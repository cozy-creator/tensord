//! The machine's caches manage themselves, with no purge verb. Each expires a TTL after its
//! last use. A disk is low when no more than its reserve is free (TensorFS `ensure::Disk`).
//! On a low disk a sweep frees what lifts it back above the reserve, but only when what it may
//! drop together covers that; otherwise it drops only the store's garbage. The plan drops, in
//! order: collected result copies, settled logs, memoized stages, downloaded wheels no
//! environment links, unused generations, then the store's least recently used model caches,
//! compiled kernels last. Only bytes an unlink frees
//! on the measured filesystem count: single-link files this process does not hold open, on
//! that filesystem. Optional cache writes are skipped on a low disk. Never evicted:
//! uncollected results (durable outputs), journal rows, a generation a run holds or an
//! installation names, a model a live executor, an unfinished run or a preparation names
//! (`keep`), a kernel namespace a live executor or kernel boot holds.
use crate::{catalog::Catalog, execution::Engine};
use fs2::FileExt;
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tensorfs_core::store::Store;

pub use tensorfs_core::ensure::Disk;

pub const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Installed or used within one sweep period counts as in use, even on a low disk: an
/// install is never evicted between its creation and its first run or binding.
pub const IDLE: Duration = Duration::from_secs(10 * 60);

/// The filesystem under `path`, now.
pub fn measure(path: &Path) -> io::Result<Disk> {
    Disk::measure(path).map_err(io::Error::other)
}

/// What one sweep removed, by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Swept {
    pub staging: usize,
    pub logs: usize,
    pub results: usize,
    pub kernels: usize,
    pub generations: usize,
    /// Ended jobs' scratch trees (`jobs::Jobs::sweep_scratch`).
    pub scratch: usize,
    /// Adapter views whose repositories went (`adapter_views::evict`).
    pub adapter_views: usize,
    /// Bytes TensorFS's GC collected: garbage, and expired or evicted model caches.
    pub store_bytes: u64,
    /// Memoized stage results (`memo::Memo`).
    pub memo: usize,
    /// Unpacked wheels of the machine's uv cache (`uv_entries`).
    pub uv_cache: usize,
}

/// The persistent compiled-kernel store (`Seal::prepare`'s `<root>/u<uid>/`). Every executor
/// and kernel boot holds its namespace's lock (`kernel_hold`) shared until it exits; a sweep
/// touches a namespace only while it holds that lock exclusively.
pub struct KernelCaches {
    pub root: PathBuf,
}

/// The lock of one kernel namespace, beside it where its identity cannot reach.
pub fn kernel_hold(kernels: &Path, namespace: &str) -> io::Result<File> {
    File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(kernels.join(format!(".{namespace}.hold")))
}

/// The models the TensorFS store caches, and those it must keep.
pub struct StoreCaches<'a> {
    pub store: &'a Store,
    /// Manifests live executors read, unfinished runs prepared, or preparations fetch.
    pub keep: Vec<String>,
}

/// Everything one sweep looks at.
pub struct Caches<'a> {
    pub engine: &'a Engine,
    pub catalog: &'a Catalog,
    /// Generations installations or configured packages still name.
    pub bound: &'a HashSet<String>,
    pub kernels: Option<&'a KernelCaches>,
    pub memo: Option<&'a crate::memo::Memo>,
    /// The machine's own uv download cache (`Publisher::uv_cache`).
    pub uv_cache: Option<&'a Path>,
    pub store: Option<StoreCaches<'a>>,
    /// The filesystem the machine's state is on (`measure`; a test's own).
    pub disk: &'a dyn Fn() -> io::Result<Disk>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Kind {
    Result,
    Log,
    Memo,
    UvCache,
    Generation,
    Kernel,
}

impl Kind {
    fn count(self, swept: &mut Swept) {
        *match self {
            Kind::Result => &mut swept.results,
            Kind::Log => &mut swept.logs,
            Kind::Memo => &mut swept.memo,
            Kind::UvCache => &mut swept.uv_cache,
            Kind::Generation => &mut swept.generations,
            Kind::Kernel => &mut swept.kernels,
        } += 1;
    }
}

/// One entry a sweep may drop. A generation's exclusive `.hold` lock, once taken, is kept
/// from planning through deletion.
struct Entry {
    kind: Kind,
    path: PathBuf,
    used: SystemTime,
    hold: Option<File>,
    /// Removed first: a uv cache entry's pointers (`uv_pointers`).
    pointers: Vec<PathBuf>,
}

impl Entry {
    fn remove(&mut self) -> io::Result<bool> {
        for pointer in &self.pointers {
            remove_pointer(pointer);
        }
        if self.kind != Kind::Generation {
            return Ok(remove(&self.path));
        }
        if self.hold.is_none() {
            let Ok(hold) = File::open(self.path.join(".hold")) else {
                return Ok(false);
            };
            if hold.try_lock_exclusive().is_err() {
                return Ok(false);
            }
            self.hold = Some(hold);
        }
        // Once renamed no resolve can find it; the lock is held until it is gone.
        let name = self.path.file_name().unwrap().to_string_lossy();
        let doomed = self.path.with_file_name(format!(".collect-{name}"));
        fs::rename(&self.path, &doomed)?;
        fs::remove_dir_all(&doomed)?;
        Ok(true)
    }
}

/// Delete generation `identity` under `generations` now (a removed installation's), unless a
/// process holds it: renamed first, so no resolve finds it. The bytes it freed, or None when
/// it is held (an executor of it still runs); reclaim removes it once nothing names it.
pub fn remove_generation(generations: &Path, identity: &str) -> io::Result<Option<u64>> {
    let path = generations.join(identity);
    let Ok(filesystem) = fs::metadata(&path).map(|m| m.dev()) else {
        return Ok(Some(0));
    };
    let bytes = releasable(&path, filesystem, &open_files()).unwrap_or(0);
    let mut entry = Entry { kind: Kind::Generation, path, used: SystemTime::now(), hold: None, pointers: vec![] };
    Ok(entry.remove()?.then_some(bytes))
}

/// Newest write, or read of a file, anywhere inside: an entry's last use. A read stamps a
/// file's access at most daily (relatime); a filesystem that never does falls back to
/// writes. A directory's own access time is this scan's, so it never counts.
fn last_use(path: &Path) -> SystemTime {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return SystemTime::UNIX_EPOCH;
    };
    let written = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    if metadata.is_dir() {
        let children = fs::read_dir(path).into_iter().flatten().flatten();
        children
            .map(|entry| last_use(&entry.path()))
            .fold(written, SystemTime::max)
    } else {
        written.max(metadata.accessed().unwrap_or(SystemTime::UNIX_EPOCH))
    }
}

fn modified(path: &Path) -> SystemTime {
    fs::symlink_metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn unused_for(used: SystemTime, age: Duration) -> bool {
    SystemTime::now()
        .duration_since(used)
        .is_ok_and(|unused| unused > age)
}

fn remove(path: &Path) -> bool {
    let removed = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    match removed {
        Ok(()) => true,
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => {
            eprintln!("reclaim {}: {error}", path.display());
            false
        }
    }
}

/// Files this process holds open, by (device, inode): unlinking one frees nothing yet.
fn open_files() -> HashSet<(u64, u64)> {
    let Ok(descriptors) = fs::read_dir("/proc/self/fd") else {
        return HashSet::new();
    };
    descriptors
        .flatten()
        .filter_map(|fd| fs::metadata(fd.path()).ok())
        .map(|m| (m.dev(), m.ino()))
        .collect()
}

/// Bytes unlinking `path` frees on `filesystem`, or None when something under it is open
/// here. Multiply linked files free nothing (uv environments share payload with a cache);
/// symlinks are not followed and another mount is never counted.
fn releasable(path: &Path, filesystem: u64, open: &HashSet<(u64, u64)>) -> Option<u64> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if metadata.dev() != filesystem || metadata.file_type().is_symlink() {
        return Some(0);
    }
    if open.contains(&(metadata.dev(), metadata.ino())) {
        return None;
    }
    let own = metadata.blocks() * 512;
    if metadata.is_dir() {
        fs::read_dir(path)
            .ok()?
            .flatten()
            .try_fold(own, |total, entry| {
                Some(total + releasable(&entry.path(), filesystem, open)?)
            })
    } else if metadata.nlink() == 1 {
        Some(own)
    } else {
        Some(0)
    }
}

/// Kernel entries of every namespace no live process holds: one compiled kernel (a `triton`
/// or `flash-attn4` entry) or one generation's torch kernels. The returned locks keep those
/// namespaces until the sweep ends.
fn kernel_entries(caches: &KernelCaches) -> (Vec<PathBuf>, Vec<File>) {
    let (mut entries, mut locks) = (vec![], vec![]);
    let Ok(namespaces) = fs::read_dir(&caches.root) else {
        return (entries, locks);
    };
    for namespace in namespaces.flatten() {
        if !namespace.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue; // never through a link, and not a lock file
        }
        let lock = kernel_hold(&caches.root, &namespace.file_name().to_string_lossy());
        let Some(lock) = lock.ok().filter(|lock| lock.try_lock_exclusive().is_ok()) else {
            continue;
        };
        locks.push(lock);
        for child in fs::read_dir(namespace.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            match child.file_name().to_str() {
                Some("triton" | "flash-attn4") => entries.extend(
                    fs::read_dir(child.path())
                        .into_iter()
                        .flatten()
                        .flatten()
                        .map(|e| e.path()),
                ),
                _ => entries.push(child.path()),
            }
        }
    }
    (entries, locks)
}

/// The uv cache's unpacked wheels (`archive-v0/<id>`) no environment links, each with the
/// last time one linked or let go of it (its newest ctime) and its pointers. Environments
/// hard-link these files, so an entry with a file linked twice is a live environment's and
/// frees nothing: it stays. uv takes the cache's `.lock` shared for each command and the
/// publisher for a whole environment build; the returned exclusive hold keeps both out until
/// the sweep ends. An entry removed with its pointers is a cache miss to uv, which unpacks the
/// wheel again. A layout uv no longer writes lists nothing.
type UvEntry = (PathBuf, SystemTime, Vec<PathBuf>);
fn uv_entries(cache: &Path) -> (Vec<UvEntry>, Option<File>) {
    let lock = File::options().write(true).open(cache.join(".lock"));
    let Some(lock) = lock.ok().filter(|lock| lock.try_lock_exclusive().is_ok()) else {
        return (vec![], None);
    };
    let mut pointers = HashMap::new();
    for bucket in fs::read_dir(cache).into_iter().flatten().flatten() {
        if bucket.file_name() != "archive-v0" {
            uv_pointers(&bucket.path(), &mut pointers);
        }
    }
    let archives = fs::read_dir(cache.join("archive-v0"))
        .into_iter()
        .flatten()
        .flatten();
    let entries = archives
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| {
            let used = unlinked_since(&entry.path())?;
            let pointers = pointers.remove(&entry.file_name()).unwrap_or_default();
            Some((entry.path(), used, pointers))
        })
        .collect();
    (entries, Some(lock))
}

/// The symlinks under `dir` into `archive-v0`, by the archive's name: uv's pointers to an
/// unpacked wheel. Each has sidecars beside it (`<pointer>.rev`, `.http`, `.msgpack`).
fn uv_pointers(dir: &Path, found: &mut HashMap<std::ffi::OsString, Vec<PathBuf>>) {
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            uv_pointers(&entry.path(), found);
        } else if let Ok(target) = kind
            .is_symlink()
            .then(|| fs::read_link(entry.path()))
            .transpose()
        {
            let archive = target
                .as_deref()
                .filter(|t| t.parent().and_then(Path::file_name) == Some("archive-v0".as_ref()));
            if let Some(name) = archive.and_then(Path::file_name) {
                found.entry(name.to_owned()).or_default().push(entry.path());
            }
        }
    }
}

/// A uv pointer and its sidecars, before the entry it names: never a pointer to nothing.
fn remove_pointer(pointer: &Path) {
    let (Some(dir), Some(name)) = (pointer.parent(), pointer.file_name()) else {
        return;
    };
    let sidecar = format!("{}.", name.to_string_lossy());
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        if entry.file_name().to_string_lossy().starts_with(&sidecar) {
            remove(&entry.path());
        }
    }
    remove(pointer);
}

/// The newest ctime under `path` (a link or unlink changes it), or None while a file under it
/// has another link, or cannot be read.
fn unlinked_since(path: &Path) -> Option<SystemTime> {
    let metadata = fs::symlink_metadata(path).ok()?;
    let nanos = Duration::new(metadata.ctime().max(0) as u64, metadata.ctime_nsec() as u32);
    let changed = UNIX_EPOCH + nanos;
    if metadata.is_dir() {
        let children = fs::read_dir(path).ok()?.flatten();
        children
            .map(|entry| unlinked_since(&entry.path()))
            .try_fold(changed, |newest, child| Some(newest.max(child?)))
    } else if metadata.is_file() && metadata.nlink() > 1 {
        None
    } else {
        Some(changed)
    }
}

/// Bytes one store GC call freed, or None while a download holds the store: nothing of the
/// store can be dropped this pass.
fn collected(
    report: tensorfs_core::err::Result<tensorfs_core::gc::Report>,
) -> io::Result<Option<u64>> {
    match report {
        Ok(report) => Ok(Some(report.reclaimed_bytes)),
        Err(busy) if busy.code == tensorfs_core::err::Code::STORE_BUSY => Ok(None),
        Err(error) => Err(io::Error::other(error)),
    }
}

/// TTL always; on a low disk, a covering plan or only garbage.
pub fn sweep(caches: &Caches) -> io::Result<Swept> {
    let mut swept = Swept::default();
    let engine = caches.engine;
    let root = &engine.root;
    // Spools of settled or unknown runs are leftovers of a crash, never needed again.
    for entry in fs::read_dir(root.join("staging"))?.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        let settled = match engine.get(&id) {
            Ok(record) => record.state.terminal(),
            Err(error) => error.kind() == io::ErrorKind::NotFound,
        };
        if settled && remove(&entry.path()) {
            swept.staging += 1;
        }
    }
    let mut entries = vec![];
    let mut add = |kind, path: PathBuf, used, pointers| {
        entries.push(Entry {
            kind,
            path,
            used,
            hold: None,
            pointers,
        })
    };
    // Result copies the client collected; uncollected ones are durable outputs.
    for entry in fs::read_dir(root.join("results"))?.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        if engine.get(&id).is_ok_and(|record| record.collected) {
            add(Kind::Result, entry.path(), modified(&entry.path()), vec![]);
        }
    }
    // Runner logs of settled or unknown runs: diagnostics.
    if let Ok(logs) = fs::read_dir(root.join("logs")) {
        for entry in logs.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let id = name.split('.').next().unwrap_or_default();
            let settled = match engine.get(id) {
                Ok(record) => record.state.terminal(),
                Err(error) => error.kind() == io::ErrorKind::NotFound,
            };
            if settled {
                add(Kind::Log, entry.path(), modified(&entry.path()), vec![]);
            }
        }
    }
    if let Some(memo) = caches.memo {
        swept.memo += memo.sweep(); // its own TTL and cap
        for home in memo.homes() {
            let used = last_use(&home);
            add(Kind::Memo, home, used, vec![]);
        }
    }
    // Generations unbound by any installation, unfinished run or configured package: a
    // resolve touches `.hold`, and a holder's shared lock on it refuses collection.
    for entry in fs::read_dir(caches.catalog.root())?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(".collect-") {
            remove(&entry.path()); // an interrupted collection
        } else if name.len() == 32
            && name.bytes().all(|b| b.is_ascii_hexdigit())
            && !caches.bound.contains(&name)
        {
            let used = modified(&entry.path().join(".hold"));
            add(Kind::Generation, entry.path(), used, vec![]);
        }
    }
    // The TTL, low disk or not.
    if let Some(store) = &caches.store {
        // Views share their closure with what they adapt: TTL only, never low-disk relief.
        swept.adapter_views = crate::adapter_views::evict(store.store, &store.keep)?;
        let expired = tensorfs_core::gc::collect_idle(store.store.root(), &store.keep, TTL);
        swept.store_bytes += collected(expired)?.unwrap_or(0);
    }
    // Kernel namespaces stay locked from here until the sweep ends: an executor that starts
    // meanwhile waits for it.
    let (wheels, _installs) = caches.uv_cache.map(uv_entries).unwrap_or_default();
    for (path, used, pointers) in wheels {
        add(Kind::UvCache, path, used, pointers);
    }
    let (kernels, _namespaces) = caches.kernels.map(kernel_entries).unwrap_or_default();
    for path in kernels {
        let used = last_use(&path);
        add(Kind::Kernel, path, used, vec![]);
    }
    let mut kept = vec![];
    for mut entry in entries {
        if unused_for(entry.used, TTL) && entry.remove()? {
            entry.kind.count(&mut swept);
        } else {
            kept.push(entry);
        }
    }

    let filesystem = fs::metadata(root)?.dev();
    let store = caches.store.as_ref();
    // A store on its own filesystem is relieved under TensorFS's rule for it.
    let shared = match store {
        Some(store) if fs::metadata(store.store.root())?.dev() != filesystem => {
            let relief = tensorfs_core::ensure::relieve(store.store, &store.keep);
            swept.store_bytes += relief.map_err(io::Error::other)?.collected_bytes;
            None
        }
        other => other,
    };
    if (caches.disk)()?.short().is_none() {
        return Ok(swept);
    }
    // Garbage first: what nothing references costs nothing to drop.
    let mut models = 0;
    if let Some(store) = shared {
        let root = store.store.root();
        if let Some(garbage) = collected(tensorfs_core::gc::collect(root, false))? {
            swept.store_bytes += garbage;
            models = tensorfs_core::gc::reclaimable(root, &store.keep).map_err(io::Error::other)?;
        }
    }
    let Some(short) = (caches.disk)()?.short() else {
        return Ok(swept);
    };

    // The plan: everything droppable now, cheapest first, each kind least recently used first.
    let open = open_files();
    let mut plan = vec![];
    for mut entry in kept {
        if entry.kind == Kind::Generation {
            if !unused_for(entry.used, IDLE) {
                continue;
            }
            let Ok(hold) = File::open(entry.path.join(".hold")) else {
                continue;
            };
            if hold.try_lock_exclusive().is_err() {
                continue;
            }
            entry.hold = Some(hold);
        }
        if let Some(bytes) = releasable(&entry.path, filesystem, &open).filter(|b| *b > 0) {
            plan.push((entry, bytes));
        }
    }
    plan.sort_by_key(|(entry, _)| (entry.kind, entry.used));
    let local: u64 = plan.iter().map(|(_, bytes)| bytes).sum();
    if local + models < short {
        return Ok(swept);
    }
    let (kernels, before): (Vec<_>, Vec<_>) = plan
        .into_iter()
        .partition(|(entry, _)| entry.kind == Kind::Kernel);
    for (mut entry, _) in before {
        if (caches.disk)()?.short().is_none() {
            return Ok(swept);
        }
        if entry.remove()? {
            entry.kind.count(&mut swept);
        }
    }
    // Models after the local caches and before compiled kernels: the fewest least recently
    // used that cover what is still missing, or all of them when kernels must go too.
    if let (Some(store), Some(short)) = (shared.filter(|_| models > 0), (caches.disk)()?.short()) {
        let (root, need) = (store.store.root(), short.min(models));
        let report = tensorfs_core::gc::collect_cached_for(root, &store.keep, need);
        swept.store_bytes += collected(report)?.unwrap_or(0);
    }
    // Kernels go only if they alone cover what is still missing.
    let cover: u64 = kernels.iter().map(|(_, bytes)| bytes).sum();
    for (mut entry, _) in kernels {
        match (caches.disk)()?.short() {
            Some(short) if cover >= short => {}
            _ => break,
        }
        if entry.remove()? {
            entry.kind.count(&mut swept);
        }
    }
    Ok(swept)
}
