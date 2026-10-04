//! The machine's caches manage themselves, with no purge verb. Each expires a TTL after its
//! last use. A disk is low when no more than its reserve is free (TensorFS `ensure::Disk`).
//! On a low disk a sweep frees what lifts it back above the reserve, but only when what it may
//! drop together covers that; otherwise it drops only the store's garbage. The plan drops, in
//! order: collected result copies, settled logs, memoized stages, unused generations, then the
//! store's least recently used model caches, compiled kernels last. Only bytes an unlink frees
//! on the measured filesystem count: single-link files this process does not hold open, on
//! that filesystem. Optional cache writes are skipped on a low disk. Never evicted:
//! uncollected results (durable outputs), journal rows, a generation a run holds or an
//! installation names, a model a live executor, an unfinished run or a preparation names
//! (`keep`), a kernel namespace a live executor or kernel boot holds.
use crate::{catalog::Catalog, execution::Engine};
use fs2::FileExt;
use std::{
    collections::HashSet,
    fs::{self, File},
    io,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
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
    pub store: Option<StoreCaches<'a>>,
    /// The filesystem the machine's state is on (`measure`; a test's own).
    pub disk: &'a dyn Fn() -> io::Result<Disk>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Kind {
    Result,
    Log,
    Memo,
    Generation,
    Kernel,
}

impl Kind {
    fn count(self, swept: &mut Swept) {
        *match self {
            Kind::Result => &mut swept.results,
            Kind::Log => &mut swept.logs,
            Kind::Memo => &mut swept.memo,
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
}

impl Entry {
    fn remove(&mut self) -> io::Result<bool> {
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
    let mut add = |kind, path: PathBuf, used| {
        entries.push(Entry {
            kind,
            path,
            used,
            hold: None,
        })
    };
    // Result copies the client collected; uncollected ones are durable outputs.
    for entry in fs::read_dir(root.join("results"))?.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        if engine.get(&id).is_ok_and(|record| record.collected) {
            add(Kind::Result, entry.path(), modified(&entry.path()));
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
                add(Kind::Log, entry.path(), modified(&entry.path()));
            }
        }
    }
    if let Some(memo) = caches.memo {
        swept.memo += memo.sweep(); // its own TTL and cap
        for home in memo.homes() {
            let used = last_use(&home);
            add(Kind::Memo, home, used);
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
            add(
                Kind::Generation,
                entry.path(),
                modified(&entry.path().join(".hold")),
            );
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
    let (kernels, _namespaces) = caches.kernels.map(kernel_entries).unwrap_or_default();
    for path in kernels {
        let used = last_use(&path);
        add(Kind::Kernel, path, used);
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
