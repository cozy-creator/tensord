//! The machine's caches manage themselves, with no purge verb. Each expires a TTL after its
//! last use, low disk or not. A disk is low when no more than its reserve, max(1 GiB, 1/50),
//! is free (TensorFS `ensure::Disk`). On a low disk a sweep frees what lifts it back above
//! the reserve, but only when everything droppable together covers that; otherwise it drops
//! nothing but garbage. The plan drops collected result copies, settled logs, memoized
//! stages, unused generations, then the fewest least-recently-used model cache roots that
//! cover the rest, compiled kernels last. Below the reserve optional cache writes are
//! skipped. Never evicted: uncollected results (durable outputs), journal rows, a generation
//! a run holds or an installation names, a model a live executor, an unfinished run or a
//! preparation names (`keep`).
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
/// How often the machine sweeps.
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
    /// Bytes TensorFS's GC collected: garbage, and expired or evicted model cache roots.
    pub store_bytes: u64,
    /// Memoized stage results (`memo::Memo`).
    pub memo: usize,
}

/// The persistent compiled-kernel store (`Seal::prepare`'s `<root>/u<uid>/`).
pub struct KernelCaches {
    pub root: PathBuf,
    /// Namespaces (`u<uid>`) a live executor uses: never touched.
    pub busy: HashSet<String>,
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

/// Newest modification or access anywhere inside: an entry's last use. A read stamps
/// access at most daily (relatime); a filesystem that never does falls back to writes.
fn last_use(path: &Path) -> SystemTime {
    let own = fs::symlink_metadata(path).map_or(SystemTime::UNIX_EPOCH, |m| {
        let accessed = m.accessed().unwrap_or(SystemTime::UNIX_EPOCH);
        m.modified().map_or(accessed, |modified| modified.max(accessed))
    });
    match fs::read_dir(path) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| last_use(&entry.path()))
            .fold(own, SystemTime::max),
        Err(_) => own,
    }
}

/// Bytes on disk under `path`.
fn size(path: &Path) -> u64 {
    let own = fs::symlink_metadata(path).map_or(0, |m| m.blocks() * 512);
    match fs::read_dir(path) {
        Ok(entries) => own + entries.flatten().map(|entry| size(&entry.path())).sum::<u64>(),
        Err(_) => own,
    }
}

fn unused_for(path: &Path, age: Duration) -> bool {
    SystemTime::now()
        .duration_since(last_use(path))
        .is_ok_and(|unused| unused > age)
}

/// Kernel entries least recently used first: one compiled kernel (a `triton` or `flash-attn4`
/// entry) or one generation's torch kernels.
fn kernel_entries(caches: &KernelCaches) -> Vec<PathBuf> {
    let mut entries = vec![];
    let Ok(namespaces) = fs::read_dir(&caches.root) else {
        return entries;
    };
    for namespace in namespaces.flatten() {
        if caches
            .busy
            .contains(&*namespace.file_name().to_string_lossy())
        {
            continue;
        }
        for child in fs::read_dir(namespace.path()).into_iter().flatten().flatten() {
            let name = child.file_name();
            if matches!(name.to_str(), Some("triton" | "flash-attn4")) {
                entries.extend(
                    fs::read_dir(child.path())
                        .into_iter()
                        .flatten()
                        .flatten()
                        .map(|e| e.path()),
                );
            } else {
                entries.push(child.path());
            }
        }
    }
    lru(entries)
}

fn lru(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut dated: Vec<_> = paths.into_iter().map(|path| (last_use(&path), path)).collect();
    dated.sort();
    dated.into_iter().map(|(_, path)| path).collect()
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

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Result,
    Log,
    Memo,
    Generation,
    Kernel,
}

impl Kind {
    fn count(self, swept: &mut Swept) -> &mut usize {
        match self {
            Kind::Result => &mut swept.results,
            Kind::Log => &mut swept.logs,
            Kind::Memo => &mut swept.memo,
            Kind::Generation => &mut swept.generations,
            Kind::Kernel => &mut swept.kernels,
        }
    }

    /// Remove one entry of this kind; a generation only while nothing holds it.
    fn remove(self, path: &Path) -> io::Result<bool> {
        match self {
            Kind::Generation => collect(path),
            _ => Ok(remove(path)),
        }
    }
}

/// TTL always; on a low disk, a covering plan or nothing.
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
    // Runner logs of settled runs (diagnostics) and result copies the client collected
    // (uncollected ones are durable outputs), least recently used first.
    let mut logs = vec![];
    if let Ok(entries) = fs::read_dir(root.join("logs")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let id = name.split('.').next().unwrap_or_default();
            if engine.get(id).map_or(true, |record| record.state.terminal()) {
                logs.push(entry.path());
            }
        }
    }
    let mut results = vec![];
    for entry in fs::read_dir(root.join("results"))?.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        if engine.get(&id).is_ok_and(|record| record.collected) {
            results.push(entry.path());
        }
    }
    // Generations unbound and unheld, least recently used first: a resolve touches `.hold`.
    let mut generations = vec![];
    for entry in fs::read_dir(caches.catalog.root())?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(".collect-") {
            remove(&entry.path()); // an interrupted collection
        } else if name.len() == 32
            && name.bytes().all(|b| b.is_ascii_hexdigit())
            && !caches.bound.contains(&name)
        {
            generations.push(entry.path());
        }
    }
    generations.sort_by_key(|path| last_use(&path.join(".hold")));
    let mut entries: Vec<(Kind, PathBuf)> = vec![];
    entries.extend(lru(results).into_iter().map(|path| (Kind::Result, path)));
    entries.extend(lru(logs).into_iter().map(|path| (Kind::Log, path)));
    if let Some(memo) = caches.memo {
        swept.memo += memo.sweep();
        entries.extend(memo.lru().into_iter().map(|path| (Kind::Memo, path)));
    }
    entries.extend(generations.into_iter().map(|path| (Kind::Generation, path)));
    let kernels = caches.kernels.map(kernel_entries).unwrap_or_default();
    entries.extend(kernels.into_iter().map(|path| (Kind::Kernel, path)));

    // The TTL: each kind unused for it goes (memoized stages keep their own).
    for (kind, path) in &entries {
        let used = match kind {
            Kind::Generation => path.join(".hold"),
            _ => path.clone(),
        };
        if *kind != Kind::Memo && unused_for(&used, TTL) && kind.remove(path)? {
            *kind.count(&mut swept) += 1;
        }
    }
    if let Some(store) = &caches.store {
        let root = store.store.root();
        swept.adapter_views = crate::adapter_views::evict(store.store, &store.keep)?;
        let mut report = tensorfs_core::gc::collect_idle(root, &store.keep, TTL);
        if swept.adapter_views > 0 {
            report = tensorfs_core::gc::collect(root, false);
        }
        swept.store_bytes += report.map_err(io::Error::other)?.reclaimed_bytes;
    }

    // A low disk: garbage first, then a plan that covers the rest, or nothing.
    let Some(mut need) = (caches.disk)()?.short() else {
        return Ok(swept);
    };
    let mut models = 0;
    if let Some(store) = &caches.store {
        let root = store.store.root();
        let garbage = tensorfs_core::gc::collect(root, false).map_err(io::Error::other)?;
        swept.store_bytes += garbage.reclaimed_bytes;
        need = need.saturating_sub(garbage.reclaimed_bytes);
        models = tensorfs_core::gc::reclaimable(root, &store.keep).map_err(io::Error::other)?;
    }
    let plan: Vec<(Kind, PathBuf, u64)> = entries
        .into_iter()
        .filter(|(_, path)| path.exists())
        .map(|(kind, path)| {
            let bytes = size(&path);
            (kind, path, bytes)
        })
        .collect();
    if need == 0 || plan.iter().map(|(_, _, bytes)| bytes).sum::<u64>() + models < need {
        return Ok(swept);
    }
    let (kernels, before): (Vec<_>, Vec<_>) =
        plan.into_iter().partition(|(kind, ..)| *kind == Kind::Kernel);
    let mut freed = 0;
    for entry in &before {
        if freed >= need {
            return Ok(swept);
        }
        freed += take(entry, &mut swept)?;
    }
    // Models before compiled kernels: the fewest least recently used that cover the rest.
    if let Some(store) = caches.store.as_ref().filter(|_| freed < need) {
        let root = store.store.root();
        let report = tensorfs_core::gc::collect_cached_for(root, &store.keep, need - freed)
            .map_err(io::Error::other)?;
        swept.store_bytes += report.reclaimed_bytes;
        freed += report.reclaimed_bytes;
    }
    for entry in &kernels {
        if freed >= need {
            break;
        }
        freed += take(entry, &mut swept)?;
    }
    Ok(swept)
}

/// Drop one planned entry; the bytes it freed.
fn take((kind, path, bytes): &(Kind, PathBuf, u64), swept: &mut Swept) -> io::Result<u64> {
    if !kind.remove(path)? {
        return Ok(0);
    }
    *kind.count(swept) += 1;
    Ok(*bytes)
}

/// Remove one generation only while nothing holds it: a run or executor keeps a shared
/// lock on `.hold` (the executor inherits it), so the exclusive lock is the proof.
fn collect(path: &Path) -> io::Result<bool> {
    let hold = match File::open(path.join(".hold")) {
        Ok(hold) => hold,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if hold.try_lock_exclusive().is_err() {
        return Ok(false);
    }
    let name = path.file_name().unwrap().to_string_lossy();
    let doomed = path.with_file_name(format!(".collect-{name}"));
    // Once renamed no resolve can find it; the lock is held until it is gone.
    fs::rename(path, &doomed)?;
    fs::remove_dir_all(&doomed)?;
    Ok(true)
}
