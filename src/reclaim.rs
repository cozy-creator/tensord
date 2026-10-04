//! The machine's caches manage themselves: a TTL, plus the host's storage-pressure policy
//! (TensorFS `ensure::admission`): pressure starts at the reserve, max(1 GiB, 1/50).
//! Pressure removes a covering set of releasable local bytes or nothing. TTL expiration
//! remains independent; optional cache writes are skipped at the reserve.
//! Never evicted: uncollected results (durable outputs), journal rows, an environment
//! generation any queued, running or retained run holds, or one an installation names.
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

pub const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Installed or used within one sweep period counts as in use, even under pressure: an
/// install is never evicted between its creation and its first run or binding.
pub const IDLE: Duration = Duration::from_secs(10 * 60);

#[derive(Clone, Copy, Debug)]
pub struct Disk {
    pub capacity: u64,
    pub available: u64,
}
impl Disk {
    pub fn measure(path: &Path) -> io::Result<Self> {
        let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(io::Error::other)?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: valid path and writable statvfs.
        if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let unit = stat.f_frsize.max(1);
        Ok(Self {
            capacity: stat.f_blocks.saturating_mul(unit),
            available: stat.f_bavail.saturating_mul(unit),
        })
    }
    pub fn reserve(&self) -> u64 {
        (1u64 << 30).max(self.capacity / 50)
    }
    pub fn short(&self) -> u64 {
        self.reserve()
            .saturating_add(1)
            .saturating_sub(self.available)
    }
    pub fn pressure(&self) -> bool {
        self.available <= self.reserve()
    }
    pub fn relieved(&self) -> bool {
        self.available > self.reserve()
    }
    /// Optional cache writes are skipped below this.
    pub fn below_reserve(&self) -> bool {
        self.pressure()
    }
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
    /// Bytes TensorFS's GC collected under storage pressure (`ensure::relieve`).
    pub store_bytes: u64,
    /// Memoized stage results (`memo::Memo::sweep`).
    pub memo: usize,
}

/// The persistent compiled-kernel store (`Seal::prepare`'s `<root>/u<uid>/`).
/// A busy snapshot is conservative status only, not held deletion authority.
pub struct KernelCaches {
    pub root: PathBuf,
    /// Namespaces (`u<uid>`) a live executor uses: never touched.
    pub busy: HashSet<String>,
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

/// TTL always; under pressure only a covering plan on the machine-state filesystem.
/// TensorFS model caches plan independently against their own filesystem. We never add
/// another mount's bytes or an in-use generation's size to this plan.
pub fn sweep(
    engine: &Engine,
    catalog: &Catalog,
    bound: &HashSet<String>,
    kernels: Option<&KernelCaches>,
) -> io::Result<Swept> {
    sweep_with_disk(engine, catalog, bound, kernels, &|| {
        Disk::measure(&engine.root)
    })
}

#[doc(hidden)]
pub fn sweep_with_disk(
    engine: &Engine,
    catalog: &Catalog,
    bound: &HashSet<String>,
    kernels: Option<&KernelCaches>,
    disk: &dyn Fn() -> io::Result<Disk>,
) -> io::Result<Swept> {
    let mut swept = Swept::default();
    let root = &engine.root;
    let filesystem = fs::metadata(root)?.dev();
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
    let mut candidates = Vec::new();
    if let Ok(entries) = fs::read_dir(root.join("logs")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let id = name.split('.').next().unwrap_or_default();
            let settled = match engine.get(id) {
                Ok(record) => record.state.terminal(),
                Err(error) => error.kind() == io::ErrorKind::NotFound,
            };
            if settled {
                candidates.push((Kind::Log, entry.path()));
            }
        }
    }
    for entry in fs::read_dir(root.join("results"))?.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        if engine.get(&id).is_ok_and(|record| record.collected) {
            candidates.push((Kind::Result, entry.path()));
        }
    }
    for entry in fs::read_dir(catalog.root())?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(".collect-") {
            remove(&entry.path());
            continue;
        }
        if name.len() == 32 && name.bytes().all(|b| b.is_ascii_hexdigit()) && !bound.contains(&name)
        {
            candidates.push((Kind::Generation, entry.path()));
        }
    }
    // Kernel namespace snapshots are not deletion leases. Keep compiled caches until
    // an owner provides held exclusion through deletion; no pressure/TTL false eligibility.
    let _ = kernels;
    // A generation last use is its held eligibility marker; other copies expire by TTL.
    let used = |kind: Kind, path: &Path| match kind {
        Kind::Generation => fs::symlink_metadata(path.join(".hold"))
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH),
        _ => fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH),
    };
    candidates.sort_by_key(|(kind, path)| (*kind, used(*kind, path)));
    let mut plan = Vec::new();
    for (kind, path) in candidates {
        let old = SystemTime::now()
            .duration_since(used(kind, &path))
            .is_ok_and(|age| age > TTL);
        if old && kind.remove(&path, None)? {
            kind.count(&mut swept);
            continue;
        }
        if kind == Kind::Generation
            && SystemTime::now()
                .duration_since(used(kind, &path))
                .is_ok_and(|age| age <= IDLE)
        {
            continue;
        }
        // Raw body readers can retain log/result inodes after collection. Their copies
        // expire by TTL but cannot cover pressure without an active-reader exclusion.
        if kind != Kind::Generation {
            continue;
        }
        // Generation locks are eligibility evidence and stay held through plan/deletion.
        let hold = if kind == Kind::Generation {
            let Ok(hold) = File::open(path.join(".hold")) else {
                continue;
            };
            if hold.try_lock_exclusive().is_err() {
                continue;
            }
            Some(hold)
        } else {
            None
        };
        let Some(bytes) = releasable(&path, filesystem) else {
            continue;
        };
        if bytes > 0 {
            plan.push((kind, path, bytes, hold));
        }
    }
    let need = disk()?.short();
    if need == 0 || plan.iter().map(|(_, _, bytes, _)| bytes).sum::<u64>() < need {
        return Ok(swept);
    }
    for (kind, path, _, hold) in plan {
        if disk()?.relieved() {
            break;
        }
        if kind.remove(&path, hold.as_ref())? {
            kind.count(&mut swept);
        }
    }
    Ok(swept)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Result,
    Log,
    Generation,
}
impl Kind {
    fn count(self, swept: &mut Swept) {
        *match self {
            Self::Result => &mut swept.results,
            Self::Log => &mut swept.logs,
            Self::Generation => &mut swept.generations,
        } += 1;
    }
    fn remove(self, path: &Path, hold: Option<&File>) -> io::Result<bool> {
        if self == Self::Generation {
            match hold {
                Some(hold) => collect_held(path, hold),
                None => collect(path),
            }
        } else {
            Ok(remove(path))
        }
    }
}

/// Count only allocated blocks this unlink can release on the measured filesystem.
/// Multiply linked files are conservatively excluded (uv environments commonly share
/// their payload with an external cache). Never follow symlinks or descend another mount.
fn releasable(path: &Path, filesystem: u64) -> Option<u64> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if metadata.dev() != filesystem {
        return None;
    }
    if metadata.file_type().is_symlink() {
        return Some(0);
    }
    if metadata.is_dir() {
        let children = fs::read_dir(path).ok()?.try_fold(0u64, |total, entry| {
            let bytes = releasable(&entry.ok()?.path(), filesystem)?;
            Some(total.saturating_add(bytes))
        })?;
        Some(
            metadata
                .blocks()
                .saturating_mul(512)
                .saturating_add(children),
        )
    } else if metadata.is_file() && metadata.nlink() == 1 {
        Some(metadata.blocks().saturating_mul(512))
    } else {
        Some(0)
    }
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
    collect_held(path, &hold)
}

fn collect_held(path: &Path, _hold: &File) -> io::Result<bool> {
    let name = path.file_name().unwrap().to_string_lossy();
    let doomed = path.with_file_name(format!(".collect-{name}"));
    // Once renamed no resolve can find it; the lock is held until it is gone.
    fs::rename(path, &doomed)?;
    fs::remove_dir_all(&doomed)?;
    Ok(true)
}
