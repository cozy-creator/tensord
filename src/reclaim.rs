//! The machine's caches manage themselves: a TTL, plus the host's storage-pressure policy
//! (TensorFS `ensure::admission`): pressure starts below 1/10 of the filesystem free and
//! ends above 1/5; below the reserve, max(1 GiB, 1/50), optional cache writes are skipped.
//! Never evicted: uncollected results (durable outputs), journal rows, an environment
//! generation any queued, running or retained run holds, or one an installation names.
use crate::{catalog::Catalog, execution::Engine};
use fs2::FileExt;
use std::{
    collections::HashSet,
    fs::{self, File},
    io,
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
    pub fn pressure(&self) -> bool {
        self.available < self.capacity / 10
    }
    pub fn relieved(&self) -> bool {
        self.available > self.capacity / 5
    }
    /// Optional cache writes are skipped below this.
    pub fn below_reserve(&self) -> bool {
        self.available < (1u64 << 30).max(self.capacity / 50)
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
}

/// The persistent compiled-kernel store (`Seal::prepare`'s `<root>/u<uid>/`). Recompiling is
/// cheaper than reinstalling, so it goes before generations, and only under pressure.
pub struct KernelCaches {
    pub root: PathBuf,
    /// Namespaces (`u<uid>`) a live executor uses: never touched.
    pub busy: HashSet<String>,
}

/// Newest modification anywhere inside: an entry's last use.
fn last_use(path: &Path) -> SystemTime {
    let own = fs::symlink_metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    match fs::read_dir(path) {
        Ok(entries) => entries
            .flatten()
            .map(|entry| last_use(&entry.path()))
            .fold(own, SystemTime::max),
        Err(_) => own,
    }
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
    let mut dated: Vec<_> = entries.into_iter().map(|path| (last_use(&path), path)).collect();
    dated.sort();
    dated.into_iter().map(|(_, path)| path).collect()
}

fn older_than(path: &Path, age: Duration) -> bool {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .is_ok_and(|modified| SystemTime::now().duration_since(modified).is_ok_and(|a| a > age))
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

/// TTL always; under pressure each tier in turn, stopping once the disk is relieved.
/// `bound`: generations installations or configured packages still name.
pub fn sweep(
    engine: &Engine,
    catalog: &Catalog,
    bound: &HashSet<String>,
    kernels: Option<&KernelCaches>,
) -> io::Result<Swept> {
    let mut swept = Swept::default();
    let root = &engine.root;
    let disk = || Disk::measure(root);
    let mut pressure = disk()?.pressure();
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
    // Runner logs of settled runs: diagnostics, kept for the TTL.
    if let Ok(entries) = fs::read_dir(root.join("logs")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let id = name.split('.').next().unwrap_or_default();
            let settled = engine.get(id).map_or(true, |record| record.state.terminal());
            if settled && (pressure || older_than(&entry.path(), TTL)) && remove(&entry.path()) {
                swept.logs += 1;
            }
        }
    }
    pressure = pressure && !disk()?.relieved();
    // Result copies the client already collected; uncollected ones are durable outputs.
    for entry in fs::read_dir(root.join("results"))?.flatten() {
        let id = entry.file_name().to_string_lossy().into_owned();
        let collected = engine.get(&id).is_ok_and(|record| record.collected);
        if collected && (pressure || older_than(&entry.path(), TTL)) && remove(&entry.path()) {
            swept.results += 1;
        }
    }
    pressure = pressure && !disk()?.relieved();
    if let Some(kernels) = kernels.filter(|_| pressure) {
        for entry in kernel_entries(kernels) {
            if remove(&entry) {
                swept.kernels += 1;
            }
            if disk()?.relieved() {
                pressure = false;
                break;
            }
        }
    }
    // Generations: unbound, unheld, and unused for the TTL (or the disk is short).
    let mut candidates: Vec<_> = fs::read_dir(catalog.root())?
        .flatten()
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".collect-") {
                return true; // an interrupted collection
            }
            name.len() == 32 && name.bytes().all(|b| b.is_ascii_hexdigit()) && !bound.contains(&name)
        })
        .collect();
    // Least recently used first: a resolve touches `.hold`.
    candidates.sort_by_key(|entry| {
        fs::metadata(entry.path().join(".hold"))
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    for entry in candidates {
        let path = entry.path();
        if entry.file_name().to_string_lossy().starts_with(".collect-") {
            remove(&path);
            continue;
        }
        let hold = path.join(".hold");
        let expired = older_than(&hold, TTL) || (pressure && older_than(&hold, IDLE));
        if expired && collect(&path)? {
            swept.generations += 1;
            pressure = pressure && !disk()?.relieved();
        }
    }
    Ok(swept)
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
