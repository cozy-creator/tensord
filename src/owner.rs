use crate::{
    os,
    protocol::{Body, Object},
};
use fs2::FileExt;
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, PermissionsExt},
            net::UnixStream,
        },
    },
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tensorfs_core::{
    ids::{self, ObjectRef},
    store::{Fault, Store},
};

pub type Shared = Arc<Mutex<Owner>>;
struct Cached {
    file: File,
    last_used: Instant,
    clock: u64,
}
struct Peer {
    pidfd: File,
    disconnected: bool,
}
struct Lease {
    peer: u64,
    object: String,
    host: bool,
}
pub struct Owner {
    store: Arc<Store>,
    _lock: File,
    _store_lock: File,
    cache: HashMap<String, Cached>,
    peers: HashMap<u64, Peer>,
    leases: HashMap<u64, Lease>,
    next_id: u64,
    clock: u64,
    budget: u64,
    ttl: Duration,
    stopping: bool,
    pub incarnation: String,
    /// Weight peers: hello, import, attach, release, stats.
    pub socket: PathBuf,
    /// The owner's administration: submit, executions, cancel, results, shutdown.
    pub admin: PathBuf,
}
fn failure(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
fn validate(object: &Object) -> io::Result<()> {
    ids::hex64("object", &object.sha256).map_err(io::Error::other)?;
    Ok(())
}
/// The store's own owner lock, taken exclusively; None when it has none yet and `create`
/// is false.
fn lock_store(store: &Path, create: bool) -> io::Result<Option<File>> {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        .open(store.join("owner.lock"));
    let lock = match lock {
        Ok(lock) => lock,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    lock.try_lock_exclusive().map_err(|_| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "this TensorFS store already has a machine owner",
        )
    })?;
    Ok(Some(lock))
}
impl Owner {
    /// `store` is the TensorFS store this owner serves, which may live outside `root`.
    pub fn new(root: &Path, store: &Path, budget: u64, ttl: Duration) -> io::Result<Shared> {
        std::fs::create_dir_all(root)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("owner.lock"))?;
        lock.try_lock_exclusive().map_err(|_| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "state already has a machine owner",
            )
        })?;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
        // One machine per store, whichever state directory or path names it. A store another
        // machine owns is refused before its catalog is opened; a new one is locked once
        // TensorFS has made it (it refuses a directory that already holds a foreign file).
        let owned = lock_store(store, false)?;
        let store = Arc::new(Store::ensure(store).map_err(io::Error::other)?);
        let store_lock = match owned {
            Some(lock) => lock,
            None => lock_store(store.root(), true)?.expect("created"),
        };
        // The machine is the store's owner: its read leases pin in memory, so its GC can make
        // room while layouts are served.
        tensorfs_core::meta::own(&store);
        let incarnation = format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        Ok(Arc::new(Mutex::new(Self {
            store,
            _lock: lock,
            _store_lock: store_lock,
            cache: HashMap::new(),
            peers: HashMap::new(),
            leases: HashMap::new(),
            next_id: 1,
            clock: 0,
            budget,
            ttl,
            stopping: false,
            incarnation,
            socket: control_socket(root, "machine")?,
            admin: control_socket(root, "admin")?,
        })))
    }
    fn id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
    pub fn connect(&mut self, stream: &UnixStream) -> io::Result<u64> {
        self.accepting()?;
        let pidfd = os::peer_pidfd(stream)?;
        self.reap();
        if self.peers.len() >= 64 {
            return Err(io::Error::other("64 active client limit reached"));
        }
        let id = self.id();
        self.peers.insert(
            id,
            Peer {
                pidfd,
                disconnected: false,
            },
        );
        Ok(id)
    }
    pub fn store(&self) -> Arc<Store> {
        self.store.clone()
    }
    pub fn disconnect(&mut self, peer: u64) {
        if self.leases.values().any(|lease| lease.peer == peer) {
            if let Some(row) = self.peers.get_mut(&peer) {
                row.disconnected = true;
            }
        } else {
            self.peers.remove(&peer);
        }
        self.reap();
    }
    fn reap(&mut self) {
        let dead: HashSet<_> = self
            .peers
            .iter()
            .filter(|(_, p)| os::ended(&p.pidfd))
            .map(|(id, _)| *id)
            .collect();
        self.leases.retain(|_, lease| !dead.contains(&lease.peer));
        self.peers.retain(|id, peer| {
            !dead.contains(id)
                && (!peer.disconnected || self.leases.values().any(|l| l.peer == *id))
        });
        let held: HashSet<_> = self
            .leases
            .values()
            .filter(|lease| lease.host)
            .map(|l| &l.object)
            .collect();
        let ttl = self.ttl;
        self.cache
            .retain(|id, c| held.contains(id) || c.last_used.elapsed() < ttl);
    }
    fn held(&self, id: &str) -> bool {
        self.leases
            .values()
            .any(|lease| lease.host && lease.object == id)
    }
    fn host_bytes(&self) -> u64 {
        self.cache
            .values()
            .map(|c| {
                c.file
                    .metadata()
                    .map(|m| m.blocks() * 512)
                    .unwrap_or(u64::MAX)
            })
            .sum()
    }
    pub fn import(&mut self, object: Object, mut file: File) -> io::Result<Body> {
        self.accepting()?;
        validate(&object)?;
        if os::seals(&file)? & os::FULL_SEALS != os::FULL_SEALS {
            return Err(failure("input fd must have full immutable seals"));
        }
        if file.metadata()?.len() != object.length {
            return Err(failure("input length differs"));
        }
        file.seek(SeekFrom::Start(0))?;
        let expected = ObjectRef {
            sha256: object.sha256.clone(),
            length: object.length,
        };
        let put = self
            .store
            .put_stream(&mut file, Some(&expected), &Fault::default())
            .map_err(io::Error::other)?;
        Ok(Body::Imported {
            object,
            admitted: put.admitted,
        })
    }
    pub fn attach(&mut self, peer: u64, object: Object) -> io::Result<(Body, File)> {
        self.accepting()?;
        validate(&object)?;
        self.reap();
        self.clock += 1;
        let source = self
            .store
            .open_verified(&object.sha256)
            .map_err(io::Error::other)?;
        if source.len() != object.length {
            return Err(failure("stored object length differs"));
        }
        let mut source = source.into_file();
        if !self.cache.contains_key(&object.sha256) && object.length <= self.budget {
            // Charges page-rounded backing, not the logical byte count.
            // SAFETY: sysconf takes a scalar key and returns the kernel's page size.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page <= 0 {
                return Err(io::Error::other("page size is unavailable"));
            }
            let page = page as u64;
            let pages = object
                .length
                .div_ceil(page)
                .checked_mul(page)
                .ok_or_else(|| failure("object span overflows"))?;
            while self.host_bytes().saturating_add(pages) > self.budget {
                let idle = self
                    .cache
                    .iter()
                    .filter(|(id, _)| !self.held(id))
                    .min_by_key(|(_, c)| c.clock)
                    .map(|(id, _)| id.clone());
                match idle {
                    Some(id) => {
                        self.cache.remove(&id);
                    }
                    None => break,
                }
            }
            if self.host_bytes().saturating_add(pages) <= self.budget {
                let cached = (|| -> io::Result<File> {
                    let mut file = os::memfd()?;
                    io::copy(&mut source, &mut file)?;
                    file.flush()?;
                    os::seal(&file)?;
                    Ok(file)
                })();
                if let Ok(file) = cached {
                    self.cache.insert(
                        object.sha256.clone(),
                        Cached {
                            file,
                            clock: self.clock,
                            last_used: Instant::now(),
                        },
                    );
                } else {
                    // Optional cache failure does not refuse the object. The
                    // failed backing is dropped, and the verified inode remains.
                    source.seek(SeekFrom::Start(0))?;
                }
            }
        }
        let (tier, fd) = match self.cache.get_mut(&object.sha256) {
            Some(cached) => {
                cached.clock = self.clock;
                cached.last_used = Instant::now();
                let fd = File::open(format!("/proc/self/fd/{}", cached.file.as_raw_fd()))?;
                ("host", fd)
            }
            None => ("disk", source),
        };
        let lease = self.id();
        self.leases.insert(
            lease,
            Lease {
                peer,
                object: object.sha256.clone(),
                host: tier == "host",
            },
        );
        Ok((
            Body::Attached {
                object,
                lease,
                incarnation: self.incarnation.clone(),
                tier,
            },
            fd,
        ))
    }
    pub fn release(&mut self, peer: u64, id: u64, incarnation: &str) -> io::Result<Body> {
        if incarnation != self.incarnation {
            return Err(failure("lease belongs to another machine incarnation"));
        }
        if self.leases.get(&id).map(|l| l.peer) != Some(peer) {
            return Err(failure("lease is absent or owned by another connection"));
        }
        self.leases.remove(&id);
        self.reap();
        Ok(Body::Released)
    }
    pub fn stats(&mut self) -> Body {
        self.reap();
        Body::Stats {
            host_bytes: self.host_bytes(),
            cached_objects: self.cache.len(),
            active_leases: self.leases.len(),
            host_budget: self.budget,
        }
    }
    fn accepting(&self) -> io::Result<()> {
        if self.stopping {
            Err(io::Error::other("machine is stopping"))
        } else {
            Ok(())
        }
    }
    pub fn idle(&mut self) -> bool {
        self.reap();
        self.leases.is_empty()
    }
    pub fn stop(&mut self) -> bool {
        if !self.idle() {
            return false;
        }
        self.stopping = true;
        true
    }
}
fn control_socket(root: &Path, name: &str) -> io::Result<PathBuf> {
    let path = root.join(format!("{name}.sock"));
    if std::os::unix::net::SocketAddr::from_pathname(&path).is_ok() {
        return Ok(path);
    }
    let uid = unsafe { libc::geteuid() };
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir);
    let directory = runtime.join(format!("cozy-machine-{uid}"));
    match std::fs::create_dir(&directory) {
        Ok(()) => std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error),
    }
    let metadata = std::fs::symlink_metadata(&directory)?;
    if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o077 != 0 {
        return Err(failure(
            "control socket runtime directory must be owned and private",
        ));
    }
    let canonical = root.canonicalize()?;
    let identity = tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(
        canonical.as_os_str().as_encoded_bytes(),
    ));
    let path = directory.join(format!("{}.{name}.sock", &identity[..24]));
    std::os::unix::net::SocketAddr::from_pathname(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorfs_core::{meta::Meta, read, store::Fault};

    /// The machine owns its store: a layout's live read lease no longer stops TensorFS GC,
    /// which keeps the leased bytes and takes what nothing holds.
    #[test]
    fn the_machines_gc_runs_while_a_layout_is_leased() {
        let root = std::env::temp_dir().join(format!("cm-owner-gc-{}", uuid::Uuid::new_v4()));
        let owner = Owner::new(&root, &root.join("tensorfs"), 1 << 20, Duration::from_secs(60)).unwrap();
        let store = owner.lock().unwrap().store();
        let meta = Meta::open(&store).unwrap();
        let put = |bytes: &[u8]| store.put_stream(&mut &bytes[..], None, &Fault::default()).unwrap().obj;
        let leased = put(b"weights a served layout reads");
        let garbage = put(b"weights nothing references");
        let (lease, _) = read::acquire(&store, &meta, "layout", vec![leased.clone()]).unwrap();
        tensorfs_core::gc::collect(store.root(), false).unwrap();
        assert!(store.object_path(&leased.sha256).is_file());
        assert!(!store.object_path(&garbage.sha256).exists());
        lease.release(&meta).unwrap();
        drop(owner);
        let _ = std::fs::remove_dir_all(root);
    }
}
