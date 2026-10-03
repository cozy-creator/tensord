//! Optional root-sealed identity; this is not a public caller-controlled credential.
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io,
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LaunchIdentity {
    pub uid: u32,
    pub gid: u32,
}
impl LaunchIdentity {
    pub fn validate(self) -> io::Result<Self> {
        // A nonprivileged owner can only retain its existing identity.
        if unsafe { libc::geteuid() } != 0
            && (self.uid != unsafe { libc::geteuid() } || self.gid != unsafe { libc::getegid() })
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "configured executor identity requires privilege the owner does not have",
            ));
        }
        Ok(self)
    }
    pub(crate) fn traverse(self, path: &Path) -> io::Result<()> {
        self.directory(path, unsafe { libc::geteuid() }, 0o710)
    }
    pub(crate) fn own(self, path: &Path) -> io::Result<()> {
        self.directory(path, self.uid, 0o700)
    }
    fn directory(self, path: &Path, uid: u32, mode: u32) -> io::Result<()> {
        self.validate()?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY)
            .open(path)?;
        let metadata = file.metadata()?;
        if metadata.uid() != unsafe { libc::geteuid() } && metadata.uid() != self.uid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "launch path belongs to another owner",
            ));
        }
        // Exact opened directory only; never walk/chown an authored tree or generation.
        if unsafe { libc::fchown(file.as_raw_fd(), uid, self.gid) } != 0
            || unsafe { libc::fchmod(file.as_raw_fd(), mode) } != 0
        {
            return Err(io::Error::last_os_error());
        }
        file.sync_all()
    }
    pub(crate) fn socket(self, path: &Path) -> io::Result<()> {
        self.validate()?;
        // O_PATH/NOFOLLOW binds ownership to the actual socket inode created by this owner.
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_PATH | libc::O_NOFOLLOW)
            .open(path)?;
        use std::os::unix::fs::FileTypeExt;
        if !file.metadata()?.file_type().is_socket() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "executor endpoint is not the owned socket",
            ));
        }
        if unsafe {
            libc::fchownat(
                file.as_raw_fd(),
                c"".as_ptr(),
                self.uid,
                self.gid,
                libc::AT_EMPTY_PATH,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub(crate) fn readable(self, path: &Path) -> io::Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        if !file.metadata()?.is_file() || file.metadata()?.uid() != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "launch metadata is not the owner's regular file",
            ));
        }
        if unsafe { libc::fchown(file.as_raw_fd(), libc::geteuid(), self.gid) } != 0
            || unsafe { libc::fchmod(file.as_raw_fd(), 0o440) } != 0
        {
            return Err(io::Error::last_os_error());
        }
        file.sync_all()
    }
}

/// Reuse Runtime's pre-CUDA seal: expected parent, parent-death SIGKILL, no_new_privs,
/// OOM score and the optional identity. Every child gets its own process group, so a kill
/// reaches the descendants that stayed in it and a terminal signal to the machine does not.
pub fn trampoline(
    python: &Path,
    identity: Option<LaunchIdentity>,
    scope: Option<&crate::cgroup::CgroupScope>,
) -> io::Result<Command> {
    let mut command = Command::new(python);
    command
        .args([
            "-I",
            "-m",
            "cozy_runtime.internal.trampoline",
            "--expect-parent",
        ])
        .arg(std::process::id().to_string())
        .args(["--oom-adj", "1000"]);
    // The trampoline joins the executor's own cgroup before anything is imported; without
    // one the process group (set below) is the containment.
    match scope {
        Some(scope) => command.args(scope.trampoline_args()),
        None => command.args(["--scope-backend", "inherit"]),
    };
    if let Some(identity) = identity {
        identity.validate()?;
        command.args([
            "--uid",
            &identity.uid.to_string(),
            "--gid",
            &identity.gid.to_string(),
        ]);
    }
    command.process_group(0).arg("--").arg(python);
    Ok(command)
}

/// Runtime `child_env.ERASED_PREFIXES`: inherited names no executor sees. The seal imposes
/// the ones it needs after this erase, so no image or operator export can redirect them.
const ERASED_PREFIXES: [&str; 17] = [
    "COZY_",
    "CUDA_",
    "PYTORCH_",
    "PYTHON",
    "TORCH_",
    "TORCHINDUCTOR_",
    "TRITON_",
    "NCCL_",
    "HF_",
    "HUGGINGFACE_",
    "TENSORHUB_",
    "CIVITAI_",
    "COMFY_",
    "OMP_",
    "MKL_",
    "FLASH_ATTENTION_",
    "TMPDIR",
];

pub fn credential_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    [
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "API_KEY",
        "AUTH",
    ]
    .iter()
    .any(|word| upper.contains(word))
}

/// The process environment every machine-launched package process starts from: the
/// machine's own environment without credentials or names the seal owns.
pub fn inherited() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(name, _)| {
            !credential_name(name) && !ERASED_PREFIXES.iter().any(|p| name.starts_with(p))
        })
        .collect()
}

/// What the machine imposes on an executor before CUDA can initialize (Runtime
/// `Worker.imposed` plus its JIT/kernel scopes). The executor reports what it received in
/// Hello and the machine compares every name both know.
#[derive(Clone, Debug)]
pub struct Seal {
    /// `CUDA_VISIBLE_DEVICES`; empty means no GPU.
    pub devices: String,
    pub alloc_conf: String,
    pub threads: u32,
    /// Degree above one: NVLS off and GPU peer memory only over NVLink (Runtime cr-068).
    pub group: bool,
    /// `COZY_HOME`: attention/kernel qualification is kept here between executors.
    pub home: PathBuf,
    /// Upstream JIT caches and `TMPDIR`, scoped to one machine incarnation and generation.
    pub jit: PathBuf,
    /// This identity's persistent kernel-store namespace.
    pub kernels: PathBuf,
    pub generation: String,
}

pub const DEFAULT_ALLOC_CONF: &str = "expandable_segments:True";
pub const DEFAULT_THREADS: u32 = 4;

impl Seal {
    /// Create the identity-owned directories under `root` for one generation.
    pub fn prepare(
        root: &Path,
        identity: Option<LaunchIdentity>,
        incarnation: &str,
        generation: &str,
        devices: &str,
    ) -> io::Result<Self> {
        if incarnation.is_empty() || !incarnation.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a machine run is named by an alphanumeric incarnation",
            ));
        }
        // Directory names never carry a caller's spelling of the generation.
        let generation = &tensorfs_core::sha256::hex_digest(generation.as_bytes())[..32];
        let uid = identity.map_or_else(|| unsafe { libc::geteuid() }, |i| i.uid);
        let namespace = format!("u{uid}");
        let owned = |path: PathBuf| -> io::Result<PathBuf> {
            match fs::create_dir(&path) {
                Ok(()) => (),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
                Err(error) => return Err(error),
            }
            match identity {
                Some(identity) => identity.own(&path)?,
                None => fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?,
            }
            Ok(path)
        };
        fs::create_dir_all(root)?;
        for boundary in ["home", "kernels", "jit"] {
            boundary_directory(&root.join(boundary))?;
        }
        boundary_directory(&root.join("jit").join(incarnation))?;
        let jit = owned(root.join("jit").join(incarnation).join(generation))?;
        // Below the disk reserve the persistent kernel store is an optional write: compiled
        // kernels then live in this run's JIT scope, removed with it.
        let kernels = if crate::reclaim::Disk::measure(root)?.below_reserve() {
            owned(jit.join("kernels"))?
        } else {
            owned(root.join("kernels").join(&namespace))?
        };
        owned(kernels.join(format!("torch-kernels.{generation}")))?;
        Ok(Self {
            devices: devices.into(),
            alloc_conf: DEFAULT_ALLOC_CONF.into(),
            threads: DEFAULT_THREADS,
            group: false,
            home: owned(root.join("home").join(&namespace))?,
            jit,
            kernels,
            generation: generation.into(),
        })
    }

    pub fn imposed(&self) -> BTreeMap<String, String> {
        let path = |path: PathBuf| path.to_string_lossy().into_owned();
        let mut imposed = BTreeMap::from([
            ("CUDA_VISIBLE_DEVICES".to_string(), self.devices.clone()),
            ("PYTORCH_CUDA_ALLOC_CONF".into(), self.alloc_conf.clone()),
            ("OMP_NUM_THREADS".into(), self.threads.to_string()),
            ("COZY_HOME".into(), path(self.home.clone())),
            ("TMPDIR".into(), path(self.jit.clone())),
            ("CUDA_CACHE_PATH".into(), path(self.jit.join("cuda"))),
            (
                "PYTHONPYCACHEPREFIX".into(),
                path(self.jit.join("bytecode")),
            ),
            (
                "TORCH_EXTENSIONS_DIR".into(),
                path(self.jit.join("extensions")),
            ),
            (
                "TORCHINDUCTOR_CACHE_DIR".into(),
                path(self.jit.join("inductor")),
            ),
            ("COZY_KERNEL_CACHE".into(), path(self.kernels.clone())),
            ("TRITON_CACHE_DIR".into(), path(self.kernels.join("triton"))),
            (
                "PYTORCH_KERNEL_CACHE_PATH".into(),
                path(
                    self.kernels
                        .join(format!("torch-kernels.{}", self.generation)),
                ),
            ),
            (
                "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR".into(),
                path(self.kernels.join("flash-attn4")),
            ),
            ("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED".into(), "1".into()),
        ]);
        if self.group {
            imposed.insert("NCCL_NVLS_ENABLE".into(), "0".into());
            imposed.insert("NCCL_P2P_LEVEL".into(), "NVL".into());
        }
        imposed
    }

    /// Inherited, then explicitly configured locations, then the seal, which always wins.
    pub fn environment(&self, configured: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        let mut environment = inherited();
        environment.extend(
            configured
                .iter()
                .filter(|(name, _)| !credential_name(name))
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        environment.extend(self.imposed());
        environment
    }

    /// Imposed names the executor did not report receiving exactly. The machine and its
    /// Runtime ship together (hard cut), so every imposed name is in its allowlist.
    pub fn mismatches(&self, reported: &BTreeMap<String, String>) -> Vec<String> {
        self.imposed()
            .into_iter()
            .filter(|(name, value)| reported.get(name) != Some(value))
            .map(|(name, _)| name)
            .collect()
    }
}

/// JIT scopes belong to one machine run (Runtime: one worker boot); the executors of
/// earlier runs are gone before a new one starts, so their scopes are unowned.
pub fn remove_stale_jit(root: &Path, incarnation: &str) {
    let Ok(entries) = fs::read_dir(root.join("jit")) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() != incarnation {
            if let Err(error) = fs::remove_dir_all(entry.path()) {
                eprintln!("stale JIT scope {}: {error}", entry.path().display());
            }
        }
    }
}

/// Owner-held directory that executors may traverse but not list.
fn boundary_directory(path: &Path) -> io::Result<()> {
    match fs::create_dir(path) {
        Ok(()) => (),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error),
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.uid() != unsafe { libc::geteuid() } {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "seal boundary is not an owner directory",
        ));
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o711))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};
    #[test]
    fn owned_paths_keep_private_metadata_closed_and_peer_metadata_readable() {
        let root = std::env::temp_dir().join(format!("machine-launch-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let identity = LaunchIdentity {
            uid: unsafe { libc::geteuid() },
            gid: unsafe { libc::getegid() },
        };
        identity.traverse(&root).unwrap();
        let peer = root.join("peer");
        fs::create_dir(&peer).unwrap();
        identity.own(&peer).unwrap();
        let private = root.join("private");
        fs::create_dir(&private).unwrap();
        fs::set_permissions(&private, fs::Permissions::from_mode(0o700)).unwrap();
        let journal = private.join("journal");
        fs::write(&journal, b"private").unwrap();
        fs::set_permissions(&journal, fs::Permissions::from_mode(0o600)).unwrap();
        let surface = peer.join("interface");
        fs::write(&surface, b"{}").unwrap();
        identity.readable(&surface).unwrap();
        let socket = peer.join("socket");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        identity.socket(&socket).unwrap();
        assert_eq!(fs::metadata(&root).unwrap().mode() & 0o777, 0o710);
        assert_eq!(fs::metadata(&peer).unwrap().mode() & 0o777, 0o700);
        assert_eq!(fs::metadata(&private).unwrap().mode() & 0o777, 0o700);
        assert_eq!(fs::metadata(&journal).unwrap().mode() & 0o777, 0o600);
        assert_eq!(fs::metadata(&surface).unwrap().mode() & 0o777, 0o440);
        assert_eq!(fs::metadata(&surface).unwrap().gid(), identity.gid);
        let alias = root.join("alias");
        std::os::unix::fs::symlink(&private, &alias).unwrap();
        assert!(identity.own(&alias).is_err());
        if unsafe { libc::geteuid() } != 0 {
            assert!(LaunchIdentity {
                uid: identity.uid + 1,
                gid: identity.gid
            }
            .validate()
            .is_err());
        }
        drop(listener);
        fs::remove_dir_all(root).unwrap();
    }
}
