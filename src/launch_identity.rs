//! Optional root-sealed identity; this is not a public caller-controlled credential.
use serde::{Deserialize, Serialize};
use std::{
    fs::OpenOptions,
    io,
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            process::CommandExt,
        },
    },
    path::Path,
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

/// Reuse Runtime's pre-CUDA seal. Defaults preserve the inherited identity/group.
pub fn trampoline(python: &Path, identity: Option<LaunchIdentity>) -> io::Result<Command> {
    let mut command = Command::new(python);
    command
        .args([
            "-I",
            "-m",
            "cozy_runtime.internal.trampoline",
            "--expect-parent",
        ])
        .arg(std::process::id().to_string())
        .args(["--oom-adj", "1000", "--scope-backend", "inherit"]);
    if let Some(identity) = identity {
        identity.validate()?;
        command.args([
            "--uid",
            &identity.uid.to_string(),
            "--gid",
            &identity.gid.to_string(),
        ]);
        command.process_group(0);
    }
    command.arg("--").arg(python);
    Ok(command)
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
