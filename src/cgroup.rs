//! An executor's own cgroup-v2 scope below the machine's (Runtime `proctree.CgroupScope`).
//! `cgroup.kill` reaches every descendant, including those that call `setsid` or
//! double-fork. Lifecycle containment, not a sandbox: same-UID code that can write a
//! delegated ancestor's `cgroup.procs` can still move itself out.
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, Write},
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::{Path, PathBuf},
};

const ROOT: &str = "/sys/fs/cgroup";

/// One exact scope directory, protected against path reuse by its inode.
#[derive(Debug)]
pub struct CgroupScope {
    pub relative: String,
    inode: u64,
}

/// Opaque, stable per state root, so a machine only ever sweeps its own scopes.
pub fn namespace(state: &Path) -> String {
    let path = state.canonicalize().unwrap_or_else(|_| state.to_path_buf());
    tensorfs_core::sha256::hex_digest(path.as_os_str().as_encoded_bytes())[..16].to_string()
}

fn own_cgroup() -> io::Result<PathBuf> {
    let entries = fs::read_to_string("/proc/self/cgroup")?;
    let relative = entries
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| io::Error::other("no unified cgroup-v2 membership"))?;
    Ok(Path::new(ROOT).join(relative.trim().trim_start_matches('/')))
}

fn writable_unified() -> bool {
    if !Path::new(ROOT).join("cgroup.controllers").is_file() {
        return false;
    }
    let path = std::ffi::CString::new(ROOT).unwrap();
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: valid path and writable statvfs.
    unsafe { libc::statvfs(path.as_ptr(), &mut stat) == 0 && stat.f_flag & libc::ST_RDONLY == 0 }
}

impl CgroupScope {
    /// A fresh scope below the machine's cgroup. `None` where the unified hierarchy is
    /// read-only or not delegated (most containers): the process group then contains it.
    pub fn create(namespace: &str) -> io::Result<Option<Self>> {
        if !writable_unified() {
            return Ok(None);
        }
        let parent = own_cgroup()?;
        let name = format!(
            "cozy-executor-{namespace}-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..16]
        );
        let path = parent.join(name);
        if let Err(error) = fs::create_dir(&path) {
            return match error.kind() {
                io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem => Ok(None),
                _ => Err(error),
            };
        }
        // An undelegated hierarchy shows here, before any executor exists.
        let usable = ["cgroup.procs", "cgroup.kill"]
            .iter()
            .all(|name| OpenOptions::new().write(true).open(path.join(name)).is_ok())
            && path.join("cgroup.events").is_file();
        if !usable {
            let _ = fs::remove_dir(&path);
            return Ok(None);
        }
        Ok(Some(Self {
            relative: format!("/{}", path.strip_prefix(ROOT).unwrap().display()),
            inode: fs::metadata(&path)?.ino(),
        }))
    }

    fn path(&self) -> io::Result<PathBuf> {
        let path = Path::new(ROOT).join(self.relative.trim_start_matches('/'));
        if fs::metadata(&path)?.ino() != self.inode {
            return Err(io::Error::other(format!(
                "executor cgroup {} was replaced",
                self.relative
            )));
        }
        Ok(path)
    }

    /// The trampoline joins this exact scope before it imports anything.
    pub fn trampoline_args(&self) -> [String; 6] {
        [
            "--scope-backend".into(),
            "cgroup_v2".into(),
            "--cgroup".into(),
            self.relative.clone(),
            "--cgroup-inode".into(),
            self.inode.to_string(),
        ]
    }

    /// Live processes in the scope and every nested cgroup.
    pub fn processes(&self) -> io::Result<usize> {
        fn count(path: &Path) -> io::Result<usize> {
            let mut total = fs::read_to_string(path.join("cgroup.procs"))?
                .lines()
                .filter(|line| !line.is_empty())
                .count();
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    total += count(&entry.path())?;
                }
            }
            Ok(total)
        }
        count(&self.path()?)
    }

    /// Move a running process into this scope (a fork starts in its parent's).
    pub fn adopt(&self, pid: u32) -> io::Result<()> {
        OpenOptions::new()
            .write(true)
            .open(self.path()?.join("cgroup.procs"))?
            .write_all(pid.to_string().as_bytes())
    }

    pub fn kill(&self) -> io::Result<()> {
        OpenOptions::new()
            .write(true)
            .open(self.path()?.join("cgroup.kill"))?
            .write_all(b"1")
    }

    /// Block until nothing in the scope is alive: woken by `cgroup.events`, no deadline.
    pub fn wait_empty(&self) -> io::Result<()> {
        let mut events = File::open(self.path()?.join("cgroup.events"))?;
        loop {
            let mut text = String::new();
            events.rewind()?;
            events.read_to_string(&mut text)?;
            if text.lines().any(|line| line == "populated 0") {
                return Ok(());
            }
            let mut poll = libc::pollfd {
                fd: events.as_raw_fd(),
                events: libc::POLLPRI,
                revents: 0,
            };
            // SAFETY: one live descriptor; kernfs wakes it when the file changes.
            if unsafe { libc::poll(&mut poll, 1, -1) } < 0 {
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            }
        }
    }

    /// Kill whatever outlived the executor, wait for it, and remove the scope.
    /// Returns how many processes were still inside.
    pub fn end(&self) -> io::Result<usize> {
        let stragglers = self.processes()?;
        if stragglers > 0 {
            self.kill()?;
        }
        self.wait_empty()?;
        remove_tree(&self.path()?)?;
        Ok(stragglers)
    }

    /// At machine start every executor scope of this namespace belongs to an earlier run:
    /// kill, wait for and remove each. Returns how many processes they still held.
    pub fn sweep(namespace: &str) -> io::Result<usize> {
        if !writable_unified() {
            return Ok(0);
        }
        let prefix = format!("cozy-executor-{namespace}-");
        let mut held = 0;
        for entry in fs::read_dir(own_cgroup()?)? {
            let entry = entry?;
            if !entry.file_name().to_string_lossy().starts_with(&prefix) {
                continue;
            }
            let scope = Self {
                relative: format!("/{}", entry.path().strip_prefix(ROOT).unwrap().display()),
                inode: entry.metadata()?.ino(),
            };
            held += scope.end()?;
        }
        Ok(held)
    }
}

fn remove_tree(path: &Path) -> io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            remove_tree(&entry.path())?;
        }
    }
    fs::remove_dir(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    #[test]
    fn kill_reaches_a_setsid_descendant_and_the_scope_is_removed() {
        let Some(scope) = CgroupScope::create("test").unwrap() else {
            eprintln!("no delegated cgroup-v2 here; process-group containment applies");
            return;
        };
        // The leader detaches a daemon into its own session, then exits.
        let mut leader = Command::new("/bin/sh")
            .args(["-c", "read go; setsid sleep 1000 </dev/null >/dev/null 2>&1 & echo started"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        fs::write(scope.path().unwrap().join("cgroup.procs"), leader.id().to_string()).unwrap();
        leader.stdin.take().unwrap().write_all(b"go\n").unwrap();
        let mut started = String::new();
        leader.stdout.take().unwrap().read_to_string(&mut started).unwrap();
        assert!(leader.wait().unwrap().success());
        assert_eq!(started.trim(), "started");
        // The leader is gone; its setsid daemon is not, and the scope still counts it.
        assert_eq!(scope.processes().unwrap(), 1);
        let path = scope.path().unwrap();
        assert_eq!(scope.end().unwrap(), 1);
        assert!(!path.exists());
    }
}
