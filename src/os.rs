use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;

pub const FULL_SEALS: i32 =
    libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
fn fd_result(fd: i32) -> io::Result<File> {
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a successful create/open syscall transfers this descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}
pub fn memfd() -> io::Result<File> {
    let name = CString::new("cozy-machine-weights").unwrap();
    // SAFETY: valid NUL-terminated name, no borrowed output pointers.
    fd_result(unsafe {
        libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
    })
}
pub fn seals(file: &File) -> io::Result<i32> {
    // SAFETY: live descriptor; fcntl has no pointer argument for this operation.
    let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    if result < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(result)
}
pub fn seal(file: &File) -> io::Result<()> {
    // SAFETY: live descriptor, scalar seal bitmask.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, FULL_SEALS) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
pub fn peer_credentials(stream: &UnixStream) -> io::Result<libc::ucred> {
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: correctly sized writable credential structure and size pointer.
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut cred as *mut _ as *mut libc::c_void,
            &mut size,
        )
    } < 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: geteuid has no memory arguments.
    if cred.uid != unsafe { libc::geteuid() } {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "peer UID differs",
        ));
    }
    Ok(cred)
}

/// Whether the connecting process descends from this machine (an executor, runner or
/// anything package code started). Same-UID package code is not a sandbox, but it cannot
/// administer the machine through its own process tree.
pub fn peer_descends_from_machine(stream: &UnixStream) -> io::Result<bool> {
    let mut pid = peer_credentials(stream)?.pid;
    let machine = std::process::id() as i32;
    while pid > 1 {
        if pid == machine {
            return Ok(true);
        }
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
        pid = stat
            .rsplit_once(") ")
            .and_then(|(_, rest)| rest.split_whitespace().nth(1))
            .and_then(|ppid| ppid.parse().ok())
            .ok_or_else(|| io::Error::other("invalid process stat"))?;
    }
    Ok(false)
}

pub fn peer_pidfd(stream: &UnixStream) -> io::Result<File> {
    let cred = peer_credentials(stream)?;
    // Linux SO_PEERPIDFD (UAPI value 77, kernel 6.5) pins the connecting process atomically.
    let mut fd: i32 = -1;
    let mut fd_size = std::mem::size_of::<i32>() as libc::socklen_t;
    // SAFETY: correctly sized scalar output and size pointer.
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            77,
            &mut fd as *mut _ as *mut libc::c_void,
            &mut fd_size,
        )
    } < 0
    {
        let error = io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ENOPROTOOPT | libc::EINVAL) => pidfd_of_live(cred.pid),
            _ => Err(error),
        };
    }
    let file = fd_result(fd)?;
    // SAFETY: valid descriptor; scalar close-on-exec flag.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

/// Older kernels: pin the credential's PID with pidfd_open, accepted only when the same
/// birth is read on both sides of the open. A peer that exited and whose PID was reused in
/// that instant is the remaining (negligible) window.
fn pidfd_of_live(pid: i32) -> io::Result<File> {
    let birth = crate::process::process_birth(u32::try_from(pid).map_err(io::Error::other)?)?;
    crate::process::Exact::open(&birth)?
        .map(crate::process::Exact::into_file)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "peer exited before it was pinned"))
}

pub fn ended(pidfd: &File) -> bool {
    let mut poll = libc::pollfd {
        fd: pidfd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live poll descriptor, instantaneous readiness observation, no elapsed-time policy.
    unsafe { libc::poll(&mut poll, 1, 0) > 0 && poll.revents != 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn older_kernel_fallback_pins_a_live_peer_and_refuses_a_gone_one() {
        let pinned = pidfd_of_live(std::process::id() as i32).unwrap();
        assert!(!ended(&pinned));
        let mut child = std::process::Command::new("/bin/true").spawn().unwrap();
        let pid = child.id() as i32;
        child.wait().unwrap();
        assert!(pidfd_of_live(pid).is_err());
    }

    #[test]
    fn admin_peer_check_sees_the_machines_own_descendants() {
        let (same, _keep) = UnixStream::pair().unwrap();
        assert!(peer_descends_from_machine(&same).unwrap());
        let path = std::env::temp_dir().join(format!("admin-peer-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "exec python3 -c \"import socket,sys;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);s.recv(1)\" \"$0\"", path.to_str().unwrap()])
            .spawn()
            .unwrap();
        let (grandchild, _) = listener.accept().unwrap();
        assert!(peer_descends_from_machine(&grandchild).unwrap());
        drop(grandchild);
        child.wait().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
