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
pub fn peer_pidfd(stream: &UnixStream) -> io::Result<File> {
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
    // Linux SO_PEERPIDFD (UAPI value 77) pins the connecting process atomically,
    // unlike a later pidfd_open of the numeric SO_PEERCRED PID. This experimental
    // host capability requires kernel support; no weaker identity is fabricated.
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
        return Err(io::Error::other(format!(
            "peer-pidfd capability unavailable: {}",
            io::Error::last_os_error()
        )));
    }
    let file = fd_result(fd)?;
    // SAFETY: valid descriptor; scalar close-on-exec flag.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
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

