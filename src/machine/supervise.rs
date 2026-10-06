//! The machine's stable parent, forked before any thread exists. It reaps orphans (a pod's
//! PID 1 must), forwards stop signals and starts the service again after an abnormal exit, as
//! the Go guardian restarts its request plane. A clean exit (a stop or an accepted release)
//! ends it, and so does a second consecutive exit before readiness: measured progress, no timer.
//! The parent is never replaced: an activated Runtime update runs as the service child
//! (`tensord service`, readiness pipe on fd 3), and a candidate that never proves
//! readiness is rolled back by this same parent.
use nix::{
    errno::Errno,
    fcntl::OFlag,
    sys::{
        signal::{kill, SigSet, Signal},
        wait::{waitpid, WaitPidFlag, WaitStatus},
    },
    unistd::{fork, pipe2, ForkResult, Pid},
};
use std::{
    fs::File,
    io::{self, Read, Write},
    os::fd::{AsRawFd, FromRawFd},
};

/// The fd an exec'd service reports readiness on.
const READY_FD: i32 = 3;

/// Forks the service. Returns only in a service child that runs this executable, with the pipe
/// it reports readiness on; the parent exits with the service's final status.
pub fn supervise(paths: &super::update::Paths) -> io::Result<Ready> {
    let mut signals = SigSet::empty();
    for signal in [Signal::SIGTERM, Signal::SIGINT, Signal::SIGCHLD] {
        signals.add(signal);
    }
    signals.thread_block().map_err(io::Error::from)?;
    // Orphans of the service reparent here, not to a PID 1 that may not reap them.
    nix::sys::prctl::set_child_subreaper(true).map_err(io::Error::from)?;
    let parent = nix::unistd::getpid();
    let mut failures_before_ready = 0;
    loop {
        let activated = super::update::activated_binary(paths);
        let (read, write) = pipe2(OFlag::O_CLOEXEC).map_err(io::Error::from)?;
        // SAFETY: no thread exists yet in this process; the child continues single-threaded.
        let child = match unsafe { fork() }.map_err(io::Error::from)? {
            ForkResult::Child => {
                drop(read);
                // The service never outlives its supervisor.
                nix::sys::prctl::set_pdeathsig(Signal::SIGKILL).map_err(io::Error::from)?;
                if nix::unistd::getppid() != parent {
                    std::process::exit(1);
                }
                SigSet::empty().thread_set_mask().map_err(io::Error::from)?;
                let Some(binary) = activated else {
                    return Ok(Ready(Some(File::from(write))));
                };
                exec_service(&binary, write);
            }
            ForkResult::Parent { child } => child,
        };
        drop(write);
        let (status, stopping) = wait_service(child, &signals)?;
        let mut byte = [0u8; 1];
        let ready = File::from(read).read(&mut byte).unwrap_or(0) == 1;
        let code = match status {
            WaitStatus::Exited(_, code) => code,
            WaitStatus::Signaled(_, signal, _) => 128 + signal as i32,
            _ => 1,
        };
        if stopping || code == 0 {
            std::process::exit(code);
        }
        if code == super::update::REPLACE_EXIT {
            eprintln!("tensord: starting the service on its updated software");
            failures_before_ready = 0;
            continue;
        }
        failures_before_ready = if ready { 0 } else { failures_before_ready + 1 };
        if failures_before_ready >= 2 {
            let cause = format!("the service exited twice before readiness ({code})");
            if super::update::rollback_pending(paths, &cause)? {
                eprintln!("tensord: {cause}; Runtime update rolled back");
                failures_before_ready = 0;
                continue;
            }
            eprintln!("tensord: {cause}; stopping");
            std::process::exit(code);
        }
        eprintln!("tensord: the service exited ({code}); starting it again");
    }
}

/// Runs an activated machine binary as this service child; never returns.
fn exec_service(binary: &std::path::Path, ready: std::os::fd::OwnedFd) -> ! {
    use std::os::unix::process::CommandExt;
    // SAFETY: plain fd calls in a single-threaded child; dup2 clears close-on-exec on the copy.
    let placed = unsafe {
        if ready.as_raw_fd() == READY_FD {
            libc::fcntl(READY_FD, libc::F_SETFD, 0)
        } else {
            libc::dup2(ready.as_raw_fd(), READY_FD)
        }
    };
    if placed < 0 {
        std::process::exit(1);
    }
    std::mem::forget(ready);
    let error = std::process::Command::new(binary).arg("service").exec();
    eprintln!("tensord: cannot run {}: {error}", binary.display());
    std::process::exit(1)
}

/// The readiness pipe an exec'd service inherited from its parent.
pub fn inherited() -> Ready {
    // SAFETY: the parent placed the pipe's write end at READY_FD before exec.
    Ready(Some(unsafe { File::from_raw_fd(READY_FD) }))
}

/// Waits for the service to end, reaping every orphan meanwhile and forwarding stop signals.
fn wait_service(child: Pid, signals: &SigSet) -> io::Result<(WaitStatus, bool)> {
    let mut stopping = false;
    loop {
        match signals.wait().map_err(io::Error::from)? {
            signal @ (Signal::SIGTERM | Signal::SIGINT) => {
                stopping = true;
                let _ = kill(child, signal);
            }
            _ => loop {
                match waitpid(None, Some(WaitPidFlag::WNOHANG)) {
                    Ok(WaitStatus::StillAlive) | Err(Errno::ECHILD) => break,
                    Ok(status) if status.pid() == Some(child) => return Ok((status, stopping)),
                    Ok(_) | Err(Errno::EINTR) => continue,
                    Err(error) => return Err(error.into()),
                }
            },
        }
    }
}

/// The service's readiness report to its parent: written once, when this boot has proved itself.
pub struct Ready(Option<File>);
impl Ready {
    pub fn report(&mut self) {
        if let Some(mut pipe) = self.0.take() {
            let _ = pipe.write_all(b"r");
        }
    }
}
