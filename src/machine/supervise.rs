//! The machine's stable parent, forked before any thread exists. It reaps orphans (a pod's
//! PID 1 must), forwards stop signals and starts the service again after an abnormal exit, as
//! the Go guardian restarts its request plane. A clean exit (a stop or an accepted release)
//! ends it, and so does a second consecutive exit before readiness: measured progress, no timer.
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
};

/// Forks the service. Returns only in the service child, with the pipe it reports readiness
/// on; the parent exits with the service's final status.
pub fn supervise() -> io::Result<Ready> {
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
                return Ok(Ready(Some(File::from(write))));
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
        failures_before_ready = if ready { 0 } else { failures_before_ready + 1 };
        if failures_before_ready >= 2 {
            eprintln!("cozy-machine: the service exited twice before readiness ({code}); stopping");
            std::process::exit(code);
        }
        eprintln!("cozy-machine: the service exited ({code}); starting it again");
    }
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
