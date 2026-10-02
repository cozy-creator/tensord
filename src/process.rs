//! Exact executor processes: birth identity, group kill, exit observation and the progress
//! meter. Nothing here kills because time passed. Callers kill on a measured wedge
//! (`Pace`), an owner restart, or a launch that never reached authored code.
use crate::journal::ProcessBirth;
use std::{
    fs::{self, File},
    io,
    os::fd::{AsRawFd, FromRawFd},
    time::{Duration, Instant},
};

pub fn process_birth(pid: u32) -> io::Result<ProcessBirth> {
    let stat = process_stat(pid)?;
    Ok(ProcessBirth {
        pid,
        boot_id: boot_id()?,
        start_ticks: stat.start_ticks,
    })
}

pub fn process_ended(birth: &ProcessBirth) -> io::Result<bool> {
    if boot_id()? != birth.boot_id {
        return Ok(true);
    }
    match process_stat(birth.pid) {
        Ok(stat) => Ok(stat.start_ticks != birth.start_ticks || matches!(stat.state, 'Z' | 'X')),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error),
    }
}

fn boot_id() -> io::Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .into())
}

struct Stat {
    state: char,
    pgrp: i32,
    start_ticks: u64,
}

fn process_stat(pid: u32) -> io::Result<Stat> {
    let record = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let (_, suffix) = record
        .rsplit_once(") ")
        .ok_or_else(|| io::Error::other("invalid process stat"))?;
    let fields: Vec<_> = suffix.split_whitespace().collect();
    let field = |index: usize| {
        fields
            .get(index)
            .copied()
            .ok_or_else(|| io::Error::other("process stat is truncated"))
    };
    Ok(Stat {
        state: field(0)?.chars().next().unwrap_or('?'),
        pgrp: field(2)?.parse().map_err(io::Error::other)?,
        start_ticks: field(19)?.parse().map_err(io::Error::other)?,
    })
}

/// A pidfd proven to name one recorded birth. It never names a reused PID.
pub struct Exact {
    pidfd: File,
    pub birth: ProcessBirth,
}

impl Exact {
    /// `None` when that birth has already been reaped or the PID belongs to another process.
    pub fn open(birth: &ProcessBirth) -> io::Result<Option<Self>> {
        if boot_id()? != birth.boot_id {
            return Ok(None);
        }
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, birth.pid, 0) } as i32;
        if raw < 0 {
            let error = io::Error::last_os_error();
            return match error.raw_os_error() {
                Some(libc::ESRCH) => Ok(None),
                _ => Err(error),
            };
        }
        // SAFETY: pidfd_open returned a new descriptor that we now own.
        let pidfd = unsafe { File::from_raw_fd(raw) };
        // Read after opening: the pidfd now pins this exact task.
        match process_stat(birth.pid) {
            Ok(stat) if stat.start_ticks == birth.start_ticks => Ok(Some(Self {
                pidfd,
                birth: birth.clone(),
            })),
            Ok(_) => Ok(None),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(Self {
            pidfd: self.pidfd.try_clone()?,
            birth: self.birth.clone(),
        })
    }

    pub fn as_file(&self) -> &File {
        &self.pidfd
    }

    /// SIGKILL this birth and, while it still anchors its own process group, every member.
    /// A group outlives its leader's number while members remain, so it cannot be reused.
    pub fn kill(&self) -> io::Result<()> {
        let leads_group = process_stat(self.birth.pid).is_ok_and(|stat| {
            stat.start_ticks == self.birth.start_ticks && stat.pgrp as u32 == self.birth.pid
        });
        if unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.pidfd.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        } < 0
        {
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        if leads_group && unsafe { libc::killpg(self.birth.pid as i32, libc::SIGKILL) } < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn ended(&self) -> bool {
        crate::os::ended(&self.pidfd)
    }

    /// Block until this exact process has exited. No deadline: exit is the only answer.
    pub fn wait(&self) -> io::Result<()> {
        wait_readable(&self.pidfd)
    }
}

pub fn wait_readable(file: &File) -> io::Result<()> {
    let mut poll = libc::pollfd {
        fd: file.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        let result = unsafe { libc::poll(&mut poll, 1, -1) };
        if result > 0 && poll.revents & libc::POLLIN != 0 {
            return Ok(());
        }
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if poll.revents != 0 {
            return Err(io::Error::other("process exit is not observable"));
        }
    }
}

/// CPU nanoseconds plus every byte moved (`rchar`, `wchar`, `read_bytes`, `write_bytes`):
/// Runtime's `proctree.progress_burn`. `None` is unreadable and decides nothing.
pub fn burn(pid: u32) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<_> = stat.rsplit_once(") ")?.1.split_whitespace().collect();
    let ticks = fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?;
    let hertz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64;
    let cpu = ticks.saturating_mul(1_000_000_000 / hertz);
    let moved = fs::read_to_string(format!("/proc/{pid}/io"))
        .ok()
        .and_then(|io| {
            io.lines()
                .filter_map(|line| line.split_once(": "))
                .filter(|(name, _)| {
                    matches!(*name, "rchar" | "wchar" | "read_bytes" | "write_bytes")
                })
                .map(|(_, value)| value.trim().parse::<u64>().ok())
                .sum::<Option<u64>>()
        });
    Some(cpu.saturating_add(moved.unwrap_or(0)))
}

/// Sampling resolution and noise floor of the wedge rule (Runtime `liveness.py`). A sample
/// period is how often the meter is read, never a deadline.
#[derive(Clone, Copy, Debug)]
pub struct Liveness {
    pub sample: Duration,
}
impl Default for Liveness {
    fn default() -> Self {
        Self {
            sample: Duration::from_secs(5),
        }
    }
}
impl Liveness {
    const STILL_SAMPLES: u32 = 6;
    pub fn floor(self) -> Duration {
        self.sample * Self::STILL_SAMPLES
    }
}

/// One subject's observed pace. Wedged means still for longer than eight times the longest
/// pause it has already shown, and longer than the floor.
#[derive(Clone, Debug, Default)]
pub struct Pace {
    reading: Option<u64>,
    moved_at: Option<Instant>,
    pub worst_pause: Duration,
    changes: u64,
}
impl Pace {
    const STILL_FACTOR: u32 = 8;
    pub fn seeded(worst_pause: Duration) -> Self {
        Self {
            worst_pause,
            ..Self::default()
        }
    }
    pub fn observe(&mut self, reading: Option<u64>, now: Instant) {
        let Some(reading) = reading else { return };
        match (self.reading, self.moved_at) {
            (Some(previous), _) if previous == reading => {}
            (Some(_), Some(moved_at)) => {
                self.worst_pause = self.worst_pause.max(now - moved_at);
                self.changes += 1;
                self.reading = Some(reading);
                self.moved_at = Some(now);
            }
            _ => {
                self.reading = Some(reading);
                self.moved_at = Some(now);
            }
        }
    }
    /// Time spent waiting on someone else (the machine answering a request) is not stillness.
    pub fn excuse(&mut self, now: Instant) {
        if self.moved_at.is_some() {
            self.moved_at = Some(now);
        }
    }
    pub fn still_since(&self) -> Option<Instant> {
        self.moved_at
    }
    pub fn patience(&self, floor: Duration) -> Duration {
        (self.worst_pause * Self::STILL_FACTOR).max(floor)
    }
    pub fn verdict(&self, still: Duration, floor: Duration) -> String {
        format!(
            "no measurable progress for {:.1} s; its longest earlier pause was {:.1} s over {} \
             advance(s), so it was given {:.1} s",
            still.as_secs_f64(),
            self.worst_pause.as_secs_f64(),
            self.changes,
            self.patience(floor).as_secs_f64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    #[test]
    fn pace_learns_its_own_pauses_and_judges_against_the_floor() {
        let start = Instant::now();
        let floor = Duration::from_secs(30);
        let mut pace = Pace::default();
        pace.observe(Some(1), start);
        pace.observe(Some(2), start + Duration::from_secs(10));
        assert_eq!(pace.worst_pause, Duration::from_secs(10));
        assert_eq!(pace.patience(floor), Duration::from_secs(80));
        pace.observe(None, start + Duration::from_secs(500)); // unreadable decides nothing
        pace.observe(Some(2), start + Duration::from_secs(500));
        assert_eq!(pace.still_since(), Some(start + Duration::from_secs(10)));
        assert_eq!(Pace::default().patience(floor), floor);
    }

    #[test]
    fn exact_kill_takes_the_group_and_never_a_reused_birth() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "sleep 1000 & wait"])
            .process_group(0)
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let birth = process_birth(child.id()).unwrap();
        let mut stale = birth.clone();
        stale.start_ticks += 1;
        assert!(Exact::open(&stale).unwrap().is_none());
        let exact = Exact::open(&birth).unwrap().unwrap();
        // The grandchild is in the leader's group: wait until it exists.
        let grandchild = loop {
            let children =
                fs::read_to_string(format!("/proc/{0}/task/{0}/children", child.id())).unwrap();
            if let Some(pid) = children.split_whitespace().next() {
                break pid.parse::<u32>().unwrap();
            }
            std::thread::yield_now();
        };
        let grandchild = Exact::open(&process_birth(grandchild).unwrap())
            .unwrap()
            .unwrap();
        exact.kill().unwrap();
        exact.wait().unwrap();
        grandchild.wait().unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
        assert!(process_ended(&birth).unwrap());
    }

    #[test]
    fn burn_reads_this_process() {
        assert!(burn(std::process::id()).is_some_and(|value| value > 0));
        assert!(burn(u32::MAX).is_none());
    }
}
