//! Exact executor processes: birth identity, group kill, exit observation and the progress
//! meter. Nothing here kills because time passed. Callers kill on a measured wedge
//! (`Pace`), an owner restart, or a launch that never reached authored code.
use crate::journal::ProcessBirth;
use std::{
    fs::{self, File},
    io,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::process::ExitStatusExt,
    },
    process::{Child, ExitStatus},
    sync::{Arc, Condvar, Mutex},
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
    /// The executor's own cgroup, when the host delegates one: kills reach every descendant.
    cgroup: Option<Arc<crate::cgroup::CgroupScope>>,
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
                // ESRCH: reaped. EINVAL: a leader already exiting (its threads tearing down).
                Some(libc::ESRCH | libc::EINVAL) => Ok(None),
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
                cgroup: None,
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
            cgroup: self.cgroup.clone(),
        })
    }

    pub fn with_cgroup(mut self, cgroup: Option<Arc<crate::cgroup::CgroupScope>>) -> Self {
        self.cgroup = cgroup;
        self
    }

    pub fn as_file(&self) -> &File {
        &self.pidfd
    }

    pub fn into_file(self) -> File {
        self.pidfd
    }

    /// SIGKILL this birth and everything it started: its cgroup when it has one, else the
    /// process group it still anchors (a group outlives its leader's number while members
    /// remain, so it cannot be reused).
    pub fn kill(&self) -> io::Result<()> {
        if let Some(cgroup) = &self.cgroup {
            cgroup.kill()?;
        }
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

/// Every process still in `leader`'s process group and born with or after it: an executor's
/// followers (Runtime `RankGroup` keeps rank 0's PGID). A group number is not reused while a
/// member lives, so a scan before or right after the leader is reaped names only its own.
pub fn group_members(leader: &ProcessBirth) -> Vec<ProcessBirth> {
    let Ok(boot) = boot_id() else {
        return vec![];
    };
    if boot != leader.boot_id {
        return vec![];
    }
    let Ok(entries) = fs::read_dir("/proc") else {
        return vec![];
    };
    entries
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| *pid != leader.pid)
        .filter_map(|pid| {
            let stat = process_stat(pid).ok()?;
            (stat.pgrp as u32 == leader.pid && stat.start_ticks >= leader.start_ticks).then(|| {
                ProcessBirth {
                    pid,
                    boot_id: boot.clone(),
                    start_ticks: stat.start_ticks,
                }
            })
        })
        .collect()
}

/// Whether a member of `leader`'s group born before `before` (start ticks) still lives: a
/// previous machine's follower, killed with its leader, until its exit is observed.
pub fn group_outlives(leader: &ProcessBirth, before: u64) -> bool {
    group_members(leader)
        .iter()
        .any(|member| member.start_ticks < before && !process_ended(member).unwrap_or(true))
}

/// This process's own birth, in start ticks.
pub fn own_start_ticks() -> io::Result<u64> {
    Ok(process_stat(std::process::id())?.start_ticks)
}

/// The leader's meter plus its group members': a follower's imports, fills and collectives
/// are the executor's work. `None` only when the leader itself is unreadable.
pub fn group_burn(leader: &ProcessBirth) -> Option<u64> {
    let own = burn(leader.pid)?;
    Some(
        group_members(leader)
            .iter()
            .filter_map(|member| burn(member.pid))
            .fold(own, u64::saturating_add),
    )
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Meter {
    /// CPU plus bytes moved by the process (Runtime `ExecutorChild.watched`).
    Burn,
    /// Frames of the running invocation, judged only after a cancel (Runtime `cancel_stall`).
    Frames,
}

#[derive(Default)]
struct WatchState {
    done: bool,
    serving: bool,
    frames: u64,
    canceled_at: Option<Instant>,
    killed: Option<String>,
    worst_gap: Duration,
}

/// One exchange's progress observer, shared with whoever may cancel it. It kills the exact
/// process (and its group) only on a measured wedge; the kill closes the channel, so the
/// blocked exchange returns.
pub struct Watch {
    meter: Meter,
    state: Mutex<WatchState>,
    changed: Condvar,
}
impl Watch {
    fn run(&self, exact: Exact, liveness: Liveness, worst_gap: Duration, what: &str) {
        let floor = liveness.floor();
        // Frame gaps teach only the frame meter; CPU-plus-bytes starts its own pace.
        let mut pace = match self.meter {
            Meter::Frames => Pace::seeded(worst_gap),
            Meter::Burn => Pace::default(),
        };
        let mut state = self.state.lock().unwrap();
        while !state.done {
            let now = Instant::now();
            let reading = match self.meter {
                Meter::Burn => group_burn(&exact.birth),
                Meter::Frames => Some(state.frames),
            };
            pace.observe(reading, now);
            if state.serving {
                pace.excuse(now);
            }
            state.worst_gap = pace.worst_pause;
            let since = match self.meter {
                Meter::Burn => pace.still_since(),
                Meter::Frames => state
                    .canceled_at
                    .map(|at| pace.still_since().map_or(at, |moved| moved.max(at))),
            };
            if let Some(since) = since {
                let still = now - since;
                if still > pace.patience(floor) {
                    let verdict = format!("wedged during {what}: {}", pace.verdict(still, floor));
                    match exact.kill() {
                        Ok(()) => state.killed = Some(verdict),
                        Err(error) => eprintln!("{verdict}; kill failed: {error}"),
                    }
                    return;
                }
            }
            state = self.changed.wait_timeout(state, liveness.sample).unwrap().0;
        }
    }
    pub fn frame(&self) {
        self.state.lock().unwrap().frames += 1;
    }
    /// Time the machine spends answering this process is not the process's stillness.
    pub fn serving(&self, serving: bool) {
        self.state.lock().unwrap().serving = serving;
    }
    pub fn canceled(&self) {
        let mut state = self.state.lock().unwrap();
        state.canceled_at.get_or_insert_with(Instant::now);
        self.changed.notify_all();
    }
}

/// A running watch; dropping it stops the observer on every path.
pub struct Watching {
    watch: Arc<Watch>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Watching {
    pub fn start(
        exact: Exact,
        meter: Meter,
        liveness: Liveness,
        worst_gap: Duration,
        what: &'static str,
    ) -> io::Result<Self> {
        let watch = Arc::new(Watch {
            meter,
            state: Mutex::new(WatchState::default()),
            changed: Condvar::new(),
        });
        let observer = watch.clone();
        let thread = std::thread::Builder::new()
            .name(format!("watch-{what}"))
            .spawn(move || observer.run(exact, liveness, worst_gap, what))?;
        Ok(Self {
            watch,
            thread: Some(thread),
        })
    }
    pub fn watch(&self) -> Arc<Watch> {
        self.watch.clone()
    }
    /// The kill's measurement, if one was needed, and the longest frame gap seen.
    pub fn finish(mut self) -> (Option<String>, Duration) {
        self.stop();
        let state = self.watch.state.lock().unwrap();
        (state.killed.clone(), state.worst_gap)
    }
    fn stop(&mut self) {
        if let Some(thread) = self.thread.take() {
            self.watch.state.lock().unwrap().done = true;
            self.watch.changed.notify_all();
            let _ = thread.join();
        }
    }
}
impl Drop for Watching {
    fn drop(&mut self) {
        self.stop();
    }
}

/// After its channel closed: wait for this exact process to exit, killing it only on a
/// measured wedge, then reap it. The sample period is the meter's cadence, not a deadline.
/// Then its cgroup: whatever outlived it (a `setsid` daemon) is killed and counted, and the
/// scope is removed.
pub fn reap(
    exact: &Exact,
    child: Option<&mut Child>,
    liveness: Liveness,
) -> io::Result<Reaped> {
    let floor = liveness.floor();
    let mut pace = Pace::default();
    let mut killed = None;
    while !readable_within(exact.as_file(), liveness.sample)? {
        let now = Instant::now();
        pace.observe(group_burn(&exact.birth), now);
        if killed.is_some() {
            continue;
        }
        if let Some(since) = pace.still_since() {
            let still = now - since;
            if still > pace.patience(floor) {
                exact.kill()?;
                killed = Some(format!(
                    "wedged while ending: {}",
                    pace.verdict(still, floor)
                ));
            }
        }
    }
    let status = match child {
        Some(child) => child.wait()?,
        None => ExitStatus::from_raw(0),
    };
    let stragglers = match &exact.cgroup {
        Some(cgroup) => cgroup.end()?,
        None => 0,
    };
    Ok(Reaped {
        status,
        killed,
        stragglers,
    })
}

/// How a process ended: its status, the measurement behind a kill, and how many of its
/// descendants were still alive in its cgroup and were killed after it.
#[derive(Debug)]
pub struct Reaped {
    pub status: ExitStatus,
    pub killed: Option<String>,
    pub stragglers: usize,
}

/// `reap` the leader, then every member left in its group (followers die with their leader
/// by parent-death signal; their device teardown is waited for, a wedged one killed).
pub fn reap_group(
    exact: &Exact,
    child: Option<&mut Child>,
    liveness: Liveness,
) -> io::Result<Reaped> {
    // With a cgroup the leader's reap already ended its followers (they share its scope).
    let mut reaped = reap(exact, child, liveness)?;
    for member in group_members(&exact.birth) {
        let Some(member) = Exact::open(&member)? else {
            continue;
        };
        if let Some(verdict) = reap(&member, None, liveness)?.killed {
            reaped.killed = Some(match reaped.killed.take() {
                Some(first) => format!("{first}; process {}: {verdict}", member.birth.pid),
                None => format!("process {}: {verdict}", member.birth.pid),
            });
        }
    }
    Ok(reaped)
}

fn readable_within(file: &File, period: Duration) -> io::Result<bool> {
    let mut poll = libc::pollfd {
        fd: file.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let milliseconds = period.as_millis().clamp(1, i32::MAX as u128) as i32;
    loop {
        let result = unsafe { libc::poll(&mut poll, 1, milliseconds) };
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if result == 0 {
            return Ok(false);
        }
        if poll.revents & libc::POLLIN != 0 {
            return Ok(true);
        }
        return Err(io::Error::other("process exit is not observable"));
    }
}

/// An OS shortage that may clear by itself; nothing else justifies requeueing a launch.
pub fn transient(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EAGAIN | libc::ENOMEM | libc::EMFILE | libc::ENFILE | libc::EINTR)
    )
}

/// The last bytes of a log, for a failure reason.
pub fn tail(path: &std::path::Path) -> String {
    const TAIL: u64 = 2048;
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    use std::io::{Read, Seek};
    let length = file.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = file.seek(io::SeekFrom::Start(length.saturating_sub(TAIL)));
    let mut bytes = Vec::new();
    let _ = file.read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).trim().to_string()
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
    fn reap_group_waits_for_members_left_behind_and_ends_a_still_one() {
        // The leader leaves two members in its group; killing only the leader (no group
        // signal) orphans them, as a follower can outlive rank 0 for a moment.
        let mut child = Command::new("/bin/sh")
            .args(["-c", "sleep 1000 & sleep 1000 & wait"])
            .process_group(0)
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let birth = process_birth(child.id()).unwrap();
        let members = loop {
            let members = group_members(&birth);
            if members.len() == 2 {
                break members;
            }
            std::thread::yield_now();
        };
        let exact = Exact::open(&birth).unwrap().unwrap();
        unsafe { libc::kill(birth.pid as i32, libc::SIGKILL) };
        let liveness = Liveness {
            sample: Duration::from_millis(50),
        };
        let reaped = reap_group(&exact, Some(&mut child), liveness).unwrap();
        let (status, killed) = (reaped.status, reaped.killed);
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        // A sleeping member shows no progress: it is ended on that measurement, then waited.
        let killed = killed.expect("still members are ended on their measurement");
        assert!(killed.contains("no measurable progress"), "{killed}");
        for member in members {
            assert!(process_ended(&member).unwrap(), "{member:?}");
        }
        assert!(group_members(&birth).is_empty());
    }

    #[test]
    fn burn_reads_this_process() {
        assert!(burn(std::process::id()).is_some_and(|value| value > 0));
        assert!(burn(u32::MAX).is_none());
    }
}
