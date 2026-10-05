//! The host asks for memory back (MEM/HOST-PRESSURE.md). A PSI trigger wakes one thread blocked
//! in poll() with no timeout: in `shared` mode on the whole system's memory pressure (anyone
//! stalled, the owner's apps included), in `dedicated` mode on this cgroup's (every task of it
//! stalled). Each event gives one rung back while each rung lowers the stall; one that did not
//! stops it, since that stall is not ours to fix (page-cache refaults on a slow disk), until the
//! stall rises above where it stopped (`Feedback`).
use serde::Deserialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, OpenOptionsExt},
    },
    path::PathBuf,
    time::{Duration, Instant},
};

/// How the machine shares its host (`host.mode` in the GPU config): a guest on someone's
/// computer (the default), or the whole of a rented pod (its image or grant says so).
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HostMode {
    #[default]
    Shared,
    Dedicated,
}

/// The trigger's window: the shortest an unprivileged process may arm (a multiple of 2 s on
/// Linux 6.x and 7.0). The threshold is the smallest measurable stall, so an event means
/// "something stalled on memory in the last window".
pub const WINDOW: Duration = Duration::from_secs(2);

/// One armed PSI trigger, and in dedicated mode this cgroup's `memory.events` (its `high`
/// and `max` counters: throttled or at the wall), which wakes poll() when they change.
pub struct Pressure {
    trigger: File,
    path: PathBuf,
    kind: &'static str,
    events: Option<File>,
}

impl Pressure {
    pub fn arm(mode: HostMode) -> io::Result<Self> {
        let cgroup = own_cgroup().filter(|cgroup| cgroup.join("memory.pressure").exists());
        match (mode, cgroup) {
            (HostMode::Dedicated, Some(cgroup)) => {
                let mut pressure = Self::arm_at(cgroup.join("memory.pressure"), "full")?;
                pressure.events = File::open(cgroup.join("memory.events")).ok();
                Ok(pressure)
            }
            (HostMode::Dedicated, None) => Self::arm_at("/proc/pressure/memory".into(), "full"),
            (HostMode::Shared, _) => Self::arm_at("/proc/pressure/memory".into(), "some"),
        }
    }

    /// A trigger on one pressure file (`memory.pressure` of a cgroup, or the system's).
    pub fn arm_at(path: PathBuf, kind: &'static str) -> io::Result<Self> {
        let mut trigger = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)?;
        trigger.write_all(format!("{kind} 1 {}\0", WINDOW.as_micros()).as_bytes())?;
        Ok(Self {
            trigger,
            path,
            kind,
            events: None,
        })
    }

    /// Blocks until the kernel reports a stall (or, dedicated, a `memory.events` change): no
    /// timeout, no work in between.
    pub fn wait(&self) -> io::Result<()> {
        let mut fds = [self.trigger.as_raw_fd(), -1].map(|fd| libc::pollfd {
            fd,
            events: libc::POLLPRI,
            revents: 0,
        });
        if let Some(events) = &self.events {
            fds[1].fd = events.as_raw_fd();
            // kernfs re-arms a change notice when this descriptor reads the file
            let _ = events.read_at(&mut [0u8; 512], 0);
        }
        loop {
            // SAFETY: live descriptors (-1 is ignored by poll).
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) } > 0 {
                return match fds[0].revents & libc::POLLERR {
                    0 => Ok(()),
                    _ => Err(io::Error::other("the PSI trigger is gone")),
                };
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }

    /// Microseconds stalled so far, of this trigger's kind.
    pub fn stalled_us(&self) -> io::Result<u64> {
        let text = fs::read_to_string(&self.path)?;
        text.lines()
            .find(|line| line.starts_with(self.kind))
            .and_then(|line| line.split_whitespace().find_map(|f| f.strip_prefix("total=")))
            .and_then(|total| total.parse().ok())
            .ok_or_else(|| io::Error::other("no PSI total"))
    }
}

/// This process's cgroup v2 directory.
fn own_cgroup() -> Option<PathBuf> {
    let text = fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = text.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}

/// Whether the next event gives a rung back. Each event brings the share of time stalled since
/// the one before. After a rung, a share lower than the share before it by more than the noise
/// says the rung helped, and another may go; one that is not stops giving, and the noise is then
/// measured: the largest change between consecutive shares while nothing is given. Giving
/// resumes only once the share rises above where it stopped by more than that noise. So a stall
/// giving back cannot touch (page-cache refaults on a slow disk) sheds a rung or two, however
/// long it lasts, while a hog's stall falls rung by rung and a rising one is answered. Nothing
/// here is a constant: the window is the trigger's, the noise is measured. An event more than
/// two windows after the last means a whole window passed without a stall: a new episode.
#[derive(Debug)]
pub struct Feedback {
    total: u64,
    at: Instant,
    last: Option<f64>,
    rung: Option<f64>,
    stopped: Option<f64>,
    noise: Option<f64>,
}

impl Feedback {
    pub fn new(total: u64, at: Instant) -> Self {
        Self {
            total,
            at,
            last: None,
            rung: None,
            stopped: None,
            noise: None,
        }
    }

    pub fn give(&mut self, total: u64, at: Instant) -> bool {
        let elapsed = at.saturating_duration_since(self.at);
        let share = total.saturating_sub(self.total) as f64 / elapsed.as_micros().max(1) as f64;
        (self.total, self.at) = (total, at);
        if elapsed > 2 * WINDOW {
            *self = Self::new(total, at);
        }
        let previous = self.last.replace(share);
        if let Some(level) = self.stopped {
            if self.noise.is_some_and(|noise| share > level + noise) {
                (self.stopped, self.rung) = (None, Some(share));
                return true;
            }
            if let Some(previous) = previous {
                let change = (share - previous).abs();
                self.noise = Some(self.noise.map_or(change, |noise| noise.max(change)));
            }
            return false;
        }
        let noise = self.noise.unwrap_or(0.0);
        if self.rung.is_some_and(|before| share >= before - noise) {
            self.stopped = Some(share);
            return false;
        }
        self.rung = Some(share);
        true
    }

    /// Nothing was left to give: wait for more pressure than now.
    pub fn exhausted(&mut self) {
        self.stopped = self.rung.take();
    }

    /// The share giving stopped at, for the record; None while it gives.
    pub fn stopped_at(&self) -> Option<f64> {
        self.stopped
    }
}

/// What shared mode leaves the host's other programs beyond their use now: the most they used
/// recently. Raised to their use at every reading and every stall; each bring-back pass that
/// found no stall since the previous one halves the excess, so a spike that caused no pressure
/// (a compile, a browser burst) holds warming back for a few calls, not until a restart.
/// Measured, never a constant. Dedicated mode leaves nothing.
#[derive(Debug, Default)]
pub struct Reserve {
    peak: u64,
}

impl Reserve {
    /// The other programs use `others` bytes now: the bytes discretionary holdings must leave.
    pub fn observe(&mut self, others: u64) -> u64 {
        self.peak = self.peak.max(others);
        self.peak - others
    }

    /// A bring-back pass with no stall since the last one: the excess halves.
    pub fn quiet(&mut self, others: u64) {
        self.peak = others.max(self.peak - self.peak.saturating_sub(others) / 2);
    }
}

/// Bytes the host's other programs use: everything in use but what the machine holds.
pub fn others(host: &crate::host_memory::HostMemory, total: u64, ours: u64) -> Option<u64> {
    let available = u64::try_from(host.mem_available).ok()?;
    Some(total.saturating_sub(available).saturating_sub(ours))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Events every window with these stall shares; true where a rung went.
    fn run(shares: &[f64]) -> Vec<bool> {
        let start = Instant::now();
        let mut total = 0;
        let mut feedback = Feedback::new(0, start);
        shares
            .iter()
            .enumerate()
            .map(|(i, share)| {
                total += (share * WINDOW.as_micros() as f64) as u64;
                feedback.give(total, start + WINDOW * (i as u32 + 1))
            })
            .collect()
    }

    #[test]
    fn rungs_go_while_each_lowers_the_stall() {
        // A memory hog: each rung given back lowers the stall, until it is gone.
        assert_eq!(run(&[0.6, 0.4, 0.2, 0.05]), [true; 4]);
    }

    #[test]
    fn a_stall_giving_back_cannot_touch_sheds_one_rung_however_long_it_lasts() {
        // Refaults on a slow disk: the share wanders around 0.3 whatever is given.
        let refaults = [0.3, 0.31, 0.29, 0.3, 0.32, 0.3, 0.28, 0.31, 0.3, 0.29];
        let given = run(&refaults);
        assert_eq!(given.iter().filter(|g| **g).count(), 1, "{given:?}");
    }

    #[test]
    fn a_stall_rising_beyond_the_noise_after_a_stop_is_answered() {
        let given = run(&[0.3, 0.31, 0.29, 0.3, 0.6, 0.4]);
        assert_eq!(given, [true, false, false, false, true, true]);
    }

    #[test]
    fn a_quiet_window_after_a_rung_starts_a_new_episode() {
        let start = Instant::now();
        let mut feedback = Feedback::new(0, start);
        assert!(feedback.give(600_000, start + WINDOW));
        assert!(!feedback.give(1_200_000, start + WINDOW * 2), "no lower: stopped");
        // No event for three windows (the stall stopped), then a stall again.
        assert!(feedback.give(1_800_000, start + WINDOW * 5 + WINDOW / 2));
    }

    #[test]
    fn nothing_left_to_give_waits_for_more_pressure_than_the_noise() {
        let start = Instant::now();
        let mut feedback = Feedback::new(0, start);
        assert!(feedback.give(400_000, start + WINDOW));
        feedback.exhausted();
        assert!(!feedback.give(900_000, start + WINDOW * 2), "noise is measured first");
        assert!(!feedback.give(1_380_000, start + WINDOW * 3), "within the noise");
        assert!(feedback.give(2_580_000, start + WINDOW * 4), "well above it");
    }

    #[test]
    fn the_reserve_keeps_a_recent_peak_and_forgets_it_on_quiet_passes() {
        let mut reserve = Reserve::default();
        assert_eq!(reserve.observe(4 << 30), 0);
        // A 6 GiB burst came and went: 6 GiB stays theirs for now.
        assert_eq!(reserve.observe(10 << 30), 0);
        assert_eq!(reserve.observe(4 << 30), 6 << 30);
        reserve.quiet(4 << 30);
        assert_eq!(reserve.observe(4 << 30), 3 << 30);
        reserve.quiet(4 << 30);
        reserve.quiet(4 << 30);
        assert_eq!(reserve.observe(4 << 30), (3 << 30) / 4);
    }

    #[test]
    fn a_trigger_arms_on_the_systems_memory_pressure() {
        // Kernels without PSI (or a sandbox that hides it) have nothing to arm.
        if !std::path::Path::new("/proc/pressure/memory").exists() {
            return;
        }
        let pressure = Pressure::arm(HostMode::Shared).expect("an unprivileged 2 s trigger");
        assert!(pressure.stalled_us().is_ok());
    }
}
