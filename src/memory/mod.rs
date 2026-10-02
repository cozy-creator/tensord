//! The one owner of every GPU memory decision (PLAN decision 3, design rust-machine.md §2.1–2.2).
//! `policy` decides from samples and executor facts; `nvml` samples on its own thread; this
//! module joins them and keeps the floor while a call runs.
pub mod nvml;
pub mod policy;

use policy::{Decision, Facts, Gpu, Holding, Round, Sample, Step};
use serde::Deserialize;
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    os::unix::fs::FileExt,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

/// Budget-cell `wanted` at or below this asks for a process cap of `CAP - wanted` bytes
/// (Runtime `budget_cell.CAP`, `process_cap/1`).
const CELL_CAP: i64 = -2;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Raises the free floor kept on the GPU (never lowers the default).
    pub floor_bytes: u64,
    /// Append every one-second reading here (JSON lines), for evidence.
    pub sample_log: Option<PathBuf>,
}

/// Ask a running executor for a smaller (or larger) process cap at its next block boundary.
/// Wanted first, then the counter it guards, as the executor reads them.
pub fn ask_cap(cell: &File, cap: u64) -> io::Result<i64> {
    let mut word = [0u8; 8];
    cell.read_exact_at(&mut word, 0)?;
    let asked = i64::from_le_bytes(word) + 1;
    cell.write_all_at(&(CELL_CAP - cap as i64).to_le_bytes(), 8)?;
    cell.write_all_at(&asked.to_le_bytes(), 0)?;
    Ok(asked)
}

struct Running {
    plan: String,
    cell: Option<File>,
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn note(event: serde_json::Value) {
    eprintln!("memory: {event}");
}

/// One GPU's ledger, sampler and floor watchdog. Without NVML nothing is decided here and
/// executors derive their own budgets (`grant` answers None).
pub struct GpuMemory {
    ledger: Arc<Mutex<Gpu>>,
    running: Arc<Mutex<Option<Running>>>,
    sampler: Option<nvml::Sampler>,
}

impl GpuMemory {
    pub fn start(device: &str, config: &MemoryConfig) -> Self {
        let ledger = Arc::new(Mutex::new(Gpu::with_floor(config.floor_bytes)));
        let running: Arc<Mutex<Option<Running>>> = Arc::new(Mutex::new(None));
        let log = config.sample_log.as_ref().and_then(|path| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|error| eprintln!("memory: sample log {path:?} unavailable: {error}"))
                .ok()
        });
        let sampler = match nvml::Device::open(device) {
            Ok(opened) => {
                let (watched, cells) = (ledger.clone(), running.clone());
                let log = Mutex::new(log);
                nvml::Sampler::start(opened, move |sample| {
                    let mut ledger = watched.lock().unwrap();
                    if let Some(file) = log.lock().unwrap().as_mut() {
                        let _ = writeln!(
                            file,
                            "{}",
                            serde_json::json!({"t_ms": now_ms() as u64, "total": sample.total,
                                "free": sample.free, "floor": ledger.floor(sample),
                                "display": sample.display, "external": ledger.external(sample)})
                        );
                    }
                    let Some((plan, cap)) = ledger.below_floor(sample) else {
                        return;
                    };
                    let running = cells.lock().unwrap();
                    let Some(Running {
                        plan: active,
                        cell: Some(cell),
                    }) = running.as_ref()
                    else {
                        return;
                    };
                    if *active != plan {
                        return;
                    }
                    match ask_cap(cell, cap) {
                        Ok(_) => {
                            ledger.set_cap(&plan, cap);
                            note(serde_json::json!({"event":"floor","plan":plan,"cap":cap,
                                "free":sample.free,"floor":ledger.floor(sample)}));
                        }
                        Err(error) => note(serde_json::json!({"event":"floor_cell_failed",
                            "plan":plan,"error":error.to_string()})),
                    }
                })
                .map_err(|error| eprintln!("memory: sampler did not start: {error}"))
                .ok()
            }
            Err(error) => {
                eprintln!("memory: NVML unavailable for {device:?}: {error}; executors derive their own budgets");
                None
            }
        };
        Self {
            ledger,
            running,
            sampler,
        }
    }

    pub fn sample(&self) -> Option<Sample> {
        self.sampler.as_ref().and_then(nvml::Sampler::now)
    }

    pub fn with<T>(&self, f: impl FnOnce(&mut Gpu) -> T) -> T {
        f(&mut self.ledger.lock().unwrap())
    }

    /// Decide for `plan` until it has its cap: `holdings` reads custody before each look,
    /// `carry_out` performs each step (and returns whether it did). Never refuses by size.
    /// None: no NVML, nothing decided here.
    pub fn decide(
        &self,
        plan: &str,
        spawn: bool,
        holdings: impl Fn() -> Vec<Holding>,
        mut carry_out: impl FnMut(&Step) -> io::Result<bool>,
    ) -> io::Result<Option<u64>> {
        let mut round = Round::default();
        loop {
            let Some(sample) = self.sample() else {
                return Ok(None);
            };
            let held = holdings();
            let decision = self.with(|gpu| {
                gpu.holdings = held;
                let (want, need) = if spawn {
                    let need = gpu.spawn_need(plan);
                    (Some(need), need)
                } else {
                    (gpu.want(plan), gpu.need(plan))
                };
                let decision = gpu.decide(plan, want, need, &sample, &mut round);
                note(
                    serde_json::json!({"event": if spawn {"admit"} else {"grant"}, "plan": plan,
                    "free": sample.free, "external": gpu.external(&sample),
                    "room": gpu.room(plan, &sample), "want": want, "need": need,
                    "decision": format!("{decision:?}")}),
                );
                decision
            });
            match decision {
                Decision::Go(cap) => return Ok(Some(cap)),
                // Calls on this GPU run one at a time, so a waiting decision has nothing to
                // wait for: it takes the room there is.
                Decision::Wait => {
                    return Ok(self.with(|gpu| Some(gpu.room(plan, &sample))));
                }
                Decision::Step(step) => {
                    if let Step::Shrink(other, cap) = &step {
                        // The cell is cloned out: the watchdog takes the ledger, then this lock.
                        let cell = self
                            .running
                            .lock()
                            .unwrap()
                            .as_ref()
                            .filter(|running| running.plan == *other)
                            .and_then(|running| running.cell.as_ref().map(File::try_clone))
                            .transpose()?;
                        if let Some(cell) = cell {
                            ask_cap(&cell, *cap)?;
                            self.with(|gpu| gpu.set_cap(other, *cap));
                        }
                        continue;
                    }
                    carry_out(&step)?;
                }
            }
        }
    }

    /// Idle tenants give room until `free_bytes` are free beside the floor (weights first, then
    /// processes); then `plan`'s cap rises into what is there. None: no NVML.
    pub fn make_room(
        &self,
        plan: &str,
        free_bytes: u64,
        holdings: impl Fn() -> Vec<Holding>,
        mut carry_out: impl FnMut(&Step) -> io::Result<bool>,
    ) -> io::Result<Option<u64>> {
        let mut round = Round::default();
        loop {
            let Some(sample) = self.sample() else {
                return Ok(None);
            };
            let held = holdings();
            let step = self.with(|gpu| {
                gpu.holdings = held;
                if sample.free >= free_bytes + gpu.floor(&sample) {
                    return None;
                }
                match gpu.decide(plan, None, u64::MAX, &sample, &mut round) {
                    Decision::Step(step @ (Step::Unmap(_) | Step::Revoke(_) | Step::End(_))) => {
                        Some(step)
                    }
                    _ => None,
                }
            });
            note(
                serde_json::json!({"event":"room","plan":plan,"free_bytes":free_bytes,
                "free":sample.free,"step":format!("{step:?}")}),
            );
            match step {
                Some(step) => {
                    carry_out(&step)?;
                }
                None => {
                    return Ok(Some(self.with(|gpu| {
                        let cap = gpu
                            .room(plan, &sample)
                            .max(gpu.tenant(plan).map_or(0, |t| t.cap));
                        gpu.set_cap(plan, cap);
                        cap
                    })));
                }
            }
        }
    }

    pub fn observe(&self, plan: &str, facts: Facts, mapped: Option<bool>) {
        self.with(|gpu| gpu.observe(plan, facts, mapped));
    }

    /// A call starts with `cap`; its cell (if any) is what the floor watchdog lowers.
    pub fn running(&self, plan: &str, cap: u64, cell: Option<File>) {
        self.with(|gpu| gpu.active(plan, cap));
        *self.running.lock().unwrap() = Some(Running {
            plan: plan.into(),
            cell,
        });
    }

    pub fn finished(&self, plan: &str) {
        *self.running.lock().unwrap() = None;
        self.with(|gpu| gpu.idle(plan));
    }
}
