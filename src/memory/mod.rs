//! The one owner of every GPU memory decision (PLAN decision 3, design rust-machine.md §2.1–2.2).
//! `policy` decides from samples and executor facts; `nvml` samples on its own thread; this
//! module joins them and keeps the floor while a call runs.
pub mod host;
pub mod learned;
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

pub(crate) fn note(event: serde_json::Value) {
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
    /// `root` keeps what executors measured across runs (`memory-learned.json`).
    pub fn start(device: &str, config: &MemoryConfig, root: &std::path::Path) -> Self {
        let mut gpu = Gpu::with_floor(config.floor_bytes);
        gpu.learned = learned::Learned::open(&root.join("memory-learned.json"));
        gpu.device = device.into();
        let ledger = Arc::new(Mutex::new(gpu));
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
                ledger.lock().unwrap().device = opened.key.clone();
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
                                "gpu":ledger.device,"free":sample.free,"floor":ledger.floor(sample)}));
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
                    (gpu.grant_want(plan), gpu.need(plan))
                };
                let decision = gpu.decide(plan, want, need, &sample, &mut round);
                note(
                    serde_json::json!({"event": if spawn {"admit"} else {"grant"}, "plan": plan,
                    "gpu": gpu.device, "free": sample.free, "external": gpu.external(&sample),
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

    /// Whether `plan` takes Degree 2 now (`Gpu::fits_resident`). False when NVML cannot say.
    pub fn fits_resident(&self, plan: &str, holdings: impl Fn() -> Vec<Holding>) -> bool {
        let Some(sample) = self.sample() else {
            return false;
        };
        let held = holdings();
        let (fits, want) = self.with(|gpu| {
            gpu.holdings = held;
            (gpu.fits_resident(plan, &sample), gpu.want(plan))
        });
        note(serde_json::json!({"event": "degree2", "plan": plan, "fits": fits, "want": want}));
        fits
    }

    /// `plan`'s executor asked custody for holding `id` or offered it: kept so its next
    /// executor's fit counts that weight set once when another tenant reads it.
    pub fn learn_holding(&self, plan: &str, id: &str) {
        self.with(|gpu| {
            if gpu.learned.holding(plan, policy::holding_name(id)) {
                if let Err(error) = gpu.learned.save() {
                    note(
                        serde_json::json!({"event": "learned_unsaved", "error": error.to_string()}),
                    );
                }
            }
        });
    }

    /// Whether `plan`'s executor and first working set fit in the room there is now, with no
    /// step on another tenant (a prewarm never makes room). True without NVML: nothing is
    /// decided here then.
    pub fn admits(&self, plan: &str, holdings: impl Fn() -> Vec<Holding>) -> bool {
        let Some(sample) = self.sample() else {
            return true;
        };
        let held = holdings();
        self.with(|gpu| {
            gpu.holdings = held;
            gpu.room(plan, &sample) >= gpu.spawn_need(plan)
        })
    }

    /// Until `free_bytes` are free beside the floor: holdings `plan`'s own executor let go
    /// are dropped, then idle tenants give room (weights first, then processes); then `plan`'s
    /// cap rises into what is there. None: no NVML.
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
                gpu.own_unread(plan, &mut round).or_else(|| {
                    match gpu.decide(plan, None, u64::MAX, &sample, &mut round) {
                        Decision::Step(
                            step @ (Step::Unmap(_) | Step::Revoke(_) | Step::End(_)),
                        ) => Some(step),
                        _ => None,
                    }
                })
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
                            .max(gpu.tenant(plan).map_or(0, |t| t.cap))
                            .max(gpu.physical_cap(plan, &sample));
                        gpu.set_cap(plan, cap);
                        cap
                    })));
                }
            }
        }
    }

    /// Host pinned budgets, `first` ahead and then most recently used first, from the live
    /// host (cgroup path and `MemAvailable`). None: the host is unreadable.
    pub fn pinned_budgets(&self, first: &str) -> Option<std::collections::BTreeMap<String, u64>> {
        let order = self.with(|gpu| gpu.pinned_order(first));
        let memory = crate::host_memory::read();
        host::pinned_split(memory.available, memory.shmem, &order)
    }

    /// The floor this GPU keeps now (the executor's own floor follows it).
    pub fn floor(&self) -> Option<u64> {
        let sample = self.sample()?;
        Some(self.with(|gpu| gpu.floor(&sample)))
    }

    /// One call's measurements, kept for later executors and runs: its shape's activation
    /// growth (the call's peak and each method's) and the context it measured.
    pub fn learn_call(
        &self,
        plan: &str,
        shape: &str,
        peak: u64,
        methods: &std::collections::BTreeMap<String, u64>,
        context: Option<u64>,
        mapped: Option<u64>,
    ) {
        self.with(|gpu| {
            gpu.learned.call(plan, shape, peak, methods);
            if let Some(mapped) = mapped {
                gpu.learned.mapped(plan, mapped);
            }
            if let Some(context) = context.filter(|c| *c > 0) {
                let device = gpu.device.clone();
                gpu.learned.context(&device, context);
            }
            if let Err(error) = gpu.learned.save() {
                note(serde_json::json!({"event": "learned_unsaved", "error": error.to_string()}));
            }
        });
    }

    /// Rooms a call saw squeeze its stage methods, kept for later executors of its shape.
    pub fn learn_squeezed(
        &self,
        plan: &str,
        shape: &str,
        rooms: &std::collections::BTreeMap<String, u64>,
    ) {
        if rooms.is_empty() {
            return;
        }
        self.with(|gpu| {
            gpu.learned.squeezed(plan, shape, rooms);
            let shape = serde_json::json!({"plan": plan, "shape": shape, "rooms": rooms});
            note(serde_json::json!({"event": "squeezed", "learned": shape}));
            if let Err(error) = gpu.learned.save() {
                note(serde_json::json!({"event": "learned_unsaved", "error": error.to_string()}));
            }
        });
    }

    /// A load's weights (as stages count them) and their floor, kept for the next load.
    pub fn learn_load(&self, plan: &str, weights: Option<u64>, floor: Option<u64>) {
        let Some(weights) = weights.filter(|w| *w > 0) else {
            return;
        };
        self.with(|gpu| {
            gpu.learned.load(plan, weights, floor.unwrap_or(0));
            if let Err(error) = gpu.learned.save() {
                note(serde_json::json!({"event": "learned_unsaved", "error": error.to_string()}));
            }
        });
    }

    /// A process's private host bytes (PSS less shared memory), kept across runs.
    pub fn learn_host(&self, plan: &str, bytes: u64) {
        self.with(|gpu| {
            gpu.learned.host(plan, bytes);
            if let Err(error) = gpu.learned.save() {
                note(serde_json::json!({"event": "learned_unsaved", "error": error.to_string()}));
            }
        });
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
