//! TensorD coordinates device execution on one GPU or a group of K (rank 0 forms followers).
//! It supplies weight sources/CPU buffers and budgets; Runtime constructs models and performs
//! CUDA transfers/mappings through its TensorFS plane. Engine owns acceptance and run custody.
use crate::{
    catalog::HeldGeneration,
    device_executor::{
        self, channel_lost, Answer, Binding, Budgets, Cancellation, DeviceCommand,
        DeviceExecutor, ExecutorConfig, Forked, Frame, Group, Kind, ModelLoad, Services,
    },
    execution::{process_ended, Engine},
    host_tier::{HostGrant, HostTier, HostTierConfig, SealedRequest},
    journal::{
        AssetBinding, Execution, Failure, Outcome, OutputChecksum, Preparation, ProcessBirth, State,
    },
    launch_identity::Seal,
    memory::{
        policy::{Facts, Holding, Step},
        GpuMemory, MemoryConfig,
    },
    model_sources::{ModelSources, SelectedManifest},
    resident_custody::{HoldingFacts, HoldingKey, Offered, Reader, ResidentCustody},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read, Write},
    os::fd::{AsRawFd, OwnedFd},
    path::{Path, PathBuf},
    sync::{
        atomic::Ordering,
        Arc, Condvar, Mutex, Weak,
    },
    time::Instant,
};
use tensorfs_core::store::Store;

fn threads() -> u32 {
    crate::launch_identity::DEFAULT_THREADS
}
fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelGrant {
    pub package: String,
    pub slot: String,
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub manifest: String,
    pub components: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PublishedPackage {
    pub package: String,
    pub distribution: String,
    pub release: String,
    pub generation: String,
}

/// The machine's sealed host tier (`host_tier.rs`); always on, sized by live headroom.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct HostOptions {
    #[serde(default)]
    pub fill_threads: Option<usize>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    /// `shared` (the default): a guest that gives memory back whenever anything on the host
    /// stalls on it; `dedicated` (a rented pod's image or grant): only when this cgroup does.
    #[serde(default)]
    pub mode: crate::host_pressure::HostMode,
}

/// Root-sealed configuration, never peer-controlled paths or environment logic switches.
#[derive(Clone, Debug, Deserialize)]
pub struct GpuConfig {
    #[serde(default)]
    pub identity: Option<crate::launch_identity::LaunchIdentity>,
    /// The GPU envelope, `"0"` or `"0,1,..."`: a degree-K plan runs on the first K.
    pub devices: String,
    /// Load's device limit where NVML cannot read the device total.
    pub authorized_device_limit_bytes: Option<u64>,
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Explicitly configured locations; the executor seal is imposed on top of them.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// `OMP_NUM_THREADS` the seal imposes. torch's allocator is the Runtime's, set in code.
    #[serde(default = "threads")]
    pub threads: u32,
    /// Explicit authority for already verified cached catalog bytes, for runs without a
    /// Hub. Published runs resolve under the owner's Hub access instead; a cache hit alone
    /// grants no private model.
    #[serde(default)]
    pub models: Vec<ModelGrant>,
    #[serde(default)]
    pub packages: Vec<PublishedPackage>,
    #[serde(default)]
    pub host: HostOptions,
    /// Keep one import-only executor per installed GPU generation and fork executors from
    /// it (`fork/1`), so a new executor's imports are already done. It costs that process's
    /// host memory (no device memory); false spawns every executor, which imports again.
    #[serde(default = "yes")]
    pub prespawn: bool,
}
impl GpuConfig {
    pub fn load(path: &Path) -> io::Result<Self> {
        let config: Self = serde_json::from_reader(File::open(path)?).map_err(io::Error::other)?;
        let envelope = config.envelope();
        let unique: std::collections::BTreeSet<_> = envelope.iter().collect();
        if envelope.iter().any(String::is_empty) || unique.len() != envelope.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GPU scope names one device or a list of distinct devices",
            ));
        }
        if config
            .environment
            .keys()
            .any(|key| key.contains("TOKEN") || key.contains("SECRET") || key.contains("PASSWORD"))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "device executor environment cannot carry credentials",
            ));
        }
        Ok(config)
    }
    /// The configured devices in order (`CUDA_VISIBLE_DEVICES` entries).
    pub fn envelope(&self) -> Vec<String> {
        self.devices
            .split(',')
            .map(|d| d.trim().to_string())
            .collect()
    }
}

/// One declared model slot of a plan and its bound checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanSlot {
    pub binding: Binding,
    /// Native selected encoded source bytes; not a GPU-fit or full-memory charge.
    pub selected_encoded_bytes: u64,
}

/// One executor's identity: the package's generation, entrypoint, bound models and group. Not
/// who submitted: every submitter of the same construction is served by the one executor, a
/// request at a time. What an actor may run is its own journal rows (installation, preparation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuPlan {
    pub id: String,
    pub installation: String,
    pub generation: String,
    pub entrypoint: String,
    /// Every model slot the entrypoint declares, in declaration order.
    pub slots: Vec<PlanSlot>,
    /// GPUs of the group this plan runs on (the first `degree` envelope devices).
    pub degree: u32,
}
impl GpuPlan {
    pub fn selections(&self) -> Vec<SelectedManifest> {
        let mut rows: Vec<SelectedManifest> = vec![];
        for slot in &self.slots {
            match rows
                .iter_mut()
                .find(|r| r.manifest == slot.binding.snapshot)
            {
                Some(row) => {
                    for c in &slot.binding.components {
                        if !row.components.contains(c) {
                            row.components.push(c.clone());
                        }
                    }
                }
                None => rows.push(SelectedManifest {
                    manifest: slot.binding.snapshot.clone(),
                    components: slot.binding.components.clone(),
                }),
            }
        }
        rows
    }
}

/// The widest group every slot declares (`sequence_parallel.degrees`, one always) that fits
/// `available` GPUs, or `wanted` exactly; Runtime `machine_lanes.widths`.
pub fn group_degree(
    models: &[serde_json::Value],
    wanted: u32,
    available: usize,
) -> io::Result<u32> {
    let mut common: std::collections::BTreeSet<u32> = (1..=64).collect();
    for model in models {
        let mut declared = std::collections::BTreeSet::from([1u32]);
        if let Some(rows) = model
            .pointer("/sequence_parallel/degrees")
            .and_then(serde_json::Value::as_array)
        {
            declared.extend(rows.iter().filter_map(|d| d.as_u64()).map(|d| d as u32));
        }
        common = common.intersection(&declared).copied().collect();
    }
    if wanted > 0 {
        if !common.contains(&wanted) || wanted as usize > available {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "gpu_count_unavailable: {wanted} GPUs is not a group every model slot declares \
                     ({common:?}) within this machine's {available}"
                ),
            ));
        }
        return Ok(wanted);
    }
    Ok(common
        .into_iter()
        .filter(|d| *d as usize <= available)
        .max()
        .unwrap_or(1))
}

/// How far a function is prepared on this machine: a warm set member's `level`, and what
/// Status says a member or an installation `holds`. Each level includes the ones before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Installed,
    Downloaded,
    /// A process with its imports done: no device, no VRAM.
    Imported,
    /// An executor with the model constructed, its weights in the host tier.
    Host,
    /// Its weights on the GPU.
    Gpu,
}
impl Level {
    pub const ALL: [Level; 5] = [
        Level::Installed,
        Level::Downloaded,
        Level::Imported,
        Level::Host,
        Level::Gpu,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Level::Installed => "installed",
            Level::Downloaded => "downloaded",
            Level::Imported => "imported",
            Level::Host => "host",
            Level::Gpu => "gpu",
        }
    }
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|level| level.name() == name)
    }
}

/// A warm set member as the journal keeps it: enough to bring it back after a restart, and
/// to show it in Status as its caller sent it.
#[derive(Serialize, Deserialize)]
pub struct KeptMember {
    pub installation: String,
    pub level: String,
    /// Its function's bound preparation; empty when it binds no model on a GPU.
    pub preparation: String,
    /// The `WarmItem` the caller sent, base64.
    pub item: String,
}

/// Why a member stays at `installed`: nothing of it loads on a GPU.
const UNBOUND: &str = "its function binds no model on a GPU, or the store let its weights go";
/// What a member holds on a machine with no GPU: its code (a function that binds a model is
/// refused there).
pub fn without_gpus(asked: Level) -> (Level, &'static str) {
    (Level::Installed, if asked > Level::Installed { UNBOUND } else { "" })
}

/// One member of a caller's warm set, as the pool keeps it.
pub struct Member {
    pub held: HeldGeneration,
    /// The models its function binds. None: it binds none on a GPU, or the store let them go.
    pub plan: Option<GpuPlan>,
    /// The level the controller asked for.
    pub level: Level,
    /// Why it holds less than `level`, from the last pass over the set.
    held_back: Mutex<&'static str>,
}
impl Member {
    pub fn new(held: HeldGeneration, plan: Option<GpuPlan>, level: Level) -> Arc<Self> {
        Arc::new(Self {
            held,
            plan,
            level,
            held_back: Mutex::new(""),
        })
    }
}

struct Session {
    plan: String,
    generation: String,
    /// The App it serves: its environment's root, or a callee's App in it.
    application: String,
    /// GPUs of its group; its followers (rank 1 first) once it started, held by pidfd so a
    /// call that fails can name the GPU whose process ended first.
    degree: u32,
    followers: Vec<crate::process::Exact>,
    loaded: bool,
    executor: DeviceExecutor,
    /// Budget cells by rank: rank 0's own, and each follower's (`rank_cells/1`).
    budget_cells: BTreeMap<u32, File>,
    /// Its selected models' headers and assets, served on `model_source`.
    sources: Arc<ModelSources>,
    /// This executor in the host tier and the layouts it may adopt.
    peer: u64,
    grants: Vec<HostGrant>,
    /// Its weights stay on the GPU under custody (Degree 2), decided once at its load.
    sharing: bool,
    launch: Launch,
    /// It has served a call: its next one is not its first.
    invoked: bool,
    /// Regions its plane had evicted when its last call ended (a running total).
    evictions: u64,
    /// Why its weights come from the store, not the sealed host tier (an older SDK).
    unsealed: Option<String>,
}

/// How a session's executor came to exist, for the load record.
#[derive(Clone, Debug, Serialize)]
struct Launch {
    /// `fork` (from the generation's import-only executor) or `spawn`.
    mode: &'static str,
    /// From the launch decision to the executor's Hello.
    ms: f64,
    /// Its Start: wall time and the executor's own legs.
    start_ms: f64,
    start: Vec<(String, f64)>,
}

/// A generation's import-only executor that executors fork from (`fork/1`). Its children
/// die with it, so it ends only with the pool or, childless, to give its host memory back.
#[derive(Default)]
struct Zygote {
    state: Mutex<ZygoteState>,
    changed: Condvar,
    /// Executors forked from it that may still live.
    children: Mutex<Vec<ProcessBirth>>,
    /// Its last fork (or its start): childless parents end least recently used first.
    used: Mutex<Option<Instant>>,
}
#[derive(Default)]
enum ZygoteState {
    #[default]
    Starting,
    Ready(Box<DeviceExecutor>),
    /// Executors of this generation spawn: its Runtime does not fork, or the parent ended.
    Off(String),
    /// Its import-only start failed: this launch spawns, the next one starts a parent again
    /// (a slow or broken first import is not a reason to stop forking for the machine's life).
    Failed(String),
}
impl Zygote {
    fn set(&self, state: ZygoteState) {
        if let ZygoteState::Off(reason) | ZygoteState::Failed(reason) = &state {
            eprintln!("executor prespawn off: {reason}");
        }
        *self.state.lock().unwrap() = state;
        self.changed.notify_all();
    }
    /// Wait while the parent imports.
    fn wait_started(&self) {
        let mut state = self.state.lock().unwrap();
        while matches!(*state, ZygoteState::Starting) {
            state = self.changed.wait(state).unwrap();
        }
    }
    /// Fork an executor, waiting while the parent imports. A refusal or an ended parent
    /// gives the configuration back for a spawn.
    fn fork(
        &self,
        config: ExecutorConfig,
        on_birth: impl FnOnce(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Forked> {
        let mut state = self.state.lock().unwrap();
        while matches!(*state, ZygoteState::Starting) {
            state = self.changed.wait(state).unwrap();
        }
        let ZygoteState::Ready(parent) = &mut *state else {
            return Ok(Forked::Refused(
                Box::new(config),
                "no import-only executor".into(),
            ));
        };
        let forked = parent.fork(config, on_birth)?;
        match &forked {
            Forked::Ready(child) => {
                self.children.lock().unwrap().push(child.birth.clone());
                *self.used.lock().unwrap() = Some(Instant::now());
            }
            Forked::Lost(..) => {
                // Its children ended with it; the pool replaces it (`new_session`).
                *state = ZygoteState::Off(ENDED.into());
            }
            Forked::Refused(..) => {}
        }
        Ok(forked)
    }
    fn failed(&self) -> bool {
        matches!(*self.state.lock().unwrap(), ZygoteState::Failed(_))
    }
    fn ready(&self) -> bool {
        matches!(*self.state.lock().unwrap(), ZygoteState::Ready(_))
    }
    /// Ready and with no live child: ending it ends nothing else.
    fn childless(&self) -> bool {
        if !matches!(*self.state.lock().unwrap(), ZygoteState::Ready(_)) {
            return false;
        }
        let mut children = self.children.lock().unwrap();
        children.retain(|birth| !process_ended(birth).unwrap_or(true));
        children.is_empty()
    }
    /// Its private host bytes (PSS less shared memory), 0 when not running.
    fn private_bytes(&self) -> u64 {
        match &*self.state.lock().unwrap() {
            ZygoteState::Ready(parent) => crate::host_memory::process(parent.birth.pid)
                .map_or(0, |memory| memory.pss.saturating_sub(memory.pss_shmem)),
            _ => 0,
        }
    }
}

/// A parent that died (killed, out of memory): the pool starts a new one.
const ENDED: &str = "import-only executor ended";

/// What a parent costs before any is measured: an import-only SDXL or Anima executor is
/// 0.78 GiB RSS (J/RESULTS.md).
const UNMEASURED_PARENT: u64 = 1 << 30;

/// The learned-host key of a generation's import-only executor.
fn parent_key(held: &HeldGeneration) -> String {
    format!("parent:{}:{}", held.record.identity, held.record.application)
}

/// An App of an environment: `(generation, application)`. Executors are keyed by it.
type App = (String, String);

fn parent_app(held: &HeldGeneration) -> App {
    (held.record.identity.clone(), held.record.application.clone())
}

/// The manifests live executors read. A download's GC must not evict them: an evicted file a
/// session still reads frees no disk and breaks the session's next load.
#[derive(Default)]
struct Serving {
    next: std::sync::atomic::AtomicU64,
    held: Mutex<BTreeMap<u64, Vec<String>>>,
}
/// One executor's entry in `Serving`, removed when its process has exited.
struct ServingHold {
    serving: Arc<Serving>,
    id: u64,
}
impl Drop for ServingHold {
    fn drop(&mut self) {
        self.serving.held.lock().unwrap().remove(&self.id);
    }
}

pub struct GpuPool {
    root: PathBuf,
    serving: Arc<Serving>,
    /// Scopes JIT caches to this machine run; earlier runs' scopes are removed at start.
    incarnation: String,
    config: GpuConfig,
    store: Arc<Store>,
    reserved: Arc<crate::gpu_reservation::Slot>,
    /// One retained executor per plan; the memory policy decides which keep weights mapped.
    sessions: Mutex<BTreeMap<String, Session>>,
    /// Per generation, the import-only executor sessions fork from. Declared after
    /// `sessions`: executors end before the parent they were forked from.
    zygotes: Mutex<BTreeMap<(String, String), Arc<Zygote>>>,
    /// Each generation's kernel boot of this machine run: its process while it runs.
    kernel_boots: Mutex<BTreeMap<String, Option<crate::process::Exact>>>,
    /// Each caller's warm set, in the order it sent it.
    members: Mutex<BTreeMap<String, Vec<Arc<Member>>>>,
    /// One pass over the warm sets at a time.
    keeping: Mutex<()>,
    /// Each live executor's (generation, level) by plan: what Status reads while a call holds
    /// `sessions`.
    levels: Mutex<BTreeMap<String, (App, Level)>>,
    /// Each envelope GPU with its own memory decisions, in envelope order.
    devices: Vec<Device>,
    host: Arc<HostTier>,
    /// Shared mode's room for the host's other programs, and the stalls the watcher saw: the
    /// count at the last bring-back pass says whether one came since (`host_pressure`).
    reserve: Mutex<crate::host_pressure::Reserve>,
    stalls: std::sync::atomic::AtomicU64,
    stalls_seen: std::sync::atomic::AtomicU64,
    /// The memory policy's host ledger; the tier's limit reads it.
    host_ledger: Arc<crate::memory::host::HostLedger>,
    /// Degree 2: GPU weights kept across executors. None on a GPU that drives a display.
    custody: Option<Mutex<ResidentCustody>>,
    /// Memoized Model methods' results, shared by every run on this machine.
    memo: Arc<crate::memo::Memo>,
    // Drop session/resource custody before ending the actual spawning thread.
    launcher: crate::child_launcher::ChildLauncher,
}
struct Device {
    entry: String,
    memory: GpuMemory,
}

pub(crate) struct Permit {
    pool: Arc<GpuPool>,
    engine: Weak<Engine>,
    token: Option<crate::gpu_reservation::Permit>,
}
pub(crate) struct JobAllocation {
    permit: Permit,
    pub devices: String,
}
impl JobAllocation {
    pub fn prepare(&self) -> io::Result<()> {
        // A raw transform has no model plan to drive eviction. End only idle
        // serving sessions under their existing lock, then revoke held mappings.
        let pool = &self.permit.pool;
        let mut sessions = pool.sessions.lock().unwrap();
        let plans: Vec<_> = sessions.keys().cloned().collect();
        for plan in plans {
            pool.carry_out(&Step::End(plan), &mut sessions)?;
        }
        for holding in pool.holdings() {
            pool.carry_out(&Step::Revoke(holding.id), &mut sessions)?;
        }
        Ok(())
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        drop(self.token.take());
        if let Some(engine) = self.engine.upgrade() {
            engine.notify_activity();
        }
    }
}
pub(crate) struct WakeOnExit(pub(crate) Weak<Engine>);
impl Drop for WakeOnExit {
    fn drop(&mut self) {
        if let Some(engine) = self.0.upgrade() {
            engine.notify_activity();
        }
    }
}

/// Whether an accepted call waits in the queue for a GPU.
fn gpu_request_queued(engine: &Engine) -> bool {
    let mut cursor = 0;
    while let Ok(page) = engine.ready_after(cursor, 256) {
        let Some(last) = page.last() else {
            return false;
        };
        let queued = |record: &Execution| {
            let gpu = record.invocation.accelerator || record.submission.as_ref().is_some_and(|s| !s.preparation_id.is_empty());
            gpu && record.state == State::Queued && record.waiting_reason.is_none()
        };
        if page.iter().any(queued) {
            return true;
        }
        cursor = last.id.parse().unwrap_or(u64::MAX);
    }
    false
}

impl GpuPool {
    pub fn new(root: &Path, config: GpuConfig, store: Arc<Store>) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        if let Some(identity) = config.identity {
            identity.traverse(root)?;
            identity.traverse(
                root.parent()
                    .ok_or_else(|| io::Error::other("GPU root has no owned state parent"))?,
            )?;
        }
        let defaults = HostTierConfig::default();
        let host_ledger = Arc::new(crate::memory::host::HostLedger::default());
        let host = HostTier::new(
            store.clone(),
            HostTierConfig {
                fill_threads: config.host.fill_threads.unwrap_or(defaults.fill_threads),
                ttl: config
                    .host
                    .ttl_seconds
                    .map_or(defaults.ttl, std::time::Duration::from_secs),
                plans: Some(root.join("host-plans")),
            },
            Box::new(crate::memory::host::TierPolicy(host_ledger.clone())),
        )?;
        // Fill leases and GPU custody hold one descriptor per object or chunk.
        raise_fd_limit();
        let envelope = config.envelope();
        // Degree 2 is world-one, on the first device; off where it drives a display.
        let custody =
            (!display_active(&envelope[0])).then(|| Mutex::new(ResidentCustody::default()));
        let incarnation = uuid::Uuid::new_v4().simple().to_string();
        crate::launch_identity::remove_stale_jit(root, &incarnation);
        remove_old_executor_roots(root);
        let mut devices = vec![];
        for entry in &envelope {
            // Per GPU: its own sample log and learned file (two writers would tear them).
            let mut memory = config.memory.clone();
            let mut learned = root.to_path_buf();
            if envelope.len() > 1 {
                memory.sample_log = memory.sample_log.map(|path| {
                    let mut name = path.into_os_string();
                    name.push(format!(".gpu{entry}"));
                    name.into()
                });
                learned = root.join(format!("memory-gpu{entry}"));
                fs::create_dir_all(&learned)?;
            }
            devices.push(Device {
                entry: entry.clone(),
                memory: GpuMemory::start(entry, &memory, &learned),
            });
        }
        let pool = Arc::new(Self {
            launcher: crate::child_launcher::ChildLauncher::new()?,
            root: root.to_path_buf(),
            serving: Arc::default(),
            devices,
            incarnation,
            config,
            store,
            reserved: Arc::new(crate::gpu_reservation::Slot::default()),
            sessions: Mutex::new(BTreeMap::new()),
            zygotes: Mutex::new(BTreeMap::new()),
            kernel_boots: Mutex::new(BTreeMap::new()),
            members: Mutex::new(BTreeMap::new()),
            keeping: Mutex::new(()),
            levels: Mutex::new(BTreeMap::new()),
            host,
            reserve: Mutex::default(),
            stalls: Default::default(),
            stalls_seen: Default::default(),
            host_ledger,
            custody,
            memo: Arc::new(crate::memo::Memo::open(root.join("stage-memo"))?),
        });
        pool.watch_host();
        Ok(pool)
    }
    /// The host-pressure watcher (`host_pressure`): one thread blocked in poll() until the
    /// kernel reports a stall, then a rung given back when the feedback rule says so.
    fn watch_host(self: &Arc<Self>) {
        let mode = self.config.host.mode;
        let pressure = match crate::host_pressure::Pressure::arm(mode) {
            Ok(pressure) => pressure,
            Err(error) => {
                eprintln!("host pressure: no PSI trigger ({error}); memory goes back only as calls plan");
                return;
            }
        };
        let pool = Arc::downgrade(self);
        let started = std::thread::Builder::new()
            .name("host-pressure".into())
            .spawn(move || {
                let mut feedback = crate::host_pressure::Feedback::new(
                    pressure.sample().map(|sample| sample.total()).unwrap_or(0),
                    std::time::Instant::now(),
                );
                while pressure.wait().is_ok() {
                    let Some(pool) = pool.upgrade() else { break };
                    let Ok(sample) = pressure.sample() else { break };
                    let Some(give) = feedback.observe(sample, std::time::Instant::now()) else { continue };
                    pool.stalls.fetch_add(1, Ordering::AcqRel);
                    pool.host_room_now();
                    let gave = give && pool.give_back_one();
                    if give && !gave {
                        feedback.exhausted();
                    }
                    crate::memory::note(serde_json::json!({"event": "host_pressure",
                        "mode": format!("{mode:?}"), "stalled_us": sample.stalled_us(),
                        "limit_hits": sample.limit_hits(), "gave": gave,
                        "stopped_at": feedback.stopped_at()}));
                }
            });
        if let Err(error) = started {
            eprintln!("host pressure: {error}");
        }
    }
    /// One rung back to the host, in the agreed order: sealed layouts no tenant holds, an
    /// import-only parent with no live child, then an idle executor (outside every warm set
    /// first). Never anything a running call reads: with a call running, its sessions are
    /// left alone. Whether anything went back.
    /// Host bytes discretionary holdings may take now (a warm member, an import ahead): what
    /// the host has before its tightest limit, less, in shared mode, what its other programs
    /// used recently beyond their use now (`host_pressure::Reserve`). None: unreadable.
    fn host_room_now(&self) -> Option<u64> {
        let host = crate::host_memory::read();
        let available = u64::try_from(host.available).ok()?;
        if self.config.host.mode == crate::host_pressure::HostMode::Dedicated {
            return Some(available);
        }
        let ours = crate::host_memory::tree_private(std::process::id())
            + self.host.facts().charged_bytes;
        let total = u64::try_from(crate::host_memory::total()).ok()?;
        let others = crate::host_pressure::others(&host, total, ours)?;
        Some(available.saturating_sub(self.reserve.lock().unwrap().observe(others)))
    }
    /// A bring-back pass: with no stall since the last one, the reserve forgets half.
    fn quiet_pass(&self) {
        let stalls = self.stalls.load(Ordering::Acquire);
        if self.stalls_seen.swap(stalls, Ordering::AcqRel) != stalls {
            return;
        }
        let host = crate::host_memory::read();
        let ours = crate::host_memory::tree_private(std::process::id())
            + self.host.facts().charged_bytes;
        let others = u64::try_from(crate::host_memory::total())
            .ok()
            .and_then(|total| crate::host_pressure::others(&host, total, ours));
        if let Some(others) = others {
            self.reserve.lock().unwrap().quiet(others);
        }
    }
    pub fn give_back_one(&self) -> bool {
        if self.host.release(1) > 0 || self.end_idle_parent(false) > 0 {
            return true;
        }
        let Ok(mut sessions) = self.sessions.try_lock() else {
            return false;
        };
        match self.first().with(|gpu| gpu.lru_idle("", false)) {
            Some(victim) => self.carry_out(&Step::End(victim), &mut sessions).unwrap_or(false),
            None => false,
        }
    }
    pub fn memo(&self) -> &crate::memo::Memo {
        &self.memo
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
    /// The manifests this pool's live executors read (`Serving`).
    pub fn serving(&self) -> Vec<String> {
        let held = self.serving.held.lock().unwrap();
        let mut manifests: Vec<String> = held.values().flatten().cloned().collect();
        manifests.sort();
        manifests.dedup();
        manifests
    }
    /// The kernel store; live executors and kernel boots hold their namespace's lock.
    pub fn kernel_caches(&self) -> crate::reclaim::KernelCaches {
        crate::reclaim::KernelCaches {
            root: self.root.join("kernels"),
        }
    }
    /// The first GPU's ledger: every plan runs there (groups take the first K), so it also
    /// orders the host's pinned shares.
    fn first(&self) -> &GpuMemory {
        &self.devices[0].memory
    }
    /// GPUs this machine may group.
    pub fn width(&self) -> usize {
        self.devices.len()
    }
    /// The GPUs a plan of `degree` runs on: the first `degree` envelope devices.
    fn lane(&self, degree: u32) -> io::Result<&[Device]> {
        let wanted = degree.max(1) as usize;
        self.devices.get(..wanted).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "gpu_count_unavailable: a {wanted}-GPU group on a machine with {} GPU(s)",
                    self.devices.len()
                ),
            )
        })
    }
    fn lane_devices(&self, degree: u32) -> io::Result<String> {
        Ok(self
            .lane(degree)?
            .iter()
            .map(|d| d.entry.as_str())
            .collect::<Vec<_>>()
            .join(","))
    }
    /// `plan`'s executor is gone: no GPU charges it any more.
    fn ended(&self, plan: &str) {
        self.levels.lock().unwrap().remove(plan);
        for device in &self.devices {
            device.memory.with(|gpu| gpu.ended(plan));
        }
        self.host_ledger.private(plan, None);
    }
    /// Each GPU of `plan`'s lane observes its own rank's facts; returns the first GPU's.
    fn observe(
        &self,
        plan: &str,
        degree: u32,
        facts: Facts,
        ranks: &[Option<device_executor::PlaneFacts>],
        mapped: Option<bool>,
    ) -> Facts {
        let each = rank_facts(facts, ranks, degree);
        for (device, facts) in self.lane(degree).unwrap_or_default().iter().zip(&each) {
            device.memory.observe(plan, *facts, mapped);
        }
        // Its weights left the GPU or came back: a loaded executor holds `host` or `gpu`.
        if let (Some(mapped), Some((_, level))) = (mapped, self.levels.lock().unwrap().get_mut(plan))
        {
            *level = if mapped { Level::Gpu } else { Level::Host };
        }
        each[0]
    }

    /// Replace `actor`'s warm set; `keep` then brings its members up. A member dropped from
    /// it becomes an ordinary tenant: it stays as it is until its memory is needed.
    pub fn set_members(&self, actor: &str, members: Vec<Arc<Member>>) {
        let plans: std::collections::BTreeSet<String> = {
            let mut sets = self.members.lock().unwrap();
            match members.is_empty() {
                true => drop(sets.remove(actor)),
                false => drop(sets.insert(actor.into(), members)),
            }
            let plans = sets.values().flatten().filter_map(|member| member.plan.as_ref());
            plans.map(|plan| plan.id.clone()).collect()
        };
        for device in &self.devices {
            device.memory.with(|gpu| gpu.members = plans.clone());
        }
    }
    /// One pass over `actor`'s warm set: each member is brought as far toward its level as
    /// free room and the room of idle tenants outside every warm set allow. It never takes
    /// from another member or a running call, and keeps the reason when it stops short.
    pub fn keep(self: &Arc<Self>, engine: &Arc<Engine>, actor: &str) {
        let _pass = self.keeping.lock().unwrap();
        self.quiet_pass();
        for member in self.warm_set(actor) {
            let held_back = match (&member.plan, member.level) {
                (_, Level::Installed) => "",
                (None, _) => UNBOUND,
                (_, Level::Downloaded) => "",
                (_, Level::Imported) => self.import(&member.held),
                (Some(plan), Level::Host | Level::Gpu) => match self.import(&member.held) {
                    "" => self.load_member(engine, &member.held, plan, member.level),
                    held_back => held_back,
                },
            };
            *member.held_back.lock().unwrap() = held_back;
        }
    }
    /// Every warm set's pass, off the caller's thread: after a call ends, when room may
    /// have come back for a member a request pushed down.
    pub fn keep_all(self: &Arc<Self>, engine: &Arc<Engine>) {
        let actors: Vec<String> = self.members.lock().unwrap().keys().cloned().collect();
        if actors.is_empty() {
            return;
        }
        let (pool, engine) = (self.clone(), engine.clone());
        let started = std::thread::Builder::new()
            .name("warm-set".into())
            .spawn(move || actors.iter().for_each(|actor| pool.keep(&engine, actor)));
        if let Err(error) = started {
            eprintln!("warm set: {error}");
        }
    }
    /// A member's executor loaded when the call slot is free: empty once it is, else why not.
    fn load_member(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        held: &HeldGeneration,
        plan: &GpuPlan,
        level: Level,
    ) -> &'static str {
        let holds = self.levels.lock().unwrap().get(&plan.id).map(|(_, held)| *held);
        if holds.is_some_and(|holds| holds >= level) {
            return "";
        }
        let Some(_permit) = self.permit(engine, None, true) else {
            return "a request holds the GPU; it loads when that ends";
        };
        // A request queued for the GPU goes first; the member comes back after it.
        if gpu_request_queued(engine) {
            return "a request is queued; it loads after";
        }
        let asked = Instant::now();
        let mut sessions = self.sessions.lock().unwrap();
        let mut result = self.prewarm_locked(engine, held, plan.clone(), &mut sessions, true);
        if let (Ok("loaded" | "already loaded"), Level::Gpu) = (&result, level) {
            result = self.map_locked(plan, &mut sessions);
        }
        for device in &self.devices {
            device.memory.finished(&plan.id);
        }
        if !sessions.contains_key(&plan.id) {
            self.ended(&plan.id);
        }
        self.note_prewarm(&plan.id, Default::default(), asked.elapsed(), &result);
        match result {
            Ok("loaded" | "already loaded" | "mapped") => "",
            Ok(held_back) => held_back,
            Err(error) => {
                eprintln!("warm set: loading {}: {error}", plan.id);
                "its load failed (the machine's log says why)"
            }
        }
    }
    /// A loaded member's weights mapped onto its GPU (`gpu`), inside the room a member may
    /// take: "mapped", or why they stay in the host tier.
    fn map_locked(
        &self,
        plan: &GpuPlan,
        sessions: &mut BTreeMap<String, Session>,
    ) -> io::Result<&'static str> {
        let level = self.levels.lock().unwrap().get(&plan.id).map(|(_, level)| *level);
        if level == Some(Level::Gpu) {
            return Ok("mapped");
        }
        if plan.degree != 1 {
            return Ok("a group of GPUs maps its weights at its first call");
        }
        let offers = |s: &Session| s.executor.hello.offers("weights.map/1");
        if !sessions.get(&plan.id).is_some_and(offers) {
            return Ok("its Runtime maps no weights ahead of a request");
        }
        let device = &self.lane(1)?[0];
        let step = |step: &Step| self.carry_out(step, sessions);
        let Some(cap) = device.memory.admit_member(&plan.id, true, || self.holdings(), step)? else {
            return Ok("no GPU room for its weights beside the warm set and the running call");
        };
        let Some(session) = sessions.get_mut(&plan.id) else {
            return Ok("its executor ended");
        };
        let reply = session.executor.command(
            &DeviceCommand::Map {
                construction: plan.id.clone(),
                cap_bytes: cap,
                floor_bytes: self.first().floor(),
            },
            &mut device_executor::Baseline,
        )?;
        if !reply.ok {
            eprintln!("warm set: mapping {}: {} {}", plan.id, reply.code, reply.detail);
            return Ok("its weights could not be mapped (the machine's log says why)");
        }
        // Every region on the card, placed or attached from custody (`complete`), is `gpu`.
        let complete = reply.mapped_bytes.unwrap_or(0) > 0 && reply.complete.unwrap_or(true);
        let facts = plane_facts(reply.plane.as_ref());
        self.observe(&plan.id, 1, facts, &reply.rank_planes, Some(complete));
        match complete {
            true => Ok("mapped"),
            false => Ok("part of its weights stay in host memory: the GPU has no room for them all"),
        }
    }
    /// What each of `actor`'s members holds now, and why that is lower than it asked.
    pub fn members(&self, actor: &str) -> Vec<(Level, &'static str)> {
        let holds = |member: &Arc<Member>| {
            let (holds, held_back) = (self.holding(member), *member.held_back.lock().unwrap());
            match (holds < member.level, holds, held_back) {
                (false, ..) => (holds, ""),
                (true, _, "") => (holds, "a request took its room; it returns when room does"),
                (true, ..) => (holds, held_back),
            }
        };
        self.warm_set(actor).iter().map(holds).collect()
    }
    fn warm_set(&self, actor: &str) -> Vec<Arc<Member>> {
        let sets = self.members.lock().unwrap();
        sets.get(actor).cloned().unwrap_or_default()
    }
    fn holding(&self, member: &Member) -> Level {
        let executor = member.plan.as_ref().and_then(|plan| {
            let levels = self.levels.lock().unwrap();
            levels.get(&plan.id).map(|(_, level)| *level)
        });
        let imported = || {
            let zygotes = self.zygotes.lock().unwrap();
            zygotes.get(&parent_app(&member.held)).cloned()
        };
        match (executor, &member.plan) {
            (Some(level), _) => level,
            _ if imported().is_some_and(|parent| parent.ready()) => Level::Imported,
            (None, Some(plan)) if self.holds(plan) => Level::Downloaded,
            _ => Level::Installed,
        }
    }
    /// Note what a loaded executor that is kept holds now, for Status.
    fn note_level(&self, session: &Session) {
        let mapped = |gpu: &mut crate::memory::policy::Gpu| {
            gpu.tenant(&session.plan).is_some_and(|tenant| tenant.mapped)
        };
        let level = match self.first().with(mapped) {
            true => Level::Gpu,
            false => Level::Host,
        };
        let noted = ((session.generation.clone(), session.application.clone()), level);
        self.levels.lock().unwrap().insert(session.plan.clone(), noted);
    }
    /// Whether the store holds every model `plan` binds.
    pub fn holds(&self, plan: &GpuPlan) -> bool {
        plan.selections().iter().all(|selected| {
            let hex = selected.manifest.trim_start_matches("sha256:");
            self.store.manifest_path(hex).exists()
        })
    }
    /// The highest level each App of each generation holds now (keyed `(generation,
    /// application)`): an import-only parent, or an executor. A callee's executors in a
    /// caller's environment are its own, never the caller's.
    pub fn levels(&self) -> BTreeMap<App, Level> {
        let parents: Vec<_> = self.zygotes.lock().unwrap().clone().into_iter().collect();
        let mut highest: BTreeMap<App, Level> = parents
            .into_iter()
            .filter(|(_, parent)| parent.ready())
            .map(|(app, _)| (app, Level::Imported))
            .collect();
        for (app, level) in self.levels.lock().unwrap().values() {
            let held = highest.entry(app.clone()).or_insert(*level);
            *held = (*held).max(*level);
        }
        highest
    }
    /// `held`'s import-only parent, started here when the host has room for it: empty once
    /// it is up, else why it is not.
    fn import(&self, held: &HeldGeneration) -> &'static str {
        let known = self.zygotes.lock().unwrap().contains_key(&parent_app(held));
        if !known && !self.parent_fits(held) {
            return "no host memory free for its imports";
        }
        let Some((parent, start)) = self.zygote(held) else {
            return "this machine imports nothing ahead for it (no fork, or it binds no model)";
        };
        if start {
            parent.set(self.import_only(held));
        }
        parent.wait_started();
        match (parent.ready(), parent.failed()) {
            (true, _) => "",
            (_, false) => "its Runtime forks no executors: nothing imports ahead of a request",
            (_, true) => {
                // The next pass imports again; a request that needs it starts one itself.
                self.forget_parent(held, &parent);
                "its imports failed (the machine's log says why)"
            }
        }
    }
    /// Each GPU of the group decides for itself, in device order (so two groups never wait
    /// on each other): one cap per GPU, rank 0's first. None: that GPU could not say.
    fn decide(
        &self,
        plan: &str,
        degree: u32,
        spawn: bool,
        sessions: &mut BTreeMap<String, Session>,
    ) -> io::Result<Vec<Option<u64>>> {
        let mut caps = vec![];
        for (index, device) in self.lane(degree)?.iter().enumerate() {
            caps.push(device.memory.decide(
                plan,
                spawn,
                || if index == 0 { self.holdings() } else { vec![] },
                |step| self.carry_out(step, sessions),
            )?);
        }
        Ok(caps)
    }
    pub fn host_tier(&self) -> &Arc<HostTier> {
        &self.host
    }
    /// GPU weights kept across executors (Degree 2), for the memory policy. Lets go of
    /// revoked generations whose last reader ended first.
    pub fn resident(&self) -> Vec<HoldingFacts> {
        self.custody.as_ref().map_or_else(Vec::new, |c| {
            let mut custody = c.lock().unwrap();
            log_released(custody.collect());
            custody.holdings()
        })
    }
    /// Revoke one held generation: no new attachments now; the bytes stay charged until each
    /// reader released it at an idle boundary (now, when the pool is idle) or ended.
    pub fn revoke(&self, key: &HoldingKey, generation: u64) -> io::Result<()> {
        let custody = self
            .custody
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, "no GPU weight custody"))?;
        custody.lock().unwrap().begin_revoke(key, generation)?;
        if let Ok(mut sessions) = self.sessions.try_lock() {
            for session in sessions.values_mut() {
                if let Err(error) = release_revoked(custody, &mut session.executor) {
                    eprintln!("resident revoke left a lease charged: {error}");
                }
            }
        }
        log_released(custody.lock().unwrap().collect());
        Ok(())
    }
    pub fn source_facts(&self, selections: &[SelectedManifest]) -> io::Result<ModelSources> {
        ModelSources::open_shared(self.store.clone(), selections)
    }
    pub fn published_installation(
        &self,
        service: &crate::service::Service,
        actor: &str,
        package: &str,
        release: &str,
    ) -> io::Result<Option<crate::journal::Installation>> {
        let Some(selected) = self
            .config
            .packages
            .iter()
            .find(|p| p.package == package && p.release == release)
        else {
            return Ok(None);
        };
        let held = service.catalog.resolve(&selected.generation)?;
        if held.record.package != selected.distribution || held.record.version != release {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "published catalog mapping differs from actual held package",
            ));
        }
        let alias = format!(
            "published-{}",
            tensorfs_core::sha256::hex_digest(
                format!("{package}\0{release}\0{}", selected.generation).as_bytes()
            )
        );
        service
            .engine
            .bind_installation(crate::journal::Installation {
                actor: actor.into(),
                alias,
                generation: selected.generation.clone(),
                package: package.into(),
                release: release.into(),
                interface: serde_json::to_vec(&held.record.interface)?,
            })
            .map(Some)
    }

    /// Derive each model slot's binding from the installed SDK's real static interface and
    /// the models the Hub resolved under the owner's access (`resolved`, one per slot), or
    /// configured cached byte authority. `degree` 0 takes the widest group every slot
    /// declares on this machine. No model code executes during description/preparation.
    pub fn prepare_root(
        &self,
        installed: &crate::journal::Installation,
        entrypoint: &str,
        choices: &[crate::api::domain::ModelChoice],
        resolved: &[ModelGrant],
        degree: u32,
    ) -> io::Result<GpuPlan> {
        let invalid = |detail: &str| io::Error::new(io::ErrorKind::InvalidData, detail.to_string());
        let interface: serde_json::Value =
            serde_json::from_slice(&installed.interface).map_err(io::Error::other)?;
        let entry = interface
            .get("entrypoints")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| {
                rows.iter().find(|row| {
                    row.get("name").and_then(serde_json::Value::as_str) == Some(entrypoint)
                })
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "declared GPU entrypoint absent",
                )
            })?;
        let models = entry
            .get("models")
            .and_then(serde_json::Value::as_array)
            .filter(|models| !models.is_empty())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "model declarations absent")
            })?;
        let prefix = format!("{entrypoint}.models.");
        let slot_of = |model: &serde_json::Value| -> io::Result<(String, String)> {
            let path = model
                .get("path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("declared model path absent"))?;
            let parameter = path
                .strip_prefix(&prefix)
                .filter(|v| !v.is_empty() && !v.contains('.'))
                .ok_or_else(|| invalid("declared model path is not a root parameter"))?;
            Ok((path.to_string(), parameter.to_string()))
        };
        let slots = models.iter().map(slot_of).collect::<io::Result<Vec<_>>>()?;
        for choice in choices {
            if !slots.iter().any(|(path, parameter)| {
                choice.parameter == *path || choice.parameter == *parameter
            }) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "model choice does not name a declared root slot",
                ));
            }
            // Hub-resolved grants already carry a slot's adapters (as its adapter view);
            // configured cached authority has no resolution step to apply them.
            if (!choice.source.is_empty() || !choice.profiles.is_empty() || !choice.adapters.is_empty())
                && resolved.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "provider sources and adapters need the run's preparation (configured grants take neither)",
                ));
            }
        }
        let degree = group_degree(models, degree, self.devices.len())?;
        let mut planned = vec![];
        for (model, (path, parameter)) in models.iter().zip(&slots) {
            let class = model
                .get("class")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("declared model class absent"))?;
            let chosen = choices
                .iter()
                .find(|c| c.parameter == *path || c.parameter == *parameter);
            let authority: Vec<&ModelGrant> = if resolved.is_empty() {
                self.config.models.iter().collect()
            } else {
                resolved.iter().collect()
            };
            let grants: Vec<_> = authority
                .into_iter()
                .filter(|grant| {
                    grant.package == installed.package
                        && grant.slot == *path
                        // A resolved grant was resolved from the choice itself.
                        && (!resolved.is_empty() || chosen.is_none_or(|choice| {
                            (choice.repository.is_empty() || choice.repository == grant.repository)
                                && (choice.release.is_empty() || choice.release == grant.release)
                                && (choice.lane.is_empty() || choice.lane == grant.lane)
                                && choice.manifest.as_ref().is_none_or(|reference| {
                                    tensorfs_core::sha256::hex(&reference.digest)
                                        == grant.manifest.trim_start_matches("sha256:")
                                })
                        }))
                })
                .collect();
            if grants.len() != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("cached model selection for {path} has no unique configured authority"),
                ));
            }
            let grant = grants[0];
            let mut wanted = std::collections::BTreeSet::new();
            if let Some(use_map) = model
                .get("component_use")
                .and_then(serde_json::Value::as_object)
            {
                for components in use_map.values() {
                    for component in components
                        .as_array()
                        .ok_or_else(|| invalid("declared component use is not an array"))?
                    {
                        wanted.insert(
                            component
                                .as_str()
                                .ok_or_else(|| invalid("declared component is not a name"))?
                                .to_string(),
                        );
                    }
                }
            }
            let components: Vec<_> = grant
                .components
                .iter()
                .filter(|c| wanted.is_empty() || wanted.contains(*c))
                .cloned()
                .collect();
            if components.is_empty() || wanted.iter().any(|c| !components.contains(c)) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("declared components of {path} exceed authorized cached model scope"),
                ));
            }
            let sources = self.source_facts(&[SelectedManifest {
                manifest: grant.manifest.clone(),
                components: components.clone(),
            }])?;
            let (_, selected_encoded_bytes, length) = sources.selected_facts(&grant.manifest)?;
            if chosen
                .and_then(|c| c.manifest.as_ref())
                .is_some_and(|r| r.length != 0 && r.length != length)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "selected manifest length differs from cached bytes",
                ));
            }
            let binding = Binding {
                application: interface
                    .get("application")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| invalid("declared application absent"))?
                    .into(),
                model_class: class.into(),
                model_binding_path: path.clone(),
                model_parameter_name: parameter.clone(),
                model_parameter_names: vec![parameter.clone()],
                component: components[0].clone(),
                components: components.clone(),
                snapshots: components
                    .into_iter()
                    .map(|c| (c, grant.manifest.clone()))
                    .collect(),
                snapshot: grant.manifest.clone(),
                release: format!("{}@{}", installed.package, installed.release),
                package: installed.package.clone(),
                model: format!("{}@{}", grant.repository, grant.release),
                ..Default::default()
            };
            planned.push(PlanSlot {
                binding,
                selected_encoded_bytes,
            });
        }
        let semantic = serde_json::json!({"generation":installed.generation,"entrypoint":entrypoint,"slots":planned,"degree":degree});
        let canonical = serde_json_canonicalizer::to_vec(&semantic).map_err(io::Error::other)?;
        Ok(GpuPlan {
            id: format!("gpu-{}", tensorfs_core::sha256::hex_digest(&canonical)),
            installation: installed.alias.clone(),
            generation: installed.generation.clone(),
            entrypoint: entrypoint.into(),
            slots: planned,
            degree,
        })
    }
    pub fn plan(&self, preparation: &Preparation) -> io::Result<GpuPlan> {
        let plan: GpuPlan =
            serde_json::from_slice(&preparation.document).map_err(io::Error::other)?;
        if plan.id != preparation.id || plan.installation != preparation.installation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GPU preparation differs from owned journal binding",
            ));
        }
        Ok(plan)
    }
    fn permit(self: &Arc<Self>, engine: &Arc<Engine>, family: Option<&str>, call: bool) -> Option<Permit> {
        let token = self.reserved.take(family, call)?;
        Some(Permit { pool: self.clone(), engine: Arc::downgrade(engine), token: Some(token) })
    }
    pub fn joins_family(&self, engine: &Engine, record: &Execution) -> io::Result<bool> {
        Ok(crate::gpu_reservation::family(record, |id| engine.get(id))?
            .is_some_and(|family| self.reserved.family(&family)))
    }
    pub(crate) fn job_allocation(self: &Arc<Self>, engine: &Arc<Engine>, record: &Execution) -> io::Result<Option<JobAllocation>> {
        let family = crate::gpu_reservation::family(record, |id| engine.get(id))?;
        let devices = self.lane_devices(1)?;
        Ok(self.permit(engine, family.as_deref(), false).map(|permit| JobAllocation { permit, devices }))
    }
    pub fn dispatch(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        record: &Execution,
        held: HeldGeneration,
        plan: GpuPlan,
    ) -> io::Result<bool> {
        let family = crate::gpu_reservation::family(record, |id| engine.get(id))?;
        let Some(permit) = self.permit(engine, family.as_deref(), true) else {
            return Ok(false);
        };
        engine.dispatch_managed(&record.id, move |engine, id| {
            let pool = permit.pool.clone();
            let result = pool.run(&engine, &id, held, plan);
            if let Err(error) = &result {
                settle(&engine, &id, error)?;
            }
            // The call slot is free: a warm set member this call pushed down may come back.
            drop(permit);
            pool.keep_all(&engine);
            result
        })
    }
    /// One line per Load in `loads.jsonl`: the executor's load facts, the host tier's, and
    /// the machine's and executor's RSS/PSS.
    fn record_load(
        &self,
        plan: &str,
        took: std::time::Duration,
        loaded: &Frame,
        executor: u32,
        launch: &Launch,
    ) -> io::Result<()> {
        use crate::host_memory::{process, ProcessMemory};
        #[derive(Serialize)]
        struct Line<'a> {
            plan: &'a str,
            load_ms: f64,
            facts: &'a Option<device_executor::LoadFacts>,
            host_tier: crate::host_tier::HostTierFacts,
            machine: ProcessMemory,
            executor: ProcessMemory,
            launch: &'a Launch,
        }
        let line = Line {
            plan,
            load_ms: took.as_secs_f64() * 1e3,
            facts: &loaded.facts,
            host_tier: self.host.facts(),
            machine: process(std::process::id()).unwrap_or_default(),
            executor: process(executor).unwrap_or_default(),
            launch,
        };
        let mut bytes = serde_json::to_vec(&line)?;
        bytes.push(b'\n');
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("loads.jsonl"))?
            .write_all(&bytes)
    }
    /// One line per call in `invokes.jsonl`: wall time, whether it was the executor's first,
    /// and the executor's weight-movement facts and metrics.
    fn record_invoke(&self, plan: &str, first: bool, took: std::time::Duration, reply: &Frame) {
        let line = serde_json::json!({
            "plan": plan,
            "first": first,
            "invoke_ms": took.as_secs_f64() * 1e3,
            "plane": reply.plane,
            "metrics": reply.metrics,
        });
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("invokes.jsonl"))
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(error) = written {
            eprintln!("invokes.jsonl: {error}");
        }
    }

    /// A running parent's hint that it will call `plan` next (`model_prefetch`, H3 long-form's
    /// `prefetch(motion_segment)`): load it in the background where it fits beside the tenants
    /// already there (it never makes room), without waiting for the parent's own run to end.
    /// It takes the GPU's call slot like any call, so a child call in flight finishes first.
    pub fn prefetch(self: &Arc<Self>, engine: &Arc<Engine>, held: HeldGeneration, plan: GpuPlan, family: Option<String>) {
        let (pool, engine) = (self.clone(), engine.clone());
        let started = std::thread::Builder::new()
            .name("executor-prefetch".into())
            .spawn(move || {
                let asked = Instant::now();
                if let Some((zygote, start)) = pool.zygote(&held) {
                    if start {
                        zygote.set(pool.import_only(&held));
                    }
                    zygote.wait_started();
                }
                let _permit = loop {
                    if let Some(permit) = pool.permit(&engine, family.as_deref(), true) { break permit; }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                };
                let key = plan.id.clone();
                let waited = asked.elapsed();
                let mut sessions = pool.sessions.lock().unwrap();
                let result = pool.prewarm_locked(&engine, &held, plan, &mut sessions, false);
                for device in &pool.devices {
                    device.memory.finished(&key);
                }
                if !sessions.contains_key(&key) {
                    pool.ended(&key);
                }
                pool.note_prewarm(&key, waited, asked.elapsed() - waited, &result);
            });
        if let Err(error) = started {
            eprintln!("executor prefetch: {error}");
        }
    }

    /// One line per prewarm in `prewarm.jsonl`: what it did and how long it waited first.
    fn note_prewarm(
        &self,
        plan: &str,
        waited: std::time::Duration,
        took: std::time::Duration,
        result: &io::Result<&'static str>,
    ) {
        let line = serde_json::json!({
            "plan": plan,
            "outcome": result.as_ref().map_or_else(|e| format!("failed: {e}"), |o| o.to_string()),
            "waited_ms": waited.as_secs_f64() * 1e3,
            "took_ms": took.as_secs_f64() * 1e3,
        });
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("prewarm.jsonl"))
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(error) = written {
            eprintln!("prewarm.jsonl: {error}");
        }
    }

    fn prewarm_locked(
        &self,
        engine: &Arc<Engine>,
        held: &HeldGeneration,
        plan: GpuPlan,
        sessions: &mut BTreeMap<String, Session>,
        member: bool,
    ) -> io::Result<&'static str> {
        if sessions.contains_key(&plan.id) {
            return Ok("already loaded");
        }
        let lane = self.lane(plan.degree)?;
        let mut load_caps = vec![];
        for (index, device) in lane.iter().enumerate() {
            let holdings = || if index == 0 { self.holdings() } else { vec![] };
            // A warm set member may take the room of idle tenants outside every set; a
            // prefetch only what is free. Neither touches a member or a running call.
            let cap = match member {
                true => {
                    let step = |step: &Step| self.carry_out(step, sessions);
                    device.memory.admit_member(&plan.id, false, holdings, step)?
                }
                false => device.memory.admits(&plan.id, holdings).then_some(None),
            };
            match cap {
                Some(cap) => load_caps.push(cap),
                None => return Ok("no GPU room beside the warm set and the running call"),
            }
        }
        if !self.host_room(&plan.id, sessions, true) {
            return Ok("host_memory: no host room beside the warm set and what this host's other programs used recently");
        }
        if !member {
            load_caps = self.decide(&plan.id, plan.degree, true, sessions)?;
        }
        for device in lane {
            device
                .memory
                .with(|gpu| gpu.starting(&plan.id, gpu.spawn_need(&plan.id)));
        }
        let mut session = self.new_session(engine, held, &plan, |birth, _| {
            self.first().with(|gpu| gpu.spawned(&plan.id, birth.pid));
            Ok(())
        })?;
        match self.call(
            engine,
            "",
            held,
            plan,
            &load_caps,
            &mut session,
            sessions,
            true,
        ) {
            Ok(_) => {
                self.note_level(&session);
                sessions.insert(session.plan.clone(), session);
                Ok("loaded")
            }
            // A prewarm is no run: there is no triage to keep.
            Err(error) => Err(ended_with(engine, "", error, session.executor)),
        }
    }

    pub fn stop(&self) -> io::Result<()> {
        let sessions = std::mem::take(&mut *self.sessions.lock().unwrap());
        for (plan, session) in sessions {
            session.executor.shutdown()?;
            self.ended(&plan);
        }
        for zygote in std::mem::take(&mut *self.zygotes.lock().unwrap()).into_values() {
            if let ZygoteState::Ready(parent) = std::mem::take(&mut *zygote.state.lock().unwrap()) {
                parent.shutdown()?;
            }
        }
        let boots = std::mem::take(&mut *self.kernel_boots.lock().unwrap());
        for boot in boots.into_values().flatten() {
            boot.kill()?;
        }
        Ok(())
    }

    /// Compile a generation's kernels for every card of the envelope in the background
    /// (machine start, after an install, so beside the model download): the Runtime's
    /// `machine_kernels` under the generation's own seal, so each key is its executors' and
    /// a kernel already in the store is not built again. Once per generation and machine
    /// run. It opens no device; its builds are niced and as wide as the host memory free
    /// when each starts. An executor that needs a kernel still compiling waits for it.
    pub fn kernel_boot(self: &Arc<Self>, held: HeldGeneration) {
        let generation = held.record.identity.clone();
        if !binds_models(&held.record.interface) {
            return;
        }
        {
            let mut boots = self.kernel_boots.lock().unwrap();
            if boots.contains_key(&generation) {
                return;
            }
            boots.insert(generation.clone(), None);
        }
        let pool = Arc::downgrade(self);
        let started = std::thread::Builder::new()
            .name("kernel-boot".into())
            .spawn(move || {
                let began = Instant::now();
                let Some(launched) = pool.upgrade().map(|pool| pool.launch_kernel_boot(&held))
                else {
                    return;
                };
                // Ended by its exit; killed only on a measured wedge of its process group.
                let ended = launched.and_then(|(mut child, boot, log)| {
                    let liveness = crate::process::Liveness::default();
                    let reaped = crate::process::reap(&boot, Some(&mut child), liveness)?;
                    Ok((reaped, log))
                });
                if let Some(pool) = pool.upgrade() {
                    if let Some(boot) = pool.kernel_boots.lock().unwrap().get_mut(&generation) {
                        *boot = None;
                    }
                    pool.note_kernel_boot(&generation, began.elapsed(), ended);
                }
            });
        if let Err(error) = started {
            eprintln!("kernel boot: {error}");
        }
    }

    fn launch_kernel_boot(
        &self,
        held: &HeldGeneration,
    ) -> io::Result<(std::process::Child, crate::process::Exact, PathBuf)> {
        let generation = &held.record.identity;
        let devices: Vec<_> = self.devices.iter().map(|d| d.entry.as_str()).collect();
        let seal = Seal::prepare(
            &self.root,
            self.config.identity,
            &self.incarnation,
            generation,
            &devices.join(","),
        )?;
        let log = self.root.join(format!("kernel-boot.{generation}.log"));
        let output = File::create(&log)?;
        let scope = Arc::new(crate::scope::Scope::create(&crate::scope::namespace(
            self.root.parent().unwrap_or(&self.root),
        ))?);
        let launch = || {
            let mut command = crate::launch_identity::trampoline(
                &held.record.python,
                self.config.identity,
                Some(&scope),
            )?;
            command
                .args(["-I", "-m", "cozy_runtime.internal.machine_kernels"])
                .env_clear()
                .envs(seal.environment(&self.config.environment))
                .envs(scope.environment())
                .stdin(std::process::Stdio::null())
                .stdout(output.try_clone()?)
                .stderr(output.try_clone()?);
            crate::launch_identity::inherit(&mut command, &[seal.kernel_hold.as_ref()]);
            let child = self.launcher.spawn(command)?;
            let birth = crate::execution::process_birth(child.id())?;
            let boot = crate::process::Exact::open(&birth)?
                .ok_or_else(|| io::Error::other("launched kernel boot has no exact birth"))?;
            io::Result::Ok((child, boot))
        };
        match launch() {
            Ok((child, boot)) => {
                let boot = boot.with_scope(Some(scope));
                let mut boots = self.kernel_boots.lock().unwrap();
                boots.insert(generation.clone(), Some(boot.try_clone()?));
                Ok((child, boot, log))
            }
            Err(error) => {
                let _ = scope.end();
                Err(error)
            }
        }
    }

    /// One line per kernel boot in `kernel-boot.jsonl`: how it ended and what each kernel
    /// said last (`kernel-boot.<generation>.log` holds its progress too).
    fn note_kernel_boot(
        &self,
        generation: &str,
        took: std::time::Duration,
        ended: io::Result<(crate::process::Reaped, PathBuf)>,
    ) {
        let mut line = serde_json::json!({
            "generation": generation,
            "took_ms": took.as_secs_f64() * 1e3,
        });
        match ended {
            Ok((reaped, log)) => {
                let kernels: Vec<serde_json::Value> = fs::read_to_string(log)
                    .unwrap_or_default()
                    .lines()
                    .filter_map(|row| serde_json::from_str::<serde_json::Value>(row).ok())
                    .filter(|row| row.is_object() && row["state"] != "compiling")
                    .collect();
                line["status"] = reaped.status.to_string().into();
                line["killed"] = reaped.killed.into();
                line["stragglers"] = reaped.stragglers.into();
                line["kernels"] = kernels.into();
            }
            Err(error) => line["failed"] = error.to_string().into(),
        }
        eprintln!("kernel boot: {line}");
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("kernel-boot.jsonl"))
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(error) = written {
            eprintln!("kernel-boot.jsonl: {error}");
        }
    }

    /// Start a generation's import-only executor in the background after its install, so its
    /// imports overlap the model download before the first request.
    pub fn prespawn(self: &Arc<Self>, held: HeldGeneration) {
        if !self.parent_fits(&held) {
            eprintln!(
                "executor prespawn skipped for {}: no host room",
                held.record.identity
            );
            return;
        }
        let Some((zygote, true)) = self.zygote(&held) else {
            return;
        };
        *zygote.used.lock().unwrap() = Some(Instant::now());
        let pool = Arc::downgrade(self);
        let started = std::thread::Builder::new()
            .name("executor-prespawn".into())
            .spawn(move || {
                let state = match pool.upgrade() {
                    Some(pool) => pool.import_only(&held),
                    None => ZygoteState::Off("the GPU pool ended".into()),
                };
                zygote.set(state);
            });
        if let Err(error) = started {
            eprintln!("executor prespawn: {error}");
        }
    }

    /// Whether the host has room for `generation`'s parent beside what it holds now: its
    /// measured private bytes (else the largest any parent measured). A prespawn never
    /// makes room; a request that needs a parent always gets one.
    fn parent_fits(&self, held: &HeldGeneration) -> bool {
        let need = self.first().with(|gpu| {
            let learned = |key: &str| gpu.learned.plans.get(key).map(|plan| plan.host_bytes);
            learned(&parent_key(held))
                .or_else(|| {
                    gpu.learned
                        .plans
                        .iter()
                        .filter(|(key, _)| key.starts_with("parent:"))
                        .map(|(_, plan)| plan.host_bytes)
                        .max()
                })
                .filter(|bytes| *bytes > 0)
                .unwrap_or(UNMEASURED_PARENT)
        });
        self.host_room_now().is_none_or(|room| room >= need)
    }

    /// Drop a dead parent, if it is still the generation's: the next launch starts another.
    fn forget_parent(&self, held: &HeldGeneration, dead: &Arc<Zygote>) {
        let mut zygotes = self.zygotes.lock().unwrap();
        let key = parent_app(held);
        if zygotes
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, dead))
        {
            zygotes.remove(&key);
        }
    }

    /// End a parent with no live child, for host room: one of a generation outside every
    /// warm set before a member's (`spare`: never a member's), least recently used first.
    /// Returns the bytes it held privately (at least 1 when one ended unmeasured), 0 when
    /// there is none to end.
    fn end_idle_parent(&self, spare: bool) -> u64 {
        let kept: std::collections::BTreeSet<(String, String)> = {
            let sets = self.members.lock().unwrap();
            let members = sets.values().flatten().filter(|member| member.level >= Level::Imported);
            members.map(|member| parent_app(&member.held)).collect()
        };
        let victim = {
            let mut zygotes = self.zygotes.lock().unwrap();
            let chosen = zygotes
                .iter()
                .filter(|(generation, zygote)| zygote.childless() && !(spare && kept.contains(*generation)))
                .min_by_key(|(generation, zygote)| (kept.contains(*generation), *zygote.used.lock().unwrap()))
                .map(|(generation, _)| generation.clone());
            chosen.and_then(|generation| zygotes.remove(&generation))
        };
        let Some(zygote) = victim else {
            return 0;
        };
        let freed = zygote.private_bytes();
        if let ZygoteState::Ready(parent) = std::mem::take(&mut *zygote.state.lock().unwrap()) {
            if let Err(error) = parent.shutdown() {
                eprintln!("ending an idle import-only executor: {error}");
            }
        }
        freed.max(1)
    }

    /// The generation's import-only executor, and whether the caller must start it. None
    /// when this pool spawns every executor or the generation binds no model.
    fn zygote(&self, held: &HeldGeneration) -> Option<(Arc<Zygote>, bool)> {
        if !self.config.prespawn || !binds_models(&held.record.interface) {
            return None;
        }
        let mut zygotes = self.zygotes.lock().unwrap();
        let key = parent_app(held);
        if let Some(zygote) = zygotes.get(&key) {
            return Some((zygote.clone(), false));
        }
        let zygote = Arc::new(Zygote::default());
        zygotes.insert(key, zygote.clone());
        Some((zygote, true))
    }

    /// Spawn an executor of `held` and import torch, the Runtime and the package with no
    /// device (`Start.import_only`), to fork executors from.
    fn import_only(&self, held: &HeldGeneration) -> ZygoteState {
        let started = (|| {
            let (root, socket, directory) = self.executor_endpoint()?;
            let mut executor = DeviceExecutor::spawn_owned(
                self.executor_config(held, root, socket, 1)?,
                &self.launcher,
                |_, _| Ok(()),
            )?;
            executor.retain_until_exit(directory);
            if !executor.hello.offers(device_executor::FORK)
                || !executor.hello.offers("import_only")
            {
                executor.shutdown()?;
                return Ok(ZygoteState::Off(format!(
                    "Runtime {} does not fork executors",
                    held.record.identity
                )));
            }
            let interface = self.interface_file(&executor, held)?;
            let reply = executor.command(
                &DeviceCommand::Start {
                    devices: self.lane_devices(1)?,
                    application: held.record.application.clone(),
                    package_interface: interface,
                    sequence_parallel_degree: 1,
                    import_only: true,
                },
                &mut device_executor::Baseline,
            )?;
            if !reply.ok {
                executor.shutdown()?;
                return Ok(ZygoteState::Failed(format!(
                    "import-only start refused: {}: {}",
                    reply.code, reply.detail
                )));
            }
            if let Ok(memory) = crate::host_memory::process(executor.birth.pid) {
                self.first().learn_host(
                    &parent_key(held),
                    memory.pss.saturating_sub(memory.pss_shmem),
                );
            }
            io::Result::Ok(ZygoteState::Ready(Box::new(executor)))
        })();
        started
            .unwrap_or_else(|error| ZygoteState::Failed(format!("import-only executor: {error}")))
    }

    /// A new executor root and its socket path. The socket is named through this process's
    /// descriptor of the root (Linux `sun_path` limit), or inside it for another identity.
    fn executor_endpoint(&self) -> io::Result<(PathBuf, PathBuf, File)> {
        let root = self.root.join(uuid::Uuid::new_v4().simple().to_string());
        fs::create_dir(&root)?;
        let directory = File::open(&root)?;
        let socket = if self.config.identity.is_some() {
            // Another UID cannot traverse this owner's /proc/fd magic link.
            // A deliberately short owned state root is required for this operation.
            let socket = root.join("executor");
            use std::os::unix::ffi::OsStrExt;
            if socket.as_os_str().as_bytes().len() > 107 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "configured identity needs a short owned executor socket path (Linux sun_path)",
                ));
            }
            socket
        } else {
            PathBuf::from(format!(
                "/proc/{}/fd/{}/executor",
                std::process::id(),
                directory.as_raw_fd()
            ))
        };
        Ok((root, socket, directory))
    }

    fn executor_config(
        &self,
        held: &HeldGeneration,
        root: PathBuf,
        socket: PathBuf,
        degree: u32,
    ) -> io::Result<ExecutorConfig> {
        let mut seal = Seal::prepare(
            &self.root,
            self.config.identity,
            &self.incarnation,
            &held.record.identity,
            &self.lane_devices(degree)?,
        )?;
        // A group's NCCL seal (NVLS off, peer memory only over NVLink).
        seal.group = degree > 1;
        seal.threads = self.config.threads;
        Ok(ExecutorConfig {
            python: held.record.python.clone(),
            root,
            socket,
            environment: self.config.environment.clone(),
            seal,
            generation_hold: Some(held.retention()),
            identity: self.config.identity,
            cgroup_namespace: Some(crate::scope::namespace(
                self.root.parent().unwrap_or(&self.root),
            )),
        })
    }

    /// The installed package interface, written where the executor (its identity) reads it.
    fn interface_file(
        &self,
        executor: &DeviceExecutor,
        held: &HeldGeneration,
    ) -> io::Result<PathBuf> {
        let path = executor.root_path().join("package-interface.json");
        fs::write(&path, serde_json::to_vec(&held.record.interface)?)?;
        if let Some(identity) = self.config.identity {
            identity.readable(&path)?;
        }
        Ok(path)
    }
    fn run(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
        plan: GpuPlan,
    ) -> io::Result<()> {
        let mut sessions = self.sessions.lock().unwrap();
        let key = plan.id.clone();
        let result = self.run_locked(engine, id, held, plan, &mut sessions);
        for device in &self.devices {
            device.memory.finished(&key);
        }
        if result.is_err() {
            // Broken/failed exchanges close the owner channel. Sources and the
            // journal reservation survive until the exact receiver has exited.
            sessions.remove(&key);
        }
        if !sessions.contains_key(&key) {
            self.ended(&key);
        }
        result
    }

    /// Custody's holdings as the memory policy reads them.
    fn holdings(&self) -> Vec<Holding> {
        self.resident()
            .into_iter()
            .map(|h| Holding {
                id: holding_id(&h.key, h.generation),
                bytes: h.bytes,
                readers: h.readers.iter().map(|birth| birth.pid).collect(),
                idle_ms: h.idle_ms,
                revoking: h.phase == crate::resident_custody::Phase::Revoking,
            })
            .collect()
    }

    /// One memory step on idle executors or custody of this GPU.
    fn carry_out(&self, step: &Step, sessions: &mut BTreeMap<String, Session>) -> io::Result<bool> {
        let plan = match step {
            Step::Revoke(id) => {
                let Some(custody) = &self.custody else {
                    return Ok(false);
                };
                let found = custody
                    .lock()
                    .unwrap()
                    .holdings()
                    .into_iter()
                    .find(|h| holding_id(&h.key, h.generation) == *id);
                let Some(holding) = found else {
                    return Ok(false);
                };
                custody
                    .lock()
                    .unwrap()
                    .begin_revoke(&holding.key, holding.generation)?;
                // Every other executor is idle: each releases at once.
                for session in sessions.values_mut() {
                    if let Err(error) = release_revoked(custody, &mut session.executor) {
                        eprintln!("resident revoke left a lease charged: {error}");
                    }
                }
                log_released(custody.lock().unwrap().collect());
                return Ok(true);
            }
            Step::Trim(id, bytes) => {
                let Some(custody) = &self.custody else {
                    return Ok(false);
                };
                let found = custody
                    .lock()
                    .unwrap()
                    .holdings()
                    .into_iter()
                    .find(|h| holding_id(&h.key, h.generation) == *id);
                let Some(holding) = found else {
                    return Ok(false);
                };
                let regions = custody.lock().unwrap().trim_regions(
                    &holding.key,
                    holding.generation,
                    *bytes,
                );
                // Every reader must be an executor here that trims (`weights.trim/1`); an
                // older one, or a reader this service does not hold, gets the whole revoke.
                let trims = |s: &Session| s.executor.hello.offers("weights.trim/1");
                let readers = sessions
                    .values()
                    .filter(|s| holding.readers.contains(&s.executor.birth))
                    .collect::<Vec<_>>();
                let Some(regions) = regions.filter(|_| {
                    readers.len() == holding.readers.len() && readers.iter().all(|s| trims(s))
                }) else {
                    return self.carry_out(&Step::Revoke(id.clone()), sessions);
                };
                for session in sessions
                    .values_mut()
                    .filter(|s| holding.readers.contains(&s.executor.birth))
                {
                    let reply = session.executor.command(
                        &DeviceCommand::Revoke {
                            layout: holding.key.layout.clone(),
                            generation: holding.generation,
                            regions: regions.clone(),
                        },
                        &mut device_executor::Baseline,
                    )?;
                    if !reply.ok {
                        // Its mapping stays a reference: the regions stay charged, nothing freed.
                        eprintln!("trim of {} refused: {} {}", id, reply.code, reply.detail);
                        return Ok(false);
                    }
                }
                let freed =
                    custody
                        .lock()
                        .unwrap()
                        .trim(&holding.key, holding.generation, &regions);
                eprintln!(
                    "resident GPU weights trimmed: {} on {}, {freed} bytes in {} regions",
                    holding.key.layout,
                    holding.key.device,
                    regions.len()
                );
                return Ok(true);
            }
            Step::Unmap(plan) => {
                let Some(session) = sessions.get_mut(plan) else {
                    return Ok(false);
                };
                if session.executor.hello.offers("weight_plane/1") {
                    let reply = session.executor.command(
                        &DeviceCommand::Budget {
                            vram_bytes: 0,
                            pinned_bytes: -1,
                            cap_bytes: None,
                        },
                        &mut device_executor::Baseline,
                    );
                    if let Some(reply) = reply.ok().filter(|reply| reply.ok) {
                        self.observe(
                            plan,
                            session.degree,
                            plane_facts(reply.plane.as_ref()),
                            &reply.rank_planes,
                            Some(false),
                        );
                        return Ok(true);
                    }
                }
                plan
            }
            Step::End(plan) => plan,
            Step::Shrink(..) => return Ok(false),
        };
        let Some(session) = sessions.remove(plan) else {
            return Ok(false);
        };
        // Its context leaves the device before anything is granted in its place. Another
        // tenant's broken channel is its own failure, never this call's.
        if let Err(error) = session.executor.shutdown() {
            eprintln!("memory: ending idle executor {plan}: {error}");
        }
        self.ended(plan);
        Ok(true)
    }

    fn run_locked(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
        plan: GpuPlan,
        sessions: &mut BTreeMap<String, Session>,
    ) -> io::Result<()> {
        if let Some(custody) = &self.custody {
            log_released(custody.lock().unwrap().collect());
        }
        let mut ended = vec![];
        for (key, session) in sessions.iter() {
            if process_ended(&session.executor.birth)? {
                ended.push(key.clone());
            }
        }
        for key in ended {
            sessions.remove(&key);
            self.ended(&key);
        }
        let cold = !sessions.contains_key(&plan.id);
        let mut load_caps = vec![];
        if cold {
            self.host_room(&plan.id, sessions, false);
            // A context and the first working set are reserved on every GPU of the group
            // before the process exists.
            load_caps = self.decide(&plan.id, plan.degree, true, sessions)?;
            for device in self.lane(plan.degree)? {
                device
                    .memory
                    .with(|gpu| gpu.starting(&plan.id, gpu.spawn_need(&plan.id)));
            }
            let on_birth = |birth: &ProcessBirth, cancel: &Cancellation| {
                // Rank 0 drives the first GPU; followers are named after Start.
                self.first().with(|gpu| gpu.spawned(&plan.id, birth.pid));
                let cancel = cancel.clone();
                let request = id.to_string();
                engine.register_managed(
                    id,
                    birth.clone(),
                    Arc::new(move || cancel.cancel(&request)),
                )
            };
            let session = match self.new_session(engine, &held, &plan, on_birth) {
                Err(error) if error.kind() == io::ErrorKind::Unsupported => {
                    engine.finish(id, Outcome::Failed(error.to_string()))?;
                    return Ok(());
                }
                session => session?,
            };
            sessions.insert(plan.id.clone(), session);
        } else {
            let session = &sessions[&plan.id];
            let cancel = session.executor.cancellation();
            let request = id.to_string();
            engine.register_managed(
                id,
                session.executor.birth.clone(),
                Arc::new(move || cancel.cancel(&request)),
            )?;
        }
        let executor = &sessions[&plan.id].executor;
        let facts = crate::journal::ExecutorFacts {
            pid: executor.birth.pid,
            runtime_version: executor.hello.runtime_version.clone(),
            tensorfs_version: executor.hello.tensorfs_version.clone(),
        };
        if !engine.authorize_managed(id, Some(facts))? {
            engine.finish_stopped(id)?;
            return Ok(());
        }
        let fresh = plan.clone();
        let error = {
            // Out of the map for the call: its requests may unmap or end the others.
            let mut session = sessions.remove(&plan.id).expect("session retained above");
            match self.call(
                engine,
                id,
                &held,
                plan,
                &load_caps,
                &mut session,
                sessions,
                false,
            ) {
                Ok(true) => {
                    self.note_level(&session);
                    sessions.insert(session.plan.clone(), session);
                    return Ok(());
                }
                // Not reusable: gone (exit observed) before its context is released.
                Ok(false) => return session.executor.terminate().map(drop),
                Err(error) if undelivered(&error) => ending(error, session.executor),
                Err(error) => return Err(ended_with(engine, id, error, session.executor)),
            }
        }; // the rest of the ended session (sources, grants) goes here
        // Never started: once that exit is observed, this dispatch runs it on a fresh executor
        // (cold, so a second loss is FAILED). An unproven exit stays charged.
        if !undelivered(&error) {
            return Err(error);
        }
        eprintln!("execution {id}: {error}; starting it on a fresh executor");
        for device in &self.devices {
            device.memory.finished(&fresh.id);
        }
        self.ended(&fresh.id);
        if !engine.redeliver(id)? {
            return Ok(()); // canceled meanwhile
        }
        self.run_locked(engine, id, held, fresh, sessions)
    }

    /// Launch an executor for `plan` (fork from its generation's import-only executor, else
    /// spawn) and open its selected metadata sources and CPU-buffer grants. The advertised
    /// capabilities select sealed-tier delivery or the supported legacy peer route.
    fn new_session(
        &self,
        engine: &Arc<Engine>,
        held: &HeldGeneration,
        plan: &GpuPlan,
        on_birth: impl Fn(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Session> {
        let launched = Instant::now();
        let (root, socket, directory) = self.executor_endpoint()?;
        let config = self.executor_config(held, root, socket, plan.degree)?;
        // A forked child takes its lane and seal with its environment, so a group forks from
        // the same import-only parent (sealed to the first GPU) as a single GPU does.
        let mut config = Some(config);
        let mut forked = None;
        // A parent found dead is replaced at once: importing a new one costs this executor
        // what a spawn would, and the next ones fork again.
        for _ in 0..2 {
            let Some((zygote, start)) = self.zygote(held) else {
                break;
            };
            if start {
                zygote.set(self.import_only(held));
            }
            match zygote.fork(config.take().expect("one config per launch"), &on_birth)? {
                Forked::Ready(executor) => {
                    forked = Some(*executor);
                    break;
                }
                Forked::Refused(returned, reason) => {
                    eprintln!("executor fork refused, spawning: {reason}");
                    config = Some(*returned);
                    if zygote.failed() {
                        self.forget_parent(held, &zygote);
                    }
                    break;
                }
                Forked::Lost(returned, reason) => {
                    eprintln!("import-only executor lost ({reason}); starting another");
                    config = Some(*returned);
                    self.forget_parent(held, &zygote);
                }
            }
        }
        let (mut executor, mode) = match forked {
            Some(executor) => (executor, "fork"),
            None => (
                DeviceExecutor::spawn_owned(
                    config.expect("a refused fork returns its config"),
                    &self.launcher,
                    &on_birth,
                )?,
                "spawn",
            ),
        };
        let launch = Launch {
            mode,
            ms: launched.elapsed().as_secs_f64() * 1e3,
            start_ms: 0.0,
            start: vec![],
        };
        executor.retain_until_exit(directory);
        executor.retain_until_exit(WakeOnExit(Arc::downgrade(engine)));
        executor.keep_memo(self.memo.clone());
        // Weights come from the machine's sealed host tier: handed out while it fills, and
        // streamed when it does not fit. An executor whose SDK cannot adopt a sealed layout
        // (an older TensorFS or Runtime: version skew) still serves: it reads its weights from
        // the store and the page cache itself, and every call it serves says so in its log.
        let sealed = ["weight_plane/1", "host_tiers.sealed/1"]
            .into_iter()
            .find(|needed| !executor.hello.offers(needed));
        let unsealed = sealed.map(|missing| {
            let text = format!(
                "its executor (Runtime {}, TensorFS {}) lacks {missing}: its weights are read \
                 from the store and the page cache, without the machine's sealed host tier",
                executor.hello.runtime_version, executor.hello.tensorfs_version
            );
            eprintln!("{}: {text}", plan.id);
            text
        });
        let selections = plan.selections();
        let id = self.serving.next.fetch_add(1, Ordering::Relaxed);
        self.serving.held.lock().unwrap().insert(
            id,
            selections.iter().map(|s| s.manifest.clone()).collect(),
        );
        // Kept until the process is seen to exit, not until its session is dropped: an
        // executor whose exit is unproven may still read these files.
        executor.retain_until_exit(ServingHold {
            serving: self.serving.clone(),
            id,
        });
        let sources = Arc::new(ModelSources::open_shared(self.store.clone(), &selections)?);
        // An executor that reads holes from object files gets only what it streams staged.
        let staged = executor.hello.offers("host_tiers.staged/1");
        let peer = self.host.register_peer(executor.observer_pidfd()?, staged);
        let grants = selections
            .iter()
            .map(|selected| {
                Ok(HostGrant {
                    manifest: selected.manifest.clone(),
                    header: sources.authorized_header(&selected.manifest)?,
                    components: selected.components.iter().cloned().collect(),
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        // TensorD starts filling CPU layouts through TensorFS while Runtime imports and
        // constructs the model. GPU allocation/transfers still happen in the executor.
        if unsealed.is_none() {
            self.host.prepare(grants.clone(), staged);
        }
        Ok(Session {
            plan: plan.id.clone(),
            generation: plan.generation.clone(),
            application: held.record.application.clone(),
            degree: plan.degree,
            followers: vec![],
            loaded: false,
            executor,
            budget_cells: BTreeMap::new(),
            sources,
            peer,
            grants,
            sharing: false,
            launch,
            invoked: false,
            evictions: 0,
            unsealed,
        })
    }

    /// Load (once), grant and invoke (unless `load_only`: a prewarm, no request). Ok(false):
    /// the executor must not be reused.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: &HeldGeneration,
        plan: GpuPlan,
        load_caps: &[Option<u64>],
        session: &mut Session,
        others: &mut BTreeMap<String, Session>,
        load_only: bool,
    ) -> io::Result<bool> {
        let retained = session.loaded;
        if retained {
            session.executor.begin_request();
        }
        // Every rank of a group reads its own GPU's cap and cell (`rank_cells/1`).
        let ranked = plan.degree > 1;
        if !session.loaded {
            session.sharing = self.degree2(&plan.id, plan.degree, &session.executor)?;
        }
        let sharing = session.sharing;
        let lane = self.lane_devices(plan.degree)?;
        let mut callbacks = Callbacks {
            engine,
            id,
            cells: &mut session.budget_cells,
            completed: 0,
            host: &self.host,
            sources: &session.sources,
            peer: session.peer,
            grants: &session.grants,
            birth: session.executor.birth.clone(),
            custody: self.custody.as_ref().filter(|_| sharing),
            exit: session.executor.observer_pidfd()?,
            pool: self,
            plan: &plan.id,
            degree: plan.degree,
            others,
            store: &self.store,
            spool: None,
        };
        if !session.loaded {
            let interface_path = self.interface_file(&session.executor, held)?;
            let used: Vec<_> = self.lane(plan.degree)?.iter().map(|d| d.memory.used()).collect();
            let starting = Instant::now();
            // Rank 0 spawns and forms every follower inside this command; its watch meters
            // the whole group's work, so formation ends only on measured lack of progress.
            let started = command_ok(session.executor.command_with(
                &DeviceCommand::Start {
                    devices: lane.clone(),
                    application: held.record.application.clone(),
                    package_interface: interface_path.clone(),
                    sequence_parallel_degree: plan.degree,
                    import_only: false,
                },
                &Group {
                    rank_cells: ranked,
                    ..Group::default()
                },
                &mut callbacks,
            )?)?;
            if ranked && (1..plan.degree).any(|rank| !callbacks.cells.contains_key(&rank)) {
                return Err(io::Error::other(Refused {
                    code: "group_unformed".into(),
                    detail: "a follower's budget cell did not reach the machine".into(),
                }));
            }
            // An executor reads its context from its own NVML row, which a container whose
            // pids are not NVML's lacks (vast: every want then priced it at 1 GiB, against
            // ~0.25 real). Until it reports one, its context is what each GPU's use grew by
            // across its start: no call runs beside it and no weight is loaded yet.
            for (device, before) in self.lane(plan.degree)?.iter().zip(used) {
                if let Some(grew) = before
                    .zip(device.memory.used())
                    .map(|(before, after)| after.saturating_sub(before))
                    .filter(|grew| *grew > 0)
                {
                    let context = Facts {
                        context: Some(grew),
                        ..Facts::default()
                    };
                    device.memory.observe(&plan.id, context, None);
                }
            }
            session.launch.start_ms = starting.elapsed().as_secs_f64() * 1e3;
            session.launch.start =
                serde_json::from_value(started.stages.clone()).unwrap_or_default();
            if started.follower_pids.len() + 1 != plan.degree as usize {
                return Err(io::Error::other(Refused {
                    code: "group_unformed".into(),
                    detail: format!(
                        "a {}-GPU start reported followers {:?}",
                        plan.degree, started.follower_pids
                    ),
                }));
            }
            // Only processes in rank 0's own group are taken as its followers.
            let members = crate::process::group_members(&session.executor.birth);
            session.followers = started
                .follower_pids
                .iter()
                .filter_map(|pid| members.iter().find(|m| m.pid == *pid))
                .filter_map(|birth| crate::process::Exact::open(birth).ok().flatten())
                .collect();
            // NVML charges each GPU's own process: rank r drives the group's GPU r.
            for (device, pid) in self
                .lane(plan.degree)?
                .iter()
                .skip(1)
                .zip(&started.follower_pids)
            {
                device.memory.with(|gpu| gpu.spawned(&plan.id, *pid));
            }
            let store = self.store.root().to_string_lossy().into_owned();
            let mut models: Vec<ModelLoad> = plan
                .slots
                .iter()
                .map(|slot| {
                    let mut binding = slot.binding.clone();
                    binding.package_interface = interface_path.to_string_lossy().into();
                    binding.store = store.clone();
                    ModelLoad {
                        binding,
                        budgets: Budgets {
                            declared_weight_bytes: slot.selected_encoded_bytes,
                        },
                    }
                })
                .collect();
            let first = models.first().cloned().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "GPU plan binds no model")
            })?;
            if models.len() == 1 {
                models.clear(); // the one-model form
            }
            // The group's per-GPU ceiling: the smallest card.
            let device_total = self
                .lane(plan.degree)?
                .iter()
                .map(|device| device.memory.sample().map(|sample| sample.total))
                .collect::<Option<Vec<_>>>()
                .and_then(|totals| totals.into_iter().min());
            let plane = session.executor.hello.offers("weight_plane/1");
            // The pinned budget comes with the Load, so no fill pins past it; an executor
            // without `load_pinned/1` gets it after, as before.
            let at_load = plane && session.executor.hello.offers("load_pinned/1");
            // Its share of the machine's pinned total, ahead of every other tenant.
            let pinned = self
                .first()
                .pinned_budgets(&plan.id)
                .and_then(|split| split.get(&plan.id).copied())
                .map_or(-1, |share| i64::try_from(share).unwrap_or(i64::MAX));
            let started = std::time::Instant::now();
            let (load_cap, load_group) = rank_grant(load_caps);
            let loaded = command_ok(session.executor.command_with(
                &DeviceCommand::Load {
                    construction: plan.id.clone(),
                    devices: lane.clone(),
                    sequence_parallel_degree: plan.degree,
                    binding: Box::new(first.binding),
                    budgets: first.budgets,
                    models,
                    authorized_device_limit_bytes:
                        device_total.or(self.config.authorized_device_limit_bytes),
                    attention_pin: String::new(),
                    stages: false,
                    sealed_tiers: session.unsealed.is_none(),
                    model_sources: true,
                    staged_tiers: session.unsealed.is_none(),
                    pinned_bytes: at_load.then_some(pinned),
                    device_weights: sharing,
                    cap_bytes: load_cap,
                },
                &load_group,
                &mut callbacks,
            )?)?;
            let facts = self.observe(
                &plan.id,
                plan.degree,
                load_facts(loaded.facts.as_ref()),
                &loaded.rank_planes,
                Some(false),
            );
            for device in self.lane(plan.degree)? {
                device
                    .memory
                    .learn_load(&plan.id, facts.weights, facts.weights_floor);
            }
            self.record_load(
                &plan.id,
                started.elapsed(),
                &loaded,
                session.executor.birth.pid,
                &session.launch,
            )?;
            if plane && !at_load {
                command_ok(session.executor.command(
                    &DeviceCommand::Budget {
                        vram_bytes: -1,
                        pinned_bytes: pinned,
                        cap_bytes: None,
                    },
                    &mut callbacks,
                )?)?;
            }
            command_ok(session.executor.command(
                &DeviceCommand::Activate {
                    construction: plan.id.clone(),
                },
                &mut callbacks,
            )?)?;
            session.loaded = true;
        }
        if load_only {
            return Ok(true);
        }
        if let Some(text) = &session.unsealed {
            if let Err(error) = engine.append_log(id, "warning", text) {
                eprintln!("run {id}: {text} ({error})");
            }
        }
        let record = engine.get(id)?;
        let prepared = session.executor.command(
            &DeviceCommand::PrepareRequest {
                request_id: id.into(),
                construction: plan.id.clone(),
                entrypoint: plan.entrypoint.clone(),
                payload: record.invocation.input,
                attention_kernel: record.invocation.attention_kernel.clone(),
                input_metadata: record
                    .invocation
                    .inputs
                    .iter()
                    .map(|i| (i.input_id.clone(), serde_json::json!({"input_id":i.input_id,"media_type":i.media_type,"digest":i.digest,"length":i.length,"order":i.order})))
                    .collect(),
            },
            &mut callbacks,
        );
        // A retained executor that went away while idle: PrepareRequest enters no handler,
        // so the attempt never started (see `run_locked`).
        let prepared = prepared.map_err(|error| match retained && channel_lost(&error) {
            true => io::Error::other(Undelivered(format!(
                "retained executor gone before the request reached it: {error}"
            ))),
            false => error,
        })?;
        if !prepared.ok
            && !matches!(
                prepared.code.as_str(),
                "executor_not_ready" | "poisoned_generation"
            )
        {
            // A pre-entry refusal leaves the executor Ready; the run ends with its reason.
            let terminal = if prepared.terminal.is_empty() {
                "refused"
            } else {
                &prepared.terminal
            };
            let failure =
                Failure::executor(terminal, &prepared.origin, &prepared.code, &prepared.detail);
            keep_triage(
                engine,
                id,
                &session.executor,
                &device_executor::Outcome {
                    terminal: terminal.into(),
                    origin: prepared.origin.clone(),
                    code: prepared.code.clone(),
                    message: prepared.detail.clone(),
                    traceback: prepared.traceback.clone(),
                },
            );
            engine.finish(id, Outcome::Failed(failure.encode()))?;
            return Ok(true);
        }
        let prepared = command_ok(prepared)?;
        let shape = crate::memory::learned::shape_cell(&prepared.features);
        for device in self.lane(plan.degree)? {
            device.memory.with(|gpu| gpu.set_shape(&plan.id, &shape));
        }
        if let Some((from, ratio)) = self
            .first()
            .with(|gpu| gpu.shape(&plan.id))
            .and_then(|learned| learned.estimated_from)
        {
            crate::memory::note(serde_json::json!({"event": "estimate", "plan": plan.id,
                "shape": shape, "from": from, "ratio": ratio}));
        }
        let spool = if let Some(identity) = self.config.identity {
            // Keep Journal/results/admin paths private to the core. A separate peer-owned
            // spool lives only inside this executor's already authorized output directory.
            let spool = session.executor.root_path().join(format!("output-{id}"));
            fs::create_dir(&spool)?;
            identity.own(&spool)?;
            spool
        } else {
            engine.staging(id)?
        };
        let _spool = crate::execution::Spool(spool.clone());
        let inputs = stage_inputs(
            &self.store,
            self.config.identity,
            &spool,
            &record.invocation.inputs,
        )?;
        // Degree 2 per call: on while the whole construction and its activations fit beside the
        // other tenants now. A sharing executor that no longer fits lets go of what it maps
        // first, so this call streams privately and the ladder can reclaim those holdings.
        let attaches = session.executor.hello.offers("weights.attach/1");
        let share = self.degree2(&plan.id, plan.degree, &session.executor)?;
        if session.sharing && !share {
            if let Some(custody) = &self.custody {
                detach(custody, &mut session.executor)?;
            }
        }
        session.sharing = share;
        callbacks.custody = self.custody.as_ref().filter(|_| share);
        self.shed(&plan.id, callbacks.others);
        // A real grant for the whole call: one tenant needs no per-stage turns.
        // A group's cap holds on every GPU of it (each rank caps its own process).
        let caps = self.decide(&plan.id, plan.degree, false, callbacks.others)?;
        let (cap, group) = rank_grant(&caps);
        // Every executor of the cohort caps its whole process (`process_cap/1`): the plane
        // derives its budget inside the cap.
        let (plane_budget_bytes, cap_bytes) = (-1, cap);
        // Each GPU's floor watchdog writes the cell of its own rank's process.
        for ((rank, device), own) in (0u32..).zip(self.lane(plan.degree)?).zip(&caps) {
            if let Some(own) = *own {
                let cell = callbacks
                    .cells
                    .get(&rank)
                    .map(File::try_clone)
                    .transpose()?;
                device.memory.running(&plan.id, own, cell);
            }
        }
        callbacks.spool = Some(spool.clone());
        let invoked = Instant::now();
        let first = !session.invoked;
        session.invoked = true;
        let reply = session.executor.command_with(
            &DeviceCommand::Invoke {
                request_id: id.into(),
                construction: plan.id.clone(),
                entrypoint: plan.entrypoint,
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: record.invocation.attention_kernel.clone(),
                plane_budget_bytes,
                stages: false,
                cap_bytes,
                inputs: inputs.inputs,
                trees: inputs.trees,
                floor_bytes: self.first().floor(),
                activation_bytes: self.first().with(|gpu| gpu.seeds(&plan.id)),
                squeezed_bytes: self.first().with(|gpu| gpu.squeezed(&plan.id)),
                device_weights: attaches.then_some(share),
            },
            &group,
            &mut callbacks,
        )?;
        // A call that evicted nothing kept every weight it uses mapped: that is what it maps.
        let evictions = reply.plane.as_ref().and_then(|p| p.evictions);
        let mapped = reply
            .plane
            .as_ref()
            .filter(|_| plan.degree == 1 && evictions == Some(session.evictions))
            .and_then(|p| {
                let own = u64::try_from(p.committed_bytes?).ok()?;
                Some(own + p.shared_bytes.and_then(|b| u64::try_from(b).ok()).unwrap_or(0))
            })
            .filter(|bytes| *bytes > 0);
        session.evictions = evictions.unwrap_or(session.evictions);
        self.learn(
            &plan.id,
            plan.degree,
            &shape,
            &reply,
            session.executor.birth.pid,
            mapped,
        );
        self.record_invoke(&plan.id, first, invoked.elapsed(), &reply);
        record_measurements(engine, id, &reply);
        if reply.attention_applied {
            if let Err(error) = engine.apply_attention(id) {
                eprintln!("run {id}: applied attention pin not recorded: {error}");
            }
        }
        let mut facts = plane_facts(reply.plane.as_ref());
        facts.activation = facts.activation.or_else(|| {
            reply
                .metrics
                .as_ref()
                .and_then(|m| m.activation_peak_bytes)
                .and_then(|v| u64::try_from(v).ok())
        });
        self.observe(&plan.id, plan.degree, facts, &reply.rank_planes, Some(true));
        // Every executor's weights are the tier's: none pins memory of its own.
        self.host_ledger.private(&plan.id, None);
        if let Some(plane) = &reply.plane {
            crate::memory::note(
                serde_json::json!({"event": "call", "plan": plan.id, "id": id, "degree": plan.degree,
                "cap": cap, "cap_bytes": plane.cap_bytes, "process": plane.process_bytes,
                "context": plane.context_bytes, "committed": plane.committed_bytes,
                "activation": plane.activation_peak_bytes, "oom_retries": plane.oom_retries,
                "alloc_retries": plane.alloc_retries, "cache_releases": plane.cache_releases,
                "paged_out": plane.paged_out,
                "evictions": plane.evictions, "h2d_bytes": plane.h2d_bytes,
                "rank_process": reply.rank_planes.iter()
                    .map(|r| r.as_ref().map(|r| r.process_bytes)).collect::<Vec<_>>(),
                "rank_cap": reply.rank_planes.iter()
                    .map(|r| r.as_ref().map(|r| r.cap_bytes)).collect::<Vec<_>>()}),
            );
        }
        // Rank 0 answered, so a follower that ended during the call is the group's first fault.
        let lost = self.lost_followers(plan.degree, &session.followers);
        if !reply.quiescent || !reply.poisoned.is_empty() || !lost.is_empty() {
            // The run ends once the executor is gone; its own reason travels with it, typed.
            let (code, message) = reply
                .outcome
                .as_ref()
                .filter(|o| !o.code.is_empty())
                .map(|o| (o.code.clone(), o.message.clone()))
                .unwrap_or_else(|| ("executor_poisoned".into(), String::new()));
            let (code, message) = match lost.is_empty() {
                true => (code, message),
                false => (
                    "group_broken".into(),
                    format!("{lost} during the call, first; then {code}: {message}"),
                ),
            };
            return Err(io::Error::other(Refused {
                code,
                detail: format!(
                    "{message}; the executor did not settle quiescent (poisoned: {})",
                    reply.poisoned
                ),
            }));
        }
        let outcome = reply
            .outcome
            .as_ref()
            .ok_or_else(|| io::Error::other("device outcome absent"))?;
        match outcome.terminal.as_str() {
            "succeeded" => {
                let result = (|| {
                    let (value, bindings) =
                        device_executor::postprocess(&session.executor.codec(), &spool, &reply)?;
                    engine.managed_result(id, &spool, value, output_bindings(bindings)?)
                })();
                if let Err(error) = result {
                    // The invocation is already quiescent; a codec/custody
                    // failure cannot cause authored work to be dispatched again.
                    engine.finish(
                        id,
                        Outcome::Failed(
                            crate::journal::Failure::custody(&error.to_string()).encode(),
                        ),
                    )?;
                }
            }
            "canceled" if engine.get(id)?.cancel_actor.is_some() => {
                engine.finish(id, Outcome::Canceled)?;
            }
            _ => {
                keep_triage(engine, id, &session.executor, outcome);
                engine.finish(
                    id,
                    Outcome::Failed(
                        crate::journal::Failure::executor(
                            &outcome.terminal,
                            &outcome.origin,
                            &outcome.code,
                            &outcome.message,
                        )
                        .encode(),
                    ),
                )?;
            }
        }
        // Degree 2 once this call measured that the construction fits: an executor loaded
        // without it (its plan never ran here) offers everything it holds at this boundary.
        if !session.sharing && self.degree2(&plan.id, plan.degree, &session.executor)? {
            session.sharing = true;
            callbacks.custody = self.custody.as_ref();
        }
        if let Some(custody) = callbacks.custody {
            // The call is over: the GPU regions it filled outlive this executor, and
            // generations revoked meanwhile are let go at this idle boundary.
            let shared = session
                .executor
                .command(&DeviceCommand::Share, &mut callbacks);
            if let Ok(reply) = &shared {
                // What it exported is custody's to count now: its report says what it still
                // holds. Its last one counted those bytes too, which hid that much external
                // memory and raised its next cap by it (run 4391: 22.52 GB for 21.84 GB of room).
                let facts = plane_facts(reply.plane.as_ref());
                self.observe(&plan.id, plan.degree, facts, &reply.rank_planes, None);
            }
            let released = shared.and_then(|_| release_revoked(custody, &mut session.executor));
            if let Err(error) = released {
                eprintln!("device weights share/revoke failed; replacing the executor: {error}");
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Degree 2 for `plan`'s executor: GPU weights the machine keeps across executors. It
    /// keeps every component resident until revoked, and a running call never revokes what
    /// it reads, so only when the whole construction and its activations fit beside the other
    /// tenants, as measured here or in an earlier run (unmeasured: Degree 1, where stages
    /// evict each other). World-one only: a group shares nothing.
    fn degree2(&self, plan: &str, degree: u32, executor: &DeviceExecutor) -> io::Result<bool> {
        Ok(self.custody.is_some()
            && degree == 1
            && executor.hello.offers("weights.attach/1")
            && self.lane(1)?[0]
                .memory
                .fits_resident(plan, || self.holdings()))
    }

    /// Before a spawn: the host has room for the executor's private bytes as measured in
    /// earlier runs, or gives it back in order: unheld sealed layouts (their bytes come from
    /// the page cache or the store again), then import-only parents with no live child, then
    /// idle executors, least recently used first.
    /// A plan never measured asks nothing; nothing is refused for the room that is left.
    /// Within each kind, what is outside every warm set goes before a member; `spare`
    /// (a member's own admission, a prefetch) never takes from a member. Returns whether
    /// the room is there.
    fn host_room(&self, plan: &str, sessions: &mut BTreeMap<String, Session>, spare: bool) -> bool {
        let need = self.first().with(|gpu| {
            gpu.learned
                .plans
                .get(plan)
                .map_or(0, |learned| learned.host_bytes)
        });
        if need == 0 {
            return true;
        }
        loop {
            // A warm member is discretionary: in shared mode it leaves the host's other
            // programs their recent peak. A request takes what the host has.
            let available = match spare {
                true => self.host_room_now(),
                false => u64::try_from(crate::host_memory::read().available).ok(),
            };
            let Some(available) = available else {
                return true;
            };
            if available >= need {
                return true;
            }
            // Unheld sealed layouts, then a parent with no live child (cheaper to recreate
            // than an executor with its weights), then idle executors.
            let released = self.host.release(need - available);
            let ended = released == 0
                && (self.end_idle_parent(spare) > 0
                    || match self.first().with(|gpu| gpu.lru_idle(plan, spare)) {
                        Some(victim) => self
                            .carry_out(&Step::End(victim), sessions)
                            .unwrap_or(false),
                        None => false,
                    });
            crate::memory::note(serde_json::json!({"event": "host_room", "plan": plan,
                "need": need, "available": available, "released": released, "ended": ended}));
            if released == 0 && !ended {
                return false;
            }
        }
    }

    /// What a call measured, for later executors and runs: its shape's activation growth
    /// (the call's peak and each stage method's), its context, its private host bytes.
    fn learn(
        &self,
        plan: &str,
        degree: u32,
        shape: &str,
        reply: &Frame,
        pid: u32,
        mapped: Option<u64>,
    ) {
        let metrics = reply.metrics.clone().unwrap_or_default();
        let plane = reply.plane.clone().unwrap_or_default();
        let peak = known(plane.activation_peak_bytes)
            .or(known(metrics.activation_peak_bytes))
            .unwrap_or(0);
        let methods = metrics
            .activation_peaks
            .iter()
            .filter_map(|(method, bytes)| Some((method.clone(), u64::try_from(*bytes).ok()?)))
            .collect();
        let shape = metrics.shape_cell.as_deref().unwrap_or(shape);
        // Every rank of a group runs the same shape on its own GPU, and learns only the context
        // its own process measured there.
        for (rank, device) in self.lane(degree).unwrap_or_default().iter().enumerate() {
            device
                .memory
                .learn_call(plan, shape, peak, &methods, rank_context(reply, rank), mapped);
        }
        // The squeezed rooms are those of GPU 0's process, whose facts these are.
        self.first().learn_squeezed(plan, shape, &plane.squeezed);
        if let Ok(host) = crate::host_memory::process(pid) {
            self.first()
                .learn_host(plan, host.pss.saturating_sub(host.pss_shmem));
        }
    }

    /// Idle tenants pinning more than their share of the host's pinned total give it back
    /// (their planes punch the least recently used regions; page cache and disk stay beneath)
    /// before `plan` runs. An idle executor's pinned tier is optional: a failure is noted.
    fn shed(&self, plan: &str, others: &mut BTreeMap<String, Session>) {
        let Some(split) = self.first().pinned_budgets(plan) else {
            return;
        };
        for (other, session) in others.iter_mut() {
            let budget = self.first().with(|gpu| gpu.facts(other).pinned_budget);
            let Some(share) = split.get(other).copied() else {
                continue;
            };
            if !session.executor.hello.offers("weight_plane/1")
                || budget.is_none_or(|budget| share >= budget)
            {
                continue;
            }
            let reply = session.executor.command(
                &DeviceCommand::Budget {
                    vram_bytes: -1,
                    pinned_bytes: i64::try_from(share).unwrap_or(i64::MAX),
                    cap_bytes: None,
                },
                &mut device_executor::Baseline,
            );
            match reply {
                Ok(reply) if reply.ok => {
                    let degree = session.degree;
                    let facts = plane_facts(reply.plane.as_ref());
                    self.observe(other, degree, facts, &reply.rank_planes, None);
                    crate::memory::note(serde_json::json!({"event": "shed", "plan": other,
                        "pinned_budget": share, "for": plan}));
                }
                other_reply => crate::memory::note(serde_json::json!({"event": "shed_failed",
                    "plan": other, "detail": format!("{:?}", other_reply.map(|r| r.code))})),
            }
        }
    }

    /// Followers whose process has ended, as people name their GPUs ("GPU 1's process 3224
    /// ended"); empty while the whole group lives.
    fn lost_followers(&self, degree: u32, followers: &[crate::process::Exact]) -> String {
        let lane = self.lane(degree).unwrap_or_default();
        followers
            .iter()
            .enumerate()
            .filter(|(_, follower)| follower.ended())
            .map(|(index, follower)| {
                let gpu = lane.get(index + 1).map_or("?", |d| d.entry.as_str());
                format!("GPU {gpu}'s process {} ended", follower.birth.pid)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// An executor's out-of-memory retry asks for `free_bytes` from the other tenants: idle
    /// weights leave first, then idle processes; its cap rises into what they gave.
    fn room_for(
        &self,
        plan: &str,
        degree: u32,
        free_bytes: u64,
        cells: &BTreeMap<u32, File>,
        others: &mut BTreeMap<String, Session>,
    ) -> io::Result<Option<u64>> {
        let mut caps = Vec::new();
        for (index, device) in self.lane(degree)?.iter().enumerate() {
            caps.push(device.memory.make_room(
                plan,
                free_bytes,
                cells.get(&(index as u32)),
                || if index == 0 { self.holdings() } else { vec![] },
                |step| self.carry_out(step, others),
            )?);
        }
        // Each rank's own cell carries its cap. The scalar reply belongs to rank 0 alone,
        // just as in Load and Invoke; a sibling's cap says nothing about its GPU.
        Ok(rank_grant(&caps).0)
    }
}

/// A grant from one cap per GPU: rank 0's, and for a group every rank's own. A GPU that
/// cannot say lifts its own rank's cap (-1). None: rank 0's GPU could not say.
fn rank_grant(caps: &[Option<u64>]) -> (Option<u64>, Group) {
    let rank_caps = if caps.len() > 1 {
        caps.iter()
            .map(|cap| cap.map_or(-1, |cap| i64::try_from(cap).unwrap_or(i64::MAX)))
            .collect()
    } else {
        vec![]
    };
    (
        caps.first().copied().flatten(),
        Group {
            rank_caps,
            ..Group::default()
        },
    )
}

fn holding_id(key: &HoldingKey, generation: u64) -> String {
    let variant = match key.variant.as_str() {
        "" => String::new(),
        variant => format!("+{variant}"),
    };
    format!("{}/{}{variant}#{generation}", key.device, key.layout)
}

fn known(value: Option<i64>) -> Option<u64> {
    value.and_then(|v| u64::try_from(v).ok())
}

/// The context rank `rank`'s own process measured: rank 0's plane, else its `rank_planes` row.
fn rank_context(reply: &Frame, rank: usize) -> Option<u64> {
    let own = match rank {
        0 => reply.plane.as_ref(),
        _ => reply.rank_planes.get(rank - 1).and_then(Option::as_ref),
    };
    own.and_then(|plane| known(plane.context_bytes))
}

fn plane_facts(plane: Option<&device_executor::PlaneFacts>) -> Facts {
    plane.map_or_else(Facts::default, |plane| Facts {
        context: known(plane.context_bytes),
        process: known(plane.process_bytes),
        activation: known(plane.activation_peak_bytes).filter(|v| *v > 0),
        pinned: known(plane.pinned_bytes),
        pinned_budget: known(plane.pinned_budget_bytes),
        ..Facts::default()
    })
}

/// One `Facts` per GPU of a group from rank 0's and each follower's own plane facts
/// (`rank_planes`, rank 1 first): a GPU's own process and context; weights and activation
/// are rank 0's (its activation is the group's largest peak), as is a follower's whole set
/// before it stated its own. The host tier is the group's, so the first GPU's pinned bytes
/// and budget are every rank's.
fn rank_facts(
    facts: Facts,
    ranks: &[Option<device_executor::PlaneFacts>],
    degree: u32,
) -> Vec<Facts> {
    let host = Facts {
        pinned: None,
        pinned_budget: None,
        ..facts
    };
    let mut each = vec![facts];
    for rank in 1..degree.max(1) as usize {
        each.push(match ranks.get(rank - 1).and_then(Option::as_ref) {
            Some(own) => Facts {
                weights: facts.weights,
                weights_floor: facts.weights_floor,
                activation: facts.activation,
                pinned: None,
                pinned_budget: None,
                ..plane_facts(Some(own))
            },
            None => host,
        });
    }
    let stated = |of: fn(&device_executor::PlaneFacts) -> Option<i64>| -> u64 {
        ranks.iter().flatten().filter_map(|r| known(of(r))).sum()
    };
    each[0].pinned = facts.pinned.map(|own| own + stated(|r| r.pinned_bytes));
    each[0].pinned_budget = facts
        .pinned_budget
        .map(|own| own + stated(|r| r.pinned_budget_bytes));
    each
}

/// A load's facts: its weights as stages count them (decoded copies included where the
/// executor says), and the largest component's floor as its lowest rung.
fn load_facts(facts: Option<&device_executor::LoadFacts>) -> Facts {
    let Some(facts) = facts else {
        return Facts::default();
    };
    let mut out = plane_facts(facts.plane.as_ref());
    let layouts = facts.layouts.values();
    if facts.layouts.is_empty() {
        out.weights = facts.filled_bytes;
    } else {
        out.weights = Some(
            layouts
                .clone()
                .map(device_executor::Layout::planned_total)
                .sum(),
        );
        out.weights_floor = layouts.map(device_executor::Layout::planned_floor).max();
    }
    out
}

/// Ask the executor to release every revoked generation it reads (it is idle here).
/// `executor` lets go of every holding it maps (queued work first): its weights become its
/// own again. Holdings others still read stay; the rest are unread, for the ladder to reclaim.
fn detach(custody: &Mutex<ResidentCustody>, executor: &mut DeviceExecutor) -> io::Result<()> {
    let read = custody.lock().unwrap().read_by(&executor.birth);
    for (key, generation) in read {
        let reply = executor.command(
            &DeviceCommand::Revoke {
                layout: key.layout.clone(),
                generation,
                regions: vec![],
            },
            &mut device_executor::Baseline,
        )?;
        if reply.ok {
            custody
                .lock()
                .unwrap()
                .released(&key, generation, &executor.birth);
        }
        crate::memory::note(serde_json::json!({"event": "detach", "layout": key.layout,
            "ok": reply.ok}));
    }
    Ok(())
}

fn release_revoked(
    custody: &Mutex<ResidentCustody>,
    executor: &mut DeviceExecutor,
) -> io::Result<()> {
    let pending: Vec<(HoldingKey, u64)> = {
        let custody = custody.lock().unwrap();
        let revoking: std::collections::BTreeSet<_> = custody
            .holdings()
            .into_iter()
            .filter(|h| h.phase == crate::resident_custody::Phase::Revoking)
            .map(|h| (h.key, h.generation))
            .collect();
        custody
            .read_by(&executor.birth)
            .into_iter()
            .filter(|row| revoking.contains(row))
            .collect()
    };
    for (key, generation) in pending {
        let reply = executor.command(
            &DeviceCommand::Revoke {
                layout: key.layout.clone(),
                generation,
                regions: vec![],
            },
            &mut device_executor::Baseline,
        )?;
        if reply.ok {
            custody
                .lock()
                .unwrap()
                .released(&key, generation, &executor.birth);
        } else {
            // A reader that cannot release keeps its lease charged until it ends: no kill.
            eprintln!(
                "revoke of {} refused: {} {}",
                key.layout, reply.code, reply.detail
            );
        }
    }
    Ok(())
}

fn log_released(released: Vec<(HoldingKey, u64)>) {
    for (key, bytes) in released {
        eprintln!(
            "resident GPU weights released: {} on {}, {bytes} bytes",
            key.layout, key.device
        );
    }
}

/// Degree 2 stays off on a GPU that drives a display until qualified there. Unknown is "yes".
/// Whether any entrypoint of an installed interface binds a model: its executors are GPU ones.
fn binds_models(interface: &serde_json::Value) -> bool {
    interface
        .get("entrypoints")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|entries| {
            entries.iter().any(|entry| {
                entry
                    .get("models")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|models| !models.is_empty())
            })
        })
}

fn display_active(devices: &str) -> bool {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=display_active",
            "--format=csv,noheader",
            "-i",
            devices,
        ])
        .output();
    !matches!(output, Ok(o) if o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "Disabled")
}

/// Each held chunk is one fd: let the soft descriptor limit reach the hard one.
fn raise_fd_limit() {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: writable rlimit structure; a failure leaves the current limit.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0 && limit.rlim_cur < limit.rlim_max
        {
            limit.rlim_cur = limit.rlim_max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &limit);
        }
    }
}

/// An error ends its executor; the exact exit is observed before the run settles, and a
/// kill's measurement joins the reason.
fn ended_with(engine: &Engine, id: &str, error: io::Error, executor: DeviceExecutor) -> io::Error {
    let (pid, stderr_tail) = (executor.birth.pid, executor.stderr_tail());
    let error = ending(error, executor);
    keep_lifecycle_triage(engine, id, pid, &error.to_string(), &stderr_tail);
    error
}

fn ending(error: io::Error, executor: DeviceExecutor) -> io::Error {
    match executor.terminate() {
        Ok(device_executor::Ended {
            killed: Some(killed),
            ..
        }) => match refused(&error) {
            Some(refusal) => io::Error::other(Refused {
                code: refusal.code.clone(),
                detail: format!("{}; {killed}", refusal.detail),
            }),
            None if undelivered(&error) => io::Error::other(Undelivered(format!("{error}; {killed}"))),
            None => io::Error::other(format!("{error}; {killed}")),
        },
        Ok(_) => error,
        Err(unproven) => io::Error::other(format!("{error}; executor exit unproven: {unproven}")),
    }
}

/// Every error ends the run once its executor is gone: FAILED with the reason, or CANCELED
/// when a cancel was journaled. Only a never-authorized attempt hit by a transient OS
/// shortage returns to the queue; a deterministic pre-start failure is FAILED.
pub(crate) fn settle(engine: &Arc<Engine>, id: &str, error: &io::Error) -> io::Result<()> {
    let record = engine.get(id)?;
    if record.state.terminal() {
        return Ok(());
    }
    let ended = record
        .process
        .as_ref()
        .map(process_ended)
        .transpose()?
        .unwrap_or(true);
    if !ended {
        return Ok(()); // exit unproven: the reservation stays charged and nonterminal
    }
    if record.state == State::Starting && crate::process::transient(error) {
        return engine.defer_managed(id, format!("device startup unavailable: {error}"));
    }
    if let Some(ended) = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<device_executor::EndedBeforeStart>())
    {
        let pid = record.process.as_ref().map_or(0, |birth| birth.pid);
        keep_lifecycle_triage(engine, id, pid, &ended.to_string(), &ended.stderr_tail);
    }
    let outcome = if record.cancel_actor.is_some() {
        Outcome::Canceled
    } else if record.invocation.job && record.pause_actor.is_some() {
        // A pausing job's root, however it stopped, is replayed on resume.
        Outcome::Paused
    } else if let Some(refusal) = refused(error) {
        // The executor's own reason, typed: a group's first fault names its GPU.
        Outcome::Failed(
            Failure::executor("failed", "runtime", &refusal.code, &refusal.detail).encode(),
        )
    } else if record.state == State::Starting {
        Outcome::Failed(
            Failure::abandoned(&format!("device executor did not start: {error}")).encode(),
        )
    } else {
        Outcome::Failed(Failure::abandoned(&format!("device executor ended: {error}")).encode())
    };
    engine.finish(id, outcome).map(drop)
}

/// Executor roots keep each ended executor's logs for a day; every executor of an earlier
/// machine run has ended before this pool exists.
pub(crate) fn remove_old_executor_roots(root: &Path) {
    const KEEP: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let executor = name.len() == 32
            && name
                .to_string_lossy()
                .bytes()
                .all(|b| b.is_ascii_hexdigit());
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|modified| modified.elapsed().is_ok_and(|age| age > KEEP));
        if executor && old {
            if let Err(error) = fs::remove_dir_all(entry.path()) {
                eprintln!("old executor root {}: {error}", entry.path().display());
            }
        }
    }
}

/// A tree input's media type: its digest names a tree manifest (`entries` of files by path).
pub const TREE_MEDIA: &str = "application/vnd.cozy.tree-manifest";

/// A call's inputs in its spool: file inputs by field path as read-only copies, and input
/// trees by reference (`Tree` fields) as read-only directories with their manifest digest.
pub(crate) struct Staged {
    pub inputs: BTreeMap<String, serde_json::Value>,
    pub trees: BTreeMap<String, (PathBuf, String)>,
}

pub(crate) fn stage_inputs(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    spool: &Path,
    inputs: &[crate::journal::InputFile],
) -> io::Result<Staged> {
    let mut staged = Staged {
        inputs: BTreeMap::new(),
        trees: BTreeMap::new(),
    };
    if inputs.is_empty() {
        return Ok(staged);
    }
    let directory = spool.join("inputs");
    fs::create_dir(&directory)?;
    for (position, input) in inputs.iter().enumerate() {
        let field: String = input
            .input_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .take(96)
            .collect();
        let local = directory.join(format!("{position:03}-{field}"));
        if input.media_type == TREE_MEDIA {
            let manifest = held_bytes(store, &input.digest, input.length)?;
            materialize_tree(store, identity, &manifest, &local)?;
            // A payload names a tree by its field's ref or by its manifest digest.
            staged.trees.insert(input.digest.clone(), (local.clone(), input.digest.clone()));
            staged.trees.insert(input.input_id.clone(), (local, input.digest.clone()));
            continue;
        }
        copy_held(store, identity, &input.digest, input.length, &local)?;
        staged.inputs.insert(
            input.input_id.clone(),
            serde_json::json!({"local":local,"media_type":input.media_type,"digest":input.digest,"length":input.length,"order":input.order,"file_state":null}),
        );
    }
    if let Some(identity) = identity {
        identity.readable(&directory)?;
    }
    Ok(staged)
}

/// One held object copied read-only to `local`, exactly `length` bytes.
fn copy_held(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    digest: &str,
    length: u64,
    local: &Path,
) -> io::Result<()> {
    let sha = digest.strip_prefix("sha256:").unwrap_or(digest);
    let mut source = store.open_verified(sha).map_err(io::Error::other)?.into_file();
    let mut copy = std::os::unix::fs::OpenOptionsExt::mode(
        fs::OpenOptions::new().write(true).create_new(true),
        0o444,
    )
    .open(local)?;
    if std::io::copy(&mut source, &mut copy)? != length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "an input's held bytes changed length",
        ));
    }
    copy.sync_all()?;
    if let Some(identity) = identity {
        identity.readable(local)?;
    }
    Ok(())
}

fn held_bytes(store: &Store, digest: &str, length: u64) -> io::Result<Vec<u8>> {
    let sha = digest.strip_prefix("sha256:").unwrap_or(digest);
    let mut bytes = Vec::new();
    store
        .open_verified(sha)
        .map_err(io::Error::other)?
        .into_file()
        .take(length + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != length {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "a tree manifest changed length"));
    }
    Ok(bytes)
}

/// A tree manifest's files under `root`: each a relative path below it, each a held object.
fn materialize_tree(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    manifest: &[u8],
    root: &Path,
) -> io::Result<()> {
    let invalid = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("tree manifest: {why}"));
    let document: serde_json::Value = serde_json::from_slice(manifest).map_err(|_| invalid("not JSON"))?;
    let entries = document["entries"].as_array().ok_or_else(|| invalid("no entries"))?;
    fs::create_dir(root)?;
    for entry in entries {
        let (Some("file"), Some(path), Some(sha), Some(length)) = (
            entry["kind"].as_str(),
            entry["path"].as_str(),
            entry["blob"]["sha256"].as_str(),
            entry["blob"]["length"].as_u64(),
        ) else {
            return Err(invalid("an entry is not a file with a blob"));
        };
        let relative = Path::new(path);
        let local = relative.components().all(|c| matches!(c, std::path::Component::Normal(_)));
        if !local || path.contains(['\\', '\0']) {
            return Err(invalid("a path leaves its tree"));
        }
        let target = root.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        copy_held(store, identity, &format!("sha256:{sha}"), length, &target)?;
    }
    if let Some(identity) = identity {
        for entry in walk_dirs(root)? {
            identity.readable(&entry)?;
        }
    }
    Ok(())
}

pub(crate) fn walk_dirs(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found = vec![root.to_path_buf()];
    let mut index = 0;
    while index < found.len() {
        for entry in fs::read_dir(&found[index])? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                found.push(entry.path());
            }
        }
        index += 1;
    }
    Ok(found)
}

/// The failed attempt's triage bundle, kept before the run settles: the executor's own terminal
/// and traceback, and its stderr tail.
/// A lifecycle failure's bundle, when the executor wrote no terminal of its own: the reason
/// (with any kill measurement) and the end of its stderr. A canceled run keeps none.
fn keep_lifecycle_triage(engine: &Engine, id: &str, pid: u32, reason: &str, stderr_tail: &str) {
    let Ok(record) = engine.get(id) else {
        return;
    };
    if record.cancel_actor.is_some() || engine.triage(id).ok().flatten().is_some() {
        return;
    }
    let request = record
        .submission
        .as_ref()
        .map_or_else(|| id.to_string(), |s| s.request_id.clone());
    engine.record_triage(
        id,
        &crate::triage::Facts {
            request_id: &request,
            attempt: record.attempt,
            terminal: "failed",
            origin: "machine",
            code: "executor_ended",
            message: reason,
            traceback: "",
            executor_pid: pid,
            stderr_tail,
        },
    );
}

pub(crate) fn keep_triage(
    engine: &Engine,
    id: &str,
    executor: &DeviceExecutor,
    outcome: &device_executor::Outcome,
) {
    let record = engine.get(id).ok();
    let request = record
        .as_ref()
        .and_then(|record| record.submission.as_ref())
        .map_or_else(|| id.to_string(), |s| s.request_id.clone());
    engine.record_triage(
        id,
        &crate::triage::Facts {
            request_id: &request,
            attempt: record.map_or(0, |record| record.attempt),
            terminal: &outcome.terminal,
            origin: &outcome.origin,
            code: &outcome.code,
            message: &outcome.message,
            traceback: &outcome.traceback,
            executor_pid: executor.birth.pid,
            stderr_tail: &executor.stderr_tail(),
        },
    );
}

pub(crate) fn output_bindings(
    bindings: Vec<device_executor::AssetBinding>,
) -> io::Result<Vec<AssetBinding>> {
    bindings
        .into_iter()
        .map(|binding| {
            let (algorithm, value) =
                if let Some(value) = binding.producer_digest.strip_prefix("blake2b:") {
                    ("blake2b-128", value)
                } else if let Some(value) = binding.producer_digest.strip_prefix("sha256:") {
                    ("sha256", value)
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "SDK output checksum operation unavailable",
                    ));
                };
            Ok(AssetBinding {
                relative_path: binding.name,
                asset_ref: binding.asset_ref,
                media_type: binding.media_type,
                checksum: OutputChecksum {
                    algorithm: algorithm.into(),
                    value: value.into(),
                },
                length: binding.length,
            })
        })
        .collect()
}

/// An executor's refusal of a lifecycle command (start, load, activate), in its own words:
/// a group's names the GPU whose fault came first (Runtime `RankGroup` latches it).
#[derive(Debug)]
struct Refused {
    code: String,
    detail: String,
}
impl std::fmt::Display for Refused {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for Refused {}
/// The request never reached a handler; see `call`.
#[derive(Debug)]
struct Undelivered(String);
impl std::fmt::Display for Undelivered {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for Undelivered {}
fn undelivered(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|inner| inner.is::<Undelivered>())
}

fn refused(error: &io::Error) -> Option<&Refused> {
    error.get_ref()?.downcast_ref::<Refused>()
}

/// What the call's executor measured, kept as the run's (`run show`): its stage and step
/// tracks, its ranks' execution records, and its attention observations (the kernels that
/// served, Sol's dense and sparse calls).
pub(crate) fn record_measurements(engine: &Engine, id: &str, reply: &Frame) {
    let attention: Vec<&serde_json::Value> = reply
        .observations
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| {
            row["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("attention."))
        })
        .collect();
    let measurements = serde_json::json!({"attribution": reply.attribution,
        "execution": reply.execution, "observations": attention});
    let recorded = serde_json::to_vec(&measurements)
        .map_err(io::Error::other)
        .and_then(|bytes| engine.with_journal(|journal| journal.record_measurements(id, &bytes)));
    if let Err(error) = recorded {
        eprintln!("run {id}: measurements not kept: {error}");
    }
}


pub(crate) fn command_ok(frame: Frame) -> io::Result<Frame> {
    if frame.ok {
        Ok(frame)
    } else {
        Err(io::Error::other(Refused {
            code: frame.code,
            detail: frame.detail,
        }))
    }
}
struct Callbacks<'a> {
    engine: &'a Arc<Engine>,
    id: &'a str,
    cells: &'a mut BTreeMap<u32, File>,
    completed: u64,
    host: &'a Arc<HostTier>,
    sources: &'a Arc<ModelSources>,
    peer: u64,
    grants: &'a [HostGrant],
    birth: ProcessBirth,
    custody: Option<&'a Mutex<ResidentCustody>>,
    /// The executor's pidfd: a reader lease ends when it does.
    exit: File,
    pool: &'a GpuPool,
    plan: &'a str,
    degree: u32,
    others: &'a mut BTreeMap<String, Session>,
    store: &'a Store,
    /// The invoking request's spool, where its published assets' bytes are.
    spool: Option<PathBuf>,
}
impl Services for Callbacks<'_> {
    fn object_files(&mut self, frame: &Frame, plan: Option<File>) -> io::Result<(Answer, Vec<File>)> {
        let mut answer = Answer::unavailable(frame.seq);
        let Some(plan) = plan else {
            return Ok((answer, Vec::new()));
        };
        let request = SealedRequest {
            sha256: &frame.sha256,
            length: frame.length,
            stage: None,
        };
        match self.host.object_files(self.peer, self.grants, request, plan) {
            Ok(files) => {
                answer.ok = true;
                answer.code.clear();
                answer.detail.clear();
                answer.objects_sha256 = crate::host_tier::objects_digest(&files);
                Ok((answer, files.into_iter().map(|(_, file)| file).collect()))
            }
            Err(error) => {
                answer.code = "object_files_refused".into();
                answer.detail = error.to_string();
                Ok((answer, Vec::new()))
            }
        }
    }
    fn device_tier(
        &mut self,
        frame: &Frame,
        fds: Vec<OwnedFd>,
    ) -> io::Result<(Answer, Vec<OwnedFd>)> {
        let mut answer = Answer::unavailable(frame.seq);
        let Some(custody) = self.custody else {
            return Ok((answer, Vec::new()));
        };
        let mut custody = custody.lock().unwrap();
        // An ask for "*" is for the layout's latest baked variant: the executor learns which
        // from the answer, and reads its bytes only if its own bake would be the same.
        let variant = match frame.variant.as_str() {
            "*" if !frame.offer => custody
                .latest_variant(&frame.device, &frame.layout)
                .unwrap_or_else(|| "*".into()),
            named => named.to_string(),
        };
        let key = HoldingKey {
            device: frame.device.clone(),
            layout: frame.layout.clone(),
            variant,
        };
        // This plan names this weight set: its next executor's Degree 2 fit counts it once
        // when another tenant holds and reads it.
        if key.variant != "*" {
            self.pool
                .first()
                .learn_holding(self.plan, &holding_id(&key, 0));
        }
        // The reader's lease is a connection: the executor keeps one end while it maps the
        // holding, and its close (release or death) ends the lease.
        let (reader, lease) = Reader::lease(self.birth.clone(), self.exit.try_clone()?)?;
        answer.ok = true;
        answer.code.clear();
        answer.detail.clear();
        answer.layout = key.layout.clone();
        answer.variants = true;
        if frame.offer {
            match custody.offer(key, &frame.name, frame.regions.clone(), fds, reader) {
                Ok(Offered::Kept { generation }) => {
                    answer.generation = generation;
                    answer.lease = true;
                    return Ok((answer, vec![lease]));
                }
                Ok(Offered::Duplicate) => answer.duplicate = true,
                Err(error) => {
                    answer.ok = false;
                    answer.code = "device_tier_refused".into();
                    answer.detail = error.to_string();
                }
            }
            return Ok((answer, Vec::new()));
        }
        drop(fds);
        Ok(match custody.attach(&key, reader)? {
            Some(mut attached) => {
                answer.held = true;
                answer.lease = true;
                answer.variant = key.variant.clone();
                answer.generation = attached.generation;
                answer.regions = attached.regions;
                attached.fds.push(lease);
                (answer, attached.fds)
            }
            None => (answer, Vec::new()),
        })
    }
    fn progress(&mut self, frame: &Frame) {
        if !(frame.request_id.is_empty() || frame.request_id == self.id) || frame.stage.is_empty() {
            return;
        }
        // Zero-advance frames are telemetry (stage/position), not completed work.
        self.completed = self.completed.saturating_add(frame.advance);
        // The Python worker's progress payload: stage and step_ms always, the rest when known.
        let mut payload = serde_json::json!({"stage": frame.stage.chars().take(120).collect::<String>(), "step_ms": frame.step_ms.unwrap_or(0.0)});
        for (name, value) in [
            (
                "stage_fraction",
                frame.stage_fraction.map(serde_json::Value::from),
            ),
            (
                "overall_fraction",
                frame.overall_fraction.map(serde_json::Value::from),
            ),
            ("position", frame.position.map(serde_json::Value::from)),
            ("total", frame.total.map(serde_json::Value::from)),
        ] {
            if let Some(value) = value {
                payload[name] = value;
            }
        }
        let _ = self
            .engine
            .observe_progress(self.id, self.completed, payload.to_string());
    }
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        if frame.kind == Kind::SealedPrefetch {
            let mut answer = Answer::unavailable(frame.seq);
            let plans =
                descriptor.ok_or_else(|| io::Error::other("sealed prefetch omitted its plans"))?;
            let request = SealedRequest {
                sha256: &frame.sha256,
                length: frame.length,
                stage: None,
            };
            match self.host.prefetch(self.peer, self.grants, request, plans) {
                Ok(()) => {
                    answer.ok = true;
                    answer.code.clear();
                    answer.detail.clear();
                }
                Err(error) => answer.detail = error.to_string(),
            }
            return Ok((answer, None));
        }
        if frame.kind == Kind::SealedTier {
            let mut answer = Answer::unavailable(frame.seq);
            let plan = descriptor
                .ok_or_else(|| io::Error::other("sealed tier request omitted its plan"))?;
            let request = SealedRequest {
                sha256: &frame.sha256,
                length: frame.length,
                stage: frame.stage_regions.as_deref(),
            };
            // A refusal belongs to this weight set; the executor determines whether another
            // supported source remains. It does not grant permission to open the Store.
            match self.host.seal(self.peer, self.grants, request, plan) {
                Ok(granted) => {
                    (answer.ok, answer.held) = (true, granted.is_some());
                    answer.code.clear();
                    answer.detail.clear();
                    return Ok((answer, granted));
                }
                Err(error) => {
                    answer.code = "sealed_tier_refused".into();
                    answer.detail = error.to_string();
                    return Ok((answer, None));
                }
            }
        }
        if frame.kind == Kind::SealedStage {
            let mut answer = Answer::unavailable(frame.seq);
            let plan =
                descriptor.ok_or_else(|| io::Error::other("sealed stage omitted its plan"))?;
            let request = SealedRequest {
                sha256: &frame.sha256,
                length: frame.length,
                stage: frame.stage_regions.as_deref(),
            };
            match self.host.stage(self.peer, self.grants, request, plan) {
                Ok(staged) => {
                    answer.ok = true;
                    answer.code.clear();
                    answer.detail.clear();
                    answer.staged = staged;
                }
                Err(error) => answer.detail = error.to_string(),
            }
            return Ok((answer, None));
        }
        if frame.kind == Kind::ModelSource {
            drop(descriptor);
            // A refusal is an answer: that executor fails its load with the reason.
            return Ok(match self.sources.serve(frame) {
                Ok((answer, file)) => (answer, Some(file)),
                Err(error) => {
                    let mut answer = Answer::unavailable(frame.seq);
                    answer.code = "model_source_refused".into();
                    answer.detail = error.to_string();
                    (answer, None)
                }
            });
        }
        if frame.kind == Kind::Publish {
            drop(descriptor);
            let answer = match &self.spool {
                Some(spool) => {
                    crate::products::publish(self.store, self.engine, self.id, spool, frame)
                }
                None => Answer::unavailable(frame.seq),
            };
            return Ok((answer, None));
        }
        let mut answer = Answer::unavailable(frame.seq);
        match frame.kind {
            Kind::BudgetCell => {
                let file =
                    descriptor.ok_or_else(|| io::Error::other("budget cell descriptor absent"))?;
                if file.metadata()?.len() != 32 {
                    return Err(io::Error::other("budget cell layout differs"));
                }
                self.cells.insert(frame.rank, file);
                answer.ok = true;
            }
            // Calls never ask for turns (Invoke sends `stages: false`); keep the budget.
            Kind::StageEnter | Kind::StageExit => {
                drop(descriptor);
                answer.ok = true;
            }
            Kind::DeviceRoom => {
                drop(descriptor);
                if let Some(held) = known(frame.holding_bytes).filter(|_| self.degree == 1) {
                    // Without it the ledger keeps its load-time report, and the bytes it holds
                    // since read as external: a holding dropped for it raised no cap (run 4557).
                    let facts = crate::memory::policy::Facts {
                        process: Some(held),
                        ..Default::default()
                    };
                    self.pool.first().observe(self.plan, facts, None);
                }
                let cap = self.pool.room_for(
                    self.plan,
                    self.degree,
                    frame.free_bytes,
                    self.cells,
                    self.others,
                )?;
                answer.ok = true;
                answer.cap_bytes = cap.map_or(-1, |cap| i64::try_from(cap).unwrap_or(i64::MAX));
            }
            _ => drop(descriptor),
        }
        if answer.ok {
            answer.code.clear();
            answer.detail.clear();
        }
        Ok((answer, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_rank_of_a_group_gets_its_own_gpus_cap() {
        let (cap, group) = rank_grant(&[Some(30), Some(20)]);
        assert_eq!((cap, group.rank_caps), (Some(30), vec![30, 20]));
        // A GPU that cannot say lifts only its own rank's cap.
        let (cap, group) = rank_grant(&[Some(30), None]);
        assert_eq!((cap, group.rank_caps), (Some(30), vec![30, -1]));
        // A sibling's reading cannot manufacture an unmeasured leader's authority.
        let (cap, group) = rank_grant(&[None, Some(20)]);
        assert_eq!((cap, group.rank_caps), (None, vec![-1, 20]));
        let (cap, group) = rank_grant(&[Some(30)]);
        assert_eq!((cap, group.is_empty()), (Some(30), true));
        assert_eq!(rank_grant(&[]).0, None);
    }

    #[test]
    fn a_gpu_learns_only_the_context_its_own_rank_measured() {
        let reply: Frame = serde_json::from_value(json!({
            "plane": {"context_bytes": 100},
            "rank_planes": [{"context_bytes": 200}, null]
        }))
        .unwrap();
        let each: Vec<_> = (0..4).map(|rank| rank_context(&reply, rank)).collect();
        assert_eq!(each, vec![Some(100), Some(200), None, None]);
    }

    #[test]
    fn each_gpu_of_a_group_is_charged_its_own_ranks_process() {
        let rank0 = Facts {
            process: Some(9),
            context: Some(1),
            weights: Some(6),
            activation: Some(3),
            pinned: Some(4),
            pinned_budget: Some(5),
            ..Facts::default()
        };
        let follower = device_executor::PlaneFacts {
            process_bytes: Some(7),
            context_bytes: Some(2),
            pinned_bytes: Some(4),
            pinned_budget_bytes: Some(5),
            ..Default::default()
        };
        let each = rank_facts(rank0, &[Some(follower), None], 3);
        assert_eq!(each.len(), 3);
        // The host tier is the group's: the first GPU holds every stated rank's pinned bytes.
        assert_eq!(
            (each[0].process, each[0].pinned, each[0].pinned_budget),
            (Some(9), Some(8), Some(10))
        );
        assert_eq!(
            (
                each[1].process,
                each[1].context,
                each[1].weights,
                each[1].activation,
                each[1].pinned
            ),
            (Some(7), Some(2), Some(6), Some(3), None)
        );
        // A follower that has not stated its own yet is charged rank 0's.
        assert_eq!((each[2].process, each[2].pinned), (Some(9), None));
        assert_eq!(rank_facts(rank0, &[], 1), vec![rank0]);
    }

    #[test]
    fn a_group_is_the_widest_degree_every_slot_declares_on_this_machine() {
        let h3 = [
            json!({"path":"e.models.base_model","sequence_parallel":{"degrees":[2,4,7,8]}}),
            json!({"path":"e.models.turbo_lora","sequence_parallel":{"degrees":[2,4,7,8]}}),
        ];
        assert_eq!(group_degree(&h3, 0, 1).unwrap(), 1);
        assert_eq!(group_degree(&h3, 0, 2).unwrap(), 2);
        assert_eq!(group_degree(&h3, 0, 6).unwrap(), 4);
        assert_eq!(group_degree(&h3, 1, 4).unwrap(), 1);
        assert_eq!(group_degree(&h3, 2, 4).unwrap(), 2);
        let refused = group_degree(&h3, 3, 4).unwrap_err().to_string();
        assert!(refused.starts_with("gpu_count_unavailable"), "{refused}");
        assert!(group_degree(&h3, 4, 2).is_err());
        // A slot without a declaration runs on one GPU only, so the group is one.
        let mixed = [h3[0].clone(), json!({"path":"e.models.vae"})];
        assert_eq!(group_degree(&mixed, 0, 4).unwrap(), 1);
    }
}
