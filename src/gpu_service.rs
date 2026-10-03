//! Trusted device execution on one GPU or a group of K (one executor sealed to all K; rank 0
//! forms the followers); acceptance, scheduling and custody stay in Engine.
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
    io::{self, Write},
    os::fd::{AsRawFd, OwnedFd},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex, Weak,
    },
    time::Instant,
};
use tensorfs_core::store::Store;

fn alloc_conf() -> String {
    crate::launch_identity::DEFAULT_ALLOC_CONF.into()
}
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
    /// `PYTORCH_CUDA_ALLOC_CONF` and `OMP_NUM_THREADS` the seal imposes (worker defaults).
    #[serde(default = "alloc_conf")]
    pub alloc_conf: String,
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
    /// At machine start, load each installed GPU generation's most recently used
    /// construction where it fits beside the tenants already there (it never makes room).
    /// They stay as idle tenants the memory policy evicts least recently used first.
    #[serde(default = "yes")]
    pub prewarm: bool,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuPlan {
    pub actor: String,
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

struct Session {
    plan: String,
    /// GPUs of its group; its followers (rank 1 first) once it started, held by pidfd so a
    /// call that fails can name the GPU whose process ended first.
    degree: u32,
    followers: Vec<crate::process::Exact>,
    loaded: bool,
    executor: DeviceExecutor,
    /// Budget cells by rank: rank 0's own, and each follower's (`rank_cells/1`).
    budget_cells: BTreeMap<u32, File>,
    actor: String,
    /// This executor in the host tier and the layouts it may adopt.
    peer: u64,
    grants: Vec<HostGrant>,
    /// Its weights stay on the GPU under custody (Degree 2), decided once at its load.
    sharing: bool,
    launch: Launch,
    /// It has served a call: its next one is not its first.
    invoked: bool,
    /// Its manifests stay out of every download's GC while it lives.
    _serving: ServingHold,
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
fn parent_key(generation: &str) -> String {
    format!("parent:{generation}")
}

/// The manifests live executors read. A download's GC must not evict them: an evicted file a
/// session still reads frees no disk and breaks the session's next load.
#[derive(Default)]
struct Serving {
    next: std::sync::atomic::AtomicU64,
    held: Mutex<BTreeMap<u64, Vec<String>>>,
}
/// One session's entry in `Serving`, removed when the session ends.
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
    reserved: AtomicBool,
    /// One retained executor per plan; the memory policy decides which keep weights mapped.
    sessions: Mutex<BTreeMap<String, Session>>,
    /// Per generation, the import-only executor sessions fork from. Declared after
    /// `sessions`: executors end before the parent they were forked from.
    zygotes: Mutex<BTreeMap<String, Arc<Zygote>>>,
    /// Each envelope GPU with its own memory decisions, in envelope order.
    devices: Vec<Device>,
    host: Arc<HostTier>,
    /// The memory policy's host ledger; the tier's limit reads it.
    host_ledger: Arc<crate::memory::host::HostLedger>,
    /// Degree 2: GPU weights kept across executors. None on a GPU that drives a display.
    custody: Option<Mutex<ResidentCustody>>,
    // Drop session/resource custody before ending the actual spawning thread.
    launcher: crate::child_launcher::ChildLauncher,
}
struct Device {
    entry: String,
    memory: GpuMemory,
}

struct Permit {
    pool: Arc<GpuPool>,
    engine: Weak<Engine>,
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.pool.reserved.store(false, Ordering::Release);
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
        Ok(Arc::new(Self {
            launcher: crate::child_launcher::ChildLauncher::new()?,
            root: root.to_path_buf(),
            serving: Arc::default(),
            devices,
            incarnation,
            config,
            store,
            reserved: AtomicBool::new(false),
            sessions: Mutex::new(BTreeMap::new()),
            zygotes: Mutex::new(BTreeMap::new()),
            host,
            host_ledger,
            custody,
        }))
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
    /// The kernel store and the namespaces live executors use. The pool's executors share
    /// one identity; a call in progress (the sessions lock is held) counts as live.
    pub fn kernel_caches(&self) -> crate::reclaim::KernelCaches {
        let uid = self
            .config
            .identity
            .map_or_else(|| unsafe { libc::geteuid() }, |identity| identity.uid);
        let live = match (self.sessions.try_lock(), self.zygotes.try_lock()) {
            (Ok(sessions), Ok(zygotes)) => !sessions.is_empty() || !zygotes.is_empty(),
            _ => true,
        };
        crate::reclaim::KernelCaches {
            root: self.root.join("kernels"),
            busy: if live {
                std::collections::HashSet::from([format!("u{uid}")])
            } else {
                std::collections::HashSet::new()
            },
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
        each[0]
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
        actor: &str,
        installed: &crate::journal::Installation,
        entrypoint: &str,
        choices: &[crate::api::pb::ModelChoice],
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
            if !choice.source.is_empty()
                || !choice.profiles.is_empty()
                || (!choice.adapters.is_empty() && resolved.is_empty())
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "provider sources (and adapters without Hub resolution) are not taken by this machine yet",
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
        let semantic = serde_json::json!({"actor":actor,"generation":installed.generation,"entrypoint":entrypoint,"slots":planned,"degree":degree});
        let canonical = serde_json_canonicalizer::to_vec(&semantic).map_err(io::Error::other)?;
        Ok(GpuPlan {
            actor: actor.into(),
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
        if plan.actor != preparation.actor
            || plan.id != preparation.id
            || plan.installation != preparation.installation
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GPU preparation differs from owned journal binding",
            ));
        }
        Ok(plan)
    }
    pub fn dispatch(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        record: &Execution,
        held: HeldGeneration,
        plan: GpuPlan,
    ) -> io::Result<bool> {
        if self
            .reserved
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(false);
        }
        let permit = Permit {
            pool: self.clone(),
            engine: Arc::downgrade(engine),
        };
        engine.dispatch_managed(&record.id, move |engine, id| {
            let _permit = permit;
            let result = _permit.pool.run(&engine, &id, held, plan);
            if let Err(error) = &result {
                settle(&engine, &id, error)?;
            }
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

    /// Load the given constructions in the background, each only where it fits beside the
    /// tenants already there (a prewarm never makes room), most recent first. Each waits for
    /// the GPU's call slot, so a request already running finishes first.
    /// `fence`: a previous run's executors; new GPU work waits until each has exited.
    pub fn prewarm(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        plans: Vec<(HeldGeneration, GpuPlan)>,
        fence: Vec<ProcessBirth>,
    ) {
        if !self.config.prewarm || plans.is_empty() {
            return;
        }
        let (pool, engine) = (self.clone(), engine.clone());
        let started = std::thread::Builder::new()
            .name("executor-prewarm".into())
            .spawn(move || {
                while !fence
                    .iter()
                    .all(|birth| process_ended(birth).unwrap_or(true))
                {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                for (held, plan) in plans {
                    let asked = Instant::now();
                    // Its parent imports first, off the call slot; a request in flight or
                    // queued always goes ahead of a prewarm.
                    if let Some((zygote, start)) = pool.zygote(&held) {
                        if start {
                            zygote.set(pool.import_only(&held));
                        }
                        zygote.wait_started();
                    }
                    while !engine.nonterminal(1).is_ok_and(|work| work.is_empty())
                        || pool
                            .reserved
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                            .is_err()
                    {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    let _permit = Permit {
                        pool: pool.clone(),
                        engine: Arc::downgrade(&engine),
                    };
                    let key = plan.id.clone();
                    let waited = asked.elapsed();
                    let mut sessions = pool.sessions.lock().unwrap();
                    let result = pool.prewarm_locked(&engine, &held, plan, &mut sessions);
                    for device in &pool.devices {
                        device.memory.finished(&key);
                    }
                    if !sessions.contains_key(&key) {
                        pool.ended(&key);
                    }
                    pool.note_prewarm(&key, waited, asked.elapsed() - waited, &result);
                }
            });
        if let Err(error) = started {
            eprintln!("executor prewarm: {error}");
        }
    }

    /// A running parent's hint that it will call `plan` next (`model_prefetch`, H3 long-form's
    /// `prefetch(motion_segment)`): load it in the background where it fits beside the tenants
    /// already there, as a prewarm does, but without waiting for the parent's own run to end.
    /// It takes the GPU's call slot like any call, so a child call in flight finishes first.
    pub fn prefetch(self: &Arc<Self>, engine: &Arc<Engine>, held: HeldGeneration, plan: GpuPlan) {
        if !self.config.prewarm {
            return;
        }
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
                while pool
                    .reserved
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                let _permit = Permit {
                    pool: pool.clone(),
                    engine: Arc::downgrade(&engine),
                };
                let key = plan.id.clone();
                let waited = asked.elapsed();
                let mut sessions = pool.sessions.lock().unwrap();
                let result = pool.prewarm_locked(&engine, &held, plan, &mut sessions);
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
    ) -> io::Result<&'static str> {
        if sessions.contains_key(&plan.id) {
            return Ok("already loaded");
        }
        let lane = self.lane(plan.degree)?;
        let fits = lane.iter().enumerate().all(|(index, device)| {
            device.memory.admits(
                &plan.id,
                || {
                    if index == 0 {
                        self.holdings()
                    } else {
                        vec![]
                    }
                },
            )
        });
        if !fits {
            return Ok("no room beside the other tenants");
        }
        self.host_room(&plan.id, sessions);
        let load_caps = self.decide(&plan.id, plan.degree, true, sessions)?;
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
        Ok(())
    }

    /// Start a generation's import-only executor in the background (machine start, after
    /// an install), so its imports overlap everything before the first request.
    pub fn prespawn(self: &Arc<Self>, held: HeldGeneration) {
        if !self.parent_fits(&held.record.identity) {
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
    fn parent_fits(&self, generation: &str) -> bool {
        let need = self.first().with(|gpu| {
            let learned = |key: &str| gpu.learned.plans.get(key).map(|plan| plan.host_bytes);
            learned(&parent_key(generation))
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
        let host = crate::host_memory::read();
        host.available < 0 || host.available as u64 >= need
    }

    /// Drop a dead parent, if it is still the generation's: the next launch starts another.
    fn forget_parent(&self, generation: &str, dead: &Arc<Zygote>) {
        let mut zygotes = self.zygotes.lock().unwrap();
        if zygotes
            .get(generation)
            .is_some_and(|current| Arc::ptr_eq(current, dead))
        {
            zygotes.remove(generation);
        }
    }

    /// End the least recently used parent with no live child, for host room: the bytes it
    /// held privately (at least 1 when one ended unmeasured), 0 when there is none to end.
    fn end_idle_parent(&self) -> u64 {
        let victim = {
            let mut zygotes = self.zygotes.lock().unwrap();
            let chosen = zygotes
                .iter()
                .filter(|(_, zygote)| zygote.childless())
                .min_by_key(|(_, zygote)| *zygote.used.lock().unwrap())
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
        if let Some(zygote) = zygotes.get(&held.record.identity) {
            return Some((zygote.clone(), false));
        }
        let zygote = Arc::new(Zygote::default());
        zygotes.insert(held.record.identity.clone(), zygote.clone());
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
                    &parent_key(&held.record.identity),
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
        seal.alloc_conf = self.config.alloc_conf.clone();
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
            self.host_room(&plan.id, sessions);
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
            engine.finish(id, Outcome::Canceled)?;
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
    /// spawn) and open its model sources and host-tier grants. Unsupported: an executor that
    /// cannot take its weights from the sealed host tier.
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
                        self.forget_parent(&held.record.identity, &zygote);
                    }
                    break;
                }
                Forked::Lost(returned, reason) => {
                    eprintln!("import-only executor lost ({reason}); starting another");
                    config = Some(*returned);
                    self.forget_parent(&held.record.identity, &zygote);
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
        // Weights come only from the machine's sealed host tier: handed out while it fills, and
        // streamed when it does not fit. Header and configs come from the store. (An executor
        // whose TensorFS predates #313/#314 refuses such a layout and reads the store itself.)
        for needed in ["weight_plane/1", "host_tiers.sealed/1"] {
            if !executor.hello.offers(needed) {
                executor.shutdown()?;
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("executor lacks {needed}: its weights would come from the store"),
                ));
            }
        }
        let selections = plan.selections();
        let id = self.serving.next.fetch_add(1, Ordering::Relaxed);
        self.serving.held.lock().unwrap().insert(
            id,
            selections.iter().map(|s| s.manifest.clone()).collect(),
        );
        let serving = ServingHold {
            serving: self.serving.clone(),
            id,
        };
        let sources = ModelSources::open_shared(self.store.clone(), &selections)?;
        let peer = self.host.register_peer(executor.observer_pidfd()?);
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
        // Start on this model's layouts while the executor imports and constructs.
        self.host.prepare(grants.clone());
        Ok(Session {
            plan: plan.id.clone(),
            degree: plan.degree,
            followers: vec![],
            loaded: false,
            executor,
            budget_cells: BTreeMap::new(),
            actor: plan.actor.clone(),
            peer,
            grants,
            sharing: false,
            launch,
            invoked: false,
            _serving: serving,
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
            peer: session.peer,
            grants: &session.grants,
            actor: &session.actor,
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
                    sealed_tiers: true,
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
                inputs,
                floor_bytes: self.first().floor(),
                activation_bytes: self.first().with(|gpu| gpu.seeds(&plan.id)),
            },
            &group,
            &mut callbacks,
        )?;
        self.learn(
            &plan.id,
            plan.degree,
            &shape,
            &reply,
            session.executor.birth.pid,
        );
        self.record_invoke(&plan.id, first, invoked.elapsed(), &reply);
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
    fn host_room(&self, plan: &str, sessions: &mut BTreeMap<String, Session>) {
        let need = self.first().with(|gpu| {
            gpu.learned
                .plans
                .get(plan)
                .map_or(0, |learned| learned.host_bytes)
        });
        if need == 0 {
            return;
        }
        loop {
            let host = crate::host_memory::read();
            let Ok(available) = u64::try_from(host.available) else {
                return;
            };
            if available >= need {
                return;
            }
            // Unheld sealed layouts, then a parent with no live child (cheaper to recreate
            // than an executor with its weights), then idle executors.
            let released = self.host.release(need - available);
            let ended = released == 0
                && (self.end_idle_parent() > 0
                    || match self.first().with(|gpu| gpu.lru_idle(plan)) {
                        Some(victim) => self
                            .carry_out(&Step::End(victim), sessions)
                            .unwrap_or(false),
                        None => false,
                    });
            crate::memory::note(serde_json::json!({"event": "host_room", "plan": plan,
                "need": need, "available": available, "released": released, "ended": ended}));
            if released == 0 && !ended {
                return;
            }
        }
    }

    /// What a call measured, for later executors and runs: its shape's activation growth
    /// (the call's peak and each stage method's), its context, its private host bytes.
    fn learn(&self, plan: &str, degree: u32, shape: &str, reply: &Frame, pid: u32) {
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
        // Every rank of a group runs the same shape on its own GPU.
        for device in self.lane(degree).unwrap_or_default() {
            device
                .memory
                .learn_call(plan, shape, peak, &methods, known(plane.context_bytes));
        }
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
        others: &mut BTreeMap<String, Session>,
    ) -> io::Result<Option<u64>> {
        let mut cap: Option<u64> = None;
        for (index, device) in self.lane(degree)?.iter().enumerate() {
            let got = device.memory.make_room(
                plan,
                free_bytes,
                || if index == 0 { self.holdings() } else { vec![] },
                |step| self.carry_out(step, others),
            )?;
            cap = match (cap, got) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        Ok(cap)
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
    format!("{}/{}/{}#{generation}", key.actor, key.device, key.layout)
}

fn known(value: Option<i64>) -> Option<u64> {
    value.and_then(|v| u64::try_from(v).ok())
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
    let (pid, stderr_tail) = (
        executor.birth.pid,
        crate::process::tail(&executor.root_path().join("stderr.log")),
    );
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

/// Read-only copies of the run's file inputs in its spool (`inputs/NNN-<field>`), from
/// the bytes the caller's import holds in the store; the executor binds them by field.
pub(crate) fn stage_inputs(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    spool: &Path,
    inputs: &[crate::journal::InputFile],
) -> io::Result<BTreeMap<String, serde_json::Value>> {
    stage_inputs_with(identity, spool, inputs, |input| {
        let sha = input
            .digest
            .strip_prefix("sha256:")
            .unwrap_or(&input.digest);
        Ok(store
            .open_verified(sha)
            .map_err(io::Error::other)?
            .into_file())
    })
}

/// Read-only copies of `inputs` in `spool`, each read from `open`, by field path.
pub(crate) fn stage_inputs_with(
    identity: Option<crate::launch_identity::LaunchIdentity>,
    spool: &Path,
    inputs: &[crate::journal::InputFile],
    open: impl Fn(&crate::journal::InputFile) -> io::Result<File>,
) -> io::Result<BTreeMap<String, serde_json::Value>> {
    let mut granted = BTreeMap::new();
    if inputs.is_empty() {
        return Ok(granted);
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
        let mut source = open(input)?;
        let mut copy = std::os::unix::fs::OpenOptionsExt::mode(
            fs::OpenOptions::new().write(true).create_new(true),
            0o444,
        )
        .open(&local)?;
        if std::io::copy(&mut source, &mut copy)? != input.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "an input's held bytes changed length",
            ));
        }
        copy.sync_all()?;
        if let Some(identity) = identity {
            identity.readable(&local)?;
        }
        granted.insert(
            input.input_id.clone(),
            serde_json::json!({"local":local,"media_type":input.media_type,"digest":input.digest,"length":input.length,"order":input.order,"file_state":null}),
        );
    }
    if let Some(identity) = identity {
        identity.readable(&directory)?;
    }
    Ok(granted)
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
            stderr_tail: &crate::process::tail(&executor.root_path().join("stderr.log")),
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
    peer: u64,
    grants: &'a [HostGrant],
    actor: &'a str,
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
    fn device_tier(
        &mut self,
        frame: &Frame,
        fds: Vec<OwnedFd>,
    ) -> io::Result<(Answer, Vec<OwnedFd>)> {
        let mut answer = Answer::unavailable(frame.seq);
        let Some(custody) = self.custody else {
            return Ok((answer, Vec::new()));
        };
        let key = HoldingKey {
            actor: self.actor.to_string(),
            device: frame.device.clone(),
            layout: frame.layout.clone(),
        };
        let reader = Reader {
            birth: self.birth.clone(),
            exit: self.exit.try_clone()?,
        };
        answer.ok = true;
        answer.code.clear();
        answer.detail.clear();
        let mut custody = custody.lock().unwrap();
        if frame.offer {
            match custody.offer(key, &frame.name, frame.regions.clone(), fds, reader) {
                Ok(Offered::Kept { generation }) => answer.generation = generation,
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
            Some(attached) => {
                answer.held = true;
                answer.generation = attached.generation;
                answer.regions = attached.regions;
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
            };
            // A refusal is an answer: that weight set reads the store, the session goes on.
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
                let cap =
                    self.pool
                        .room_for(self.plan, self.degree, frame.free_bytes, self.others)?;
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
        let (cap, group) = rank_grant(&[Some(30)]);
        assert_eq!((cap, group.is_empty()), (Some(30), true));
        assert_eq!(rank_grant(&[]).0, None);
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
