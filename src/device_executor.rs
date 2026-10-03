//! Existing Runtime device-executor control seam. Models and codecs remain in Python SDK.
use crate::{
    execution::open_artifact,
    journal::ProcessBirth,
    launch_identity::{LaunchIdentity, Seal},
    process::{process_birth, reap_group, tail, Exact, Liveness, Meter, Watch, Watching},
    protocol,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::{OpenOptionsExt, PermissionsExt},
            net::{UnixListener, UnixStream},
            process::{CommandExt, ExitStatusExt},
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

pub const MAX_DEVICE_FRAME: usize = 64 * 1024;
/// Chunks of one weight set's shared GPU regions crossing in one exchange.
const MAX_SHARED_FDS: u32 = 1 << 16;
pub const RESULT_DOCUMENT: &str = "result.canonical";

#[derive(Clone, Debug)]
pub struct ExecutorConfig {
    pub python: PathBuf,
    pub root: PathBuf,
    pub socket: PathBuf,
    /// Explicitly configured locations. Credentials never enter this child.
    pub environment: BTreeMap<String, String>,
    pub seal: Seal,
    pub generation_hold: Option<Arc<File>>,
    pub identity: Option<LaunchIdentity>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Hello {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub runtime_version: String,
    pub tensorfs_version: String,
    pub executor_protocol_revision: u64,
    pub memory: Vec<String>,
    pub sealed: BTreeMap<String, String>,
    pub torch_loaded: bool,
}
impl Hello {
    pub fn offers(&self, capability: &str) -> bool {
        self.memory.iter().any(|offered| offered == capability)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Adapter {
    pub model_id: String,
    pub r#ref: String,
    pub scale: f64,
    pub kind: String,
    pub component: String,
    pub source_component: String,
    pub family: String,
}
impl Default for Adapter {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            r#ref: String::new(),
            scale: 1.0,
            kind: "lora".into(),
            component: String::new(),
            source_component: String::new(),
            family: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Binding {
    pub application: String,
    pub package_interface: String,
    pub model_class: String,
    pub model_binding_path: String,
    pub model_parameter_name: String,
    pub model_parameter_names: Vec<String>,
    pub component: String,
    pub components: Vec<String>,
    pub snapshots: BTreeMap<String, String>,
    pub store: String,
    pub snapshot: String,
    pub release: String,
    /// Hardware derivation variant, e.g. sm86; empty lets the SDK measure it. Never a catalog lane.
    pub variant: String,
    pub development: bool,
    pub custody: String,
    pub objective: String,
    pub steps_basis: u64,
    pub placement: String,
    pub package: String,
    pub model: String,
    pub adapters: Vec<Adapter>,
    pub window_bytes: u64,
    pub slots: u64,
    pub readers: u64,
    pub inflight: u64,
}
impl Default for Binding {
    fn default() -> Self {
        Self {
            application: String::new(),
            package_interface: String::new(),
            model_class: String::new(),
            model_binding_path: String::new(),
            model_parameter_name: String::new(),
            model_parameter_names: Vec::new(),
            component: String::new(),
            components: Vec::new(),
            snapshots: BTreeMap::new(),
            store: String::new(),
            snapshot: String::new(),
            release: String::new(),
            variant: String::new(),
            development: false,
            custody: "canonical".into(),
            objective: String::new(),
            steps_basis: 0,
            placement: String::new(),
            package: String::new(),
            model: String::new(),
            adapters: Vec::new(),
            window_bytes: 4 << 20,
            slots: 16,
            readers: 16,
            inflight: 8,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Budgets {
    pub declared_weight_bytes: u64,
}

/// One model of a many-model construction (an entrypoint with several model slots).
#[derive(Clone, Debug, Serialize)]
pub struct ModelLoad {
    pub binding: Binding,
    pub budgets: Budgets,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum DeviceCommand {
    Hello,
    Shutdown,
    Start {
        devices: String,
        application: String,
        package_interface: PathBuf,
        sequence_parallel_degree: u32,
        #[serde(skip_serializing_if = "is_false")]
        import_only: bool,
    },
    Load {
        construction: String,
        devices: String,
        sequence_parallel_degree: u32,
        binding: Box<Binding>,
        budgets: Budgets,
        /// Several model slots in one construction (H3 turbo: base + LoRA); when set, the
        /// executor loads these and ignores `binding`/`budgets`.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        models: Vec<ModelLoad>,
        #[serde(skip_serializing_if = "Option::is_none")]
        authorized_device_limit_bytes: Option<u64>,
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_pin: String,
        stages: bool,
        #[serde(skip_serializing_if = "is_false")]
        descriptor_sources: bool,
        /// `host_tiers.sealed/1`: every weight set asks for the machine's sealed layout.
        #[serde(skip_serializing_if = "is_false")]
        sealed_tiers: bool,
        /// `load_pinned/1`: the pinned budget, applied before any weight set registers.
        #[serde(skip_serializing_if = "Option::is_none")]
        pinned_bytes: Option<i64>,
        /// Attach GPU weights the machine keeps (`weights.attach/1`).
        #[serde(skip_serializing_if = "is_false")]
        device_weights: bool,
        /// The process's device cap (`process_cap/1`).
        #[serde(skip_serializing_if = "Option::is_none")]
        cap_bytes: Option<u64>,
    },
    Activate {
        construction: String,
    },
    PrepareRequest {
        request_id: String,
        construction: String,
        entrypoint: String,
        payload: Value,
        /// Checked before entry, so a bad pin is the request's refusal, not a fault.
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_kernel: String,
        /// File inputs by field path (identity only; the bytes arrive with Invoke).
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        input_metadata: BTreeMap<String, Value>,
    },
    Invoke {
        request_id: String,
        construction: String,
        entrypoint: String,
        spool: PathBuf,
        deadline_s: Option<f64>,
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_kernel: String,
        plane_budget_bytes: i64,
        stages: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cap_bytes: Option<u64>,
        /// File inputs by field path: read-only copies in this call's spool.
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        inputs: BTreeMap<String, Value>,
        /// The machine's physical free floor on this GPU.
        #[serde(skip_serializing_if = "Option::is_none")]
        floor_bytes: Option<u64>,
        /// Activation growth per stage method learned for this request's shape.
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        activation_bytes: BTreeMap<String, u64>,
    },
    Budget {
        vram_bytes: i64,
        pinned_bytes: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        cap_bytes: Option<u64>,
    },
    Prefetch {
        construction: String,
    },
    /// Offer the machine every GPU region filled and not shared yet (`weights.attach/1`).
    Share,
    /// Release the machine's GPU regions of one layout (`weights.revoke/1`).
    Revoke {
        layout: String,
        generation: u64,
    },
    Unload {
        construction: String,
    },
    Probe {
        collect: bool,
    },
    /// Fork this import-only executor into a new one on its own seam (`fork/1`).
    Fork {
        socket: PathBuf,
        root: PathBuf,
        environment: BTreeMap<String, String>,
    },
}
impl DeviceCommand {
    fn name(&self) -> &'static str {
        match self {
            Self::Hello => "hello",
            Self::Shutdown => "shutdown",
            Self::Start { .. } => "start",
            Self::Load { .. } => "load",
            Self::Activate { .. } => "activate",
            Self::PrepareRequest { .. } => "prepare_request",
            Self::Invoke { .. } => "invoke",
            Self::Budget { .. } => "budget",
            Self::Prefetch { .. } => "prefetch",
            Self::Share => "share",
            Self::Revoke { .. } => "revoke",
            Self::Unload { .. } => "unload",
            Self::Probe { .. } => "probe",
            Self::Fork { .. } => "fork",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Progress,
    Request,
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    ModelSourceRead,
    SealedTier,
    SealedPrefetch,
    DeviceTier,
    BudgetCell,
    StageEnter,
    StageExit,
    DeviceRoom,
    Publish,
    StageMemoLookup,
    StageMemoStore,
    Progress,
    ExecutionActivity,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceRole {
    Header,
    Asset,
    Object,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResultRef {
    pub digest: String,
    pub length: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Outcome {
    pub terminal: String,
    #[serde(default)]
    pub origin: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
    /// The raising traceback of a failed invocation (diagnostics, never a terminal field).
    #[serde(default)]
    pub traceback: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Output {
    pub output_id: String,
    pub asset_ref: String,
    pub kind: String,
    #[serde(default)]
    pub media_type: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HostFrame {
    pub handle: String,
    pub codec: String,
    pub raw: String,
    pub raw_bytes: u64,
    #[serde(default)]
    pub media_type: String,
    #[serde(default)]
    pub facts: BTreeMap<String, Value>,
}

/// SDK observations: absent fields stay unknown; signed byte counters preserve its -1 sentinel.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct PlaneFacts {
    pub budget_bytes: Option<i64>,
    pub committed_bytes: Option<i64>,
    /// GPU regions mapped from custody (`weights.attach/1`): outside `committed_bytes`.
    pub shared_bytes: Option<i64>,
    pub leased_bytes: Option<i64>,
    pub pinned_budget_bytes: Option<i64>,
    pub pinned_bytes: Option<i64>,
    pub context_bytes: Option<i64>,
    pub reserved_bytes: Option<i64>,
    /// Context + torch reserved + the plane's own maps.
    pub process_bytes: Option<i64>,
    pub cap_bytes: Option<i64>,
    pub activation_peak_bytes: Option<i64>,
    pub resident: BTreeMap<String, i64>,
    pub streamed: BTreeMap<String, Streamed>,
    pub h2d_bytes: Option<u64>,
    pub h2d_gbps: Option<f64>,
    pub disk_copy_bytes: Option<u64>,
    pub disk_read_bytes: Option<u64>,
    /// RAM of the machine's sealed layouts the executor maps (the machine's, not its own).
    pub sealed_bytes: Option<u64>,
    pub late: Option<u64>,
    pub stall_ns: Option<u64>,
    pub misses: Option<u64>,
    pub evictions: Option<u64>,
    pub oom_retries: Option<u64>,
    pub modes: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Streamed {
    pub blocks: u64,
    pub resident_blocks: u64,
    pub window: u64,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Metrics {
    pub handler_ms: Option<f64>,
    pub device_lease_ms: Option<f64>,
    pub d2h_wait_ms: Option<f64>,
    pub peak_vram_bytes: Option<i64>,
    pub activation_peak_bytes: Option<i64>,
    pub activation_peaks: BTreeMap<String, i64>,
    pub allocated_at_start_bytes: Option<i64>,
    pub working_peak_vram_bytes: Option<i64>,
    pub rss_at_end_bytes: Option<i64>,
    pub started_unix: Option<f64>,
    pub gpu_count: Option<u32>,
    pub shape_cell: Option<String>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct LoadFacts {
    pub cuda_init_ms: Option<f64>,
    pub prepare_ms: Option<f64>,
    pub stages: Vec<(String, f64)>,
    pub warm_ms: Option<f64>,
    pub filled_bytes: Option<u64>,
    pub device_free_bytes: Option<i64>,
    pub device_total_bytes: Option<i64>,
    pub allocator_bytes: Option<i64>,
    pub reserved_bytes: Option<i64>,
    pub rss_bytes: Option<i64>,
    pub plane: Option<PlaneFacts>,
    pub layouts: BTreeMap<String, Layout>,
}

/// One weight set as the plane holds it; `planned_*` count decoded copies (Runtime 0.18.103+).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Layout {
    pub common: u64,
    pub blocks: Vec<u64>,
    pub planned_total: Option<u64>,
    pub planned_floor: Option<u64>,
}
impl Layout {
    pub fn planned_total(&self) -> u64 {
        self.planned_total
            .unwrap_or(self.common + self.blocks.iter().sum::<u64>())
    }
    /// Common and the largest block: the least a stage of it runs in.
    pub fn planned_floor(&self) -> u64 {
        self.planned_floor
            .unwrap_or(self.common + self.blocks.iter().max().copied().unwrap_or(0))
    }
}

/// One decode per frame; only consumed business fields are represented.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Frame {
    pub event: Option<Event>,
    pub reply: String,
    pub ok: bool,
    pub code: String,
    pub detail: String,
    /// A refusal's own terminal (`refused`/`failed`) and origin (`request`/`author`/`runtime`).
    pub terminal: String,
    pub origin: String,
    /// A traced command refusal's raising traceback.
    pub traceback: String,
    #[serde(flatten)]
    pub hello: Hello,
    pub outcome: Option<Outcome>,
    pub quiescent: bool,
    pub poisoned: String,
    pub reused: bool,
    /// `start` of a group (degree > 1): the follower ranks' pids, rank 1 first.
    pub follower_pids: Vec<u32>,
    pub result_ref: Option<ResultRef>,
    pub outputs: Vec<Output>,
    pub frames: Vec<HostFrame>,
    pub max_output_bytes: Option<u64>,
    pub request_id: String,
    pub seq: u64,
    pub kind: Kind,
    pub descriptor: bool,
    pub name: String,
    pub layout: String,
    pub manifest: String,
    pub sha256: String,
    pub role: SourceRole,
    pub length: u64,
    pub offer: bool,
    pub method: String,
    pub components: Vec<String>,
    pub growth_bytes: u64,
    pub wall_ns: u64,
    pub stall_ns: u64,
    pub passes: Option<u64>,
    pub yielded: bool,
    pub free_bytes: u64,
    pub facts: Option<LoadFacts>,
    pub plane: Option<PlaneFacts>,
    pub metrics: Option<Metrics>,
    pub stage: String,
    pub position: Option<u64>,
    pub total: Option<u64>,
    pub stage_fraction: Option<f64>,
    pub overall_fraction: Option<f64>,
    pub step_ms: Option<f64>,
    pub advance: u64,
    /// PrepareRequest: the request's normalized features (its shape).
    pub features: BTreeMap<String, Value>,
    /// `device_tier`: fds that follow the frame, one per chunk of `regions`.
    pub descriptors: u32,
    pub device: String,
    pub regions: Vec<crate::resident_custody::SharedRegion>,
    pub shared_bytes: u64,
    pub released_bytes: u64,
    // `publish` (Outputs.publish): one product of a declared output.
    pub output: String,
    pub label: String,
    pub asset_ref: String,
    pub asset_kind: String,
    pub media_type: String,
    pub size_bytes: u64,
    pub digest: String,
    pub parts: Vec<PublishPart>,
    /// `start`: the executor's legs, `[[name, ms], ...]`; read leniently.
    pub stages: Value,
}

/// One part of a composite product: a spool file and the media time it adds.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct PublishPart {
    pub local: String,
    pub duration_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    pub event: &'static str,
    pub seq: u64,
    pub ok: bool,
    pub code: String,
    pub detail: String,
    pub held: bool,
    pub budget_bytes: i64,
    /// A `DeviceRoom` answer: the process cap raised into the room made; -1 keeps it.
    pub cap_bytes: i64,
    pub descriptor: bool,
    pub sha256: String,
    pub length: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub generation: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<crate::resident_custody::SharedRegion>,
    #[serde(skip_serializing_if = "is_false")]
    pub duplicate: bool,
    #[serde(skip_serializing_if = "is_zero")]
    pub descriptors: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub digest: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub sequence: u64,
}
impl Answer {
    pub fn unavailable(seq: u64) -> Self {
        Self {
            event: "answer",
            seq,
            ok: false,
            code: "capability_unavailable".into(),
            detail: "owner has not qualified this operation".into(),
            held: false,
            budget_bytes: -1,
            cap_bytes: -1,
            descriptor: false,
            sha256: String::new(),
            length: 0,
            generation: 0,
            regions: Vec::new(),
            duplicate: false,
            descriptors: 0,
            digest: String::new(),
            sequence: 0,
        }
    }
}

pub trait Services {
    fn progress(&mut self, _: &Frame) {}
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        drop(descriptor);
        Ok((Answer::unavailable(frame.seq), None))
    }
    /// `device_tier` (`weights.attach/1`): `fds` arrived with the request; the returned ones
    /// follow the answer.
    fn device_tier(
        &mut self,
        frame: &Frame,
        fds: Vec<std::os::fd::OwnedFd>,
    ) -> io::Result<(Answer, Vec<std::os::fd::OwnedFd>)> {
        drop(fds);
        Ok((Answer::unavailable(frame.seq), Vec::new()))
    }
}

pub struct Baseline;
impl Services for Baseline {}

pub struct DeviceExecutor {
    child: Option<Child>,
    /// Forked from an import-only executor (`fork/1`), which is its parent and reaps it.
    forked: bool,
    exact: Exact,
    stream: UnixStream,
    pub birth: ProcessBirth,
    pub hello: Hello,
    /// Sampling resolution of the wedge rule. Tests shorten it; nothing else changes it.
    pub liveness: Liveness,
    root: PathBuf,
    socket: PathBuf,
    _generation_hold: Option<Arc<File>>,
    codec: Arc<Codec>,
    retained: Vec<Box<dyn Send>>,
    identity: Option<LaunchIdentity>,
    watched: Watched,
    /// Longest gap between frames this executor has shown during invocations.
    worst_gap: Duration,
}

/// Cooperative first: the executor stops at its next safe point. The running invocation's
/// frame meter then decides whether it stopped moving and must be killed.
#[derive(Clone)]
pub struct Cancellation {
    root: PathBuf,
    identity: Option<LaunchIdentity>,
    watched: Watched,
}
impl Cancellation {
    pub fn cancel(&self, request_id: &str) -> io::Result<()> {
        let temporary = self
            .root
            .join(format!("executor.cancel.{}.pending", uuid::Uuid::new_v4()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(request_id.as_bytes())?;
        file.sync_all()?;
        if let Some(identity) = self.identity {
            identity.readable(&temporary)?;
        }
        fs::rename(temporary, self.root.join("executor.cancel"))?;
        File::open(&self.root)?.sync_all()?;
        if let Some(watch) = self.watched.lock().unwrap().as_ref() {
            watch.canceled();
        }
        Ok(())
    }
}

type Watched = Arc<Mutex<Option<Arc<Watch>>>>;

/// Executors fork from an import-only executor that offers this (Runtime `fork/1`).
pub const FORK: &str = "fork/1";

pub enum Forked {
    Ready(Box<DeviceExecutor>),
    /// No process was made; the configuration is free for a spawn.
    Refused(Box<ExecutorConfig>, String),
}

/// A forked executor's wait status while it is its parent's zombie (`/proc` `exit_code`).
fn zombie_status(birth: &ProcessBirth) -> Option<ExitStatus> {
    let stat = fs::read_to_string(format!("/proc/{}/stat", birth.pid)).ok()?;
    let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    let start: u64 = fields.get(19)?.parse().ok()?;
    let code: i32 = fields.get(49)?.parse().ok()?;
    (fields.first() == Some(&"Z") && start == birth.start_ticks).then(|| ExitStatus::from_raw(code))
}

/// The executor's root and listening socket, and the sealed environment it starts with.
fn endpoint(config: &ExecutorConfig) -> io::Result<(UnixListener, BTreeMap<String, String>)> {
    fs::create_dir_all(&config.root)?;
    fs::set_permissions(&config.root, fs::Permissions::from_mode(0o700))?;
    if let Some(identity) = config.identity {
        identity.own(&config.root)?;
    }
    let listener = UnixListener::bind(&config.socket)?;
    fs::set_permissions(&config.socket, fs::Permissions::from_mode(0o600))?;
    if let Some(identity) = config.identity {
        identity.socket(&config.socket)?;
    }
    Ok((listener, config.seal.environment(&config.environment)))
}

fn parent_of(pid: u32) -> io::Result<u32> {
    fs::read_to_string(format!("/proc/{pid}/status"))?
        .lines()
        .find_map(|line| line.strip_prefix("PPid:"))
        .and_then(|value| value.trim().parse().ok())
        .ok_or_else(|| io::Error::other("process status names no parent"))
}

/// The executor exited before it connected: no authored code ran. Deterministic causes
/// (an SDK without this module, a broken environment) fail the run with this evidence.
#[derive(Debug)]
pub struct EndedBeforeStart {
    pub status: ExitStatus,
    pub stderr_tail: String,
}
impl std::fmt::Display for EndedBeforeStart {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "executor exited before connecting ({}): {}",
            self.status, self.stderr_tail
        )
    }
}
impl std::error::Error for EndedBeforeStart {}

/// How an executor ended: its reaped status and the measurement behind a kill, if any.
#[derive(Debug)]
pub struct Ended {
    pub status: ExitStatus,
    pub killed: Option<String>,
}

#[derive(Clone)]
pub struct CodecConfig {
    pub python: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub generation_hold: Option<Arc<File>>,
}

/// The selected SDK's encoder for codecs this machine does not encode itself (audio,
/// video): one helper process per executor, started at its first such output and ended
/// with the executor (or when the machine dies: it exits on stdin EOF).
pub struct Codec {
    config: CodecConfig,
    log: PathBuf,
    helper: Mutex<Option<CodecHelper>>,
}
struct CodecHelper {
    child: Child,
    input: std::process::ChildStdin,
    output: std::io::BufReader<std::process::ChildStdout>,
}
impl Drop for CodecHelper {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Codec {
    pub fn python(&self) -> &Path {
        &self.config.python
    }
    pub fn new(config: CodecConfig, log: PathBuf) -> Self {
        Self {
            config,
            log,
            helper: Mutex::new(None),
        }
    }
    fn spawn(&self) -> io::Result<CodecHelper> {
        let hold = self
            .config
            .generation_hold
            .as_ref()
            .map(|hold| hold.as_raw_fd());
        let mut command = Command::new(&self.config.python);
        command
            .args([
                "-I",
                "-c",
                include_str!("../python/cozy_machine_client/device_codec.py"),
            ])
            .env_clear()
            .envs(&self.config.environment)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .mode(0o600)
                    .open(&self.log)?,
            );
        // SAFETY: only async-signal-safe fcntl on the retained generation hold.
        unsafe {
            command.pre_exec(move || {
                if let Some(descriptor) = hold {
                    if libc::fcntl(descriptor, libc::F_SETFD, 0) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let mut child = command.spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("codec stdin missing"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("codec stdout missing"))?;
        Ok(CodecHelper {
            child,
            input,
            output: std::io::BufReader::new(output),
        })
    }
    fn encode(&self, spool: &Path, reply: &Frame) -> io::Result<PostReply> {
        use std::io::BufRead;
        let mut helper = self.helper.lock().unwrap();
        if helper.is_none() {
            *helper = Some(self.spawn()?);
        }
        let request = PostRequest {
            spool,
            frames: &reply.frames,
            outputs: &reply.outputs,
            max_output_bytes: reply.max_output_bytes,
        };
        let mut line = serde_json::to_vec(&request)?;
        line.push(b'\n');
        let mut answer = String::new();
        let exchanged = (|| {
            let current = helper.as_mut().expect("spawned codec helper");
            current.input.write_all(&line)?;
            current.input.flush()?;
            current.output.read_line(&mut answer)
        })();
        match exchanged {
            Ok(n) if n > 0 => {}
            Ok(_) | Err(_) => {
                helper.take(); // the next output starts a fresh helper
                return Err(io::Error::other(format!(
                    "selected SDK encoder stopped; see {}",
                    self.log.display()
                )));
            }
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Answer {
            Done(PostReply),
            Refused { error: String },
        }
        match serde_json::from_str::<Answer>(&answer).map_err(io::Error::other)? {
            Answer::Done(post) => Ok(post),
            Answer::Refused { error } => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("selected SDK encoder refused: {error}"),
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AssetBinding {
    pub output_id: String,
    pub asset_ref: String,
    pub name: String,
    pub kind: String,
    pub media_type: String,
    pub length: u64,
    pub producer_digest: String,
    pub sha256: String,
}
#[derive(Debug, Deserialize)]
struct PostReply {
    bindings: Vec<AssetBinding>,
}
#[derive(Serialize)]
struct PostRequest<'a> {
    spool: &'a Path,
    frames: &'a [HostFrame],
    outputs: &'a [Output],
    max_output_bytes: Option<u64>,
}

pub fn postprocess(
    codec: &Codec,
    spool: &Path,
    reply: &Frame,
) -> io::Result<(Value, Vec<AssetBinding>)> {
    let result = read_result(spool, reply)?;
    let directory = File::open(spool)?;
    let post = match encode_native(spool, reply)? {
        Some(bindings) => PostReply { bindings },
        None => codec.encode(spool, reply)?,
    };
    for binding in &post.bindings {
        let mut file = open_artifact(spool, Path::new(&binding.name))?;
        let mut hash = tensorfs_core::sha256::Sha256::new();
        let mut length = 0;
        let mut buffer = [0; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            length += count as u64;
        }
        if length != binding.length || tensorfs_core::sha256::hex(&hash.finish()) != binding.sha256
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SDK-bound encoded bytes changed before custody",
            ));
        }
        file.sync_all()?;
    }
    directory.sync_all()?;
    Ok((result, post.bindings))
}
/// Encodes registered PNG/WebP frames in this process exactly as the SDK's post thread
/// does (lossless RGB; PNG at zlib's fastest level, WebP lossless) and binds every output
/// to its spool file. None when a frame needs a codec only the SDK has.
fn encode_native(spool: &Path, reply: &Frame) -> io::Result<Option<Vec<AssetBinding>>> {
    if reply
        .frames
        .iter()
        .any(|f| !matches!(f.codec.as_str(), "png" | "webp"))
    {
        return Ok(None);
    }
    let invalid = |detail: &str| io::Error::new(io::ErrorKind::InvalidData, detail.to_string());
    let blob = |reference: &str| -> io::Result<String> {
        let tail: Vec<_> = reference.rsplitn(3, '/').collect();
        match tail.as_slice() {
            [name, kind, _]
                if !name.is_empty()
                    && !name.contains(['/', '\0'])
                    && *name != "."
                    && *name != ".." =>
            {
                Ok(format!("{kind}-{name}"))
            }
            _ => Err(invalid("SDK output has no owned spool binding")),
        }
    };
    let mut media = BTreeMap::new();
    for frame in &reply.frames {
        let mut raw = Vec::new();
        open_artifact(spool, Path::new(&frame.raw))?.read_to_end(&mut raw)?;
        let dimension = |name: &str| {
            frame
                .facts
                .get(name)
                .and_then(Value::as_u64)
                .filter(|v| *v > 0 && *v <= u32::MAX as u64)
                .map(|v| v as u32)
        };
        let (Some(width), Some(height)) = (dimension("width"), dimension("height")) else {
            return Err(invalid("registered frame has no dimensions"));
        };
        if raw.len() as u64 != frame.raw_bytes
            || raw.len() as u64 != width as u64 * height as u64 * 3
        {
            return Err(invalid("registered raw frame length changed"));
        }
        let mut encoded = Vec::new();
        if frame.codec == "png" {
            let mut encoder = png::Encoder::new(&mut encoded, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_compression(png::Compression::Fast);
            encoder
                .write_header()
                .and_then(|mut w| w.write_image_data(&raw))
                .map_err(io::Error::other)?;
        } else {
            image_webp::WebPEncoder::new(&mut encoded)
                .encode(&raw, width, height, image_webp::ColorType::Rgb8)
                .map_err(io::Error::other)?;
        }
        let name = blob(&frame.handle)?;
        let temporary = spool.join(format!(".codec-{}", uuid::Uuid::new_v4().simple()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary)?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        fs::rename(&temporary, spool.join(&name))?;
        media.insert(
            frame.handle.clone(),
            if frame.codec == "png" {
                "image/png"
            } else {
                "image/webp"
            },
        );
    }
    let mut total = 0u64;
    let mut bindings = vec![];
    for output in &reply.outputs {
        let name = blob(&output.asset_ref)?;
        let mut encoded = Vec::new();
        open_artifact(spool, Path::new(&name))?.read_to_end(&mut encoded)?;
        use blake2::digest::{Update, VariableOutput};
        let mut blake = blake2::Blake2bVar::new(16).map_err(io::Error::other)?;
        blake.update(&encoded);
        let mut digest = [0; 16];
        blake
            .finalize_variable(&mut digest)
            .map_err(io::Error::other)?;
        let producer = format!("blake2b:{}", tensorfs_core::sha256::hex(&digest));
        if !output.digest.is_empty() && output.digest != producer
            || output.size_bytes.is_some_and(|n| n != encoded.len() as u64)
        {
            return Err(invalid("SDK output bytes changed before custody"));
        }
        total += encoded.len() as u64;
        if reply.max_output_bytes.is_some_and(|limit| total > limit) {
            return Err(invalid(
                "encoded output exceeds the authored aggregate allowance",
            ));
        }
        bindings.push(AssetBinding {
            output_id: output.output_id.clone(),
            asset_ref: output.asset_ref.clone(),
            name,
            kind: output.kind.clone(),
            media_type: media
                .get(&output.asset_ref)
                .map(|m| m.to_string())
                .unwrap_or_else(|| output.media_type.clone()),
            length: encoded.len() as u64,
            producer_digest: producer,
            sha256: tensorfs_core::sha256::hex_digest(&encoded),
        });
    }
    Ok(Some(bindings))
}

impl DeviceExecutor {
    pub fn spawn(config: ExecutorConfig) -> io::Result<Self> {
        Self::spawn_observed(config, |_, _| Ok(()))
    }

    pub fn spawn_observed(
        config: ExecutorConfig,
        on_birth: impl FnOnce(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        Self::spawn_internal(config, None, on_birth)
    }
    pub fn spawn_owned(
        config: ExecutorConfig,
        launcher: &crate::child_launcher::ChildLauncher,
        on_birth: impl FnOnce(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        Self::spawn_internal(config, Some(launcher), on_birth)
    }
    fn spawn_internal(
        config: ExecutorConfig,
        launcher: Option<&crate::child_launcher::ChildLauncher>,
        on_birth: impl FnOnce(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        let (listener, environment) = endpoint(&config)?;
        let mut command = crate::launch_identity::trampoline(&config.python, config.identity)?;
        command
            .args(["-I", "-m", "cozy_runtime.internal.executor", "--socket"])
            .arg(&config.socket)
            .arg("--root")
            .arg(&config.root)
            .env_clear()
            .envs(&environment)
            .stdin(Stdio::null())
            .stdout(File::create(config.root.join("stdout.log"))?)
            .stderr(File::create(config.root.join("stderr.log"))?);
        if let Some(hold) = &config.generation_hold {
            let fd = hold.as_raw_fd();
            // SAFETY: only async-signal-safe fcntl; this retained shared hold survives core death.
            unsafe {
                command.pre_exec(move || {
                    if libc::fcntl(fd, libc::F_SETFD, 0) < 0 {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                });
            }
        }
        let child = match launcher {
            Some(launcher) => launcher.spawn(command)?,
            None => command.spawn()?,
        };
        let launched = Unready {
            pid: child.id(),
            parent: std::process::id(),
            child: Some(child),
            exact: None,
            birth: None,
        };
        Self::connect(config, listener, environment, launched, on_birth)
    }

    /// An executor forked from this import-only one (`fork/1`), its imports done. It dials
    /// its own seam and is checked exactly as a spawned one is, with this process as its
    /// parent. Refused (no process exists) gives the configuration back for a spawn.
    pub fn fork(
        &mut self,
        config: ExecutorConfig,
        on_birth: impl FnOnce(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Forked> {
        let (listener, environment) = endpoint(&config)?;
        let reply = self.command(
            &DeviceCommand::Fork {
                socket: config.socket.clone(),
                root: config.root.clone(),
                environment: environment.clone(),
            },
            &mut Baseline,
        );
        let pid = match reply {
            Ok(reply) if reply.ok && reply.hello.pid != 0 => reply.hello.pid,
            refused => {
                drop(listener);
                fs::remove_file(&config.socket)?;
                let reason = match refused {
                    Ok(reply) => format!("{}: {}", reply.code, reply.detail),
                    Err(error) => error.to_string(),
                };
                return Ok(Forked::Refused(Box::new(config), reason));
            }
        };
        let launched = Unready {
            pid,
            parent: self.birth.pid,
            child: None,
            exact: None,
            birth: None,
        };
        Self::connect(config, listener, environment, launched, on_birth)
            .map(|executor| Forked::Ready(Box::new(executor)))
    }

    /// Wait for the launched process to dial or end, prove the connection is that process,
    /// and take its Hello. Until then every early return kills and reaps it: nothing
    /// authored has run, so no measurement is needed to end it.
    fn connect(
        config: ExecutorConfig,
        listener: UnixListener,
        environment: BTreeMap<String, String>,
        mut unready: Unready,
        on_birth: impl FnOnce(&ProcessBirth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Self> {
        let codec = Arc::new(Codec::new(
            CodecConfig {
                python: config.python.clone(),
                environment,
                generation_hold: config.generation_hold.clone(),
            },
            config.root.join("codec.stderr.log"),
        ));
        let pid = unready.pid;
        let birth = process_birth(pid)?;
        let exact = Exact::open(&birth)?
            .ok_or_else(|| io::Error::other("launched executor has no exact birth"))?;
        unready.exact = Some(exact.try_clone()?);
        unready.birth = Some(birth.clone());
        if unready.child.is_none() && parent_of(pid)? != unready.parent {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "forked executor is not a child of the process it was forked from",
            ));
        }
        let watched: Watched = Arc::default();
        on_birth(
            &birth,
            &Cancellation {
                root: config.root.clone(),
                identity: config.identity,
                watched: watched.clone(),
            },
        )?;
        // Wait for connection OR observed process death. No elapsed-time startup kill.
        let mut poll = [
            libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: exact.as_file().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        loop {
            let result = unsafe { libc::poll(poll.as_mut_ptr(), 2, -1) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if poll[0].revents & libc::POLLIN != 0 {
                break;
            }
            if poll[1].revents != 0 {
                let status = unready.ended()?;
                return Err(io::Error::other(EndedBeforeStart {
                    status,
                    stderr_tail: tail(&config.root.join("stderr.log")),
                }));
            }
        }
        let (stream, _) = listener.accept()?;
        let mut credential: libc::ucred = unsafe { std::mem::zeroed() };
        let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        if unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut credential as *mut libc::ucred).cast(),
                &mut size,
            )
        } < 0
        {
            return Err(io::Error::last_os_error());
        }
        let expected_uid = config
            .identity
            .map_or_else(|| unsafe { libc::geteuid() }, |identity| identity.uid);
        let expected_gid = config
            .identity
            .map_or_else(|| unsafe { libc::getegid() }, |identity| identity.gid);
        if credential.pid != pid as i32
            || credential.uid != expected_uid
            || credential.gid != expected_gid
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "device executor connection differs from launched process",
            ));
        }
        let parent = unready.parent;
        let mut executor = Self {
            child: None,
            forked: unready.child.is_none(),
            exact,
            stream,
            birth,
            hello: Hello::default(),
            liveness: Liveness::default(),
            root: config.root,
            socket: config.socket,
            _generation_hold: config.generation_hold,
            codec,
            retained: Vec::new(),
            identity: config.identity,
            watched,
            worst_gap: Duration::ZERO,
        };
        let hello = executor.command(&DeviceCommand::Hello, &mut Baseline)?;
        // Older Runtimes omit ppid/pgid; a reported value must match (Runtime worker hello).
        let mismatched = config.seal.mismatches(&hello.hello.sealed);
        if !hello.ok
            || hello.hello.pid != executor.birth.pid
            || (hello.hello.ppid != 0 && hello.hello.ppid != parent)
            || (hello.hello.pgid != 0 && hello.hello.pgid != executor.birth.pid)
            || !mismatched.is_empty()
        {
            return Err(io::Error::other(format!(
                "stock executor hello did not identify the launched, sealed process \
                 (pid {}, parent {}, group {}; differing sealed names {mismatched:?})",
                hello.hello.pid, hello.hello.ppid, hello.hello.pgid
            )));
        }
        // Proven: the executor now owns the child; the launch guard no longer kills it.
        executor.child = unready.child.take();
        unready.exact = None;
        unready.pid = 0;
        executor.hello = hello.hello;
        Ok(executor)
    }
    pub fn codec(&self) -> Arc<Codec> {
        self.codec.clone()
    }
    pub fn root_path(&self) -> &Path {
        &self.root
    }
    pub fn cancellation(&self) -> Cancellation {
        Cancellation {
            root: self.root.clone(),
            identity: self.identity,
            watched: self.watched.clone(),
        }
    }

    pub fn observer_pidfd(&self) -> io::Result<File> {
        self.exact.as_file().try_clone()
    }

    /// Retain a source/resource until this exact receiver exits, including owner-handle loss.
    /// Observer lifetimes must never own the DeviceExecutor handle.
    pub fn retain_until_exit(&mut self, resource: impl Send + 'static) {
        self.retained.push(Box::new(resource));
    }

    fn offered(&self, command: &DeviceCommand) -> io::Result<()> {
        if matches!(command, DeviceCommand::Fork { .. }) && !self.hello.offers(FORK) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "fork capability absent",
            ));
        }
        if let DeviceCommand::Start {
            import_only: true, ..
        } = command
        {
            if !self.hello.offers("import_only") {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "import-only startup capability absent",
                ));
            }
        }
        if let DeviceCommand::Load {
            sealed_tiers,
            pinned_bytes,
            ..
        } = command
        {
            let plane = self.hello.offers("weight_plane/1");
            if (*sealed_tiers && !(plane && self.hello.offers("host_tiers.sealed/1")))
                || (pinned_bytes.is_some() && !(plane && self.hello.offers("load_pinned/1")))
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "sealed host tiers or a Load pinned budget without the executor capability",
                ));
            }
        }
        if matches!(
            command,
            DeviceCommand::Load { stages: true, .. } | DeviceCommand::Invoke { stages: true, .. }
        ) && !self.hello.offers("stage/1")
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "stage-turn capability absent",
            ));
        }
        let sharing = match command {
            DeviceCommand::Load { device_weights, .. } => *device_weights,
            DeviceCommand::Share => true,
            _ => false,
        };
        if (sharing && !self.hello.offers("weights.attach/1"))
            || (matches!(command, DeviceCommand::Revoke { .. })
                && !self.hello.offers("weights.revoke/1"))
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "shared GPU weight capability absent",
            ));
        }
        if matches!(command, DeviceCommand::Budget { .. }) && !self.hello.offers("weight_plane/1") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "plane budget capability absent; legacy residency remains available",
            ));
        }
        if let DeviceCommand::Load {
            descriptor_sources: true,
            sequence_parallel_degree,
            ..
        } = command
        {
            if *sequence_parallel_degree != 1 || !self.hello.offers("model_sources.descriptors/1") {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "descriptor model sources require qualified world-one capability",
                ));
            }
        }
        Ok(())
    }

    /// One command, observed: a wedged executor is killed on its measured lack of progress,
    /// which closes the channel and returns this call with the measurement.
    pub fn command(
        &mut self,
        command: &DeviceCommand,
        services: &mut impl Services,
    ) -> io::Result<Frame> {
        self.offered(command)?;
        let meter = match command {
            DeviceCommand::Invoke { .. } => Meter::Frames,
            _ => Meter::Burn,
        };
        let watching = Watching::start(
            self.exact.try_clone()?,
            meter,
            self.liveness,
            self.worst_gap,
            command.name(),
        )?;
        *self.watched.lock().unwrap() = Some(watching.watch());
        let result = self.exchange(command, services, &watching.watch());
        *self.watched.lock().unwrap() = None;
        let (killed, worst_gap) = watching.finish();
        if meter == Meter::Frames {
            self.worst_gap = self.worst_gap.max(worst_gap);
        }
        match killed {
            Some(verdict) => Err(io::Error::other(verdict)),
            None => result,
        }
    }

    fn exchange(
        &mut self,
        command: &DeviceCommand,
        services: &mut impl Services,
        watch: &Watch,
    ) -> io::Result<Frame> {
        write_frame(&mut self.stream, command)?;
        loop {
            let frame = read_frame(&mut self.stream)?
                .ok_or_else(|| io::Error::other("stock executor EOF before reply"))?;
            watch.frame();
            match frame.event {
                Some(Event::Progress) => services.progress(&frame),
                Some(Event::Request) if frame.kind == Kind::DeviceTier => {
                    if frame.descriptors > MAX_SHARED_FDS {
                        return Err(io::Error::other("device tier names too many descriptors"));
                    }
                    let mut fds = Vec::with_capacity(frame.descriptors as usize);
                    for _ in 0..frame.descriptors {
                        fds.push(protocol::recv_fd(&self.stream)?);
                    }
                    let (mut answer, out) = services.device_tier(&frame, fds)?;
                    answer.event = "answer";
                    answer.seq = frame.seq;
                    answer.descriptor = false;
                    answer.descriptors = out.len() as u64;
                    write_frame(&mut self.stream, &answer)?;
                    for fd in &out {
                        protocol::send_fd(&self.stream, fd)?;
                    }
                }
                Some(Event::Request) => {
                    // The machine's own answer time is not the executor's stillness.
                    watch.serving(true);
                    let answered = self.answer(&frame, services);
                    watch.serving(false);
                    answered?;
                }
                Some(Event::Unknown) => (),
                None if frame.reply == command.name() => return Ok(frame),
                None => return Err(io::Error::other("out-of-order stock executor reply")),
            }
        }
    }

    fn answer(&mut self, frame: &Frame, services: &mut impl Services) -> io::Result<()> {
        let descriptor = if frame.descriptor {
            Some(File::from(protocol::recv_fd(&self.stream)?))
        } else {
            None
        };
        let (mut answer, descriptor) = services.request(frame, descriptor)?;
        if frame.kind == Kind::ModelSourceRead && answer.ok {
            let source = descriptor
                .as_ref()
                .ok_or_else(|| io::Error::other("successful model source omitted readonly file"))?;
            let flags = unsafe { libc::fcntl(source.as_raw_fd(), libc::F_GETFL) };
            if flags < 0 {
                return Err(io::Error::last_os_error());
            }
            if flags & libc::O_ACCMODE != libc::O_RDONLY
                || flags & libc::O_PATH != 0
                || !source.metadata()?.is_file()
                || source.metadata()?.len() != answer.length
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "model source is not the declared readonly regular file",
                ));
            }
        }
        answer.event = "answer";
        answer.seq = frame.seq;
        answer.descriptor = descriptor.is_some();
        write_frame(&mut self.stream, &answer)?;
        if let Some(descriptor) = descriptor {
            protocol::send_fd(&self.stream, &descriptor)?;
        }
        Ok(())
    }

    /// Attempt-keyed cooperative cancellation; observer teardown never calls this.
    pub fn cancel(&self, request_id: &str) -> io::Result<()> {
        self.cancellation().cancel(request_id)
    }

    /// Ask the executor to exit, then observe that exact exit (killing only a wedge).
    pub fn shutdown(mut self) -> io::Result<()> {
        let asked = self.command(&DeviceCommand::Shutdown, &mut Baseline);
        let ended = self.terminate()?;
        asked?;
        if !ended.status.success() {
            return Err(io::Error::other(format!(
                "stock executor shutdown: {}",
                ended.status
            )));
        }
        Ok(())
    }

    /// Close the channel, let the executor stop at its next exchange, kill it only on a
    /// measured wedge, then reap it and release what it held. Blocks until it is gone.
    pub fn terminate(mut self) -> io::Result<Ended> {
        let parts = self.parts();
        parts.end()
    }

    fn parts(&mut self) -> Ending {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        Ending {
            exact: self.exact.try_clone(),
            child: self.child.take(),
            forked: std::mem::take(&mut self.forked),
            retained: std::mem::take(&mut self.retained),
            socket: std::mem::take(&mut self.socket),
            liveness: self.liveness,
        }
    }
}

/// A launched process before its Hello proves it: dropped, it is killed (and reaped by its
/// parent: this machine, or the executor it was forked from).
struct Unready {
    /// 0 once the executor owns the process.
    pid: u32,
    /// The process it must be a child of: this machine, or the executor it was forked from.
    parent: u32,
    child: Option<Child>,
    exact: Option<Exact>,
    birth: Option<ProcessBirth>,
}
impl Unready {
    /// Its exit status, once the process is seen to have exited.
    fn ended(&mut self) -> io::Result<ExitStatus> {
        match self.child.as_mut() {
            Some(child) => child.wait(),
            None => Ok(self
                .birth
                .as_ref()
                .and_then(zombie_status)
                .unwrap_or_else(|| ExitStatus::from_raw(0))),
        }
    }
}
impl Drop for Unready {
    fn drop(&mut self) {
        if self.pid == 0 {
            return;
        }
        let exact = self.exact.take().or_else(|| {
            process_birth(self.pid)
                .ok()
                .and_then(|birth| Exact::open(&birth).ok().flatten())
        });
        if let Some(exact) = &exact {
            let _ = exact.kill();
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Everything an executor's teardown must hold until its exit is observed.
struct Ending {
    exact: io::Result<Exact>,
    child: Option<Child>,
    forked: bool,
    retained: Vec<Box<dyn Send>>,
    socket: PathBuf,
    liveness: Liveness,
}
impl Ending {
    fn end(mut self) -> io::Result<Ended> {
        let exact = match &self.exact {
            Ok(exact) => exact,
            Err(error) => {
                // An unobservable receiver is not authority to release its sources.
                std::mem::forget(std::mem::take(&mut self.retained));
                return Err(io::Error::other(format!(
                    "executor exit unobservable: {error}"
                )));
            }
        };
        // Followers (degree > 1) stay in the leader's group and are waited for too.
        let (mut status, killed) = match reap_group(exact, self.child.as_mut(), self.liveness) {
            Ok(ended) => ended,
            Err(error) => {
                std::mem::forget(std::mem::take(&mut self.retained));
                return Err(error);
            }
        };
        if self.forked {
            status = zombie_status(&exact.birth).unwrap_or(status);
        }
        if !self.socket.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.socket);
        }
        drop(std::mem::take(&mut self.retained));
        Ok(Ended { status, killed })
    }
}

/// Dropping the handle ends the executor before returning: whoever releases its slot or
/// reservation next does so after its exit, on every path including early returns.
impl Drop for DeviceExecutor {
    fn drop(&mut self) {
        if self.child.is_none() && !self.forked && self.retained.is_empty() {
            return;
        }
        if let Err(error) = self.parts().end() {
            eprintln!("executor teardown: {error}");
        }
    }
}

pub fn read_result(spool: &Path, reply: &Frame) -> io::Result<Value> {
    if !reply.ok
        || reply
            .outcome
            .as_ref()
            .is_none_or(|outcome| outcome.terminal != "succeeded")
    {
        return Err(io::Error::other(
            "stock invocation has no successful typed outcome",
        ));
    }
    if !reply.quiescent || !reply.poisoned.is_empty() {
        return Err(io::Error::other(
            "stock invocation did not settle quiescent",
        ));
    }
    let reference = reply
        .result_ref
        .as_ref()
        .ok_or_else(|| io::Error::other("stock invocation has no retained result reference"))?;
    let mut bytes = Vec::new();
    open_artifact(spool, Path::new(RESULT_DOCUMENT))?.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != reference.length
        || format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes)) != reference.digest
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "spooled result identity differs from executor reference",
        ));
    }
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
fn is_false(value: &bool) -> bool {
    !*value
}

fn write_frame(
    stream: &mut std::os::unix::net::UnixStream,
    value: &impl Serialize,
) -> io::Result<()> {
    let value = serde_json::to_value(value)?;
    scan(&value)?;
    let bytes = serde_json::to_vec(&value)?;
    if bytes.len() > MAX_DEVICE_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "device control frame exceeds64KiB; data belongs in spool",
        ));
    }
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(&bytes)
}
/// SDK control framing shared with the CPU source-qualification receiver.
pub fn read_frame(stream: &mut std::os::unix::net::UnixStream) -> io::Result<Option<Frame>> {
    let mut header = [0; 4];
    if stream.read(&mut header[..1])? == 0 {
        return Ok(None);
    }
    stream.read_exact(&mut header[1..])?;
    let size = u32::from_be_bytes(header) as usize;
    if size == 0 || size > MAX_DEVICE_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid device frame length",
        ));
    }
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes)?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(io::Error::other)
}
pub fn write_answer(
    stream: &mut std::os::unix::net::UnixStream,
    answer: &Answer,
) -> io::Result<()> {
    write_frame(stream, answer)
}
fn scan(value: &Value) -> io::Result<()> {
    match value {
        Value::Object(fields) => {
            for (name, value) in fields {
                if ["token", "credential", "authorization", "secret", "jwt"]
                    .contains(&name.to_lowercase().as_str())
                {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "credential cannot cross device executor seam",
                    ));
                }
                scan(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                scan(value)?;
            }
        }
        _ => (),
    }
    Ok(())
}

#[cfg(test)]
mod codec_tests {
    use super::*;

    fn pid(codec: &Codec) -> Option<u32> {
        codec.helper.lock().unwrap().as_ref().map(|h| h.child.id())
    }

    #[test]
    #[ignore = "requires an explicitly configured installed SDK interpreter; no GPU imports"]
    fn one_sdk_encoder_serves_every_output_of_an_executor() {
        let python = PathBuf::from(std::env::var("COZY_MACHINE_CPU_TEST_PYTHON").unwrap());
        let root = std::env::temp_dir().join(format!("codec-{}", uuid::Uuid::new_v4().simple()));
        let spool = root.join("spool");
        fs::create_dir_all(&spool).unwrap();
        let codec = Codec::new(
            CodecConfig {
                python,
                environment: BTreeMap::new(),
                generation_hold: None,
            },
            root.join("codec.log"),
        );
        let reply = |index: usize, raw_bytes: u64| {
            let name = format!("image-{index:04}");
            fs::write(
                spool.join(format!("{name}.raw")),
                vec![index as u8 * 40; 4 * 2 * 3],
            )
            .unwrap();
            Frame {
                frames: vec![HostFrame {
                    handle: format!("asset/image/{index:04}"),
                    codec: "png".into(),
                    raw: format!("{name}.raw"),
                    raw_bytes,
                    media_type: "image/png".into(),
                    facts: BTreeMap::from([
                        ("width".into(), 4.into()),
                        ("height".into(), 2.into()),
                    ]),
                }],
                outputs: vec![Output {
                    output_id: format!("out-{index}"),
                    asset_ref: format!("asset/image/{index:04}"),
                    kind: "image".into(),
                    media_type: "image/png".into(),
                    size_bytes: None,
                    digest: String::new(),
                }],
                ..Default::default()
            }
        };
        let first = codec.encode(&spool, &reply(1, 24)).unwrap();
        let helper = pid(&codec);
        let second = codec.encode(&spool, &reply(2, 24)).unwrap();
        assert_eq!(
            pid(&codec),
            helper,
            "one encoder process serves the executor's outputs"
        );
        for (post, name) in [(first, "image-0001"), (second, "image-0002")] {
            let bytes = fs::read(spool.join(name)).unwrap();
            assert_eq!(post.bindings[0].name, name);
            assert_eq!(post.bindings[0].media_type, "image/png");
            assert_eq!(
                post.bindings[0].sha256,
                tensorfs_core::sha256::hex_digest(&bytes)
            );
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
        // A refused output is that output's failure; the encoder keeps serving.
        assert!(codec.encode(&spool, &reply(3, 25)).is_err());
        assert_eq!(pid(&codec), helper);
        codec.encode(&spool, &reply(4, 24)).unwrap();
        drop(codec);
        fs::remove_dir_all(root).unwrap();
    }
}
