//! Existing Runtime device-executor control seam. Models and codecs remain in Python SDK.
use crate::{
    execution::{open_artifact, process_birth},
    journal::ProcessBirth,
    protocol,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            fs::{OpenOptionsExt, PermissionsExt},
            net::UnixListener,
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
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
    /// Explicit configuration only. Credentials never enter this child.
    pub environment: BTreeMap<String, String>,
    pub generation_hold: Option<Arc<File>>,
    pub identity: Option<crate::launch_identity::LaunchIdentity>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Hello {
    pub pid: u32,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        authorized_device_limit_bytes: Option<u64>,
        #[serde(skip_serializing_if = "String::is_empty")]
        attention_pin: String,
        host_tier: bool,
        stages: bool,
        #[serde(skip_serializing_if = "is_false")]
        descriptor_sources: bool,
        #[serde(skip_serializing_if = "is_false")]
        host_tier_owner: bool,
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
    HostTierPrepare,
    HostTier,
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
    #[serde(flatten)]
    pub hello: Hello,
    pub outcome: Option<Outcome>,
    pub quiescent: bool,
    pub poisoned: String,
    pub reused: bool,
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
    /// `device_tier`: fds that follow the frame, one per chunk of `regions`.
    pub descriptors: u32,
    pub device: String,
    pub regions: Vec<crate::resident_custody::SharedRegion>,
    pub shared_bytes: u64,
    pub released_bytes: u64,
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
    stream: std::os::unix::net::UnixStream,
    pub birth: ProcessBirth,
    pub hello: Hello,
    root: PathBuf,
    socket: PathBuf,
    _generation_hold: Option<Arc<File>>,
    codec: CodecConfig,
    exit: Option<File>,
    retained: Vec<Box<dyn Send>>,
    identity: Option<crate::launch_identity::LaunchIdentity>,
}

#[derive(Clone)]
pub struct Cancellation {
    root: PathBuf,
    identity: Option<crate::launch_identity::LaunchIdentity>,
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
        File::open(&self.root)?.sync_all()
    }
}

#[derive(Clone)]
pub struct CodecConfig {
    pub python: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub generation_hold: Option<Arc<File>>,
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
    frames: &'a [HostFrame],
    outputs: &'a [Output],
    max_output_bytes: Option<u64>,
}

pub fn postprocess(
    config: &CodecConfig,
    spool: &Path,
    reply: &Frame,
) -> io::Result<(Value, Vec<AssetBinding>)> {
    let result = read_result(spool, reply)?;
    let directory = File::open(spool)?;
    let post = match encode_native(spool, reply)? {
        Some(bindings) => PostReply { bindings },
        None => encode_with_sdk(config, spool, reply)?,
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
/// The selected SDK's post helper, one process per request: only for codecs this
/// machine does not encode itself (audio, video).
fn encode_with_sdk(config: &CodecConfig, spool: &Path, reply: &Frame) -> io::Result<PostReply> {
    let directory = File::open(spool)?;
    let fd = directory.as_raw_fd();
    let hold = config.generation_hold.as_ref().map(|hold| hold.as_raw_fd());
    let mut command = Command::new(&config.python);
    let log_name = format!("codec-{}.stderr.log", uuid::Uuid::new_v4());
    command
        .args([
            "-I",
            "-c",
            include_str!("../python/cozy_machine_client/device_codec.py"),
            "--spool-fd",
        ])
        .arg(fd.to_string())
        .env_clear()
        .envs(&config.environment)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(spool.join(&log_name))?,
        );
    // SAFETY: only async-signal-safe fcntl on retained directory/generation fds.
    unsafe {
        command.pre_exec(move || {
            for descriptor in std::iter::once(fd).chain(hold) {
                if libc::fcntl(descriptor, libc::F_SETFD, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = command.spawn()?;
    let request = PostRequest {
        frames: &reply.frames,
        outputs: &reply.outputs,
        max_output_bytes: reply.max_output_bytes,
    };
    child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("codec stdin missing"))?
        .write_all(&serde_json::to_vec(&request)?)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "selected SDK post helper failed: {}; see {log_name}",
            output.status
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(io::Error::other)
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
        let codec = CodecConfig {
            python: config.python.clone(),
            environment: config.environment.clone(),
            generation_hold: config.generation_hold.clone(),
        };
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
        let mut command = crate::launch_identity::trampoline(&config.python, config.identity)?;
        command
            .args(["-I", "-m", "cozy_runtime.internal.executor", "--socket"])
            .arg(&config.socket)
            .arg("--root")
            .arg(&config.root)
            .env_clear()
            .envs(config.environment)
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
        let mut child = match launcher {
            Some(launcher) => launcher.spawn(command)?,
            None => command.spawn()?,
        };
        let birth = process_birth(child.id())?;
        on_birth(
            &birth,
            &Cancellation {
                root: config.root.clone(),
                identity: config.identity,
            },
        )?;
        // Wait for connection OR observed process death. No elapsed-time startup kill.
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, child.id(), 0) } as i32;
        if raw < 0 {
            return Err(io::Error::last_os_error());
        }
        let pidfd = unsafe { File::from_raw_fd(raw) };
        let mut poll = [
            libc::pollfd {
                fd: listener.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: pidfd.as_raw_fd(),
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
                return Err(io::Error::other(format!(
                    "stock executor ended before connection: {}",
                    child.wait()?
                )));
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
        if credential.pid != child.id() as i32
            || credential.uid != expected_uid
            || credential.gid != expected_gid
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "device executor connection differs from launched process",
            ));
        }
        let mut executor = Self {
            child: Some(child),
            stream,
            birth,
            hello: Hello::default(),
            root: config.root,
            socket: config.socket,
            _generation_hold: config.generation_hold,
            codec,
            exit: Some(pidfd),
            retained: Vec::new(),
            identity: config.identity,
        };
        let hello = executor.command(&DeviceCommand::Hello, &mut Baseline)?;
        if !hello.ok || hello.hello.pid != executor.birth.pid {
            return Err(io::Error::other(
                "stock executor hello did not identify launched process",
            ));
        }
        executor.hello = hello.hello;
        Ok(executor)
    }
    pub fn codec(&self) -> CodecConfig {
        self.codec.clone()
    }
    pub fn root_path(&self) -> &Path {
        &self.root
    }
    pub fn cancellation(&self) -> Cancellation {
        Cancellation {
            root: self.root.clone(),
            identity: self.identity,
        }
    }

    pub fn observer_pidfd(&self) -> io::Result<File> {
        self.exit
            .as_ref()
            .ok_or_else(|| io::Error::other("device exit observer absent"))?
            .try_clone()
    }

    /// Retain a source/resource until this exact receiver exits, including owner-handle loss.
    /// Observer lifetimes must never own the DeviceExecutor handle.
    pub fn retain_until_exit(&mut self, resource: impl Send + 'static) {
        self.retained.push(Box::new(resource));
    }

    pub fn command(
        &mut self,
        command: &DeviceCommand,
        services: &mut impl Services,
    ) -> io::Result<Frame> {
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
            host_tier: true, ..
        } = command
        {
            if !self.hello.offers("weight_plane/1") {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "host weight tier capability absent",
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
        if let DeviceCommand::Load {
            host_tier_owner: true,
            host_tier,
            sequence_parallel_degree,
            ..
        } = command
        {
            if !*host_tier
                || *sequence_parallel_degree != 1
                || !self.hello.offers("host_tiers.owner/1")
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "machine host-tier registration requires qualified world-one capability",
                ));
            }
        }
        write_frame(&mut self.stream, command)?;
        loop {
            let frame = read_frame(&mut self.stream)?
                .ok_or_else(|| io::Error::other("stock executor EOF before reply"))?;
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
                    let descriptor = if frame.descriptor {
                        Some(File::from(protocol::recv_fd(&self.stream)?))
                    } else {
                        None
                    };
                    let (mut answer, descriptor) = services.request(&frame, descriptor)?;
                    if frame.kind == Kind::ModelSourceRead && answer.ok {
                        let source = descriptor.as_ref().ok_or_else(|| {
                            io::Error::other("successful model source omitted readonly file")
                        })?;
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
                }
                Some(Event::Unknown) => (),
                None if frame.reply == command.name() => return Ok(frame),
                None => return Err(io::Error::other("out-of-order stock executor reply")),
            }
        }
    }

    /// Attempt-keyed cooperative cancellation; observer teardown never calls this.
    pub fn cancel(&self, request_id: &str) -> io::Result<()> {
        self.cancellation().cancel(request_id)
    }
    pub fn shutdown(mut self) -> io::Result<()> {
        self.command(&DeviceCommand::Shutdown, &mut Baseline)?;
        self.stream.shutdown(std::net::Shutdown::Both)?;
        let status = self
            .child
            .as_mut()
            .ok_or_else(|| io::Error::other("device child handle absent"))?
            .wait()?;
        fs::remove_file(&self.socket)?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "stock executor shutdown: {status}"
            )));
        }
        Ok(())
    }
}

impl Drop for DeviceExecutor {
    fn drop(&mut self) {
        // Close the private owner channel, without writing the explicit cancel marker.
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        let Some(exit) = self.exit.take() else {
            return;
        };
        let held = Arc::new(Mutex::new(Some((
            exit,
            std::mem::take(&mut self.retained),
            self.child.take(),
        ))));
        let task = Arc::clone(&held);
        let monitor = move || {
            let Some((exit, resources, mut child)) = task.lock().unwrap().take() else {
                return;
            };
            if let Err(error) = wait_exit(&exit) {
                // An unobservable receiver is not authority to release its sources.
                eprintln!("receiver-exit observation failed; retaining sources: {error}");
                std::mem::forget((resources, child));
                return;
            }
            if let Some(child) = child.as_mut() {
                let _ = child.wait();
            }
        };
        if let Err(error) = std::thread::Builder::new()
            .name("device-source-custody".into())
            .spawn(monitor)
        {
            eprintln!("receiver monitor unavailable; observing exit synchronously: {error}");
            // The original Arc retains resources if thread creation discards its closure.
            if let Some((exit, resources, mut child)) = held.lock().unwrap().take() {
                if let Err(error) = wait_exit(&exit) {
                    eprintln!("receiver-exit observation failed; retaining sources: {error}");
                    std::mem::forget((resources, child));
                    return;
                }
                if let Some(child) = child.as_mut() {
                    let _ = child.wait();
                }
            }
        }
    }
}

fn wait_exit(exit: &File) -> io::Result<()> {
    let mut poll = libc::pollfd {
        fd: exit.as_raw_fd(),
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
            return Err(io::Error::other("receiver pidfd is not observable"));
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
