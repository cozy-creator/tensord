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
    sync::Arc,
};

pub const MAX_DEVICE_FRAME: usize = 64 * 1024;
pub const RESULT_DOCUMENT: &str = "result.canonical";

#[derive(Clone, Debug)]
pub struct ExecutorConfig {
    pub python: PathBuf,
    pub root: PathBuf,
    pub socket: PathBuf,
    /// Explicit configuration only. Credentials never enter this child.
    pub environment: BTreeMap<String, String>,
    pub generation_hold: Option<Arc<File>>,
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
    },
    Budget {
        vram_bytes: i64,
        pinned_bytes: i64,
    },
    Prefetch {
        construction: String,
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
    HostTier,
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
    pub advance: u64,
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
    pub descriptor: bool,
    pub sha256: String,
    pub length: u64,
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
            descriptor: false,
            sha256: String::new(),
            length: 0,
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
}
pub struct Baseline;
impl Services for Baseline {}

pub struct DeviceExecutor {
    child: Child,
    stream: std::os::unix::net::UnixStream,
    pub birth: ProcessBirth,
    pub hello: Hello,
    root: PathBuf,
    socket: PathBuf,
    _generation_hold: Option<Arc<File>>,
    codec: CodecConfig,
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
    let post: PostReply = serde_json::from_slice(&output.stdout)?;
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
impl DeviceExecutor {
    pub fn spawn(config: ExecutorConfig) -> io::Result<Self> {
        let codec = CodecConfig {
            python: config.python.clone(),
            environment: config.environment.clone(),
            generation_hold: config.generation_hold.clone(),
        };
        fs::create_dir_all(&config.root)?;
        fs::set_permissions(&config.root, fs::Permissions::from_mode(0o700))?;
        let listener = UnixListener::bind(&config.socket)?;
        fs::set_permissions(&config.socket, fs::Permissions::from_mode(0o600))?;
        let mut command = Command::new(config.python);
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
        let mut child = command.spawn()?;
        let birth = process_birth(child.id())?;
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
        if credential.pid != child.id() as i32 || credential.uid != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "device executor connection differs from launched process",
            ));
        }
        let mut executor = Self {
            child,
            stream,
            birth,
            hello: Hello::default(),
            root: config.root,
            socket: config.socket,
            _generation_hold: config.generation_hold,
            codec,
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
        write_frame(&mut self.stream, command)?;
        loop {
            let frame = read_frame(&mut self.stream)?
                .ok_or_else(|| io::Error::other("stock executor EOF before reply"))?;
            match frame.event {
                Some(Event::Progress) => services.progress(&frame),
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
        let temporary = self.root.join("executor.cancel.pending");
        let mut file = File::create(&temporary)?;
        file.write_all(request_id.as_bytes())?;
        file.sync_all()?;
        fs::rename(temporary, self.root.join("executor.cancel"))?;
        File::open(&self.root)?.sync_all()
    }
    pub fn shutdown(mut self) -> io::Result<()> {
        self.command(&DeviceCommand::Shutdown, &mut Baseline)?;
        self.stream.shutdown(std::net::Shutdown::Both)?;
        let status = self.child.wait()?;
        fs::remove_file(&self.socket)?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "stock executor shutdown: {status}"
            )));
        }
        Ok(())
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
fn read_frame(stream: &mut std::os::unix::net::UnixStream) -> io::Result<Option<Frame>> {
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
