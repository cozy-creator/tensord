//! Trusted world-one device execution; acceptance, scheduling and custody stay in Engine.
use crate::{
    catalog::HeldGeneration,
    device_executor::{
        self, Answer, Binding, Budgets, DeviceCommand, DeviceExecutor, ExecutorConfig, Frame, Kind,
        Services,
    },
    execution::{process_ended, Engine},
    host_tier::{HalfOfHeadroom, HostGrant, HostTier, HostTierConfig, SealedRequest},
    journal::{
        AssetBinding, Execution, Failure, Outcome, OutputChecksum, Preparation, ProcessBirth, State,
    },
    launch_identity::Seal,
    memory::{
        policy::{Facts, Holding, Step, MARGIN},
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
        Arc, Mutex, Weak,
    },
};
use tensorfs_core::store::Store;

fn alloc_conf() -> String {
    crate::launch_identity::DEFAULT_ALLOC_CONF.into()
}
fn threads() -> u32 {
    crate::launch_identity::DEFAULT_THREADS
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceMode {
    #[default]
    Auto,
    Legacy,
    Descriptors,
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
    #[serde(default)]
    pub source_mode: SourceMode,
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
}
impl GpuConfig {
    pub fn load(path: &Path) -> io::Result<Self> {
        let config: Self = serde_json::from_reader(File::open(path)?).map_err(io::Error::other)?;
        if config.devices.is_empty() || config.devices.contains(',') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GPU scope requires one configured device",
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuPlan {
    pub actor: String,
    pub id: String,
    pub installation: String,
    pub generation: String,
    pub entrypoint: String,
    pub binding: Binding,
    /// Native selected encoded source bytes; not a GPU-fit or full-memory charge.
    pub selected_encoded_bytes: u64,
}

struct Session {
    plan: String,
    loaded: bool,
    executor: DeviceExecutor,
    sources: Arc<Mutex<ModelSources>>,
    budget_cells: Vec<File>,
    actor: String,
    /// This executor in the host tier, the layouts it may adopt, and whether it adopts them.
    peer: u64,
    grants: Vec<HostGrant>,
    sealed: bool,
    descriptors: bool,
    /// Its weights stay on the GPU under custody (Degree 2), decided once at its load.
    sharing: bool,
}
pub struct GpuPool {
    root: PathBuf,
    /// Scopes JIT caches to this machine run; earlier runs' scopes are removed at start.
    incarnation: String,
    config: GpuConfig,
    store: Arc<Store>,
    reserved: AtomicBool,
    /// One retained executor per plan; the memory policy decides which keep weights mapped.
    sessions: Mutex<BTreeMap<String, Session>>,
    memory: GpuMemory,
    host: Arc<HostTier>,
    /// Degree 2: GPU weights kept across executors. None on a GPU that drives a display.
    custody: Option<Mutex<ResidentCustody>>,
    // Drop session/resource custody before ending the actual spawning thread.
    launcher: crate::child_launcher::ChildLauncher,
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
struct WakeOnExit(Weak<Engine>);
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
            Box::new(HalfOfHeadroom),
        )?;
        // Fill leases and GPU custody hold one descriptor per object or chunk.
        raise_fd_limit();
        let custody =
            (!display_active(&config.devices)).then(|| Mutex::new(ResidentCustody::default()));
        let incarnation = uuid::Uuid::new_v4().simple().to_string();
        crate::launch_identity::remove_stale_jit(root, &incarnation);
        remove_old_executor_roots(root);
        Ok(Arc::new(Self {
            launcher: crate::child_launcher::ChildLauncher::new()?,
            root: root.to_path_buf(),
            memory: GpuMemory::start(&config.devices, &config.memory),
            incarnation,
            config,
            store,
            reserved: AtomicBool::new(false),
            sessions: Mutex::new(BTreeMap::new()),
            host,
            custody,
        }))
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
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

    /// Derive a binding from the installed SDK's real static interface and the model the
    /// Hub resolved under the owner's access (or configured cached byte authority). No
    /// model code executes during description/preparation.
    pub fn prepare_root(
        &self,
        actor: &str,
        installed: &crate::journal::Installation,
        entrypoint: &str,
        choices: &[crate::api::pb::ModelChoice],
        resolved: Option<&ModelGrant>,
    ) -> io::Result<GpuPlan> {
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
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "model declarations absent")
            })?;
        if models.len() != 1 || choices.len() > 1 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "this GPU operation supports one declared model slot/world-one only",
            ));
        }
        let model = &models[0];
        let path = model
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "declared model path absent")
            })?;
        let prefix = format!("{entrypoint}.models.");
        let parameter = path
            .strip_prefix(&prefix)
            .filter(|v| !v.is_empty() && !v.contains('.'))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "declared model path is not a root parameter",
                )
            })?;
        let class = model
            .get("class")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "declared model class absent")
            })?;
        let chosen = choices.first();
        if let Some(choice) = chosen {
            if choice.parameter != path && choice.parameter != parameter {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "model choice does not name the declared root slot",
                ));
            }
            if !choice.source.is_empty()
                || !choice.profiles.is_empty()
                || !choice.adapters.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "provider sources/adapters need their own qualified model-resolution operation",
                ));
            }
        }
        let grants: Vec<_> = resolved
            .into_iter()
            .chain(self.config.models.iter().filter(|_| resolved.is_none()))
            .filter(|grant| {
                grant.package == installed.package
                    && grant.slot == path
                    && chosen.is_none_or(|choice| {
                        (choice.repository.is_empty() || choice.repository == grant.repository)
                            && (choice.release.is_empty() || choice.release == grant.release)
                            && (choice.lane.is_empty() || choice.lane == grant.lane)
                            && choice.manifest.as_ref().is_none_or(|reference| {
                                tensorfs_core::sha256::hex(&reference.digest)
                                    == grant.manifest.trim_start_matches("sha256:")
                            })
                    })
            })
            .collect();
        if grants.len() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "cached model selection has no unique configured authority",
            ));
        }
        let grant = grants[0];
        let mut wanted = std::collections::BTreeSet::new();
        if let Some(use_map) = model
            .get("component_use")
            .and_then(serde_json::Value::as_object)
        {
            for components in use_map.values() {
                for component in components.as_array().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "declared component use is not an array",
                    )
                })? {
                    wanted.insert(
                        component
                            .as_str()
                            .ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "declared component is not a name",
                                )
                            })?
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
                "declared components exceed authorized cached model scope",
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
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "declared application absent")
                })?
                .into(),
            model_class: class.into(),
            model_binding_path: path.into(),
            model_parameter_name: parameter.into(),
            model_parameter_names: vec![parameter.into()],
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
        let semantic = serde_json::json!({"actor":actor,"generation":installed.generation,"entrypoint":entrypoint,"binding":binding});
        let canonical = serde_json_canonicalizer::to_vec(&semantic).map_err(io::Error::other)?;
        Ok(GpuPlan {
            actor: actor.into(),
            id: format!("gpu-{}", tensorfs_core::sha256::hex_digest(&canonical)),
            installation: installed.alias.clone(),
            generation: installed.generation.clone(),
            entrypoint: entrypoint.into(),
            binding,
            selected_encoded_bytes,
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
        sealed: bool,
        took: std::time::Duration,
        loaded: &Frame,
        executor: u32,
    ) -> io::Result<()> {
        use crate::host_memory::{process, ProcessMemory};
        #[derive(Serialize)]
        struct Line<'a> {
            plan: &'a str,
            sealed: bool,
            load_ms: f64,
            facts: &'a Option<device_executor::LoadFacts>,
            host_tier: crate::host_tier::HostTierFacts,
            machine: ProcessMemory,
            executor: ProcessMemory,
        }
        let line = Line {
            plan,
            sealed,
            load_ms: took.as_secs_f64() * 1e3,
            facts: &loaded.facts,
            host_tier: self.host.facts(),
            machine: process(std::process::id()).unwrap_or_default(),
            executor: process(executor).unwrap_or_default(),
        };
        let mut bytes = serde_json::to_vec(&line)?;
        bytes.push(b'\n');
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("loads.jsonl"))?
            .write_all(&bytes)
    }
    pub fn stop(&self) -> io::Result<()> {
        let sessions = std::mem::take(&mut *self.sessions.lock().unwrap());
        for (plan, session) in sessions {
            session.executor.shutdown()?;
            self.memory.with(|gpu| gpu.ended(&plan));
        }
        Ok(())
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
        self.memory.finished(&key);
        if result.is_err() {
            // Broken/failed exchanges close the owner channel. Sources and the
            // journal reservation survive until the exact receiver has exited.
            sessions.remove(&key);
        }
        if !sessions.contains_key(&key) {
            self.memory.with(|gpu| gpu.ended(&key));
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
                        self.memory
                            .observe(plan, plane_facts(reply.plane.as_ref()), Some(false));
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
        self.memory.with(|gpu| gpu.ended(plan));
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
            self.memory.with(|gpu| gpu.ended(&key));
        }
        let cold = !sessions.contains_key(&plan.id);
        let mut load_cap = None;
        if cold {
            // A context and the first working set are reserved before the process exists.
            load_cap = self.memory.decide(
                &plan.id,
                true,
                || self.holdings(),
                |step| self.carry_out(step, sessions),
            )?;
            self.memory
                .with(|gpu| gpu.starting(&plan.id, gpu.spawn_need(&plan.id)));
            let root = self.root.join(uuid::Uuid::new_v4().simple().to_string());
            fs::create_dir(&root)?;
            let directory = File::open(&root)?;
            // Linux pathname limit is independent of the owned state directory length.
            let socket = if self.config.identity.is_some() {
                // Another UID cannot traverse this owner's /proc/fd magic link.
                // A deliberately short owned state root is required for this operation.
                let socket = root.join("executor");
                use std::os::unix::ffi::OsStrExt;
                if socket.as_os_str().as_bytes().len() > 107 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput,"configured identity needs a short owned executor socket path (Linux sun_path)"));
                }
                socket
            } else {
                PathBuf::from(format!(
                    "/proc/{}/fd/{}/executor",
                    std::process::id(),
                    directory.as_raw_fd()
                ))
            };
            let mut seal = Seal::prepare(
                &self.root,
                self.config.identity,
                &self.incarnation,
                &held.record.identity,
                &self.config.devices,
            )?;
            seal.alloc_conf = self.config.alloc_conf.clone();
            seal.threads = self.config.threads;
            let mut executor = DeviceExecutor::spawn_owned(
                ExecutorConfig {
                    python: held.record.python.clone(),
                    root: root.clone(),
                    socket,
                    environment: self.config.environment.clone(),
                    seal,
                    generation_hold: Some(held.retention()),
                    identity: self.config.identity,
                },
                &self.launcher,
                |birth, cancel| {
                    self.memory.with(|gpu| gpu.spawned(&plan.id, birth.pid));
                    let cancel = cancel.clone();
                    let request = id.to_string();
                    engine.register_managed(
                        id,
                        birth.clone(),
                        Arc::new(move || cancel.cancel(&request)),
                    )
                },
            )?;
            executor.retain_until_exit(directory);
            executor.retain_until_exit(WakeOnExit(Arc::downgrade(engine)));
            // Weights from the machine's sealed tier when the executor adopts them; its header
            // and configs then come from the store (descriptors would lease every object).
            let sealed = executor.hello.offers("weight_plane/1")
                && executor.hello.offers("host_tiers.sealed/1");
            let descriptors = match self.config.source_mode {
                SourceMode::Legacy => false,
                SourceMode::Auto => !sealed && executor.hello.offers("model_sources.descriptors/1"),
                SourceMode::Descriptors if executor.hello.offers("model_sources.descriptors/1") => {
                    true
                }
                SourceMode::Descriptors => {
                    executor.shutdown()?;
                    engine.finish(id, Outcome::Failed("requested descriptor-source operation is unavailable; automatic or legacy mode remains available".into()))?;
                    return Ok(());
                }
            };
            let sources = Arc::new(Mutex::new(ModelSources::open_shared(
                self.store.clone(),
                &[SelectedManifest {
                    manifest: plan.binding.snapshot.clone(),
                    components: plan.binding.components.clone(),
                }],
            )?));
            executor.retain_until_exit(sources.clone());
            let peer = self.host.register_peer(executor.observer_pidfd()?);
            let grants = vec![HostGrant {
                manifest: plan.binding.snapshot.clone(),
                header: sources
                    .lock()
                    .unwrap()
                    .authorized_header(&plan.binding.snapshot)?,
                components: plan.binding.components.iter().cloned().collect(),
            }];
            // Layouts this model had before refill while the executor imports and starts.
            if sealed {
                self.host.prefill(grants.clone());
            }
            sessions.insert(
                plan.id.clone(),
                Session {
                    plan: plan.id.clone(),
                    loaded: false,
                    executor,
                    sources,
                    budget_cells: vec![],
                    actor: plan.actor.clone(),
                    peer,
                    grants,
                    sealed,
                    descriptors,
                    sharing: false,
                },
            );
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
        // Out of the map for the call: its requests may unmap or end the others.
        let mut session = sessions.remove(&plan.id).expect("session retained above");
        match self.call(engine, id, &held, plan, load_cap, &mut session, sessions) {
            Ok(true) => {
                sessions.insert(session.plan.clone(), session);
                Ok(())
            }
            // Not reusable: gone (exit observed) before its context is released.
            Ok(false) => session.executor.terminate().map(drop),
            Err(error) => Err(ended_with(error, session.executor)),
        }
    }

    /// Load (once), grant and invoke. Ok(false): the executor must not be reused.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: &HeldGeneration,
        plan: GpuPlan,
        load_cap: Option<u64>,
        session: &mut Session,
        others: &mut BTreeMap<String, Session>,
    ) -> io::Result<bool> {
        let capped = session.executor.hello.offers("process_cap/1");
        if !session.loaded {
            // Degree 2 keeps every component resident until revoked, and a running call never
            // revokes what it reads: only when the whole construction and its activations fit
            // beside the other tenants. Unmeasured: off (Degree 1 lets stages evict each other).
            session.sharing = self.custody.is_some()
                && session.executor.hello.offers("weights.attach/1")
                && self.memory.fits_resident(&plan.id, || self.holdings());
        }
        let sharing = session.sharing;
        let mut callbacks = Callbacks {
            engine,
            id,
            sources: &session.sources,
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
            others,
            store: &self.store,
            spool: None,
        };
        if !session.loaded {
            let interface_path = session.executor.root_path().join("package-interface.json");
            fs::write(&interface_path, serde_json::to_vec(&held.record.interface)?)?;
            if let Some(identity) = self.config.identity {
                identity.readable(&interface_path)?;
            }
            command_ok(session.executor.command(
                &DeviceCommand::Start {
                    devices: self.config.devices.clone(),
                    application: held.record.application.clone(),
                    package_interface: interface_path.clone(),
                    sequence_parallel_degree: 1,
                    import_only: false,
                },
                &mut callbacks,
            )?)?;
            let mut binding = plan.binding.clone();
            binding.package_interface = interface_path.to_string_lossy().into();
            binding.store = if session.descriptors {
                String::new()
            } else {
                self.store.root().to_string_lossy().into()
            };
            let device_total = self.memory.sample().map(|sample| sample.total);
            let plane = session.executor.hello.offers("weight_plane/1");
            // The pinned budget comes with the Load, so no fill pins past it; an executor
            // without `load_pinned/1` gets it after, as before.
            let at_load = plane && session.executor.hello.offers("load_pinned/1");
            // Its share of the machine's pinned total, ahead of every other tenant.
            let pinned = self
                .memory
                .pinned_budgets(&plan.id)
                .and_then(|split| split.get(&plan.id).copied())
                .map_or(-1, |share| i64::try_from(share).unwrap_or(i64::MAX));
            let started = std::time::Instant::now();
            let loaded = command_ok(session.executor.command(
                &DeviceCommand::Load {
                    construction: plan.id.clone(),
                    devices: self.config.devices.clone(),
                    sequence_parallel_degree: 1,
                    binding: Box::new(binding),
                    budgets: Budgets {
                        declared_weight_bytes: plan.selected_encoded_bytes,
                    },
                    authorized_device_limit_bytes:
                        device_total.or(self.config.authorized_device_limit_bytes),
                    attention_pin: String::new(),
                    stages: false,
                    descriptor_sources: session.descriptors,
                    sealed_tiers: session.sealed,
                    pinned_bytes: at_load.then_some(pinned),
                    device_weights: sharing,
                    cap_bytes: load_cap.filter(|_| capped),
                },
                &mut callbacks,
            )?)?;
            self.memory
                .observe(&plan.id, load_facts(loaded.facts.as_ref()), Some(false));
            self.record_load(
                &plan.id,
                session.sealed,
                started.elapsed(),
                &loaded,
                session.executor.birth.pid,
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
        )?;
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
            engine.finish(id, Outcome::Failed(failure.encode()))?;
            return Ok(true);
        }
        command_ok(prepared)?;
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
        let cap = self.memory.decide(
            &plan.id,
            false,
            || self.holdings(),
            |step| self.carry_out(step, callbacks.others),
        )?;
        let (plane_budget_bytes, cap_bytes) = match cap {
            Some(cap) if capped => (-1, Some(cap)),
            Some(cap) if session.executor.hello.offers("weight_plane/1") => {
                let facts = self.memory.with(|gpu| gpu.facts(&plan.id));
                let context = facts
                    .context
                    .unwrap_or(self.memory.with(|gpu| gpu.context_estimate()));
                let plane = cap.saturating_sub(context + facts.activation.unwrap_or(0) + MARGIN);
                (i64::try_from(plane).unwrap_or(i64::MAX), None)
            }
            _ => (-1, None),
        };
        if let Some(cap) = cap {
            let cell = callbacks.cells.first().map(File::try_clone).transpose()?;
            self.memory.running(&plan.id, cap, cell);
        }
        callbacks.spool = Some(spool.clone());
        let reply = session.executor.command(
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
            },
            &mut callbacks,
        )?;
        let mut facts = plane_facts(reply.plane.as_ref());
        facts.activation = facts.activation.or_else(|| {
            reply
                .metrics
                .as_ref()
                .and_then(|m| m.activation_peak_bytes)
                .and_then(|v| u64::try_from(v).ok())
        });
        self.memory.observe(&plan.id, facts, Some(true));
        if let Some(plane) = &reply.plane {
            crate::memory::note(
                serde_json::json!({"event": "call", "plan": plan.id, "id": id,
                "cap": cap, "cap_bytes": plane.cap_bytes, "process": plane.process_bytes,
                "context": plane.context_bytes, "committed": plane.committed_bytes,
                "activation": plane.activation_peak_bytes, "oom_retries": plane.oom_retries,
                "evictions": plane.evictions, "h2d_bytes": plane.h2d_bytes}),
            );
        }
        if !reply.quiescent || !reply.poisoned.is_empty() {
            // The run ends once the executor is gone; its own reason travels with it.
            let reason = reply
                .outcome
                .as_ref()
                .map(|o| format!("{}: {}; ", o.code, o.message))
                .unwrap_or_default();
            return Err(io::Error::other(format!(
                "{reason}the executor did not settle quiescent"
            )));
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

    /// Idle tenants pinning more than their share of the host's pinned total give it back
    /// (their planes punch the least recently used regions; page cache and disk stay beneath)
    /// before `plan` runs. An idle executor's pinned tier is optional: a failure is noted.
    fn shed(&self, plan: &str, others: &mut BTreeMap<String, Session>) {
        let Some(split) = self.memory.pinned_budgets(plan) else {
            return;
        };
        for (other, session) in others.iter_mut() {
            let budget = self.memory.with(|gpu| gpu.facts(other).pinned_budget);
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
                    self.memory
                        .observe(other, plane_facts(reply.plane.as_ref()), None);
                    crate::memory::note(serde_json::json!({"event": "shed", "plan": other,
                        "pinned_budget": share, "for": plan}));
                }
                other_reply => crate::memory::note(serde_json::json!({"event": "shed_failed",
                    "plan": other, "detail": format!("{:?}", other_reply.map(|r| r.code))})),
            }
        }
    }

    /// An executor's out-of-memory retry asks for `free_bytes` from the other tenants: idle
    /// weights leave first, then idle processes; its cap rises into what they gave.
    fn room_for(
        &self,
        plan: &str,
        free_bytes: u64,
        others: &mut BTreeMap<String, Session>,
    ) -> io::Result<Option<u64>> {
        self.memory.make_room(
            plan,
            free_bytes,
            || self.holdings(),
            |step| self.carry_out(step, others),
        )
    }
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
fn ended_with(error: io::Error, executor: DeviceExecutor) -> io::Error {
    match executor.terminate() {
        Ok(device_executor::Ended {
            killed: Some(killed),
            ..
        }) => io::Error::other(format!("{error}; {killed}")),
        Ok(_) => error,
        Err(unproven) => io::Error::other(format!("{error}; executor exit unproven: {unproven}")),
    }
}

/// Every error ends the run once its executor is gone: FAILED with the reason, or CANCELED
/// when a cancel was journaled. Only a never-authorized attempt hit by a transient OS
/// shortage returns to the queue; a deterministic pre-start failure is FAILED.
fn settle(engine: &Arc<Engine>, id: &str, error: &io::Error) -> io::Result<()> {
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
    let outcome = if record.cancel_actor.is_some() {
        Outcome::Canceled
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
fn remove_old_executor_roots(root: &Path) {
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
        let sha = input
            .digest
            .strip_prefix("sha256:")
            .unwrap_or(&input.digest);
        let mut source = store
            .open_verified(sha)
            .map_err(io::Error::other)?
            .into_file();
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

fn command_ok(frame: Frame) -> io::Result<Frame> {
    if frame.ok {
        Ok(frame)
    } else {
        Err(io::Error::other(format!(
            "{}: {}",
            frame.code, frame.detail
        )))
    }
}
struct Callbacks<'a> {
    engine: &'a Arc<Engine>,
    id: &'a str,
    sources: &'a Arc<Mutex<ModelSources>>,
    cells: &'a mut Vec<File>,
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
        if frame.kind == Kind::ModelSourceRead {
            drop(descriptor);
            let (answer, file) =
                crate::model_source_driver::answer(&mut self.sources.lock().unwrap(), frame)?;
            return Ok((answer, Some(file)));
        }
        let mut answer = Answer::unavailable(frame.seq);
        match frame.kind {
            Kind::BudgetCell => {
                let file =
                    descriptor.ok_or_else(|| io::Error::other("budget cell descriptor absent"))?;
                if file.metadata()?.len() != 32 {
                    return Err(io::Error::other("budget cell layout differs"));
                }
                self.cells.push(file);
                answer.ok = true;
            }
            // Calls never ask for turns (Invoke sends `stages: false`); keep the budget.
            Kind::StageEnter | Kind::StageExit => {
                drop(descriptor);
                answer.ok = true;
            }
            Kind::DeviceRoom => {
                drop(descriptor);
                let cap = self
                    .pool
                    .room_for(self.plan, frame.free_bytes, self.others)?;
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
