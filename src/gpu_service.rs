//! Trusted world-one device execution; acceptance, scheduling and custody stay in Engine.
use crate::{
    catalog::HeldGeneration,
    device_executor::{
        self, Answer, Binding, Budgets, DeviceCommand, DeviceExecutor, ExecutorConfig, Frame, Kind,
        Services,
    },
    execution::{process_ended, Engine},
    journal::{AssetBinding, Execution, Outcome, OutputChecksum, Preparation, State},
    model_sources::{ModelSources, SelectedManifest},
    shared_host_plane::{HostConfig, HostPeer, HostScope, SharedHostPlane},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, Weak,
    },
};
use tensorfs_core::store::Store;

fn measured_budget() -> i64 {
    -1
}
fn enabled() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceMode {
    #[default]
    Auto,
    Legacy,
    Descriptors,
}

#[derive(Clone, Debug, Deserialize)]
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

#[derive(Clone, Debug, Deserialize)]
pub struct HostOptions {
    pub budget_bytes: u64,
    pub readers: usize,
    pub max_entries: usize,
}

/// Root-sealed configuration, never peer-controlled paths or environment logic switches.
#[derive(Clone, Debug, Deserialize)]
pub struct GpuConfig {
    #[serde(default)]
    pub identity: Option<crate::launch_identity::LaunchIdentity>,
    #[serde(default)]
    pub source_mode: SourceMode,
    pub devices: String,
    pub authorized_device_limit_bytes: Option<u64>,
    #[serde(default = "measured_budget")]
    pub plane_budget_bytes: i64,
    pub pinned_budget_bytes: i64,
    #[serde(default = "enabled")]
    pub stages: bool,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// Explicit authority for already verified cached catalog bytes. Hub download/grant
    /// resolution is a separate operation; a cache hit alone grants no private model.
    pub models: Vec<ModelGrant>,
    #[serde(default)]
    pub packages: Vec<PublishedPackage>,
    #[serde(default)]
    pub host: Option<HostOptions>,
}
impl GpuConfig {
    pub fn load(path: &Path) -> io::Result<Self> {
        let config: Self = serde_json::from_reader(File::open(path)?).map_err(io::Error::other)?;
        if config.devices.is_empty() || config.devices.contains(',') || config.models.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GPU scope requires one configured device and cached-model authority",
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
    peer: HostPeer,
    descriptors: bool,
}
pub struct GpuPool {
    root: PathBuf,
    config: GpuConfig,
    store: Arc<Store>,
    reserved: AtomicBool,
    session: Mutex<Option<Session>>,
    host: Option<Arc<SharedHostPlane>>,
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
        let host = config
            .host
            .as_ref()
            .map(|options| {
                SharedHostPlane::new(
                    store.root(),
                    HostConfig {
                        budget_bytes: options.budget_bytes,
                        readers: options.readers,
                        max_entries: options.max_entries,
                    },
                )
            })
            .transpose()?;
        Ok(Arc::new(Self {
            launcher: crate::child_launcher::ChildLauncher::new()?,
            root: root.to_path_buf(),
            config,
            store,
            reserved: AtomicBool::new(false),
            session: Mutex::new(None),
            host,
        }))
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
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

    /// Derive a binding from the installed SDK's real static interface and configured
    /// cached byte authority. No model code executes during description/preparation.
    pub fn prepare_root(
        &self,
        actor: &str,
        installed: &crate::journal::Installation,
        entrypoint: &str,
        choices: &[crate::api::pb::ModelChoice],
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
        let grants: Vec<_> = self
            .config
            .models
            .iter()
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
                let record = engine.get(&id)?;
                if record.state == State::Starting && record.process.is_none() {
                    engine.defer_managed(&id, format!("device startup unavailable: {error}"))?;
                } else if record
                    .process
                    .as_ref()
                    .map(process_ended)
                    .transpose()?
                    .unwrap_or(false)
                {
                    engine.finish(
                        &id,
                        Outcome::Failed(format!("device executor ended: {error}")),
                    )?;
                }
                // An unquiesced live/unknown birth remains charged and nonterminal.
            }
            result
        })
    }
    pub fn stop(&self) -> io::Result<()> {
        if let Some(session) = self.session.lock().unwrap().take() {
            session.executor.shutdown()?;
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
        let mut slot = self.session.lock().unwrap();
        let result = self.run_locked(engine, id, held, plan, &mut slot);
        if result.is_err() {
            // Broken/failed exchanges close the owner channel. Sources and the
            // journal reservation survive until the exact receiver has exited.
            slot.take();
        }
        result
    }

    fn run_locked(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
        plan: GpuPlan,
        slot: &mut Option<Session>,
    ) -> io::Result<()> {
        if let Some(session) = slot.as_ref() {
            if process_ended(&session.executor.birth)? {
                slot.take();
            } else if session.plan != plan.id {
                // Closing the old context is observed before creating its replacement.
                slot.take().unwrap().executor.shutdown()?;
            }
        }
        let cold = slot.is_none();
        if cold {
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
            let mut environment = self.config.environment.clone();
            environment.insert("CUDA_VISIBLE_DEVICES".into(), self.config.devices.clone());
            let mut executor = DeviceExecutor::spawn_owned(
                ExecutorConfig {
                    python: held.record.python.clone(),
                    root: root.clone(),
                    socket,
                    environment,
                    generation_hold: Some(held.retention()),
                    identity: self.config.identity,
                },
                &self.launcher,
                |birth, cancel| {
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
            let descriptors = match self.config.source_mode {
                SourceMode::Legacy => false,
                SourceMode::Auto => executor.hello.offers("model_sources.descriptors/1"),
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
            let peer = HostPeer {
                actor: plan.actor.clone(),
                plan: plan.id.clone(),
                birth: executor.birth.clone(),
            };
            if let Some(host) = &self.host {
                host.register_peer(peer.clone(), executor.observer_pidfd()?)?;
                let sources = sources.lock().unwrap();
                host.authorize(
                    HostScope {
                        actor: plan.actor.clone(),
                        plan: plan.id.clone(),
                    },
                    plan.binding.snapshot.clone(),
                    sources.authorized_header(&plan.binding.snapshot)?,
                    plan.binding.components.clone(),
                )?;
            }
            *slot = Some(Session {
                plan: plan.id.clone(),
                loaded: false,
                executor,
                sources,
                budget_cells: vec![],
                peer,
                descriptors,
            });
        } else {
            let session = slot.as_ref().unwrap();
            let cancel = session.executor.cancellation();
            let request = id.to_string();
            engine.register_managed(
                id,
                session.executor.birth.clone(),
                Arc::new(move || cancel.cancel(&request)),
            )?;
        }
        if !engine.authorize_managed(id)? {
            engine.finish(id, Outcome::Canceled)?;
            return Ok(());
        }
        let session = slot.as_mut().unwrap();
        let stages = self.config.stages && session.executor.hello.offers("stage/1");
        let mut callbacks = Callbacks {
            engine,
            id,
            sources: &session.sources,
            cells: &mut session.budget_cells,
            budget: self.config.plane_budget_bytes,
            completed: 0,
            host: self.host.as_ref(),
            peer: &session.peer,
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
            command_ok(session.executor.command(
                &DeviceCommand::Load {
                    construction: plan.id.clone(),
                    devices: self.config.devices.clone(),
                    sequence_parallel_degree: 1,
                    binding: Box::new(binding),
                    budgets: Budgets {
                        declared_weight_bytes: plan.selected_encoded_bytes,
                    },
                    authorized_device_limit_bytes: self.config.authorized_device_limit_bytes,
                    attention_pin: String::new(),
                    host_tier: self.host.is_some()
                        && session.executor.hello.offers("host_tiers.owner/1"),
                    stages,
                    descriptor_sources: session.descriptors,
                    host_tier_owner: self.host.is_some()
                        && session.executor.hello.offers("host_tiers.owner/1"),
                },
                &mut callbacks,
            )?)?;
            if session.executor.hello.offers("weight_plane/1") {
                command_ok(session.executor.command(
                    &DeviceCommand::Budget {
                        vram_bytes: self.config.plane_budget_bytes,
                        pinned_bytes: self.config.pinned_budget_bytes,
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
            },
            &mut callbacks,
        )?;
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
        let reply = session.executor.command(
            &DeviceCommand::Invoke {
                request_id: id.into(),
                construction: plan.id.clone(),
                entrypoint: plan.entrypoint,
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: self.config.plane_budget_bytes,
                stages,
            },
            &mut callbacks,
        )?;
        if !reply.quiescent || !reply.poisoned.is_empty() {
            slot.take();
            return Err(io::Error::other("device reply lacks quiescence; wait for exact executor exit before releasing reservation"));
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
                        Outcome::Failed(format!("result custody failed: {error}")),
                    )?;
                }
            }
            "canceled" if engine.get(id)?.cancel_actor.is_some() => {
                engine.finish(id, Outcome::Canceled)?;
            }
            _ => {
                engine.finish(
                    id,
                    Outcome::Failed(format!("{}: {}", outcome.code, outcome.message)),
                )?;
            }
        }
        Ok(())
    }
}

fn output_bindings(bindings: Vec<device_executor::AssetBinding>) -> io::Result<Vec<AssetBinding>> {
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
    budget: i64,
    completed: u64,
    host: Option<&'a Arc<SharedHostPlane>>,
    peer: &'a HostPeer,
}
impl Services for Callbacks<'_> {
    fn progress(&mut self, frame: &Frame) {
        if frame.request_id.is_empty() || frame.request_id == self.id {
            self.completed = self.completed.saturating_add(frame.advance.max(1));
            let _ = self
                .engine
                .observe_progress(self.id, self.completed, frame.stage.clone());
        }
    }
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        if matches!(frame.kind, Kind::HostTier | Kind::HostTierPrepare) {
            if let Some(host) = self.host {
                return host.request(self.peer, frame, descriptor).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Unsupported,
                        "host-tier operation unavailable",
                    )
                })?;
            }
            drop(descriptor);
            let mut answer = Answer::unavailable(frame.seq);
            answer.ok = true;
            answer.code.clear();
            answer.detail.clear();
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
            Kind::StageEnter | Kind::StageExit => {
                drop(descriptor);
                answer.ok = true;
                answer.budget_bytes = self.budget;
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
