//! Jobs and their child runs. A job (`@app.job`) runs in a deviceless executor (`run_job`).
//! Each call it makes to one of its package's invocables arrives on the seam (`child_call`)
//! and becomes a run of its own, `<parent>/<call_index>`, under the parent's signer: same
//! records, dispatched like any run, canceled with the parent. A child's result and files
//! reach the parent through the seam (`CallState`), its files copied into the parent's spool.
use crate::{
    catalog::HeldGeneration,
    device_executor::{
        self, Answer, Binding, Budgets, CallInterface, DeviceCommand, DeviceExecutor,
        ExecutorConfig, Frame, Kind, Services,
    },
    execution::Engine,
    gpu_service::{
        command_ok, keep_triage, output_bindings, settle, stage_inputs, WakeOnExit,
    },
    journal::{Execution, ExecutorFacts, Failure, InputFile, Outcome, State},
    launch_identity::{LaunchIdentity, Seal},
    runs::Runs,
    service::Service,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{self, Write},
    os::{fd::AsRawFd, unix::net::UnixStream},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};
use tensorfs_core::store::Store;

/// The weightless construction a model-less child is served from.
const WEIGHTLESS: &str = "weightless";

pub struct Jobs {
    root: PathBuf,
    incarnation: String,
    environment: BTreeMap<String, String>,
    identity: Option<LaunchIdentity>,
    store: Arc<Store>,
    service: Weak<Service>,
    /// Children prepare as runs do, with their job's context.
    runs: Weak<Runs>,
    /// Running jobs by execution id: what their child calls need.
    parents: Mutex<HashMap<String, Arc<Parent>>>,
}

/// A running job: its signer, spool, callables and calls.
struct Parent {
    id: String,
    request: String,
    actor: String,
    spool: PathBuf,
    /// `(module, export)` of each own invocable, and the entrypoint it is registered as.
    callables: HashMap<(String, String), String>,
    /// The parent's own file inputs: a child may be handed any of them.
    inputs: Vec<InputFile>,
    calls: Mutex<Calls>,
}

#[derive(Default)]
struct Calls {
    /// The parent's nudge socket (`child_events`): a byte when one of its calls may have moved.
    watcher: Option<UnixStream>,
    by_index: HashMap<u64, ChildCall>,
    /// Child files granted to the parent, by digest: later children may be handed them.
    received: HashMap<String, Received>,
}

struct ChildCall {
    child: String,
    request: String,
    /// The answered result and grants, once the child succeeded (materialized once).
    settled: Option<(String, Vec<Value>)>,
    progress: (u64, Option<String>),
}

#[derive(Clone)]
struct Received {
    length: u64,
    media_type: String,
}

impl Jobs {
    pub fn new(
        root: &Path,
        store: Arc<Store>,
        environment: BTreeMap<String, String>,
        identity: Option<LaunchIdentity>,
        service: &Arc<Service>,
        runs: &Arc<Runs>,
    ) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        let incarnation = uuid::Uuid::new_v4().simple().to_string();
        crate::launch_identity::remove_stale_jit(root, &incarnation);
        crate::gpu_service::remove_old_executor_roots(root);
        let jobs = Arc::new(Self {
            root: root.into(),
            incarnation,
            environment,
            identity,
            store,
            service: Arc::downgrade(service),
            runs: Arc::downgrade(runs),
            parents: Mutex::new(HashMap::new()),
        });
        let (watching, engine) = (Arc::downgrade(&jobs), service.engine.clone());
        std::thread::Builder::new()
            .name("job-nudges".into())
            .spawn(move || nudge(watching, engine))?;
        Ok(jobs)
    }

    /// Whether this record runs here: a job, or a child that needs no GPU.
    pub fn takes(record: &Execution) -> bool {
        let gpu = record
            .submission
            .as_ref()
            .is_some_and(|s| !s.preparation_id.is_empty());
        record.invocation.job || (!record.invocation.parent.is_empty() && !gpu)
    }

    pub fn dispatch(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        record: &Execution,
        held: HeldGeneration,
    ) -> io::Result<bool> {
        let (jobs, job) = (self.clone(), record.invocation.job);
        engine.dispatch_managed(&record.id, move |engine, id| {
            let result = match job {
                true => jobs.job(&engine, &id, held),
                false => jobs.child(&engine, &id, held),
            };
            if let Err(error) = &result {
                settle(&engine, &id, error)?;
            }
            result
        })
    }

    /// A deviceless executor for `id`, authorized to run; None when the run was canceled first.
    fn launch(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        id: &str,
        held: &HeldGeneration,
    ) -> io::Result<Option<DeviceExecutor>> {
        let root = self.root.join(uuid::Uuid::new_v4().simple().to_string());
        fs::create_dir(&root)?;
        let directory = File::open(&root)?;
        let socket = match self.identity {
            Some(_) => root.join("executor"),
            None => PathBuf::from(format!(
                "/proc/{}/fd/{}/executor",
                std::process::id(),
                directory.as_raw_fd()
            )),
        };
        let config = ExecutorConfig {
            python: held.record.python.clone(),
            root,
            socket,
            environment: self.environment.clone(),
            seal: Seal::prepare(
                &self.root,
                self.identity,
                &self.incarnation,
                &held.record.identity,
                "",
            )?,
            generation_hold: Some(held.retention()),
            identity: self.identity,
            cgroup_namespace: Some(crate::scope::namespace(
                self.root.parent().unwrap_or(&self.root),
            )),
        };
        let jobs = Arc::downgrade(self);
        let mut executor = DeviceExecutor::spawn_observed(config, |birth, cancel| {
            let (cancel, request) = (cancel.clone(), id.to_string());
            engine.register_managed(
                id,
                birth.clone(),
                Arc::new(move || {
                    let canceled = cancel.cancel(&request);
                    if let Some(jobs) = jobs.upgrade() {
                        jobs.end_children(&request);
                    }
                    canceled
                }),
            )
        })?;
        executor.retain_until_exit(directory);
        executor.retain_until_exit(WakeOnExit(Arc::downgrade(engine)));
        let facts = ExecutorFacts {
            pid: executor.birth.pid,
            runtime_version: executor.hello.runtime_version.clone(),
            tensorfs_version: executor.hello.tensorfs_version.clone(),
        };
        if !engine.authorize_managed(id, Some(facts))? {
            engine.finish(id, Outcome::Canceled)?;
            executor.shutdown()?;
            return Ok(None);
        }
        Ok(Some(executor))
    }

    fn interface(&self, executor: &DeviceExecutor, held: &HeldGeneration) -> io::Result<PathBuf> {
        let path = executor.root_path().join("package-interface.json");
        fs::write(&path, serde_json::to_vec(&held.record.interface)?)?;
        if let Some(identity) = self.identity {
            identity.readable(&path)?;
        }
        Ok(path)
    }

    fn spool(&self, executor: &DeviceExecutor, id: &str) -> io::Result<PathBuf> {
        let spool = executor.root_path().join(format!("output-{id}"));
        fs::create_dir(&spool)?;
        if let Some(identity) = self.identity {
            identity.own(&spool)?;
        }
        Ok(spool)
    }

    fn job(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
    ) -> io::Result<()> {
        let record = engine.get(id)?;
        let Some(mut executor) = self.launch(engine, id, &held)? else {
            return Ok(());
        };
        let interface = self.interface(&executor, &held)?;
        let spool = self.spool(&executor, id)?;
        let inputs = stage_inputs(&self.store, self.identity, &spool, &record.invocation.inputs)?;
        let (call_interfaces, callables) = own_invocables(&held.record.interface, &interface);
        let parent = Arc::new(Parent {
            id: id.into(),
            request: record
                .submission
                .as_ref()
                .map_or_else(|| id.to_string(), |s| s.request_id.clone()),
            actor: record
                .submission
                .as_ref()
                .map(|s| s.actor.clone())
                .unwrap_or_default(),
            spool: spool.clone(),
            callables,
            inputs: record.invocation.inputs.clone(),
            calls: Mutex::new(Calls::default()),
        });
        self.parents
            .lock()
            .unwrap()
            .insert(id.into(), parent.clone());
        let mut services = Seam {
            engine,
            id,
            store: &self.store,
            spool: &spool,
            completed: 0,
            publish: true,
            job: Some((self, &parent)),
        };
        let reply = executor.command(
            &DeviceCommand::RunJob {
                request_id: id.into(),
                job: record.invocation.entrypoint.clone(),
                payload: record.invocation.input.clone(),
                application: held.record.application.clone(),
                package_interface: interface,
                spool: spool.clone(),
                deadline_s: None,
                inputs,
                call_interfaces,
            },
            &mut services,
        );
        // Its children end with it, whatever it returned.
        self.end_children(id);
        self.parents.lock().unwrap().remove(id);
        if let Some(runs) = self.runs.upgrade() {
            runs.end_job(id);
        }
        conclude(engine, id, &executor, &spool, command_ok(reply?)?)?;
        executor.shutdown()
    }

    fn child(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
    ) -> io::Result<()> {
        let record = engine.get(id)?;
        let Some(mut executor) = self.launch(engine, id, &held)? else {
            return Ok(());
        };
        let interface = self.interface(&executor, &held)?;
        let spool = self.spool(&executor, id)?;
        let mut services = Seam {
            engine,
            id,
            store: &self.store,
            spool: &spool,
            completed: 0,
            publish: false,
            job: None,
        };
        let application = held.record.application.clone();
        command_ok(executor.command(
            &DeviceCommand::Start {
                devices: String::new(),
                application: application.clone(),
                package_interface: interface.clone(),
                sequence_parallel_degree: 1,
                import_only: false,
            },
            &mut services,
        )?)?;
        command_ok(executor.command(
            &DeviceCommand::Load {
                construction: WEIGHTLESS.into(),
                devices: String::new(),
                sequence_parallel_degree: 1,
                binding: Box::new(Binding {
                    application,
                    package_interface: interface.to_string_lossy().into_owned(),
                    ..Binding::default()
                }),
                budgets: Budgets::default(),
                models: vec![],
                authorized_device_limit_bytes: None,
                attention_pin: String::new(),
                stages: false,
                sealed_tiers: false,
                pinned_bytes: None,
                device_weights: false,
                cap_bytes: None,
            },
            &mut services,
        )?)?;
        command_ok(executor.command(
            &DeviceCommand::Activate {
                construction: WEIGHTLESS.into(),
            },
            &mut services,
        )?)?;
        let invocation = &record.invocation;
        let prepared = executor.command(
            &DeviceCommand::PrepareRequest {
                request_id: id.into(),
                construction: WEIGHTLESS.into(),
                entrypoint: invocation.entrypoint.clone(),
                payload: invocation.input.clone(),
                attention_kernel: String::new(),
                input_metadata: invocation
                    .inputs
                    .iter()
                    .map(|i| (i.input_id.clone(), json!({"input_id": i.input_id, "media_type": i.media_type, "digest": i.digest, "length": i.length, "order": i.order})))
                    .collect(),
            },
            &mut services,
        )?;
        if !prepared.ok {
            let terminal = match prepared.terminal.as_str() {
                "" => "refused",
                terminal => terminal,
            };
            let failure =
                Failure::executor(terminal, &prepared.origin, &prepared.code, &prepared.detail);
            engine.finish(id, Outcome::Failed(failure.encode()))?;
            return executor.shutdown();
        }
        let inputs = stage_inputs(&self.store, self.identity, &spool, &invocation.inputs)?;
        let reply = executor.command(
            &DeviceCommand::Invoke {
                request_id: id.into(),
                construction: WEIGHTLESS.into(),
                entrypoint: invocation.entrypoint.clone(),
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: -1,
                stages: false,
                cap_bytes: None,
                inputs,
                floor_bytes: None,
                activation_bytes: BTreeMap::new(),
            },
            &mut services,
        )?;
        conclude(engine, id, &executor, &spool, command_ok(reply)?)?;
        executor.shutdown()
    }

    /// Cancels every unfinished child of `parent` (its cancel, or its end).
    fn end_children(&self, parent: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        let Ok(records) = service.engine.nonterminal(usize::MAX) else {
            return;
        };
        for child in records.iter().filter(|r| r.invocation.parent == parent) {
            let actor = child
                .submission
                .as_ref()
                .map(|s| s.actor.as_str())
                .unwrap_or_default();
            if let Err(error) = service.engine.cancel(&child.id, actor) {
                eprintln!("child run {} of {parent} remains: {error}", child.id);
            }
        }
    }

    /// `child_call`: the call's run, accepted once per index (its intent must not change).
    fn call(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let service = self
            .service
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let entrypoint = parent
            .callables
            .get(&(frame.module.clone(), frame.export.clone()))
            .ok_or((
                "child_undeclared",
                format!(
                    "{}.{} is not an invocable of this package",
                    frame.module, frame.export
                ),
            ))?;
        let input: Value = crate::boundary_json::parse(frame.payload.as_bytes()).map_err(|e| {
            (
                "child_call_refused",
                format!("call request is not JSON: {e}"),
            )
        })?;
        if !input.is_object() {
            return Err((
                "child_call_refused",
                "call request must be an object".into(),
            ));
        }
        let runs = self
            .runs
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let mut calls = parent.calls.lock().unwrap();
        let inputs = child_inputs(&input, parent, &calls.received);
        let request = format!("{}/{}", parent.request, frame.call_index);
        let intent = format!(
            "sha256:{}",
            tensorfs_core::sha256::hex_digest(
                &serde_json_canonicalizer::to_vec(&json!([frame.module, frame.export, input]))
                    .map_err(|e| ("child_call_refused", e.to_string()))?
            )
        );
        let job = service
            .engine
            .get(&parent.id)
            .map_err(|e| ("child_call_refused", e.to_string()))?;
        let record = runs
            .child(&job, &request, &intent, entrypoint, input, inputs)
            .map_err(|refusal| match refusal.code {
                "run_id_conflict" => (
                    "child_call_refused",
                    "an existing call index changed its exact intent".to_string(),
                ),
                code => (code, refusal.message),
            })?;
        calls.by_index.entry(frame.call_index).or_insert(ChildCall {
            child: record.id.clone(),
            request: request.clone(),
            settled: None,
            progress: (0, None),
        });
        drop(calls);
        self.state(parent, frame)
    }

    /// `child_poll`: the call's state; a success carries its result and the parent's grants.
    fn state(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let service = self
            .service
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let mut calls = parent.calls.lock().unwrap();
        let Calls {
            by_index, received, ..
        } = &mut *calls;
        let call = by_index.get_mut(&frame.call_index).ok_or((
            "child_call_refused",
            "parent call has not been activated".into(),
        ))?;
        let record = service
            .engine
            .get(&call.child)
            .map_err(|e| ("child_call_refused", e.to_string()))?;
        let mut answer = Answer::ok(frame.seq);
        answer.child_request_id = call.request.clone();
        if record.progress != call.progress.1 && record.progress.is_some() {
            call.progress = (call.progress.0 + 1, record.progress.clone());
        }
        if let Some(detail) = &call.progress.1 {
            let mut payload = serde_json::from_str::<Value>(detail).unwrap_or_else(|_| json!({}));
            if let Some(fields) = payload.as_object_mut() {
                fields.insert("call_request".into(), call.request.clone().into());
                fields.insert("call_attempt".into(), record.attempt.into());
            }
            answer.progress = Some(json!({"sequence": call.progress.0, "payload": payload}));
        }
        match record.state {
            State::Completed => {
                if call.settled.is_none() {
                    let result = record
                        .result
                        .as_ref()
                        .ok_or(("child_failed", "child result is absent".into()))?;
                    let directory = parent
                        .spool
                        .join("child-results")
                        .join(frame.call_index.to_string());
                    let settled = grant(&service.engine.root, result, &directory, self.identity)
                        .map_err(|e| ("child_result_unavailable", e.to_string()))?;
                    // Each file is the signer's object too: a later child may be handed it.
                    let runs = self
                        .runs
                        .upgrade()
                        .ok_or(("child_call_refused", "machine is stopping".into()))?;
                    for row in &settled.1 {
                        let digest = row["digest"].as_str().unwrap_or_default();
                        let length = row["length"].as_u64().unwrap_or_default();
                        let object = tensorfs_core::ids::ObjectRef {
                            sha256: digest.trim_start_matches("sha256:").into(),
                            length,
                        };
                        let local = Path::new(row["local"].as_str().unwrap_or_default());
                        runs.objects
                            .adopt(&parent.actor, local, &object)
                            .map_err(|e| ("child_result_unavailable", e.to_string()))?;
                        received.insert(
                            digest.into(),
                            Received {
                                length,
                                media_type: row["media_type"].as_str().unwrap_or_default().into(),
                            },
                        );
                    }
                    call.settled = Some(settled);
                }
                let (result, grants) = call.settled.clone().expect("settled above");
                answer.state = "succeeded".into();
                answer.result = result;
                answer.byte_grants = grants;
            }
            State::Failed => {
                let message =
                    Failure::decode(record.failure.as_deref().unwrap_or_default()).message;
                let code = match message.split_once(": ") {
                    Some((code, _)) if !code.is_empty() && !code.contains(' ') => code.to_string(),
                    _ => "child_failed".into(),
                };
                answer.ok = false;
                answer.code = code;
                answer.detail = message.chars().take(1024).collect();
            }
            State::Canceled => {
                answer.ok = false;
                answer.code = "child_canceled".into();
                answer.detail = "the child run was canceled".into();
            }
            _ => answer.state = "pending".into(),
        }
        Ok(answer)
    }

    fn seam(&self, parent: &Parent, frame: &Frame) -> io::Result<(Answer, Option<File>)> {
        let answered = match frame.kind {
            Kind::ChildCall => self.call(parent, frame),
            Kind::ChildPoll => self.state(parent, frame),
            Kind::ChildCancel => {
                let call = parent
                    .calls
                    .lock()
                    .unwrap()
                    .by_index
                    .get(&frame.call_index)
                    .map(|c| c.child.clone());
                if let (Some(child), Some(service)) = (call, self.service.upgrade()) {
                    let _ = service.engine.cancel(&child, &parent.actor);
                }
                let mut answer = Answer::ok(frame.seq);
                answer.state = "pending".into();
                Ok(answer)
            }
            Kind::ChildForget => {
                let mut calls = parent.calls.lock().unwrap();
                let settled = calls.by_index.get(&frame.call_index).and_then(|c| {
                    let record = self.service.upgrade()?.engine.get(&c.child).ok()?;
                    Some(record.state.terminal())
                });
                if settled == Some(true) {
                    calls.by_index.remove(&frame.call_index);
                }
                Ok(Answer::ok(frame.seq))
            }
            Kind::ChildEvents => {
                let (ours, theirs) = UnixStream::pair()?;
                ours.set_nonblocking(true)?;
                parent.calls.lock().unwrap().watcher = Some(ours);
                return Ok((
                    Answer::ok(frame.seq),
                    Some(File::from(std::os::fd::OwnedFd::from(theirs))),
                ));
            }
            // The job will call it next: its models prepare and load now.
            Kind::ModelPrefetch => {
                let callable = (frame.module.clone(), frame.export.clone());
                if let (Some(entrypoint), Some(runs), Some(service)) = (
                    parent.callables.get(&callable),
                    self.runs.upgrade(),
                    self.service.upgrade(),
                ) {
                    if let Ok(job) = service.engine.get(&parent.id) {
                        runs.prefetch(&job, entrypoint);
                    }
                }
                Ok(Answer::ok(frame.seq))
            }
            // A deviceless parent holds no GPU.
            Kind::GpuRelease => Ok(Answer::ok(frame.seq)),
            _ => return Ok((Answer::unavailable(frame.seq), None)),
        };
        Ok((
            answered.unwrap_or_else(|(code, detail)| Answer::refused(frame.seq, code, detail)),
            None,
        ))
    }
}

/// Every executor exchange of one job or child run.
struct Seam<'a> {
    engine: &'a Arc<Engine>,
    id: &'a str,
    store: &'a Store,
    spool: &'a Path,
    completed: u64,
    /// A job's products show to its owner; a child's do not (its parent publishes).
    publish: bool,
    job: Option<(&'a Arc<Jobs>, &'a Arc<Parent>)>,
}
impl Services for Seam<'_> {
    fn progress(&mut self, frame: &Frame) {
        if !(frame.request_id.is_empty() || frame.request_id == self.id) || frame.stage.is_empty() {
            return;
        }
        self.completed = self.completed.saturating_add(frame.advance);
        let mut payload = json!({"stage": frame.stage.chars().take(120).collect::<String>(), "step_ms": frame.step_ms.unwrap_or(0.0)});
        for (name, value) in [
            ("stage_fraction", frame.stage_fraction.map(Value::from)),
            ("overall_fraction", frame.overall_fraction.map(Value::from)),
            ("position", frame.position.map(Value::from)),
            ("total", frame.total.map(Value::from)),
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
        drop(descriptor);
        match (frame.kind, self.job) {
            (Kind::Publish, _) if self.publish => Ok((
                crate::products::publish(self.store, self.engine, self.id, self.spool, frame),
                None,
            )),
            // A child call's product shows nothing (`sequence` 0): its parent decides.
            (Kind::Publish, _) => Ok((Answer::ok(frame.seq), None)),
            (Kind::StageEnter | Kind::StageExit, _) => Ok((Answer::ok(frame.seq), None)),
            (_, Some((jobs, parent))) => jobs.seam(parent, frame),
            _ => Ok((Answer::unavailable(frame.seq), None)),
        }
    }
}

/// A run's executor reply settles it: its result in custody, or its typed failure.
fn conclude(
    engine: &Arc<Engine>,
    id: &str,
    executor: &DeviceExecutor,
    spool: &Path,
    reply: Frame,
) -> io::Result<()> {
    let outcome = reply
        .outcome
        .as_ref()
        .ok_or_else(|| io::Error::other("executor outcome absent"))?;
    match outcome.terminal.as_str() {
        "succeeded" => {
            let custody = device_executor::postprocess(&executor.codec(), spool, &reply).and_then(
                |(value, bindings)| {
                    engine.managed_result(id, spool, value, output_bindings(bindings)?)
                },
            );
            if let Err(error) = custody {
                engine.finish(
                    id,
                    Outcome::Failed(Failure::custody(&error.to_string()).encode()),
                )?;
            }
        }
        // A journaled cancel ends it CANCELED, however it stopped (a job may first see its
        // child canceled with it).
        _ if engine.get(id)?.cancel_actor.is_some() => {
            engine.finish(id, Outcome::Canceled)?;
        }
        _ => {
            keep_triage(engine, id, executor, outcome);
            let failure = Failure::executor(
                &outcome.terminal,
                &outcome.origin,
                &outcome.code,
                &outcome.message,
            );
            engine.finish(id, Outcome::Failed(failure.encode()))?;
        }
    }
    Ok(())
}

/// The package's own invocables as the job's call interfaces, and the entrypoint each is.
fn own_invocables(
    interface: &Value,
    path: &Path,
) -> (Vec<CallInterface>, HashMap<(String, String), String>) {
    let mut rows = vec![];
    let mut callables = HashMap::new();
    for (section, kind) in [("jobs", "job"), ("entrypoints", "entrypoint")] {
        for entry in interface[section].as_array().into_iter().flatten() {
            let declared = &entry["invocable"];
            let (Some(module), Some(export), Some(name)) = (
                declared["module"].as_str(),
                declared["export"].as_str(),
                entry["name"].as_str(),
            ) else {
                continue;
            };
            if callables
                .insert((module.to_string(), export.to_string()), name.to_string())
                .is_none()
            {
                rows.push(CallInterface {
                    module: module.into(),
                    export: export.into(),
                    interface_path: path.into(),
                    self_call: true,
                    kind: kind.into(),
                });
            }
        }
    }
    (rows, callables)
}

/// The child's file inputs: every string of its request naming a file the parent holds (its
/// own inputs, or files its earlier children returned), by field path, as the SDK names them.
fn child_inputs(
    input: &Value,
    parent: &Parent,
    received: &HashMap<String, Received>,
) -> Vec<InputFile> {
    fn walk(
        value: &Value,
        path: &mut Vec<String>,
        order: u32,
        found: &mut Vec<(String, u32, String)>,
    ) {
        match value {
            Value::String(text) if text.starts_with("sha256:") => {
                found.push((path.join("."), order, text.clone()))
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    walk(item, path, index as u32, found);
                    path.pop();
                }
            }
            Value::Object(fields) => {
                for (name, item) in fields {
                    path.push(name.clone());
                    walk(item, path, order, found);
                    path.pop();
                }
            }
            _ => (),
        }
    }
    let mut found = vec![];
    walk(input, &mut vec![], 0, &mut found);
    found
        .into_iter()
        .filter_map(|(input_id, order, digest)| {
            let (length, media_type) = match parent.inputs.iter().find(|i| i.digest == digest) {
                Some(own) => (own.length, own.media_type.clone()),
                None => received
                    .get(&digest)
                    .map(|r| (r.length, r.media_type.clone()))?,
            };
            Some(InputFile {
                input_id,
                digest,
                length,
                media_type,
                order,
            })
        })
        .collect()
}

/// A completed child's result as its parent receives it: each file leaf named by its bytes
/// (`asset_ref` = `digest` = sha256) and granted at a copy in the parent's spool.
fn grant(
    root: &Path,
    result: &crate::journal::ResultRecord,
    directory: &Path,
    identity: Option<LaunchIdentity>,
) -> io::Result<(String, Vec<Value>)> {
    fn leaves(value: &Value, path: &mut Vec<String>, found: &mut Vec<String>) {
        match value {
            Value::Object(fields) if fields.contains_key("asset_ref") => found.push(path.join(".")),
            Value::Object(fields) => {
                for (name, item) in fields {
                    path.push(name.clone());
                    leaves(item, path, found);
                    path.pop();
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    leaves(item, path, found);
                    path.pop();
                }
            }
            _ => (),
        }
    }
    let mut value = result.value.clone();
    let mut paths = vec![];
    leaves(&value, &mut vec![], &mut paths);
    let mut grants = vec![];
    for output_id in paths {
        let pointer = format!("/{}", output_id.replace('.', "/"));
        let node = value
            .pointer_mut(if output_id.is_empty() { "" } else { &pointer })
            .ok_or_else(|| io::Error::other("result leaf vanished"))?;
        let reference = node["asset_ref"].as_str().unwrap_or_default().to_string();
        let binding = result
            .asset_bindings
            .iter()
            .find(|b| b.asset_ref == reference)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "child result file has no binding",
                )
            })?;
        let artifact = result
            .artifacts
            .iter()
            .find(|a| a.name == binding.relative_path)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "child result file is not held")
            })?;
        fs::create_dir_all(directory)?;
        let local = directory.join(format!("{}-{}", grants.len(), artifact.sha256));
        fs::copy(root.join(&artifact.path), &local)?;
        if let Some(identity) = identity {
            identity.readable(&local)?;
        }
        let digest = format!("sha256:{}", artifact.sha256);
        node["asset_ref"] = digest.clone().into();
        node["digest"] = digest.clone().into();
        node["size_bytes"] = artifact.length.into();
        node["media_type"] = binding.media_type.clone().into();
        grants.push(json!({
            "output_id": output_id,
            "kind": node["kind"],
            "digest": digest,
            "local": local,
            "length": artifact.length,
            "content_bytes": artifact.length,
            "media_type": binding.media_type,
        }));
    }
    let canonical = serde_json_canonicalizer::to_string(&value).map_err(io::Error::other)?;
    Ok((canonical, grants))
}


/// One byte to every running job's nudge socket each time anything moved: its calls re-poll.
fn nudge(jobs: Weak<Jobs>, engine: Arc<Engine>) {
    let mut seen = engine.activity_epoch();
    loop {
        seen = engine.wait_activity(seen, None);
        let Some(jobs) = jobs.upgrade() else {
            return;
        };
        if seen == u64::MAX {
            return;
        }
        for parent in jobs.parents.lock().unwrap().values() {
            let calls = parent.calls.lock().unwrap();
            if let (Some(mut watcher), false) = (calls.watcher.as_ref(), calls.by_index.is_empty())
            {
                // A full socket already holds a nudge; one is enough.
                let _ = watcher.write(&[0]);
            }
        }
    }
}
