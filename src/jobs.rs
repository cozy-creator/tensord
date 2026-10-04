//! Jobs and their child runs. A job (`@app.job`) runs in a deviceless executor (`run_job`).
//! Each call it makes to one of its package's invocables arrives on the seam (`child_call`)
//! and becomes a run of its own, `<parent>/<call_index>`, under the parent's signer: same
//! records, dispatched like any run, canceled with the parent. A child's result and files
//! reach the parent through the seam (`CallState`), its files copied into the parent's spool.
//!
//! Only a job pauses. Its root stops (as a cancel stops it) and the run rests `paused`; its
//! unstarted children are held, started ones run to their end. Resume replays only the root:
//! each call it makes again finds its child by index and intent, finished children answer
//! with their results, held ones go on. The run's scratch and checkpoint declarations persist
//! across attempts; the self-managing sweep removes the scratch once the run has ended.
use crate::{
    catalog::HeldGeneration,
    device_executor::{
        self, Answer, Binding, Budgets, CallInterface, DeviceCommand, DeviceExecutor,
        ExecutorConfig, Frame, Kind, Services,
    },
    execution::Engine,
    gpu_service::{
        command_ok, keep_triage, output_bindings, record_measurements, settle, stage_inputs,
        WakeOnExit,
    },
    journal::{Execution, ExecutorFacts, Failure, InputFile, Outcome, State},
    launch_identity::{LaunchIdentity, Seal},
    runs::Runs,
    service::Service,
};
use serde_json::{json, Value};
use crate::objects::Refused;
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
    /// A job's weights sources and outputs (`weights_writer`).
    weights: crate::weights::Weights,
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

impl Parent {
    /// A call of this job has not ended: its executor is watched, and the root waits on it.
    fn unfinished(&self, engine: &Engine) -> bool {
        let calls = self.calls.lock().unwrap();
        calls.by_index.values().any(|call| {
            engine
                .get(&call.child)
                .is_ok_and(|record| !record.state.terminal())
        })
    }
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
    /// The machine's CPU execution, configured on `service`: jobs and model-less calls in
    /// deviceless executors, with the GPU executors' environment and identity when a GPU is
    /// configured, else a minimal one. `runs` prepares a job's children (none: no children).
    pub fn configure(
        service: &Arc<Service>,
        store: Arc<Store>,
        runs: Option<&Arc<Runs>>,
    ) -> io::Result<Arc<Self>> {
        let root = service
            .engine
            .root
            .parent()
            .ok_or_else(|| io::Error::other("the engine has no machine root"))?
            .join("cpu");
        let (environment, identity) = match service.gpu() {
            Some(gpu) => (gpu.config().environment.clone(), gpu.config().identity),
            None => {
                let home = root.join("home");
                fs::create_dir_all(&home)?;
                let home = home.to_string_lossy().into_owned();
                let minimal = [("PATH", "/usr/local/bin:/usr/bin:/bin"), ("LANG", "C.UTF-8"), ("HOME", &home)];
                (minimal.into_iter().map(|(k, v)| (k.into(), v.into())).collect(), None)
            }
        };
        let jobs = Self::new(&root, store, environment, identity, service, runs)?;
        service.configure_jobs(jobs.clone());
        Ok(jobs)
    }

    fn new(
        root: &Path,
        store: Arc<Store>,
        environment: BTreeMap<String, String>,
        identity: Option<LaunchIdentity>,
        service: &Arc<Service>,
        runs: Option<&Arc<Runs>>,
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
            store: store.clone(),
            service: Arc::downgrade(service),
            runs: runs.map_or_else(Weak::new, Arc::downgrade),
            parents: Mutex::new(HashMap::new()),
            weights: crate::weights::Weights::new(store)?,
        });
        // The startup sweep ran before jobs were configured: ended runs' scratch goes now.
        jobs.sweep_scratch(&service.engine);
        let (watching, engine) = (Arc::downgrade(&jobs), service.engine.clone());
        std::thread::Builder::new()
            .name("job-nudges".into())
            .spawn(move || nudge(watching, engine))?;
        Ok(jobs)
    }

    /// Whether this record runs here: a job, or a call that needs no GPU.
    pub fn takes(record: &Execution) -> bool {
        record.invocation.job
            || record
                .submission
                .as_ref()
                .is_none_or(|s| s.preparation_id.is_empty())
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
                false => jobs.call(&engine, &id, held),
            };
            let settled = match &result {
                Err(error) => settle(&engine, &id, error),
                Ok(()) => Ok(()),
            };
            if job {
                jobs.ended(&id);
            }
            settled.and(result)
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
                        jobs.stop_children(&request);
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
            engine.finish_stopped(id)?;
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
        let scratch = self.scratch(id)?;
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
        let (waiting, journal) = (parent.clone(), engine.clone());
        executor.waits = Some(Arc::new(move || waiting.unfinished(&journal)));
        let (grant, models) = self.weights_grant(engine, &record, &held.record.interface, &spool)?;
        let mut services = Seam {
            engine,
            id,
            store: &self.store,
            spool: &spool,
            completed: 0,
            publish: true,
            job: Some((self, &parent)),
            appended: HashMap::new(),
            weights: Some((&self.weights, &grant)),
        };
        let reply = executor.command(
            &DeviceCommand::RunJob {
                request_id: id.into(),
                job: record.invocation.entrypoint.clone(),
                payload: record.invocation.input.clone(),
                application: held.record.application.clone(),
                package_interface: interface,
                spool: spool.clone(),
                scratch,
                deadline_s: None,
                inputs,
                call_interfaces,
                models,
                weights: true,
            },
            &mut services,
        );
        self.parents.lock().unwrap().remove(id);
        conclude(engine, id, &executor, &spool, command_ok(reply?)?)?;
        executor.shutdown()
    }

    /// The job attempt's weights grant: its model inputs as sources (and as the executor's
    /// `models`), its declared weights outputs, and where they are published.
    fn weights_grant(
        &self,
        engine: &Arc<Engine>,
        record: &Execution,
        interface: &Value,
        spool: &Path,
    ) -> io::Result<(crate::weights::Grant, BTreeMap<String, Value>)> {
        let context = match self.runs.upgrade() {
            Some(runs) => runs.job_weights(&record.id)?,
            None => None,
        };
        let (inputs, destination) = context.map_or_else(Default::default, |c| (c.inputs, c.destination));
        let models = inputs
            .iter()
            .map(|(parameter, (class, manifest))| {
                (parameter.clone(), json!({"class": class, "manifest": manifest.id(), "length": manifest.length}))
            })
            .collect();
        let sources = inputs.values().map(|(_, m)| (m.id(), m.length)).collect();
        let outputs = interface["jobs"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == record.invocation.entrypoint.as_str()))
            .and_then(|row| row["weights_outputs"].as_array())
            .into_iter()
            .flatten()
            .filter_map(|row| Some((row["output_id"].as_str()?.to_string(), row["max_bytes"].as_u64()?)))
            .collect();
        let actor = record.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        let (journal, run) = (engine.clone(), record.id.clone());
        let current = Arc::new(move || {
            journal
                .get(&run)
                .is_ok_and(|r| r.state == State::Running && r.cancel_actor.is_none() && r.pause_actor.is_none())
        });
        let grant = crate::weights::Grant::new(&actor, &record.id, spool, sources, outputs, destination, current);
        Ok((grant, models))
    }

    /// A model-less call (a run's own, or a job's child) in a weightless construction.
    fn call(
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
            // A child's products show nothing; its parent decides.
            publish: record.invocation.parent.is_empty(),
            job: None,
            appended: HashMap::new(),
            weights: None,
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
                model_sources: false,
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
                device_weights: None,
            },
            &mut services,
        )?;
        conclude(engine, id, &executor, &spool, command_ok(reply)?)?;
        executor.shutdown()
    }

    /// The run's scratch (`<root>/scratch/<id>`), kept across its attempts.
    fn scratch(&self, id: &str) -> io::Result<PathBuf> {
        let root = self.root.join("scratch");
        fs::create_dir_all(&root)?;
        let scratch = root.join(id);
        fs::create_dir_all(&scratch)?;
        if let Some(identity) = self.identity {
            identity.traverse(&root)?;
            identity.own(&scratch)?;
        }
        Ok(scratch)
    }

    /// Scratch of runs that have ended (or are unknown); a paused run keeps its own.
    pub fn sweep_scratch(&self, engine: &Engine) -> usize {
        let Ok(entries) = fs::read_dir(self.root.join("scratch")) else {
            return 0;
        };
        let mut removed = 0;
        for entry in entries.flatten() {
            let id = entry.file_name().to_string_lossy().into_owned();
            let ended = match engine.get(&id) {
                Ok(record) => record.state.terminal(),
                Err(error) => error.kind() == io::ErrorKind::NotFound,
            };
            if ended {
                match fs::remove_dir_all(entry.path()) {
                    Ok(()) => removed += 1,
                    Err(error) => eprintln!("scratch of run {id}: {error}"),
                }
            }
        }
        removed
    }

    /// Whether `parent` is pausing or paused (not canceled): its children are held, not ended.
    fn holding(engine: &Engine, parent: &str) -> bool {
        engine.get(parent).is_ok_and(|record| {
            record.pause_actor.is_some()
                && record.cancel_actor.is_none()
                && !record.state.terminal()
        })
    }

    /// One child follows its parent's stop: held while unstarted if the parent pauses (started
    /// work runs to its end), else canceled.
    fn stop_child(engine: &Engine, child: &Execution, hold: bool) {
        let actor = child
            .submission
            .as_ref()
            .map(|s| s.actor.as_str())
            .unwrap_or_default();
        let stopped = match hold {
            true => engine.pause(&child.id, actor, true),
            false => engine.cancel(&child.id, actor),
        };
        if let Err(error) = stopped {
            eprintln!("child run {} of {} remains: {error}", child.id, child.invocation.parent);
        }
    }

    /// Every unfinished child of `parent` follows its stop (cancel, pause, or its end).
    fn stop_children(&self, parent: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        let hold = Self::holding(&service.engine, parent);
        let Ok(children) = service.engine.children(parent) else {
            return;
        };
        for child in &children {
            Self::stop_child(&service.engine, child, hold);
        }
    }

    /// The job's root has stopped: its children follow, and unless it rests paused its
    /// children prepare no more. A root deferred before it started runs again: nothing ends.
    fn ended(&self, id: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        if service.engine.get(id).is_ok_and(|r| r.state == State::Queued) {
            return;
        }
        self.stop_children(id);
        if let (false, Some(runs)) = (Self::holding(&service.engine, id), self.runs.upgrade()) {
            runs.end_job(id);
        }
    }

    /// Control's pause: only a job pauses.
    pub fn pause(&self, record: &Execution, actor: &str) -> Result<Execution, Refused> {
        let service = self.service()?;
        if !record.invocation.job {
            return Err(refused(
                "pause_unsupported",
                "only a job pauses; a call runs to its end or is canceled",
            ));
        }
        let paused = service.engine.pause(&record.id, actor, false)?;
        self.stop_children(&record.id);
        Ok(paused)
    }

    /// Control's resume: a paused job queues for a fresh attempt that replays its root.
    pub fn resume(&self, record: &Execution) -> Result<Execution, Refused> {
        let service = self.service()?;
        let resumed = service.engine.resume(&record.id)?;
        match resumed.state {
            _ if resumed.pause_actor.is_some() => Err(refused(
                "run_pausing",
                "the run is still pausing; resume it once it is paused",
            )),
            state if state.terminal() => Err(refused(
                "run_not_paused",
                format!("the run has {} and does not resume", state_name(state)),
            )),
            _ => Ok(resumed),
        }
    }

    /// A cancel reached a job with no root running (paused or queued): its children end too.
    pub fn canceled(&self, record: &Execution) {
        if record.invocation.job && record.state.terminal() {
            self.ended(&record.id);
        }
    }

    fn service(&self) -> Result<Arc<Service>, Refused> {
        self.service
            .upgrade()
            .ok_or_else(|| refused("machine_stopping", "the machine is stopping"))
    }

    /// `child_call`: the call's run, accepted once per index (its intent must not change).
    fn child_call(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
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
        // A pausing job starts nothing new; a resumed one's held call goes on.
        let engine = &service.engine;
        let held = match Self::holding(engine, &parent.id) {
            true => engine.pause(&record.id, &parent.actor, true),
            false if record.state == State::Paused || record.pause_actor.is_some() => {
                engine.resume(&record.id)
            }
            false => Ok(record.clone()),
        };
        held.map_err(|e| ("child_call_refused", e.to_string()))?;
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

    /// `checkpoint`: the declaration is journaled on the run (the bytes stay in its scratch or
    /// spool); a repeat replays its receipt, other content under the same keys is refused.
    fn checkpoint(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        if frame.logical_key.is_empty() || !frame.content_digest.starts_with("sha256:") {
            return Err((
                "checkpoint_invalid",
                "a checkpoint names a logical key and a sha256 content digest".into(),
            ));
        }
        let service = self
            .service
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let attempt = service.engine.get(&parent.id).map_or(0, |record| record.attempt);
        let (receipt, replayed) = service
            .engine
            .declare_checkpoint(
                &parent.id,
                attempt,
                &frame.operation_key,
                &frame.logical_key,
                &frame.content_digest,
                frame.length,
            )
            .map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => ("checkpoint_conflict", e.to_string()),
                _ => ("checkpoint_unrecorded", e.to_string()),
            })?;
        let mut answer = Answer::ok(frame.seq);
        answer.receipt_id = receipt;
        answer.replayed = replayed;
        Ok(answer)
    }

    fn seam(&self, parent: &Parent, frame: &Frame) -> io::Result<(Answer, Option<File>)> {
        let answered = match frame.kind {
            Kind::ChildCall => self.child_call(parent, frame),
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
                    if let Ok(record) = service.engine.get(&child) {
                        let hold = Self::holding(&service.engine, &parent.id);
                        Self::stop_child(&service.engine, &record, hold);
                    }
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
            Kind::Checkpoint => self.checkpoint(parent, frame),
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
    /// This attempt's list items per output: a resumed job's replay adds no duplicates.
    appended: HashMap<String, usize>,
    /// A job's weights broker and this attempt's grant.
    weights: Option<(&'a crate::weights::Weights, &'a crate::weights::Grant)>,
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
                crate::products::publish_replayed(
                    self.store,
                    self.engine,
                    self.id,
                    self.spool,
                    frame,
                    &mut self.appended,
                ),
                None,
            )),
            // A child call's product shows nothing (`sequence` 0): its parent decides.
            (Kind::Publish, _) => Ok((Answer::ok(frame.seq), None)),
            (Kind::StageEnter | Kind::StageExit, _) => Ok((Answer::ok(frame.seq), None)),
            (Kind::WeightsWriter, _) => {
                let Some((weights, grant)) = self.weights else {
                    return Ok((Answer::unavailable(frame.seq), None));
                };
                let (mut answer, channel, adopted) = weights.answer(grant, frame);
                if let Some(adopted) = adopted {
                    if let Err(error) = crate::products::record_manifest(self.engine, self.id, &adopted.output, &adopted.manifest) {
                        answer = Answer::refused(frame.seq, "publish_refused", error.to_string());
                    }
                }
                Ok((answer, channel))
            }
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
    record_measurements(engine, id, &reply);
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
        // child canceled with it); a pause's stop leaves it PAUSED.
        terminal if engine.get(id).is_ok_and(|r| {
            r.cancel_actor.is_some() || (terminal == "canceled" && r.pause_actor.is_some())
        }) =>
        {
            engine.finish_stopped(id)?;
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

fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

fn state_name(state: State) -> &'static str {
    match state {
        State::Completed => "succeeded",
        State::Failed => "failed",
        _ => "been canceled",
    }
}
