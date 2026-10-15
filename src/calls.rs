//! Child calls and each attempt's weights grant. Every call an attempt makes to a callable of
//! its environment arrives on the seam (`child_call`) and becomes a run of its own,
//! `<parent>/<call_index>`, under the parent's signer: same records, dispatched like any run,
//! canceled with the parent. A child's result and files reach the parent through the seam
//! (`state`), its files copied into the parent's spool. A memoized child's success is held when
//! it completes, whatever its parent then does: a parent canceled or failed and run again finds
//! its finished calls answered.
use crate::{
    catalog::HeldGeneration,
    device_executor::{Answer, CallInterface, Frame, Kind},
    execution::Engine,
    gpu_service::Plan,
    journal::{Execution, Failure, InputFile, State},
    launch_identity::LaunchIdentity,
    runs::Runs,
    service::Service,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::{self, Write},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
};
use tensorfs_core::store::Store;

pub struct Calls {
    identity: Option<LaunchIdentity>,
    service: Weak<Service>,
    /// Children prepare as runs do, with their parent's context.
    runs: Weak<Runs>,
    /// Running attempts that may call, by execution id.
    parents: Mutex<HashMap<String, Arc<Parent>>>,
    /// Every attempt's weights sources and outputs (`weights_writer`).
    weights: crate::weights::Weights,
}

/// A running attempt that calls: its signer, spool, callables and calls.
struct Parent {
    id: String,
    request: String,
    actor: String,
    spool: PathBuf,
    /// `(module, export)` of each invocable it may call: the App it belongs to (empty: its own
    /// package's; else a callee's in its environment) and the entrypoint registered for it.
    callables: Callables,
    /// It is itself memoized: each call it makes must be too, or replaying its result would
    /// skip that call's effects.
    pure: bool,
    /// The callables declared `memoize=True` (with or without an operation identity).
    pure_callables: std::collections::HashSet<(String, String)>,
    /// The parent's own file inputs: a child may be handed any of them.
    inputs: Vec<InputFile>,
    calls: Mutex<CallTable>,
}

impl Parent {
    /// The bytes done of the download the child call `request` is in now, of its total.
    fn child_bytes(&self, engine: &Engine, request: &str) -> Option<(u64, u64)> {
        let child = self.calls.lock().unwrap().by_index.values().find(|c| c.request == request)?.child.clone();
        let progress: Value = serde_json::from_str(&engine.get(&child).ok()?.progress?).ok()?;
        let total = progress["bytes_total"].as_u64().filter(|total| *total > 0)?;
        Some((progress["bytes_done"].as_u64()?.min(total), total))
    }

    fn callable(&self, frame: &Frame) -> Result<&(String, String), (&'static str, String)> {
        self.callables
            .get(&(frame.module.clone(), frame.export.clone()))
            .ok_or((
                "child_undeclared",
                format!(
                    "{}.{} is not an invocable of this package or its environment's",
                    frame.module, frame.export
                ),
            ))?
            .as_ref()
            .ok_or((
                "child_ambiguous",
                format!(
                    "{}.{} is registered by more than one App in this environment",
                    frame.module, frame.export
                ),
            ))
    }

    /// A call of this attempt has not ended: its executor is watched, and the parent waits on it.
    fn unfinished(&self, engine: &Engine) -> bool {
        let calls = self.calls.lock().unwrap();
        calls.by_index.values().any(|call| {
            engine
                .get(&call.child)
                .is_ok_and(|record| !record.state.terminal())
        })
    }
}

/// An attempt that may call: its calls' record, kept for the nudges while it lives.
pub(crate) struct Caller {
    calls: Arc<Calls>,
    parent: Arc<Parent>,
}
impl Caller {
    /// One of its attempt's call frames: a child call, poll, cancel, prefetch or GPU release.
    pub(crate) fn seam(&self, frame: &Frame) -> io::Result<(Answer, Option<File>)> {
        self.calls.seam(&self.parent, frame)
    }
    /// Its executor is still while a call it waits on runs: that call's executor is watched.
    pub(crate) fn waits(&self, engine: &Arc<Engine>) -> crate::process::Waits {
        let (parent, engine) = (self.parent.clone(), engine.clone());
        Arc::new(move || parent.unfinished(&engine))
    }
}
impl Drop for Caller {
    fn drop(&mut self) {
        self.calls.parents.lock().unwrap().remove(&self.parent.id);
    }
}

/// What a package's executor may call, sent on its Start (`interface`: its App's interface
/// beside the executor).
pub(crate) fn call_interfaces(held: &HeldGeneration, interface: &Path) -> Vec<CallInterface> {
    invocables(&held.record, &held.record.application, &held.record.interface, interface).0
}

#[derive(Default)]
struct CallTable {
    /// The parent's nudge socket (`child_events`): a byte when one of its calls may have moved.
    watcher: Option<UnixStream>,
    by_index: HashMap<u64, ChildCall>,
    /// Child files granted to the parent, by digest: later children may be handed them.
    received: HashMap<String, Received>,
}

struct ChildCall {
    child: String,
    /// The callee and the author's label for it, for the parent's `call` event.
    module: String,
    function: String,
    label: String,
    /// Its `call` event is in the parent's log.
    recorded: bool,
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

impl Calls {
    /// The machine's call broker; `runs` prepares children (none: no children).
    pub fn configure(
        service: &Arc<Service>,
        store: Arc<Store>,
        runs: Option<&Arc<Runs>>,
    ) -> io::Result<Arc<Self>> {
        let calls = Arc::new(Self {
            identity: service.pool().and_then(|pool| pool.config().identity),
            service: Arc::downgrade(service),
            runs: runs.map_or_else(Weak::new, Arc::downgrade),
            parents: Mutex::new(HashMap::new()),
            weights: crate::weights::Weights::new(store)?,
        });
        if let Err(error) = calls.sweep_canceled_outputs(&service.engine) {
            eprintln!("startup canceled native outputs: {error}");
        }
        let (watching, engine) = (Arc::downgrade(&calls), service.engine.clone());
        std::thread::Builder::new()
            .name("call-nudges".into())
            .spawn(move || nudge(watching, engine))?;
        service.configure_calls(calls.clone());
        Ok(calls)
    }

    /// `id`'s attempt waits on a call it made.
    pub(crate) fn waiting(&self, engine: &Engine, id: &str) -> bool {
        self.parents.lock().unwrap().get(id).is_some_and(|parent| parent.unfinished(engine))
    }

    pub(crate) fn weights(&self) -> &crate::weights::Weights {
        &self.weights
    }

    /// `record`'s attempt as a caller of its environment's callables (`document`: its App's
    /// interface, at `interface` beside its executor), and the call interfaces it is sent.
    pub(crate) fn caller(
        self: &Arc<Self>,
        record: &Execution,
        held: &HeldGeneration,
        document: &Value,
        interface: &Path,
        spool: &Path,
    ) -> (Caller, Vec<CallInterface>) {
        let (rows, callables) = invocables(&held.record, &record.invocation.module, document, interface);
        let submission = record.submission.as_ref();
        let parent = Arc::new(Parent {
            id: record.id.clone(),
            request: submission.map_or_else(|| record.id.clone(), |s| s.request_id.clone()),
            actor: submission.map(|s| s.actor.clone()).unwrap_or_default(),
            spool: spool.into(),
            callables,
            pure: declarations(document).any(|(declared, entry)| {
                entry["name"].as_str() == Some(record.invocation.entrypoint.as_str())
                    && declared["memoize"].as_bool() == Some(true)
            }),
            pure_callables: pure_callables(&held.record),
            inputs: record.invocation.inputs.clone(),
            calls: Mutex::new(CallTable::default()),
        });
        self.parents.lock().unwrap().insert(record.id.clone(), parent.clone());
        (Caller { calls: self.clone(), parent }, rows)
    }

    /// The attempt's weights grant: its plan's `Weights[M]` sources, the weights outputs its
    /// function declares, where they are published and how a source held in part fetches what
    /// it lacks. None when it reads and writes no weights.
    pub(crate) fn grant(
        &self,
        engine: &Arc<Engine>,
        record: &Execution,
        plan: &Plan,
        document: &Value,
        spool: &Path,
    ) -> io::Result<Option<crate::weights::Grant>> {
        let entry = document["entrypoints"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == record.invocation.entrypoint.as_str()))
            .cloned()
            .unwrap_or_default();
        let outputs: BTreeMap<String, u64> = entry["weights_outputs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|row| Some((row["output_id"].as_str()?.to_string(), row["max_bytes"].as_u64()?)))
            .collect();
        if plan.sources.is_empty() && outputs.is_empty() {
            return Ok(None);
        }
        let (destination, fetch) = match self.runs.upgrade() {
            Some(runs) => runs.weights_access(&record.id, plan)?,
            None => (None, None),
        };
        let sources = plan.sources.iter().map(|s| (s.manifest.clone(), s.length)).collect();
        let actor = record.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        let (run, attempt) = (record.id.clone(), record.attempt);
        let current = {
            let (engine, run) = (engine.clone(), run.clone());
            Arc::new(move || {
                engine.get(&run).is_ok_and(|r| {
                    r.attempt == attempt && r.state == State::Running && r.cancel_actor.is_none()
                })
            })
        };
        let bind_output = {
            let engine = engine.clone();
            Arc::new(move |transaction: &str, slot: &str| {
                engine.with_journal(|journal| journal.bind_derived_output(&run, attempt, transaction, slot))
            })
        };
        // A committed derivation is replayed for the same code, declaration and inputs (the
        // raw declaration the writer opens), never for its declaration alone.
        let replay = self.service.upgrade().and_then(|service| {
            let held = service.catalog.resolve(&record.invocation.generation).ok()?;
            let operation = crate::call_memo::operation(&held.record, &record.invocation.module, &entry["invocable"])?;
            let declared = serde_json_canonicalizer::to_vec(&entry).ok()?;
            Some(format!("{operation}\0{}", tensorfs_core::sha256::hex_digest(&declared)))
        });
        Ok(Some(crate::weights::Grant::new(
            &actor,
            &record.id,
            spool,
            sources,
            outputs,
            destination,
            crate::weights::GrantAuthority { current, bind_output, fetch, replay },
        )))
    }

    /// Write roots no unfinished run needs (`objects::Objects::sweep`).
    pub fn sweep_objects(&self) -> io::Result<usize> {
        match self.runs.upgrade() {
            Some(runs) => runs.objects.sweep(),
            None => Ok(0),
        }
    }

    /// Retry native cancellation cleanup from the existing durable execution associations.
    pub fn sweep_canceled_outputs(&self, engine: &Engine) -> io::Result<usize> {
        let bindings = engine.with_journal(|journal| journal.canceled_derived_outputs())?;
        let mut released = 0;
        let mut failure = None;
        for binding in bindings {
            let result = self
                .weights
                .cancel_output(&binding.transaction)
                .and_then(|()| {
                    engine.with_journal(|journal| journal.release_derived_output_binding(&binding))
                });
            match result {
                Ok(()) => released += 1,
                Err(error) => {
                    failure.get_or_insert_with(|| {
                        io::Error::other(format!(
                            "run {} output {}: {error}",
                            binding.execution, binding.slot
                        ))
                    });
                }
            }
        }
        failure.map_or(Ok(released), Err)
    }

    /// Every unfinished child of `parent` is canceled with it.
    pub(crate) fn stop_children(&self, parent: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        let Ok(children) = service.engine.children(parent) else {
            return;
        };
        for child in &children {
            cancel(&service.engine, child);
        }
    }

    /// A run's attempt has stopped: its children follow and prepare no more. A run that is
    /// queued again (deferred before it started) runs again and nothing ends.
    pub(crate) fn ended(&self, id: &str) {
        let Some(service) = self.service.upgrade() else {
            return;
        };
        if let Err(error) = self.sweep_canceled_outputs(&service.engine) {
            eprintln!("canceled native outputs: {error}");
        }
        if service.engine.get(id).is_ok_and(|r| r.state == State::Queued) {
            return;
        }
        self.stop_children(id);
        if let Some(runs) = self.runs.upgrade() {
            runs.end(id);
        }
    }

    /// A cancel reached a run: its unfinished native outputs are let go at once, and when no
    /// attempt runs (it was queued) its children end too.
    pub fn canceled(&self, record: &Execution) {
        if record.cancel_actor.is_none() {
            return;
        }
        if let Some(service) = self.service.upgrade() {
            if let Err(error) = self.sweep_canceled_outputs(&service.engine) {
                eprintln!("canceled native outputs: {error}");
            }
        }
        if record.state.terminal() {
            self.ended(&record.id);
        }
    }

    /// `child_call`: the call's run, accepted once per index (its intent must not change).
    fn child_call(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let service = self
            .service
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let (application, entrypoint) = parent.callable(frame)?;
        let callable = (frame.module.clone(), frame.export.clone());
        if parent.pure && !parent.pure_callables.contains(&callable) {
            return Err((
                "child_impure",
                format!(
                    "a memoized function cannot call {}.{}, which is not memoized",
                    frame.module, frame.export
                ),
            ));
        }
        let input: Value = crate::boundary_json::parse(frame.payload.as_bytes()).map_err(|e| {
            (
                "child_call_refused",
                format!("call request is not JSON: {e}"),
            )
        })?;
        // Every call is a serving call: its request, and the models it passes by value to the
        // callee's slots (null leaves a slot to the choice its parent carries for it).
        let (request_input, choices) = serving_call(&input, entrypoint)?;
        let runs = self
            .runs
            .upgrade()
            .ok_or(("child_call_refused", "machine is stopping".into()))?;
        let mut calls = parent.calls.lock().unwrap();
        let inputs = child_inputs(&request_input, parent, &calls.received);
        let request = format!("{}/{}", parent.request, frame.call_index);
        let intent = child_intent(&frame.module, &frame.export, &input);
        let caller = service
            .engine
            .get(&parent.id)
            .map_err(|e| ("child_call_refused", e.to_string()))?;
        let record = runs
            .child(&caller, &request, &intent, application, entrypoint, request_input, inputs, choices)
            .map_err(|refusal| match refusal.code {
                "run_id_conflict" => (
                    "child_call_refused",
                    "an existing call index changed its exact intent".to_string(),
                ),
                code => (code, refusal.message),
            })?;
        calls.by_index.entry(frame.call_index).or_insert(ChildCall {
            child: record.id.clone(),
            module: frame.module.clone(),
            function: frame.export.clone(),
            label: frame.progress_label.clone(),
            recorded: false,
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
        let CallTable {
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
                    let (text, rows, held) = receive(&service.engine.root, result, &parent.spool, &directory, self.identity)
                        .map_err(|e| ("child_result_unavailable", e.to_string()))?;
                    let settled = (text, rows);
                    // Each file is the signer's object too: a later child may be handed it.
                    let runs = self
                        .runs
                        .upgrade()
                        .ok_or(("child_call_refused", "machine is stopping".into()))?;
                    for (local, object) in &held {
                        runs.objects
                            .adopt(&parent.actor, &parent.id, local, object)
                            .map_err(|e| ("child_result_unavailable", e.to_string()))?;
                    }
                    for row in &settled.1 {
                        let digest = row["digest"].as_str().unwrap_or_default();
                        let length = row["length"].as_u64().unwrap_or_default();
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
                let failure = Failure::decode(record.failure.as_deref().unwrap_or_default())
                    .map_err(|error| ("failure_record_invalid", error.to_string()))?;
                answer.ok = false;
                answer.code = failure.code;
                answer.detail = failure.message.chars().take(1024).collect();
            }
            State::Canceled => {
                answer.ok = false;
                answer.code = "child_canceled".into();
                answer.detail = "the child run was canceled".into();
            }
            _ => answer.state = "pending".into(),
        }
        if record.state.terminal() && !call.recorded {
            call.recorded = record_call(&service.engine, parent, frame.call_index, call, &record);
        }
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
                        cancel(&service.engine, &record);
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
            // A hint about a future call: TensorD may prepare its weights and ask Runtime
            // to build the construction. This answer does not promise completed loading.
            Kind::ModelPrefetch => self.model_prefetch(parent, frame),
            Kind::GpuRelease => {
                let service = self.service.upgrade()
                    .ok_or_else(|| io::Error::other("the machine is stopping"))?;
                let record = service.engine.get(&parent.id)?;
                // Only a deviceless root ends its lane: a call on a GPU cannot give up its own
                // device safely, and a child cannot end its root's future GPU work.
                if let (true, Some(pool)) = (record.invocation.parent.is_empty() && record.invocation.degree == 0, service.pool()) {
                    pool.release(&service.engine, record.root_seq());
                }
                Ok(Answer::ok(frame.seq))
            }
            _ => return Ok((Answer::unavailable(frame.seq), None)),
        };
        Ok((
            answered.unwrap_or_else(|(code, detail)| Answer::refused(frame.seq, code, detail)),
            None,
        ))
    }

    fn model_prefetch(&self, parent: &Parent, frame: &Frame) -> Result<Answer, (&'static str, String)> {
        let (application, entrypoint) = parent.callable(frame)?;
        // Explicit choices use the ordinary child-call decoder and selection path; a hint
        // cannot turn a model name into byte authority.
        let raw = if frame.payload.is_empty() { "{}" } else { &frame.payload };
        let models = crate::boundary_json::parse(raw.as_bytes())
            .map_err(|error| ("child_call_refused", format!("prefetch choices are not JSON: {error}")))?;
        let choices = model_choices(&models, entrypoint)?;
        let runs = self.runs.upgrade().ok_or((
            "model_prefetch_unsupported", "machine does not prepare child models".into(),
        ))?;
        let service = self.service.upgrade().ok_or((
            "model_prefetch_unsupported", "machine is stopping".into(),
        ))?;
        let caller = service.engine.get(&parent.id)
            .map_err(|error| ("child_call_refused", error.to_string()))?;
        runs.prefetch(&caller, application, entrypoint, choices)
            .map_err(|error| (error.code, error.message))?;
        Ok(Answer::ok(frame.seq))
    }
}

/// A child follows its parent's stop: canceled.
fn cancel(engine: &Engine, child: &Execution) {
    let actor = child.submission.as_ref().map(|s| s.actor.as_str()).unwrap_or_default();
    if let Err(error) = engine.cancel(&child.id, actor) {
        eprintln!("child run {} of {} remains: {error}", child.id, child.invocation.parent);
    }
}

/// One progress frame of run `id`'s executor: a package log line, or its stage (with the
/// download bytes of the child call it relays, when it is one).
pub(crate) fn observe(engine: &Engine, id: &str, completed: &mut u64, frame: &Frame, caller: Option<&Caller>) {
    if !(frame.request_id.is_empty() || frame.request_id == id) {
        return;
    }
    if let Some((level, text)) = frame.log() {
        let _ = engine.append_package_log(id, level, &text);
        return;
    }
    if frame.stage.is_empty() {
        return;
    }
    // `advance` is the attempt's monotone position in completed units (Runtime
    // `Telemetry`); a zero one is telemetry, not work.
    *completed = (*completed).max(frame.advance);
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
    let call = frame.call_request.as_deref().filter(|call| !call.is_empty());
    if let Some((done, total)) = caller.zip(call).and_then(|(caller, call)| caller.parent.child_bytes(engine, call)) {
        (payload["bytes_done"], payload["bytes_total"]) = (done.into(), total.into());
    }
    let _ = engine.observe_progress(id, *completed, payload.to_string());
}

/// `(module, export)` of every invocable of the environment declared `memoize=True`.
fn pure_callables(generation: &crate::catalog::Generation) -> std::collections::HashSet<(String, String)> {
    std::iter::once(&generation.interface)
        .chain(generation.callees.iter().map(|c| &c.interface))
        .flat_map(declarations)
        .filter(|(declared, _)| declared["memoize"].as_bool() == Some(true))
        .filter_map(|(declared, _)| {
            Some((declared["module"].as_str()?.into(), declared["export"].as_str()?.into()))
        })
        .collect()
}

/// Each declared invocable of an interface: its declaration and its entry.
fn declarations(interface: &Value) -> impl Iterator<Item = (&Value, &Value)> {
    interface["entrypoints"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|entry| (&entry["invocable"], entry))
}

/// Every App this environment holds: its own exports, including internal ones, and the
/// other Apps' public exports. A callee can call its own App or another dependency too.
/// None names an export owned by different Apps: only that call is refused.
type Callables = HashMap<(String, String), Option<(String, String)>>;
fn invocables(
    generation: &crate::catalog::Generation,
    application: &str,
    own: &Value,
    path: &Path,
) -> (Vec<CallInterface>, Callables) {
    let mut rows = vec![];
    let mut callables: Callables = HashMap::new();
    // A callee module's typed callers need only its own callables: its first row carries that
    // part of its App's interface, so the whole App travels once, not once per export.
    let mut documented = std::collections::HashSet::new();
    let apps = std::iter::once((application, own, false))
        .chain(std::iter::once((generation.application.as_str(), &generation.interface, false)))
        .chain(generation.callees.iter().map(|c| {
            (c.application.as_str(), &c.interface, c.package == crate::catalog::BUILTIN_PACKAGE)
        }));
    for (owner, interface, builtin) in apps {
        let own = owner == application;
        for (declared, entry) in declarations(interface) {
            let (Some(module), Some(export), Some(name)) = (
                declared["module"].as_str(),
                declared["export"].as_str(),
                entry["name"].as_str(),
            ) else {
                continue;
            };
            // Another package's internal callable is its own business.
            if !own && entry["internal"].as_bool() == Some(true) {
                continue;
            }
            let key = (module.to_string(), export.to_string());
            if let Some(existing) = callables.get_mut(&key) {
                if existing.as_ref().is_some_and(|(app, _)| app != owner) {
                    *existing = None;
                }
                continue;
            }
            callables.insert(key, Some((owner.to_string(), name.to_string())));
            rows.push(CallInterface {
                module: module.into(),
                export: export.into(),
                interface_path: if own { path.into() } else { PathBuf::new() },
                self_call: own,
                // The Runtime binds its own operations from its own code, not a document.
                interface_document: (!own && !builtin && documented.insert(module.to_string()))
                    .then(|| module_interface(interface, module)),
                builtin: (builtin && !own).then(|| "operations".into()),
            });
        }
    }
    (rows, callables)
}

/// An App's interface narrowed to one module's exported callables (the Runtime generates that
/// module's callers from it).
fn module_interface(interface: &Value, module: &str) -> Value {
    let mut narrowed = interface.clone();
    if let Some(Value::Array(entries)) = narrowed.get_mut("entrypoints") {
        entries.retain(|entry| {
            entry["invocable"]["module"].as_str() == Some(module)
                && entry["internal"].as_bool() != Some(true)
        });
    }
    narrowed
}

/// A serving call's request and the model choices it carries: each `models` entry that is a
/// ModelArtifact becomes the exact checkpoint of the callee's slot `<entrypoint>.models.<name>`.
fn serving_call(
    call: &Value,
    entrypoint: &str,
) -> Result<(Value, Vec<crate::api::domain::ModelChoice>), (&'static str, String)> {
    let refuse = |why: &str| ("child_call_refused", format!("serving call {why}"));
    let request = call.get("payload").filter(|p| p.is_object()).ok_or_else(|| refuse("has no request object"))?;
    Ok((request.clone(), model_choices(&call["models"], entrypoint)?))
}

fn model_choices(
    models: &Value,
    entrypoint: &str,
) -> Result<Vec<crate::api::domain::ModelChoice>, (&'static str, String)> {
    let refuse = |why: &str| ("child_call_refused", format!("model choices {why}"));
    if !models.is_null() && !models.is_object() {
        return Err(refuse("must be an object"));
    }
    let mut choices = vec![];
    for (name, artifact) in models.as_object().into_iter().flatten() {
        if artifact.is_null() {
            continue;
        }
        let digest = artifact["manifest"]["digest"].as_str().and_then(|d| d.strip_prefix("sha256:"))
            .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| refuse(&format!("passes {name} without an exact checkpoint")))?;
        let digest = (0..64).step_by(2).map(|i| u8::from_str_radix(&digest[i..i + 2], 16).unwrap()).collect();
        let length = artifact["manifest"]["length"].as_u64().filter(|length| *length > 0)
            .ok_or_else(|| refuse(&format!("passes {name} without a positive checkpoint length")))?;
        choices.push(crate::api::domain::ModelChoice {
            parameter: format!("{entrypoint}.models.{name}"),
            manifest: Some(crate::api::domain::Ref {
                digest,
                length,
            }),
            ..Default::default()
        });
    }
    Ok(choices)
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
/// The files a parent now holds from a child's result, to adopt as its own objects.
type Held = Vec<(PathBuf, tensorfs_core::ids::ObjectRef)>;

fn receive(
    root: &Path,
    result: &crate::journal::ResultRecord,
    spool: &Path,
    directory: &Path,
    identity: Option<LaunchIdentity>,
) -> io::Result<(String, Vec<Value>, Held)> {
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
    let (mut grants, mut held) = (vec![], vec![]);
    // Each file the parent now holds, also in its spool as `sha256-<hex>`: it may return it
    // as its own output by the digest it names it by (`device_executor::tree_binding`).
    let mut hold = |local: &Path, sha: &str, length: u64| -> io::Result<()> {
        match fs::hard_link(local, spool.join(format!("sha256-{sha}"))) {
            Err(error) if error.kind() != io::ErrorKind::AlreadyExists => return Err(error),
            _ => (),
        }
        let object = tensorfs_core::ids::ObjectRef { sha256: sha.into(), length };
        held.push((local.to_path_buf(), object));
        Ok(())
    };
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
        hold(&local, &artifact.sha256, artifact.length)?;
        let digest = format!("sha256:{}", artifact.sha256);
        if binding.media_type == crate::gpu_service::TREE_MEDIA {
            // A tree: its manifest's members materialized as the directory the parent reads.
            let tree = directory.join(format!("{}-tree-{}", grants.len(), artifact.sha256));
            let manifest: Value = serde_json::from_slice(&fs::read(&local)?).map_err(io::Error::other)?;
            let mut content = 0;
            for entry in manifest["entries"].as_array().into_iter().flatten() {
                let (Some(path), Some(sha), Some(length)) = (entry["path"].as_str(),
                    entry["blob"]["sha256"].as_str(), entry["blob"]["length"].as_u64()) else {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "child tree manifest has an invalid entry"));
                };
                let relative = Path::new(path);
                if !relative.components().all(|c| matches!(c, std::path::Component::Normal(_))) {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "child tree member leaves its tree"));
                }
                let member = result.artifacts.iter().find(|a| a.sha256 == sha && a.length == length)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "child tree member is not held"))?;
                let target = tree.join(relative);
                fs::create_dir_all(target.parent().unwrap_or(&tree))?;
                fs::copy(root.join(&member.path), &target)?;
                if let Some(identity) = identity {
                    identity.readable(&target)?;
                }
                hold(&target, sha, length)?;
                content += length;
            }
            if let Some(identity) = identity {
                for entry in crate::gpu_service::walk_dirs(&tree)? {
                    identity.readable(&entry)?;
                }
            }
            node["asset_ref"] = digest.clone().into();
            node["digest"] = digest.clone().into();
            node["size_bytes"] = content.into();
            grants.push(json!({"output_id": output_id, "kind": "tree", "digest": digest, "local": tree,
                "length": artifact.length, "content_bytes": content, "media_type": binding.media_type}));
            continue;
        }
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
    let exact = String::from_utf8(crate::boundary_json::exact(&value)).map_err(io::Error::other)?;
    Ok((exact, grants, held))
}

/// A child call's intent: its callable and its input, numbers exact.
fn child_intent(module: &str, export: &str, input: &Value) -> String {
    let semantic = crate::boundary_json::exact(&json!([module, export, input]));
    format!("sha256:{}", tensorfs_core::sha256::hex_digest(&semantic))
}


/// One byte to every calling attempt's nudge socket each time anything moved: its calls re-poll.
fn nudge(calls: Weak<Calls>, engine: Arc<Engine>) {
    let mut seen = engine.activity_epoch();
    loop {
        seen = engine.wait_activity(seen, None);
        let Some(calls) = calls.upgrade() else {
            return;
        };
        if seen == u64::MAX {
            return;
        }
        for parent in calls.parents.lock().unwrap().values() {
            let calls = parent.calls.lock().unwrap();
            if let (Some(mut watcher), false) = (calls.watcher.as_ref(), calls.by_index.is_empty())
            {
                // A full socket already holds a nudge; one is enough.
                let _ = watcher.write(&[0]);
            }
        }
    }
}

/// Journals a weights output's publication as a `call` event on its run (`run show` lists it
/// beside the run's child calls), before the run settles. Keyed by output and transaction, so
/// a replay adds no second row.
pub(crate) fn record_upload(engine: &Engine, id: &str, upload: &crate::weights::Upload) {
    let body = json!({
        "request": format!("{id}/upload/{}", upload.output), "parent": id, "index": 0, "attempt": 1,
        "module": "cozy_runtime.author.publication", "export": "upload_checkpoint",
        "label": format!("Upload checkpoint to {}", upload.destination),
        "status": if upload.error.is_some() { "failed" } else { "succeeded" },
        "error": upload.error.clone().unwrap_or_default(),
        "error_code": if upload.error.is_some() { "weights_upload_failed" } else { "" },
        "called_unix_ms": upload.called_ms, "finished_unix_ms": upload.finished_ms, "measurements": Value::Null,
    });
    let key = format!("weights:{}/{}", upload.output, upload.transaction);
    if let Err(error) = engine.append_call(id, &key, &crate::boundary_json::exact(&body)) {
        eprintln!("run {id}: upload call: {error}");
    }
}

/// Journals a settled child as a `call` event on its parent: the callee, the author's label,
/// how it ended and when, and what its executor measured (`run show` lists each call). True
/// once it is in the parent's log, or the parent has ended and takes no more.
fn record_call(
    engine: &Engine,
    parent: &Parent,
    index: u64,
    call: &ChildCall,
    record: &Execution,
) -> bool {
    let (status, error_code, error) = match record.state {
        State::Completed => ("succeeded", String::new(), String::new()),
        State::Canceled => ("canceled", "child_canceled".into(), "the child run was canceled".into()),
        _ => match Failure::decode(record.failure.as_deref().unwrap_or_default()) {
            Ok(failure) => ("failed", failure.code, failure.message),
            Err(error) => {
                eprintln!("run {}: invalid child failure record: {error}", parent.id);
                return false;
            }
        },
    };
    let measurements = engine
        .with_journal(|journal| journal.measurements(&record.id))
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .unwrap_or(Value::Null);
    let body = json!({
        "request": call.request, "parent": parent.request, "index": index,
        "attempt": record.attempt.max(1), "module": call.module, "export": call.function,
        "label": call.label, "status": status, "error_code": error_code, "error": error,
        "called_unix_ms": record.accepted_at_ms, "finished_unix_ms": record.finished_at_ms,
        "measurements": measurements, "computation_digest": record.memo.computation,
        "memoized": record.memo.memoized, "unmemoized": record.memo.unmemoized,
    });
    match engine.append_call(&parent.id, &record.id, &crate::boundary_json::exact(&body)) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("run {}: call {index}: {error}", parent.id);
            false
        }
    }
}

#[cfg(test)]
mod prefetch_tests;
#[cfg(test)]
mod progress_tests;

#[cfg(test)]
mod exact_tests {
    use super::*;

    /// A serving call (the Runtime's typed caller of a callable) is its request plus models: a
    /// passed ModelArtifact is the exact checkpoint of the callee's slot, a null one leaves the
    /// slot to the callee's ladder.
    #[test]
    fn a_serving_call_unwraps_into_its_request_and_the_callees_slot_choices() {
        let hex = "ab".repeat(32);
        let call = json!({"payload": {"prompt": "a fox", "steps": 4}, "models": {
            "model": {"producer_request_id": "job-1/0", "output_slot": "weights",
                "manifest": {"digest": format!("sha256:{hex}"), "length": 812},
                "tensorfs_receipt_digest": format!("sha256:{hex}")},
            "vae": null}});
        let (request, choices) = serving_call(&call, "generate_image").unwrap();
        assert_eq!(request, json!({"prompt": "a fox", "steps": 4}));
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].parameter, "generate_image.models.model");
        let reference = choices[0].manifest.as_ref().unwrap();
        assert_eq!((tensorfs_core::sha256::hex(&reference.digest), reference.length), (hex, 812));
        assert!(choices[0].repository.is_empty());
        assert!(serving_call(&json!({"prompt": "a fox"}), "generate_image").is_err());
    }

    /// A callee App with many exports across modules: each module's first row carries only its
    /// own callables, so its Start fits the control frame where one full interface per export
    /// (cozy-eval 0.7.5: 8.7 KB on 9 rows) did not.
    #[test]
    fn a_callee_interface_travels_once_per_module() {
        let entry = |module: &str, name: &str| json!({"name": name, "description": "x".repeat(900),
            "invocable": {"module": module, "export": name}});
        let modules = ["eval.a", "eval.b", "eval.c", "eval.d", "eval.e", "eval.f"];
        let mut entries: Vec<Value> = modules.iter().flat_map(|m| [entry(m, &format!("{}_1", &m[5..])), entry(m, &format!("{}_2", &m[5..]))]).collect();
        entries.push(json!({"name": "hidden", "internal": true, "invocable": {"module": "eval.a", "export": "hidden"}}));
        let callee = json!({"application": "eval:app", "format": "interface/1", "entrypoints": entries});
        let own = json!({"entrypoints": [{"name": "root", "invocable": {"module": "root", "export": "root"}}]});
        let mut generation: crate::catalog::Generation = serde_json::from_value(json!({
            "identity": "a".repeat(32), "package": "root", "version": "1", "application": "root:app",
            "python": "/usr/bin/python3", "interface": own,
        })).unwrap();
        generation.callees.push(crate::catalog::Callee {
            distribution: "eval".into(), package: "local/eval".into(), version: "1".into(),
            application: "eval:app".into(), source_digest: String::new(), interface: callee.clone(),
        });
        let (rows, _) = invocables(&generation, "root:app", &own, Path::new("interface.json"));
        let documented: Vec<_> = rows.iter().filter_map(|r| Some((r.module.as_str(), r.interface_document.as_ref()?))).collect();
        assert_eq!(documented.iter().map(|(m, _)| *m).collect::<Vec<_>>(), modules);
        for (module, document) in &documented {
            let names: Vec<_> = document["entrypoints"].as_array().unwrap().iter().map(|e| e["invocable"]["module"].as_str().unwrap()).collect();
            assert_eq!(names, [*module, *module], "only its own exported callables");
            assert_eq!(document["application"], "eval:app");
        }
        let bytes: usize = rows.iter().map(|row| serde_json::to_vec(row).unwrap().len()).sum();
        assert!(bytes < crate::device_executor::MAX_DEVICE_FRAME, "{bytes} B");
    }

    #[test]
    fn a_shared_export_is_refused_only_when_different_apps_own_it() {
        let own = json!({"entrypoints":[
            {"name":"apply","invocable":{"module":"shared","export":"apply"}},
            {"name":"alias","invocable":{"module":"shared","export":"apply"}},
            {"name":"unrelated","invocable":{"module":"root","export":"unrelated"}}
        ]});
        let mut generation: crate::catalog::Generation = serde_json::from_value(json!({
            "identity":"a".repeat(32),"package":"root","version":"1","application":"root:app",
            "python":"/usr/bin/python3","interface":own,
        })).unwrap();
        let (_, calls) = invocables(&generation,"root:app",&own,Path::new("interface.json"));
        let shared = ("shared".into(),"apply".into());
        assert_eq!(calls[&shared], Some(("root:app".into(),"apply".into())));
        generation.callees.push(crate::catalog::Callee {
            distribution:"dep".into(),package:"second/dep".into(),version:"1".into(),application:"dep:app".into(),
            source_digest:String::new(),interface:json!({"entrypoints":[
                {"name":"apply","invocable":{"module":"shared","export":"apply"}}
            ]}),
        });
        let (_, calls) = invocables(&generation,"root:app",&own,Path::new("interface.json"));
        assert_eq!(calls[&shared], None);
        assert_eq!(calls[&("root".into(),"unrelated".into())], Some(("root:app".into(),"unrelated".into())));
        let root = std::env::temp_dir().join(format!("cm-ambiguous-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"),&root.join("generations"),1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let broker = Calls::configure(&service,store,None).unwrap();
        let parent = Parent { id:"parent".into(),request:"parent".into(),actor:"alice".into(),spool:root.clone(),
            callables:calls,pure:false,pure_callables:Default::default(),
            inputs:vec![],calls:Default::default() };
        let frame = Frame { module:"shared".into(),export:"apply".into(),..Default::default() };
        assert_eq!(broker.child_call(&parent,&frame).unwrap_err().0,"child_ambiguous");
        assert!(service.engine.nonterminal(usize::MAX).unwrap().is_empty());
        // A memoized function calls only memoized callables: replaying its result skips its calls.
        let unrelated = Frame { module:"root".into(),export:"unrelated".into(),payload:r#"{"payload":{}}"#.into(),..Default::default() };
        let pure = Parent { pure:true, ..parent };
        assert_eq!(broker.child_call(&pure,&unrelated).unwrap_err().0,"child_impure");
        let pure = Parent { pure_callables:[("root".into(),"unrelated".into())].into(), ..pure };
        assert_ne!(broker.child_call(&pure,&unrelated).unwrap_err().0,"child_impure");
        assert!(service.engine.nonterminal(usize::MAX).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_a_deviceless_root_releases_its_gpu_lane() {
        let root = std::env::temp_dir().join(format!("cm-release-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let config = serde_json::from_value(json!({"devices": "GPU-test-release", "prespawn": false})).unwrap();
        let pool = crate::gpu_service::Pool::new(&root.join("gpu"), config, store.clone()).unwrap();
        service.configure_pool(pool.clone()).unwrap();
        let broker = Calls::configure(&service, store, None).unwrap();
        let submit = |key: &str, parent: &str| service.engine.submit(key, crate::journal::Invocation {
            parent: parent.into(), ..Default::default()
        }).unwrap();
        let (job, next) = (submit("root", ""), submit("next", ""));
        let (_owner, _next_owner) = (pool.root(&service.engine, &job, 1).unwrap(), pool.root(&service.engine, &next, 1).unwrap());
        let lane = |record: &Execution| pool.root_lane(&service.engine, record.root_seq(), &mut [None]);
        assert!(lane(&job).is_some());
        assert!(lane(&next).is_none());
        let nested = submit("nested", &job.id);
        assert_eq!(nested.root_seq(), job.root_seq(), "a child is stamped with its root at acceptance");
        let mut parent = Parent {
            id: nested.id, request: "root".into(), actor: "alice".into(), spool: root.clone(),
            callables: Default::default(), pure: false, pure_callables: Default::default(),
            inputs: vec![], calls: Default::default(),
        };
        let release = Frame { kind: Kind::GpuRelease, seq: 7, ..Default::default() };
        assert!(broker.seam(&parent, &release).unwrap().0.ok);
        assert!(lane(&next).is_none(), "a child must not release its root's lane");
        parent.id = job.id;
        let (answer, file) = broker.seam(&parent, &release).unwrap();
        assert!(answer.ok && file.is_none());
        assert!(lane(&next).is_some(), "the live root's release lets the next root bind");
        pool.stop().unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    /// Seeds past 2^53 name distinct child calls and come back from a child exact.
    #[test]
    fn large_integers_stay_exact() {
        let seed = |text: &str| crate::boundary_json::parse(text.as_bytes()).unwrap();
        let intent = |input: &Value| child_intent("package", "call", input);
        let first = seed(r#"{"seed":9007199254740992,"nested":{"b":2,"a":1}}"#);
        assert_eq!(intent(&first), intent(&seed(r#" {"nested":{"a":1,"b":2}, "seed":9007199254740992} "#)));
        assert_ne!(intent(&first), intent(&seed(r#"{"seed":9007199254740993,"nested":{"b":2,"a":1}}"#)));
        let result = crate::journal::ResultRecord { value: json!({"seed": u64::MAX, "float": 1.0}), artifacts: vec![], asset_bindings: vec![] };
        let (sent, _, _) = receive(Path::new("unused"), &result, Path::new("unused"), Path::new("unused"), None).unwrap();
        let received = crate::boundary_json::parse(sent.as_bytes()).unwrap();
        assert_eq!(received["seed"].as_u64(), Some(u64::MAX));
        assert!(received["float"].is_f64());
    }
}
