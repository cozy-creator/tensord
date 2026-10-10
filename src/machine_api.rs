//! Deployed machine RPCs over the sole journal and authoritative TensorFS store.
use crate::{
    api::{
        auth::{Authority, VerifiedActor},
        domain, v1,
        MachineBackend,
    },
    gpu_service::{without_gpus, KeptMember, Level},
    journal::{Execution, PublicTerminal, State},
    service::Service,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use prost::Message;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{self, Read, Seek, SeekFrom},
    sync::{Arc, Mutex},
    time::Duration,
};
use tensorfs_core::{
    ids::ObjectRef,
    sha256,
    store::Store,
};
use tonic::Status;

pub struct NativeBackend {
    pub service: Arc<Service>,
    pub authority: Authority,
    pub store: Arc<Store>,
    pub installer: Option<crate::api::install::InstallerConfig>,
    pub publisher: Option<Arc<crate::published::Publisher>>,
    /// `cozy.machine.v1` Run sources and Write.
    pub runs: Option<Arc<crate::runs::Runs>>,
    // Serialize native projection, not inference or observation. Only one result
    // projection may establish a given immutable output's native custody at once.
    projection: Mutex<()>,
}
impl NativeBackend {
    pub fn new(
        service: Arc<Service>,
        authority: Authority,
        store: Arc<Store>,
    ) -> Self {
        Self {
            service,
            authority,
            store,
            installer: None,
            publisher: None,
            runs: None,
            projection: Mutex::new(()),
        }
    }
    fn workspace_id(&self) -> String {
        self.service.engine.workspace_id()
    }
    fn query(
        &self,
        actor: VerifiedActor,
        query: domain::MachineExecutionQuery,
    ) -> Result<Execution, Status> {
        if query.expected_execution_workspace_id != self.workspace_id() {
            return Err(refusal(
                "execution_workspace_changed",
                "the requested execution journal is not this workspace",
            ));
        }
        self.service
            .engine
            .get_public(&actor_id(actor), &query.request_id)
            .map_err(problem)
    }
    fn state(&self, record: &Execution) -> Result<domain::MachineExecutionState, Status> {
        let context = record
            .submission
            .as_ref()
            .ok_or_else(|| Status::internal("public submission context absent"))?;
        let sequence = self
            .service
            .engine
            .public_terminal(&record.id)
            .map_err(problem)?
            .map(|held| {
                crate::archive::decode_machine_execution_event_page(held.events.as_slice())
                    .map(|page| page.head_sequence)
                    .map_err(|_| Status::data_loss("durable event projection corrupt"))
            })
            .transpose()?
            .unwrap_or(record.revision)
            .max(record.revision);
        Ok(domain::MachineExecutionState {
            request_id: context.request_id.clone(),
            attempt_ordinal: record.attempt.max(1) as u64,
            generation: record.attempt as u64,
            collected: record.collected,
            state: match record.state {
                State::Completed => "succeeded",
                State::Failed => "failed",
                State::Canceled => "canceled",
                // Only Run (`cozy.machine.v1`) accepts a run before it is prepared.
                State::Queued
                    if record.waiting_reason.as_deref() == Some(crate::journal::PREPARING) =>
                {
                    "preparing"
                }
                State::Queued => "queued",
                // The worker protocol has no "starting": an attempt is queued until it runs.
                State::Starting => "queued",
                State::Running => "running",
                State::Paused => "paused",
                State::Unknown => "unknown",
            }
            .into(),
            sequence,
            worker_id: self.authority.worker_id.clone(),
            worker_boot_id: self.authority.boot_id.clone(),
            execution_workspace_id: self.workspace_id(),
            number: record
                .id
                .parse()
                .map_err(|_| Status::internal("invalid journal run number"))?,
            accepted_at_ms: record.accepted_at_ms,
            finished_at_ms: record.finished_at_ms,
            target: Some(domain::MachineExecutionTarget {
                package: record.invocation.package.clone(),
                entrypoint: record.invocation.entrypoint.clone(),
                installation_id: record.invocation.generation.clone(),
                ..Default::default()
            }),
        })
    }
    /// The run's journaled log: its published products, its memoized calls' results and its
    /// settled calls, as events with their journaled sequences.
    fn product_events(&self, record: &Execution) -> Result<Vec<domain::MachineExecutionEvent>, Status> {
        let attempt = record.attempt.max(1) as u64;
        let engine = &self.service.engine;
        let mut events = engine
            .products(&record.id)
            .map_err(problem)?
            .iter()
            .map(|stored| {
                let product = crate::products::decode(stored).map_err(problem)?;
                Ok(domain::MachineExecutionEvent {
                    sequence: stored.sequence,
                    attempt_ordinal: attempt,
                    at_ms: stored.at_ms,
                    kind: "product".into(),
                    body_canonical_bytes: product_document(&product)?,
                    product: Some(product),
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, Status>>()?;
        let calls = engine.calls(&record.id).map_err(problem)?.into_iter().map(|c| ("call", c));
        let logs = engine.logs(&record.id).map_err(problem)?.into_iter().map(|l| ("log", l));
        for (kind, stored) in calls.chain(logs) {
            events.push(domain::MachineExecutionEvent {
                sequence: stored.sequence,
                attempt_ordinal: attempt,
                at_ms: stored.at_ms,
                kind: kind.into(),
                body_canonical_bytes: stored.product,
                ..Default::default()
            });
        }
        events.sort_by_key(|event| event.sequence);
        Ok(events)
    }
    fn terminal(&self, record: &Execution) -> Result<domain::MachineExecutionEventPage, Status> {
        let _guard = self.projection.lock().unwrap();
        if let Some(held) = self
            .service
            .engine
            .public_terminal(&record.id)
            .map_err(problem)?
        {
            return crate::archive::decode_machine_execution_event_page(held.events.as_slice())
                .map_err(|_| Status::data_loss("durable event projection is corrupt"));
        }
        if !record.state.terminal() {
            return Err(Status::failed_precondition("execution is not terminal"));
        }
        let context = record
            .submission
            .as_ref()
            .ok_or_else(|| Status::internal("public submission context absent"))?;
        let mut value = record.result.as_ref().map(|r| r.value.clone());
        let mut products = vec![];
        let mut schema_digest = None;
        // A warm run's result describes its preparation; no callable produced it.
        if let Some(result) = record
            .result
            .as_ref()
            .filter(|_| !record.invocation.generation.is_empty())
        {
            let held = self
                .service
                .engine
                .installation_for_generation(&context.actor, &record.invocation.generation)
                .map_err(problem)?
                .ok_or_else(|| Status::data_loss("held installation interface absent"))?;
            let interface: Value = serde_json::from_slice(&held.interface)
                .map_err(|_| Status::data_loss("held installation interface corrupt"))?;
            let declaration =
                entrypoint(&interface, &record.invocation.entrypoint, record.invocation.job)?;
            schema_digest =
                Some(identity(declaration.get("result").ok_or_else(|| {
                    Status::failed_precondition("result schema absent")
                })?)?);
            let mut references = std::collections::HashMap::new();
            for binding in &result.asset_bindings {
                let index = result
                    .artifacts
                    .iter()
                    .position(|a| a.name == binding.relative_path)
                    .ok_or_else(|| Status::data_loss("output binding artifact absent"))?;
                let artifact = &result.artifacts[index];
                let object = ObjectRef {
                    sha256: artifact.sha256.clone(),
                    length: artifact.length,
                };
                let owner = identity(
                    &json!({"workspace":self.workspace_id(),"actor":context.actor,"request":context.request_id,"asset":binding.asset_ref}),
                )?;
                let mut source = self
                    .service
                    .engine
                    .open_result(&record.id, index)
                    .map_err(problem)?;
                verify_checksum(&mut source, binding)?;
                let native = crate::products::retain(
                    &self.store,
                    &self.service.engine,
                    &context.actor,
                    &owner,
                    &mut source,
                    &object,
                )
                .map_err(problem)?;
                // A tree's members are held too: Read serves them, a child may be handed it.
                if binding.media_type == crate::gpu_service::TREE_MEDIA {
                    let mut manifest = Vec::new();
                    self.service.engine.open_result(&record.id, index).map_err(problem)?
                        .read_to_end(&mut manifest).map_err(problem)?;
                    for (sha, length) in crate::execution::tree_members(&manifest).map_err(problem)? {
                        let member = result.artifacts.iter().position(|a| a.sha256 == sha && a.length == length)
                            .ok_or_else(|| Status::data_loss("a tree member is not held"))?;
                        let owner = identity(&json!({"workspace":self.workspace_id(),"actor":context.actor,
                            "request":context.request_id,"asset":binding.asset_ref,"member":sha}))?;
                        crate::products::retain(&self.store, &self.service.engine, &context.actor, &owner,
                            &mut self.service.engine.open_result(&record.id, member).map_err(problem)?,
                            &ObjectRef { sha256: sha, length }).map_err(problem)?;
                    }
                }
                references.insert(
                    binding.asset_ref.clone(),
                    (artifact.clone(), binding.media_type.clone(), native),
                );
            }
            if let Some(value) = &mut value {
                rewrite_assets(
                    value,
                    declaration
                        .get("result")
                        .ok_or_else(|| Status::data_loss("result schema absent"))?,
                    "",
                    domain::RunProductOp::Set,
                    0,
                    &references,
                    &mut products,
                )?;
            }
        }
        let output_entries = output_entries(&products)?;
        // The log's published products keep their sequences; the result publishes only what
        // the log does not already show, after them.
        let mut events = self.product_events(record)?;
        let shown: Vec<_> = events.iter().filter_map(|e| e.product.clone()).collect();
        // An observer that last saw it queued still sees it start before it ends.
        if record.running_revision > 0 {
            events.push(domain::MachineExecutionEvent {
                sequence: record.running_revision,
                attempt_ordinal: record.attempt.max(1) as u64,
                at_ms: record.started_at_ms,
                kind: "running".into(),
                body_canonical_bytes: canonical(&json!({"generation":record.attempt.max(1)}))?,
                ..Default::default()
            });
            events.sort_by_key(|event| event.sequence);
        }
        let first_sequence = record
            .revision
            .checked_add(1)
            .ok_or_else(|| Status::resource_exhausted("event cursor exhausted"))?;
        let fresh = products
            .into_iter()
            .filter(|product| !crate::products::shown(&shown, product));
        for (index, product) in fresh.enumerate() {
            events.push(domain::MachineExecutionEvent {
                sequence: first_sequence + index as u64,
                attempt_ordinal: record.attempt.max(1) as u64,
                at_ms: record.finished_at_ms,
                kind: "product".into(),
                body_canonical_bytes: product_document(&product)?,
                product: Some(product),
                ..Default::default()
            });
        }
        // The Python worker's run facts: which Runtime executed it, and for how long.
        let attempt = record.attempt.max(1) as u64;
        let mut facts = vec![];
        if let Some(executor) = record
            .executor
            .as_ref()
            .filter(|e| !e.runtime_version.is_empty())
        {
            facts.push(("executor", json!({"request":context.request_id,"attempt":attempt,"pid":executor.pid,"runtime_version":executor.runtime_version,"tensorfs_version":executor.tensorfs_version})));
        }
        let mut timing = json!({"attempt":attempt,"terminal":true});
        if record.started_at_ms > 0 && record.finished_at_ms >= record.started_at_ms {
            timing["execution_ms"] = json!((record.finished_at_ms - record.started_at_ms) as f64);
        }
        facts.push(("run.timing", timing));
        for (kind, body) in facts {
            events.push(domain::MachineExecutionEvent {
                sequence: first_sequence + events.len() as u64,
                attempt_ordinal: attempt,
                at_ms: record.finished_at_ms,
                kind: kind.into(),
                body_canonical_bytes: canonical(&body)?,
                ..Default::default()
            });
        }
        let mut body = json!({"format":"cozy.machine.outcome/1", "request_id":context.request_id, "attempt_ordinal":record.attempt.max(1), "invocation_spec_digest":context.invocation_digest});
        if record.process.is_some() {
            body["execution_started"] = json!(true);
        }
        let (status, code, origin, message, error_code) = match record.state {
            State::Completed => (1, 0, 2, "completed".to_string(), String::new()),
            State::Canceled => (4, 11, 6, "explicitly canceled".to_string(), "canceled".into()),
            _ => {
                let failure = crate::journal::Failure::decode(
                    record
                        .failure
                        .as_deref()
                        .unwrap_or_default(),
                ).map_err(problem)?;
                (
                    failure.status,
                    failure.cause,
                    failure.origin,
                    safe(&failure.message, 4096),
                    failure.code,
                )
            }
        };
        body["status"] = json!(status);
        body["cause"] = if code == 0 {
            json!({"origin":origin})
        } else {
            json!({"code":code,"origin":origin,"detail":safe(&message, 1024)})
        };
        body["safe_message"] = json!(message);
        if !error_code.is_empty() {
            body["error_code"] = json!(error_code);
        }
        // Observation only: the failed attempt's bundle, written before the run settled.
        if let Some((triage, _)) = self.service.engine.triage(&record.id).map_err(problem)? {
            body["triage_bundle"] = json!({"subject_id":triage.subject_id,
                "write_receipt_digest":format!("sha256:{}", triage.sha256),"length":triage.length});
        }
        if !output_entries.is_empty() {
            body["output_manifest"] = json!({"outputs":output_entries});
        }
        if let Some(value) = value {
            body["result"] = json!({"result_schema_digest":schema_digest, "inline_result":STANDARD.encode(crate::boundary_json::exact(&value))});
        }
        let bytes = canonical(&body)?;
        let digest = sha256::digest(&bytes);
        let outcome = domain::AttemptOutcome {
            worker_boot_id: record.acceptance_boot_id.clone(),
            request_id: context.request_id.clone(),
            attempt_ordinal: record.attempt.max(1) as u64,
            invocation_spec_digest: if record.canceled_before_acceptance {
                Vec::new() // cancellation accepted no invocation spec
            } else {
                digest_bytes(&context.invocation_digest)?
            },
            outcome_id: format!("out-{}", sha256::hex(&digest)),
            outcome_digest: digest.to_vec(),
            outcome_canonical_bytes: bytes,
        };
        let sequence = events
            .last()
            .map_or(record.revision, |event| event.sequence.max(record.revision))
            + 1;
        events.push(domain::MachineExecutionEvent {
            sequence,
            attempt_ordinal: outcome.attempt_ordinal,
            at_ms: record.finished_at_ms,
            kind: "outcome".into(),
            body_canonical_bytes: canonical(&json!({
                "state":match record.state { State::Completed => "completed", State::Canceled => "canceled", _ => "failed" },
                "outcome_id":outcome.outcome_id,
                "outcome_digest":format!("sha256:{}",sha256::hex(&outcome.outcome_digest))
            }))?,
            outcome: Some(outcome.clone()),
            ..Default::default()
        });
        let page = domain::MachineExecutionEventPage {
            events,
            next_after: sequence,
            head_sequence: sequence,
            compacted_through: record.revision,
        };
        let committed = self
            .service
            .engine
            .commit_public_terminal(
                &record.id,
                PublicTerminal {
                    outcome: crate::archive::encode_attempt_outcome(&outcome),
                    events: crate::archive::encode_machine_execution_event_page(&page),
                },
            )
            .map_err(problem)?;
        crate::archive::decode_machine_execution_event_page(committed.events.as_slice())
            .map_err(|_| Status::data_loss("durable event projection is corrupt"))
    }
}
/// Printable ASCII only, bounded: a reason is shown to the client verbatim.
fn safe(text: &str, limit: usize) -> String {
    text.chars()
        .map(|c| if (' '..='~').contains(&c) { c } else { ' ' })
        .take(limit)
        .collect()
}
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
impl MachineBackend for NativeBackend {
    fn runs(&self) -> Option<Arc<crate::runs::Runs>> {
        self.runs.clone()
    }
    fn open_output(
        &self,
        actor: VerifiedActor,
        run: u64,
        output: &str,
        index: Option<u32>,
    ) -> Result<crate::api::backend::OutputSnapshot, Status> {
        let absent = || Status::not_found("this machine has no such run or output");
        let actor = actor_id(actor);
        let record = match self.service.engine.get(&run.to_string()) {
            Ok(record) if record.submission.as_ref().is_some_and(|s| s.actor == actor) => record,
            Ok(_) => return Err(absent()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Err(absent()),
            Err(error) => return Err(problem(error)),
        };
        let terminal = record.state.terminal();
        let events = if terminal {
            self.terminal(&record)?.events
        } else {
            self.product_events(&record)?
        };
        let (op, position) = match index {
            Some(index) => (
                domain::RunProductOp::Append,
                index.checked_sub(1).ok_or_else(absent)?,
            ),
            None => (domain::RunProductOp::Set, 0),
        };
        let revisions: Vec<_> = events
            .into_iter()
            .filter_map(|event| event.product)
            .filter(|p| p.output == output && p.op == op as i32 && p.index == position)
            .collect();
        let current = revisions.last().ok_or_else(absent)?;
        let content = current
            .content
            .as_ref()
            .ok_or_else(|| Status::data_loss("product content absent"))?;
        let refs: Vec<&domain::Ref> = if current.parts.is_empty() {
            vec![content]
        } else {
            current
                .parts
                .iter()
                .filter_map(|p| p.content.as_ref())
                .collect()
        };
        let parts = refs
            .into_iter()
            .map(|r| {
                let file = self
                    .store
                    .open_verified(&sha256::hex(&r.digest))
                    .map_err(storage)?
                    .into_file();
                Ok((file, r.length))
            })
            .collect::<Result<Vec<_>, Status>>()?;
        Ok(crate::api::backend::OutputSnapshot {
            parts,
            length: content.length,
            rev: revisions.len() as u64,
            media_type: current.media_type.clone(),
            sha256: terminal.then(|| format!("sha256:{}", sha256::hex(&content.digest))),
        })
    }
    fn open_member(
        &self,
        actor: VerifiedActor,
        run: u64,
        output: &str,
        index: Option<u32>,
        member: &str,
    ) -> Result<crate::api::backend::OutputSnapshot, Status> {
        let tree = self.open_output(actor, run, output, index)?;
        if tree.media_type != crate::gpu_service::TREE_MEDIA {
            return Err(Status::invalid_argument("only a tree output has members"));
        }
        let mut manifest = Vec::new();
        for (file, length) in tree.parts {
            file.take(length).read_to_end(&mut manifest).map_err(problem)?;
        }
        let document: Value = serde_json::from_slice(&manifest)
            .map_err(|_| Status::data_loss("tree manifest is not JSON"))?;
        let entry = document["entries"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|entry| entry["path"] == member)
            .ok_or_else(|| Status::not_found("the tree has no such member"))?;
        let (Some(sha), Some(length)) = (entry["blob"]["sha256"].as_str(), entry["blob"]["length"].as_u64()) else {
            return Err(Status::data_loss("tree member names no blob"));
        };
        let file = self.store.open_verified(sha).map_err(storage)?.into_file();
        Ok(crate::api::backend::OutputSnapshot {
            parts: vec![(file, length)],
            length,
            rev: tree.rev,
            media_type: "application/octet-stream".into(),
            sha256: Some(format!("sha256:{sha}")),
        })
    }
    fn read_triage(
        &self,
        actor: VerifiedActor,
        request: domain::MachineExecutionTriageQuery,
    ) -> Result<domain::MachineExecutionTriage, Status> {
        let record = self.query(
            actor,
            request
                .execution
                .ok_or_else(|| Status::invalid_argument("execution query absent"))?,
        )?;
        if request.attempt_ordinal != 0 && request.attempt_ordinal != record.attempt.max(1) as u64
        {
            return Err(Status::not_found("this machine keeps one attempt per run"));
        }
        let (triage, bytes) = self
            .service
            .engine
            .triage(&record.id)
            .map_err(problem)?
            .ok_or_else(|| Status::not_found("this attempt kept no triage bundle"))?;
        Ok(domain::MachineExecutionTriage {
            bundle: Some(domain::TriageBundleRef {
                subject_id: triage.subject_id,
                write_receipt_digest: digest_bytes(&format!("sha256:{}", triage.sha256))?,
                length: triage.length,
            }),
            bundle_canonical_bytes: bytes,
        })
    }
    fn read_machine_log(
        &self,
        _: VerifiedActor,
        request: domain::MachineLogQuery,
    ) -> Result<Vec<u8>, Status> {
        if request.log != domain::MachineLog::TensorfsTransport as i32 {
            return Err(Status::not_found(format!(
                "this machine keeps no log {}",
                request.log
            )));
        }
        // TensorFS appends here and rotates to one older file.
        let logs = self.store.root().join("logs");
        let mut data = vec![];
        for name in ["transport.log.1", "transport.log"] {
            match std::fs::read(logs.join(name)) {
                Ok(kept) => data.extend(kept),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(Status::internal(format!("read {name}: {error}"))),
            }
        }
        let tail = request.tail_bytes as usize;
        if tail > 0 && data.len() > tail {
            data.drain(..data.len() - tail);
            if let Some(line) = data.iter().position(|b| *b == b'\n') {
                data.drain(..=line);
            }
        }
        Ok(data)
    }
    fn object(&self, actor: VerifiedActor, sha256: &str) -> Result<std::fs::File, Status> {
        // Only an object this signer wrote: another signer's reads as absent.
        let held = self
            .service
            .engine
            .with_journal(|j| j.object(&actor_id(actor), sha256))
            .map_err(problem)?;
        if held.is_none() {
            return Err(Status::not_found("this signer wrote no such object"));
        }
        Ok(self.store.open_verified(sha256).map_err(storage)?.into_file())
    }
    fn list_packages(
        &self,
        actor: VerifiedActor,
        _: domain::PackageListQuery,
    ) -> Result<domain::PackageList, Status> {
        let mut packages = vec![];
        for installed in self
            .service
            .engine
            .installations(&actor_id(actor))
            .map_err(problem)?
        {
            let held = match self.service.catalog.resolve(&installed.generation) {
                Ok(held) => held,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(problem(error)),
            };
            let warning = held.record.sdk_fallback.clone();
            let sdk = held
                .record
                .dependencies
                .into_iter()
                .filter(|d| matches!(d.name.as_str(), "cozy-runtime" | "tensorfs"))
                .map(|d| domain::ImageDistribution {
                    distribution: d.name,
                    version: d.version,
                })
                .collect();
            let interface: Value = serde_json::from_slice(&installed.interface)
                .map_err(|_| Status::data_loss("held interface corrupt"))?;
            let mut entrypoints = interface
                .get("entrypoints")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|e| e.get("name").and_then(Value::as_str).map(str::to_string))
                .collect::<Vec<_>>();
            entrypoints.sort();
            packages.push(domain::MachinePackage {
                installation_id: installed.alias,
                package: installed.package,
                release: installed.release,
                origin: "local".into(),
                sdk,
                entrypoints,
                warning,
                ..Default::default()
            });
        }
        packages.sort_by(|a, b| {
            (&a.package, &a.release, &a.installation_id).cmp(&(
                &b.package,
                &b.release,
                &b.installation_id,
            ))
        });
        Ok(domain::PackageList { packages })
    }
    fn levels(&self, actor: VerifiedActor) -> Result<BTreeMap<String, &'static str>, Status> {
        let (actor, gpu) = (actor_id(actor), self.service.gpu());
        let held = gpu.as_ref().map(|gpu| gpu.levels()).unwrap_or_default();
        let mut levels = BTreeMap::new();
        for installed in self.service.engine.installations(&actor).map_err(problem)? {
            // `downloaded`: the store still holds every model of a preparation bound to it.
            let bound = self
                .service
                .engine
                .with_journal(|journal| journal.preparations_of(&actor, &installed.alias))
                .map_err(problem)?;
            let downloaded = gpu.as_ref().is_some_and(|gpu| {
                bound.iter().any(|bound| gpu.plan(bound).is_ok_and(|plan| gpu.holds(&plan)))
            });
            let base = match downloaded {
                true => Level::Downloaded,
                false => Level::Installed,
            };
            // Only the installation's own App: a callee's executors in its environment are the
            // callee's.
            let application = self
                .service
                .catalog
                .resolve(&installed.generation)
                .map(|g| g.record.application)
                .unwrap_or_default();
            let level = held
                .get(&(installed.generation.clone(), application))
                .map_or(base, |held| base.max(*held));
            levels.insert(installed.alias, level.name());
        }
        Ok(levels)
    }
    fn warm_set(&self, actor: VerifiedActor) -> Result<Vec<v1::WarmItem>, Status> {
        let actor = actor_id(actor);
        let kept = self
            .service
            .engine
            .with_journal(|journal| journal.warm_sets())
            .map_err(problem)?;
        let holds = self.service.gpu().map(|gpu| gpu.members(&actor));
        let mut items = vec![];
        fn corrupt<E>(_: E) -> Status {
            Status::data_loss("kept warm set corrupt")
        }
        for (position, (_, record)) in kept.iter().filter(|(of, _)| *of == actor).enumerate() {
            let kept: KeptMember = serde_json::from_str(record).map_err(corrupt)?;
            let sent = STANDARD.decode(&kept.item).map_err(corrupt)?;
            let mut item = v1::WarmItem::decode(&sent[..]).map_err(corrupt)?;
            // Before the pool has taken the set up (a restart in progress) it holds its code.
            let (level, held_back) = match &holds {
                Some(holds) => holds.get(position).copied().unwrap_or((Level::Installed, "")),
                None => without_gpus(Level::named(&kept.level).ok_or_else(|| corrupt(()))?),
            };
            (item.holds, item.held_back) = (level.name().into(), held_back.into());
            items.push(item);
        }
        Ok(items)
    }
    fn workspace(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionWorkspaceQuery,
    ) -> Result<domain::MachineExecutionWorkspace, Status> {
        Ok(domain::MachineExecutionWorkspace {
            worker_id: self.authority.worker_id.clone(),
            worker_boot_id: self.authority.boot_id.clone(),
            execution_workspace_id: self.workspace_id(),
        })
    }
    fn get(
        &self,
        actor: VerifiedActor,
        query: domain::MachineExecutionQuery,
    ) -> Result<domain::MachineExecutionState, Status> {
        self.state(&self.query(actor, query)?)
    }
    fn execution_ms(
        &self,
        actor: VerifiedActor,
        query: domain::MachineExecutionQuery,
    ) -> Result<u64, Status> {
        Ok(self.query(actor, query)?.executed_ms)
    }
    fn measurements(
        &self,
        actor: VerifiedActor,
        query: domain::MachineExecutionQuery,
    ) -> Result<Option<Vec<u8>>, Status> {
        let record = self.query(actor, query)?;
        Ok(self
            .service
            .engine
            .with_journal(|journal| journal.measurements(&record.id))?)
    }
    fn events(
        &self,
        actor: VerifiedActor,
        request: domain::MachineExecutionEventsQuery,
    ) -> Result<domain::MachineExecutionEventPage, Status> {
        let query = request
            .execution
            .ok_or_else(|| Status::invalid_argument("execution query absent"))?;
        let mut record = self.query(actor, query.clone())?;
        if request.wait && !record.state.terminal() && record.revision <= request.after {
            let epoch = self.service.engine.activity_epoch();
            record = self.query(actor, query.clone())?;
            if !record.state.terminal() && record.revision <= request.after {
                self.service
                    .engine
                    .wait_activity(epoch, Some(Duration::from_secs(30)));
                record = self.query(actor, query)?;
            }
        }
        if record.state.terminal() {
            let mut page = self.terminal(&record)?;
            page.events.retain(|event| event.sequence > request.after);
            page.events.truncate(if request.limit == 0 {
                256
            } else {
                request.limit.min(256)
            } as usize);
            page.next_after = page
                .events
                .last()
                .map(|e| e.sequence)
                .unwrap_or(request.after);
            return Ok(page);
        }
        // The Python worker's live kinds: `running` once, then the latest `progress`.
        let attempt = record.attempt.max(1) as u64;
        let event = |sequence: u64,
                     kind: &str,
                     body: &Value|
         -> Result<domain::MachineExecutionEvent, Status> {
            Ok(domain::MachineExecutionEvent {
                sequence,
                attempt_ordinal: attempt,
                at_ms: now_ms(),
                kind: kind.into(),
                body_canonical_bytes: canonical(body)?,
                ..Default::default()
            })
        };
        // Published products keep their journaled sequences; see `products.rs`.
        let mut events = self.product_events(&record)?;
        let published = events.last().map_or(0, |event| event.sequence);
        events.retain(|event| event.sequence > request.after);
        let running = record.state == State::Running && record.running_revision > 0;
        if running && record.running_revision > request.after {
            let mut started = event(
                record.running_revision,
                "running",
                &json!({"generation":attempt}),
            )?;
            started.at_ms = record.started_at_ms;
            events.push(started);
        }
        let floor = request
            .after
            .max(if running { record.running_revision } else { 0 });
        if record.revision > floor && record.revision > published {
            // A run preparing inside itself (`runs`) shows its stage and bytes as progress.
            let preparing = record.state == State::Queued
                && record.waiting_reason.as_deref() == Some(crate::journal::PREPARING);
            if let Some(progress) = record.progress.as_ref().filter(|_| running || preparing) {
                let payload = serde_json::from_str::<Value>(progress)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({"stage":progress,"step_ms":0.0}));
                events.push(event(
                    record.revision,
                    "progress",
                    &json!({"type":"progress","payload":payload}),
                )?);
            } else {
                // Every cursor up to the head is answered, or a waiting client would spin.
                events.push(event(record.revision, "state", &json!({"state":self.state(&record)?.state,"completed_units":record.completed_units,"waiting_reason":record.waiting_reason}))?);
            }
        }
        events.sort_by_key(|event| event.sequence);
        events.truncate(if request.limit == 0 {
            256
        } else {
            request.limit.min(256)
        } as usize);
        Ok(domain::MachineExecutionEventPage {
            next_after: events.last().map(|e| e.sequence).unwrap_or(request.after),
            events,
            head_sequence: record.revision,
            compacted_through: record.revision.saturating_sub(1),
        })
    }
    fn control(
        &self,
        actor: VerifiedActor,
        request: domain::MachineExecutionControl,
    ) -> Result<domain::MachineExecutionState, Status> {
        let query = request.execution
            .ok_or_else(|| Status::invalid_argument("execution query absent"))?;
        if request.action == domain::MachineExecutionAction::Cancel as i32 {
            if query.expected_execution_workspace_id != self.workspace_id() {
                return Err(refusal("execution_workspace_changed", "the requested execution journal is not this workspace"));
            }
            let canceled = self.service.engine.cancel_run(&actor_id(actor), &query.request_id)
                .map_err(|error| match error.kind() {
                    std::io::ErrorKind::Unsupported => refusal("update_control_unsupported", &error.to_string()),
                    _ => problem(error),
                })?;
            if let Some(jobs) = self.service.jobs() {
                jobs.canceled(&canceled);
            }
            return self.state(&canceled);
        }
        let record = self.query(actor, query)?;
        let actor = actor_id(actor);
        let jobs = self.service.jobs();
        let typed = |refused: crate::objects::Refused| {
            refusal(refused.code, &refused.message)
        };
        let changed = match domain::MachineExecutionAction::try_from(request.action) {
            Ok(domain::MachineExecutionAction::Pause) => jobs
                .ok_or_else(|| refusal("pause_unsupported", "this machine runs no jobs"))?
                .pause(&record, &actor)
                .map_err(typed)?,
            Ok(domain::MachineExecutionAction::Resume) => jobs
                .ok_or_else(|| refusal("run_not_paused", "this machine runs no jobs"))?
                .resume(&record)
                .map_err(typed)?,
            _ => return Err(Status::invalid_argument("control names an action")),
        };
        self.state(&changed)
    }
    fn list(
        &self,
        actor: VerifiedActor,
        request: domain::MachineExecutionListQuery,
    ) -> Result<domain::MachineExecutionList, Status> {
        let actor = actor_id(actor);
        let head_number = self.service.engine.actor_head(&actor).map_err(problem)?;
        let records = self
            .service
            .engine
            .actor_page(
                &actor,
                if request.newest_first {
                    0
                } else {
                    request.after_number
                },
                if request.newest_first {
                    request.before_number
                } else {
                    0
                },
                request.newest_first,
                &request.states,
                if request.limit == 0 {
                    64
                } else {
                    request.limit.min(256)
                } as usize,
            )
            .map_err(problem)?;
        let executions = records
            .iter()
            .map(|r| self.state(r))
            .collect::<Result<Vec<_>, _>>()?;
        if request.wait {
            return Err(Status::unimplemented(
                "list wait is not qualified by this build",
            ));
        }
        Ok(domain::MachineExecutionList {
            executions,
            head_number,
            execution_workspace_id: self.workspace_id(),
        })
    }
}
pub fn actor_id(actor: VerifiedActor) -> String {
    sha256::hex(&actor.public_key)
}
fn verify_checksum(
    source: &mut std::fs::File,
    binding: &crate::journal::AssetBinding,
) -> Result<(), Status> {
    use blake2::digest::{Update, VariableOutput};
    let mut blake = blake2::Blake2bVar::new(16)
        .map_err(|_| Status::internal("checksum configuration invalid"))?;
    let mut sha = sha256::Sha256::new();
    let mut length = 0;
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = source.read(&mut buffer).map_err(problem)?;
        if count == 0 {
            break;
        }
        match binding.checksum.algorithm.as_str() {
            "blake2b-128" => Update::update(&mut blake, &buffer[..count]),
            "sha256" => sha.update(&buffer[..count]),
            _ => {
                return Err(Status::unimplemented(
                    "producer output checksum algorithm is unavailable",
                ))
            }
        }
        length += count as u64;
    }
    let digest = match binding.checksum.algorithm.as_str() {
        "blake2b-128" => {
            let mut bytes = [0; 16];
            blake
                .finalize_variable(&mut bytes)
                .map_err(|_| Status::internal("checksum finalization invalid"))?;
            sha256::hex(&bytes)
        }
        "sha256" => sha256::hex(&sha.finish()),
        _ => {
            return Err(Status::unimplemented(
                "producer output checksum algorithm is unavailable",
            ))
        }
    };
    source.seek(SeekFrom::Start(0)).map_err(problem)?;
    if length != binding.length || digest != binding.checksum.value {
        return Err(Status::data_loss(
            "SDK output checksum differs from held result bytes",
        ));
    }
    Ok(())
}
fn canonical(value: &Value) -> Result<Vec<u8>, Status> {
    serde_json_canonicalizer::to_vec(value)
        .map_err(|_| Status::invalid_argument("JSON value is outside canonical profile"))
}
fn identity(value: &Value) -> Result<String, Status> {
    Ok(format!(
        "sha256:{}",
        sha256::hex(&sha256::digest(&canonical(value)?))
    ))
}
fn digest_bytes(value: &str) -> Result<Vec<u8>, Status> {
    let value = value
        .strip_prefix("sha256:")
        .ok_or_else(|| Status::data_loss("digest algorithm differs"))?;
    if value.len() != 64 {
        return Err(Status::data_loss("digest length differs"));
    }
    (0..32)
        .map(|i| {
            u8::from_str_radix(&value[2 * i..2 * i + 2], 16)
                .map_err(|_| Status::data_loss("digest spelling differs"))
        })
        .collect()
}
/// A callable's declaration: an entrypoint, or a job's (`jobs`).
fn entrypoint<'a>(interface: &'a Value, name: &str, job: bool) -> Result<&'a Value, Status> {
    interface
        .get(if job { "jobs" } else { "entrypoints" })
        .and_then(Value::as_array)
        .and_then(|entries| {
            entries
                .iter()
                .find(|entry| entry.get("name").and_then(Value::as_str) == Some(name))
        })
        .ok_or_else(|| Status::invalid_argument("entrypoint is not declared by this held package"))
}
fn problem(error: io::Error) -> Status {
    match error.kind() {
        io::ErrorKind::NotFound => Status::not_found(error.to_string()),
        io::ErrorKind::AlreadyExists => Status::already_exists(error.to_string()),
        io::ErrorKind::InvalidInput => Status::invalid_argument(error.to_string()),
        io::ErrorKind::PermissionDenied => Status::permission_denied(error.to_string()),
        _ => Status::failed_precondition(error.to_string()),
    }
}
fn storage(error: impl std::fmt::Display) -> Status {
    Status::failed_precondition(format!("native custody: {error}"))
}
fn refusal(code: &str, detail: &str) -> Status {
    let mut status = Status::failed_precondition(detail.to_owned());
    if let Ok(code) = code.parse() {
        status.metadata_mut().insert("cozy-error-code", code);
    }
    status
}
type AssetSources = std::collections::HashMap<
    String,
    (
        crate::journal::Artifact,
        String,
        domain::NativeByteRetentionRequest,
    ),
>;
fn rewrite_assets(
    value: &mut Value,
    schema: &Value,
    path: &str,
    op: domain::RunProductOp,
    index: u32,
    sources: &AssetSources,
    products: &mut Vec<domain::RunProduct>,
) -> Result<(), Status> {
    if value.is_null() {
        return Ok(());
    }
    // A returned tree (`{"input": "tree"}`) is an output like an asset: its manifest's bytes.
    if schema.get("asset").is_some() || schema["input"] == "tree" {
        let object = value
            .as_object_mut()
            .ok_or_else(|| Status::data_loss("asset result differs from declared schema"))?;
        if let Some(reference) = object.get("asset_ref").and_then(Value::as_str) {
            let (artifact, mime, source) = sources.get(reference).ok_or_else(|| {
                Status::failed_precondition(
                    "executor did not provide native output binding for this asset",
                )
            })?;
            object.insert(
                "digest".into(),
                json!(format!("sha256:{}", artifact.sha256)),
            );
            products.push(domain::RunProduct {
                output: path.into(),
                op: op as i32,
                index,
                content: Some(domain::Ref {
                    digest: digest_bytes(&format!("sha256:{}", artifact.sha256))?,
                    length: artifact.length,
                }),
                media_type: mime.clone(),
                source: Some(source.clone()),
                ..Default::default()
            });
        } else {
            return Err(Status::data_loss("declared asset has no reference"));
        }
    } else if let Some(fields) = schema.get("fields").and_then(Value::as_array) {
        let object = value
            .as_object_mut()
            .ok_or_else(|| Status::data_loss("result record differs from declared schema"))?;
        for field in fields {
            let name = field
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| Status::data_loss("result field name absent"))?;
            if let Some(child) = object.get_mut(name) {
                let path = if path.is_empty() {
                    name.to_string()
                } else {
                    format!("{path}.{name}")
                };
                rewrite_assets(
                    child,
                    field
                        .get("type")
                        .ok_or_else(|| Status::data_loss("result field type absent"))?,
                    &path,
                    op,
                    index,
                    sources,
                    products,
                )?;
            }
        }
    } else if let Some(element) = schema.get("list") {
        let items = value
            .as_array_mut()
            .ok_or_else(|| Status::data_loss("result list differs from declared schema"))?;
        for (position, child) in items.iter_mut().enumerate() {
            if element.get("asset").is_some() {
                rewrite_assets(
                    child,
                    element,
                    path,
                    domain::RunProductOp::Append,
                    position.try_into().map_err(|_| {
                        Status::out_of_range("list output index is outside deployed protocol")
                    })?,
                    sources,
                    products,
                )?;
            } else {
                rewrite_assets(
                    child,
                    element,
                    &format!("{path}[{position}]"),
                    op,
                    index,
                    sources,
                    products,
                )?;
            }
        }
    }
    Ok(())
}
fn product_document(product: &domain::RunProduct) -> Result<Vec<u8>, Status> {
    canonical(
        &crate::products::document(product)
            .map_err(|_| Status::data_loss("product reference absent"))?,
    )
}

fn output_entries(products: &[domain::RunProduct]) -> Result<Vec<Value>, Status> {
    let mut entries = std::collections::BTreeMap::new();
    for product in products {
        let document: Value = serde_json::from_slice(&product_document(product)?)
            .map_err(|_| Status::internal("product document is invalid"))?;
        let output_id = if product.op == domain::RunProductOp::Append as i32 {
            format!("{}[{}]", product.output, product.index)
        } else {
            product.output.clone()
        };
        let entry = json!({
            "output_id":output_id,
            "digest":document["content"]["digest"],
            "length":document["content"]["length"],
            "mime_type":product.media_type,
            "native_tree":document["source"]["source"]
        });
        if entries.insert(output_id, entry).is_some() {
            return Err(Status::data_loss("output binding path is ambiguous"));
        }
    }
    Ok(entries.into_values().collect())
}

/// The SDK's `Outputs.publish` against the real Runtime executor of a real installed package:
/// products stream on the run's log while it runs, and its result adds only what is new.
#[cfg(test)]
mod product_log_tests {
    use super::*;
    use crate::{
        api::MachineIdentity,
        device_executor::{
            self, Baseline, Binding, Budgets, DeviceCommand, DeviceExecutor, ExecutorConfig, Frame,
            Services,
        },
        execution::Engine as Execution_Engine,
        journal::{Installation, Invocation, SubmissionContext},
    };
    use std::{
        collections::BTreeMap,
        fs::File,
        path::{Path, PathBuf},
        process::Command,
    };

    const INSTALL: &str = r#"
import json, subprocess, sys
from pathlib import Path
from cozy_machine_client.packages import install
fixture, out = Path(sys.argv[1]), Path(sys.argv[2])
subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out / "client"), "."], check=True, capture_output=True)
generation = install(fixture, out / "generations", next((out / "client").glob("*.whl")), python="3.12")
print(json.dumps({"identity": generation.identity}))
"#;

    /// A real uv-installed generation of `tests/fixtures/<name>` (released cozy-runtime).
    fn install_fixture(root: &Path) -> String {
        install_named(root, "cpu_publish")
    }
    fn install_named(root: &Path, name: &str) -> String {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = Command::new("uv")
            .current_dir(repo)
            .args([
                "run", "--locked", "--extra", "test", "python", "-c", INSTALL,
            ])
            .arg(repo.join("tests/fixtures").join(name))
            .arg(root)
            .output()
            .expect("uv runs the repository's installer");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let line = String::from_utf8(output.stdout).unwrap();
        let value: Value = serde_json::from_str(line.lines().last().unwrap()).unwrap();
        value["identity"].as_str().unwrap().to_owned()
    }

    struct Publisher<'a> {
        store: &'a Store,
        engine: &'a Execution_Engine,
        id: &'a str,
        spool: PathBuf,
    }
    impl Services for Publisher<'_> {
        fn request(
            &mut self,
            frame: &Frame,
            descriptor: Option<File>,
        ) -> io::Result<(device_executor::Answer, Option<File>)> {
            drop(descriptor);
            if frame.kind == device_executor::Kind::Publish {
                let answer =
                    crate::products::publish(self.store, self.engine, self.id, &self.spool, frame);
                return Ok((answer, None));
            }
            Ok((device_executor::Answer::unavailable(frame.seq), None))
        }
    }

    fn command(executor: &mut DeviceExecutor, command: &DeviceCommand) -> io::Result<()> {
        let reply = executor.command(command, &mut Baseline)?;
        if !reply.ok {
            return Err(io::Error::other(format!(
                "{}: {}",
                reply.code, reply.detail
            )));
        }
        Ok(())
    }

    fn products(page: &domain::MachineExecutionEventPage) -> Vec<(u64, String, i32, u32, String)> {
        page.events
            .iter()
            .filter_map(|event| {
                let product = event.product.as_ref()?;
                Some((
                    event.sequence,
                    product.output.clone(),
                    product.op,
                    product.index,
                    product.label.clone(),
                ))
            })
            .collect()
    }

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// One public run of a fixture entrypoint on the real executor, settled as the GPU pool
    /// settles it (`products::publish` for publishes, `keep_triage` for a failure).
    struct Fixture {
        root: PathBuf,
        engine: Arc<Execution_Engine>,
        backend: NativeBackend,
        actor: VerifiedActor,
        failure: Arc<Mutex<Option<String>>>,
        until: std::time::Instant,
        _scratch: Scratch,
    }
    impl Fixture {
        fn start(entrypoint: &'static str, input: impl FnOnce(&Path) -> Value) -> Self {
        let scratch =
            Scratch(std::env::temp_dir().join(format!("cm-products-{}", uuid::Uuid::new_v4())));
        let root = scratch.0.clone();
        std::fs::create_dir_all(&root).unwrap();
        let identity = install_fixture(&root);
        let state = root.join("state");
        let service = Service::open(&state, &root.join("generations"), 1).unwrap();
        assert!(
            service.stop().unwrap(),
            "this test dispatches the run itself"
        );
        let engine = service.engine.clone();
        let store = Arc::new(Store::ensure(&state.join("tensorfs")).unwrap());
        let signer = ed25519_dalek::SigningKey::from_bytes(&[41; 32]);
        let machine = MachineIdentity::ephemeral(
            "products".into(),
            vec![signer.verifying_key()],
            vec![7; 32],
        )
        .unwrap();
        let actor = VerifiedActor {
            public_key: signer.verifying_key().to_bytes(),
        };
        let backend = NativeBackend::new(
            service.clone(),
            machine.authority.clone(),
            store.clone(),
        );
        let held = service.catalog.resolve(&identity).unwrap();
        engine
            .bind_installation(Installation {
                actor: actor_id(actor),
                alias: "fixture".into(),
                generation: identity.clone(),
                package: held.record.package.clone(),
                release: held.record.version.clone(),
                interface: serde_json::to_vec(&held.record.interface).unwrap(),
            })
            .unwrap();
        let digest = format!("sha256:{}", "a".repeat(64));
        let record = engine
            .submit_public(
                SubmissionContext {
                    actor: actor_id(actor),
                    request_id: "request-1".into(),
                    submission_id: "submission-1".into(),
                    expected_workspace_id: engine.workspace_id(),
                    capture_digest: digest.clone(),
                    invocation_digest: digest.clone(),
                    payload_digest: digest,
                    ..Default::default()
                },
                Invocation {
                    package: held.record.package.clone(),
                    generation: identity,
                    module: held.record.application.clone(),
                    entrypoint: entrypoint.into(),
                    input: input(&root),
                    attention_kernel: String::new(),
                    inputs: vec![],
                    ..Default::default()
                },
            )
            .unwrap();
        let executor_root = root.join("executor");
        let task_store = store.clone();
        let failure = Arc::new(Mutex::new(None::<String>));
        let failed = failure.clone();
        assert!(engine
            .dispatch_managed(&record.id, move |engine, id| {
                let result = (|| -> io::Result<()> {
                    std::fs::create_dir(&executor_root)?;
                    let mut executor = DeviceExecutor::spawn_observed(
                        ExecutorConfig {
                            python: held.record.python.clone(),
                            root: executor_root.clone(),
                            socket: executor_root.join("e.sock"),
                            environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
                            // No device is visible (empty devices): this package never imports torch.
                            seal: {
                                let mut seal = crate::launch_identity::Seal::prepare(
                                    &executor_root,
                                    None,
                                    "products",
                                    &held.record.identity,
                                    "",
                                )?;
                                seal.threads = 1;
                                seal
                            },
                            generation_hold: Some(held.retention()),
                            identity: None,
                            cgroup_namespace: None,
                        },
                        |birth, cancel| {
                            let (cancel, request) = (cancel.clone(), id.clone());
                            engine.register_managed(
                                &id,
                                birth.clone(),
                                Arc::new(move || cancel.cancel(&request)),
                            )
                        },
                    )?;
                    assert!(!executor.hello.torch_loaded);
                    assert!(engine.authorize_managed(&id, None)?);
                    let interface = executor_root.join("package-interface.json");
                    std::fs::write(&interface, serde_json::to_vec(&held.record.interface)?)?;
                    command(
                        &mut executor,
                        &DeviceCommand::Start {
                            devices: String::new(),
                            application: held.record.application.clone(),
                            package_interface: interface.clone(),
                            sequence_parallel_degree: 1,
                            import_only: false,
                        },
                    )?;
                    command(
                        &mut executor,
                        &DeviceCommand::Load {
                            construction: "fixture".into(),
                            devices: String::new(),
                            sequence_parallel_degree: 1,
                            binding: Box::new(Binding {
                                application: held.record.application.clone(),
                                package_interface: interface.display().to_string(),
                                ..Binding::default()
                            }),
                            budgets: Budgets::default(),
                            models: Vec::new(),
                            authorized_device_limit_bytes: None,
                            attention_pin: String::new(),
                            stages: false,
                            device_weights: false,
                            cap_bytes: None,
                            sealed_tiers: false,
                            model_sources: false,
                            staged_tiers: false,
                            pinned_bytes: None,
                        },
                    )?;
                    command(
                        &mut executor,
                        &DeviceCommand::Activate {
                            construction: "fixture".into(),
                        },
                    )?;
                    let record = engine.get(&id)?;
                    command(
                        &mut executor,
                        &DeviceCommand::PrepareRequest {
                            request_id: id.clone(),
                            construction: "fixture".into(),
                            entrypoint: entrypoint.into(),
                            payload: record.invocation.input,
                            attention_kernel: String::new(),
                            input_metadata: Default::default(),
                        },
                    )?;
                    let spool = engine.staging(&id)?;
                    let reply = executor.command(
                        &DeviceCommand::Invoke {
                            max_output_bytes: crate::output_capacity::DEFAULT_BYTES,
                            request_id: id.clone(),
                            construction: "fixture".into(),
                            entrypoint: entrypoint.into(),
                            spool: spool.clone(),
                            deadline_s: None,
                            attention_kernel: String::new(),
                            plane_budget_bytes: -1,
                            stages: false,
                            cap_bytes: None,
                            inputs: Default::default(),
                            trees: Default::default(),
                            floor_bytes: None,
                            activation_bytes: Default::default(),
                            squeezed_bytes: Default::default(),
                            device_weights: None,
                        },
                        &mut Publisher {
                            store: &task_store,
                            engine: &engine,
                            id: &id,
                            spool: spool.clone(),
                        },
                    )?;
                    if let Some(outcome) =
                        reply.outcome.as_ref().filter(|o| o.terminal != "succeeded")
                    {
                        crate::gpu_service::keep_triage(&engine, &id, &executor, outcome);
                        let failure = crate::journal::Failure::executor(
                            &outcome.terminal,
                            &outcome.origin,
                            &outcome.code,
                            &outcome.message,
                        );
                        engine.finish(&id, crate::journal::Outcome::Failed(failure))?;
                        return executor.shutdown();
                    }
                    let (value, bindings) =
                        device_executor::postprocess(&executor.codec(), &spool, &reply)?;
                    engine.managed_result(
                        &id,
                        &spool,
                        value,
                        crate::gpu_service::output_bindings(bindings)?,
                    )?;
                    executor.shutdown()
                })();
                if let Err(error) = &result {
                    *failed.lock().unwrap() = Some(error.to_string());
                }
                result
            })
            .unwrap());
        Self {
            root,
            engine,
            backend,
            actor,
            failure,
            // Harness bound on a broken run only; the product never kills by elapsed time.
            until: std::time::Instant::now() + Duration::from_secs(300),
            _scratch: scratch,
        }
        }
        fn check(&self) {
            assert!(
                self.failure.lock().unwrap().is_none(),
                "run failed: {:?}",
                self.failure.lock().unwrap()
            );
            assert!(std::time::Instant::now() < self.until, "run made no progress");
        }
        fn query(&self, after: u64) -> domain::MachineExecutionEventsQuery {
            domain::MachineExecutionEventsQuery {
                execution: Some(self.execution()),
                after,
                limit: 0,
                wait: true,
            }
        }
        fn execution(&self) -> domain::MachineExecutionQuery {
            domain::MachineExecutionQuery {
                request_id: "request-1".into(),
                expected_execution_workspace_id: self.engine.workspace_id(),
            }
        }
        fn terminal_page(&self) -> domain::MachineExecutionEventPage {
            loop {
                self.check();
                let page = self.backend.events(self.actor, self.query(0)).unwrap();
                if page.events.iter().any(|event| event.kind == "outcome") {
                    return page;
                }
            }
        }
    }

    #[test]
    fn published_products_stream_while_running_and_the_result_adds_only_what_is_new() {
        let fixture = Fixture::start("make", |root| json!({"gate": root.join("gate")}));
        let (engine, backend, actor) = (&fixture.engine, &fixture.backend, fixture.actor);
        let gate = fixture.root.join("gate");
        let check = || fixture.check();
        let query = |after| fixture.query(after);
        let (append, set) = (
            domain::RunProductOp::Append as i32,
            domain::RunProductOp::Set as i32,
        );
        // While the run waits at its gate, the log already shows its first two products.
        let mut cursor = 0;
        let mut live = vec![];
        while live.len() < 2 {
            check();
            let page = backend.events(actor, query(cursor)).unwrap();
            cursor = page.next_after;
            live.extend(products(&page));
        }
        assert_eq!(engine.get("1").unwrap().state, State::Running);
        assert_eq!(
            live.iter()
                .map(|p| (p.1.as_str(), p.2, p.3, p.4.as_str()))
                .collect::<Vec<_>>(),
            [
                ("frames", append, 0, "Frame 1"),
                ("preview", set, 0, "Draft")
            ]
        );
        std::fs::write(&gate, b"").unwrap();
        let page = loop {
            check();
            let page = backend.events(actor, query(0)).unwrap();
            if page.events.iter().any(|event| event.kind == "outcome") {
                break page;
            }
        };
        let log = products(&page);
        assert_eq!(
            &log[..2],
            &live[..],
            "published products keep their sequences"
        );
        assert_eq!(
            log.iter()
                .map(|p| (p.1.as_str(), p.2, p.3, p.4.as_str()))
                .collect::<Vec<_>>(),
            [
                ("frames", append, 0, "Frame 1"),
                ("preview", set, 0, "Draft"),
                ("frames", append, 1, "Frame 2"),
                ("frames", append, 2, ""),
                ("preview", set, 0, ""),
            ],
            "the result adds only the unpublished third frame and the new preview"
        );
        assert!(page
            .events
            .windows(2)
            .all(|w| w[0].sequence < w[1].sequence));
        assert_eq!(page.events.last().unwrap().kind, "outcome");
        // `cozy run show` reads the execution time from the terminal run.timing event.
        let timing = page
            .events
            .iter()
            .find(|e| e.kind == "run.timing")
            .expect("run.timing event");
        let timing: Value = serde_json::from_slice(&timing.body_canonical_bytes).unwrap();
        assert_eq!(
            (timing["attempt"].as_u64(), timing["terminal"].as_bool()),
            (Some(1), Some(true))
        );
        assert!(timing["execution_ms"].as_f64().is_some_and(|ms| ms >= 0.0));
        // Each output's current bytes are what Read serves the actor.
        let number = backend.get(actor, fixture.execution()).unwrap().number;
        let read = |output: &str, index: Option<u32>| -> Vec<u8> {
            let snapshot = backend.open_output(actor, number, output, index).unwrap();
            let mut bytes = vec![];
            for (file, length) in snapshot.parts {
                file.take(length).read_to_end(&mut bytes).unwrap();
            }
            bytes
        };
        assert_eq!(
            [read("frames", Some(1)), read("frames", Some(2)), read("frames", Some(3)), read("preview", None)],
            [&b"frame-1"[..], b"frame-2", b"frame-3", b"preview-final"]
        );
    }

    #[test]
    fn a_failed_attempt_keeps_a_triage_bundle_its_outcome_names_and_the_owner_can_read() {
        let fixture = Fixture::start("explode", |_| json!({}));
        let page = fixture.terminal_page();
        let outcome = page.events.last().unwrap().outcome.clone().unwrap();
        let body: Value = serde_json::from_slice(&outcome.outcome_canonical_bytes).unwrap();
        assert_eq!(body["status"], 3, "{body}");
        let named = &body["triage_bundle"];
        let triage = fixture
            .backend
            .read_triage(
                fixture.actor,
                domain::MachineExecutionTriageQuery {
                    execution: Some(fixture.execution()),
                    attempt_ordinal: 0,
                },
            )
            .unwrap();
        let reference = triage.bundle.unwrap();
        // What the CLI checks: the bytes are exactly the ones the outcome names.
        assert_eq!(reference.length, triage.bundle_canonical_bytes.len() as u64);
        assert_eq!(
            reference.write_receipt_digest,
            sha256::digest(&triage.bundle_canonical_bytes).to_vec()
        );
        assert_eq!(named["subject_id"], json!(reference.subject_id));
        assert_eq!(named["length"], json!(reference.length));
        assert_eq!(
            named["write_receipt_digest"],
            json!(format!("sha256:{}", sha256::hex(&reference.write_receipt_digest)))
        );
        let bundle: Value = serde_json::from_slice(&triage.bundle_canonical_bytes).unwrap();
        let traceback = bundle["terminal"]["traceback"].as_str().unwrap();
        assert!(traceback.contains("ValueError") && traceback.contains("boom from the fixture"), "{bundle}");
        assert_eq!(bundle["request_id"], "request-1");
        assert!(bundle["executor"]["pid"].as_u64().unwrap() > 0);
        // A later attempt ordinal names no bundle on this machine.
        let later = fixture.backend.read_triage(
            fixture.actor,
            domain::MachineExecutionTriageQuery {
                execution: Some(fixture.execution()),
                attempt_ordinal: 9,
            },
        );
        assert_eq!(later.unwrap_err().code(), tonic::Code::NotFound);
    }
}

#[cfg(test)]
mod terminal_tests {
    use super::*;
    use crate::journal::{Invocation, Outcome, SubmissionContext};

    /// A run seen queued and next read finished still shows it ran, before its outcome.
    #[test]
    fn a_finished_run_shows_its_start() {
        let root = std::env::temp_dir().join(format!("cm-terminal-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        assert!(service.stop().unwrap(), "the test moves the run itself");
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let signer = ed25519_dalek::SigningKey::from_bytes(&[44; 32]);
        let machine =
            crate::api::MachineIdentity::ephemeral("terminal".into(), vec![signer.verifying_key()], vec![7; 32]).unwrap();
        let actor = VerifiedActor { public_key: signer.verifying_key().to_bytes() };
        let backend = NativeBackend::new(service.clone(), machine.authority.clone(), store);
        let engine = &service.engine;
        let digest = format!("sha256:{}", "a".repeat(64));
        let (queued, started) = engine.with_journal(|journal| {
            let context = SubmissionContext {
                actor: actor_id(actor),
                request_id: "request-1".into(),
                submission_id: "submission-1".into(),
                expected_workspace_id: journal.workspace_id().into(),
                capture_digest: digest.clone(),
                invocation_digest: digest.clone(),
                payload_digest: digest.clone(),
                ..Default::default()
            };
            let invocation = Invocation { package: "org/pkg".into(), entrypoint: "run".into(), input: json!({}), ..Default::default() };
            let queued = journal.accept_public(context, invocation)?;
            assert!(journal.claim(&queued.id)?);
            journal.register_process(&queued.id, crate::execution::process_birth(std::process::id())?)?;
            let started = journal.running(&queued.id, None)?;
            journal.finish(&queued.id, Outcome::Failed(crate::journal::Failure::executor("failed", "author", "author_failed", "it stopped")))?;
            Ok((queued, started))
        }).unwrap();
        let execution = domain::MachineExecutionQuery {
            request_id: "request-1".into(),
            expected_execution_workspace_id: engine.workspace_id(),
        };
        let query = domain::MachineExecutionEventsQuery { execution: Some(execution), after: queued.revision, limit: 0, wait: false };
        let kinds: Vec<_> = backend.events(actor, query).unwrap().events.into_iter().map(|e| (e.kind, e.sequence)).collect();
        assert_eq!(kinds.first(), Some(&("running".to_string(), started.running_revision)), "{kinds:?}");
        assert_eq!(kinds.last().map(|(kind, _)| kind.as_str()), Some("outcome"));
        let _ = std::fs::remove_dir_all(root);
    }
}
