//! Deployed machine RPCs over the sole journal and authoritative TensorFS store.
use crate::{
    api::{
        auth::{Authority, VerifiedActor},
        pb,
        workspaces::WorkspaceUploads,
        MachineBackend,
    },
    journal::{Execution, PublicTerminal, State, SubmissionContext},
    service::Service,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use prost::Message;
use serde_json::{json, Value};
use std::{
    io::{self, Read, Seek, SeekFrom},
    sync::{Arc, Mutex},
    time::Duration,
};
use tensorfs_core::{
    ids::{ObjectRef, StoredDoc},
    sha256, source_artifact,
    store::Store,
};
use tonic::Status;

pub struct NativeBackend {
    pub service: Arc<Service>,
    pub authority: Authority,
    pub store: Arc<Store>,
    pub uploads: Arc<WorkspaceUploads>,
    pub installer: Option<crate::api::install::InstallerConfig>,
    pub publisher: Option<Arc<crate::published::Publisher>>,
    /// On a rental: its own Hub, read with the pod's worker capability.
    pub own_hub: Option<crate::hub::Source>,
    /// `cozy.machine.v1` Run sources and Write.
    pub runs: Option<Arc<crate::runs::Runs>>,
    // Serialize native projection, not inference or observation. Only one result
    // projection may establish a given immutable output's native custody at once.
    projection: Mutex<()>,
    installation: Mutex<()>,
}
impl NativeBackend {
    pub fn new(
        service: Arc<Service>,
        authority: Authority,
        store: Arc<Store>,
        uploads: Arc<WorkspaceUploads>,
    ) -> Self {
        Self {
            service,
            authority,
            store,
            uploads,
            installer: None,
            publisher: None,
            own_hub: None,
            runs: None,
            projection: Mutex::new(()),
            installation: Mutex::new(()),
        }
    }
    fn workspace_id(&self) -> String {
        self.service.engine.workspace_id()
    }
    fn query(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionQuery,
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
    fn receipt(&self, record: &Execution) -> Result<pb::MachineExecutionReceipt, Status> {
        let context = record
            .submission
            .as_ref()
            .ok_or_else(|| Status::internal("public submission context absent"))?;
        Ok(pb::MachineExecutionReceipt {
            request_id: context.request_id.clone(),
            submission_id: context.submission_id.clone(),
            capture_digest: digest_bytes(&context.capture_digest)?,
            invocation_spec_digest: digest_bytes(&context.invocation_digest)?,
            accepted_at_ms: record.accepted_at_ms,
            worker_id: self.authority.worker_id.clone(),
            worker_boot_id: record.acceptance_boot_id.clone(),
            execution_workspace_id: self.workspace_id(),
            publication_authorization_id: context.publication_authorization_id.clone(),
            number: record
                .id
                .parse()
                .map_err(|_| Status::internal("invalid journal run number"))?,
        })
    }
    fn state(&self, record: &Execution) -> Result<pb::MachineExecutionState, Status> {
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
                pb::MachineExecutionEventPage::decode(held.events.as_slice())
                    .map(|page| page.head_sequence)
                    .map_err(|_| Status::data_loss("durable event projection corrupt"))
            })
            .transpose()?
            .unwrap_or(record.revision)
            .max(record.revision);
        Ok(pb::MachineExecutionState {
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
            target: Some(pb::MachineExecutionTarget {
                package: record.invocation.package.clone(),
                entrypoint: record.invocation.entrypoint.clone(),
                installation_id: record.invocation.generation.clone(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }
    /// The run's published products, as events with their journaled sequences.
    fn product_events(&self, record: &Execution) -> Result<Vec<pb::MachineExecutionEvent>, Status> {
        let attempt = record.attempt.max(1) as u64;
        self.service
            .engine
            .products(&record.id)
            .map_err(problem)?
            .iter()
            .map(|stored| {
                let product = crate::products::decode(stored).map_err(problem)?;
                Ok(pb::MachineExecutionEvent {
                    sequence: stored.sequence,
                    attempt_ordinal: attempt,
                    at_ms: stored.at_ms,
                    kind: "product".into(),
                    body_canonical_bytes: product_document(&product)?,
                    product: Some(product),
                    ..Default::default()
                })
            })
            .collect()
    }
    fn terminal(&self, record: &Execution) -> Result<pb::MachineExecutionEventPage, Status> {
        let _guard = self.projection.lock().unwrap();
        if let Some(held) = self
            .service
            .engine
            .public_terminal(&record.id)
            .map_err(problem)?
        {
            return pb::MachineExecutionEventPage::decode(held.events.as_slice())
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
                    pb::RunProductOp::Set,
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
        let first_sequence = record
            .revision
            .checked_add(1)
            .ok_or_else(|| Status::resource_exhausted("event cursor exhausted"))?;
        let fresh = products
            .into_iter()
            .filter(|product| !crate::products::shown(&shown, product));
        for (index, product) in fresh.enumerate() {
            events.push(pb::MachineExecutionEvent {
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
            events.push(pb::MachineExecutionEvent {
                sequence: first_sequence + events.len() as u64,
                attempt_ordinal: attempt,
                at_ms: record.finished_at_ms,
                kind: kind.into(),
                body_canonical_bytes: canonical(&body)?,
                ..Default::default()
            });
        }
        let mut body = json!({"format":"cozy.worker.v1.AttemptOutcomeBody/1", "request_id":context.request_id, "attempt_ordinal":record.attempt.max(1), "invocation_spec_digest":context.invocation_digest});
        if record.process.is_some() {
            body["execution_started"] = json!(true);
        }
        let (status, code, origin, message) = match record.state {
            State::Completed => (1, 0, 2, "completed".to_string()),
            State::Canceled => (4, 11, 6, "explicitly canceled".to_string()),
            _ => {
                let failure = crate::journal::Failure::decode(
                    record
                        .failure
                        .as_deref()
                        .unwrap_or("the run failed without a recorded reason"),
                );
                (
                    failure.status,
                    failure.cause,
                    failure.origin,
                    safe(&failure.message, 4096),
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
        // Observation only: the failed attempt's bundle, written before the run settled.
        if let Some((triage, _)) = self.service.engine.triage(&record.id).map_err(problem)? {
            body["triage_bundle"] = json!({"subject_id":triage.subject_id,
                "write_receipt_digest":format!("sha256:{}", triage.sha256),"length":triage.length});
        }
        if !output_entries.is_empty() {
            body["output_manifest"] = json!({"outputs":output_entries});
        }
        if let Some(value) = value {
            body["result"] = json!({"result_schema_digest":schema_digest, "inline_result":STANDARD.encode(crate::boundary_json::intent_bytes(&value).map_err(|_| Status::internal("result JSON cannot be serialized"))?)});
        }
        let bytes = canonical(&body)?;
        let digest = sha256::digest(&bytes);
        let outcome = pb::AttemptOutcome {
            worker_boot_id: record.acceptance_boot_id.clone(),
            request_id: context.request_id.clone(),
            attempt_ordinal: record.attempt.max(1) as u64,
            invocation_spec_digest: digest_bytes(&context.invocation_digest)?,
            outcome_id: format!("out-{}", sha256::hex(&digest)),
            outcome_digest: digest.to_vec(),
            outcome_canonical_bytes: bytes,
            ..Default::default()
        };
        let sequence = events
            .last()
            .map_or(record.revision, |event| event.sequence.max(record.revision))
            + 1;
        events.push(pb::MachineExecutionEvent {
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
        let page = pb::MachineExecutionEventPage {
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
                    outcome: outcome.encode_to_vec(),
                    events: page.encode_to_vec(),
                },
            )
            .map_err(problem)?;
        pb::MachineExecutionEventPage::decode(committed.events.as_slice())
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
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
impl NativeBackend {
    /// The root's file inputs, each the committed import of this request and field: one
    /// `payload` file whose bytes the binding names. Trees are not taken yet.
    fn file_inputs(
        &self,
        actor: &str,
        request_id: &str,
        root: &pb::ReleaseRoot,
    ) -> Result<Vec<crate::journal::InputFile>, Status> {
        const FILE_BYTES: u64 = 64 << 20;
        const TOTAL_BYTES: u64 = 256 << 20;
        let unprepared = |id: &str| {
            refusal(
                "input_unprepared",
                &format!("input {id} was not imported for this request"),
            )
        };
        if root.inputs.len() != root.input_access.len() {
            return Err(Status::invalid_argument(
                "every input binding needs exactly one access",
            ));
        }
        let mut files = vec![];
        let mut total = 0;
        for binding in &root.inputs {
            let access: Vec<_> = root
                .input_access
                .iter()
                .filter(|a| a.input_id == binding.input_id)
                .collect();
            let [access] = access.as_slice() else {
                return Err(Status::invalid_argument(
                    "every input binding needs exactly one access",
                ));
            };
            let tree = access
                .native_tree
                .as_ref()
                .ok_or_else(|| Status::unimplemented("only imported (native) inputs are taken"))?;
            let state = self
                .service
                .engine
                .with_journal(|j| j.intake(actor, &tree.retention_id))
                .map_err(problem)?
                .ok_or_else(|| unprepared(&binding.input_id))?;
            if state.released
                || state.spec.request_id != request_id
                || state.spec.input_id != binding.input_id
            {
                return Err(unprepared(&binding.input_id));
            }
            let receipt = state.receipt.ok_or_else(|| unprepared(&binding.input_id))?;
            let committed = pb::NativeByteRetentionResult::decode(receipt.as_slice())
                .map_err(|_| Status::data_loss("input receipt corrupt"))?;
            let source = tree
                .source
                .as_ref()
                .filter(|s| committed.source.as_ref() == Some(*s))
                .ok_or_else(|| unprepared(&binding.input_id))?;
            let manifest = source
                .manifest
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("input tree names no manifest"))?;
            let manifest = self
                .store
                .read_manifest(&ObjectRef {
                    sha256: sha256::hex(&manifest.digest),
                    length: manifest.length,
                })
                .map_err(storage)?;
            let object = match manifest.entries() {
                [(name, tensorfs_core::manifest::Entry::File(object))] if name == "payload" => {
                    object.clone()
                }
                _ => {
                    return Err(Status::unimplemented(
                        "directory (tree) inputs are not taken yet",
                    ))
                }
            };
            if format!("sha256:{}", object.sha256) != binding.digest
                || object.length != binding.length
            {
                return Err(Status::invalid_argument(format!(
                    "input {} differs from its imported bytes",
                    binding.input_id
                )));
            }
            total += object.length;
            if object.length > FILE_BYTES || total > TOTAL_BYTES {
                return Err(refusal(
                    "input_too_large",
                    "file inputs are limited to 64 MiB each and 256 MiB in all",
                ));
            }
            files.push(crate::journal::InputFile {
                input_id: binding.input_id.clone(),
                digest: binding.digest.clone(),
                length: object.length,
                media_type: binding.kind_mime.clone(),
                order: binding.order,
            });
        }
        files.sort_by(|a, b| a.input_id.cmp(&b.input_id));
        Ok(files)
    }
    /// Where a published root's packages and models come from. A root naming a Hub uses the
    /// owner's delegated access there; one naming none uses a rental's own Hub with the pod's
    /// worker capability (as the Go agent and Python worker do); elsewhere there is none.
    fn hub_source(&self, actor: &str, origin: &str) -> Result<Option<crate::hub::Source>, Status> {
        if origin.is_empty() {
            return Ok(self.own_hub.clone());
        }
        match self.hub_grant(actor, origin) {
            Ok(grant) => Ok(Some(crate::hub::Source::delegated(&grant.access))),
            Err(refused) => self
                .own_hub
                .as_ref()
                .filter(|own| crate::hub::origin_key(&own.origin) == crate::hub::origin_key(origin))
                .map(|own| Some(own.clone()))
                .ok_or(refused),
        }
    }
    /// This owner's usable access at a Hub: present, bound to this leaf, unexpired.
    fn hub_grant(&self, actor: &str, origin: &str) -> Result<crate::hub::Grant, Status> {
        let key = crate::hub::origin_key(origin)
            .filter(|_| crate::hub::valid_origin(origin))
            .ok_or_else(|| Status::invalid_argument("release root names an invalid Hub origin"))?;
        let grant = self
            .service
            .engine
            .with_journal(|j| j.hub_grant(actor, &key))
            .map_err(problem)?
            .filter(|g| g.leaf == sha256::hex(&self.authority.leaf_digest))
            .ok_or_else(|| refusal("hub_access_absent", &format!("this machine holds no execution access for {origin}; the next run from a signed-in CLI delivers it")))?;
        if grant.expired(unix_now()) {
            return Err(refusal("hub_access_expired", &format!("execution access for {origin} has expired; the next run from a signed-in CLI renews it")));
        }
        Ok(grant)
    }
}
impl MachineBackend for NativeBackend {
    fn runs(&self) -> Option<Arc<crate::runs::Runs>> {
        self.runs.clone()
    }
    fn hub_access(
        &self,
        actor: VerifiedActor,
        mut access: crate::hub::Access,
    ) -> Result<(String, i64), crate::api::backend::HubAccessRefusal> {
        access.origin = access.origin.trim_end_matches('/').to_string();
        crate::hub::validate(&access, unix_now())
            .map_err(|m| (400, "invalid_access", m.to_string()))?;
        let key = crate::hub::origin_key(&access.origin).expect("validated origin");
        let (origin, expires_at) = (access.origin.clone(), access.expires_at);
        let grant = crate::hub::Grant {
            principal: crate::hub::principal(&access.token),
            leaf: sha256::hex(&self.authority.leaf_digest),
            access,
        };
        match self.service.engine.with_journal(|j| j.put_hub_grant(&actor_id(actor), &key, &grant)) {
            Ok(true) => Ok((origin, expires_at)),
            Ok(false) => Err((409, "hub_access_principal_conflict", "this machine holds another account at this Hub; remove its access with DELETE /v1/hubs/access".into())),
            Err(_) => Err((503, "hub_access_unavailable", "cannot retain the Hub access grant".into())),
        }
    }
    fn forget_hub_access(
        &self,
        actor: VerifiedActor,
        origin: &str,
    ) -> Result<(), crate::api::backend::HubAccessRefusal> {
        let key = crate::hub::origin_key(origin)
            .filter(|_| crate::hub::valid_origin(origin))
            .ok_or((
                400,
                "invalid_access",
                "send one valid Hub origin".to_string(),
            ))?;
        // Accepted work needs no Hub: removal never waits on it.
        self.service
            .engine
            .with_journal(|j| j.forget_hub_grant(&actor_id(actor), &key))
            .map_err(|_| {
                (
                    503,
                    "hub_access_unavailable",
                    "cannot remove the Hub access grant".to_string(),
                )
            })
    }
    fn begin_input_tree(
        &self,
        actor: VerifiedActor,
        header: pb::InputTreeImportHeader,
    ) -> Result<Box<dyn crate::api::backend::InputTreeReceiver>, Status> {
        crate::native_inputs::SourceIntake::begin(
            self.store.clone(),
            self.service.engine.clone(),
            &self.service.engine.root.join("input-staging"),
            &self.workspace_id(),
            actor,
            header,
        )
    }
    fn describe_runtime(&self, _: VerifiedActor) -> Result<pb::MachineRuntime, Status> {
        Ok(pb::MachineRuntime {
            wire_minor: crate::api::WIRE_MINOR,
            minimum_wire_minor: crate::api::WIRE_MINIMUM,
            tensorfs_version: tensorfs_core::VERSION.into(),
            accelerator_backend: "none".into(),
            execution_workspace_id: self.workspace_id(),
            ..Default::default()
        })
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
                pb::RunProductOp::Append,
                index.checked_sub(1).ok_or_else(absent)?,
            ),
            None => (pb::RunProductOp::Set, 0),
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
        let refs: Vec<&pb::Ref> = if current.parts.is_empty() {
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
    fn forget_package(
        &self,
        actor: VerifiedActor,
        request: pb::ForgetPackageCall,
    ) -> Result<pb::ForgetPackageResult, Status> {
        let package = request.package.trim();
        if package.is_empty() || package.len() > 256 || !package.contains('/') {
            return Err(Status::invalid_argument("package must name org/name"));
        }
        // Held installations are keyed by their exact release; the owner's bindings are the
        // model resolutions, read again on the next run.
        self.service
            .engine
            .with_journal(|j| j.forget_resolutions(&actor_id(actor), package))
            .map_err(problem)?;
        Ok(pb::ForgetPackageResult {})
    }
    fn read_triage(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionTriageQuery,
    ) -> Result<pb::MachineExecutionTriage, Status> {
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
        Ok(pb::MachineExecutionTriage {
            bundle: Some(pb::TriageBundleRef {
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
        request: pb::MachineLogQuery,
    ) -> Result<Vec<u8>, Status> {
        if request.log != pb::MachineLog::TensorfsTransport as i32 {
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
        _: pb::PackageListQuery,
    ) -> Result<pb::PackageList, Status> {
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
            let sdk = held
                .record
                .dependencies
                .into_iter()
                .filter(|d| matches!(d.name.as_str(), "cozy-runtime" | "tensorfs"))
                .map(|d| pb::ImageDistribution {
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
            packages.push(pb::MachinePackage {
                installation_id: installed.alias,
                package: installed.package,
                release: installed.release,
                origin: "local".into(),
                sdk,
                entrypoints,
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
        Ok(pb::PackageList { packages })
    }
    fn retain_bytes(
        &self,
        actor: VerifiedActor,
        request: pb::NativeByteRetentionCall,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        let _guard = self.projection.lock().unwrap();
        let request = request
            .request
            .ok_or_else(|| Status::invalid_argument("native retention request absent"))?;
        let source = request
            .source
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("native source absent"))?;
        let actor = actor_id(actor);
        if self
            .service
            .engine
            .native_owner(&request.retention_id)
            .map_err(problem)?
            .is_some_and(|held| held != actor)
        {
            return Err(Status::permission_denied(
                "native retention belongs to another actor",
            ));
        }
        let bytes = self
            .service
            .engine
            .native_output(&actor, &source.producer_root_id)
            .map_err(problem)?
            .ok_or_else(|| {
                Status::permission_denied("native producer is not owned by this actor")
            })?;
        let donor = pb::NativeByteRetentionRequest::decode(bytes.as_slice())
            .map_err(|_| Status::data_loss("native source record corrupt"))?;
        if donor.source != request.source {
            return Err(Status::invalid_argument(
                "native source differs from owned producer",
            ));
        }
        source_artifact::retain(&self.store, &donor.retention_id, &request.retention_id)
            .map_err(storage)?;
        self.service
            .engine
            .bind_native_output(&actor, &request.retention_id, &request.encode_to_vec())
            .map_err(problem)?;
        Ok(pb::NativeByteRetentionResult {
            source: request.source,
            retention_id: request.retention_id,
            released: false,
        })
    }
    fn release_bytes(
        &self,
        actor: VerifiedActor,
        request: pb::NativeByteRetentionCall,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        let _guard = self.projection.lock().unwrap();
        let request = request
            .request
            .ok_or_else(|| Status::invalid_argument("native retention request absent"))?;
        let bytes = self
            .service
            .engine
            .native_output(&actor_id(actor), &request.retention_id)
            .map_err(problem)?
            .ok_or_else(|| {
                Status::permission_denied("native recipient is not held by this actor")
            })?;
        let expected = pb::NativeByteRetentionRequest::decode(bytes.as_slice())
            .map_err(|_| Status::data_loss("native source record corrupt"))?;
        if expected != request {
            return Err(Status::invalid_argument("native release subject changed"));
        }
        source_artifact::release(&self.store, &request.retention_id).map_err(storage)?;
        Ok(pb::NativeByteRetentionResult {
            source: request.source,
            retention_id: request.retention_id,
            released: true,
        })
    }
    fn workspace(
        &self,
        _: VerifiedActor,
        query: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status> {
        if query.describe.is_some() {
            return Err(Status::unimplemented(
                "published release description is not qualified by this CPU build",
            ));
        }
        Ok(pb::MachineExecutionWorkspace {
            worker_id: self.authority.worker_id.clone(),
            worker_boot_id: self.authority.boot_id.clone(),
            execution_workspace_id: self.workspace_id(),
            accelerator_backend: "none".into(),
            run_output_log: true,
            release_root_owner: true,
            submission_close: true,
            ..Default::default()
        })
    }
    fn submit(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionSubmit,
    ) -> Result<pb::MachineExecutionReceipt, Status> {
        let root = request.release_root.as_ref().ok_or_else(|| {
            Status::unimplemented("captured offers are not qualified by this CPU build")
        })?;
        let offer = request
            .offer
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("request offer absent"))?;
        let actor = actor_id(actor);
        // A replay of accepted work needs nothing from the Hub, even after access expired.
        if let Some(prior) = self
            .service
            .engine
            .with_journal(|j| j.accepted_public(&actor, &offer.request_id, &request.submission_id))
            .map_err(problem)?
        {
            if request.expected_execution_workspace_id != self.workspace_id() {
                return Err(refusal(
                    "execution_workspace_changed",
                    "the requested execution journal is not this workspace",
                ));
            }
            return self.receipt(&prior);
        }
        if root.job
            || root.deadline_unix_ms != 0
            || root.capture.is_some()
            || !request.publication_authorization_id.is_empty()
        {
            return Err(Status::unimplemented(
                "this machine runs callable roots without deadlines or publication",
            ));
        }
        if !request.capture_digest.is_empty()
            || !request.capture_canonical_bytes.is_empty()
            || request.prepared_state.is_some()
            || !offer.invocation_spec_canonical_bytes.is_empty()
        {
            return Err(Status::invalid_argument(
                "release root carries an independently prepared offer",
            ));
        }
        let source = if root.installation_id.is_empty() {
            self.hub_source(&actor, &root.hub)?
        } else {
            None
        };
        let (installed, published_plan) = if let Some(source) = source {
            if root.package.is_empty() || root.release.is_empty() {
                return Err(Status::invalid_argument(
                    "a published root names its package and release",
                ));
            }
            let publisher = self.publisher.as_ref().ok_or_else(|| {
                Status::unimplemented(
                    "published package preparation is not configured on this machine",
                )
            })?;
            let published = crate::published::Request {
                source,
                package: root.package.clone(),
                release: root.release.clone(),
                installed: None,
                owner: String::new(),
                binding_revision: String::new(),
                providers: Default::default(),
                entrypoint: root.entrypoint.clone(),
                choices: root.models.clone(),
            };
            match publisher.prepare(&self.service, &actor, &request.submission_id, published) {
                crate::published::Progress::Ready(prepared) => {
                    (prepared.installation.clone(), Some(prepared.plan.clone()))
                }
                crate::published::Progress::Failed(code, detail) => {
                    return Err(refusal(code, &format!("{code}: {detail}")))
                }
                crate::published::Progress::Preparing {
                    stage,
                    moved,
                    total,
                } => {
                    let mut status = Status::unavailable(stage);
                    status.metadata_mut().insert(
                        "cozy-error-code",
                        "release_root_preparing".parse().expect("ASCII"),
                    );
                    if total > 0 {
                        if let Ok(value) = format!("{moved} {total}").parse() {
                            status.metadata_mut().insert("cozy-progress-bytes", value);
                        }
                    }
                    return Err(status);
                }
            }
        } else {
            let installed = if root.installation_id.is_empty() {
                self.service
                    .gpu()
                    .ok_or_else(|| {
                        Status::unimplemented(
                            "published cache-only GPU package execution is not configured",
                        )
                    })?
                    .published_installation(&self.service, &actor, &root.package, &root.release)
                    .map_err(problem)?
            } else {
                self.service
                    .engine
                    .installation(&actor, &root.installation_id)
                    .map_err(problem)?
            }
            .ok_or_else(|| {
                refusal(
                    "release_root_installation_absent",
                    "this owner has not prepared the named installation",
                )
            })?;
            if (root.installation_id.is_empty() && root.release != installed.release)
                || (!root.installation_id.is_empty() && !root.release.is_empty())
                || (!root.package.is_empty() && root.package != installed.package)
            {
                return Err(Status::invalid_argument(
                    "release root differs from its held installation",
                ));
            }
            (installed, None)
        };
        let interface: Value = serde_json::from_slice(&installed.interface)
            .map_err(|_| Status::data_loss("held installation interface is corrupt"))?;
        let entry = entrypoint(&interface, &root.entrypoint, false)?;
        let gpu_plan = if let Some(plan) = published_plan {
            plan
        } else if entry
            .get("models")
            .and_then(Value::as_array)
            .is_some_and(|models| !models.is_empty())
        {
            let gpu = self.service.gpu().ok_or_else(|| Status::unimplemented("this installed callable needs the GPU execution operation; CPU peers remain usable"))?;
            let plan = gpu
                .prepare_root(&installed, &root.entrypoint, &root.models, &[], 0)
                .map_err(problem)?;
            self.service
                .engine
                .bind_preparation(crate::journal::Preparation {
                    actor: actor.clone(),
                    id: plan.id.clone(),
                    installation: installed.alias.clone(),
                    document: serde_json::to_vec(&plan)
                        .map_err(|_| Status::internal("GPU preparation encoding failed"))?,
                })
                .map_err(problem)?;
            Some(plan)
        } else {
            if !root.models.is_empty() {
                return Err(Status::invalid_argument(
                    "model choices do not name a declared model slot",
                ));
            }
            None
        };
        let input: Value = crate::boundary_json::parse(&request.payload_canonical_bytes)
            .map_err(|_| Status::invalid_argument("payload is invalid JSON"))?;
        if !input.is_object() {
            return Err(refusal(
                "invalid_request",
                "the payload must be a JSON object of the function's parameters",
            ));
        }
        let canonical_input = canonical(&input)?;
        let payload_digest = format!("sha256:{}", sha256::hex(&sha256::digest(&canonical_input)));
        let binding = identity(entry)?;
        let mut spec = json!({"format":"cozy.worker.v1.InvocationSpec/1","installation_id":installed.alias,"payload_digest":payload_digest,"serving":{"entrypoint_binding_digest":binding,"attempt_binding_id":binding,"bindings_digest":binding}});
        if !root.attention_kernel.is_empty() {
            spec["attention_kernel"] = json!(root.attention_kernel);
        }
        if let Some(plan) = &gpu_plan {
            spec["model_preparation"] = json!(plan.id);
        }
        let inputs = self.file_inputs(&actor, &offer.request_id, root)?;
        if !inputs.is_empty() {
            if gpu_plan.is_none() {
                return Err(Status::unimplemented("file inputs reach device executors only; this CPU callable cannot take them yet"));
            }
            spec["inputs"] = serde_json::to_value(&inputs)
                .map_err(|_| Status::internal("input encoding failed"))?;
        }
        let mut capture = json!({"installation":installed.alias,"generation":installed.generation,"entrypoint":root.entrypoint,"owner":root.owner,"hub":root.hub});
        if let Some(plan) = &gpu_plan {
            capture["model_preparation"] = json!(plan.id);
        }
        let context = SubmissionContext {
            actor,
            request_id: offer.request_id.clone(),
            submission_id: request.submission_id,
            expected_workspace_id: request.expected_execution_workspace_id,
            capture_digest: identity(&capture)?,
            invocation_digest: identity(&spec)?,
            payload_digest,
            publication_authorization_id: String::new(),
            preparation_id: gpu_plan
                .as_ref()
                .map(|plan| plan.id.clone())
                .unwrap_or_default(),
        };
        let record = self
            .service
            .submit_public(
                context,
                &installed.generation,
                crate::service::Call {
                    entrypoint: root.entrypoint.clone(),
                    input,
                    attention_kernel: root.attention_kernel.clone(),
                    inputs,
                    ..Default::default()
                },
                &self.authority.boot_id,
            )
            .map_err(problem)?;
        self.receipt(&record)
    }
    fn get(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionQuery,
    ) -> Result<pb::MachineExecutionState, Status> {
        self.state(&self.query(actor, query)?)
    }
    fn measurements(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionQuery,
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
        request: pb::MachineExecutionEventsQuery,
    ) -> Result<pb::MachineExecutionEventPage, Status> {
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
         -> Result<pb::MachineExecutionEvent, Status> {
            Ok(pb::MachineExecutionEvent {
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
            events.push(event(
                record.running_revision,
                "running",
                &json!({"generation":attempt}),
            )?);
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
        Ok(pb::MachineExecutionEventPage {
            next_after: events.last().map(|e| e.sequence).unwrap_or(request.after),
            events,
            head_sequence: record.revision,
            compacted_through: record.revision.saturating_sub(1),
        })
    }
    fn control(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionControl,
    ) -> Result<pb::MachineExecutionState, Status> {
        let record = self.query(
            actor,
            request
                .execution
                .ok_or_else(|| Status::invalid_argument("execution query absent"))?,
        )?;
        if request.expected_generation != 0 && request.expected_generation != record.attempt as u64
        {
            return Err(Status::aborted("execution generation changed"));
        }
        let actor = actor_id(actor);
        let jobs = self.service.jobs();
        let typed = |refused: crate::objects::Refused| {
            refusal(refused.code, &format!("{}: {}", refused.code, refused.message))
        };
        let changed = match pb::MachineExecutionAction::try_from(request.action) {
            Ok(pb::MachineExecutionAction::Cancel) => {
                let canceled = self
                    .service
                    .engine
                    .cancel(&record.id, &actor)
                    .map_err(problem)?;
                if let Some(jobs) = &jobs {
                    jobs.canceled(&canceled);
                }
                canceled
            }
            Ok(pb::MachineExecutionAction::Pause) => jobs
                .ok_or_else(|| refusal("pause_unsupported", "this machine runs no jobs"))?
                .pause(&record, &actor)
                .map_err(typed)?,
            Ok(pb::MachineExecutionAction::Resume) => jobs
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
        request: pb::MachineExecutionListQuery,
    ) -> Result<pb::MachineExecutionList, Status> {
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
        Ok(pb::MachineExecutionList {
            executions,
            head_number,
            execution_workspace_id: self.workspace_id(),
        })
    }
    fn close_submission(
        &self,
        actor: VerifiedActor,
        request: pb::MachineSubmissionClose,
    ) -> Result<pb::MachineSubmissionClosure, Status> {
        let held = self
            .service
            .engine
            .close_submission(
                &actor_id(actor),
                &request.submission_id,
                &request.request_id,
                &request.expected_execution_workspace_id,
            )
            .map_err(problem)?;
        Ok(pb::MachineSubmissionClosure {
            submission_id: request.submission_id,
            request_id: request.request_id,
            execution_workspace_id: self.workspace_id(),
            receipt: held.as_ref().map(|r| self.receipt(r)).transpose()?,
        })
    }
    fn uploads(&self) -> Option<Arc<WorkspaceUploads>> {
        Some(self.uploads.clone())
    }
    fn collect(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionCollect,
    ) -> Result<pb::AttemptOutcome, Status> {
        let record = self.query(
            actor,
            request
                .execution
                .ok_or_else(|| Status::invalid_argument("execution query absent"))?,
        )?;
        if request.attempt_ordinal != 0 && request.attempt_ordinal != record.attempt.max(1) as u64 {
            return Err(Status::not_found(
                "requested attempt is not retained by this execution",
            ));
        }
        self.terminal(&record)?
            .events
            .into_iter()
            .find_map(|e| e.outcome)
            .ok_or_else(|| Status::data_loss("durable terminal outcome absent"))
    }
    fn ack_collection(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionCollectionAck,
    ) -> Result<pb::MachineExecutionState, Status> {
        let record = self.query(
            actor,
            request
                .execution
                .ok_or_else(|| Status::invalid_argument("execution query absent"))?,
        )?;
        let ack = request
            .outcome
            .ok_or_else(|| Status::invalid_argument("outcome acknowledgement absent"))?;
        let outcome = self
            .terminal(&record)?
            .events
            .into_iter()
            .find_map(|e| e.outcome)
            .ok_or_else(|| Status::data_loss("durable terminal outcome absent"))?;
        if !ack.worker_boot_id.is_empty() && ack.worker_boot_id != outcome.worker_boot_id {
            return Err(Status::invalid_argument(
                "acknowledgement names a different outcome boot",
            ));
        }
        if (
            ack.request_id,
            ack.attempt_ordinal,
            ack.invocation_spec_digest,
            ack.outcome_id,
            ack.outcome_digest,
        ) != (
            outcome.request_id,
            outcome.attempt_ordinal,
            outcome.invocation_spec_digest,
            outcome.outcome_id,
            outcome.outcome_digest,
        ) {
            return Err(Status::invalid_argument(
                "acknowledgement differs from the exact retained outcome",
            ));
        }
        let events = if !ack.retain_work {
            // This slice admits no asset inputs. A canceled terminal has no result/native
            // output binding, and terminal settlement has already ended its runner hold.
            // Output trees of completed runs retain their explicit source-release authority.
            let _guard = self.projection.lock().unwrap();
            let held = self
                .service
                .engine
                .public_terminal(&record.id)
                .map_err(problem)?
                .ok_or_else(|| Status::data_loss("terminal projection absent"))?;
            let mut page = pb::MachineExecutionEventPage::decode(held.events.as_slice())
                .map_err(|_| Status::data_loss("terminal event projection corrupt"))?;
            if !page
                .events
                .iter()
                .any(|event| event.kind == "retention_released")
            {
                let sequence = page
                    .head_sequence
                    .checked_add(1)
                    .ok_or_else(|| Status::resource_exhausted("event cursor exhausted"))?;
                page.events.push(pb::MachineExecutionEvent {
                    sequence,
                    attempt_ordinal: record.attempt.max(1) as u64,
                    at_ms: record.finished_at_ms,
                    kind: "retention_released".into(),
                    body_canonical_bytes: canonical(
                        &json!({"reason":"final custody acknowledged"}),
                    )?,
                    ..Default::default()
                });
                page.head_sequence = sequence;
                page.next_after = sequence;
            }
            Some(page.encode_to_vec())
        } else {
            None
        };
        self.state(
            &self
                .service
                .engine
                .acknowledge_collection_events(&record.id, events.as_deref())
                .map_err(problem)?,
        )
    }
    fn prepare_local(
        &self,
        actor: VerifiedActor,
        request: pb::PrepareLocalPackageCall,
        uploaded: Option<crate::api::workspaces::UploadedPackage>,
    ) -> Result<Vec<pb::PrepareEvent>, Status> {
        let verified_actor = actor;
        if !request.hub.is_empty() {
            return Err(Status::unimplemented(
                "scoped Hub grant routing is not qualified by this CPU installer",
            ));
        }
        let selection = request
            .local_package_set
            .as_ref()
            .and_then(|s| s.package.as_ref())
            .ok_or_else(|| Status::invalid_argument("local package metadata absent"))?;
        let alias = selection.installation_id.clone();
        let actor = actor_id(verified_actor);
        let _guard = self.installation.lock().unwrap();
        let installed = if let Some(prior) = self
            .service
            .engine
            .installation(&actor, &alias)
            .map_err(problem)?
        {
            if prior.package != selection.package || prior.release != selection.release {
                return Err(Status::already_exists(
                    "installation alias names different package semantics",
                ));
            }
            prior
        } else {
            let uploaded = uploaded.ok_or_else(|| {
                Status::failed_precondition("captured package uploads are incomplete")
            })?;
            let config = self
                .installer
                .as_ref()
                .ok_or_else(|| Status::unimplemented("package installer is not configured"))?;
            let prepared = crate::api::install::prepare_uploaded(config, &uploaded)?;
            let record = self
                .service
                .engine
                .bind_installation(crate::journal::Installation {
                    actor,
                    alias,
                    generation: prepared.record.identity,
                    package: uploaded.root.package.clone(),
                    release: uploaded.root.release.clone(),
                    interface: prepared.interface_bytes,
                })
                .map_err(problem)?;
            self.service.changed_environment().map_err(problem)?;
            self.uploads
                .release_after_install(verified_actor, &uploaded.root.operation_id)?;
            record
        };
        Ok(vec![pb::PrepareEvent {
            stage: pb::PrepareStage::Prepared as i32,
            installed_package: Some(pb::InstalledPackage {
                installation_id: installed.alias,
                package: installed.package,
                release: installed.release,
                package_interface: installed.interface,
            }),
            ..Default::default()
        }])
    }
    fn read_stream(
        &self,
        actor: VerifiedActor,
        request: pb::NativeByteReadCall,
    ) -> Result<crate::api::backend::NativeByteStream, Status> {
        let source = request
            .source
            .ok_or_else(|| Status::invalid_argument("native source absent"))?;
        let expected = self
            .service
            .engine
            .native_output(&actor_id(actor), &source.retention_id)
            .map_err(problem)?
            .ok_or_else(|| Status::not_found("native output is not retained for this actor"))?;
        let expected = pb::NativeByteRetentionRequest::decode(expected.as_slice())
            .map_err(|_| Status::data_loss("native output record corrupt"))?;
        if expected != source {
            return Err(Status::invalid_argument(
                "native source differs from its retained record",
            ));
        }
        let object = request
            .object
            .ok_or_else(|| Status::invalid_argument("byte object absent"))?;
        let root = source_artifact::read(&self.store, &source.retention_id)
            .map_err(storage)?
            .ok_or_else(|| Status::not_found("native output root absent"))?;
        if root.released || !root.complete {
            return Err(Status::not_found(
                "native recipient is released or incomplete",
            ));
        }
        let object_ref = ObjectRef {
            sha256: sha256::hex(&object.digest),
            length: object.length,
        };
        if (!root.objects.contains(&object_ref) && root.manifest != object_ref)
            || request.offset > object.length
        {
            return Err(Status::invalid_argument(
                "byte range is outside retained source",
            ));
        }
        if root.manifest == object_ref {
            let manifest = self.store.read_manifest(&object_ref).map_err(storage)?;
            let mut file = std::io::Cursor::new(manifest.canonical_bytes().map_err(storage)?);
            file.seek(SeekFrom::Start(request.offset))
                .map_err(problem)?;
            return Ok(Box::new(ByteReader {
                file: Box::new(file),
                offset: request.offset,
                remaining: object.length - request.offset,
            }));
        }
        let mut file = self
            .store
            .open_verified(&object_ref.sha256)
            .map_err(storage)?
            .into_file();
        file.seek(SeekFrom::Start(request.offset))
            .map_err(problem)?;
        Ok(Box::new(ByteReader {
            file: Box::new(file),
            offset: request.offset,
            remaining: object.length - request.offset,
        }))
    }
}
struct ByteReader {
    file: Box<dyn Read + Send>,
    offset: u64,
    remaining: u64,
}
impl Iterator for ByteReader {
    type Item = Result<pb::NativeByteReadChunk, Status>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        // worker.proto MaxNativeByteReadChunkBytes: clients refuse larger chunks.
        let mut data = vec![0; self.remaining.min(32 << 10) as usize];
        if let Err(error) = self.file.read_exact(&mut data) {
            self.remaining = 0;
            return Some(Err(problem(error)));
        }
        let offset = self.offset;
        self.offset += data.len() as u64;
        self.remaining -= data.len() as u64;
        Some(Ok(pb::NativeByteReadChunk { offset, data }))
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
        pb::NativeByteRetentionRequest,
    ),
>;
fn rewrite_assets(
    value: &mut Value,
    schema: &Value,
    path: &str,
    op: pb::RunProductOp,
    index: u32,
    sources: &AssetSources,
    products: &mut Vec<pb::RunProduct>,
) -> Result<(), Status> {
    if value.is_null() {
        return Ok(());
    }
    if schema.get("asset").is_some() {
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
            products.push(pb::RunProduct {
                output: path.into(),
                op: op as i32,
                index,
                content: Some(pb::Ref {
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
                    pb::RunProductOp::Append,
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
fn product_document(product: &pb::RunProduct) -> Result<Vec<u8>, Status> {
    canonical(
        &crate::products::document(product)
            .map_err(|_| Status::data_loss("product reference absent"))?,
    )
}

fn output_entries(products: &[pb::RunProduct]) -> Result<Vec<Value>, Status> {
    let mut entries = std::collections::BTreeMap::new();
    for product in products {
        let document: Value = serde_json::from_slice(&product_document(product)?)
            .map_err(|_| Status::internal("product document is invalid"))?;
        let output_id = if product.op == pb::RunProductOp::Append as i32 {
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
        journal::{Installation, Invocation},
    };
    use std::{
        collections::BTreeMap,
        fs::File,
        path::{Path, PathBuf},
        process::Command,
    };
    use tensorfs_core::manifest::{Draft, Entry};

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

    fn products(page: &pb::MachineExecutionEventPage) -> Vec<(u64, String, i32, u32, String)> {
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

    #[test]
    fn imported_file_inputs_are_verified_at_submit_and_granted_to_the_executor() {
        let root = std::env::temp_dir().join(format!("cm-inputs-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let identity = install_named(&root, "cpu_input");
        let state = root.join("state");
        let service = Service::open(&state, &root.join("generations"), 1).unwrap();
        assert!(service.stop().unwrap());
        let store = Arc::new(Store::ensure(&state.join("tensorfs")).unwrap());
        let signer = ed25519_dalek::SigningKey::from_bytes(&[43; 32]);
        let machine =
            MachineIdentity::ephemeral("inputs".into(), vec![signer.verifying_key()], vec![7; 32])
                .unwrap();
        let actor = VerifiedActor {
            public_key: signer.verifying_key().to_bytes(),
        };
        let backend = NativeBackend::new(
            service.clone(),
            machine.authority.clone(),
            store.clone(),
            WorkspaceUploads::open(&state.join("uploads"), store.clone()).unwrap(),
        );
        // The CLI's import: a one-file tree whose member is `payload`.
        let bytes = b"%PDF-1.4 a document the package reads".to_vec();
        let object = ObjectRef::of(&bytes);
        let tree = Draft {
            entries: vec![("payload".into(), Entry::File(object.clone()))],
        }
        .seal()
        .unwrap();
        let manifest = tree.object_ref().unwrap();
        let reference = |o: &ObjectRef| pb::Ref {
            digest: digest_bytes(&format!("sha256:{}", o.sha256)).unwrap(),
            length: o.length,
        };
        let import = |request: &str| {
            let mut receiver = backend
                .begin_input_tree(
                    actor,
                    pb::InputTreeImportHeader {
                        request_id: request.into(),
                        input_id: "document".into(),
                        manifest: Some(reference(&manifest)),
                        manifest_canonical_bytes: StoredDoc::canonical_bytes(&tree).unwrap(),
                        content_bytes: object.length,
                        ..Default::default()
                    },
                )
                .unwrap();
            receiver
                .blob(pb::InputTreeImportBlob {
                    object: Some(reference(&object)),
                    offset: 0,
                    data: bytes.clone(),
                })
                .unwrap();
            receiver
                .commit(pb::InputTreeImportCommit { abort: false })
                .unwrap()
        };
        let held = import("request-1");
        let release_root = |digest: String| pb::ReleaseRoot {
            inputs: vec![pb::InputBinding {
                input_id: "document".into(),
                digest,
                length: object.length,
                kind_mime: "application/pdf".into(),
                order: 0,
            }],
            input_access: vec![pb::InputAccess {
                input_id: "document".into(),
                native_tree: Some(pb::NativeByteRetentionRequest {
                    source: held.source.clone(),
                    retention_id: held.retention_id.clone(),
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let actor_key = actor_id(actor);
        let inputs = backend
            .file_inputs(
                &actor_key,
                "request-1",
                &release_root(format!("sha256:{}", object.sha256)),
            )
            .unwrap();
        assert_eq!(inputs.len(), 1);
        // Another request cannot borrow this import, and the binding must name its bytes.
        let other = backend
            .file_inputs(
                &actor_key,
                "request-2",
                &release_root(format!("sha256:{}", object.sha256)),
            )
            .unwrap_err();
        assert_eq!(
            other.metadata().get("cozy-error-code").unwrap(),
            "input_unprepared"
        );
        assert!(backend
            .file_inputs(
                &actor_key,
                "request-1",
                &release_root(format!("sha256:{}", "b".repeat(64)))
            )
            .is_err());

        let held_generation = service.catalog.resolve(&identity).unwrap();
        let executor_root = root.join("executor");
        std::fs::create_dir(&executor_root).unwrap();
        let mut executor = DeviceExecutor::spawn(ExecutorConfig {
            python: held_generation.record.python.clone(),
            root: executor_root.clone(),
            socket: executor_root.join("e.sock"),
            environment: BTreeMap::from([("PATH".into(), "/usr/bin:/bin".into())]),
            seal: {
                let mut seal = crate::launch_identity::Seal::prepare(
                    &executor_root,
                    None,
                    "inputs",
                    &identity,
                    "",
                )
                .unwrap();
                seal.threads = 1;
                seal
            },
            generation_hold: Some(held_generation.retention()),
            identity: None,
            cgroup_namespace: None,
        })
        .unwrap();
        let interface = executor_root.join("package-interface.json");
        std::fs::write(
            &interface,
            serde_json::to_vec(&held_generation.record.interface).unwrap(),
        )
        .unwrap();
        let application = held_generation.record.application.clone();
        command(
            &mut executor,
            &DeviceCommand::Start {
                devices: String::new(),
                application: application.clone(),
                package_interface: interface.clone(),
                sequence_parallel_degree: 1,
                import_only: false,
            },
        )
        .unwrap();
        command(
            &mut executor,
            &DeviceCommand::Load {
                construction: "inputs".into(),
                devices: String::new(),
                sequence_parallel_degree: 1,
                binding: Box::new(Binding {
                    application,
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
                pinned_bytes: None,
            },
        )
        .unwrap();
        command(
            &mut executor,
            &DeviceCommand::Activate {
                construction: "inputs".into(),
            },
        )
        .unwrap();
        let payload = json!({"document": format!("sha256:{}", object.sha256)});
        command(&mut executor, &DeviceCommand::PrepareRequest {
            request_id: "run-1".into(),
            construction: "inputs".into(),
            entrypoint: "measure".into(),
            payload,
            attention_kernel: String::new(),
            input_metadata: inputs.iter().map(|i| (i.input_id.clone(), json!({"input_id":i.input_id,"media_type":i.media_type,"digest":i.digest,"length":i.length,"order":i.order}))).collect(),
        })
        .unwrap();
        let spool = root.join("spool");
        std::fs::create_dir(&spool).unwrap();
        let granted = crate::gpu_service::stage_inputs(&store, None, &spool, &inputs).unwrap();
        let reply = executor
            .command(
                &DeviceCommand::Invoke {
                    request_id: "run-1".into(),
                    construction: "inputs".into(),
                    entrypoint: "measure".into(),
                    spool: spool.clone(),
                    deadline_s: None,
                    attention_kernel: String::new(),
                    plane_budget_bytes: -1,
                    stages: false,
                    cap_bytes: None,
                    inputs: granted.inputs,
                    trees: granted.trees,
                    floor_bytes: None,
                    activation_bytes: Default::default(),
                    device_weights: None,
                },
                &mut Baseline,
            )
            .unwrap();
        let result = device_executor::read_result(&spool, &reply)
            .unwrap_or_else(|e| panic!("{e}: {:?}", reply.outcome));
        assert_eq!(
            result,
            json!({"length": bytes.len(), "sha256": object.sha256, "media_type": "application/pdf"})
        );
        executor.shutdown().unwrap();
        let _ = std::fs::remove_dir_all(&root);
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
            WorkspaceUploads::open(&state.join("uploads"), store.clone()).unwrap(),
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
                        engine.finish(&id, crate::journal::Outcome::Failed(failure.encode()))?;
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
        fn query(&self, after: u64) -> pb::MachineExecutionEventsQuery {
            pb::MachineExecutionEventsQuery {
                execution: Some(self.execution()),
                after,
                limit: 0,
                wait: true,
            }
        }
        fn execution(&self) -> pb::MachineExecutionQuery {
            pb::MachineExecutionQuery {
                request_id: "request-1".into(),
                expected_execution_workspace_id: self.engine.workspace_id(),
                ..Default::default()
            }
        }
        fn terminal_page(&self) -> pb::MachineExecutionEventPage {
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
            pb::RunProductOp::Append as i32,
            pb::RunProductOp::Set as i32,
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
        // Every product's bytes are fetchable with the actor's own Claim.
        let bytes: Vec<Vec<u8>> = page
            .events
            .iter()
            .filter_map(|event| event.product.as_ref())
            .map(|product| {
                backend
                    .read_stream(
                        actor,
                        pb::NativeByteReadCall {
                            source: product.source.clone(),
                            object: product.content.clone(),
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .flat_map(|chunk| chunk.unwrap().data)
                    .collect()
            })
            .collect();
        assert_eq!(
            bytes,
            [
                &b"frame-1"[..],
                b"preview-draft",
                b"frame-2",
                b"frame-3",
                b"preview-final"
            ]
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
                pb::MachineExecutionTriageQuery {
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
            pb::MachineExecutionTriageQuery {
                execution: Some(fixture.execution()),
                attempt_ordinal: 9,
            },
        );
        assert_eq!(later.unwrap_err().code(), tonic::Code::NotFound);
    }
}
