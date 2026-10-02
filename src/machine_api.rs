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
    ids::ObjectRef,
    manifest::{Draft, Entry},
    sha256, source_artifact,
    store::{Fault, Store},
};
use tonic::Status;

pub struct NativeBackend {
    pub service: Arc<Service>,
    pub authority: Authority,
    pub store: Arc<Store>,
    pub uploads: Arc<WorkspaceUploads>,
    // Serialize native projection, not inference or observation. Only one result
    // projection may establish a given immutable output's native custody at once.
    projection: Mutex<()>,
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
            projection: Mutex::new(()),
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
        Ok(pb::MachineExecutionState {
            request_id: context.request_id.clone(),
            attempt_ordinal: record.attempt.max(1) as u64,
            generation: record.attempt as u64,
            state: match record.state {
                State::Completed => "succeeded",
                State::Failed => "failed",
                State::Canceled => "canceled",
                State::Queued => "queued",
                State::Starting => "starting",
                State::Running => "running",
            }
            .into(),
            sequence: record.revision,
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
        if let Some(result) = &record.result {
            let held = self
                .service
                .catalog
                .resolve(&record.invocation.generation)
                .map_err(problem)?;
            let declaration = entrypoint(&held.record.interface, &record.invocation.entrypoint)?;
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
                self.store
                    .put_stream_held(&mut source, Some(&object), &Fault::default(), Some(&owner))
                    .map_err(storage)?;
                let tree = Draft {
                    entries: vec![("payload".into(), Entry::File(object))],
                }
                .seal()
                .map_err(storage)?;
                let root = source_artifact::create(&self.store, &owner, &tree).map_err(storage)?;
                let receipt = root.receipt().map_err(storage)?;
                let native = pb::NativeByteRetentionRequest {
                    source: Some(pb::NativeByteTreeRef {
                        producer_root_id: root.producer,
                        receipt_digest: sha256::digest(&receipt).to_vec(),
                        manifest: Some(pb::Ref {
                            digest: digest_bytes(&format!("sha256:{}", root.manifest.sha256))?,
                            length: root.manifest.length,
                        }),
                        content_bytes: artifact.length,
                    }),
                    retention_id: owner.clone(),
                };
                self.service
                    .engine
                    .bind_native_output(&context.actor, &owner, &native.encode_to_vec())
                    .map_err(problem)?;
                references.insert(
                    binding.asset_ref.clone(),
                    (artifact.clone(), binding.media_type.clone(), native),
                );
            }
            if let Some(value) = &mut value {
                rewrite_assets(value, "", &references, &mut products)?;
            }
        }
        let mut events = vec![];
        let first_sequence = record
            .revision
            .checked_add(1)
            .ok_or_else(|| Status::resource_exhausted("event cursor exhausted"))?;
        for (index, product) in products.into_iter().enumerate() {
            events.push(pb::MachineExecutionEvent {
                sequence: first_sequence + index as u64,
                attempt_ordinal: record.attempt.max(1) as u64,
                at_ms: record.finished_at_ms,
                kind: "product".into(),
                product: Some(product),
                ..Default::default()
            });
        }
        let mut body = json!({"format":"cozy.worker.v1.AttemptOutcomeBody/1", "request_id":context.request_id, "attempt_ordinal":record.attempt.max(1), "invocation_spec_digest":context.invocation_digest});
        if record.process.is_some() {
            body["execution_started"] = json!(true);
        }
        let (status, code, origin, message) = match record.state {
            State::Completed => (1, 0, 2, "completed"),
            State::Canceled => (4, 11, 6, "explicitly canceled"),
            _ => (
                3,
                7,
                3,
                "executor failed; inspect the private execution diagnostic",
            ),
        };
        body["status"] = json!(status);
        body["cause"] = if code == 0 {
            json!({"origin":origin})
        } else {
            json!({"code":code,"origin":origin})
        };
        body["safe_message"] = json!(message);
        if let Some(value) = value {
            body["result"] = json!({"result_schema_digest":schema_digest, "inline_result":STANDARD.encode(canonical(&value)?)});
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
        let sequence = first_sequence + events.len() as u64;
        events.push(pb::MachineExecutionEvent {
            sequence,
            attempt_ordinal: outcome.attempt_ordinal,
            at_ms: record.finished_at_ms,
            kind: "outcome".into(),
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
impl MachineBackend for NativeBackend {
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
        if !root.models.is_empty()
            || !root.inputs.is_empty()
            || !root.input_access.is_empty()
            || root.job
            || root.deadline_unix_ms != 0
            || !root.attention_kernel.is_empty()
            || root.capture.is_some()
            || !request.publication_authorization_id.is_empty()
        {
            return Err(Status::unimplemented("this CPU vertical slice supports weightless callable roots without asset inputs, deadlines or publication"));
        }
        let actor = actor_id(actor);
        let installed = self
            .service
            .engine
            .installation(&actor, &root.installation_id)
            .map_err(problem)?
            .ok_or_else(|| {
                refusal(
                    "release_root_installation_absent",
                    "this owner has not prepared the named installation",
                )
            })?;
        if !root.release.is_empty()
            || (!root.package.is_empty() && root.package != installed.package)
        {
            return Err(Status::invalid_argument(
                "release root differs from its held installation",
            ));
        }
        let offer = request
            .offer
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("request offer absent"))?;
        if !request.capture_digest.is_empty()
            || !request.capture_canonical_bytes.is_empty()
            || request.prepared_state.is_some()
            || !offer.invocation_spec_canonical_bytes.is_empty()
        {
            return Err(Status::invalid_argument(
                "release root carries an independently prepared offer",
            ));
        }
        let interface: Value = serde_json::from_slice(&installed.interface)
            .map_err(|_| Status::data_loss("held installation interface is corrupt"))?;
        let entry = entrypoint(&interface, &root.entrypoint)?;
        let input: Value = serde_json::from_slice(&request.payload_canonical_bytes)
            .map_err(|_| Status::invalid_argument("payload is invalid JSON"))?;
        let payload_digest = identity(&input)?;
        let binding = identity(entry)?;
        let spec = json!({"format":"cozy.worker.v1.InvocationSpec/1","installation_id":installed.alias,"payload_digest":payload_digest,"serving":{"entrypoint_binding_digest":binding,"attempt_binding_id":binding,"bindings_digest":binding}});
        let context = SubmissionContext {
            actor,
            request_id: offer.request_id.clone(),
            submission_id: request.submission_id,
            expected_workspace_id: request.expected_execution_workspace_id,
            capture_digest: identity(
                &json!({"installation":installed.alias,"generation":installed.generation,"entrypoint":root.entrypoint,"owner":root.owner,"hub":root.hub}),
            )?,
            invocation_digest: identity(&spec)?,
            payload_digest,
            publication_authorization_id: String::new(),
        };
        let record = self
            .service
            .submit_public(
                context,
                &installed.generation,
                &root.entrypoint,
                input,
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
        let events = if record.revision > request.after {
            vec![pb::MachineExecutionEvent {
                sequence: record.revision,
                attempt_ordinal: record.attempt.max(1) as u64,
                at_ms: record.accepted_at_ms,
                kind: "state".into(),
                body_canonical_bytes: canonical(
                    &json!({"state":self.state(&record)?.state,"completed_units":record.completed_units,"waiting_reason":record.waiting_reason}),
                )?,
                ..Default::default()
            }]
        } else {
            vec![]
        };
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
        if request.action != pb::MachineExecutionAction::Cancel as i32 {
            return Err(Status::unimplemented(
                "this slice implements explicit cancellation",
            ));
        }
        if request.expected_generation != 0 && request.expected_generation != record.attempt as u64
        {
            return Err(Status::aborted("execution generation changed"));
        }
        self.state(
            &self
                .service
                .engine
                .cancel(&record.id, &actor_id(actor))
                .map_err(problem)?,
        )
    }
    fn list(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionListQuery,
    ) -> Result<pb::MachineExecutionList, Status> {
        let records = self
            .service
            .engine
            .list_actor(&actor_id(actor), usize::MAX)
            .map_err(problem)?;
        let head_number = records.last().and_then(|r| r.id.parse().ok()).unwrap_or(0);
        let mut executions = records
            .iter()
            .map(|r| self.state(r))
            .collect::<Result<Vec<_>, _>>()?;
        executions.retain(|r| {
            if request.newest_first {
                (request.before_number == 0 || r.number < request.before_number)
                    && (request.states.is_empty() || request.states.contains(&r.state))
            } else {
                r.number > request.after_number
                    && (request.states.is_empty() || request.states.contains(&r.state))
            }
        });
        if request.newest_first {
            executions.reverse();
        }
        executions.truncate(if request.limit == 0 {
            64
        } else {
            request.limit.min(256)
        } as usize);
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
    fn read_bytes(
        &self,
        actor: VerifiedActor,
        request: pb::NativeByteReadCall,
    ) -> Result<Vec<pb::NativeByteReadChunk>, Status> {
        let source = request
            .source
            .ok_or_else(|| Status::invalid_argument("native source absent"))?;
        let expected = self
            .service
            .engine
            .native_output(&actor_id(actor), &source.retention_id)
            .map_err(problem)?
            .ok_or_else(|| Status::not_found("native output is not retained for this actor"))?;
        if expected != source.encode_to_vec() {
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
        let object_ref = ObjectRef {
            sha256: sha256::hex(&object.digest),
            length: object.length,
        };
        if !root.objects.contains(&object_ref) || request.offset > object.length {
            return Err(Status::invalid_argument(
                "byte range is outside retained source",
            ));
        }
        let mut file = self
            .store
            .open_verified(&object_ref.sha256)
            .map_err(storage)?
            .into_file();
        file.seek(SeekFrom::Start(request.offset))
            .map_err(problem)?;
        let mut chunks = vec![];
        let mut offset = request.offset;
        loop {
            let mut data = vec![0; 1 << 20];
            let count = file.read(&mut data).map_err(problem)?;
            if count == 0 {
                break;
            }
            data.truncate(count);
            chunks.push(pb::NativeByteReadChunk { offset, data });
            offset += count as u64;
        }
        Ok(chunks)
    }
}
fn actor_id(actor: VerifiedActor) -> String {
    sha256::hex(&actor.public_key)
}
fn verify_checksum(source: &mut std::fs::File, binding: &crate::journal::AssetBinding) -> Result<(), Status> {
    use blake2::digest::{Update, VariableOutput};
    let mut blake = blake2::Blake2bVar::new(16).map_err(|_| Status::internal("checksum configuration invalid"))?;
    let mut sha = sha256::Sha256::new();
    let mut length = 0;
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = source.read(&mut buffer).map_err(problem)?;
        if count == 0 { break; }
        match binding.checksum.algorithm.as_str() {
            "blake2b-128" => Update::update(&mut blake, &buffer[..count]),
            "sha256" => sha.update(&buffer[..count]),
            _ => return Err(Status::unimplemented("producer output checksum algorithm is unavailable")),
        }
        length += count as u64;
    }
    let digest = match binding.checksum.algorithm.as_str() {
        "blake2b-128" => { let mut bytes = [0;16]; blake.finalize_variable(&mut bytes).map_err(|_|Status::internal("checksum finalization invalid"))?; sha256::hex(&bytes) },
        "sha256" => sha256::hex(&sha.finish()),
        _ => return Err(Status::unimplemented("producer output checksum algorithm is unavailable")),
    };
    source.seek(SeekFrom::Start(0)).map_err(problem)?;
    if length != binding.length || digest != binding.checksum.value {return Err(Status::data_loss("SDK output checksum differs from held result bytes"));}
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
fn entrypoint<'a>(interface: &'a Value, name: &str) -> Result<&'a Value, Status> {
    interface
        .get("entrypoints")
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
    path: &str,
    sources: &AssetSources,
    products: &mut Vec<pb::RunProduct>,
) -> Result<(), Status> {
    match value {
        Value::Object(object) => {
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
                    op: pb::RunProductOp::Set as i32,
                    content: Some(pb::Ref {
                        digest: digest_bytes(&format!("sha256:{}", artifact.sha256))?,
                        length: artifact.length,
                    }),
                    media_type: mime.clone(),
                    source: Some(source.clone()),
                    ..Default::default()
                });
            } else {
                for (key, child) in object {
                    let child_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    rewrite_assets(child, &child_path, sources, products)?;
                }
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter_mut().enumerate() {
                rewrite_assets(child, &format!("{path}[{index}]"), sources, products)?;
            }
        }
        _ => (),
    }
    Ok(())
}
