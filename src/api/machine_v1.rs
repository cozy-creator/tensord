//! `cozy.machine.v1` (G/API.md): Run, Control and Read over the machine's engine, every call
//! authorized by one `Cozy-Cap`. Until `worker.v1` is deleted at cutover this service reuses
//! the backend's operations; Status and `Run kind: update` are `machine_status` and
//! `machine_update` (D2); Write (D1) answers UNIMPLEMENTED until it lands.
use super::{
    auth::{StreamAuthority, VerifiedActor},
    backend::MachineBackend,
    capability::{self, Grant},
    pb, v1, MachineIdentity,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::Value;
use std::{
    collections::HashMap,
    io::Read,
    pin::Pin,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_stream::{Stream, StreamExt};
use tonic::{metadata::MetadataMap, Request, Response, Status};

type Events<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

pub(super) struct MachineV1<B> {
    pub identity: Arc<MachineIdentity>,
    pub backend: Arc<B>,
}

/// Who may do what: a machine-scope cap does everything as its signer; a run-scope cap may only
/// attach to and read that run.
struct Caller {
    actor: VerifiedActor,
    grant: Grant,
}
impl Caller {
    fn machine(&self) -> Result<(), Status> {
        if self.grant.action == capability::MACHINE {
            Ok(())
        } else {
            Err(Status::permission_denied(
                "this call needs a machine-scope Cozy-Cap",
            ))
        }
    }
    /// An output-limited run grant (`run play`'s link), which sees only those outputs.
    fn limit(&self) -> Option<Grant> {
        let limited = self.grant.action != capability::MACHINE && !self.grant.outputs.is_empty();
        limited.then(|| self.grant.clone())
    }
    fn run(&self, id: &str, output: Option<(&str, Option<u32>)>) -> Result<(), Status> {
        let allowed = self.grant.action == capability::MACHINE
            || match output {
                Some((name, index)) => self.grant.allows(id, name, index),
                None => self.grant.run == id,
            };
        if allowed {
            Ok(())
        } else {
            Err(Status::permission_denied(
                "the capability does not grant this",
            ))
        }
    }
}

/// What an output-limited grant sees of an event: its outputs' products, and the outcome's
/// status and inventory of them. Results, reasons, triage, measurements, waiting details,
/// progress and logs stay the run owner's.
fn visible(grant: &Grant, event: &mut v1::RunEvent) -> bool {
    let allows = |p: &v1::Product| grant.allows(&grant.run, &p.output, (p.index > 0).then_some(p.index));
    match &mut event.event {
        Some(v1::run_event::Event::Product(product)) => allows(&*product),
        Some(v1::run_event::Event::Outcome(outcome)) => {
            outcome.outputs.retain(allows);
            outcome.result.clear();
            outcome.reason = None;
            outcome.triage = false;
            outcome.measurements.clear();
            true
        }
        Some(v1::run_event::Event::State(state)) => {
            state.waiting.clear();
            true
        }
        _ => false,
    }
}

impl<B: MachineBackend> MachineV1<B> {
    /// A v1 stream lives until its key stops authorizing; the cap's expiry held at open.
    fn authority(&self, caller: &Caller) -> StreamAuthority {
        StreamAuthority::new(self.identity.authority.keys.clone(), caller.actor, None)
    }

    fn caller(&self, metadata: &MetadataMap) -> Result<Caller, Status> {
        let token = metadata
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Cozy-Cap "))
            .ok_or_else(|| Status::unauthenticated("the call carries no Cozy-Cap"))?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() as i64);
        let authority = &self.identity.authority;
        let (grant, signer) = capability::verify_signer(
            token,
            &authority.worker_id,
            &authority.keys.admitted(),
            now,
            "",
        )
        .map_err(|refusal| Status::unauthenticated(refusal.to_string()))?;
        Ok(Caller {
            actor: VerifiedActor {
                public_key: signer.to_bytes(),
            },
            grant,
        })
    }

    async fn call<T: Send + 'static>(
        backend: &Arc<B>,
        f: impl FnOnce(&B) -> Result<T, Status> + Send + 'static,
    ) -> Result<T, Status> {
        let backend = backend.clone();
        tokio::task::spawn_blocking(move || f(&backend))
            .await
            .map_err(|_| Status::internal("machine operation stopped"))?
    }
}

pub(super) fn query(
    backend: &impl MachineBackend,
    actor: VerifiedActor,
    id: &str,
) -> Result<pb::MachineExecutionQuery, Status> {
    let workspace = backend.workspace(actor, pb::MachineExecutionWorkspaceQuery::default())?;
    Ok(pb::MachineExecutionQuery {
        request_id: id.into(),
        expected_execution_workspace_id: workspace.execution_workspace_id,
        ..Default::default()
    })
}

/// The run's state as the log's first frame (sequence 0: a snapshot, not a log entry).
pub(super) fn state(id: &str, state: &pb::MachineExecutionState) -> v1::RunState {
    v1::RunState {
        id: id.into(),
        number: state.number,
        state: match state.state.as_str() {
            "starting" => "queued".into(),
            other => other.into(),
        },
        sequence: state.sequence,
        attempt: state.attempt_ordinal as u32,
        waiting: String::new(),
    }
}

/// A Run spec as this machine's run sources (`runs`). Its digest (the token cleared) makes the
/// id idempotent: the same id with another spec is refused.
fn spec_of(mut spec: v1::RunSpec) -> Result<crate::runs::Spec, Status> {
    let (warm, job) = match v1::RunKind::try_from(spec.kind) {
        Ok(v1::RunKind::Call) => (false, false),
        Ok(v1::RunKind::Warm) => (true, false),
        Ok(v1::RunKind::Job) => (false, true),
        _ => {
            return Err(Status::unimplemented(
                "Run takes calls, jobs, warm-ups and updates",
            ))
        }
    };
    let input: Value = if spec.payload.is_empty() {
        serde_json::json!({})
    } else {
        crate::boundary_json::parse(&spec.payload)
            .map_err(|_| Status::invalid_argument("the payload is not JSON"))?
    };
    let hub = spec.hub.take().filter(|hub| !hub.token.is_empty());
    let providers = spec.providers.take();
    // What the run is, not how it reaches its sources: a spec re-sent with refreshed access,
    // binding hints, known results or a reformatted payload names the same run.
    let identity_digest = {
        let mut identity = spec.clone();
        identity.payload = crate::boundary_json::exact(&input);
        identity.hub = hub.as_ref().map(|hub| v1::HubAccess {
            origin: hub.origin.trim_end_matches('/').into(),
            ..Default::default()
        });
        identity.binding_revision.clear();
        identity.known_results.clear();
        identity.publication.clear();
        format!(
            "sha256:{}",
            tensorfs_core::sha256::hex_digest(&prost::Message::encode_to_vec(&identity))
        )
    };
    let hub = match hub {
        None => None,
        Some(hub) => {
            if !crate::hub::valid_origin(&hub.origin) {
                return Err(Status::invalid_argument("the run's Hub origin is invalid"));
            }
            Some(crate::hub::Source {
                origin: hub.origin.trim_end_matches('/').to_string(),
                credential: format!("bearer {}", hub.token),
                ca_der: (!hub.ca_der.is_empty()).then_some(hub.ca_der),
                object_hosts: hub.object_hosts,
            })
        }
    };
    let source = match spec.source {
        Some(v1::run_spec::Source::Release(release)) => crate::runs::Source::Release {
            package: release.package,
            release: release.release,
        },
        Some(v1::run_spec::Source::Installation(alias)) => crate::runs::Source::Installation(alias),
        Some(v1::run_spec::Source::Local(local)) => crate::runs::Source::Local(local.manifest),
        // A warm run of model choices alone makes them (and uploads to its destination).
        None if warm => crate::runs::Source::Models,
        None => return Err(Status::invalid_argument("a run spec names its source")),
    };
    let models = spec
        .models
        .into_iter()
        .map(|choice| -> Result<pb::ModelChoice, Status> {
            Ok(pb::ModelChoice {
                parameter: choice.parameter,
                repository: choice.repository,
                release: choice.release,
                lane: choice.lane,
                manifest: if choice.manifest.is_empty() {
                    None
                } else {
                    Some(pb::Ref {
                        digest: digest(&choice.manifest)?,
                        length: choice.manifest_length,
                    })
                },
                source: choice.source,
                profiles: choice.profiles,
                adapters: choice
                    .adapters
                    .into_iter()
                    .map(|a| pb::DownloadAdapterRef {
                        component: a.component,
                        model: a.model,
                        release: a.release,
                        lane: a.lane,
                        manifest: a.manifest,
                        scale: a.scale,
                        source: a.source,
                        profiles: a.profiles,
                        ..Default::default()
                    })
                    .collect(),
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(crate::runs::Spec {
        warm,
        job,
        parent: String::new(),
        source,
        entrypoint: spec.entrypoint,
        input,
        inputs: spec
            .inputs
            .into_iter()
            .map(|file| crate::journal::InputFile {
                input_id: file.field,
                digest: file.digest,
                length: file.length,
                media_type: file.media_type,
                order: file.order,
            })
            .collect(),
        models,
        binding_revision: spec.binding_revision,
        attention_kernel: spec.attention_kernel,
        hub,
        providers: providers
            .map(|p| crate::published::Providers {
                huggingface: p.huggingface,
                civitai: p.civitai,
            })
            .unwrap_or_default(),
        weights_destination: spec.weights_destination,
        publication: spec.publication,
        owner: spec.owner,
        digest: identity_digest,
    })
}

/// A typed refusal: the status carries its code as `cozy-error-code`, its message as text.
fn refused(code: &str, message: &str) -> Status {
    let status = match code {
        "run_id_conflict" => Status::already_exists(format!("{code}: {message}")),
        c if c.starts_with("invalid") => Status::invalid_argument(format!("{code}: {message}")),
        "object_storage_failed" => Status::unavailable(format!("{code}: {message}")),
        _ => Status::failed_precondition(format!("{code}: {message}")),
    };
    let mut status = status;
    if let Ok(value) = code.parse() {
        status.metadata_mut().insert("cozy-error-code", value);
    }
    status
}
fn refusal(refused_: crate::objects::Refused) -> Status {
    refused(refused_.code, &refused_.message)
}

fn digest(text: &str) -> Result<Vec<u8>, Status> {
    let hex = text
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| Status::invalid_argument("a digest is sha256:<64 hex>"))?;
    (0..64)
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|_| Status::invalid_argument("a digest is hex"))
        })
        .collect()
}

fn spell(bytes: &[u8]) -> String {
    format!("sha256:{}", tensorfs_core::sha256::hex(bytes))
}

/// The outcome's typed reason: the code its message leads with, and who caused it.
fn reason(body: &Value) -> v1::Reason {
    let message = body["safe_message"].as_str().unwrap_or_default();
    let code = message
        .split_once(": ")
        .map(|(code, _)| code)
        .filter(|code| {
            !code.is_empty()
                && code
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b == b'_' || b == b'.')
        })
        .unwrap_or(if body["status"].as_u64() == Some(4) {
            "canceled"
        } else {
            "failed"
        });
    let origin = match body["cause"]["origin"].as_u64() {
        Some(1) => "author",
        Some(2) => "runtime",
        Some(6) => "request",
        _ => "machine",
    };
    v1::Reason {
        code: code.into(),
        message: message.into(),
        origin: origin.into(),
    }
}

/// One revision's parts: (digest, length) each.
type Parts = Vec<(Vec<u8>, u64)>;

/// Translates the engine's event pages into the run's log, numbering each output's revisions.
#[derive(Default)]
struct Log {
    revisions: HashMap<(String, u32), u64>,
    products: Vec<v1::Product>,
    /// Each item's latest revision as parts (digest, length): a revision whose parts extend
    /// them appends.
    parts: HashMap<(String, u32), Parts>,
}
impl Log {
    fn event(&mut self, event: pb::MachineExecutionEvent) -> Option<v1::RunEvent> {
        let body: Value =
            serde_json::from_slice(&event.body_canonical_bytes).unwrap_or(Value::Null);
        let kind = match event.kind.as_str() {
            "state" | "running" => v1::run_event::Event::State(v1::RunState {
                state: body["state"]
                    .as_str()
                    .unwrap_or("running")
                    .replace("starting", "queued"),
                sequence: event.sequence,
                attempt: event.attempt_ordinal as u32,
                waiting: body["waiting_reason"].as_str().unwrap_or_default().into(),
                ..Default::default()
            }),
            "progress" => {
                let payload = &body["payload"];
                v1::run_event::Event::Progress(v1::Progress {
                    step_ms: payload["step_ms"].as_f64().unwrap_or(0.0),
                    stage: payload["stage"].as_str().unwrap_or_default().into(),
                    fraction: payload["overall_fraction"].as_f64().unwrap_or(-1.0),
                    completed: payload["position"].as_u64().unwrap_or(0),
                    total: payload["total"].as_u64().unwrap_or(0),
                    bytes_done: payload["bytes_done"].as_u64().unwrap_or(0),
                    bytes_total: payload["bytes_total"].as_u64().unwrap_or(0),
                    stage_fraction: payload["stage_fraction"].as_f64(),
                })
            }
            "product" => {
                let product = event.product?;
                let list = product.op == pb::RunProductOp::Append as i32;
                let index = if list { product.index + 1 } else { 0 };
                let rev = self
                    .revisions
                    .entry((product.output.clone(), index))
                    .or_default();
                *rev += 1;
                let content = product.content.unwrap_or_default();
                let parts: Parts = product
                    .parts
                    .iter()
                    .filter_map(|p| p.content.as_ref().map(|c| (c.digest.clone(), c.length)))
                    .collect();
                let previous = self
                    .parts
                    .insert((product.output.clone(), index), parts.clone());
                let appended_from = previous
                    .filter(|before| {
                        !before.is_empty()
                            && before.len() <= parts.len()
                            && parts.starts_with(before)
                    })
                    .map(|before| before.iter().map(|(_, length)| length).sum());
                let converted = v1::Product {
                    output: product.output,
                    index,
                    rev: *rev,
                    length: content.length,
                    digest: spell(&content.digest),
                    media_type: product.media_type,
                    label: product.label,
                    duration_us: product.parts.iter().map(|p| p.duration_us).sum(),
                    appended_from,
                };
                self.products.retain(|p| {
                    (p.output.as_str(), p.index) != (converted.output.as_str(), converted.index)
                });
                self.products.push(converted.clone());
                v1::run_event::Event::Product(converted)
            }
            "outcome" => {
                let outcome = event.outcome?;
                let body: Value =
                    serde_json::from_slice(&outcome.outcome_canonical_bytes).unwrap_or(Value::Null);
                let status = match body["status"].as_u64() {
                    Some(1) => "succeeded",
                    Some(4) => "canceled",
                    _ => "failed",
                };
                let result = body["result"]["inline_result"]
                    .as_str()
                    .and_then(|encoded| STANDARD.decode(encoded).ok())
                    .unwrap_or_default();
                v1::run_event::Event::Outcome(v1::Outcome {
                    status: status.into(),
                    reason: (status != "succeeded").then(|| reason(&body)),
                    result,
                    outputs: self.products.clone(),
                    triage: body.get("triage_bundle").is_some(),
                    measurements: vec![], // `stream_run` adds them
                })
            }
            _ => return None,
        };
        Some(v1::RunEvent {
            sequence: event.sequence,
            at_ms: event.at_ms as i64,
            event: Some(kind),
        })
    }
}

#[tonic::async_trait]
impl<B: MachineBackend> v1::machine_server::Machine for MachineV1<B> {
    async fn status(
        &self,
        request: Request<v1::StatusRequest>,
    ) -> Result<Response<Events<v1::StatusFrame>>, Status> {
        let caller = match request.metadata().get("authorization") {
            None => None,
            Some(_) => Some(self.caller(request.metadata())?),
        };
        let machine = caller.filter(|c| c.machine().is_ok()).map(|c| c.actor);
        let keepalive = request.into_inner().keepalive;
        super::machine_status::status(
            self.identity.clone(),
            self.backend.clone(),
            machine,
            keepalive,
        )
        .await
        .map(Response::new)
    }

    async fn run(
        &self,
        request: Request<v1::RunRequest>,
    ) -> Result<Response<Events<v1::RunEvent>>, Status> {
        let caller = self.caller(request.metadata())?;
        let request = request.into_inner();
        if request.id.is_empty() || request.id.len() > 256 {
            return Err(Status::invalid_argument("a run id is 1-256 bytes"));
        }
        if super::machine_update::owns(&self.identity, &request) {
            caller.machine()?;
            let (identity, backend) = (self.identity.clone(), self.backend.clone());
            return super::machine_update::run(identity, backend, caller.actor, request)
                .await
                .map(|events| Response::new(self.authority(&caller).wrap(events)));
        }
        match &request.spec {
            Some(_) => caller.machine()?,
            None => caller.run(&request.id, None)?,
        }
        let (sender, receiver) = tokio::sync::mpsc::channel(16);
        let (backend, actor, limit) = (self.backend.clone(), caller.actor, caller.limit());
        tokio::spawn(async move {
            let result = stream_run(backend, actor, request, limit, sender.clone()).await;
            if let Err(status) = result {
                let _ = sender.send(Err(status)).await;
            }
        });
        let events = Box::pin(tokio_stream::wrappers::ReceiverStream::new(receiver));
        Ok(Response::new(self.authority(&caller).wrap(events)))
    }

    /// One object's bytes from the frame's offset; a header-only stream answers what is held.
    async fn write(
        &self,
        request: Request<tonic::Streaming<v1::WriteFrame>>,
    ) -> Result<Response<v1::WriteResult>, Status> {
        let caller = self.caller(request.metadata())?;
        caller.machine()?;
        let runs = self
            .backend
            .runs()
            .ok_or_else(|| Status::unimplemented("this machine takes no writes"))?;
        let actor = crate::machine_api::actor_id(caller.actor);
        let mut frames = self.authority(&caller).wrap(Box::pin(request.into_inner()));
        let first = frames
            .next()
            .await
            .transpose()?
            .ok_or_else(|| Status::invalid_argument("a write names its object first"))?;
        let (digest, length, offset) = (first.digest.clone(), first.length, first.offset);
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<Vec<u8>>(4);
        let object = digest.clone();
        let writer = tokio::task::spawn_blocking(move || -> Result<u64, Status> {
            let mut writer = runs
                .objects
                .begin(&actor, &object, length, offset)
                .map_err(refusal)?;
            while let Some(data) = receiver.blocking_recv() {
                writer.append(&data).map_err(refusal)?;
            }
            writer.finish().map_err(refusal)
        });
        let mut data = first.data;
        loop {
            // A refused writer drops its receiver; its refusal is the answer.
            if !data.is_empty() && sender.send(data).await.is_err() {
                break;
            }
            match frames.next().await.transpose()? {
                Some(frame) => data = frame.data,
                None => break,
            }
        }
        drop(sender);
        let held = writer
            .await
            .map_err(|_| Status::internal("machine operation stopped"))??;
        Ok(Response::new(v1::WriteResult { digest, held }))
    }

    async fn control(
        &self,
        request: Request<v1::ControlRequest>,
    ) -> Result<Response<v1::RunState>, Status> {
        let caller = self.caller(request.metadata())?;
        caller.machine()?;
        let request = request.into_inner();
        let action = match v1::Action::try_from(request.action) {
            Ok(v1::Action::Cancel) => pb::MachineExecutionAction::Cancel,
            Ok(v1::Action::Pause) => pb::MachineExecutionAction::Pause,
            Ok(v1::Action::Resume) => pb::MachineExecutionAction::Resume,
            _ => return Err(Status::invalid_argument("control names an action")),
        };
        let actor = caller.actor;
        let id = request.id.clone();
        let changed = Self::call(&self.backend, move |backend| {
            backend.control(
                actor,
                pb::MachineExecutionControl {
                    execution: Some(query(backend, actor, &id)?),
                    command_id: uuid::Uuid::new_v4().to_string(),
                    action: action as i32,
                    ..Default::default()
                },
            )
        })
        .await?;
        Ok(Response::new(state(&request.id, &changed)))
    }

    async fn read(
        &self,
        request: Request<v1::ReadRequest>,
    ) -> Result<Response<Events<v1::ReadFrame>>, Status> {
        let caller = self.caller(request.metadata())?;
        let request = request.into_inner();
        let actor = caller.actor;
        let (offset, if_rev) = (request.offset, request.if_rev);
        let (meta, bytes): (v1::ReadFrame, Box<dyn Read + Send>) = match request.target {
            Some(v1::read_request::Target::Output(target)) => {
                let index = (target.index > 0).then_some(target.index);
                caller.run(&target.run, Some((&target.output, index)))?;
                let snapshot = Self::call(&self.backend, move |backend| {
                    let run = backend.get(actor, query(backend, actor, &target.run)?)?;
                    backend.open_output(actor, run.number, &target.output, index)
                })
                .await?;
                if if_rev != 0 && if_rev != snapshot.rev {
                    return Err(Status::failed_precondition(format!(
                        "the output is at revision {}",
                        snapshot.rev
                    )));
                }
                let meta = v1::ReadFrame {
                    rev: snapshot.rev,
                    length: snapshot.length,
                    digest: snapshot.sha256.clone().unwrap_or_default(),
                    media_type: snapshot.media_type.clone(),
                    data: vec![],
                };
                let reader = snapshot.parts.into_iter().fold(
                    Box::new(std::io::empty()) as Box<dyn Read + Send>,
                    |chain, (file, length)| Box::new(chain.chain(file.take(length))),
                );
                (meta, reader)
            }
            Some(v1::read_request::Target::Triage(run)) => {
                caller.run(&run, None)?;
                if caller.limit().is_some() {
                    return Err(Status::permission_denied("triage is the run owner's"));
                }
                let triage = Self::call(&self.backend, move |backend| {
                    backend.read_triage(
                        actor,
                        pb::MachineExecutionTriageQuery {
                            execution: Some(query(backend, actor, &run)?),
                            attempt_ordinal: 0,
                        },
                    )
                })
                .await?;
                let bytes = triage.bundle_canonical_bytes;
                let meta = v1::ReadFrame {
                    rev: 1,
                    length: bytes.len() as u64,
                    digest: spell(&tensorfs_core::sha256::digest(&bytes)),
                    media_type: "application/json".into(),
                    data: vec![],
                };
                (meta, Box::new(std::io::Cursor::new(bytes)))
            }
            Some(v1::read_request::Target::Log(name)) => {
                caller.machine()?;
                let log = match name.as_str() {
                    "tensorfs-transport" => pb::MachineLog::TensorfsTransport,
                    _ => {
                        return Err(Status::not_found(format!(
                            "this machine keeps no log {name}"
                        )))
                    }
                };
                let tail = request.tail;
                let bytes = Self::call(&self.backend, move |backend| {
                    backend.read_machine_log(
                        actor,
                        pb::MachineLogQuery {
                            log: log as i32,
                            tail_bytes: tail,
                            ..Default::default()
                        },
                    )
                })
                .await?;
                let meta = v1::ReadFrame {
                    rev: 0,
                    length: bytes.len() as u64,
                    digest: String::new(),
                    media_type: "text/plain".into(),
                    data: vec![],
                };
                (meta, Box::new(std::io::Cursor::new(bytes)))
            }
            None => return Err(Status::invalid_argument("read names a target")),
        };
        if offset > meta.length {
            return Err(Status::out_of_range("offset is past the end"));
        }
        let (sender, receiver) = tokio::sync::mpsc::channel(4);
        tokio::task::spawn_blocking(move || {
            let (mut bytes, mut left) = (bytes, meta.length - offset);
            let ended_early = || Err(Status::data_loss("output bytes ended early"));
            if sender.blocking_send(Ok(meta)).is_err() {
                return;
            }
            if std::io::copy(&mut (&mut bytes).take(offset), &mut std::io::sink()).ok() != Some(offset) {
                let _ = sender.blocking_send(ended_early());
                return;
            }
            let mut buffer = vec![0; 1 << 20];
            loop {
                match bytes.read(&mut buffer) {
                    Ok(0) => {
                        if left > 0 {
                            let _ = sender.blocking_send(ended_early());
                        }
                        return;
                    }
                    Ok(n) => {
                        left = left.saturating_sub(n as u64);
                        let frame = v1::ReadFrame {
                            data: buffer[..n].to_vec(),
                            ..Default::default()
                        };
                        if sender.blocking_send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = sender.blocking_send(Err(Status::data_loss(error.to_string())));
                        return;
                    }
                }
            }
        });
        let frames = Box::pin(tokio_stream::wrappers::ReceiverStream::new(receiver));
        Ok(Response::new(self.authority(&caller).wrap(frames)))
    }
}

/// Accepts a run, or answers the one a re-sent spec already names whatever its access now
/// says: expired Hub access refuses only a run not yet accepted.
fn accept(runs: &Arc<crate::runs::Runs>, actor: &str, id: &str, spec: crate::runs::Spec, expired: bool) -> Result<(), Status> {
    if expired && runs.existing(actor, id, &spec.digest).map_err(refusal)?.is_none() {
        return Err(refused("hub_access_expired", "the run's Hub access has expired"));
    }
    runs.submit(actor, id, spec).map(drop).map_err(refusal)
}

/// Submit when asked, then follow the log until the outcome or the caller leaves. An
/// output-limited grant (`limit`) sees only what [`visible`] keeps.
pub(super) async fn stream_run<B: MachineBackend>(
    backend: Arc<B>,
    actor: VerifiedActor,
    request: v1::RunRequest,
    limit: Option<Grant>,
    sender: tokio::sync::mpsc::Sender<Result<v1::RunEvent, Status>>,
) -> Result<(), Status> {
    let id = request.id.clone();
    if let Some(spec) = request.spec {
        let runs = backend
            .runs()
            .ok_or_else(|| Status::unimplemented("this machine takes no runs"))?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs() as i64);
        let expired = spec.hub.as_ref().is_some_and(|hub| {
            !hub.token.is_empty() && hub.expires_at != 0 && hub.expires_at <= now
        });
        let (id, spec) = (id.clone(), spec_of(spec)?);
        tokio::task::spawn_blocking(move || {
            accept(&runs, &crate::machine_api::actor_id(actor), &id, spec, expired)
        })
        .await
        .map_err(|_| Status::internal("machine operation stopped"))??;
    }
    // The log is read from its start, so a reattach numbers each output's revisions and lists
    // the outcome's outputs as the run made them; only entries past the cursor are sent.
    let (mut log, mut after, cursor) = (Log::default(), 0, request.after);
    let (first_backend, first_id) = (backend.clone(), id.clone());
    let current = tokio::task::spawn_blocking(move || {
        first_backend.get(actor, query(&*first_backend, actor, &first_id)?)
    })
    .await
    .map_err(|_| Status::internal("machine operation stopped"))??;
    let snapshot = v1::RunEvent {
        sequence: 0,
        at_ms: 0,
        event: Some(v1::run_event::Event::State(state(&id, &current))),
    };
    if sender.send(Ok(snapshot)).await.is_err() {
        return Ok(());
    }
    loop {
        let (page_backend, page_id) = (backend.clone(), id.clone());
        let page = tokio::task::spawn_blocking(move || {
            page_backend.events(
                actor,
                pb::MachineExecutionEventsQuery {
                    execution: Some(query(&*page_backend, actor, &page_id)?),
                    after,
                    limit: 256,
                    wait: true,
                },
            )
        })
        .await
        .map_err(|_| Status::internal("machine operation stopped"))??;
        let mut ended = false;
        for event in page.events {
            after = after.max(event.sequence);
            let Some(mut converted) = log.event(event) else { continue };
            ended |= matches!(converted.event, Some(v1::run_event::Event::Outcome(_)));
            let hidden = limit.as_ref().is_some_and(|grant| !visible(grant, &mut converted));
            if converted.sequence <= cursor || hidden {
                continue;
            }
            if let (Some(v1::run_event::Event::Outcome(outcome)), None) = (&mut converted.event, &limit) {
                let (kept_backend, kept_id) = (backend.clone(), id.clone());
                outcome.measurements = tokio::task::spawn_blocking(move || {
                    kept_backend.measurements(actor, query(&*kept_backend, actor, &kept_id)?)
                })
                .await
                .map_err(|_| Status::internal("machine operation stopped"))??
                .unwrap_or_default();
            }
            if sender.send(Ok(converted)).await.is_err() {
                return Ok(());
            }
        }
        if ended {
            return Ok(());
        }
        if sender.is_closed() {
            return Ok(());
        }
        after = after.max(page.next_after);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authored() -> v1::RunSpec {
        let input = |field: &str, hex: &str, order| v1::InputFile {
            field: field.into(), digest: format!("sha256:{}", hex.repeat(64)), length: 4, order, ..Default::default()
        };
        let model = |parameter: &str, repository: &str| v1::ModelChoice {
            parameter: parameter.into(), repository: repository.into(), release: "1".into(), ..Default::default()
        };
        v1::RunSpec {
            source: Some(v1::run_spec::Source::Release(v1::Release { package: "org/package".into(), release: "1.0.0".into() })),
            entrypoint: "render".into(),
            payload: br#" { "seed":9007199254740993, "nested":{"b":2,"a":1} } "#.to_vec(),
            inputs: vec![input("reference", "a", 0), input("reference", "b", 1)],
            models: vec![model("model", "org/first"), model("other", "org/second")],
            binding_revision: "7".into(),
            known_results: vec![v1::MemoResult::default()],
            publication: "p1".into(),
            hub: Some(v1::HubAccess {
                origin: "https://hub.example.test/".into(), token: "t1".into(), expires_at: i64::MAX,
                ca_der: vec![1], object_hosts: vec!["objects-1.example.test".into()],
            }),
            providers: Some(v1::ProviderAccess { huggingface: "h1".into(), civitai: "c1".into() }),
            ..Default::default()
        }
    }

    /// A spec re-sent after a daemon crash, with refreshed (even expired) access, new hints and
    /// a reformatted payload, names the run it named; what the run is still tells runs apart.
    #[test]
    fn a_resent_spec_names_the_run_it_named() {
        let original = authored();
        let digest = spec_of(original.clone()).unwrap().digest;
        let mut resent = original.clone();
        resent.payload = br#"{"nested":{"a":1,"b":2},"seed":9007199254740993}"#.to_vec();
        resent.hub = Some(v1::HubAccess {
            origin: "https://hub.example.test".into(), token: "t2".into(), expires_at: 1,
            ca_der: vec![2], object_hosts: vec!["objects-2.example.test".into()],
        });
        (resent.binding_revision, resent.publication, resent.providers) = ("8".into(), "p2".into(), None);
        resent.known_results.clear();
        assert_eq!(spec_of(resent).unwrap().digest, digest);
        let changes: [fn(&mut v1::RunSpec); 6] = [
            |s| s.payload = br#"{"seed":9007199254740992,"nested":{"a":1,"b":2}}"#.to_vec(),
            |s| s.payload = br#"{"seed":9007199254740993.0,"nested":{"a":1,"b":2}}"#.to_vec(),
            |s| s.inputs.reverse(),
            |s| s.models.reverse(),
            |s| s.entrypoint = "other".into(),
            |s| s.hub.as_mut().unwrap().origin = "https://other.example.test".into(),
        ];
        for change in changes {
            let mut changed = original.clone();
            change(&mut changed);
            assert_ne!(spec_of(changed).unwrap().digest, digest);
        }
    }

    /// Expired access refuses a new run, never the accepted run a re-sent spec names.
    #[test]
    fn expired_access_refuses_only_a_new_run() {
        let root = std::env::temp_dir().join(format!("cm-accept-{}", uuid::Uuid::new_v4()));
        let service = crate::service::Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(crate::objects::Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap());
        let runs = Arc::new(crate::runs::Runs {
            service: service.clone(), objects, publisher: None, local: None, own_hub: None, jobs: Default::default(),
        });
        let spec = spec_of(authored()).unwrap();
        let invocation = crate::journal::Invocation { package: "org/package".into(), input: spec.input.clone(), ..Default::default() };
        let accepted = service.engine.accept_run("alice", "run-1", &spec.digest, invocation).unwrap().0;
        accept(&runs, "alice", "run-1", spec_of(authored()).unwrap(), true).unwrap();
        assert_eq!(service.engine.get_public("alice", "run-1").unwrap().id, accepted.id);
        let refusal = accept(&runs, "alice", "run-2", spec, true).unwrap_err();
        assert_eq!(refusal.metadata().get("cozy-error-code").unwrap(), "hub_access_expired");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A digest of 64 non-hex bytes is refused, not sliced mid-character.
    #[test]
    fn a_digest_is_ascii_hex() {
        for invalid in ["é".repeat(32), format!("a{}a", "é".repeat(31)), "z".repeat(64)] {
            let refusal = super::digest(&format!("sha256:{invalid}")).unwrap_err();
            assert_eq!(refusal.code(), tonic::Code::InvalidArgument);
        }
        assert_eq!(super::digest(&format!("sha256:{}", "A".repeat(64))).unwrap(), vec![0xaa; 32]);
    }
}
