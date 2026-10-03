//! `cozy.machine.v1` (G/API.md): Run, Control and Read over the machine's engine, every call
//! authorized by one `Cozy-Cap`. Until `worker.v1` is deleted at cutover this service reuses
//! the backend's operations; Status and `Run kind: update` are `machine_status` and
//! `machine_update` (D2); Write (D1) answers UNIMPLEMENTED until it lands.
use super::{
    auth::VerifiedActor,
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
use tokio_stream::Stream;
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

impl<B: MachineBackend> MachineV1<B> {
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

fn query(
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

fn submission(
    id: &str,
    spec: v1::RunSpec,
    workspace: String,
) -> Result<pb::MachineExecutionSubmit, Status> {
    if spec.kind != v1::RunKind::Call as i32 {
        return Err(Status::unimplemented(
            "this machine runs calls; jobs, warm-up and updates come next",
        ));
    }
    if !spec.inputs.is_empty() {
        return Err(Status::unimplemented("file inputs arrive with Write"));
    }
    if spec.hub.as_ref().is_some_and(|hub| !hub.token.is_empty()) {
        return Err(Status::unimplemented(
            "a run-held Hub token arrives with preparation in runs",
        ));
    }
    let mut root = pb::ReleaseRoot {
        entrypoint: spec.entrypoint,
        attention_kernel: spec.attention_kernel,
        owner: spec.owner,
        hub: spec.hub.map(|hub| hub.origin).unwrap_or_default(),
        models: spec
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
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                })
            })
            .collect::<Result<_, _>>()?,
        ..Default::default()
    };
    match spec.source {
        Some(v1::run_spec::Source::Release(release)) => {
            root.package = release.package;
            root.release = release.release;
        }
        Some(v1::run_spec::Source::Installation(installation)) => {
            root.installation_id = installation;
        }
        Some(_) => {
            return Err(Status::unimplemented(
                "local and private sources arrive with Write",
            ))
        }
        None => return Err(Status::invalid_argument("a run spec names its source")),
    }
    Ok(pb::MachineExecutionSubmit {
        offer: Some(pb::AttemptOffer {
            request_id: id.into(),
            ..Default::default()
        }),
        submission_id: id.into(),
        expected_execution_workspace_id: workspace,
        payload_canonical_bytes: spec.payload,
        release_root: Some(root),
        ..Default::default()
    })
}

fn digest(text: &str) -> Result<Vec<u8>, Status> {
    let hex = text
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64)
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

/// Translates the engine's event pages into the run's log, numbering each output's revisions.
#[derive(Default)]
struct Log {
    revisions: HashMap<(String, u32), u64>,
    products: Vec<v1::Product>,
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
                    stage: payload["stage"].as_str().unwrap_or_default().into(),
                    fraction: payload["overall_fraction"].as_f64().unwrap_or(-1.0),
                    completed: payload["position"].as_u64().unwrap_or(0),
                    total: payload["total"].as_u64().unwrap_or(0),
                    ..Default::default()
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
                let converted = v1::Product {
                    output: product.output,
                    index,
                    rev: *rev,
                    length: content.length,
                    digest: spell(&content.digest),
                    media_type: product.media_type,
                    label: product.label,
                    duration_us: product.parts.iter().map(|p| p.duration_us).sum(),
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
                .map(Response::new);
        }
        match &request.spec {
            Some(_) => caller.machine()?,
            None => caller.run(&request.id, None)?,
        }
        let (sender, receiver) = tokio::sync::mpsc::channel(16);
        let backend = self.backend.clone();
        let actor = caller.actor;
        tokio::spawn(async move {
            let result = stream_run(backend, actor, request, sender.clone()).await;
            if let Err(status) = result {
                let _ = sender.send(Err(status)).await;
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
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
                    backend.open_output(run.number, &target.output, index)
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
            let mut bytes = bytes;
            if sender.blocking_send(Ok(meta)).is_err() {
                return;
            }
            if std::io::copy(&mut (&mut bytes).take(offset), &mut std::io::sink()).is_err() {
                let _ = sender.blocking_send(Err(Status::data_loss("output bytes ended early")));
                return;
            }
            let mut buffer = vec![0; 1 << 20];
            loop {
                match bytes.read(&mut buffer) {
                    Ok(0) => return,
                    Ok(n) => {
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
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
    }
}

/// Submit when asked, then follow the log until the outcome or the caller leaves.
async fn stream_run<B: MachineBackend>(
    backend: Arc<B>,
    actor: VerifiedActor,
    request: v1::RunRequest,
    sender: tokio::sync::mpsc::Sender<Result<v1::RunEvent, Status>>,
) -> Result<(), Status> {
    let id = request.id.clone();
    if let Some(spec) = request.spec {
        loop {
            let (submit_backend, id, spec) = (backend.clone(), id.clone(), spec.clone());
            let submitted = tokio::task::spawn_blocking(move || {
                let workspace = submit_backend
                    .workspace(actor, pb::MachineExecutionWorkspaceQuery::default())?
                    .execution_workspace_id;
                submit_backend.submit(actor, submission(&id, spec, workspace)?)
            })
            .await
            .map_err(|_| Status::internal("machine operation stopped"))?;
            match submitted {
                Ok(_) => break,
                // Preparation before acceptance: its stage is the progress (D1 moves it into runs).
                Err(status)
                    if status
                        .metadata()
                        .get("cozy-error-code")
                        .and_then(|v| v.to_str().ok())
                        == Some("release_root_preparing") =>
                {
                    let (done, total) = status
                        .metadata()
                        .get("cozy-progress-bytes")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.split_once(' '))
                        .map(|(d, t)| (d.parse().unwrap_or(0), t.parse().unwrap_or(0)))
                        .unwrap_or((0, 0));
                    let event = v1::RunEvent {
                        sequence: 0,
                        at_ms: 0,
                        event: Some(v1::run_event::Event::Progress(v1::Progress {
                            stage: status.message().into(),
                            fraction: -1.0,
                            bytes_done: done,
                            bytes_total: total,
                            ..Default::default()
                        })),
                    };
                    if sender.send(Ok(event)).await.is_err() {
                        return Ok(());
                    }
                }
                Err(status) => return Err(status),
            }
        }
    }
    let mut log = Log::default();
    let mut after = request.after;
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
            if let Some(converted) = log.event(event) {
                ended |= matches!(converted.event, Some(v1::run_event::Event::Outcome(_)));
                if sender.send(Ok(converted)).await.is_err() {
                    return Ok(());
                }
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
