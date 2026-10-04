//! `Run kind: update` (G/API.md, D2): the payload names the software cohort, `{"runtime": …,
//! "tensorfs": …, "agent": "bundled" | "explicit"}`. A member is a published version (fetched
//! from PyPI, digest checked) or, when an input binds that field to an object sent with Write,
//! that wheel's file name. The update stages and verifies the pair, waits for measured idleness,
//! restarts the service on it in place and commits once the new service proves readiness, or
//! rolls back. Its log is the update's state history, so a client that lost the stream across
//! the restart attaches again with `Run{id, after}` and reads the outcome.
use super::{auth::VerifiedActor, backend::MachineBackend, v1, MachineIdentity};
use crate::machine::update::{Choice, Request, Status as Update, Updates};
use std::{collections::BTreeMap, pin::Pin, sync::Arc, time::Duration};
use tokio_stream::Stream;
use tonic::Status;

pub(super) type Events = Pin<Box<dyn Stream<Item = Result<v1::RunEvent, Status>> + Send>>;

/// The update `request` submits or attaches to, if it is one.
pub(super) fn owns(identity: &MachineIdentity, request: &v1::RunRequest) -> bool {
    match &request.spec {
        Some(spec) => spec.kind == v1::RunKind::Update as i32,
        None => identity
            .updates
            .as_ref()
            .is_some_and(|u| u.update(&request.id).is_some()),
    }
}

#[derive(serde::Deserialize)]
struct Cohort {
    runtime: Option<String>,
    tensorfs: Option<String>,
    #[serde(default)]
    agent: String,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

pub(super) async fn run<B: MachineBackend>(
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
    actor: VerifiedActor,
    request: v1::RunRequest,
) -> Result<Events, Status> {
    let updates = identity
        .updates
        .clone()
        .ok_or_else(|| Status::failed_precondition("this machine does not update in place"))?;
    let id = request.id.clone();
    if let Some(spec) = request.spec.filter(|_| updates.update(&id).is_none()) {
        if !identity.readiness.proved() {
            return Err(Status::unavailable(
                "the machine is starting; update it once it is ready",
            ));
        }
        let updates = updates.clone();
        let id = id.clone();
        tokio::task::spawn_blocking(move || submit(&updates, &*backend, actor, &id, spec))
            .await
            .map_err(|_| Status::internal("machine operation stopped"))??;
    }
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        let mut sent = request.after;
        let mut first = true;
        loop {
            let Some(update) = updates.update(&id) else {
                let _ = sender
                    .send(Err(Status::not_found("this machine has no such run")))
                    .await;
                return;
            };
            let mut events = vec![];
            if first {
                events.push(event(
                    0,
                    0,
                    v1::run_event::Event::State(state(&id, &update)),
                ));
                first = false;
            }
            for (index, step) in update.history.iter().enumerate().skip(sent as usize) {
                events.push(event(
                    index as u64 + 1,
                    step.at_ms,
                    step_event(&update, step),
                ));
            }
            sent = sent.max(update.history.len() as u64);
            for event in events {
                if sender.send(Ok(event)).await.is_err() {
                    return;
                }
            }
            if update.terminal() {
                return;
            }
            tokio::select! {
                _ = sender.closed() => return,
                _ = tokio::time::sleep(Duration::from_millis(250)) => {}
            }
        }
    });
    Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(
        receiver,
    )))
}

fn submit(
    updates: &Arc<Updates>,
    backend: &impl MachineBackend,
    actor: VerifiedActor,
    id: &str,
    spec: v1::RunSpec,
) -> Result<(), Status> {
    let cohort: Cohort = serde_json::from_slice(&spec.payload).map_err(|e| {
        Status::invalid_argument(format!("an update payload is a JSON object: {e}"))
    })?;
    for name in cohort.unknown.keys() {
        eprintln!("cozy-machine: update {id}: ignoring unknown payload field {name:?}");
    }
    let choose = |member: &str, value: Option<String>| -> Result<Option<Choice>, Status> {
        let Some(value) = value else {
            return Ok(None);
        };
        let Some(input) = spec.inputs.iter().find(|i| i.field == member) else {
            return Ok(Some(Choice {
                version: value,
                ..Default::default()
            }));
        };
        let sha256 = input
            .digest
            .strip_prefix("sha256:")
            .filter(|h| h.len() == 64)
            .ok_or_else(|| Status::invalid_argument("an input digest is sha256:<64 hex>"))?;
        let mut object = backend.object(actor, sha256).map_err(|e| {
            Status::failed_precondition(format!(
                "the {member} wheel {} is not held here; Write it first: {}",
                input.digest,
                e.message()
            ))
        })?;
        let (staged, _) = updates
            .stage(&value, &mut object)
            .map_err(|(code, message)| refusal(code, message))?;
        if staged != sha256 {
            return Err(Status::data_loss(
                "the staged wheel differs from its object",
            ));
        }
        Ok(Some(Choice {
            file: value,
            sha256: staged,
            ..Default::default()
        }))
    };
    let request = Request {
        operation: id.into(),
        agent: cohort.agent,
        pin: None,
        runtime: choose("runtime", cohort.runtime)?,
        tensorfs: choose("tensorfs", cohort.tensorfs)?,
    };
    updates
        .request(request, |code| std::process::exit(code))
        .map(|_| ())
        .map_err(|(code, message)| refusal(code, message))
}

fn refusal(code: u16, message: String) -> Status {
    match code {
        400 => Status::invalid_argument(message),
        409 => Status::failed_precondition(message),
        503 => Status::unavailable(message),
        _ => Status::internal(message),
    }
}

fn event(sequence: u64, at_ms: i64, event: v1::run_event::Event) -> v1::RunEvent {
    v1::RunEvent {
        sequence,
        at_ms,
        event: Some(event),
    }
}

fn state(id: &str, update: &Update) -> v1::RunState {
    v1::RunState {
        id: id.into(),
        number: 0,
        state: match update.state.as_str() {
            "waiting" => "queued",
            "succeeded" => "succeeded",
            "rolled_back" | "failed" => "failed",
            _ => "running",
        }
        .into(),
        sequence: update.history.len() as u64,
        attempt: 1,
        waiting: if update.state == "waiting_activation" {
            "waiting for the machine to be idle".into()
        } else {
            String::new()
        },
    }
}

fn step_event(update: &Update, step: &crate::machine::update::Step) -> v1::run_event::Event {
    let outcome = |status: &str, code: &str| {
        let pair = |p: &crate::machine::update::Pair| serde_json::json!({"runtime": p.runtime, "tensorfs": p.tensorfs});
        let result = serde_json::json!({"from": pair(&update.from), "to": pair(&update.to)});
        v1::run_event::Event::Outcome(v1::Outcome {
            status: status.into(),
            reason: (!code.is_empty()).then(|| v1::Reason {
                code: code.into(),
                message: update.error.clone(),
                origin: "machine".into(),
            }),
            result: serde_json::to_vec(&result).unwrap_or_default(),
            ..Default::default()
        })
    };
    match step.state.as_str() {
        "succeeded" => outcome("succeeded", ""),
        "rolled_back" => outcome("failed", "update_rolled_back"),
        "failed" => outcome("failed", "update_failed"),
        stage => v1::run_event::Event::Progress(v1::Progress {
            stage: stage.into(),
            fraction: -1.0,
            ..Default::default()
        }),
    }
}
