//! A real queued observer whose first event page waits for actual owned work to end.
//! Controlled CPU producer/TLS/journal proof, not model inference qualification.
use crate::{
    api::{
        self,
        auth::VerifiedActor,
        capability::{self, Grant},
        pb, v1, MachineBackend, MachineIdentity,
    },
    execution::process_birth,
    journal::Invocation,
    machine_api::NativeBackend,
    service::Service,
};
use ed25519_dalek::SigningKey;
use serde_json::{json, Value};
use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Barrier},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tonic::{
    transport::{Certificate, ClientTlsConfig, Endpoint},
    Status,
};

struct HeldPage {
    native: Arc<NativeBackend>,
    first_snapshot: Arc<Barrier>,
    id: String,
}
impl MachineBackend for HeldPage {
    fn workspace(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status> {
        self.native.workspace(actor, query)
    }
    fn get(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionQuery,
    ) -> Result<pb::MachineExecutionState, Status> {
        let state = self.native.get(actor, query)?;
        assert_eq!(state.state, "queued"); // Actual Starting maps to queued on the public API.
        assert_eq!(
            self.native.service.engine.get(&self.id).unwrap().state,
            crate::journal::State::Starting
        );
        self.first_snapshot.wait(); // Work starts only after this real queued snapshot was read.
        Ok(state)
    }
    fn events(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionEventsQuery,
    ) -> Result<pb::MachineExecutionEventPage, Status> {
        let until = Instant::now() + Duration::from_secs(15); // Harness bound, never work authority.
        loop {
            let epoch = self.native.service.engine.activity_epoch();
            if self
                .native
                .service
                .engine
                .get(&self.id)
                .map_err(|error| Status::internal(error.to_string()))?
                .state
                .terminal()
            {
                break;
            }
            assert!(
                Instant::now() < until,
                "controlled CPU producer did not finish"
            );
            self.native
                .service
                .engine
                .wait_activity(epoch, Some(Duration::from_millis(100)));
        }
        self.native.events(actor, query)
    }
    fn measurements(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionQuery,
    ) -> Result<Option<Vec<u8>>, Status> {
        self.native.measurements(actor, query)
    }
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn event_state(event: &v1::RunEvent) -> Option<&v1::RunState> {
    match &event.event {
        Some(v1::run_event::Event::State(state)) => Some(state),
        _ => None,
    }
}
fn event_progress(event: &v1::RunEvent) -> Option<&v1::Progress> {
    match &event.event {
        Some(v1::run_event::Event::Progress(progress)) => Some(progress),
        _ => None,
    }
}

#[tokio::test]
async fn skipped_running_replays_the_actual_start_before_terminal_progress() {
    let scratch =
        Scratch(std::env::temp_dir().join(format!("cm-running-history-{}", uuid::Uuid::new_v4())));
    let root = &scratch.0;
    let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
    assert!(service.stop().unwrap()); // The test owns this one execution's adapter.
    let signer = SigningKey::from_bytes(&[93; 32]);
    let actor = tensorfs_core::sha256::hex(signer.verifying_key().as_bytes());
    let (record, fresh) = service
        .engine
        .accept_run(
            &actor,
            "run",
            &format!("sha256:{}", "a".repeat(64)),
            Invocation::default(),
        )
        .unwrap();
    assert!(fresh);
    service
        .engine
        .bind_prepared(&record.id, Invocation::default(), "")
        .unwrap();
    let first_snapshot = Arc::new(Barrier::new(2));
    let gate = first_snapshot.clone();
    let (registered, wait_registered) = std::sync::mpsc::channel();
    assert!(service
        .engine
        .dispatch_managed(&record.id, move |engine, id| {
            let mut child = Command::new("/usr/bin/python3")
                .args([
                    "-u",
                    "-c",
                    r#"
import json, sys
if sys.stdin.buffer.read(1):
    for index in range(30):
        print(json.dumps({"stage":"denoise","position":index+1,"total":30}), flush=True)
    print(json.dumps({"stage":"decoding"}), flush=True)
"#,
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()?;
            let birth = process_birth(child.id())?;
            let cancel_birth = birth.clone();
            engine.register_managed(
                &id,
                birth,
                Arc::new(move || {
                    if let Some(exact) = crate::process::Exact::open(&cancel_birth)? {
                        exact.kill()?;
                    }
                    Ok(())
                }),
            )?;
            registered
                .send(())
                .map_err(|_| io::Error::other("test observer ended"))?;
            gate.wait();
            assert!(engine.authorize_managed(&id, None)?);
            child.stdin.take().unwrap().write_all(b"g")?;
            let mut count = 0;
            for line in io::BufReader::new(child.stdout.take().unwrap()).lines() {
                let detail = line?;
                let value: Value = serde_json::from_str(&detail)?;
                if value["stage"] == "denoise" {
                    count += 1;
                }
                engine.observe_progress(&id, 0, detail)?;
            }
            assert!(child.wait()?.success());
            assert_eq!(count, 30); // Actual frames from the owned CPU process, not inferred completion.
            let spool = engine.staging(&id)?;
            engine.managed_result(&id, &spool, json!({"steps":count}), vec![])?;
            Ok(())
        })
        .unwrap());
    wait_registered
        .recv_timeout(Duration::from_secs(15))
        .unwrap();

    let identity =
        MachineIdentity::ephemeral("history".into(), vec![signer.verifying_key()], vec![7; 32])
            .unwrap();
    let pem = identity.cert_pem.clone();
    let store = Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
    let uploads =
        api::workspaces::WorkspaceUploads::open(&root.join("uploads"), store.clone()).unwrap();
    let native = Arc::new(NativeBackend::new(
        service.clone(),
        identity.authority.clone(),
        store,
        uploads,
    ));
    let held = Arc::new(HeldPage {
        native: native.clone(),
        first_snapshot,
        id: record.id.clone(),
    });
    let token = capability::mint(
        &signer,
        Grant {
            machine: "history".into(),
            action: capability::MACHINE.into(),
            expires: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
                + 600,
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { api::serve(listener, identity, held).await.unwrap() });
    let channel = Endpoint::from_shared(format!("https://{address}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(pem))
                .domain_name("localhost"),
        )
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = v1::machine_client::MachineClient::new(channel);
    let mut request = tonic::Request::new(v1::RunRequest {
        id: "run".into(),
        after: 0,
        spec: None,
    });
    request.metadata_mut().insert(
        "authorization",
        format!("Cozy-Cap {token}").parse().unwrap(),
    );
    let mut stream = client.run(request).await.unwrap().into_inner();
    let snapshot = stream.message().await.unwrap().unwrap();
    assert_eq!(snapshot.sequence, 0);
    assert_eq!(event_state(&snapshot).unwrap().state, "queued");
    let mut events = vec![];
    while let Some(event) = stream.message().await.unwrap() {
        events.push(event);
    }
    let ended = service.engine.get(&record.id).unwrap();
    let running = events
        .iter()
        .position(|event| event_state(event).is_some_and(|state| state.state == "running"))
        .expect("terminal page omitted the actual running fact");
    assert_eq!(events[running].sequence, ended.running_revision);
    assert_eq!(events[running].at_ms, ended.started_at_ms as i64);
    let endpoint = events
        .iter()
        .position(|event| {
            event_progress(event)
                .is_some_and(|progress| progress.stage == "denoise" && progress.completed == 30)
        })
        .unwrap();
    let outcome = events
        .iter()
        .position(|event| matches!(&event.event, Some(v1::run_event::Event::Outcome(_))))
        .unwrap();
    assert!(running < endpoint && endpoint < outcome);
    // Its immutable terminal projection and cursors survive a fresh journal reader.
    let reopened = crate::journal::Journal::open(&root.join("state/execution")).unwrap();
    let held = reopened.public_terminal(&record.id).unwrap().unwrap();
    use prost::Message;
    let page = pb::MachineExecutionEventPage::decode(held.events.as_slice()).unwrap();
    let replayed = page
        .events
        .iter()
        .find(|event| event.kind == "running")
        .unwrap();
    assert_eq!(replayed.sequence, ended.running_revision);
    assert_eq!(replayed.at_ms, ended.started_at_ms);
    server.abort();
}
