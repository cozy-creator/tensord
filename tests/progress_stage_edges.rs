//! Genuine controlled progress frames through Engine, durable journal and real TLS Run.
//! CPU observation proof only: no model or GPU inference is qualified here.
use cozy_machine::{
    api::{
        self,
        capability::{self, Grant},
        v1, MachineIdentity,
    },
    journal::{Invocation, Outcome, MAX_PROGRESS_STAGE_EDGES},
    machine_api::NativeBackend,
    service::Service,
};
use ed25519_dalek::SigningKey;
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

struct Fixture {
    root: PathBuf,
    service: Arc<Service>,
    id: String,
}
impl Fixture {
    fn open() -> Self {
        let root = std::env::temp_dir().join(format!("cm-progress-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        assert!(service.stop().unwrap()); // No package is dispatched by this CPU fixture.
        let signer = SigningKey::from_bytes(&[91; 32]);
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
        Self {
            root,
            service,
            id: record.id,
        }
    }
    fn progress(&self, units: u64, stage: &str, position: u64) {
        self.service
            .engine
            .observe_progress(
                &self.id,
                units,
                json!({"stage":stage,"position":position,"total":30}).to_string(),
            )
            .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = self.service.stop();
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn stage_projection_is_bounded_and_only_flushes_with_a_real_transition() {
    let fixture = Fixture::open();
    let engine = &fixture.service.engine;
    fixture.progress(0, "denoise", 0); // The existing initial cursor reservation is durable.
    let before = engine
        .with_journal(|journal| journal.get(&fixture.id))
        .unwrap();
    for position in 1..=2000 {
        fixture.progress(position, "denoise", position);
    }
    let live = engine.get(&fixture.id).unwrap();
    assert_eq!(live.progress_samples.len(), 1);
    assert_eq!(live.progress_samples[0].completed_units, 2000);
    let unchanged = engine
        .with_journal(|journal| journal.get(&fixture.id))
        .unwrap();
    assert_eq!(unchanged.revision, before.revision);
    assert!(unchanged.progress_samples.is_empty());
    for index in 0..100 {
        fixture.progress(2000, &format!("phase-{index}"), index);
    }
    engine
        .observe_progress(
            &fixture.id,
            2000,
            json!({"stage":"🙂".repeat(2000)}).to_string(),
        )
        .unwrap();
    let live = engine.get(&fixture.id).unwrap();
    assert_eq!(live.progress_samples.len(), MAX_PROGRESS_STAGE_EDGES + 1);
    assert!(live
        .progress_samples
        .windows(2)
        .all(|pair| pair[0].revision < pair[1].revision));
    assert!(live
        .progress_samples
        .iter()
        .all(|sample| sample.detail.len() <= 2048));
    let finished = engine
        .end_preparation(
            &fixture.id,
            Outcome::Failed("owned CPU fixture complete".into()),
        )
        .unwrap();
    assert_eq!(
        finished.progress_samples.len(),
        MAX_PROGRESS_STAGE_EDGES + 1
    );
    let stored = engine
        .with_journal(|journal| journal.get(&fixture.id))
        .unwrap();
    assert_eq!(
        stored.progress_samples.last().unwrap().detail,
        live.progress_samples.last().unwrap().detail
    );
    let mut older = serde_json::to_value(&stored).unwrap();
    older.as_object_mut().unwrap().remove("progress_samples");
    let decoded: cozy_machine::journal::Execution = serde_json::from_value(older).unwrap();
    assert!(decoded.progress_samples.is_empty());
}

#[test]
fn prepared_binding_retains_samples_without_reusing_preparation_work_units() {
    let fixture = Fixture::open();
    fixture.progress(1000, "downloading", 1);
    fixture.progress(2000, "checking", 2);
    let bound = fixture
        .service
        .engine
        .bind_prepared(&fixture.id, Invocation::default(), "")
        .unwrap();
    assert_eq!(bound.completed_units, 0);
    assert!(bound.progress.is_none());
    assert_eq!(bound.progress_samples.len(), 2);
    assert_eq!(bound.progress_samples[0].completed_units, 1000);
}

#[test]
fn unreadable_advisory_samples_do_not_hide_an_accepted_run() {
    let fixture = Fixture::open();
    fixture.progress(30, "denoise", 30);
    let original = fixture.service.engine.get(&fixture.id).unwrap();
    for unsupported in [
        json!(null),
        json!({"future":"projection"}),
        json!([{"detail":"unknown shape"}]),
    ] {
        let mut value = serde_json::to_value(&original).unwrap();
        value["progress_samples"] = unsupported;
        let decoded: cozy_machine::journal::Execution = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.state, original.state);
        assert!(decoded.progress_samples.is_empty());
    }
    let mut value = serde_json::to_value(&original).unwrap();
    value["progress_samples"]
        .as_array_mut()
        .unwrap()
        .push(json!({"future":"sample"}));
    let decoded: cozy_machine::journal::Execution = serde_json::from_value(value).unwrap();
    assert_eq!(decoded.progress_samples.len(), 1);
    assert_eq!(
        decoded.progress_samples[0].revision,
        original.progress_samples[0].revision
    );
}

#[tokio::test]
async fn burst_stage_endpoint_survives_real_tls_observer_and_terminal_journal_reopen() {
    let fixture = Fixture::open();
    fixture.progress(29, "denoise", 29);
    fixture.progress(30, "denoise", 30);
    fixture.progress(30, "decoding", 0); // The observer has not yet had a turn.
    let service = &fixture.service;
    let signer = SigningKey::from_bytes(&[91; 32]);
    let identity =
        MachineIdentity::ephemeral("progress".into(), vec![signer.verifying_key()], vec![7; 32])
            .unwrap();
    let pem = identity.cert_pem.clone();
    let store = Arc::new(tensorfs_core::store::Store::ensure(&fixture.root.join("store")).unwrap());
    let uploads =
        api::workspaces::WorkspaceUploads::open(&fixture.root.join("uploads"), store.clone())
            .unwrap();
    let backend = Arc::new(NativeBackend::new(
        service.clone(),
        identity.authority.clone(),
        store,
        uploads,
    ));
    let token = capability::mint(
        &signer,
        Grant {
            machine: "progress".into(),
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
    let server =
        tokio::spawn(async move { api::serve(listener, identity, backend).await.unwrap() });
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
    let request = |after| {
        let mut request = tonic::Request::new(v1::RunRequest {
            id: "run".into(),
            after,
            spec: None,
        });
        request.metadata_mut().insert(
            "authorization",
            format!("Cozy-Cap {token}").parse().unwrap(),
        );
        request
    };
    let mut stream = client.run(request(0)).await.unwrap().into_inner();
    let mut samples = vec![];
    while samples.len() < 2 {
        let event = stream.message().await.unwrap().unwrap();
        if let Some(v1::run_event::Event::Progress(progress)) = event.event {
            samples.push((event.sequence, event.at_ms, progress));
        }
    }
    assert_eq!(samples[0].2.stage, "denoise");
    assert_eq!(samples[0].2.completed, 30);
    assert_eq!(samples[1].2.stage, "decoding");
    assert!(samples[0].0 < samples[1].0);
    drop(stream); // Detaching observes; it supplies no work cancellation authority.
    assert!(service
        .engine
        .get(&fixture.id)
        .unwrap()
        .cancel_actor
        .is_none());
    service
        .engine
        .end_preparation(
            &fixture.id,
            Outcome::Failed("owned CPU fixture complete".into()),
        )
        .unwrap();
    let mut terminal = client.run(request(0)).await.unwrap().into_inner();
    let mut replay = vec![];
    while let Some(event) = terminal.message().await.unwrap() {
        assert!(
            !matches!(&event.event, Some(v1::run_event::Event::State(state)) if state.state == "running"),
            "preparation-only work acquired an invented running fact"
        );
        if let Some(v1::run_event::Event::Progress(progress)) = event.event {
            replay.push((event.sequence, event.at_ms, progress));
        }
    }
    assert_eq!(replay, samples); // Original frames, cursors and observation times survive.
    let reopened =
        cozy_machine::journal::Journal::open(&fixture.root.join("state/execution")).unwrap();
    let held = reopened.get(&fixture.id).unwrap();
    assert_eq!(held.progress_samples[0].revision, samples[0].0);
    assert_eq!(held.progress_samples[0].at_ms as i64, samples[0].1);
    server.abort();
}
