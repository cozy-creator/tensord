//! Explicit negative reproductions for the independent audit. These are CPU/storage/API
//! checks, not inference qualification. Run deliberately with --ignored.
use cozy_machine::{
    api::{self, auth::VerifiedActor, capability::{self, Grant}, pb, v1, MachineBackend, MachineIdentity},
    execution::Engine,
    journal::{InputFile, Invocation},
    objects::Objects,
    owner::Owner,
};
use ed25519_dalek::SigningKey;
use std::{fs, path::PathBuf, sync::Arc, time::{Duration, SystemTime, UNIX_EPOCH}};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

struct Area(PathBuf);
impl Area {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/audit-data").join(uuid::Uuid::new_v4().to_string());
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Area {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

#[test]
#[ignore = "negative audit reproduction: uploaded inputs lack GC roots"]
fn accepted_input_survives_gc() {
    let area = Area::new();
    let state = area.0.join("state");
    let owner = Owner::new(&state, &area.0.join("store"), 0, Duration::from_secs(60)).unwrap();
    let store = owner.lock().unwrap().store();
    let engine = Engine::open(&state).unwrap();
    let objects = Objects::new(&area.0.join("writes"), store.clone(), engine.clone()).unwrap();
    let bytes = b"input promised to an accepted run";
    let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(bytes));
    let mut writer = objects.begin("actor", &digest, bytes.len() as u64, 0).unwrap();
    writer.append(bytes).unwrap();
    writer.finish().unwrap();
    let (run, _) = engine.accept_run("actor", "audit-run", "intent", Invocation {
        package: "audit/package".into(), input: serde_json::json!({}),
        inputs: vec![InputFile {input_id: "file".into(), digest: digest.clone(),
            length: bytes.len() as u64, media_type: "application/octet-stream".into(), order: 0}],
        ..Default::default()
    }).unwrap();
    assert!(!run.state.terminal());
    let report = tensorfs_core::gc::collect(store.root(), false).unwrap();
    eprintln!("GC reclaimed {} bytes while the run was accepted", report.reclaimed_bytes);
    assert!(objects.path("actor", &digest).unwrap().is_some(),
        "GC deleted the accepted run's journal-referenced input");
}

#[test]
#[ignore = "negative audit reproduction: the exclusive lock names state, not store"]
fn one_store_has_one_machine_owner() {
    let area = Area::new();
    let store = area.0.join("shared-store");
    let _first = Owner::new(&area.0.join("state-a"), &store, 0, Duration::from_secs(60)).unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "store_owner_child"])
        .env("COZY_AUDIT_AREA", &area.0).status().unwrap();
    assert!(status.success());
    let accepted = fs::read_to_string(area.0.join("second-owner.txt")).unwrap();
    assert_eq!(accepted, "false", "another process with another state root owned the same store");
}

#[test]
#[ignore = "helper subprocess for one_store_has_one_machine_owner"]
fn store_owner_child() {
    let Ok(path) = std::env::var("COZY_AUDIT_AREA") else { return };
    let root = PathBuf::from(path);
    let second = Owner::new(&root.join("state-b"), &root.join("shared-store"), 0, Duration::from_secs(60));
    fs::write(root.join("second-owner.txt"), second.is_ok().to_string()).unwrap();
}

struct Events { complete: bool }
impl MachineBackend for Events {
    fn workspace(&self, _: VerifiedActor, _: pb::MachineExecutionWorkspaceQuery)
        -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Ok(pb::MachineExecutionWorkspace { execution_workspace_id: "audit".into(), ..Default::default() })
    }
    fn get(&self, _: VerifiedActor, _: pb::MachineExecutionQuery)
        -> Result<pb::MachineExecutionState, tonic::Status> {
        Ok(pb::MachineExecutionState {number: 1, state: if self.complete {"completed"} else {"running"}.into(),
            sequence: 3, ..Default::default()})
    }
    fn events(&self, _: VerifiedActor, request: pb::MachineExecutionEventsQuery)
        -> Result<pb::MachineExecutionEventPage, tonic::Status> {
        let events = if self.complete {
            let product = |sequence, byte| pb::MachineExecutionEvent { sequence, kind: "product".into(),
                product: Some(pb::RunProduct {output: "image".into(), op: pb::RunProductOp::Set as i32,
                    content: Some(pb::Ref {digest: vec![byte; 32], length: 3}), ..Default::default()}),
                ..Default::default()};
            vec![product(1, 1), product(2, 2), pb::MachineExecutionEvent {
                sequence: 3, kind: "outcome".into(), outcome: Some(pb::AttemptOutcome {
                    outcome_canonical_bytes: br#"{"status":1}"#.to_vec(), ..Default::default()}),
                ..Default::default()}].into_iter().filter(|e| e.sequence > request.after).collect()
        } else {
            // A synthetic event source, behind the real TLS server, keeps the stream moving.
            std::thread::sleep(Duration::from_millis(20));
            vec![pb::MachineExecutionEvent {sequence: request.after + 1, kind: "progress".into(),
                body_canonical_bytes: br#"{"payload":{"stage":"audit"}}"#.to_vec(), ..Default::default()}]
        };
        Ok(pb::MachineExecutionEventPage {events, next_after: request.after + 1, ..Default::default()})
    }
}

async fn client(complete: bool) -> (v1::machine_client::MachineClient<Channel>, api::auth::Keys, String, tokio::task::JoinHandle<()>) {
    let signer = SigningKey::from_bytes(&[91; 32]);
    let identity = MachineIdentity::ephemeral("audit-machine".into(), vec![signer.verifying_key()], vec![7; 32]).unwrap();
    let keys = identity.authority.keys.clone();
    let pem = identity.cert_pem.clone();
    let token = capability::mint(&signer, Grant {machine: "audit-machine".into(), action: capability::MACHINE.into(),
        expires: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 60,
        ..Default::default()});
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {api::serve(listener, identity, Arc::new(Events {complete})).await.unwrap()});
    let channel = Endpoint::from_shared(format!("https://{address}")).unwrap()
        .tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("localhost"))
        .unwrap().connect().await.unwrap();
    (v1::machine_client::MachineClient::new(channel), keys, token, server)
}
fn authorized<T>(value: T, token: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(value);
    request.metadata_mut().insert("authorization", format!("Cozy-Cap {token}").parse().unwrap());
    request
}

#[tokio::test]
#[ignore = "negative audit reproduction: Run resets output revision counters on reattach"]
async fn resumed_run_keeps_output_revisions() {
    let (mut client, _, token, server) = client(true).await;
    let mut stream = client.run(authorized(v1::RunRequest {id: "run".into(), after: 1, spec: None}, &token))
        .await.unwrap().into_inner();
    let _snapshot = stream.message().await.unwrap().unwrap();
    let event = stream.message().await.unwrap().unwrap();
    let Some(v1::run_event::Event::Product(product)) = event.event else {panic!("expected a product")};
    server.abort();
    assert_eq!(product.rev, 2, "the latest output became revision 1 after reattach");
}

#[tokio::test]
#[ignore = "negative audit reproduction: an outcome after a product cursor loses its outputs"]
async fn resumed_outcome_keeps_its_outputs() {
    let (mut client, _, token, server) = client(true).await;
    let mut stream = client.run(authorized(v1::RunRequest {id: "run".into(), after: 2, spec: None}, &token))
        .await.unwrap().into_inner();
    let _snapshot = stream.message().await.unwrap().unwrap();
    let event = stream.message().await.unwrap().unwrap();
    let Some(v1::run_event::Event::Outcome(outcome)) = event.event else {panic!("expected an outcome")};
    server.abort();
    assert_eq!(outcome.outputs.len(), 1, "the final outcome lost the output before the cursor");
}

#[tokio::test]
#[ignore = "negative audit reproduction: v1 Run does not observe signer revocation"]
async fn revoked_signer_loses_its_open_run_stream() {
    let (mut client, keys, token, server) = client(false).await;
    let mut stream = client.run(authorized(v1::RunRequest {id: "run".into(), after: 0, spec: None}, &token))
        .await.unwrap().into_inner();
    let _snapshot = stream.message().await.unwrap().unwrap();
    keys.revoke();
    // Allow already-buffered events through. The server's channel holds 16; observing 32
    // more events proves the revoked key still receives newly generated observations.
    let mut revoked = false;
    for _ in 0..32 {
        match tokio::time::timeout(Duration::from_secs(3), stream.message()).await {
            Ok(Err(status)) if status.code() == tonic::Code::Unauthenticated => {revoked = true; break;},
            Ok(Ok(Some(_))) => {},
            other => panic!("unexpected stream termination: {other:?}"),
        }
    }
    server.abort();
    assert!(revoked, "the revoked key still received 32 new run events");
}
