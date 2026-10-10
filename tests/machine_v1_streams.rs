//! `cozy.machine.v1` streams through the real TLS server, over a backend with a fixed log:
//! what a reattaching or output-limited observer sees, when a stream's authority ends, and
//! how a short output read fails.
use cozy_machine::api::{
    self,
    auth::{Keys, VerifiedActor},
    backend::OutputSnapshot,
    capability::{self, Grant},
    domain, v1, MachineBackend, MachineIdentity,
};
use ed25519_dalek::SigningKey;
use std::{
    io::{Seek, Write},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tonic::{
    transport::{Certificate, Channel, ClientTlsConfig, Endpoint},
    Code,
};
use v1::run_event::Event;

/// A finished run whose log revises `image` twice, sets `private`, reports author progress
/// and fails with a result and a triage bundle; a live run's log only grows.
struct Run {
    live: bool,
    /// Bytes `open_output` holds of the 8 it declares.
    held: usize,
}
impl MachineBackend for Run {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionWorkspaceQuery,
    ) -> Result<domain::MachineExecutionWorkspace, tonic::Status> {
        Ok(domain::MachineExecutionWorkspace { execution_workspace_id: "w".into(), ..Default::default() })
    }
    fn get(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionQuery,
    ) -> Result<domain::MachineExecutionState, tonic::Status> {
        let state = if self.live { "running" } else { "failed" };
        Ok(domain::MachineExecutionState { number: 1, state: state.into(), ..Default::default() })
    }
    fn events(
        &self,
        _: VerifiedActor,
        query: domain::MachineExecutionEventsQuery,
    ) -> Result<domain::MachineExecutionEventPage, tonic::Status> {
        let progress = |sequence| domain::MachineExecutionEvent {
            sequence,
            kind: "progress".into(),
            body_canonical_bytes: br#"{"payload":{"stage":"the author's prompt"}}"#.to_vec(),
            ..Default::default()
        };
        let events = if self.live {
            std::thread::sleep(Duration::from_millis(20));
            vec![progress(query.after + 1)]
        } else {
            let product = |sequence, output: &str, byte| domain::MachineExecutionEvent {
                sequence,
                kind: "product".into(),
                product: Some(domain::RunProduct {
                    output: output.into(),
                    op: domain::RunProductOp::Set as i32,
                    content: Some(domain::Ref { digest: vec![byte; 32], length: 8 }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let outcome = br#"{"status":3,"safe_message":"author_failed: the author's secret","result":{"inline_result":"c2VjcmV0"},"triage_bundle":{}}"#;
            vec![
                product(1, "image", 1),
                product(2, "image", 2),
                product(3, "private", 3),
                progress(4),
                domain::MachineExecutionEvent {
                    sequence: 5,
                    kind: "outcome".into(),
                    outcome: Some(domain::AttemptOutcome {
                        outcome_canonical_bytes: outcome.to_vec(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ]
            .into_iter()
            .filter(|event| event.sequence > query.after)
            .collect()
        };
        Ok(domain::MachineExecutionEventPage { events, next_after: query.after + 1, ..Default::default() })
    }
    fn measurements(&self, _: VerifiedActor, _: domain::MachineExecutionQuery) -> Result<Option<Vec<u8>>, tonic::Status> {
        Ok(Some(br#"{"device":"the owner's"}"#.to_vec()))
    }
    fn open_output(
        &self,
        _: VerifiedActor,
        _: u64,
        _: &str,
        _: Option<u32>,
    ) -> Result<OutputSnapshot, tonic::Status> {
        let path = std::env::temp_dir().join(format!("cm-v1-output-{}", uuid::Uuid::new_v4()));
        let mut file = std::fs::File::options().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        file.write_all(&b"8 bytes!"[..self.held]).unwrap();
        // A whole output is 128 MiB, sparse: a read stays open behind HTTP/2 flow control.
        let length = if self.held == 8 { 128 << 20 } else { 8 };
        if self.held == 8 {
            file.set_len(length).unwrap();
        }
        file.rewind().unwrap();
        Ok(OutputSnapshot {
            parts: vec![(file, length)],
            length,
            rev: 1,
            media_type: "application/octet-stream".into(),
            sha256: None,
        })
    }
}

const SIGNER: [u8; 32] = [91; 32];

fn cap(grant: Grant) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
    let expires = if grant.expires == 0 { now + 60 } else { grant.expires };
    capability::mint(&SigningKey::from_bytes(&SIGNER), Grant { machine: "m".into(), expires, ..grant })
}
fn machine_cap() -> String {
    cap(Grant { action: capability::MACHINE.into(), ..Default::default() })
}

async fn serve(run: Run) -> (v1::machine_client::MachineClient<Channel>, Keys) {
    let identity =
        MachineIdentity::ephemeral("m".into(), vec![SigningKey::from_bytes(&SIGNER).verifying_key()], vec![7; 32])
            .unwrap();
    let (keys, pem) = (identity.authority.keys.clone(), identity.cert_pem.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { api::serve(listener, identity, Arc::new(run)).await.unwrap() });
    let tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("localhost");
    let channel =
        Endpoint::from_shared(format!("https://{address}")).unwrap().tls_config(tls).unwrap().connect().await.unwrap();
    (v1::machine_client::MachineClient::new(channel), keys)
}

fn authorized<T>(value: T, token: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(value);
    request.metadata_mut().insert("authorization", format!("Cozy-Cap {token}").parse().unwrap());
    request
}

async fn attach(
    client: &mut v1::machine_client::MachineClient<Channel>,
    token: &str,
    after: u64,
) -> tonic::Streaming<v1::RunEvent> {
    let request = v1::RunRequest { id: "run".into(), after, spec: None };
    let mut events = client.run(authorized(request, token)).await.unwrap().into_inner();
    let snapshot = events.message().await.unwrap().unwrap();
    assert!(matches!(snapshot.event, Some(Event::State(_))) && snapshot.sequence == 0);
    events
}

async fn read(
    client: &mut v1::machine_client::MachineClient<Channel>,
    offset: u64,
) -> tonic::Streaming<v1::ReadFrame> {
    read_range(client, offset, 0).await
}

async fn read_range(
    client: &mut v1::machine_client::MachineClient<Channel>,
    offset: u64,
    length: u64,
) -> tonic::Streaming<v1::ReadFrame> {
    let target = v1::OutputTarget { run: "run".into(), output: "image".into(), index: 0, ..Default::default() };
    let request =
        v1::ReadRequest { target: Some(v1::read_request::Target::Output(target)), offset, length, ..Default::default() };
    client.read(authorized(request, &machine_cap())).await.unwrap().into_inner()
}

async fn rest(mut frames: tonic::Streaming<v1::ReadFrame>) -> Result<Vec<u8>, tonic::Status> {
    let mut bytes = vec![];
    while let Some(frame) = frames.message().await? {
        bytes.extend(frame.data);
    }
    Ok(bytes)
}

/// A reattach (every daemon restart) past earlier products keeps each output's revision and
/// the outcome's whole inventory; a cursor at or past the outcome gets the snapshot only.
#[tokio::test]
async fn a_reattach_keeps_revisions_and_the_outcome_keeps_every_output() {
    let (mut client, _) = serve(Run { live: false, held: 8 }).await;
    let token = machine_cap();
    let mut events = attach(&mut client, &token, 1).await;
    let Some(Event::Product(product)) = events.message().await.unwrap().unwrap().event else {
        panic!("a product follows the snapshot")
    };
    assert_eq!((product.output.as_str(), product.rev), ("image", 2));

    let mut events = attach(&mut client, &token, 3).await;
    let outcome = loop {
        if let Some(Event::Outcome(outcome)) = events.message().await.unwrap().unwrap().event {
            break outcome;
        }
    };
    let outputs: Vec<_> = outcome.outputs.iter().map(|p| (p.output.as_str(), p.rev)).collect();
    assert_eq!(outputs, [("image", 2), ("private", 1)]);
    assert!(!outcome.measurements.is_empty() && !outcome.result.is_empty());

    for after in [5, 100] {
        assert!(attach(&mut client, &token, after).await.message().await.unwrap().is_none());
    }
}

/// `run play`'s output-limited link sees its outputs and the outcome's status, never the
/// author's progress, result, failure text, measurements or triage.
#[tokio::test]
async fn an_output_limited_grant_sees_only_its_outputs() {
    let (mut client, _) = serve(Run { live: false, held: 8 }).await;
    let token = cap(Grant { run: "run".into(), outputs: vec!["image".into()], ..Default::default() });
    let mut events = attach(&mut client, &token, 0).await;
    let mut outcome = None;
    while let Some(event) = events.message().await.unwrap() {
        match event.event {
            Some(Event::Product(product)) => assert_eq!(product.output, "image"),
            Some(Event::Outcome(seen)) => outcome = Some(seen),
            Some(Event::State(state)) => assert!(state.waiting.is_empty()),
            other => panic!("the limited grant saw {other:?}"),
        }
    }
    let outcome = outcome.unwrap();
    assert_eq!(outcome.status, "failed");
    assert_eq!(outcome.outputs.iter().map(|p| p.output.as_str()).collect::<Vec<_>>(), ["image"]);
    assert!(outcome.result.is_empty() && outcome.reason.is_none() && !outcome.triage);
    assert!(outcome.measurements.is_empty());
    let triage = v1::ReadRequest { target: Some(v1::read_request::Target::Triage("run".into())), ..Default::default() };
    assert_eq!(client.read(authorized(triage, &token)).await.unwrap_err().code(), Code::PermissionDenied);
}

/// A run observer and a read outlive their capability's expiry, which held when they opened;
/// revoking the key that opened them ends both.
#[tokio::test]
async fn open_streams_end_on_revocation_not_at_expiry() {
    let (mut client, keys) = serve(Run { live: true, held: 8 }).await;
    let expires = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 2;
    let token = cap(Grant { action: capability::MACHINE.into(), expires, ..Default::default() });
    let mut events = attach(&mut client, &token, 0).await;
    let mut bytes = read(&mut client, 0).await;
    assert_eq!(bytes.message().await.unwrap().unwrap().length, 128 << 20);
    let past_expiry = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < past_expiry {
        events.message().await.expect("the observer outlived its cap").unwrap();
    }
    keys.revoke();
    // What was already queued may still arrive (the server buffers 16 events); nothing new does.
    let mut queued = 0;
    let ended = loop {
        match events.message().await {
            Ok(Some(_)) => queued += 1,
            other => break other,
        }
        assert!(queued < 64, "the revoked observer still receives new events");
    };
    assert_eq!(ended.unwrap_err().code(), Code::Unauthenticated);
    let mut received = 0;
    let ended = loop {
        match bytes.message().await {
            Ok(Some(frame)) => received += frame.data.len(),
            other => break other,
        }
        assert!(received < 64 << 20, "the revoked read went on for {received} bytes");
    };
    assert_eq!(ended.unwrap_err().code(), Code::Unauthenticated);
}

/// An output whose file is shorter than its revision is DATA_LOSS, never a short success.
#[tokio::test]
async fn a_short_output_read_is_data_loss() {
    let (mut client, _) = serve(Run { live: false, held: 3 }).await;
    let mut bytes = read(&mut client, 0).await;
    assert_eq!(bytes.message().await.unwrap().unwrap().length, 8);
    assert_eq!(bytes.message().await.unwrap().unwrap().data, b"8 b");
    assert_eq!(bytes.message().await.unwrap_err().code(), Code::DataLoss);
    let mut bytes = read(&mut client, 5).await;
    bytes.message().await.unwrap().unwrap();
    assert_eq!(bytes.message().await.unwrap_err().code(), Code::DataLoss);
}

/// A range is one part of a parallel read: exactly its bytes, its end in the first frame,
/// clamped at the output's end, and a short output is short only where the range reaches.
#[tokio::test]
async fn a_range_reads_its_bytes_alone() {
    let (mut client, _) = serve(Run { live: false, held: 8 }).await;
    let mut frames = read_range(&mut client, 2, 4).await;
    let meta = frames.message().await.unwrap().unwrap();
    assert_eq!((meta.length, meta.end), (128 << 20, 6));
    assert_eq!(rest(frames).await.unwrap(), b"byte");

    let mut frames = read_range(&mut client, (128 << 20) - 3, 1 << 20).await;
    assert_eq!(frames.message().await.unwrap().unwrap().end, 128 << 20);
    assert_eq!(rest(frames).await.unwrap(), [0; 3]);

    let mut frames = read(&mut client, (128 << 20) - 1).await;
    assert_eq!(frames.message().await.unwrap().unwrap().end, 128 << 20);

    let (mut client, _) = serve(Run { live: false, held: 3 }).await;
    let mut frames = read_range(&mut client, 0, 3).await;
    assert_eq!(frames.message().await.unwrap().unwrap().end, 3);
    assert_eq!(rest(frames).await.unwrap(), b"8 b");
    let mut frames = read_range(&mut client, 1, 4).await;
    frames.message().await.unwrap().unwrap();
    assert_eq!(rest(frames).await.unwrap_err().code(), Code::DataLoss);
}
