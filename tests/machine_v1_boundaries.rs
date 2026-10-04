//! CPU checks against the real TLS API, with controlled backend events. These qualify
//! observer and authorization boundaries, not inference.
use cozy_machine::api::{
    self,
    auth::VerifiedActor,
    capability::{self, Grant},
    pb, v1, MachineBackend, MachineIdentity,
};
use ed25519_dalek::SigningKey;
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

struct Events {
    complete: bool,
    private: bool,
}
impl MachineBackend for Events {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Ok(pb::MachineExecutionWorkspace {
            execution_workspace_id: "audit".into(),
            ..Default::default()
        })
    }
    fn get(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionQuery,
    ) -> Result<pb::MachineExecutionState, tonic::Status> {
        Ok(pb::MachineExecutionState {
            number: 1,
            state: if self.complete {
                "completed"
            } else {
                "running"
            }
            .into(),
            sequence: 3,
            ..Default::default()
        })
    }
    fn events(
        &self,
        _: VerifiedActor,
        request: pb::MachineExecutionEventsQuery,
    ) -> Result<pb::MachineExecutionEventPage, tonic::Status> {
        let events = if self.complete {
            let product = |sequence, byte| pb::MachineExecutionEvent {
                sequence,
                kind: "product".into(),
                product: Some(pb::RunProduct {
                    output: "image".into(),
                    op: pb::RunProductOp::Set as i32,
                    content: Some(pb::Ref {
                        digest: vec![byte; 32],
                        length: 3,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mut log = vec![product(1, 1), product(2, 2)];
            if self.private {
                let mut hidden = product(3, 3);
                hidden.product.as_mut().unwrap().output = "private".into();
                log.push(hidden);
            }
            log.push(pb::MachineExecutionEvent {
                sequence: if self.private { 4 } else { 3 },
                kind: "outcome".into(),
                outcome: Some(pb::AttemptOutcome {
                    outcome_canonical_bytes:
                        br#"{"status":1,"result":{"inline_result":"c2VjcmV0"},"triage_bundle":{}}"#
                            .to_vec(),
                    ..Default::default()
                }),
                ..Default::default()
            });
            log.into_iter()
                .filter(|e| e.sequence > request.after)
                .collect()
        } else {
            // A synthetic event source, behind the real TLS server, keeps the stream moving.
            std::thread::sleep(Duration::from_millis(20));
            vec![pb::MachineExecutionEvent {
                sequence: request.after + 1,
                kind: "progress".into(),
                body_canonical_bytes: br#"{"payload":{"stage":"audit"}}"#.to_vec(),
                ..Default::default()
            }]
        };
        Ok(pb::MachineExecutionEventPage {
            events,
            next_after: request.after + 1,
            ..Default::default()
        })
    }
}

fn cap(mut grant: Grant) -> String {
    grant.machine = "audit-machine".into();
    if grant.expires == 0 {
        grant.expires = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            + 60;
    }
    capability::mint(&SigningKey::from_bytes(&[91; 32]), grant)
}

async fn client(
    complete: bool,
) -> (
    v1::machine_client::MachineClient<Channel>,
    api::auth::Keys,
    String,
    tokio::task::JoinHandle<()>,
) {
    serve(Arc::new(Events {
        complete,
        private: false,
    }))
    .await
}
async fn serve<B: MachineBackend>(
    backend: Arc<B>,
) -> (
    v1::machine_client::MachineClient<Channel>,
    api::auth::Keys,
    String,
    tokio::task::JoinHandle<()>,
) {
    let signer = SigningKey::from_bytes(&[91; 32]);
    let identity = MachineIdentity::ephemeral(
        "audit-machine".into(),
        vec![signer.verifying_key()],
        vec![7; 32],
    )
    .unwrap();
    let keys = identity.authority.keys.clone();
    let pem = identity.cert_pem.clone();
    let token = capability::mint(
        &signer,
        Grant {
            machine: "audit-machine".into(),
            action: capability::MACHINE.into(),
            expires: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
                + 60,
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
    (
        v1::machine_client::MachineClient::new(channel),
        keys,
        token,
        server,
    )
}
fn authorized<T>(value: T, token: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(value);
    request.metadata_mut().insert(
        "authorization",
        format!("Cozy-Cap {token}").parse().unwrap(),
    );
    request
}

#[tokio::test]
async fn resumed_run_keeps_output_revisions() {
    let (mut client, _, token, server) = client(true).await;
    let mut stream = client
        .run(authorized(
            v1::RunRequest {
                id: "run".into(),
                after: 1,
                spec: None,
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    let _snapshot = stream.message().await.unwrap().unwrap();
    let event = stream.message().await.unwrap().unwrap();
    let Some(v1::run_event::Event::Product(product)) = event.event else {
        panic!("expected a product")
    };
    server.abort();
    assert_eq!(
        product.rev, 2,
        "the latest output became revision 1 after reattach"
    );
}

#[tokio::test]
async fn resumed_outcome_keeps_its_outputs() {
    let (mut client, _, token, server) = client(true).await;
    let mut stream = client
        .run(authorized(
            v1::RunRequest {
                id: "run".into(),
                after: 2,
                spec: None,
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    let _snapshot = stream.message().await.unwrap().unwrap();
    let event = stream.message().await.unwrap().unwrap();
    let Some(v1::run_event::Event::Outcome(outcome)) = event.event else {
        panic!("expected an outcome")
    };
    server.abort();
    assert_eq!(
        outcome.outputs.len(),
        1,
        "the final outcome lost the output before the cursor"
    );
}

#[tokio::test]
async fn revoked_signer_loses_its_open_run_stream() {
    let (mut client, keys, token, server) = client(false).await;
    let mut stream = client
        .run(authorized(
            v1::RunRequest {
                id: "run".into(),
                after: 0,
                spec: None,
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    let _snapshot = stream.message().await.unwrap().unwrap();
    keys.revoke();
    // Allow already-buffered events through. The server's channel holds 16; observing 32
    // more events proves the revoked key still receives newly generated observations.
    let mut revoked = false;
    for _ in 0..32 {
        match tokio::time::timeout(Duration::from_secs(3), stream.message()).await {
            Ok(Err(status)) if status.code() == tonic::Code::Unauthenticated => {
                revoked = true;
                break;
            }
            Ok(Ok(Some(_))) => {}
            other => panic!("unexpected stream termination: {other:?}"),
        }
    }
    server.abort();
    assert!(revoked, "the revoked key still received 32 new run events");
}

#[tokio::test]
async fn terminal_cursor_at_or_beyond_outcome_closes_after_snapshot() {
    let (mut client, _, token, server) = client(true).await;
    for after in [3, 100] {
        let mut stream = client
            .run(authorized(
                v1::RunRequest {
                    id: "run".into(),
                    after,
                    spec: None,
                },
                &token,
            ))
            .await
            .unwrap()
            .into_inner();
        assert!(matches!(
            stream.message().await.unwrap().unwrap().event,
            Some(v1::run_event::Event::State(_))
        ));
        assert!(
            tokio::time::timeout(Duration::from_secs(2), stream.message())
                .await
                .unwrap()
                .unwrap()
                .is_none()
        );
    }
    server.abort();
}

#[tokio::test]
async fn output_limited_cap_hides_private_products_result_and_triage() {
    let (mut client, _, _, server) = serve(Arc::new(Events {
        complete: true,
        private: true,
    }))
    .await;
    let token = cap(Grant {
        run: "run".into(),
        outputs: vec!["image".into()],
        ..Default::default()
    });
    let mut stream = client
        .run(authorized(
            v1::RunRequest {
                id: "run".into(),
                ..Default::default()
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    let mut outcome = None;
    while let Some(event) = stream.message().await.unwrap() {
        match event.event {
            Some(v1::run_event::Event::Product(product)) => assert_eq!(product.output, "image"),
            Some(v1::run_event::Event::Outcome(value)) => outcome = Some(value),
            _ => (),
        }
    }
    let outcome = outcome.unwrap();
    assert_eq!(outcome.outputs.len(), 1);
    assert_eq!(outcome.outputs[0].output, "image");
    assert!(outcome.result.is_empty());
    assert!(!outcome.triage);
    assert!(outcome.measurements.is_empty());
    let refusal = client
        .read(authorized(
            v1::ReadRequest {
                target: Some(v1::read_request::Target::Triage("run".into())),
                ..Default::default()
            },
            &token,
        ))
        .await
        .unwrap_err();
    assert_eq!(refusal.code(), tonic::Code::PermissionDenied);
    server.abort();
}

#[tokio::test]
async fn expiry_ends_a_quiet_status_stream_and_a_run_observer() {
    let (mut client, _, _, server) = client(false).await;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let token = cap(Grant {
        action: capability::MACHINE.into(),
        expires: now + 2,
        ..Default::default()
    });
    let mut status = client
        .status(authorized(v1::StatusRequest::default(), &token))
        .await
        .unwrap()
        .into_inner();
    assert!(status.message().await.unwrap().is_some());
    let mut run = client
        .run(authorized(
            v1::RunRequest {
                id: "run".into(),
                ..Default::default()
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(run.message().await.unwrap().is_some());
    let refusal = tokio::time::timeout(Duration::from_secs(3), status.message())
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(refusal.code(), tonic::Code::Unauthenticated);
    loop {
        match tokio::time::timeout(Duration::from_secs(2), run.message())
            .await
            .unwrap()
        {
            Err(refusal) => {
                assert_eq!(refusal.code(), tonic::Code::Unauthenticated);
                break;
            }
            Ok(Some(_)) => (),
            Ok(None) => panic!("expiry ended without an authorization refusal"),
        }
    }
    // A new valid cap can still observe that same run; no Control was issued by expiry.
    let token = cap(Grant {
        action: capability::MACHINE.into(),
        ..Default::default()
    });
    let mut attached = client
        .run(authorized(
            v1::RunRequest {
                id: "run".into(),
                ..Default::default()
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    let snapshot = attached.message().await.unwrap().unwrap();
    let Some(v1::run_event::Event::State(state)) = snapshot.event else {
        panic!("state expected")
    };
    assert_eq!(state.state, "running");
    server.abort();
}

struct Output;
impl MachineBackend for Output {
    fn workspace(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Events {
            complete: false,
            private: false,
        }
        .workspace(actor, query)
    }
    fn get(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionQuery,
    ) -> Result<pb::MachineExecutionState, tonic::Status> {
        Events {
            complete: false,
            private: false,
        }
        .get(actor, query)
    }
    fn open_output(
        &self,
        _: VerifiedActor,
        _: u64,
        _: &str,
        _: Option<u32>,
    ) -> Result<api::backend::OutputSnapshot, tonic::Status> {
        // A sparse anonymous file keeps the read behind HTTP/2 backpressure without disk I/O.
        let path = std::env::temp_dir().join(format!("cm-read-{}", uuid::Uuid::new_v4()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(path).unwrap();
        let length = 128 << 20;
        file.set_len(length).unwrap();
        Ok(api::backend::OutputSnapshot {
            parts: vec![(file, length)],
            length,
            rev: 1,
            media_type: "application/octet-stream".into(),
            sha256: None,
        })
    }
}

#[tokio::test]
async fn revoked_signer_loses_a_backpressured_output_read() {
    let (mut client, keys, token, server) = serve(Arc::new(Output)).await;
    let mut stream = client
        .read(authorized(
            v1::ReadRequest {
                target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                    run: "run".into(),
                    output: "image".into(),
                    index: 0,
                })),
                ..Default::default()
            },
            &token,
        ))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stream.message().await.unwrap().unwrap().length, 128 << 20);
    keys.revoke();
    // Allow transport buffers, but require refusal well before the 128 MiB output ends.
    let mut read = 0;
    loop {
        match tokio::time::timeout(Duration::from_secs(3), stream.message())
            .await
            .unwrap()
        {
            Err(status) => {
                assert_eq!(status.code(), tonic::Code::Unauthenticated);
                break;
            }
            Ok(Some(frame)) => {
                read += frame.data.len();
                assert!(read < 64 << 20);
            }
            Ok(None) => panic!("revoked read served its whole output"),
        }
    }
    server.abort();
}

struct Writes {
    runs: Arc<cozy_machine::runs::Runs>,
    root: std::path::PathBuf,
}
impl Writes {
    fn new() -> Arc<Self> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/audit-data")
            .join(uuid::Uuid::new_v4().to_string());
        let service =
            cozy_machine::service::Service::open(&root.join("state"), &root.join("generations"), 1)
                .unwrap();
        let store = Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(
            cozy_machine::objects::Objects::new(
                &root.join("writes"),
                store,
                service.engine.clone(),
            )
            .unwrap(),
        );
        let runs = Arc::new(cozy_machine::runs::Runs {
            service,
            objects,
            publisher: None,
            local: None,
            own_hub: None,
            jobs: Default::default(),
        });
        Arc::new(Self { runs, root })
    }
}
impl Drop for Writes {
    fn drop(&mut self) {
        self.runs.service.stop().unwrap();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
impl MachineBackend for Writes {
    fn workspace(
        &self,
        actor: VerifiedActor,
        query: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Events {
            complete: false,
            private: false,
        }
        .workspace(actor, query)
    }
    fn runs(&self) -> Option<Arc<cozy_machine::runs::Runs>> {
        Some(self.runs.clone())
    }
}

#[tokio::test]
async fn revoked_signer_cannot_keep_a_write_open_or_finalize_it() {
    let writes = Writes::new();
    let (mut client, keys, token, server) = serve(writes.clone()).await;
    let bytes = b"an upload stopped by key revocation";
    let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(bytes));
    let (sender, receiver) = tokio::sync::mpsc::channel(2);
    sender
        .send(v1::WriteFrame {
            digest: digest.clone(),
            length: bytes.len() as u64,
            data: bytes[..4].to_vec(),
            ..Default::default()
        })
        .await
        .unwrap();
    let writing = tokio::spawn(async move {
        client
            .write(authorized(
                tokio_stream::wrappers::ReceiverStream::new(receiver),
                &token,
            ))
            .await
    });
    // Wait until the actual staging writer has appended; revocation then interrupts the
    // pending incoming frame without requiring the uploader to send more bytes or EOF.
    let until = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        let staged = std::fs::read_dir(writes.root.join("writes"))
            .unwrap()
            .filter_map(Result::ok)
            .any(|actor| {
                std::fs::read_dir(actor.path())
                    .unwrap()
                    .filter_map(Result::ok)
                    .any(|file| file.metadata().unwrap().len() == 4)
            });
        if staged {
            break;
        }
        assert!(tokio::time::Instant::now() < until);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    keys.revoke();
    let refusal = tokio::time::timeout(Duration::from_secs(3), writing)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(refusal.code(), tonic::Code::Unauthenticated);
    let actor =
        tensorfs_core::sha256::hex(&SigningKey::from_bytes(&[91; 32]).verifying_key().to_bytes());
    assert!(writes.runs.objects.path(&actor, &digest).unwrap().is_none());
    drop(sender);
    server.abort();
}
