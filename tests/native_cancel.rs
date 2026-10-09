//! Native cancellation uses real SQLite transactions and the TLS machine API. No SDK/GPU needed.
use cozy_machine::{
    api::{
        self,
        auth::VerifiedActor,
        capability::{self, Grant},
        domain, v1, MachineBackend, MachineIdentity,
    },
    execution::Engine,
    journal::{Invocation, Journal, Outcome, ResultRecord, State},
    machine_api::{actor_id, NativeBackend},
    objects::Objects,
    runs::Runs,
    service::Service,
};
use ed25519_dalek::SigningKey;
use fs2::FileExt;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Barrier},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tonic::{
    transport::{Certificate, Channel, ClientTlsConfig, Endpoint},
    Code, Request,
};

const ALICE: [u8; 32] = [51; 32];
const BOB: [u8; 32] = [52; 32];

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("cm-native-cancel-{}", uuid::Uuid::new_v4())))
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn actor(key: [u8; 32]) -> String {
    actor_id(VerifiedActor {
        public_key: SigningKey::from_bytes(&key).verifying_key().to_bytes(),
    })
}
fn request<T>(value: T, key: [u8; 32], run: Option<&str>) -> Request<T> {
    let expires = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 120;
    let mut grant = Grant {
        machine: "cancel-proof".into(),
        expires,
        ..Default::default()
    };
    if let Some(id) = run {
        grant.run = id.into();
    } else {
        grant.action = capability::MACHINE.into();
    }
    let token = capability::mint(&SigningKey::from_bytes(&key), grant);
    let mut request = Request::new(value);
    request.metadata_mut().insert(
        "authorization",
        format!("Cozy-Cap {token}").parse().unwrap(),
    );
    request
}
fn spec() -> v1::RunSpec {
    v1::RunSpec {
        kind: v1::RunKind::Job as i32,
        source: Some(v1::run_spec::Source::Installation(
            "unused-installation".into(),
        )),
        entrypoint: "must_not_prepare".into(),
        payload: b"{}".to_vec(),
        ..Default::default()
    }
}

struct Server {
    client: v1::machine_client::MachineClient<Channel>,
    service: Arc<Service>,
    backend: Arc<NativeBackend>,
    task: tokio::task::JoinHandle<()>,
}
impl Server {
    async fn open(root: &Path) -> Self {
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(
            Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap(),
        );
        let mut identity = MachineIdentity::ephemeral(
            "cancel-proof".into(),
            [ALICE, BOB]
                .map(|key| SigningKey::from_bytes(&key).verifying_key())
                .to_vec(),
            vec![7; 32],
        )
        .unwrap();
        let paths =
            cozy_machine::machine::update::Paths::new(&root.join("state"), &root.join("image"));
        identity.updates = Some(
            cozy_machine::machine::update::Updates::open(paths, Box::new(|| true), None).unwrap(),
        );
        let pem = identity.cert_pem.clone();
        let mut backend = NativeBackend::new(service.clone(), identity.authority.clone(), store);
        backend.runs = Some(Arc::new(Runs {
            service: service.clone(),
            objects,
            publisher: None,
            local: None,
            own_hub: None,
            leaf: None,
            jobs: Default::default(),
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = Arc::new(backend);
        let served = backend.clone();
        let task =
            tokio::spawn(async move { api::serve(listener, identity, served).await.unwrap() });
        let tls = ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(pem))
            .domain_name("localhost");
        let channel = Endpoint::from_shared(format!("https://{address}"))
            .unwrap()
            .tls_config(tls)
            .unwrap()
            .connect()
            .await
            .unwrap();
        Self {
            client: v1::machine_client::MachineClient::new(channel),
            service,
            backend,
            task,
        }
    }
    async fn cancel(&mut self, id: &str, key: [u8; 32]) -> v1::RunState {
        self.client
            .control(request(
                v1::ControlRequest {
                    id: id.into(),
                    action: v1::Action::Cancel as i32,
                },
                key,
                None,
            ))
            .await
            .unwrap()
            .into_inner()
    }
    async fn canceled(&mut self, id: &str, sent: Option<v1::RunSpec>) {
        let mut stream = self
            .client
            .run(request(
                v1::RunRequest {
                    id: id.into(),
                    spec: sent,
                    after: 0,
                },
                ALICE,
                None,
            ))
            .await
            .unwrap()
            .into_inner();
        let mut outcome = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(5), stream.message())
            .await
            .unwrap()
            .unwrap()
        {
            match event.event.unwrap() {
                v1::run_event::Event::State(state) => assert_eq!(state.state, "canceled"),
                v1::run_event::Event::Outcome(seen) => {
                    assert_eq!(seen.status, "canceled");
                    outcome = true;
                }
                other => panic!("canceled before acceptance emitted {other:?}"),
            }
        }
        assert!(outcome);
    }
    async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

#[tokio::test]
async fn canceled_ids_refuse_updates_and_another_actors_update_never_shadows_them() {
    let root = Root::new();
    let mut server = Server::open(&root.0).await;
    let canceled = server.cancel("shared", ALICE).await;
    let update = v1::RunSpec {
        kind: v1::RunKind::Update as i32,
        payload: br#"{"runtime":"0.19.0"}"#.to_vec(),
        ..Default::default()
    };
    let error = server
        .client
        .run(request(
            v1::RunRequest {
                id: "shared".into(),
                spec: Some(update),
                after: 0,
            },
            ALICE,
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    server
        .service
        .engine
        .with_journal(|journal| journal.reserve_update_run(&actor(BOB), "shared"))
        .unwrap();
    // If update admission wins after the API's ownership read, the transactional backend
    // still refuses this operation without presenting it as a missing/older API method.
    let error = server
        .backend
        .control(
            VerifiedActor {
                public_key: SigningKey::from_bytes(&BOB).verifying_key().to_bytes(),
            },
            domain::MachineExecutionControl {
                execution: Some(domain::MachineExecutionQuery {
                    request_id: "shared".into(),
                    expected_execution_workspace_id: server.service.engine.workspace_id(),
                }),
                action: domain::MachineExecutionAction::Cancel as i32,
            },
        )
        .unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    assert_eq!(
        error.metadata().get("cozy-error-code").unwrap(),
        "update_control_unsupported"
    );
    let error = server
        .client
        .control(request(
            v1::ControlRequest {
                id: "shared".into(),
                action: v1::Action::Cancel as i32,
            },
            BOB,
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    assert_eq!(
        error.metadata().get("cozy-error-code").unwrap(),
        "update_control_unsupported"
    );
    assert_eq!(server.cancel("shared", ALICE).await, canceled);
    server.canceled("shared", None).await;
    server.stop().await;
    // The update payload journal remains machine-wide; its retained status must not shadow
    // another signer's canceled ordinary run with the same text ID.
    let status = cozy_machine::machine::update::Status {
        operation: "shared".into(),
        state: "succeeded".into(),
        history: vec![cozy_machine::machine::update::Step {
            state: "succeeded".into(),
            at_ms: 1,
        }],
        ..Default::default()
    };
    fs::write(
        root.0.join("state/update/status.json"),
        serde_json::to_vec(&status).unwrap(),
    )
    .unwrap();
    let mut restarted = Server::open(&root.0).await;
    let error = restarted
        .client
        .control(request(
            v1::ControlRequest {
                id: "shared".into(),
                action: v1::Action::Cancel as i32,
            },
            BOB,
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    assert_eq!(
        error.metadata().get("cozy-error-code").unwrap(),
        "update_control_unsupported"
    );
    assert_eq!(
        restarted.cancel("shared", ALICE).await.number,
        canceled.number
    );
    restarted.canceled("shared", None).await;
    let error = restarted
        .client
        .run(request(
            v1::RunRequest {
                id: "shared".into(),
                spec: None,
                after: 0,
            },
            BOB,
            Some("shared"),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::PermissionDenied);
    let mut update = restarted
        .client
        .run(request(
            v1::RunRequest {
                id: "shared".into(),
                spec: None,
                after: 0,
            },
            BOB,
            None,
        ))
        .await
        .unwrap()
        .into_inner();
    let mut observed = false;
    while let Some(event) = update.message().await.unwrap() {
        if let Some(v1::run_event::Event::Outcome(outcome)) = event.event {
            assert_eq!(outcome.status, "succeeded");
            observed = true;
        }
    }
    assert!(observed);
    restarted.stop().await;
}

#[tokio::test]
async fn a_busy_ownership_lookup_does_not_block_the_api_executor() {
    let root = Root::new();
    let mut server = Server::open(&root.0).await;
    let engine = server.service.engine.clone();
    let (held, locked) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        engine
            .with_journal(|_| {
                held.send(()).unwrap();
                // The watchdog releases only this fixture's lock if an old blocking handler stalls
                // the single API executor, so the failure cannot hang the test process.
                let _ = released.recv_timeout(Duration::from_secs(3));
                Ok(())
            })
            .unwrap()
    });
    locked.recv().unwrap();
    let mut blocked_client = server.client.clone();
    let lookup = tokio::spawn(async move {
        blocked_client
            .run(request(
                v1::RunRequest {
                    id: "absent".into(),
                    spec: None,
                    after: 0,
                },
                ALICE,
                None,
            ))
            .await
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    let responsive = tokio::time::timeout(
        Duration::from_millis(500),
        server.client.control(request(
            v1::ControlRequest {
                id: String::new(),
                action: v1::Action::Cancel as i32,
            },
            ALICE,
            None,
        )),
    )
    .await;
    let _ = release.send(());
    writer.join().unwrap();
    assert_eq!(
        responsive
            .expect("journal ownership I/O stalled the API executor")
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    let mut observed = lookup.await.unwrap().unwrap().into_inner();
    assert_eq!(observed.message().await.unwrap_err().code(), Code::NotFound);
    server.stop().await;
}

#[test]
fn update_and_cancel_admission_serialize_across_real_sqlite_connections() {
    let root = Root::new();
    let mut journal = Journal::open(&root.0).unwrap();
    journal.reserve_update_run("alice", "update-first").unwrap();
    assert_eq!(
        journal
            .reserve_run_cancellation("alice", "update-first")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    assert_eq!(
        journal
            .accept_run("alice", "update-first", "spec", Invocation::default())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    journal
        .reserve_run_cancellation("alice", "cancel-first")
        .unwrap();
    assert_eq!(
        journal
            .reserve_update_run("alice", "cancel-first")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    journal.reserve_update_run("bob", "cancel-first").unwrap();
    drop(journal);
    for n in 0..24 {
        let id = format!("update-race-{n}");
        let mut updater = Journal::open(&root.0).unwrap();
        let mut canceller = Journal::open(&root.0).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let start = barrier.clone();
        let update_id = id.clone();
        let update = std::thread::spawn(move || {
            start.wait();
            updater.reserve_update_run("alice", &update_id)
        });
        barrier.wait();
        let canceled = canceller.reserve_run_cancellation("alice", &id);
        let updated = update.join().unwrap();
        match (updated, canceled) {
            (Ok(()), Err(error)) => assert_eq!(error.kind(), std::io::ErrorKind::Unsupported),
            (Err(error), Ok(record)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
                assert!(record.canceled_before_acceptance && record.state == State::Canceled);
            }
            results => panic!("both admissions won or failed: {results:?}"),
        }
    }
    let mut restarted = Journal::open(&root.0).unwrap();
    assert!(restarted
        .update_run_reserved("alice", "update-first")
        .unwrap());
    assert!(restarted
        .update_run_reserved("bob", "cancel-first")
        .unwrap());
    assert!(
        restarted
            .get_public("alice", "cancel-first")
            .unwrap()
            .canceled_before_acceptance
    );
    assert_eq!(
        restarted
            .reserve_run_cancellation("alice", "update-first")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
}

#[tokio::test]
async fn absent_cancel_is_observable_scoped_repeatable_and_survives_restart() {
    let root = Root::new();
    let mut server = Server::open(&root.0).await;
    for action in [v1::Action::Pause, v1::Action::Resume] {
        let error = server
            .client
            .control(request(
                v1::ControlRequest {
                    id: "never-sent".into(),
                    action: action as i32,
                },
                ALICE,
                None,
            ))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::NotFound);
    }
    for id in ["".to_string(), "x".repeat(257)] {
        let error = server
            .client
            .control(request(
                v1::ControlRequest {
                    id,
                    action: v1::Action::Cancel as i32,
                },
                ALICE,
                None,
            ))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }
    let error = server
        .client
        .control(request(
            v1::ControlRequest {
                id: "never-sent".into(),
                action: v1::Action::Cancel as i32,
            },
            ALICE,
            Some("never-sent"),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::PermissionDenied);
    assert!(server
        .service
        .engine
        .get_public(&actor(ALICE), "never-sent")
        .is_err());
    let first = server.cancel("never-sent", ALICE).await;
    assert_eq!(first.state, "canceled");
    assert_eq!(server.cancel("never-sent", ALICE).await, first);
    assert!(server
        .service
        .engine
        .get_public(&actor(BOB), "never-sent")
        .is_err());
    let (bob, new) = server
        .service
        .engine
        .accept_run(&actor(BOB), "never-sent", "bob-spec", Invocation::default())
        .unwrap();
    assert!(new);
    assert_eq!(bob.state, State::Queued);
    server.canceled("never-sent", None).await;
    server.canceled("never-sent", Some(spec())).await;
    server.stop().await;
    let mut restarted = Server::open(&root.0).await;
    restarted.canceled("never-sent", None).await;
    restarted.canceled("never-sent", Some(spec())).await;
    assert_eq!(
        restarted.cancel("never-sent", ALICE).await.number,
        first.number
    );
    restarted.stop().await;
}

#[tokio::test]
async fn cancellation_wins_while_the_real_run_acceptance_waits_on_store_custody() {
    let root = Root::new();
    let mut server = Server::open(&root.0).await;
    let path = root.0.join("store/tmp/writers/recovery.lock");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    lock.lock_exclusive().unwrap();
    let mut submitted = server
        .client
        .run(request(
            v1::RunRequest {
                id: "blocked".into(),
                spec: Some(spec()),
                after: 0,
            },
            ALICE,
            None,
        ))
        .await
        .unwrap()
        .into_inner();
    assert!(server
        .service
        .engine
        .get_public(&actor(ALICE), "blocked")
        .is_err());
    assert_eq!(server.cancel("blocked", ALICE).await.state, "canceled");
    FileExt::unlock(&lock).unwrap();
    while let Some(event) = tokio::time::timeout(Duration::from_secs(5), submitted.message())
        .await
        .unwrap()
        .unwrap()
    {
        match event.event.unwrap() {
            v1::run_event::Event::State(state) => assert_eq!(state.state, "canceled"),
            v1::run_event::Event::Outcome(outcome) => assert_eq!(outcome.status, "canceled"),
            other => panic!("the blocked acceptance escaped cancellation: {other:?}"),
        }
    }
    let record = server
        .service
        .engine
        .get_public(&actor(ALICE), "blocked")
        .unwrap();
    assert!(record.canceled_before_acceptance);
    assert_eq!(record.attempt, 0);
    assert!(record.process.is_none() && record.invocation.generation.is_empty());
    assert!(server.service.engine.ready(10).unwrap().is_empty());
    server.stop().await;
}

#[test]
fn acceptance_and_cancel_have_one_durable_winner_and_preserve_completed_outcomes() {
    let root = Root::new();
    let engine = Engine::open(&root.0).unwrap();
    for n in 0..16 {
        let id = format!("race-{n}");
        let barrier = Arc::new(Barrier::new(2));
        let accepting = engine.clone();
        let start = barrier.clone();
        let sent = id.clone();
        let accept = std::thread::spawn(move || {
            start.wait();
            accepting
                .accept_run("alice", &sent, "spec", Invocation::default())
                .unwrap()
        });
        barrier.wait();
        let canceled = engine.cancel_run("alice", &id).unwrap();
        let (accepted, new) = accept.join().unwrap();
        assert_eq!(accepted.id, canceled.id);
        assert_eq!(
            engine.get_public("alice", &id).unwrap().state,
            State::Canceled
        );
        assert_eq!(new, !canceled.canceled_before_acceptance);
    }
    let record = engine
        .accept_run("alice", "completed", "authored-spec", Invocation::default())
        .unwrap()
        .0;
    engine
        .end_preparation(
            &record.id,
            Outcome::Completed(ResultRecord {
                value: serde_json::json!(42),
                artifacts: vec![],
                asset_bindings: vec![],
            }),
        )
        .unwrap();
    let canceled = engine.cancel_run("alice", "completed").unwrap();
    assert_eq!(canceled.state, State::Completed);
    assert_eq!(canceled.result.unwrap().value, serde_json::json!(42));
    assert!(!canceled.canceled_before_acceptance);
    drop(engine);
    let mut journal = Journal::open(&root.0).unwrap();
    let before = journal.reserve_run_cancellation("alice", "before").unwrap();
    drop(journal);
    let mut journal = Journal::open(&root.0).unwrap();
    let (after, new) = journal
        .accept_run("alice", "before", "late-spec", Invocation::default())
        .unwrap();
    assert!(!new && after.canceled_before_acceptance && after.state == State::Canceled);
    assert_eq!(before.id, after.id);
}
