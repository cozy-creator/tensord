//! Real TLS owner maintenance authority on CPU; no GPU reclamation is inferred.
use cozy_machine::{
    api::{
        self,
        auth::VerifiedActor,
        capability::{self, Grant},
        pb, MachineBackend, MachineIdentity,
    },
    execution::Engine,
    gpu_service::IdleReclaim,
    journal::{Invocation, State},
};
use ed25519_dalek::SigningKey;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
struct Backend {
    calls: AtomicUsize,
    busy: AtomicBool,
}
impl MachineBackend for Backend {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Ok(Default::default())
    }
    fn reclaim_idle_memory(&self, _: VerifiedActor, _: i64) -> Result<IdleReclaim, tonic::Status> {
        if self.busy.load(Ordering::Acquire) {
            return Err(tonic::Status::failed_precondition("memory_reclaim_busy"));
        }
        self.calls.fetch_add(1, Ordering::AcqRel);
        Ok(Default::default())
    }
    fn control(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionControl,
    ) -> Result<pb::MachineExecutionState, tonic::Status> {
        panic!("maintenance must not control accepted work")
    }
}
async fn post(
    address: std::net::SocketAddr,
    config: Arc<rustls::ClientConfig>,
    token: &str,
) -> u16 {
    let tcp = tokio::net::TcpStream::connect(address).await.unwrap();
    let mut socket = tokio_rustls::TlsConnector::from(config)
        .connect(
            rustls::pki_types::ServerName::try_from("localhost").unwrap(),
            tcp,
        )
        .await
        .unwrap();
    socket.write_all(format!("POST /v1/machine/memory/reclaim HTTP/1.1\r\nHost: localhost\r\nAuthorization: Cozy-Cap {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
    let mut body = vec![];
    socket.take(65536).read_to_end(&mut body).await.unwrap();
    String::from_utf8(body)
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}
#[tokio::test]
async fn owner_http_reclaim_rejects_scoped_expired_revoked_and_busy_without_canceling_work() {
    let signer = SigningKey::from_bytes(&[87; 32]);
    let identity = MachineIdentity::ephemeral(
        "reclaim-owner".into(),
        vec![signer.verifying_key()],
        vec![7; 32],
    )
    .unwrap();
    let keys = identity.authority.keys.clone();
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(rustls::pki_types::CertificateDer::from(
            identity.cert_der.clone(),
        ))
        .unwrap();
    let config = Arc::new(
        rustls::ClientConfig::builder_with_provider(
            rustls::crypto::ring::default_provider().into(),
        )
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth(),
    );
    let root = std::env::temp_dir().join(format!("cm-http-reclaim-{}", uuid::Uuid::new_v4()));
    let engine = Engine::open(&root).unwrap();
    let run = engine
        .submit(
            "accepted",
            Invocation {
                package: "cpu".into(),
                ..Default::default()
            },
        )
        .unwrap();
    let backend = Arc::new(Backend {
        calls: AtomicUsize::new(0),
        busy: AtomicBool::new(false),
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let served = backend.clone();
    let server = tokio::spawn(async move { api::serve(listener, identity, served).await.unwrap() });
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let owner = Grant {
        machine: "reclaim-owner".into(),
        action: capability::MACHINE.into(),
        expires: now + 60,
        ..Default::default()
    };
    for denied in [
        Grant {
            action: String::new(),
            run: "1".into(),
            ..owner.clone()
        },
        Grant {
            expires: now - 1,
            ..owner.clone()
        },
        Grant {
            action: "runtime-update".into(),
            ..owner.clone()
        },
        Grant {
            outputs: vec!["video".into()],
            ..owner.clone()
        },
    ] {
        assert_eq!(
            post(address, config.clone(), &capability::mint(&signer, denied)).await,
            403
        );
    }
    assert_eq!(post(address, config.clone(), "invalid").await, 403);
    assert_eq!(backend.calls.load(Ordering::Acquire), 0);
    let token = capability::mint(&signer, owner);
    backend.busy.store(true, Ordering::Release);
    assert_eq!(post(address, config.clone(), &token).await, 409);
    backend.busy.store(false, Ordering::Release);
    assert_eq!(post(address, config.clone(), &token).await, 200);
    keys.revoke();
    assert_eq!(post(address, config.clone(), &token).await, 403);
    assert_eq!(backend.calls.load(Ordering::Acquire), 1);
    let after = engine.get(&run.id).unwrap();
    assert_eq!(after.state, State::Queued);
    assert!(after.cancel_actor.is_none());
    server.abort();
    drop(engine);
    std::fs::remove_dir_all(root).unwrap();
}
