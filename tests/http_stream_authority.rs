//! Actual TLS/HTTP output transport authority under backpressure; no inference qualification.
use cozy_machine::api::{
    self,
    auth::VerifiedActor,
    capability::{self, Grant},
    pb, MachineBackend, MachineIdentity,
};
use ed25519_dalek::SigningKey;
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
struct Output;
impl MachineBackend for Output {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Ok(pb::MachineExecutionWorkspace::default())
    }
    fn open_output(
        &self,
        _: VerifiedActor,
        _: u64,
        _: &str,
        _: Option<u32>,
    ) -> Result<api::backend::OutputSnapshot, tonic::Status> {
        let path = std::env::temp_dir().join(format!("cm-http-body-{}", uuid::Uuid::new_v4()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(path).unwrap();
        file.set_len(128 << 20).unwrap();
        Ok(api::backend::OutputSnapshot {
            parts: vec![(file, 128 << 20)],
            length: 128 << 20,
            rev: 1,
            media_type: "application/octet-stream".into(),
            sha256: None,
        })
    }
}
#[tokio::test]
async fn revoked_output_cap_stops_the_existing_http_body() {
    let signer = SigningKey::from_bytes(&[85; 32]);
    let identity = MachineIdentity::ephemeral(
        "http-output".into(),
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
    let config = rustls::ClientConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        api::serve(listener, identity, Arc::new(Output))
            .await
            .unwrap()
    });
    let tcp = tokio::net::TcpStream::connect(address).await.unwrap();
    let mut socket = tokio_rustls::TlsConnector::from(Arc::new(config))
        .connect(
            rustls::pki_types::ServerName::try_from("localhost").unwrap(),
            tcp,
        )
        .await
        .unwrap();
    let token = capability::mint(
        &signer,
        Grant {
            machine: "http-output".into(),
            run: "1".into(),
            expires: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
                + 60,
            ..Default::default()
        },
    );
    socket.write_all(format!("GET /v1/runs/1/outputs/image HTTP/1.1\r\nHost: localhost\r\nAuthorization: Cozy-Cap {token}\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
    let mut head = vec![];
    loop {
        head.push(socket.read_u8().await.unwrap());
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
        assert!(head.len() < 16 << 10);
    }
    assert!(std::str::from_utf8(&head)
        .unwrap()
        .starts_with("HTTP/1.1 200"));
    keys.revoke();
    let mut read = 0;
    let mut buffer = vec![0; 64 << 10];
    loop {
        match tokio::time::timeout(Duration::from_secs(3), socket.read(&mut buffer))
            .await
            .unwrap()
        {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                read += n;
                assert!(read < 64 << 20, "revoked output served its complete body");
            }
        }
    }
    assert!(read < 128 << 20);
    server.abort();
}
