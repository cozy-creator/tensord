//! A key that leaves the Hub lease ends the open streams it authorized, at once, through the
//! real server; accepted work is not touched (none is involved here).
use cozy_machine::api::{self, pb, MachineIdentity};
use ed25519_dalek::{Signer, SigningKey};
use std::{sync::Arc, time::Duration};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

struct Nothing;
impl api::MachineBackend for Nothing {
    fn workspace(
        &self,
        _: api::auth::VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, tonic::Status> {
        Err(tonic::Status::unimplemented("no workspace"))
    }
}

#[tokio::test]
async fn a_revoked_key_ends_its_control_stream() {
    let (owner, other) = (SigningKey::from_bytes(&[31; 32]), SigningKey::from_bytes(&[32; 32]));
    let identity = MachineIdentity::ephemeral("revocation-probe".into(), vec![owner.verifying_key()], vec![7; 32]).unwrap();
    let keys = identity.authority.keys.clone();
    let claim = pb::Claim {
        worker_id: identity.authority.worker_id.clone(),
        worker_boot_id: identity.authority.boot_id.clone(),
        record_owner_epoch: 1,
        proof: owner.sign(&identity.authority.transcript(1).unwrap()).to_bytes().to_vec(),
        ..Default::default()
    };
    let pem = identity.cert_pem.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { api::serve(listener, identity, Arc::new(Nothing)).await.unwrap() });
    let channel = Endpoint::from_shared(format!("https://{address}"))
        .unwrap()
        .tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("localhost"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = pb::worker_control_client::WorkerControlClient::new(channel);
    let (frames, requests) = tokio::sync::mpsc::channel(1);
    frames.send(pb::RecordOwnerFrame { msg: Some(pb::record_owner_frame::Msg::Claim(claim)) }).await.unwrap();
    let mut answers = client.control(tokio_stream::wrappers::ReceiverStream::new(requests)).await.unwrap().into_inner();
    match answers.message().await.unwrap().and_then(|f| f.msg) {
        Some(pb::worker_frame::Msg::ClaimAck(ack)) => assert!(ack.accepted),
        other => panic!("no accepted ClaimAck: {other:?}"),
    }
    // The stream stays open while its key authorizes; the Hub lease then names another key.
    assert!(tokio::time::timeout(Duration::from_millis(300), answers.message()).await.is_err());
    keys.renew(
        vec![api::auth::Holder::own(other.verifying_key())],
        Duration::from_secs(60),
    );
    let ended = tokio::time::timeout(Duration::from_secs(5), answers.message()).await.expect("the stream was not ended");
    assert_eq!(ended.unwrap_err().code(), tonic::Code::Unauthenticated);
    drop(frames);
}
