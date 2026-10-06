//! The machine's real listener serves cozy.machine.v1 only. A worker.v1 client is told to
//! upgrade in words released CLI 0.1.26 prints (it reads UNIMPLEMENTED as a machine too old).
use cozy_machine::api::{self, domain, retired, v1, MachineIdentity};
use ed25519_dalek::SigningKey;
use std::sync::Arc;
use tonic::{
    transport::{Certificate, ClientTlsConfig, Endpoint},
    Code,
};

struct Nothing;
impl api::MachineBackend for Nothing {
    fn workspace(
        &self,
        _: api::auth::VerifiedActor,
        _: domain::MachineExecutionWorkspaceQuery,
    ) -> Result<domain::MachineExecutionWorkspace, tonic::Status> {
        Err(tonic::Status::unimplemented("no workspace"))
    }
}

#[tokio::test]
async fn a_worker_v1_client_is_told_to_upgrade() {
    let owner = SigningKey::from_bytes(&[31; 32]).verifying_key();
    let identity = MachineIdentity::ephemeral("retired".into(), vec![owner], vec![7; 32]).unwrap();
    let pem = identity.cert_pem.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        api::serve(listener, identity, Arc::new(Nothing))
            .await
            .unwrap()
    });
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

    // A rejected old method needs no generated old message or RPC binding.
    for path in [
        "/cozy.worker.v1.PodHost/ProtocolInfo",
        "/cozy.worker.v1.WorkerControl/Control",
    ] {
        let mut rejected = tonic::client::Grpc::new(channel.clone());
        rejected.ready().await.unwrap();
        let result: Result<tonic::Response<v1::StatusFrame>, tonic::Status> = rejected
            .unary(
                tonic::Request::new(v1::StatusRequest::default()),
                path.parse().unwrap(),
                tonic_prost::ProstCodec::default(),
            )
            .await;
        let error = result.unwrap_err();
        assert_eq!(
            (error.code(), error.message()),
            (Code::FailedPrecondition, retired::WORKER_V1)
        );
    }

    // cozy.machine.v1 answers on the same listener.
    let mut status = v1::machine_client::MachineClient::new(channel)
        .status(v1::StatusRequest { keepalive: false })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        status.message().await.unwrap().unwrap().worker_id,
        "retired"
    );
}
