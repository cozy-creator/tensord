//! A worker.v1 client meeting a machine that serves only cozy.machine.v1 is told to upgrade, in
//! words released CLI 0.1.26 prints (it reads UNIMPLEMENTED as a machine too old for it).
use cozy_machine::api::{pb, retired};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{service::Routes, transport::Server, Code};

#[tokio::test]
async fn a_worker_v1_client_is_told_to_upgrade() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let mut routes = Routes::default();
    *routes.axum_router_mut() = retired::worker_v1(routes.axum_router_mut().clone());
    tokio::spawn(
        Server::builder()
            .add_routes(routes)
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );

    // The first call of every CLI 0.1.26 verb.
    let mut host = pb::pod_host_client::PodHostClient::connect(address.clone())
        .await
        .unwrap();
    let refused = host
        .protocol_info(pb::ProtocolInfoRequest::default())
        .await
        .unwrap_err();
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::FailedPrecondition, retired::WORKER_V1)
    );

    // Its Claim stream.
    let mut control = pb::worker_control_client::WorkerControlClient::connect(address.clone())
        .await
        .unwrap();
    let refused = control
        .control(tokio_stream::iter([pb::RecordOwnerFrame::default()]))
        .await
        .unwrap_err();
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::FailedPrecondition, retired::WORKER_V1)
    );

    // A path nothing routes still answers UNIMPLEMENTED: the new CLI tells sides apart by it.
    let mut machine = cozy_machine::api::v1::machine_client::MachineClient::connect(address)
        .await
        .unwrap();
    let unrouted = machine
        .status(cozy_machine::api::v1::StatusRequest { keepalive: false })
        .await
        .unwrap_err();
    assert_eq!(unrouted.code(), Code::Unimplemented);
}
