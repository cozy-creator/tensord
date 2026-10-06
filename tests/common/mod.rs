//! What every readiness test asks a machine root's listener.
use tensord::api::v1::{machine_client::MachineClient, StatusRequest};
use std::path::Path;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

/// The sealed receipt Status carries with no capability, as the Hub reads it; None until the
/// listener answers and the boot is sealed. Runs on its own thread, so sync and async tests
/// both call it.
pub fn receipt(root: &Path, port: u16) -> Option<Vec<u8>> {
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).ok()?;
    let read = move || -> Option<Vec<u8>> {
        tokio::runtime::Runtime::new().ok()?.block_on(async {
            let tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("cozy-worker");
            let endpoint = Endpoint::from_shared(format!("https://127.0.0.1:{port}")).ok()?.tls_config(tls).ok()?;
            let mut frames = MachineClient::new(endpoint.connect().await.ok()?)
                .status(StatusRequest::default())
                .await
                .ok()?
                .into_inner();
            let frame = frames.message().await.ok()??;
            (!frame.receipt.is_empty()).then_some(frame.receipt)
        })
    };
    std::thread::spawn(read).join().ok()?
}
