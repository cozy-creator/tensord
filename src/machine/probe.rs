//! The two listener observations the Hub's readiness reader requires, made against the real
//! listener over loopback: it presents exactly this machine's leaf, and it refuses a foreign Claim.
use crate::api::pb;
use std::sync::Arc;
use tokio_rustls::rustls::{
    self,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider},
    pki_types::{CertificateDer, ServerName, UnixTime},
    DigitallySignedStruct, SignatureScheme,
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

/// True when a TLS handshake on the listener presents exactly `leaf`.
pub async fn presents_leaf(port: u16, leaf: &[u8]) -> bool {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let Ok(config) = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
    else {
        return false;
    };
    let config = config
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pinned(leaf.to_vec(), provider)))
        .with_no_client_auth();
    let Ok(stream) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
        return false;
    };
    let name = ServerName::try_from(super::identity::SERVER_NAME).expect("a DNS name");
    tokio_rustls::TlsConnector::from(Arc::new(config))
        .connect(name, stream)
        .await
        .is_ok()
}

/// True when the listener answers a current-schema Claim with a foreign proof by
/// `ClaimAck { accepted: false, rejection: UNAUTHENTICATED }`.
pub async fn refuses_foreign_claim(
    port: u16,
    cert_pem: &str,
    worker_id: &str,
    boot_id: &str,
) -> bool {
    let tls = ClientTlsConfig::new()
        .ca_certificate(Certificate::from_pem(cert_pem))
        .domain_name(super::identity::SERVER_NAME);
    let Ok(endpoint) =
        Endpoint::from_shared(format!("https://127.0.0.1:{port}")).and_then(|e| e.tls_config(tls))
    else {
        return false;
    };
    let Ok(channel) = endpoint.connect().await else {
        return false;
    };
    let claim = pb::Claim {
        record_owner_epoch: 1,
        record_owner_id: "readiness-negative-arm".into(),
        worker_id: worker_id.into(),
        worker_boot_id: boot_id.into(),
        wire_minor: crate::api::WIRE_MINOR,
        proof: [0x5a; 64].to_vec(),
        ..Default::default()
    };
    let frame = pb::RecordOwnerFrame {
        msg: Some(pb::record_owner_frame::Msg::Claim(claim)),
    };
    let mut client = pb::worker_control_client::WorkerControlClient::new(channel);
    let Ok(response) = client.control(tokio_stream::iter([frame])).await else {
        return false;
    };
    match response.into_inner().message().await {
        Ok(Some(pb::WorkerFrame {
            msg: Some(pb::worker_frame::Msg::ClaimAck(ack)),
        })) => !ack.accepted && ack.rejection == pb::ClaimRejection::Unauthenticated as i32,
        _ => false,
    }
}

#[derive(Debug)]
struct Pinned(Vec<u8>, Arc<CryptoProvider>);
impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        if end_entity.as_ref() == self.0 {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(
                "the listener presented another leaf".into(),
            ))
        }
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.1.signature_verification_algorithms,
        )
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.1.signature_verification_algorithms,
        )
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.1.signature_verification_algorithms.supported_schemes()
    }
}
