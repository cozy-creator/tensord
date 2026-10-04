//! The two listener observations the Hub's readiness reader requires, made against the real
//! listener over loopback: it presents exactly this machine's leaf, and it refuses a foreign
//! credential on `cozy.machine.v1`.
use crate::api::{capability, v1};
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

/// True when the listener refuses a `cozy.machine.v1` call whose Cozy-Cap is well formed,
/// names this machine and is unexpired, but is signed by a key made for this one probe.
pub async fn refuses_foreign_capability(port: u16, cert_pem: &str, worker_id: &str) -> bool {
    match super::identity::random::<32>() {
        Ok(seed) => {
            let foreign = ed25519_dalek::SigningKey::from_bytes(&seed);
            refuses_capability_of(port, cert_pem, worker_id, &foreign).await
        }
        Err(_) => false,
    }
}

/// True when `Status` carrying a machine-scope Cozy-Cap signed by `key` answers
/// UNAUTHENTICATED while `Status` with no capability answers: the refusal is the credential's,
/// not a closed door.
pub async fn refuses_capability_of(
    port: u16,
    cert_pem: &str,
    worker_id: &str,
    key: &ed25519_dalek::SigningKey,
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
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() as i64);
    let grant = capability::Grant {
        machine: worker_id.into(),
        action: capability::MACHINE.into(),
        expires: now + 60,
        ..Default::default()
    };
    let Ok(authorization) = format!("Cozy-Cap {}", capability::mint(key, grant)).parse() else {
        return false;
    };
    let mut client = v1::machine_client::MachineClient::new(channel);
    let open = client.status(v1::StatusRequest { keepalive: false }).await;
    let mut request = tonic::Request::new(v1::StatusRequest { keepalive: false });
    request
        .metadata_mut()
        .insert("authorization", authorization);
    let refused = client.status(request).await;
    open.is_ok() && refused.is_err_and(|status| status.code() == tonic::Code::Unauthenticated)
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
