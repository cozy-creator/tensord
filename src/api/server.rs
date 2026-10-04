use super::{auth::Authority, backend::MachineBackend};
use crate::machine::receipt::{self, Readiness};
use axum::{
    http::{HeaderMap, StatusCode},
    routing::get,
};
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_stream::{wrappers::TcpListenerStream, Stream};
use tonic::transport::{Identity, Server, ServerTlsConfig};

pub struct MachineIdentity {
    pub authority: Authority,
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
    pub readiness: Arc<Readiness>,
    pub started_at_ms: u64,
    /// A rental's idle ledger; None on development front doors.
    pub lifecycle: Option<Arc<crate::machine::lifecycle::Lifecycle>>,
    /// The Hubs this machine is registered with: (origin, worker id there).
    pub hubs: Vec<(String, String)>,
    /// `runtime-update/1`; None on development front doors.
    pub updates: Option<Arc<crate::machine::update::Updates>>,
    /// A granted media port: the CLI's machine launcher reads the receipt there.
    pub media: Option<std::net::TcpListener>,
    /// The WebRTC listener: `cozy/1` for browsers (ICE-TCP).
    pub webrtc: Option<std::net::TcpListener>,
    /// Where browsers reach it; Status reports it with the port `serve` bound.
    pub player: crate::machine::player::Reach,
    pub(crate) webrtc_port: Option<u16>,
    /// The store's directory; Status reports its filesystem. None on development front doors.
    pub store: Option<std::path::PathBuf>,
}

impl MachineIdentity {
    pub fn ephemeral(
        worker_id: String,
        keys: Vec<ed25519_dalek::VerifyingKey>,
        receipt_key: Vec<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if keys.is_empty() || receipt_key.len() < 32 {
            return Err("authorized owner keys and a >=32-byte readiness key are required".into());
        }
        let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/uuid")?
            .trim()
            .to_owned();
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["cozy.worker".into(), "localhost".into()])?;
        let cert_der = cert.der().to_vec();
        let authority = Authority {
            worker_id,
            boot_id,
            leaf_digest: tensorfs_core::sha256::digest(&cert_der),
            keys: keys.into(),
        };
        authority.transcript(1)?;
        Ok(Self {
            authority,
            cert_pem: cert.pem(),
            key_pem: signing_key.serialize_pem(),
            cert_der,
            readiness: Readiness::open(None, Some(receipt_key), false)?,
            started_at_ms: now_ms(),
            lifecycle: None,
            hubs: vec![],
            updates: None,
            media: None,
            store: None,
            webrtc: None,
            player: Default::default(),
            webrtc_port: None,
        })
    }
}

pub async fn serve<B: MachineBackend>(
    listener: tokio::net::TcpListener,
    mut identity: MachineIdentity,
    backend: Arc<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let port = listener.local_addr()?.port();
    let readiness = identity.readiness.clone();
    if let Some(media) = identity.media.take() {
        media.set_nonblocking(true)?;
        let media = tokio::net::TcpListener::from_std(media)?;
        let tls = ServerTlsConfig::new()
            .identity(Identity::from_pem(&identity.cert_pem, &identity.key_pem));
        let readiness = readiness.clone();
        let mut routes = tonic::service::Routes::default();
        *routes.axum_router_mut() = axum::Router::new().route(
            "/v1/bootstrap/receipt",
            get(move || {
                let envelope = readiness.envelope();
                async move {
                    match envelope {
                        Some(body) => (StatusCode::OK, body),
                        None => (StatusCode::SERVICE_UNAVAILABLE, vec![]),
                    }
                }
            }),
        );
        let server = Server::builder()
            .accept_http1(true)
            .tls_config(tls)?
            .add_routes(routes)
            .serve_with_incoming(no_delay(media));
        tokio::spawn(async move {
            if let Err(error) = server.await {
                eprintln!("cozy-machine: the receipt listener stopped: {error}");
            }
        });
    }
    let webrtc = match identity.webrtc.take() {
        Some(listener) => {
            listener.set_nonblocking(true)?;
            Some(tokio::net::TcpListener::from_std(listener)?)
        }
        None => None,
    };
    identity.webrtc_port = match &webrtc {
        Some(listener) => Some(listener.local_addr()?.port()),
        None => None,
    };
    let mut measured = MeasuredIdentity::of(&identity);
    measured.webrtc_port = identity.webrtc_port;
    tokio::spawn(prove_readiness(port, readiness.clone(), measured));
    let tls =
        ServerTlsConfig::new().identity(Identity::from_pem(&identity.cert_pem, &identity.key_pem));
    let identity = Arc::new(identity);
    if let Some(listener) = webrtc {
        let media = super::cozy1::Media::new(identity.clone(), backend.clone())?;
        tokio::spawn(super::cozy1::serve(media, listener));
    }
    let machine = super::machine_v1::MachineV1 {
        identity: identity.clone(),
        backend,
    };
    let mut routes = tonic::service::Routes::new(
        super::v1::machine_server::MachineServer::new(machine)
            .max_decoding_message_size(16 << 20)
            .max_encoding_message_size(16 << 20),
    );
    // A client that still speaks cozy.worker.v1 is told to upgrade, not that this machine is old.
    *routes.axum_router_mut() = super::retired::worker_v1(routes.axum_router_mut().clone())
        .route(
            "/v1/health",
            get(|headers: HeaderMap| async move {
                if headers.contains_key("authorization") {
                    StatusCode::UNAUTHORIZED
                } else {
                    StatusCode::NO_CONTENT
                }
            }),
        )
        .route(
            "/v1/bootstrap/receipt",
            get(move || {
                let envelope = readiness.envelope();
                async move {
                    let headers = [
                        ("content-type", "application/json"),
                        ("cache-control", "no-store"),
                    ];
                    match envelope {
                        Some(body) => (StatusCode::OK, headers, body),
                        // The Hub probes again on 503 until this boot has proved itself.
                        None => (StatusCode::SERVICE_UNAVAILABLE, headers, vec![]),
                    }
                }
            }),
        );
    Server::builder()
        .accept_http1(true)
        // Fixed receive windows (Write's uploads): BDP probing from 64 KiB lost 16% to a fixed
        // window on a lossy 160 ms link (G/read-bench); 16 MiB covers 150 Mbit/s at 800 ms.
        // A ping every 20 s keeps NAT mappings on a client's path alive through a quiet run, and
        // one left unanswered for 20 s ends a dead connection (its runs go on; a Run attaches).
        .http2_keepalive_interval(Some(std::time::Duration::from_secs(20)))
        .http2_keepalive_timeout(Some(std::time::Duration::from_secs(20)))
        .initial_stream_window_size(Some(16 << 20))
        .initial_connection_window_size(Some(32 << 20))
        .tls_config(tls)?
        .add_routes(routes)
        .serve_with_incoming(no_delay(listener))
        .await?;
    Ok(())
}

struct MeasuredIdentity {
    worker_id: String,
    boot_id: String,
    cert_pem: String,
    cert_der: Vec<u8>,
    webrtc_port: Option<u16>,
    capabilities: Vec<String>,
}
impl MeasuredIdentity {
    fn of(identity: &MachineIdentity) -> Self {
        Self {
            worker_id: identity.authority.worker_id.clone(),
            boot_id: identity.authority.boot_id.clone(),
            cert_pem: identity.cert_pem.clone(),
            cert_der: identity.cert_der.clone(),
            webrtc_port: None,
            capabilities: super::machine_status::capabilities(identity),
        }
    }
}

/// Observes this listener as the Hub requires and seals the result once. A retained envelope
/// of this boot is kept as is; a measurement that contradicts it is reported, never signed.
async fn prove_readiness(port: u16, readiness: Arc<Readiness>, id: MeasuredIdentity) {
    let listener_bound = crate::machine::probe::presents_leaf(port, &id.cert_der).await;
    let foreign_credential_refused =
        crate::machine::probe::refuses_foreign_capability(port, &id.cert_pem, &id.worker_id).await;
    let gpus = match readiness.measures_gpus() {
        false => Ok(vec![]),
        true => tokio::task::spawn_blocking(receipt::gpus)
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(e))),
    };
    let result = gpus.and_then(|gpus| {
        readiness.seal(
            receipt::Measured {
                boot_id: &id.boot_id,
                worker_port: port,
                cert_der: &id.cert_der,
                gpus,
                listener_bound,
                foreign_credential_refused,
                capabilities: id.capabilities,
                webrtc_port: id.webrtc_port,
            }
            .payload(),
        )
    });
    match result {
        Ok(true) => eprintln!("cozy-machine: readiness sealed (listener {listener_bound}, foreign capability refused {foreign_credential_refused})"),
        Ok(false) => eprintln!("cozy-machine: this boot's retained readiness still holds"),
        Err(error) => eprintln!("cozy-machine: readiness not proved: {error}"),
    }
}

/// Accepted connections with Nagle off, as Go's net package sets every TCP connection: a
/// response's small trailing frames otherwise wait for the ACK of the frames before them, one
/// round trip each on a far link.
fn no_delay(
    listener: tokio::net::TcpListener,
) -> impl Stream<Item = std::io::Result<tokio::net::TcpStream>> {
    use tokio_stream::StreamExt;
    TcpListenerStream::new(listener).map(|accepted| {
        accepted.inspect(|stream| {
            let _ = stream.set_nodelay(true);
        })
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
