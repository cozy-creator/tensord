//! The Hub's worker API as this machine: worker id and token headers, JSON, HTTPS to the
//! granted origin (public roots plus the granted CA). Two calls: rental authority and release.
use super::grant::HubGrant;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use bytes::Bytes;
use ed25519_dalek::VerifyingKey;
use http_body_util::{BodyExt, Full, Limited};
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use std::{
    io,
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio_rustls::rustls::{self, pki_types::ServerName, RootCertStore};

/// A bounded transport budget per call, never a limit on any work.
const CALL_BUDGET: Duration = Duration::from_secs(30);
const MAX_ANSWER_BYTES: usize = 64 << 10;

pub struct Hub {
    grant: HubGrant,
    host: String,
    port: u16,
    tls: Arc<rustls::ClientConfig>,
    /// When the Hub last answered anything (0: never in this process).
    contact_ms: AtomicI64,
}

#[derive(Debug)]
pub enum Refusal {
    /// 401/403: the Hub denies this attempt's authority.
    Denied,
    /// Unreachable, slow or any other answer: proves nothing either way.
    Transport(String),
}

impl Hub {
    pub fn new(grant: HubGrant) -> io::Result<Self> {
        let authority = grant.origin.trim_start_matches("https://");
        let (host, port) = match authority
            .rsplit_once(':')
            .map(|(h, p)| (h, p.parse::<u16>()))
        {
            Some((host, Ok(port))) => (host.to_owned(), port),
            _ => (authority.to_owned(), 443),
        };
        let mut roots = RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        if let Some(ca) = &grant.ca_der {
            roots.add(ca.clone().into()).map_err(io::Error::other)?;
        }
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let tls = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(io::Error::other)?
            .with_root_certificates(roots)
            .with_no_client_auth();
        Ok(Self {
            grant,
            host,
            port,
            tls: Arc::new(tls),
            contact_ms: AtomicI64::new(0),
        })
    }

    pub fn origin(&self) -> &str {
        &self.grant.origin
    }

    /// When the Hub last answered this machine, whatever it said; 0: never in this process.
    pub fn contact_ms(&self) -> i64 {
        self.contact_ms.load(Ordering::Acquire)
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<(StatusCode, Bytes), Refusal> {
        let exchange = async {
            let tcp = super::net::connect(&self.host, self.port).await?;
            let name = ServerName::try_from(self.host.clone()).map_err(io::Error::other)?;
            let tls = tokio_rustls::TlsConnector::from(self.tls.clone())
                .connect(name, tcp)
                .await?;
            let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tls))
                .await
                .map_err(io::Error::other)?;
            tokio::spawn(connection);
            let request = Request::builder()
                .method(method)
                .uri(path)
                .header("host", &self.host)
                .header("x-cozy-worker-id", &self.grant.worker_id)
                .header("x-cozy-worker-token", &self.grant.worker_token)
                .header("accept", "application/json")
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(body.unwrap_or_default())))
                .map_err(io::Error::other)?;
            let response = sender
                .send_request(request)
                .await
                .map_err(io::Error::other)?;
            let status = response.status();
            let answer = Limited::new(response.into_body(), MAX_ANSWER_BYTES)
                .collect()
                .await
                .map_err(io::Error::other)?
                .to_bytes();
            Ok::<_, io::Error>((status, answer))
        };
        let answered = tokio::time::timeout(CALL_BUDGET, exchange).await;
        if let Ok(Ok(_)) = &answered {
            self.contact_ms
                .store(super::lifecycle::now_ms(), Ordering::Release);
        }
        match answered {
            Ok(Ok((status, _)))
                if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN =>
            {
                Err(Refusal::Denied)
            }
            Ok(Ok(answer)) => Ok(answer),
            Ok(Err(error)) => Err(Refusal::Transport(format!("reach Tensorhub: {error}"))),
            Err(_) => Err(Refusal::Transport(
                "Tensorhub did not answer within its call budget".into(),
            )),
        }
    }

    /// The rental's current authorized keys, their lease and the owner's bindings revision (0
    /// from a Hub that states none).
    pub async fn authorized_keys(&self) -> Result<(Vec<VerifyingKey>, Duration, i64), Refusal> {
        let (status, answer) = self
            .call(Method::GET, "/v1/worker/rental/authorized-keys", None)
            .await?;
        if status != StatusCode::OK {
            return Err(Refusal::Transport(format!(
                "rental authority answered HTTP {status}"
            )));
        }
        #[derive(serde::Deserialize)]
        struct Document {
            worker_id: String,
            authorized_keys: Vec<String>,
            lease_seconds: u64,
            #[serde(default)]
            bindings_revision: i64,
        }
        let invalid = || Refusal::Transport("rental authority metadata is invalid".into());
        let document: Document = serde_json::from_slice(&answer).map_err(|_| invalid())?;
        if document.worker_id != self.grant.worker_id
            || !(1..=3600).contains(&document.lease_seconds)
        {
            return Err(invalid());
        }
        let keys = document
            .authorized_keys
            .iter()
            .map(|spelled| {
                let raw: [u8; 32] = URL_SAFE_NO_PAD.decode(spelled).ok()?.try_into().ok()?;
                VerifyingKey::from_bytes(&raw).ok()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(invalid)?;
        Ok((keys, Duration::from_secs(document.lease_seconds), document.bindings_revision))
    }

    /// Tells the Hub where this booting machine listens. Ok(false): a Hub that takes no
    /// reports and waits for the provider instead.
    pub async fn report_endpoint(&self, endpoint: &super::endpoint::Endpoint) -> Result<bool, Refusal> {
        let body = serde_json::to_vec(endpoint).expect("JSON");
        let (status, _) = self
            .call(Method::POST, "/v1/worker/rental/endpoint", Some(body))
            .await?;
        match status {
            StatusCode::NO_CONTENT => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            other => Err(Refusal::Transport(format!("the endpoint report answered HTTP {other}"))),
        }
    }

    /// Ends this rental's allocation; 204 is acceptance.
    pub async fn release(&self) -> Result<(), Refusal> {
        let body = serde_json::to_vec(&serde_json::json!({ "worker_id": self.grant.worker_id }))
            .expect("JSON");
        let (status, answer) = self
            .call(Method::POST, "/v1/worker/rental/release", Some(body))
            .await?;
        if status == StatusCode::NO_CONTENT {
            return Ok(());
        }
        #[derive(serde::Deserialize)]
        struct Refused {
            error: Detail,
        }
        #[derive(serde::Deserialize)]
        struct Detail {
            code: String,
            message: String,
        }
        Err(Refusal::Transport(
            match serde_json::from_slice::<Refused>(&answer) {
                Ok(refused) => format!(
                    "{}: {:.512} (HTTP {status})",
                    refused.error.code, refused.error.message
                ),
                Err(_) => format!("Tensorhub answered HTTP {status}"),
            },
        ))
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied => write!(f, "the Hub denied this rental's authority"),
            Self::Transport(detail) => write!(f, "{detail}"),
        }
    }
}
