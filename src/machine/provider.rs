//! A rental's own end through its provider, for when the Hub cannot hear its idle release: each
//! provider gives a pod a credential scoped to that pod (RunPod `RUNPOD_POD_ID` and
//! `RUNPOD_API_KEY`, vast.ai `CONTAINER_ID` and `CONTAINER_API_KEY`).
use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use std::{collections::HashMap, io, sync::Arc, time::Duration};
use tokio_rustls::rustls::{self, pki_types::ServerName, RootCertStore};

/// The provider API's location, when not the provider's own (a stand-in provider). Plain HTTP is
/// accepted for a loopback stand-in only.
pub const API_ORIGIN: &str = "COZY_PROVIDER_API_ORIGIN";
const CALL_BUDGET: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderSelf {
    RunPod {
        pod: String,
        key: String,
        origin: String,
    },
    Vast {
        instance: String,
        key: String,
        origin: String,
    },
}

impl ProviderSelf {
    /// The pod's own provider credential, when its provider gave one.
    pub fn from_env(env: &HashMap<String, String>) -> Option<Self> {
        let get = |name: &str| env.get(name).filter(|v| !v.is_empty()).cloned();
        let origin = |default: &str| get(API_ORIGIN).unwrap_or_else(|| default.to_owned());
        if let (Some(pod), Some(key)) = (get("RUNPOD_POD_ID"), get("RUNPOD_API_KEY")) {
            return Some(Self::RunPod {
                pod,
                key,
                origin: origin("https://api.runpod.io"),
            });
        }
        if let (Some(instance), Some(key)) = (get("CONTAINER_ID"), get("CONTAINER_API_KEY")) {
            return Some(Self::Vast {
                instance,
                key,
                origin: origin("https://console.vast.ai"),
            });
        }
        None
    }

    /// Ends this pod at its provider, so billing stops.
    pub async fn end(&self) -> io::Result<()> {
        let (origin, method, path, key, body) = match self {
            Self::RunPod { pod, key, origin } => {
                let query = format!("mutation {{ podTerminate(input: {{podId: {pod:?}}}) }}");
                let body = serde_json::to_vec(&serde_json::json!({ "query": query }))?;
                (origin, Method::POST, "/graphql".to_owned(), key, body)
            }
            Self::Vast {
                instance,
                key,
                origin,
            } => (
                origin,
                Method::DELETE,
                format!("/api/v0/instances/{instance}/"),
                key,
                vec![],
            ),
        };
        let (status, answer) =
            tokio::time::timeout(CALL_BUDGET, call(origin, method, &path, key, body))
                .await
                .map_err(|_| {
                    io::Error::other("the provider did not answer within its call budget")
                })??;
        // RunPod answers GraphQL errors with HTTP 200 and an `errors` list.
        let refused = serde_json::from_slice::<serde_json::Value>(&answer)
            .ok()
            .and_then(|v| v.get("errors").cloned())
            .filter(|e| !e.is_null());
        match (status.is_success(), refused) {
            (true, None) => Ok(()),
            (_, Some(errors)) => Err(io::Error::other(format!(
                "the provider refused: {errors:.512}"
            ))),
            (false, None) => Err(io::Error::other(format!(
                "the provider answered HTTP {status}"
            ))),
        }
    }
}

async fn call(
    origin: &str,
    method: Method,
    path: &str,
    key: &str,
    body: Vec<u8>,
) -> io::Result<(StatusCode, Bytes)> {
    let (tls, authority) = match (
        origin.strip_prefix("https://"),
        origin.strip_prefix("http://"),
    ) {
        (Some(authority), _) => (true, authority),
        (None, Some(authority)) if authority.starts_with("127.0.0.1:") => (false, authority),
        _ => {
            return Err(io::Error::other(format!(
                "{API_ORIGIN} must be https, or http on 127.0.0.1"
            )))
        }
    };
    let (host, port) = match authority
        .rsplit_once(':')
        .map(|(h, p)| (h, p.parse::<u16>()))
    {
        Some((host, Ok(port))) => (host.to_owned(), port),
        _ => (authority.to_owned(), 443),
    };
    let tcp = super::net::connect(&host, port).await?;
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("host", &host)
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("user-agent", "cozy-machine")
        .body(Full::new(Bytes::from(body)))
        .map_err(io::Error::other)?;
    let response = if tls {
        let roots = RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        let config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(io::Error::other)?
        .with_root_certificates(roots)
        .with_no_client_auth();
        let name = ServerName::try_from(host.clone()).map_err(io::Error::other)?;
        let stream = tokio_rustls::TlsConnector::from(Arc::new(config))
            .connect(name, tcp)
            .await?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(io::Error::other)?;
        tokio::spawn(connection);
        sender
            .send_request(request)
            .await
            .map_err(io::Error::other)?
    } else {
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tcp))
            .await
            .map_err(io::Error::other)?;
        tokio::spawn(connection);
        sender
            .send_request(request)
            .await
            .map_err(io::Error::other)?
    };
    let status = response.status();
    let answer = Limited::new(response.into_body(), 64 << 10)
        .collect()
        .await
        .map_err(io::Error::other)?
        .to_bytes();
    Ok((status, answer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pod_knows_its_provider_credential() {
        let env = |pairs: &[(&str, &str)]| {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        assert_eq!(
            ProviderSelf::from_env(&env(&[("RUNPOD_POD_ID", "p1"), ("RUNPOD_API_KEY", "k")])),
            Some(ProviderSelf::RunPod {
                pod: "p1".into(),
                key: "k".into(),
                origin: "https://api.runpod.io".into()
            })
        );
        assert_eq!(
            ProviderSelf::from_env(&env(&[
                ("CONTAINER_ID", "9"),
                ("CONTAINER_API_KEY", "k"),
                (API_ORIGIN, "http://127.0.0.1:9")
            ])),
            Some(ProviderSelf::Vast {
                instance: "9".into(),
                key: "k".into(),
                origin: "http://127.0.0.1:9".into()
            })
        );
        assert_eq!(
            ProviderSelf::from_env(&env(&[("RUNPOD_POD_ID", "p1")])),
            None
        );
    }
}
