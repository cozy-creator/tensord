//! A rental's lifecycle: the idle ledger (shared with the Go agent's `idle.json`), explicit
//! keepalive, the Hub lease of authorized keys and the idle release that ends billing.
use super::hub::{Hub, Refusal};
use crate::api::auth::Keys;
use std::{
    collections::BTreeMap,
    io,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tonic::Status;

/// The fixed rental idle window, `RentalIdleTimeoutSeconds` in the worker protocol and CLI.
pub const IDLE_GRACE_MS: i64 = 900_000;

#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct IdleState {
    deadline_ms: i64,
    released: bool,
    work_observed: bool,
    unknown: bool,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    keepalives: BTreeMap<String, [i64; 2]>,
}

pub struct Lifecycle {
    rental: bool,
    path: PathBuf,
    state: Mutex<(IdleState, i64)>, // the ledger and the in-memory renewal from observed work
    admissions: AtomicUsize,
}

/// Holds idle release while a call that may start work is in flight.
pub struct Admission(Arc<Lifecycle>);
impl Drop for Admission {
    fn drop(&mut self) {
        self.0.admissions.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl Lifecycle {
    /// `fresh` begins this root's first idle window; a restart keeps the persisted ledger.
    pub fn open(path: PathBuf, rental: bool, fresh: bool) -> io::Result<Arc<Self>> {
        let mut state = match std::fs::read(&path) {
            Ok(raw) if !fresh && rental => serde_json::from_slice(&raw)
                .map_err(|e| io::Error::other(format!("the idle ledger is unreadable: {e}")))?,
            Ok(_) => IdleState::default(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => IdleState::default(),
            Err(e) => return Err(e),
        };
        // A persistent machine begins a new idle session each start; it never releases.
        if state.deadline_ms == 0 && !state.released {
            state.deadline_ms = now_ms() + IDLE_GRACE_MS;
        }
        let lifecycle = Arc::new(Self {
            rental,
            path,
            state: Mutex::new((state, 0)),
            admissions: AtomicUsize::new(0),
        });
        lifecycle.save(&lifecycle.state.lock().unwrap().0)?;
        Ok(lifecycle)
    }

    fn save(&self, state: &IdleState) -> io::Result<()> {
        super::identity::write_atomic(&self.path, &serde_json::to_vec(state)?, 0o600)
    }

    fn due(state: &(IdleState, i64), now: i64) -> bool {
        state.0.released || now >= state.0.deadline_ms.max(state.1)
    }

    pub fn admit(self: &Arc<Self>) -> Result<Admission, Status> {
        let state = self.state.lock().unwrap();
        if self.rental && Self::due(&state, now_ms()) {
            return Err(Status::unavailable(
                "machine_released: this rental released itself after its idle deadline",
            ));
        }
        self.admissions.fetch_add(1, Ordering::AcqRel);
        Ok(Admission(self.clone()))
    }

    /// Accepted work renews the deadline.
    pub fn work(&self) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        state.0.deadline_ms = state.0.deadline_ms.max(now_ms() + IDLE_GRACE_MS);
        state.0.work_observed = true;
        self.save(&state.0)
    }

    /// Folds one activity sample in. Busy (or unreadable) work holds the deadline; the
    /// ledger is rewritten only when the persisted deadline falls half a window behind.
    pub fn observe(&self, busy: bool) -> io::Result<i64> {
        let now = now_ms();
        let mut state = self.state.lock().unwrap();
        if state.0.released {
            return Ok(now);
        }
        if busy {
            state.1 = state.1.max(now + IDLE_GRACE_MS);
        }
        let target = state.0.deadline_ms.max(state.1);
        if target - state.0.deadline_ms >= IDLE_GRACE_MS / 2 || busy && !state.0.work_observed {
            state.0.deadline_ms = target;
            state.0.work_observed |= busy;
            self.save(&state.0)?;
        }
        Ok(target)
    }

    /// An explicit owner keepalive, idempotent per request id (worker.v1).
    pub fn keepalive(&self, id: &str) -> Result<(i64, i64), Status> {
        self.reset(Some(id))
    }

    /// `Status{keepalive}`: one explicit reset of the idle deadline; answers the new deadline.
    pub fn renew(&self) -> Result<i64, Status> {
        self.reset(None).map(|(_, deadline)| deadline)
    }

    fn reset(&self, id: Option<&str>) -> Result<(i64, i64), Status> {
        let now = now_ms();
        let mut state = self.state.lock().unwrap();
        if self.rental && Self::due(&state, now) {
            return Err(Status::failed_precondition(
                "this machine's idle release is already due or committed",
            ));
        }
        if let Some(seen) = id.and_then(|id| state.0.keepalives.get(id)) {
            return Ok((seen[0], seen[1]));
        }
        if state.0.keepalives.len() >= 256 {
            state.0.keepalives.clear();
        }
        let deadline = now + IDLE_GRACE_MS;
        if let Some(id) = id {
            state.0.keepalives.insert(id.to_owned(), [now, deadline]);
        }
        state.0.deadline_ms = state.0.deadline_ms.max(deadline);
        self.save(&state.0).map_err(|e| {
            Status::unavailable(format!("the idle ledger could not be written: {e}"))
        })?;
        Ok((now, deadline))
    }

    /// Calls that may start work, in flight now.
    pub fn admitted(&self) -> usize {
        self.admissions.load(Ordering::Acquire)
    }

    pub fn deadline_ms(&self) -> i64 {
        let state = self.state.lock().unwrap();
        state.0.deadline_ms.max(state.1)
    }

    pub fn released(&self) -> bool {
        self.state.lock().unwrap().0.released
    }

    /// Takes the release once the deadline passed and nothing holds it. Irreversible.
    fn claim(&self) -> io::Result<bool> {
        let mut state = self.state.lock().unwrap();
        if !self.rental {
            return Ok(false);
        }
        if state.0.released {
            return Ok(true);
        }
        if self.admissions.load(Ordering::Acquire) > 0 || !Self::due(&state, now_ms()) {
            return Ok(false);
        }
        state.0.released = true;
        self.save(&state.0)?;
        Ok(true)
    }
}

/// Keeps the Hub lease of authorized keys: renewed at half its length, retried each second
/// while the Hub is unreachable or not yet ready.
pub async fn keep_authority(hub: Arc<Hub>, keys: Keys) {
    loop {
        let delay = match hub.authorized_keys().await {
            Ok((current, lease)) => {
                keys.renew(current, lease);
                lease / 2
            }
            Err(Refusal::Denied) => {
                eprintln!(
                    "cozy-machine: the Hub denied rental authority; no new control is admitted"
                );
                keys.revoke();
                Duration::from_secs(1)
            }
            Err(Refusal::Transport(_)) => Duration::from_secs(1),
        };
        tokio::time::sleep(delay).await;
    }
}

/// Watches activity and releases the rental once its idle deadline passed with nothing held.
/// Returns when the Hub accepted the release.
pub async fn release_when_idle(lifecycle: Arc<Lifecycle>, hub: Arc<Hub>, busy: impl Fn() -> bool) {
    let mut next_ask = 0;
    loop {
        if let Err(error) = lifecycle.observe(busy()) {
            eprintln!("cozy-machine: idle ledger: {error}");
        }
        let now = now_ms();
        if now >= lifecycle.deadline_ms() && now >= next_ask {
            match lifecycle.claim() {
                Ok(true) => {
                    match hub.release().await {
                        Ok(()) => {
                            eprintln!("cozy-machine: idle deadline passed; Tensorhub accepted the release");
                            return;
                        }
                        Err(error) => {
                            eprintln!("cozy-machine: idle release not accepted; retrying without extending the deadline: {error}");
                            next_ask = now + 5_000;
                        }
                    }
                }
                Ok(false) => (),
                Err(error) => {
                    eprintln!("cozy-machine: idle release could not be recorded: {error}")
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keepalive_is_idempotent_work_renews_and_release_is_irreversible() {
        let dir = std::env::temp_dir().join(format!("cozy-idle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("idle.json");
        let lifecycle = Lifecycle::open(path.clone(), true, true).unwrap();
        let first = lifecycle.keepalive("a").unwrap();
        assert_eq!(lifecycle.keepalive("a").unwrap(), first);
        let held = lifecycle.admit().unwrap();
        lifecycle.state.lock().unwrap().0.deadline_ms = 0;
        assert!(
            !lifecycle.claim().unwrap(),
            "an admitted call holds the release"
        );
        drop(held);
        assert!(
            lifecycle.admit().is_err(),
            "a due release admits no new work"
        );
        assert!(lifecycle.claim().unwrap());
        // A restart keeps the committed release; the Go agent reads the same ledger.
        let again = Lifecycle::open(path.clone(), true, false).unwrap();
        assert!(again.released());
        assert!(again.keepalive("b").is_err());
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(raw["released"], true);
        assert!(raw["deadline_ms"].is_i64());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod hub_tests {
    use super::*;
    use crate::machine::grant::HubGrant;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use ed25519_dalek::SigningKey;
    use std::collections::VecDeque;
    use tokio_rustls::rustls;

    type Answers = Arc<Mutex<VecDeque<(u16, String)>>>;
    type Seen = Arc<Mutex<Vec<String>>>;

    /// A Hub stand-in over real HTTPS: answers in order and records method, path and headers.
    async fn fake_hub(answers: Answers, seen: Seen) -> (u16, Vec<u8>) {
        let key = rcgen::KeyPair::generate().unwrap();
        let cert = rcgen::CertificateParams::new(vec!["localhost".into()])
            .unwrap()
            .self_signed(&key)
            .unwrap();
        let der = cert.der().to_vec();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![der.clone().into()],
            rustls::pki_types::PrivateKeyDer::Pkcs8(key.serialize_der().into()),
        )
        .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (tcp, _) = listener.accept().await.unwrap();
                let (acceptor, answers, seen) = (acceptor.clone(), answers.clone(), seen.clone());
                tokio::spawn(async move {
                    let tls = acceptor.accept(tcp).await.unwrap();
                    let service = hyper::service::service_fn(
                        move |request: hyper::Request<hyper::body::Incoming>| {
                            let header = |name| {
                                request
                                    .headers()
                                    .get(name)
                                    .and_then(|v| v.to_str().ok())
                                    .unwrap_or("")
                                    .to_owned()
                            };
                            seen.lock().unwrap().push(format!(
                                "{} {} {} {}",
                                request.method(),
                                request.uri().path(),
                                header("x-cozy-worker-id"),
                                header("x-cozy-worker-token")
                            ));
                            let (status, body) = answers
                                .lock()
                                .unwrap()
                                .pop_front()
                                .unwrap_or((500, String::new()));
                            async move {
                                Ok::<_, std::convert::Infallible>(
                                    hyper::Response::builder()
                                        .status(status)
                                        .body(http_body_util::Full::new(bytes::Bytes::from(body)))
                                        .unwrap(),
                                )
                            }
                        },
                    );
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(hyper_util::rt::TokioIo::new(tls), service)
                        .await;
                });
            }
        });
        (port, der)
    }

    #[tokio::test]
    async fn lease_replaces_boot_keys_denial_revokes_and_release_is_accepted() {
        let (owner, other) = (
            SigningKey::from_bytes(&[1; 32]),
            SigningKey::from_bytes(&[2; 32]),
        );
        let answers: Answers = Default::default();
        let seen: Seen = Default::default();
        let (port, ca) = fake_hub(answers.clone(), seen.clone()).await;
        let token = URL_SAFE_NO_PAD.encode([3; 32]);
        let hub = Arc::new(
            Hub::new(HubGrant {
                origin: format!("https://localhost:{port}"),
                public_origin: None,
                worker_id: "ra-1".into(),
                worker_token: token.clone(),
                ca_der: Some(ca),
                object_hosts: vec![],
            })
            .unwrap(),
        );
        let keys = Keys::fixed(vec![owner.verifying_key()]);
        let authority = crate::api::auth::Authority {
            worker_id: "ra-1".into(),
            boot_id: "boot".into(),
            leaf_digest: [9; 32],
            keys: keys.clone(),
        };
        let claim = |key: &SigningKey| crate::api::pb::Claim {
            record_owner_epoch: 1,
            worker_id: "ra-1".into(),
            worker_boot_id: "boot".into(),
            proof: ed25519_dalek::Signer::sign(key, &authority.transcript(1).unwrap())
                .to_bytes()
                .to_vec(),
            ..Default::default()
        };
        authority.verify(Some(&claim(&owner))).unwrap();
        let lease = |key: &SigningKey| {
            (200, serde_json::json!({"worker_id": "ra-1", "authorized_keys": [URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes())], "lease_seconds": 60}).to_string())
        };
        answers.lock().unwrap().extend([
            (425, String::new()),
            lease(&other),
            (401, String::new()),
            (204, String::new()),
        ]);
        let task = tokio::spawn(keep_authority(hub.clone(), keys));
        // 425 (not yet ready) keeps the boot keys; the lease then replaces them.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(authority.verify(Some(&claim(&owner))).is_err());
        authority.verify(Some(&claim(&other))).unwrap();
        task.abort();
        // An explicit denial revokes at once.
        let keys = authority.keys.clone();
        let task = tokio::spawn(keep_authority(hub.clone(), keys));
        tokio::time::sleep(Duration::from_millis(300)).await;
        task.abort();
        assert!(authority.verify(Some(&claim(&other))).is_err());
        hub.release().await.unwrap();
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen.last().unwrap(),
            &format!("POST /v1/worker/rental/release ra-1 {token}")
        );
        assert!(seen[0].starts_with("GET /v1/worker/rental/authorized-keys ra-1 "));
    }
}
