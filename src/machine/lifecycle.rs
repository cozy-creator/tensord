//! A rental's lifecycle: it releases itself after 15 minutes with no job queued for it or
//! running on it, used or not (owner ruling 2026-10-10); every run asked of it is a job (a
//! call, a job, a warm-up, an upload); an explicit keepalive restarts that clock once. The clock is durable state, never a sample held in memory: the journal's last
//! job end and the ledger's idle start (this rental's first boot, or a keepalive), so restarts
//! and updates neither shorten nor lose it. Also the Hub lease of authorized keys.
use super::hub::{Hub, Refusal};
use crate::api::auth::Keys;
use std::{
    io,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tonic::Status;

/// A rental's idle window, `RentalIdleTimeoutSeconds` in the CLI.
pub const IDLE_GRACE_MS: i64 = 900_000;

/// The journal's jobs: one queued or running, and when the last one stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Jobs {
    pub working: bool,
    pub ended_ms: i64,
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Ledger {
    /// When the idle clock last started without a job: first boot, or a keepalive.
    idle_since_ms: i64,
    released: bool,
}

struct State {
    ledger: Ledger,
    jobs: Jobs,
    /// The admission epoch `jobs` was read after: a later admission may have queued a job.
    sampled: u64,
    /// An update is activating: no new work is admitted until it exits or fails.
    activating: bool,
}

pub struct Lifecycle {
    rental: bool,
    /// The idle window: `IDLE_GRACE_MS` unless the grant names another.
    grace: i64,
    path: PathBuf,
    state: Mutex<State>,
    admissions: AtomicUsize,
    epoch: AtomicU64,
}

/// Holds the release while a call that may queue a job is in flight.
pub struct Admission(Arc<Lifecycle>);

/// Closes admission while an update waits for admitted work to drain and replaces the
/// service. Dropping it (a failed activation) reopens admission.
pub struct Activation(Arc<Lifecycle>);
impl Drop for Activation {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().activating = false;
    }
}
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
    /// `fresh` starts this root's first idle clock; a restart keeps the persisted ledger.
    pub fn open(path: PathBuf, rental: bool, fresh: bool, grace: i64) -> io::Result<Arc<Self>> {
        let mut ledger: Ledger = match std::fs::read(&path) {
            Ok(raw) if !fresh => serde_json::from_slice(&raw)
                .map_err(|e| io::Error::other(format!("the idle ledger is unreadable: {e}")))?,
            Ok(_) => Ledger::default(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ledger::default(),
            Err(e) => return Err(e),
        };
        if ledger.idle_since_ms == 0 {
            ledger.idle_since_ms = now_ms();
        }
        let lifecycle = Arc::new(Self {
            rental,
            grace,
            path,
            state: Mutex::new(State { ledger, jobs: Jobs::default(), sampled: 0, activating: false }),
            admissions: AtomicUsize::new(0),
            epoch: AtomicU64::new(0),
        });
        lifecycle.save(&lifecycle.state.lock().unwrap().ledger)?;
        Ok(lifecycle)
    }

    fn save(&self, ledger: &Ledger) -> io::Result<()> {
        super::identity::write_atomic(&self.path, &serde_json::to_vec(ledger)?, 0o600)
    }

    /// When the idle clock runs out: 15 minutes after the later of its start and the last
    /// job's end.
    fn idle_end(&self, state: &State) -> i64 {
        state.ledger.idle_since_ms.max(state.jobs.ended_ms) + self.grace
    }

    fn due(&self, state: &State, now: i64) -> bool {
        state.ledger.released || (!state.jobs.working && now >= self.idle_end(state))
    }

    /// The deadline a caller is told: none (0) while a job is queued or running, or while an
    /// admission may have queued one since the journal was last read.
    fn deadline(&self, state: &State) -> i64 {
        let working = state.jobs.working || self.admitted() > 0 || state.sampled != self.epoch();
        match working && !state.ledger.released {
            true => 0,
            false => self.idle_end(state),
        }
    }

    pub fn admit(self: &Arc<Self>) -> Result<Admission, Status> {
        let state = self.state.lock().unwrap();
        if state.activating {
            return Err(Status::unavailable(
                "machine_updating: the machine is activating an update; submit again once it is back",
            ));
        }
        if self.rental && self.due(&state, now_ms()) {
            return Err(Status::unavailable(
                "machine_released: this rental released itself after 15 minutes idle",
            ));
        }
        self.admissions.fetch_add(1, Ordering::AcqRel);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(Admission(self.clone()))
    }

    /// Closes admission under the lock `admit` takes: a call is either admitted (and counted)
    /// before activation or refused after it, never accepted while the service exits.
    pub fn activate(self: &Arc<Self>) -> Result<Activation, Status> {
        let mut state = self.state.lock().unwrap();
        if state.activating || state.ledger.released {
            return Err(Status::unavailable(
                "the machine is already activating or released",
            ));
        }
        state.activating = true;
        Ok(Activation(self.clone()))
    }

    /// The admission epoch, read before the journal is.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// The journal's jobs as read after admission epoch `epoch`.
    pub fn observe(&self, jobs: Jobs, epoch: u64) {
        let mut state = self.state.lock().unwrap();
        state.jobs = jobs;
        state.sampled = epoch;
    }

    /// The last job end this lifecycle has read.
    pub fn ended_ms(&self) -> i64 {
        self.state.lock().unwrap().jobs.ended_ms
    }

    /// `Status{keepalive}`: restarts the idle clock once; answers the new deadline.
    pub fn renew(&self) -> Result<i64, Status> {
        if !self.rental {
            return Ok(0);
        }
        let now = now_ms();
        let mut state = self.state.lock().unwrap();
        if self.due(&state, now) {
            return Err(Status::failed_precondition(
                "this rental's idle release is already due or committed",
            ));
        }
        let mut ledger = state.ledger.clone();
        ledger.idle_since_ms = ledger.idle_since_ms.max(now);
        self.save(&ledger).map_err(|e| {
            Status::unavailable(format!("the idle ledger could not be written: {e}"))
        })?;
        state.ledger = ledger;
        Ok(self.deadline(&state))
    }

    /// Calls that may start work, in flight now.
    pub fn admitted(&self) -> usize {
        self.admissions.load(Ordering::Acquire)
    }

    /// When this rental releases itself if no job comes first. 0: no deadline, because a job
    /// is queued or running, or because this machine never releases itself.
    pub fn deadline_ms(&self) -> i64 {
        match self.rental {
            true => self.deadline(&self.state.lock().unwrap()),
            false => 0,
        }
    }

    pub fn released(&self) -> bool {
        self.state.lock().unwrap().ledger.released
    }

    /// Takes the release once the idle clock ran out on a current reading. Irreversible.
    fn claim(&self) -> io::Result<bool> {
        let mut state = self.state.lock().unwrap();
        if !self.rental {
            return Ok(false);
        }
        if state.ledger.released {
            return Ok(true);
        }
        if state.activating
            || self.admissions.load(Ordering::Acquire) > 0
            || state.sampled != self.epoch()
            || !self.due(&state, now_ms())
        {
            return Ok(false);
        }
        let mut ledger = state.ledger.clone();
        ledger.released = true;
        self.save(&ledger)?;
        state.ledger = ledger;
        Ok(true)
    }
}

/// Keeps the Hub's authorized keys: asked again at half the lease it names. While the Hub is
/// unreachable or not yet ready the keys stay as they were, and it is asked again after 1 s,
/// doubling to 5 s (the Hub reads a pod's readiness every 5 s).
pub async fn keep_authority(hub: Arc<Hub>, keys: Keys) {
    const UNANSWERED: (Duration, Duration) = (Duration::from_secs(1), Duration::from_secs(5));
    let mut unanswered = UNANSWERED.0;
    loop {
        let delay = match hub.authorized_keys().await {
            Ok((current, lease)) => {
                keys.renew(current);
                unanswered = UNANSWERED.0;
                lease / 2
            }
            Err(Refusal::Denied) => {
                eprintln!(
                    "cozy-machine: the Hub denied rental authority; no new control is admitted"
                );
                keys.revoke();
                Duration::from_secs(1)
            }
            Err(Refusal::Transport(_)) => {
                let delay = unanswered;
                unanswered = (unanswered * 2).min(UNANSWERED.1);
                delay
            }
        };
        tokio::time::sleep(delay).await;
    }
}

/// Reads the journal's jobs each second and releases the rental once 15 minutes passed with
/// none. Returns when the Hub accepted the release, or when the rental ended itself: idle past
/// its deadline with no answer from the Hub through that whole idle window, the Hub cannot end
/// it, so it ends itself at its provider (`provider`, else by stopping) and billing stops.
pub async fn release_when_idle(
    lifecycle: Arc<Lifecycle>,
    hub: Arc<Hub>,
    provider: Option<super::provider::ProviderSelf>,
    jobs: impl Fn(i64) -> io::Result<Jobs>,
) {
    let mut next_ask = 0;
    loop {
        let epoch = lifecycle.epoch();
        match jobs(lifecycle.ended_ms()) {
            Ok(read) => lifecycle.observe(read, epoch),
            Err(error) => {
                eprintln!("cozy-machine: cannot read this rental's jobs: {error}");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        }
        let now = now_ms();
        if now >= next_ask {
            match lifecycle.claim() {
                Ok(true) => match hub.release().await {
                    Ok(()) => {
                        eprintln!("cozy-machine: 15 minutes without a job; Tensorhub accepted the release");
                        return;
                    }
                    Err(error) => {
                        eprintln!("cozy-machine: idle release not accepted; retrying without extending the deadline: {error}");
                        next_ask = now + 5_000;
                        if unheard(hub.contact_ms(), lifecycle.deadline_ms(), lifecycle.grace) {
                            end_without_hub(provider.as_ref()).await;
                            return;
                        }
                    }
                },
                Ok(false) => (),
                Err(error) => {
                    eprintln!("cozy-machine: idle release could not be recorded: {error}")
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// The Hub has not answered since this idle window began: it cannot hear the release.
fn unheard(contact_ms: i64, deadline_ms: i64, grace: i64) -> bool {
    contact_ms < deadline_ms - grace
}

async fn end_without_hub(provider: Option<&super::provider::ProviderSelf>) {
    eprintln!("cozy-machine: idle past the deadline with no answer from Tensorhub through the whole idle window; ending this rental itself");
    match provider {
        Some(provider) => match provider.end().await {
            Ok(()) => eprintln!("cozy-machine: the provider accepted this pod's end"),
            Err(error) => {
                eprintln!("cozy-machine: the provider did not end this pod ({error}); stopping")
            }
        },
        None => eprintln!("cozy-machine: no provider credential; stopping"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opened(fresh: bool, dir: &std::path::Path) -> Arc<Lifecycle> {
        Lifecycle::open(dir.join("idle.json"), true, fresh, IDLE_GRACE_MS).unwrap()
    }

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Moves the idle clock's start back by `ms`, as if that time had passed.
    fn age(lifecycle: &Lifecycle, ms: i64) {
        lifecycle.state.lock().unwrap().ledger.idle_since_ms -= ms;
    }

    fn read(lifecycle: &Lifecycle, working: bool, ended_ms: i64) {
        lifecycle.observe(Jobs { working, ended_ms }, lifecycle.epoch());
    }

    #[test]
    fn a_rental_used_or_not_releases_after_fifteen_minutes_without_a_job() {
        let dir = dir("idle-rule");
        let lifecycle = opened(true, &dir);
        read(&lifecycle, false, 0);
        assert!(!lifecycle.claim().unwrap(), "a new rental has its 15 minutes");
        age(&lifecycle, IDLE_GRACE_MS);
        // A job queued or running holds it, however long the clock has been idle before.
        read(&lifecycle, true, 0);
        assert!(!lifecycle.claim().unwrap());
        assert_eq!(lifecycle.deadline_ms(), 0, "a job queued or running has no deadline");
        // The clock starts when that job ends: a used rental releases 15 minutes later.
        let ended = now_ms();
        read(&lifecycle, false, ended);
        assert_eq!(lifecycle.deadline_ms(), ended + IDLE_GRACE_MS);
        assert!(!lifecycle.claim().unwrap());
        read(&lifecycle, false, ended - IDLE_GRACE_MS);
        assert!(lifecycle.claim().unwrap(), "a used rental idle 15 minutes releases");
        assert!(lifecycle.admit().is_err(), "a released rental admits nothing");
        assert!(lifecycle.renew().is_err(), "nor restarts its clock");
        let again = opened(false, &dir);
        assert!(again.released(), "a committed release survives a restart");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn keepalive_restarts_the_clock_once_and_a_restart_keeps_it() {
        let dir = dir("idle-keepalive");
        let lifecycle = opened(true, &dir);
        age(&lifecycle, IDLE_GRACE_MS - 60_000);
        read(&lifecycle, false, 0);
        let before = lifecycle.deadline_ms();
        let renewed = lifecycle.renew().unwrap();
        assert!(renewed >= before + IDLE_GRACE_MS - 61_000, "{before} -> {renewed}");
        let restarted = opened(false, &dir);
        read(&restarted, false, 0);
        assert_eq!(restarted.deadline_ms(), renewed, "a restart neither shortens nor extends it");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_ledger_without_an_idle_start_starts_the_clock_at_boot() {
        let dir = dir("idle-old-ledger");
        std::fs::write(dir.join("idle.json"), br#"{"deadline_ms":1,"work_observed":true}"#).unwrap();
        let lifecycle = opened(false, &dir);
        read(&lifecycle, false, 0);
        assert!(lifecycle.deadline_ms() >= now_ms() + IDLE_GRACE_MS - 1_000);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_admission_or_a_reading_from_before_it_holds_the_release() {
        let dir = dir("idle-admission");
        let lifecycle = opened(true, &dir);
        let epoch = lifecycle.epoch();
        let held = lifecycle.admit().unwrap();
        age(&lifecycle, IDLE_GRACE_MS);
        lifecycle.observe(Jobs::default(), epoch);
        assert!(!lifecycle.claim().unwrap(), "an admitted call holds the release");
        drop(held);
        assert!(!lifecycle.claim().unwrap(), "a reading from before that admission is stale");
        read(&lifecycle, false, 0);
        assert!(lifecycle.claim().unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_rental_ends_itself_only_when_the_hub_was_silent_through_the_idle_window() {
        let (deadline, grace) = (10 * IDLE_GRACE_MS, IDLE_GRACE_MS);
        assert!(unheard(0, deadline, grace), "never heard");
        assert!(unheard(deadline - grace - 1, deadline, grace), "last heard before the window");
        assert!(!unheard(deadline - grace, deadline, grace), "heard as the window began");
        assert!(!unheard(deadline - 1, deadline, grace), "heard during the window");
    }

    /// A rental due for release does not release while an update activates.
    #[test]
    fn activation_holds_the_release() {
        let dir = dir("idle-activation");
        let lifecycle = opened(true, &dir);
        let activation = lifecycle.activate().unwrap();
        age(&lifecycle, IDLE_GRACE_MS);
        read(&lifecycle, false, 0);
        assert!(!lifecycle.claim().unwrap());
        drop(activation);
        assert!(lifecycle.claim().unwrap());
        assert!(
            lifecycle.activate().is_err(),
            "a released rental activates nothing"
        );
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
            keys: keys.clone(),
        };
        let authorized =
            |key: &SigningKey| authority.keys.admitted().contains(&key.verifying_key());
        assert!(authorized(&owner));
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
        assert!(!authorized(&owner));
        assert!(authorized(&other));
        task.abort();
        // An explicit denial revokes at once.
        let keys = authority.keys.clone();
        let task = tokio::spawn(keep_authority(hub.clone(), keys));
        tokio::time::sleep(Duration::from_millis(300)).await;
        task.abort();
        assert!(!authorized(&other));
        hub.release().await.unwrap();
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen.last().unwrap(),
            &format!("POST /v1/worker/rental/release ra-1 {token}")
        );
        assert!(seen[0].starts_with("GET /v1/worker/rental/authorized-keys ra-1 "));
    }
}
