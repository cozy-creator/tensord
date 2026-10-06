//! Authorized signing keys and the lifetimes of capability-authenticated streams.
use ed25519_dalek::VerifyingKey;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, RwLock},
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio_stream::Stream;
use tonic::Status;

#[derive(Clone)]
pub struct Authority {
    pub worker_id: String,
    pub boot_id: String,
    pub keys: Keys,
}

/// The signing keys this machine admits, shared by every clone of its Authority. A
/// rental's set is a Hub lease: its granted boot value holds until the Hub first answers,
/// then each lease holds until it expires; a transport failure neither revokes nor extends
/// it, a denial revokes it at once. Losing authority never cancels accepted work.
#[derive(Clone)]
pub struct Keys {
    state: Arc<RwLock<KeyState>>,
    changes: Arc<tokio::sync::watch::Sender<u64>>,
}
struct KeyState {
    keys: Vec<VerifyingKey>,
    until: Option<Instant>,
    leased: bool,
}
impl Keys {
    pub fn fixed(keys: Vec<VerifyingKey>) -> Self {
        let state = KeyState {
            keys,
            until: None,
            leased: false,
        };
        Self {
            state: Arc::new(RwLock::new(state)),
            changes: Arc::new(tokio::sync::watch::channel(0).0),
        }
    }
    fn replace(&self, state: KeyState) {
        *self.state.write().unwrap() = state;
        self.changes.send_modify(|generation| *generation += 1);
    }
    pub fn renew(&self, keys: Vec<VerifyingKey>, lease: Duration) {
        self.replace(KeyState {
            keys,
            until: Some(Instant::now() + lease),
            leased: true,
        });
    }
    pub fn revoke(&self) {
        self.replace(KeyState {
            keys: vec![],
            until: None,
            leased: true,
        });
    }
    /// The keys that may authorize a new control now: none without a current lease.
    pub fn admitted(&self) -> Vec<VerifyingKey> {
        match self.current() {
            (keys, true) => keys,
            (_, false) => vec![],
        }
    }
    fn current(&self) -> (Vec<VerifyingKey>, bool) {
        let state = self.state.read().unwrap();
        let live = !state.leased || state.until.is_some_and(|until| Instant::now() < until);
        (state.keys.clone(), live)
    }
    /// Resolves once `key` no longer authorizes: it left the set, or the lease that admits
    /// it expired. Ends only that key's open transports; accepted work is never cancelled.
    pub async fn revoked(&self, key: [u8; 32]) {
        let mut changes = self.changes.subscribe();
        loop {
            if !self.admitted().iter().any(|k| k.to_bytes() == key) {
                return;
            }
            let until = self.state.read().unwrap().until;
            let expiry = async {
                match until {
                    Some(until) => tokio::time::sleep_until(until.into()).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                _ = changes.changed() => (),
                _ = expiry => (),
            }
        }
    }
}
impl From<Vec<VerifyingKey>> for Keys {
    fn from(keys: Vec<VerifyingKey>) -> Self {
        Self::fixed(keys)
    }
}

/// Why an open transport lost its authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lapse {
    /// The key that opened it left the admitted set, or the lease admitting it ran out.
    Revoked,
    /// The expiry its signer chose passed (a play link's `--expires`).
    Expired,
}
impl Lapse {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Lapse::Revoked => "revoked",
            Lapse::Expired => "expired",
        }
    }
    pub(crate) fn message(self) -> &'static str {
        match self {
            Lapse::Revoked => "the key that opened this stream no longer authorizes it",
            Lapse::Expired => "the capability that opened this stream has expired",
        }
    }
}
impl From<Lapse> for Status {
    fn from(lapse: Lapse) -> Self {
        Status::unauthenticated(lapse.message())
    }
}

type Transport<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// One open transport's authority. A `cozy.machine.v1` stream ends only when its key stops
/// authorizing: its capability's expiry was checked when it opened, and a long run, upload or
/// read outlives it. A media session also ends at its signed expiry, which its viewer chose.
/// Losing authority detaches the transport; it never controls accepted work.
#[derive(Clone)]
pub(crate) struct StreamAuthority {
    keys: Keys,
    key: [u8; 32],
    expires: Option<i64>,
}
impl StreamAuthority {
    pub(crate) fn new(keys: Keys, actor: VerifiedActor, expires: Option<i64>) -> Self {
        Self { keys, key: actor.public_key, expires }
    }
    pub(crate) fn check(&self) -> Result<(), Lapse> {
        if self.expires.is_some_and(|expires| unix_now() >= expires) {
            return Err(Lapse::Expired);
        }
        match self.keys.admitted().iter().any(|key| key.to_bytes() == self.key) {
            true => Ok(()),
            false => Err(Lapse::Revoked),
        }
    }
    pub(crate) async fn ended(self) -> Lapse {
        let expiry = async {
            let Some(expires) = self.expires else {
                return std::future::pending().await;
            };
            // A wall-clock expiry, re-read at most daily so a distant one stays in range.
            while unix_now() < expires {
                let left = (expires - unix_now()).clamp(1, 86_400) as u64;
                tokio::time::sleep(Duration::from_secs(left)).await;
            }
        };
        tokio::select! {
            _ = self.keys.revoked(self.key) => Lapse::Revoked,
            _ = expiry => Lapse::Expired,
        }
    }
    /// Ends `stream` with UNAUTHENTICATED once authority lapses, dropping it (and the observer
    /// and files behind it) even while its peer applies backpressure.
    pub(crate) fn wrap<T: Send + 'static>(&self, stream: Transport<T>) -> Transport<T> {
        Box::pin(Authorized { stream: Some(stream), ended: Box::pin(self.clone().ended()) })
    }
}
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() as i64)
}
struct Authorized<T> {
    stream: Option<Transport<T>>,
    ended: Pin<Box<dyn Future<Output = Lapse> + Send>>,
}
impl<T> Stream for Authorized<T> {
    type Item = Result<T, Status>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.stream.is_none() {
            return Poll::Ready(None);
        }
        if let Poll::Ready(lapse) = self.ended.as_mut().poll(cx) {
            self.stream = None;
            return Poll::Ready(Some(Err(lapse.into())));
        }
        self.stream.as_mut().unwrap().as_mut().poll_next(cx)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedActor {
    pub public_key: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    /// Revocation ends a stream mid-flight; a signed expiry ends a quiet one.
    #[tokio::test]
    async fn a_transport_ends_when_its_authority_lapses() {
        use tokio_stream::StreamExt;
        let signer = SigningKey::from_bytes(&[83; 32]);
        let actor = VerifiedActor { public_key: signer.verifying_key().to_bytes() };
        let keys = Keys::fixed(vec![signer.verifying_key()]);
        let mut stream = StreamAuthority::new(keys.clone(), actor, None)
            .wrap(Box::pin(tokio_stream::iter([Ok(1), Ok(2)])));
        assert_eq!(stream.next().await.unwrap().unwrap(), 1);
        keys.revoke();
        assert_eq!(stream.next().await.unwrap().unwrap_err().message(), Lapse::Revoked.message());
        assert!(stream.next().await.is_none());

        let keys = Keys::fixed(vec![signer.verifying_key()]);
        let authority = StreamAuthority::new(keys, actor, Some(unix_now() + 1));
        assert_eq!(authority.check(), Ok(()));
        assert_eq!(authority.clone().ended().await, Lapse::Expired);
        assert_eq!(authority.check(), Err(Lapse::Expired));
    }
}
