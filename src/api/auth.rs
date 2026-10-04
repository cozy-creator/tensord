//! Deployed ClaimProof/1. Wire versions are observations, never authorization.
use super::pb::Claim;
use ed25519_dalek::{Signature, VerifyingKey};
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tensorfs_core::{canon, sha256};
use tonic::Status;

#[derive(Clone)]
pub struct Authority {
    pub worker_id: String,
    pub boot_id: String,
    pub leaf_digest: [u8; 32],
    pub keys: Keys,
}

/// The keys whose Claims this machine admits, shared by every clone of its Authority. A
/// rental's set is a Hub lease: its granted boot value holds until the Hub first answers,
/// then each lease holds until it expires; a transport failure neither revokes nor extends
/// it, a denial revokes it at once. Losing authority never cancels accepted work.
#[derive(Clone)]
pub struct Keys {
    state: Arc<RwLock<KeyState>>,
    changes: Arc<tokio::sync::watch::Sender<u64>>,
}
/// One admitted key and whose it is, as SSH authorized_keys names keys for one account. Runs,
/// outputs and workspaces belong to the actor; only an owner maintains the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Holder {
    pub key: VerifyingKey,
    pub actor: [u8; 32],
    pub owner: bool,
}
impl Holder {
    /// A key of the machine's owner: every key in a computer's authorized_keys, a rental's boot
    /// keys and its renter's leased keys. They are one actor, so the owner's work stays its own
    /// across its devices, a new key after logout, and a rental's boot keys becoming its lease.
    pub fn owner(key: VerifyingKey) -> Self {
        Self {
            key,
            actor: sha256::digest(b"cozy.machine.owner/1"),
            owner: true,
        }
    }
    /// A key of a user the renter shared the rental with: every key of that user is one actor.
    pub fn member(key: VerifyingKey, user: &str) -> Self {
        let mut named = b"cozy.machine.user/1\0".to_vec();
        named.extend_from_slice(user.as_bytes());
        Self {
            key,
            actor: sha256::digest(&named),
            owner: false,
        }
    }
}
struct KeyState {
    keys: Vec<Holder>,
    until: Option<Instant>,
    leased: bool,
}
impl Keys {
    pub fn fixed(keys: Vec<VerifyingKey>) -> Self {
        let state = KeyState {
            keys: keys.into_iter().map(Holder::owner).collect(),
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
    /// A computer's set, read from its `authorized_keys`: no lease.
    pub fn set(&self, keys: Vec<VerifyingKey>) {
        self.replace(KeyState {
            keys: keys.into_iter().map(Holder::owner).collect(),
            until: None,
            leased: false,
        });
    }
    pub fn renew(&self, keys: Vec<Holder>, lease: Duration) {
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
        self.holders().iter().map(|h| h.key).collect()
    }
    /// The admitted keys with whose each is, as one snapshot: verify a signature and name its
    /// actor against the same set.
    pub fn holders(&self) -> Vec<Holder> {
        match self.current() {
            (keys, true) => keys,
            (_, false) => vec![],
        }
    }
    fn current(&self) -> (Vec<Holder>, bool) {
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

/// The key that signed a call and whom it acts as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedActor {
    pub public_key: [u8; 32],
    /// Runs, outputs and workspaces belong to it: the key itself, or a Hub account's.
    pub actor: [u8; 32],
    /// May maintain the machine (update, keepalive): a computer's keys and a rental's renter.
    pub owner: bool,
}
impl VerifiedActor {
    /// A key that is its own actor and an owner, for tools and tests that take no key set.
    pub fn own(public_key: [u8; 32]) -> Self {
        Self {
            public_key,
            actor: public_key,
            owner: true,
        }
    }
    /// The signer among `holders`, the set its signature was verified against.
    pub fn of(holders: &[Holder], signer: &VerifyingKey) -> Option<Self> {
        holders.iter().find(|h| h.key == *signer).map(|h| Self {
            public_key: h.key.to_bytes(),
            actor: h.actor,
            owner: h.owner,
        })
    }
}

impl Authority {
    pub fn verify(&self, claim: Option<&Claim>) -> Result<VerifiedActor, Status> {
        let claim = claim.ok_or_else(|| Status::unauthenticated("the call carries no Claim"))?;
        if claim.worker_id != self.worker_id || claim.worker_boot_id != self.boot_id {
            return Err(Status::unauthenticated(
                "Claim does not name this machine and boot",
            ));
        }
        let bytes = self.transcript(claim.record_owner_epoch)?;
        let signature = Signature::from_slice(&claim.proof)
            .map_err(|_| Status::unauthenticated("Claim signature must be 64 bytes"))?;
        let (keys, live) = self.keys.current();
        if let Some(holder) = keys
            .iter()
            .find(|h| h.key.verify_strict(&bytes, &signature).is_ok())
        {
            if !live {
                return Err(Status::unavailable(
                    "rental_authority_unavailable: no current rental authority lease",
                ));
            }
            Ok(VerifiedActor {
                public_key: holder.key.to_bytes(),
                actor: holder.actor,
                owner: holder.owner,
            })
        } else {
            Err(Status::unauthenticated(
                "Claim is not signed by an authorized machine key",
            ))
        }
    }

    pub fn transcript(&self, epoch: u64) -> Result<Vec<u8>, Status> {
        // The existing profile is integer-only, bounded and printable ASCII. Validate
        // before using TensorFS's canonical writer; its Value::uint is an unchecked cast.
        if epoch > (1u64 << 53) - 1
            || [&self.worker_id, &self.boot_id]
                .into_iter()
                .any(|s| !s.bytes().all(|b| (0x20..=0x7e).contains(&b)))
        {
            return Err(Status::unauthenticated(
                "Claim identity is outside the canonical document profile",
            ));
        }
        let mut fields = vec![
            ("format", canon::Value::str("cozy.worker.v1.ClaimProof/1")),
            (
                "worker_tls_certificate_digest",
                canon::Value::str(format!("sha256:{}", sha256::hex(&self.leaf_digest))),
            ),
        ];
        // Proto3 default values are omitted by the deployed canonicalizer.
        if epoch != 0 {
            fields.push(("record_owner_epoch", canon::Value::uint(epoch)));
        }
        if !self.boot_id.is_empty() {
            fields.push(("worker_boot_id", canon::Value::str(&self.boot_id)));
        }
        if !self.worker_id.is_empty() {
            fields.push(("worker_id", canon::Value::str(&self.worker_id)));
        }
        Ok(canon::write(&canon::Value::obj(fields)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn deployed_claim_proof_fixture_matches_exactly() {
        let digest = "4c5b5699f2d99ebf9195c1e518d13c94539bbd4ba670a46199f6d79d58d15721";
        let mut raw = [0; 32];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = u8::from_str_radix(&digest[i * 2..i * 2 + 2], 16).unwrap();
        }
        let auth = Authority {
            worker_id: "wrk-4070".into(),
            boot_id: "boot-9f21".into(),
            leaf_digest: raw,
            keys: vec![].into(),
        };
        assert_eq!(
            auth.transcript(41).unwrap(),
            include_bytes!("../../vendor/worker-protocol/fixtures/claim_proof.json")
        );
    }

    #[test]
    fn identity_signature_and_leaf_are_fenced_but_version_is_not() {
        let key = SigningKey::from_bytes(&[23; 32]);
        let auth = Authority {
            worker_id: "cpu-private".into(),
            boot_id: "boot".into(),
            leaf_digest: [9; 32],
            keys: vec![key.verifying_key()].into(),
        };
        let mut claim = Claim {
            worker_id: auth.worker_id.clone(),
            worker_boot_id: auth.boot_id.clone(),
            record_owner_epoch: 1,
            proof: key.sign(&auth.transcript(1).unwrap()).to_bytes().to_vec(),
            ..Default::default()
        };
        for version in [0, 1, 64, 72, u32::MAX] {
            claim.wire_minor = version;
            auth.verify(Some(&claim)).unwrap();
        }
        claim.worker_boot_id = "stale".into();
        assert!(auth.verify(Some(&claim)).is_err());
        claim.worker_boot_id = auth.boot_id.clone();
        let other = Authority {
            leaf_digest: [8; 32],
            ..auth.clone()
        };
        assert!(other.verify(Some(&claim)).is_err());
        claim.proof[0] ^= 1;
        assert!(auth.verify(Some(&claim)).is_err());
        assert!(auth.verify(None).is_err());
    }
}
