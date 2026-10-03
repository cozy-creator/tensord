//! The signed grant a client presents to one machine (`Authorization: Cozy-Cap`, or a media
//! session's hello): an authorized key's permission to read a run's outputs, or to maintain
//! the machine, until an expiry. Verified offline; the machine stores no bearer. Same token as
//! the Go agent's `capability` package.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use tensorfs_core::sha256;

const DOMAIN: &[u8] = b"cozy-capability/1\0";

/// A member this verifier does not know refuses the whole capability: a restriction an older
/// machine cannot read must not be dropped.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    #[serde(rename = "m")]
    pub machine: String,
    #[serde(rename = "r", default, skip_serializing_if = "String::is_empty")]
    pub run: String,
    #[serde(rename = "a", default, skip_serializing_if = "String::is_empty")]
    pub action: String,
    /// "name" (every index) or "name/i"; empty: every output.
    #[serde(rename = "p", default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<String>,
    /// Unix seconds.
    #[serde(rename = "e")]
    pub expires: i64,
    /// The signer's [`key_id`].
    #[serde(rename = "k")]
    pub key: String,
    /// The client's DTLS certificate (`sha-256 AB:…`), WebRTC only.
    #[serde(rename = "x", default, skip_serializing_if = "String::is_empty")]
    pub binding: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Invalid,
    Expired,
    Scope,
}
impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Refusal::Invalid => "the capability is malformed or not signed by an authorized key",
            Refusal::Expired => "the capability has expired",
            Refusal::Scope => "the capability does not grant this",
        })
    }
}

pub fn key_id(key: &VerifyingKey) -> String {
    URL_SAFE_NO_PAD.encode(&sha256::digest(key.as_bytes())[..16])
}

/// Signs a grant (clients and tests; the machine only verifies).
pub fn mint(key: &ed25519_dalek::SigningKey, mut grant: Grant) -> String {
    use ed25519_dalek::Signer;
    grant.key = key_id(&key.verifying_key());
    let payload = serde_json::to_vec(&grant).expect("a grant serializes");
    let signature = key.sign(&[DOMAIN, &payload].concat());
    format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(&payload),
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    )
}

/// Admits `token` for `machine` at `now` (unix seconds), signed by one of `keys`. `binding`
/// is the client's DTLS fingerprint on WebRTC and empty on HTTPS: a grant naming a binding
/// holds only there.
pub fn verify(
    token: &str,
    machine: &str,
    keys: &[VerifyingKey],
    now: i64,
    binding: &str,
) -> Result<Grant, Refusal> {
    let (encoded, signature) = token.split_once('.').ok_or(Refusal::Invalid)?;
    let payload = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| Refusal::Invalid)?;
    let signature = URL_SAFE_NO_PAD
        .decode(signature)
        .ok()
        .and_then(|bytes| Signature::from_slice(&bytes).ok())
        .ok_or(Refusal::Invalid)?;
    let grant: Grant = serde_json::from_slice(&payload).map_err(|_| Refusal::Invalid)?;
    if grant.machine.is_empty()
        || grant.run.is_empty() == grant.action.is_empty()
        || grant.expires == 0
    {
        return Err(Refusal::Invalid);
    }
    let signed = [DOMAIN, &payload].concat();
    if !keys
        .iter()
        .any(|key| key_id(key) == grant.key && key.verify_strict(&signed, &signature).is_ok())
    {
        return Err(Refusal::Invalid);
    }
    if grant.machine != machine || !grant.binding.is_empty() && grant.binding != binding {
        return Err(Refusal::Invalid);
    }
    if now >= grant.expires {
        return Err(Refusal::Expired);
    }
    Ok(grant)
}

impl Grant {
    /// Whether the grant covers one output of a run; `index` is a list item's 1-based index.
    pub fn allows(&self, run: &str, output: &str, index: Option<u32>) -> bool {
        run == self.run
            && (self.outputs.is_empty()
                || self.outputs.iter().any(|granted| {
                    granted == output || index.is_some_and(|i| *granted == format!("{output}/{i}"))
                }))
    }
}
