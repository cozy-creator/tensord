//! Delegated Hub access (the Go agent's `POST`/`DELETE /v1/hubs/access`) and the catalog
//! reads it authorizes. Access is the signed-in account's execution grant, bound to this
//! machine's TLS leaf, handed over by an owner-signed capability. It is journaled per
//! owner key and origin; nothing here infers authority from a cache or a URL.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use tensorfs_core::{
    sha256,
    transport::{self, Deadline, HostToken, Ledger, SourcePolicy},
};

pub const ACTION: &str = "hub-access";

/// The access document the CLI sends; unknown members are ignored.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Access {
    pub origin: String,
    pub token: String,
    pub expires_at: i64,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default, rename = "ca_der_b64url", skip_serializing_if = "String::is_empty")]
    pub ca: String,
}

/// One journaled grant: the access, whose account it is, and the leaf it was bound to.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Grant {
    pub access: Access,
    pub principal: String,
    pub leaf: String,
}

impl Grant {
    pub fn expired(&self, now: i64) -> bool {
        self.access.expires_at <= now
    }
}

/// `scheme://host:port`, lowercase, default port spelled: one key per origin.
pub fn origin_key(origin: &str) -> Option<String> {
    let origin = origin.trim_end_matches('/');
    let (scheme, rest) = origin.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if rest.is_empty() || rest.contains(['/', '?', '#', '@']) {
        return None;
    }
    let (host, port) = match rest.rsplit_once(':') {
        Some((host, port)) if !port.contains(']') => (host, Some(port.parse::<u16>().ok()?)),
        _ => (rest, None),
    };
    let port = match (port, scheme.as_str()) {
        (Some(port), _) => port,
        (None, "https") => 443,
        (None, "http") => 80,
        _ => return None,
    };
    Some(format!("{scheme}://{}:{port}", host.to_ascii_lowercase()))
}

fn loopback_host(key: &str) -> bool {
    let host = key
        .split_once("://")
        .and_then(|(_, rest)| rest.rsplit_once(':'))
        .map(|(host, _)| host)
        .unwrap_or_default();
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

/// An HTTPS origin, or plaintext HTTP on loopback only.
pub fn valid_origin(origin: &str) -> bool {
    match origin_key(origin) {
        Some(key) => key.starts_with("https://") || key.starts_with("http://") && loopback_host(&key),
        None => false,
    }
}

/// The account a delegated token names (continuity, not authentication: Tensorhub
/// verifies every request). An unfamiliar token may refresh only with identical bytes.
pub fn principal(token: &str) -> String {
    let opaque = || format!("opaque:{}", sha256::hex_digest(token.as_bytes()));
    let parts: Vec<_> = token.split('.').collect();
    if token.len() > 32 << 10 || parts.len() != 3 || parts[2].is_empty() {
        return opaque();
    }
    #[derive(Deserialize)]
    struct Header {
        typ: String,
    }
    #[derive(Deserialize)]
    struct Claims {
        #[serde(default)]
        iss: String,
        #[serde(default)]
        delegated_sub: String,
        #[serde(default)]
        sub: String,
        #[serde(default)]
        permissions: Vec<String>,
    }
    let decode = |part: &str| URL_SAFE_NO_PAD.decode(part.trim_end_matches('=')).ok();
    let header: Option<Header> = decode(parts[0]).and_then(|raw| serde_json::from_slice(&raw).ok());
    let claims: Option<Claims> = decode(parts[1]).and_then(|raw| serde_json::from_slice(&raw).ok());
    match (header, claims) {
        (Some(header), Some(claims))
            if header.typ.trim().eq_ignore_ascii_case("delegated-access+jwt")
                && !claims.iss.is_empty()
                && claims.iss.len() <= 2048
                && !claims.delegated_sub.is_empty()
                && claims.delegated_sub.len() <= 256
                && claims.sub.is_empty()
                && claims.permissions == ["cozy.execution-access"] =>
        {
            format!(
                "delegated:{}",
                serde_json::to_string(&[claims.iss, claims.delegated_sub]).unwrap_or_default()
            )
        }
        _ => opaque(),
    }
}

pub fn validate(access: &Access, now: i64) -> Result<(), &'static str> {
    if !valid_origin(&access.origin) {
        return Err("Hub origin must be an HTTPS origin or loopback HTTP origin");
    }
    let token = &access.token;
    if token.is_empty() || token.len() > 32 << 10 || token.trim() != token || token.contains(['\r', '\n']) {
        return Err("Hub access token is invalid");
    }
    if access.expires_at <= now {
        return Err("Hub access grant has expired");
    }
    let declared = access.environment.get("TENSORHUB_ORIGIN").map(String::as_str).unwrap_or_default();
    if declared.trim_end_matches('/') != access.origin.trim_end_matches('/') {
        return Err("Hub access environment names another origin");
    }
    if !access.ca.is_empty() {
        let der = URL_SAFE_NO_PAD.decode(&access.ca).map_err(|_| "invalid Hub CA encoding")?;
        let pem = pem(&der);
        if rustls_pemfile::certs(&mut pem.as_bytes()).count() != 1 {
            return Err("invalid Hub CA certificate");
        }
    }
    Ok(())
}

fn pem(der: &[u8]) -> String {
    let body = base64::engine::general_purpose::STANDARD.encode(der);
    let lines: Vec<_> = body.as_bytes().chunks(64).map(|c| std::str::from_utf8(c).unwrap()).collect();
    format!("-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n", lines.join("\n"))
}

/// A `Cozy-Cap` capability (`base64url(JSON).base64url(sig)`, domain `cozy-capability/1\0`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Capability {
    m: String,
    #[serde(default)]
    r: String,
    #[serde(default)]
    a: String,
    #[serde(default)]
    #[allow(dead_code)]
    p: Vec<String>,
    e: i64,
    k: String,
    #[serde(default)]
    x: String,
}

pub fn key_id(key: &VerifyingKey) -> String {
    URL_SAFE_NO_PAD.encode(&sha256::digest(key.as_bytes())[..16])
}

/// The signer of a valid capability for `action` on `worker`, or None.
pub fn verify_capability(token: &str, worker: &str, keys: &[VerifyingKey], now: i64, action: &str) -> Option<VerifyingKey> {
    let (payload, signature) = token.split_once('.')?;
    let payload = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let signature = Signature::from_slice(&URL_SAFE_NO_PAD.decode(signature).ok()?).ok()?;
    let grant: Capability = serde_json::from_slice(&payload).ok()?;
    if grant.m.is_empty() || grant.r.is_empty() == grant.a.is_empty() || grant.e == 0 {
        return None;
    }
    let mut signed = b"cozy-capability/1\0".to_vec();
    signed.extend_from_slice(&payload);
    let key = keys.iter().find(|key| key_id(key) == grant.k && key.verify_strict(&signed, &signature).is_ok())?;
    (grant.m == worker && grant.x.is_empty() && now < grant.e && grant.r.is_empty() && grant.a == action).then_some(*key)
}

/// A catalog read refusal: the HTTP-level detail, never the token.
#[derive(Debug, Clone)]
pub struct Refusal(pub String);

/// Reads at the grant's Hub with its bearer, presented only to the Hub host.
pub struct Catalog {
    origin: String,
    host: String,
    token: String,
    policy: SourcePolicy,
}

impl Catalog {
    pub fn new(access: &Access) -> Result<Self, Refusal> {
        let origin = access.origin.trim_end_matches('/').to_string();
        let host = transport::base_host(&origin).map_err(|e| Refusal(e.to_string()))?;
        if !access.ca.is_empty() {
            let der = URL_SAFE_NO_PAD.decode(&access.ca).map_err(|_| Refusal("invalid Hub CA".into()))?;
            transport::trust_roots(pem(&der).as_bytes()).map_err(|e| Refusal(e.to_string()))?;
        }
        let mut allowed = vec![host.clone()];
        if let Some(hosts) = access.environment.get("TENSORHUB_OBJECT_STORAGE_HOSTS") {
            allowed.extend(hosts.split(',').map(str::trim).filter(|h| !h.is_empty()).map(String::from));
        }
        let policy = SourcePolicy {
            allowed_hosts: allowed,
            allow_local: origin_key(&origin).is_some_and(|key| loopback_host(&key)),
            ..Default::default()
        };
        Ok(Self { origin, host, token: access.token.clone(), policy })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn credential(&self) -> HostToken {
        HostToken { hosts: vec![self.host.clone()], token: self.token.clone() }
    }
    pub fn policy(&self) -> &SourcePolicy {
        &self.policy
    }
    pub fn bytes(&self, path: &str, cap: u64) -> Result<Vec<u8>, Refusal> {
        let url = format!("{}{path}", self.origin);
        let policy = SourcePolicy { allowed_hosts: vec![self.host.clone()], ..self.policy.clone() };
        transport::api_get(&url, &policy, &self.credential(), cap, Deadline::none(), &Ledger::new())
            .map(|(body, _)| body)
            .map_err(|e| Refusal(format!("{path}: {}", e.detail)))
    }
    pub fn json(&self, path: &str) -> Result<Value, Refusal> {
        serde_json::from_slice(&self.bytes(path, 4 << 20)?).map_err(|_| Refusal(format!("{path}: invalid JSON")))
    }
}

/// Path segment escaping for catalog names and refs.
pub fn escape(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn mint(key: &SigningKey, payload: &str) -> String {
        let mut signed = b"cozy-capability/1\0".to_vec();
        signed.extend_from_slice(payload.as_bytes());
        format!("{}.{}", URL_SAFE_NO_PAD.encode(payload), URL_SAFE_NO_PAD.encode(key.sign(&signed).to_bytes()))
    }

    #[test]
    fn capability_admits_only_its_signer_worker_action_and_lifetime() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let other = SigningKey::from_bytes(&[8; 32]);
        let keys = [key.verifying_key()];
        let id = key_id(&key.verifying_key());
        let good = mint(&key, &format!(r#"{{"m":"w1","a":"hub-access","e":200,"k":"{id}"}}"#));
        assert!(verify_capability(&good, "w1", &keys, 100, ACTION).is_some());
        assert!(verify_capability(&good, "w2", &keys, 100, ACTION).is_none());
        assert!(verify_capability(&good, "w1", &keys, 200, ACTION).is_none());
        assert!(verify_capability(&good, "w1", &keys, 100, "runtime-update").is_none());
        assert!(verify_capability(&good, "w1", &[other.verifying_key()], 100, ACTION).is_none());
        let forged = mint(&other, &format!(r#"{{"m":"w1","a":"hub-access","e":200,"k":"{id}"}}"#));
        assert!(verify_capability(&forged, "w1", &keys, 100, ACTION).is_none());
        let unknown = mint(&key, &format!(r#"{{"m":"w1","a":"hub-access","e":200,"k":"{id}","z":1}}"#));
        assert!(verify_capability(&unknown, "w1", &keys, 100, ACTION).is_none());
        let run = mint(&key, &format!(r#"{{"m":"w1","r":"7","e":200,"k":"{id}"}}"#));
        assert!(verify_capability(&run, "w1", &keys, 100, ACTION).is_none());
    }

    #[test]
    fn origins_compare_by_scheme_host_and_port() {
        assert_eq!(origin_key("https://Hub.Example/"), origin_key("HTTPS://hub.example:443"));
        assert_ne!(origin_key("https://hub.example"), origin_key("http://hub.example"));
        assert!(valid_origin("http://127.0.0.1:8819"));
        assert!(!valid_origin("http://hub.example"));
        assert!(!valid_origin("https://hub.example/path"));
        assert!(!valid_origin("https://user@hub.example"));
    }

    #[test]
    fn principal_is_issuer_and_delegated_subject() {
        let part = |v: &str| URL_SAFE_NO_PAD.encode(v);
        let token = |sub: &str| format!("{}.{}.sig", part(r#"{"typ":"delegated-access+jwt"}"#), part(&format!(r#"{{"iss":"https://h","delegated_sub":"{sub}","permissions":["cozy.execution-access"],"n":1}}"#)));
        assert_eq!(principal(&token("a")), principal(&token("a")));
        assert_ne!(principal(&token("a")), principal(&token("b")));
        assert!(principal("opaque-token").starts_with("opaque:"));
    }
}
