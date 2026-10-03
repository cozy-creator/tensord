//! Delegated Hub access (the Go agent's `POST`/`DELETE /v1/hubs/access`) and the catalog
//! reads it authorizes. Access is the signed-in account's execution grant, bound to this
//! machine's TLS leaf, handed over by an owner-signed capability. It is journaled per
//! owner key and origin; nothing here infers authority from a cache or a URL.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tensorfs_core::{
    sha256,
    transport::{self, Deadline, Ledger, ScopedHeaders, SourcePolicy},
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
    #[serde(
        default,
        rename = "ca_der_b64url",
        skip_serializing_if = "String::is_empty"
    )]
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
        Some(key) => {
            key.starts_with("https://") || key.starts_with("http://") && loopback_host(&key)
        }
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
            if header
                .typ
                .trim()
                .eq_ignore_ascii_case("delegated-access+jwt")
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
    if token.is_empty()
        || token.len() > 32 << 10
        || token.trim() != token
        || token.contains(['\r', '\n'])
    {
        return Err("Hub access token is invalid");
    }
    if access.expires_at <= now {
        return Err("Hub access grant has expired");
    }
    let declared = access
        .environment
        .get("TENSORHUB_ORIGIN")
        .map(String::as_str)
        .unwrap_or_default();
    if declared.trim_end_matches('/') != access.origin.trim_end_matches('/') {
        return Err("Hub access environment names another origin");
    }
    if !access.ca.is_empty() {
        let der = URL_SAFE_NO_PAD
            .decode(&access.ca)
            .map_err(|_| "invalid Hub CA encoding")?;
        let pem = pem(&der);
        if rustls_pemfile::certs(&mut pem.as_bytes()).count() != 1 {
            return Err("invalid Hub CA certificate");
        }
    }
    Ok(())
}

fn pem(der: &[u8]) -> String {
    let body = base64::engine::general_purpose::STANDARD.encode(der);
    let lines: Vec<_> = body
        .as_bytes()
        .chunks(64)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        lines.join("\n")
    )
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
pub fn verify_capability(
    token: &str,
    worker: &str,
    keys: &[VerifyingKey],
    now: i64,
    action: &str,
) -> Option<VerifyingKey> {
    let (payload, signature) = token.split_once('.')?;
    let payload = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let signature = Signature::from_slice(&URL_SAFE_NO_PAD.decode(signature).ok()?).ok()?;
    let grant: Capability = serde_json::from_slice(&payload).ok()?;
    if grant.m.is_empty() || grant.r.is_empty() == grant.a.is_empty() || grant.e == 0 {
        return None;
    }
    let mut signed = b"cozy-capability/1\0".to_vec();
    signed.extend_from_slice(&payload);
    let key = keys
        .iter()
        .find(|key| key_id(key) == grant.k && key.verify_strict(&signed, &signature).is_ok())?;
    (grant.m == worker
        && grant.x.is_empty()
        && now < grant.e
        && grant.r.is_empty()
        && grant.a == action)
        .then_some(*key)
}

/// A catalog read refusal: the HTTP-level detail, never the token.
#[derive(Debug, Clone)]
pub struct Refusal(pub String);

/// Where and as whom a machine reads a Hub: the owner's delegated access (a bearer), or on a
/// rental the pod's own worker capability, which is what a run that names no Hub uses there
/// (the Go agent and Python worker's default registration).
#[derive(Clone, Debug)]
pub struct Source {
    pub origin: String,
    /// TensorFS credential spelling: `bearer <token>` or `worker <id> <token>`.
    pub credential: String,
    pub ca_der: Option<Vec<u8>>,
    pub object_hosts: Vec<String>,
}

impl Source {
    pub fn delegated(access: &Access) -> Self {
        Self {
            origin: access.origin.trim_end_matches('/').to_string(),
            credential: format!("bearer {}", access.token),
            ca_der: URL_SAFE_NO_PAD
                .decode(&access.ca)
                .ok()
                .filter(|der| !der.is_empty()),
            object_hosts: access
                .environment
                .get("TENSORHUB_OBJECT_STORAGE_HOSTS")
                .map(|hosts| {
                    hosts
                        .split(',')
                        .map(str::trim)
                        .filter(|h| !h.is_empty())
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
    pub fn pod(
        origin: &str,
        worker_id: &str,
        worker_token: &str,
        ca_der: Option<Vec<u8>>,
        object_hosts: Vec<String>,
    ) -> Self {
        Self {
            origin: origin.trim_end_matches('/').to_string(),
            credential: format!("worker {worker_id} {worker_token}"),
            ca_der,
            object_hosts,
        }
    }
}

/// Reads at a Hub with its credential, presented only to the Hub host.
pub struct Catalog {
    origin: String,
    host: String,
    credential: String,
    policy: SourcePolicy,
}

impl Catalog {
    pub fn new(source: &Source) -> Result<Self, Refusal> {
        let origin = source.origin.clone();
        let host = transport::base_host(&origin).map_err(|e| Refusal(e.to_string()))?;
        if let Some(der) = &source.ca_der {
            transport::trust_roots(pem(der).as_bytes()).map_err(|e| Refusal(e.to_string()))?;
        }
        let mut allowed = vec![host.clone()];
        allowed.extend(source.object_hosts.iter().cloned());
        let policy = SourcePolicy {
            allowed_hosts: allowed,
            allow_local: origin_key(&origin).is_some_and(|key| loopback_host(&key)),
            ..Default::default()
        };
        Ok(Self {
            origin,
            host,
            credential: source.credential.clone(),
            policy,
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn credential(&self) -> ScopedHeaders {
        transport::credential_from_spec(&self.credential, vec![self.host.clone()]).unwrap_or(
            ScopedHeaders {
                hosts: vec![],
                headers: vec![],
            },
        )
    }
    pub fn policy(&self) -> &SourcePolicy {
        &self.policy
    }
    pub fn bytes(&self, path: &str, cap: u64) -> Result<Vec<u8>, Refusal> {
        let url = format!("{}{path}", self.origin);
        let policy = SourcePolicy {
            allowed_hosts: vec![self.host.clone()],
            ..self.policy.clone()
        };
        transport::api_get(
            &url,
            &policy,
            &self.credential(),
            cap,
            Deadline::none(),
            &Ledger::new(),
        )
        .map(|(body, _)| body)
        .map_err(|e| Refusal(format!("{path}: {}", e.detail)))
    }
    pub fn json(&self, path: &str) -> Result<Value, Refusal> {
        serde_json::from_slice(&self.bytes(path, 4 << 20)?)
            .map_err(|_| Refusal(format!("{path}: invalid JSON")))
    }
}

/// Hub writes under a machine-publication authorization: the Hub mints a short bearer for this
/// machine's leaf, proven by the run's execution access, and it is renewed at half its life.
/// Both go to the Hub host only; presigned object hosts see neither.
pub struct Publishing {
    catalog: Catalog,
    sender: String,
    authorization: String,
    bearer: Mutex<Option<(String, Instant)>>,
}

impl Publishing {
    pub fn new(source: &Source, authorization: &str) -> Result<Self, Refusal> {
        let sender = match source.credential.split_once(' ') {
            Some(("bearer", token)) if !token.is_empty() => token.to_string(),
            _ => {
                return Err(Refusal(
                    "publication needs the run's execution access".into(),
                ))
            }
        };
        let publishing = Self {
            catalog: Catalog::new(source)?,
            sender,
            authorization: authorization.to_string(),
            bearer: Mutex::new(None),
        };
        // A refused authorization surfaces here, before any byte moves.
        *publishing.bearer.lock().unwrap() = Some(publishing.renew()?);
        Ok(publishing)
    }
    pub fn origin(&self) -> &str {
        self.catalog.origin()
    }
    pub fn policy(&self) -> &SourcePolicy {
        self.catalog.policy()
    }
    fn renew(&self) -> Result<(String, Instant), Refusal> {
        let path = format!(
            "/v1/worker/machine-authorizations/{}/token",
            escape(&self.authorization)
        );
        let sender = ScopedHeaders {
            hosts: vec![self.catalog.host.clone()],
            headers: vec![("x-cozy-execution-access".into(), self.sender.clone())],
        };
        let answer =
            transport::hub_call("POST", self.origin(), &path, b"{}", &sender, self.policy())
                .map_err(|e| Refusal(format!("publication authorization: {}", e.detail)))?;
        let token = match &answer {
            tensorfs_core::canon::Value::Obj(pairs) => pairs.iter().find_map(|(k, v)| match v {
                tensorfs_core::canon::Value::Str(token) if k == "token" => Some(token.clone()),
                _ => None,
            }),
            _ => None,
        }
        .ok_or_else(|| Refusal("the Hub minted no publication token".into()))?;
        let renew_after = Duration::from_secs(remaining_life(&token) / 2);
        Ok((token, Instant::now() + renew_after))
    }
}

impl transport::CredentialProvider for Publishing {
    fn headers(&self, host: &str) -> Vec<(String, String)> {
        if !host.eq_ignore_ascii_case(&self.catalog.host) {
            return vec![];
        }
        let mut held = self.bearer.lock().unwrap();
        if held
            .as_ref()
            .is_none_or(|(_, renew)| Instant::now() >= *renew)
        {
            match self.renew() {
                Ok(fresh) => *held = Some(fresh),
                // The Hub refuses the stale bearer and the publication fails with its answer.
                Err(Refusal(why)) => eprintln!("{why}"),
            }
        }
        let mut headers = vec![("x-cozy-execution-access".into(), self.sender.clone())];
        if let Some((token, _)) = held.as_ref() {
            headers.push(("authorization".into(), format!("Bearer {token}")));
        }
        headers
    }
}

/// Seconds until a JWT's `exp`, from its own claims; 0 when it names none.
fn remaining_life(token: &str) -> u64 {
    let exp = token
        .split('.')
        .nth(1)
        .and_then(|claims| URL_SAFE_NO_PAD.decode(claims).ok())
        .and_then(|claims| serde_json::from_slice::<Value>(&claims).ok())
        .and_then(|claims| claims["exp"].as_u64())
        .unwrap_or(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    exp.saturating_sub(now)
}

/// Path segment escaping for catalog names and refs.
pub fn escape(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
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
        format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(payload),
            URL_SAFE_NO_PAD.encode(key.sign(&signed).to_bytes())
        )
    }

    #[test]
    fn capability_admits_only_its_signer_worker_action_and_lifetime() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let other = SigningKey::from_bytes(&[8; 32]);
        let keys = [key.verifying_key()];
        let id = key_id(&key.verifying_key());
        let good = mint(
            &key,
            &format!(r#"{{"m":"w1","a":"hub-access","e":200,"k":"{id}"}}"#),
        );
        assert!(verify_capability(&good, "w1", &keys, 100, ACTION).is_some());
        assert!(verify_capability(&good, "w2", &keys, 100, ACTION).is_none());
        assert!(verify_capability(&good, "w1", &keys, 200, ACTION).is_none());
        assert!(verify_capability(&good, "w1", &keys, 100, "runtime-update").is_none());
        assert!(verify_capability(&good, "w1", &[other.verifying_key()], 100, ACTION).is_none());
        let forged = mint(
            &other,
            &format!(r#"{{"m":"w1","a":"hub-access","e":200,"k":"{id}"}}"#),
        );
        assert!(verify_capability(&forged, "w1", &keys, 100, ACTION).is_none());
        let unknown = mint(
            &key,
            &format!(r#"{{"m":"w1","a":"hub-access","e":200,"k":"{id}","z":1}}"#),
        );
        assert!(verify_capability(&unknown, "w1", &keys, 100, ACTION).is_none());
        let run = mint(&key, &format!(r#"{{"m":"w1","r":"7","e":200,"k":"{id}"}}"#));
        assert!(verify_capability(&run, "w1", &keys, 100, ACTION).is_none());
    }

    #[test]
    fn a_rental_reads_its_hub_as_the_pod_and_delegated_access_as_a_bearer() {
        use tensorfs_core::transport::CredentialProvider;
        let pod = Catalog::new(&Source::pod(
            "https://hub.example/",
            "wrk-1",
            "tok",
            None,
            vec!["objects.example".into()],
        ))
        .unwrap();
        assert_eq!(pod.origin(), "https://hub.example");
        assert_eq!(
            pod.credential().headers("hub.example"),
            [
                ("x-cozy-worker-id".to_string(), "wrk-1".to_string()),
                ("x-cozy-worker-token".to_string(), "tok".to_string())
            ]
        );
        assert!(
            pod.credential().headers("objects.example").is_empty(),
            "presigned hosts never see the credential"
        );
        assert!(pod.policy().allows_host("objects.example"));
        let access = Access {
            origin: "https://hub.example".into(),
            token: "bearer-token".into(),
            expires_at: 1,
            environment: BTreeMap::from([(
                "TENSORHUB_OBJECT_STORAGE_HOSTS".into(),
                "objects.example".into(),
            )]),
            ca: String::new(),
        };
        let delegated = Catalog::new(&Source::delegated(&access)).unwrap();
        assert_eq!(
            delegated.credential().headers("hub.example"),
            [(
                "authorization".to_string(),
                "Bearer bearer-token".to_string()
            )]
        );
        assert!(delegated.policy().allows_host("objects.example"));
    }

    #[test]
    fn origins_compare_by_scheme_host_and_port() {
        assert_eq!(
            origin_key("https://Hub.Example/"),
            origin_key("HTTPS://hub.example:443")
        );
        assert_ne!(
            origin_key("https://hub.example"),
            origin_key("http://hub.example")
        );
        assert!(valid_origin("http://127.0.0.1:8819"));
        assert!(!valid_origin("http://hub.example"));
        assert!(!valid_origin("https://hub.example/path"));
        assert!(!valid_origin("https://user@hub.example"));
    }

    #[test]
    fn principal_is_issuer_and_delegated_subject() {
        let part = |v: &str| URL_SAFE_NO_PAD.encode(v);
        let token = |sub: &str| {
            format!(
                "{}.{}.sig",
                part(r#"{"typ":"delegated-access+jwt"}"#),
                part(&format!(
                    r#"{{"iss":"https://h","delegated_sub":"{sub}","permissions":["cozy.execution-access"],"n":1}}"#
                ))
            )
        };
        assert_eq!(principal(&token("a")), principal(&token("a")));
        assert_ne!(principal(&token("a")), principal(&token("b")));
        assert!(principal("opaque-token").starts_with("opaque:"));
    }
}
