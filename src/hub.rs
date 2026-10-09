//! Hub access (th-241). Public content is read anonymously. A run's private operations (reading
//! one of the owner's unpublished checkpoints, publishing to the owner's repositories) are
//! named by a capability the owner's CLI signed with its device key for this machine's TLS leaf.
//! At the run's first private operation the machine trades it, once, for a DPoP-bound token
//! through AuthKit's JWT-bearer grant: an assertion its leaf signs carries the capability. The
//! token lives in memory until the capability expires; there is no refresh and no second trade.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tensorfs_core::transport::{
    self, AccessToken, Anonymous, Ask, CredentialProvider, Deadline, DpopCredential, DpopKey,
    SourcePolicy,
};

/// The AuthKit client every TensorD asserts as.
pub const CLIENT_ID: &str = "tensord";
/// The assertion claim that carries the owner's capability (AuthKit #437).
const CAPABILITY_CLAIM: &str = "capability";
const JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// A Hub or token endpoint that has not answered by then has not answered.
const CALL_SECONDS: f64 = 30.0;
/// After an unanswered trade, the next ask waits this long before asking again.
const RETRY: Duration = Duration::from_secs(15);
/// How long an assertion is good for: one token request.
const ASSERTION_SECONDS: u64 = 60;

/// A run's reasons when its capability does not carry it (th-241's typed refusals).
pub mod reason {
    /// A private operation the run's capability does not name.
    pub const REQUIRED: &str = "capability_required";
    /// The Hub refused the capability: its trade, or its token since (a revoked device key).
    pub const REFUSED: &str = "capability_refused";
    /// The capability's `exp` passed.
    pub const EXPIRED: &str = "capability_expired";
    /// The Hub narrowed the capability below this operation.
    pub const EXCEEDED: &str = "capability_exceeded";
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

/// The origin of an absolute URL: everything before its path.
fn origin_of(url: &str) -> &str {
    let after = url.find("://").map_or(0, |at| at + 3);
    url[after..]
        .find(['/', '?', '#'])
        .map_or(url, |end| &url[..after + end])
}

/// Whether `url` is a file the run's Hub publishes (its index's file door).
pub fn publishes(source: &Source, url: &str) -> bool {
    origin_key(origin_of(url)).is_some_and(|key| origin_key(&source.origin) == Some(key))
}

/// Where its published files are read: the file door answers with one redirect to its
/// object store.
pub fn files_policy(source: &Source) -> Result<SourcePolicy, Refusal> {
    Ok(SourcePolicy {
        max_redirects: 1,
        ..Catalog::new(source)?.policy
    })
}

/// An absolute URL at a valid origin, without query or fragment.
fn valid_url(url: &str) -> bool {
    valid_origin(origin_of(url)) && !url.contains(['?', '#'])
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

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// A Hub refusal: the HTTP-level detail, never a token.
#[derive(Debug, Clone)]
pub struct Refusal(pub String);

/// The machine's TLS leaf key (P-256): it proves DPoP and signs JWT-bearer assertions. A
/// capability names its thumbprint (`cnf.jkt`).
pub struct Leaf {
    dpop: Arc<DpopKey>,
    pair: EcdsaKeyPair,
    rng: SystemRandom,
    jwk: Value,
}

impl Leaf {
    /// The identity's PKCS#8 PEM (`-----BEGIN PRIVATE KEY-----`).
    pub fn from_pem(pem: &str) -> Result<Self, Refusal> {
        let unusable = |why: &str| Refusal(format!("the machine key: {why}"));
        let der = match rustls_pemfile::private_key(&mut pem.as_bytes()) {
            Ok(Some(rustls::pki_types::PrivateKeyDer::Pkcs8(der))) => der,
            _ => return Err(unusable("not a PKCS#8 PRIVATE KEY")),
        };
        let rng = SystemRandom::new();
        let pair = EcdsaKeyPair::from_pkcs8(
            &ECDSA_P256_SHA256_FIXED_SIGNING,
            der.secret_pkcs8_der(),
            &rng,
        )
        .map_err(|_| unusable("not a P-256 key"))?;
        let point = pair.public_key().as_ref();
        let jwk = json!({"kty": "EC", "crv": "P-256",
            "x": URL_SAFE_NO_PAD.encode(&point[1..33]), "y": URL_SAFE_NO_PAD.encode(&point[33..])});
        let dpop = DpopKey::from_pkcs8(der.secret_pkcs8_der()).map_err(|e| unusable(&e.detail))?;
        Ok(Self {
            dpop: Arc::new(dpop),
            pair,
            rng,
            jwk,
        })
    }

    /// RFC 7638 thumbprint: a capability's and every token's `cnf.jkt`.
    pub fn thumbprint(&self) -> String {
        self.dpop.thumbprint()
    }

    fn jti(&self) -> Result<String, Refusal> {
        let mut raw = [0u8; 24];
        self.rng
            .fill(&mut raw)
            .map_err(|_| Refusal("the system random source failed".into()))?;
        Ok(URL_SAFE_NO_PAD.encode(raw))
    }

    /// A compact ES256 JWS: `header` with `alg` set, then `claims`, raw r‖s signature.
    fn sign(&self, mut header: Value, claims: &Value) -> Result<String, Refusal> {
        header["alg"] = "ES256".into();
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(header.to_string()),
            URL_SAFE_NO_PAD.encode(claims.to_string())
        );
        let signature = self
            .pair
            .sign(&self.rng, input.as_bytes())
            .map_err(|_| Refusal("signing with the machine key failed".into()))?;
        Ok(format!("{input}.{}", URL_SAFE_NO_PAD.encode(signature.as_ref())))
    }

    /// An RFC 7523 assertion for `token_endpoint` carrying `capability`: the key travels in
    /// the header, the same as the DPoP proof beside it; `sub` is this key's thumbprint.
    fn assertion(&self, token_endpoint: &str, capability: &str) -> Result<String, Refusal> {
        let now = now_seconds();
        self.sign(
            json!({"typ": "JWT", "jwk": self.jwk}),
            &json!({"iss": CLIENT_ID, "sub": self.thumbprint(), "aud": token_endpoint,
                "iat": now, "exp": now + ASSERTION_SECONDS, "jti": self.jti()?,
                CAPABILITY_CLAIM: capability}),
        )
    }
}

/// Where a capability is traded: the Hub's resource (its `aud`) and its authorization
/// server's token endpoint, both as the CLI found them.
#[derive(Clone, Debug)]
struct Found {
    resource: String,
    token_endpoint: String,
}

#[derive(Deserialize)]
struct Issued {
    access_token: String,
    token_type: String,
    expires_in: u64,
}

enum TradeRefusal {
    /// The issuer's OAuth error (`invalid_grant`, with AuthKit's `reason`): final.
    Refused { expired: bool, why: String },
    /// No verdict (no answer, a 5xx): the same capability may be traded again.
    Unanswered(String),
}

/// One JWT-bearer token request, proved by the leaf.
fn trade(found: &Found, leaf: &Leaf, capability: &str) -> Result<Issued, TradeRefusal> {
    let endpoint = &found.token_endpoint;
    let unanswered = |why: String| TradeRefusal::Unanswered(format!("{endpoint}: {why}"));
    let host = transport::base_host(endpoint).map_err(|e| unanswered(e.detail))?;
    let policy = SourcePolicy {
        allowed_hosts: vec![host.clone()],
        allow_local: origin_key(origin_of(endpoint)).is_some_and(|key| loopback_host(&key)),
        max_redirects: 0,
        ..Default::default()
    };
    let assertion = leaf
        .assertion(endpoint, capability)
        .map_err(|refusal| unanswered(refusal.0))?;
    let proof = DpopCredential::new(leaf.dpop.clone(), (), vec![host]);
    let (status, body) = transport::form_post(
        endpoint,
        &[
            ("grant_type", JWT_BEARER),
            ("assertion", &assertion),
            ("client_id", CLIENT_ID),
            ("resource", &found.resource),
        ],
        &policy,
        &proof,
        64 << 10,
        Deadline::after_seconds(Some(CALL_SECONDS)),
    )
    .map_err(|e| unanswered(e.detail))?;
    if status == 200 {
        let issued: Issued = serde_json::from_slice(&body)
            .map_err(|_| unanswered("an unreadable token answer".into()))?;
        if !issued.token_type.eq_ignore_ascii_case("DPoP") || issued.access_token.is_empty() {
            return Err(unanswered(format!(
                "a {} token, not a DPoP-bound one",
                issued.token_type
            )));
        }
        return Ok(issued);
    }
    #[derive(Deserialize)]
    struct OAuthError {
        error: String,
        #[serde(default)]
        error_description: String,
        #[serde(default)]
        reason: String,
    }
    match serde_json::from_slice::<OAuthError>(&body) {
        Ok(refused) if (400..500).contains(&status) => Err(TradeRefusal::Refused {
            expired: refused.reason == "capability_expired",
            why: [refused.error, refused.reason, format!("{:.256}", refused.error_description)]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(": "),
        }),
        _ => Err(unanswered(format!("HTTP {status}"))),
    }
}

/// One private operation, as a capability's `authorization_details` names it.
#[derive(Clone, Copy, Debug)]
pub enum Op<'a> {
    /// Read one checkpoint of a model: `model` is `org/name`, `manifest` `sha256:<hex>`.
    Read { model: &'a str, manifest: &'a str },
    /// Publish one checkpoint to a model repository (the whole publication flow).
    Publish { model: &'a str },
}

/// The owner's capability for one run (th-241), and the one token it trades for. Child runs
/// share it. Nothing is asked of the Hub until a private operation needs it.
pub struct Capability {
    jws: String,
    expires: u64,
    ops: Vec<Value>,
    found: Found,
    leaf: Arc<Leaf>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    token: Option<Token>,
    /// Why the capability no longer serves; final.
    ended: Option<(&'static str, String)>,
    /// After an unanswered trade, none until then.
    quiet_until: Option<Instant>,
}

struct Token {
    access: String,
    expires: Instant,
}

impl Capability {
    /// `jws` as the run spec carries it, traded at `token_endpoint`. Its claims are the CLI's
    /// statement, read here only to know which operations are private, for which Hub and until
    /// when; AuthKit and the Hub verify it.
    pub fn new(jws: &str, token_endpoint: &str, leaf: Arc<Leaf>) -> Result<Self, Refusal> {
        let invalid = |why: &str| Refusal(format!("the run's capability {why}"));
        let claims = jws
            .split('.')
            .nth(1)
            .filter(|_| jws.split('.').count() == 3 && jws.len() <= 64 << 10)
            .and_then(|part| URL_SAFE_NO_PAD.decode(part).ok())
            .and_then(|raw| serde_json::from_slice::<Value>(&raw).ok())
            .ok_or_else(|| invalid("is not a compact JWS"))?;
        if claims["cnf"]["jkt"].as_str() != Some(leaf.thumbprint().as_str()) {
            return Err(invalid("names another machine key"));
        }
        let expires = claims["exp"].as_u64().ok_or_else(|| invalid("has no exp"))?;
        let resource = claims["aud"].as_str().unwrap_or_default().trim_end_matches('/');
        if !valid_origin(resource) {
            return Err(invalid("names no Hub origin"));
        }
        if !valid_url(token_endpoint) {
            return Err(invalid("comes with no token endpoint"));
        }
        let ops = claims["authorization_details"]
            .as_array()
            .filter(|ops| !ops.is_empty())
            .cloned()
            .ok_or_else(|| invalid("names no operations"))?;
        Ok(Self {
            jws: jws.to_string(),
            expires,
            ops,
            found: Found {
                resource: resource.to_string(),
                token_endpoint: token_endpoint.to_string(),
            },
            leaf,
            state: Mutex::default(),
        })
    }

    /// Whether the capability names `op`.
    pub fn names(&self, op: Op) -> bool {
        self.ops.iter().any(|named| match op {
            Op::Read { model, manifest } => {
                named["type"] == "tensorhub_model_read"
                    && named["model"] == model
                    && named["manifest"] == manifest
            }
            Op::Publish { model } => {
                named["type"] == "tensorhub_model_publish" && named["model"] == model
            }
        })
    }

    /// Why the capability no longer serves, if it does not.
    fn ended(&self) -> Option<(&'static str, String)> {
        self.state.lock().unwrap().ended.clone()
    }

    fn expired(&self) -> bool {
        now_seconds() >= self.expires
    }

    /// The one trade: a token until the capability expires, or why there is none.
    fn trade(&self, origin: &str, state: &mut State) {
        let answer = trade(&self.found, &self.leaf, &self.jws);
        match answer {
            Ok(issued) => {
                state.token = Some(Token {
                    access: issued.access_token,
                    expires: Instant::now() + Duration::from_secs(issued.expires_in),
                });
                state.quiet_until = None;
            }
            Err(TradeRefusal::Refused { expired, why }) => {
                let code = if expired { reason::EXPIRED } else { reason::REFUSED };
                let why = format!("{origin} refused the run's capability: {why}");
                eprintln!("Hub capability: {why}");
                state.ended = Some((code, why));
            }
            Err(TradeRefusal::Unanswered(why)) => {
                eprintln!("Hub capability at {origin}: {why}");
                state.quiet_until = Some(Instant::now() + RETRY);
            }
        }
    }
}

impl std::fmt::Debug for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capability").field("expires", &self.expires).finish()
    }
}

/// The capability at one Hub, as a DPoP credential's token source.
struct Tokens {
    capability: Arc<Capability>,
    origin: String,
}

impl AccessToken for Tokens {
    fn current(&self) -> Option<String> {
        let capability = &self.capability;
        let mut state = capability.state.lock().unwrap();
        if state.ended.is_some() {
            return None;
        }
        if capability.expired() {
            state.ended = Some((reason::EXPIRED, "the run's capability expired".into()));
            return None;
        }
        let now = Instant::now();
        if state.token.is_none() && state.quiet_until.is_none_or(|until| now >= until) {
            capability.trade(&self.origin, &mut state);
        }
        let token = state.token.as_ref().filter(|token| Instant::now() < token.expires);
        token.map(|token| token.access.clone())
    }

    /// The Hub refused the token (a revoked device key, an ended account): the capability is
    /// spent, and there is no second trade.
    fn refused(&self, token: &str) {
        let capability = &self.capability;
        let mut state = capability.state.lock().unwrap();
        if state.token.as_ref().is_some_and(|held| held.access == token) {
            state.token = None;
            state.ended = Some(match capability.expired() {
                true => (reason::EXPIRED, "the run's capability expired".into()),
                false => (reason::REFUSED, format!("{} refused the run's token", self.origin)),
            });
        }
    }
}

/// Where a run reads a Hub, and the owner's capability there, if the run has private work.
#[derive(Clone)]
pub struct Source {
    pub origin: String,
    pub ca_der: Option<Vec<u8>>,
    pub object_hosts: Vec<String>,
    pub capability: Option<Arc<Capability>>,
}

impl Source {
    pub fn new(
        origin: &str,
        ca_der: Option<Vec<u8>>,
        object_hosts: Vec<String>,
        capability: Option<Capability>,
    ) -> Result<Self, Refusal> {
        if !valid_origin(origin) {
            return Err(Refusal(format!("{origin} is not an HTTPS origin")));
        }
        Ok(Self {
            origin: origin.trim_end_matches('/').to_string(),
            ca_der,
            object_hosts,
            capability: capability.map(Arc::new),
        })
    }
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source").field("origin", &self.origin).finish()
    }
}

type Presented = Arc<DpopCredential<Tokens>>;

/// The capability's token presented to the Hub host only, with a fresh proof per request
/// naming the Hub's resource origin.
struct Presenter {
    capability: Arc<Capability>,
    origin: String,
    host: String,
    built: Mutex<Option<Presented>>,
    /// The Hub answered a token-bearing request 403: the operation is outside the token.
    forbidden: std::sync::atomic::AtomicBool,
}

impl Presenter {
    fn credential(&self, ask: &Ask) -> Option<Presented> {
        if !ask.url.host.eq_ignore_ascii_case(&self.host) {
            return None;
        }
        let mut built = self.built.lock().unwrap();
        if built.is_none() {
            let tokens = Tokens {
                capability: self.capability.clone(),
                origin: self.origin.clone(),
            };
            let credential = DpopCredential::new(self.capability.leaf.dpop.clone(), tokens, vec![self.host.clone()]);
            *built = Some(Arc::new(credential.proving_origin(&self.capability.found.resource)));
        }
        built.clone()
    }
}

impl CredentialProvider for Presenter {
    fn headers(&self, ask: &Ask) -> Vec<(String, String)> {
        self.credential(ask)
            .map(|credential| credential.headers(ask))
            .unwrap_or_default()
    }
    fn answered(&self, ask: &Ask, answer: &transport::Answer) -> bool {
        let Some(credential) = self.credential(ask) else {
            return false;
        };
        if answer.status == 403 && answer.presented("authorization").is_some() {
            self.forbidden.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        credential.answered(ask, answer)
    }
}

/// Reads and publications at a Hub, credentials presented only to the Hub host; presigned
/// object hosts see nothing. Anonymous unless built for an operation the capability names.
pub struct Catalog {
    origin: String,
    host: String,
    credential: Option<Presenter>,
    policy: SourcePolicy,
}

impl Catalog {
    /// Public reads: anonymous.
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
            credential: None,
            policy,
        })
    }

    /// `op` under the run's capability when it names it; a read it does not name is public.
    /// A publication it does not name is refused before the Hub hears of it. The token is
    /// traded here, so a capability the Hub refused, or that expired, ends the operation
    /// before any request goes out without it.
    pub fn for_op(source: &Source, op: Op) -> Result<Self, (&'static str, String)> {
        let mut catalog = Self::new(source).map_err(|e| ("catalog_read_failed", e.0))?;
        match &source.capability {
            Some(capability) if capability.names(op) => {
                let tokens = Tokens { capability: capability.clone(), origin: source.origin.clone() };
                if tokens.current().is_none() {
                    if let Some(ended) = capability.ended() {
                        return Err(ended);
                    }
                }
                catalog.credential = Some(Presenter {
                    capability: capability.clone(),
                    origin: source.origin.clone(),
                    host: catalog.host.clone(),
                    built: Mutex::default(),
                    forbidden: Default::default(),
                });
            }
            _ => {
                if let Op::Publish { model } = op {
                    return Err((
                        reason::REQUIRED,
                        format!("the run's capability does not let it publish to {model}"),
                    ));
                }
            }
        }
        Ok(catalog)
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn credential(&self) -> &dyn CredentialProvider {
        match &self.credential {
            Some(presenter) => presenter,
            None => &Anonymous,
        }
    }
    pub fn policy(&self) -> &SourcePolicy {
        &self.policy
    }
    /// A failed Hub call as a run's reason: the capability's typed refusal when it no longer
    /// serves or the Hub narrowed it, else `code`.
    pub fn reason(&self, code: &'static str, detail: impl std::fmt::Display) -> (&'static str, String) {
        let Some(presenter) = &self.credential else {
            return (code, detail.to_string());
        };
        if let Some(ended) = presenter.capability.ended() {
            return ended;
        }
        if presenter.forbidden.load(std::sync::atomic::Ordering::Relaxed) {
            return (reason::EXCEEDED, format!("{}: {detail}", self.origin));
        }
        (code, detail.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf() -> Arc<Leaf> {
        let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            .unwrap()
            .serialize_pem();
        Arc::new(Leaf::from_pem(&pem).unwrap())
    }

    fn signed(leaf: &Leaf, ops: Value) -> String {
        let part = |value: Value| URL_SAFE_NO_PAD.encode(value.to_string());
        format!("{}.{}.sig", part(json!({"alg": "EdDSA", "kid": "dk-1"})),
            part(json!({"sub": "alice", "aud": "https://hub.example", "cnf": {"jkt": leaf.thumbprint()},
                "exp": now_seconds() + 600, "authorization_details": ops})))
    }

    /// Public reads are anonymous; only an operation the capability names carries it, and a
    /// publication it does not name is refused before the Hub hears of it.
    #[test]
    fn only_named_operations_carry_the_capability() {
        let leaf = leaf();
        let manifest = format!("sha256:{}", "ab".repeat(32));
        let jws = signed(&leaf, json!([
            {"type": "tensorhub_model_read", "model": "alice/private", "manifest": manifest},
            {"type": "tensorhub_model_publish", "model": "alice/out"},
        ]));
        // Nothing answers here: a trade goes unanswered and the operation keeps its capability.
        const TOKENS: &str = "http://127.0.0.1:9/v1/auth/oauth2/token";
        let capability = Capability::new(&jws, TOKENS, leaf.clone()).unwrap();
        assert!(Capability::new(&jws, "", leaf.clone()).is_err(), "a capability comes with its token endpoint");
        let source = Source::new("https://hub.example/", None, vec!["objects.example".into()], Some(capability)).unwrap();
        let public = Catalog::new(&source).unwrap();
        assert_eq!((public.origin(), public.host.as_str()), ("https://hub.example", "hub.example"));
        assert!(public.policy().allows_host("objects.example") && public.credential.is_none());
        let read = |model, manifest| Catalog::for_op(&source, Op::Read { model, manifest }).unwrap();
        assert!(read("alice/private", &manifest).credential.is_some());
        let other_manifest = format!("sha256:{}", "cd".repeat(32));
        assert!(read("alice/private", &other_manifest).credential.is_none());
        assert!(read("public/model", &manifest).credential.is_none());
        let publish = |model| Catalog::for_op(&source, Op::Publish { model });
        assert!(publish("alice/out").unwrap().credential.is_some());
        assert_eq!(publish("alice/other").err().unwrap().0, reason::REQUIRED);
        let anonymous = Source::new("https://hub.example", None, vec![], None).unwrap();
        assert_eq!(Catalog::for_op(&anonymous, Op::Publish { model: "alice/out" }).err().unwrap().0, reason::REQUIRED);
        assert!(Source::new("http://hub.example", None, vec![], None).is_err());
        let other = signed(&self::leaf(), json!([{"type": "tensorhub_model_publish", "model": "alice/out"}]));
        assert!(Capability::new(&other, TOKENS, leaf.clone()).is_err(), "a capability for another machine key");
        assert!(Capability::new("not-a-jws", TOKENS, leaf).is_err());
    }

    /// An assertion verifies under the leaf's public key, which it carries; its claims are
    /// exactly what AuthKit checks.
    #[test]
    fn the_leaf_signs_what_authkit_and_the_hub_verify() {
        use ring::signature::{UnparsedPublicKey, ECDSA_P256_SHA256_FIXED};
        let leaf = leaf();
        let public = UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, leaf.pair.public_key().as_ref().to_vec());
        let open = |jws: &str| {
            let parts: Vec<_> = jws.split('.').collect();
            let signature = URL_SAFE_NO_PAD.decode(parts[2]).unwrap();
            public.verify(format!("{}.{}", parts[0], parts[1]).as_bytes(), &signature).unwrap();
            let read = |part: &str| serde_json::from_slice::<Value>(&URL_SAFE_NO_PAD.decode(part).unwrap()).unwrap();
            (read(parts[0]), read(parts[1]))
        };
        let (header, claims) = open(&leaf.assertion("https://hub.example/v1/auth/oauth2/token", "cap.jws.sig").unwrap());
        assert_eq!(header, json!({"alg": "ES256", "typ": "JWT", "jwk": leaf.jwk}));
        assert_eq!(claims["iss"], CLIENT_ID);
        assert_eq!(claims["sub"], leaf.thumbprint());
        assert_eq!(claims["aud"], "https://hub.example/v1/auth/oauth2/token");
        assert_eq!(claims["exp"].as_u64().unwrap() - claims["iat"].as_u64().unwrap(), ASSERTION_SECONDS);
        assert_eq!(claims["jti"].as_str().unwrap().len(), 32);
        assert_eq!(claims[CAPABILITY_CLAIM], "cap.jws.sig");
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
        assert_eq!(
            origin_of("https://hub.example/v1/auth"),
            "https://hub.example"
        );
        assert_eq!(origin_of("https://hub.example"), "https://hub.example");
    }
}
