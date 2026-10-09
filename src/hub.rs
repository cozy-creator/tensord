//! Hub access (th-238). A signer's authority at a Hub arrives as an AuthKit authorization code
//! for client `cozy-machine`, approved by the signer's CLI and bound to this machine's TLS leaf
//! (`dpop_jkt`). The machine redeems it at once with a DPoP proof from that key and holds the
//! grant in memory only: refresh tokens rotate and never touch disk, and every Hub request
//! carries the access token beside a fresh proof. A rental also reads its own Hub with its
//! worker capability, and presents it beside a publication grant.
use base64::Engine as _;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tensorfs_core::transport::{
    self, AccessToken, Both, CredentialProvider, Deadline, DpopCredential, DpopKey, Ledger,
    ScopedHeaders, SourcePolicy,
};

pub const CLIENT_ID: &str = "cozy-machine";
/// `authorization_details` types: the owner's catalog reads, and writes to named repositories.
pub const EXECUTION: &str = "tensorhub_execution";
pub const PUBLICATION: &str = "tensorhub_machine_publication";
/// A token endpoint that has not answered by then has not answered.
const TOKEN_CALL_SECONDS: f64 = 30.0;
/// After an unanswered refresh, the next ask waits this long before asking again.
const REFRESH_RETRY: Duration = Duration::from_secs(15);

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

/// A Hub refusal: the HTTP-level detail, never a token.
#[derive(Debug, Clone)]
pub struct Refusal(pub String);

/// An authorization code the signer's CLI handed over (`HubAuthorization`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Authorization {
    pub issuer: String,
    pub code: String,
    pub code_verifier: String,
    pub redirect_uri: String,
    pub resource: String,
}

/// A redeemed grant: DPoP-bound tokens this machine holds, the access token renewed at half
/// its life. Refresh tokens rotate, so only the newest is kept and none is ever sent twice.
pub struct Grant {
    key: Arc<DpopKey>,
    issuer: String,
    resource: String,
    tokens: Mutex<Tokens>,
}

struct Tokens {
    access: String,
    renew: Instant,
    expires: Instant,
    refresh: Option<String>,
    /// Why the grant ended: the issuer refused its refresh token.
    ended: Option<String>,
}

#[derive(Deserialize)]
struct Issued {
    access_token: String,
    token_type: String,
    expires_in: u64,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    authorization_details: Option<Value>,
}

enum TokenRefusal {
    /// The issuer's OAuth error (`invalid_grant`: the code or grant is no longer good).
    Refused {
        error: String,
        description: String,
    },
    Unanswered(String),
}

impl TokenRefusal {
    fn detail(&self) -> String {
        match self {
            Self::Refused { error, description } if description.is_empty() => error.clone(),
            Self::Refused { error, description } => format!("{error}: {description:.256}"),
            Self::Unanswered(why) => why.clone(),
        }
    }
}

impl Grant {
    /// Redeems `authorization` with this machine's key, refusing a grant of another kind.
    pub fn redeem(
        authorization: &Authorization,
        kind: &str,
        key: Arc<DpopKey>,
        ca_der: Option<&[u8]>,
    ) -> Result<Arc<Grant>, Refusal> {
        let Authorization {
            issuer,
            code,
            code_verifier,
            redirect_uri,
            resource,
        } = authorization;
        let issuer = issuer.trim_end_matches('/');
        if !valid_origin(origin_of(issuer)) || issuer.contains(['?', '#']) {
            return Err(Refusal(
                "the authorization's issuer is not an HTTPS URL".into(),
            ));
        }
        if !valid_origin(origin_of(resource)) {
            return Err(Refusal(
                "the authorization's resource is not an HTTPS URL".into(),
            ));
        }
        if [code, code_verifier, redirect_uri]
            .iter()
            .any(|v| v.is_empty() || v.len() > 4096)
        {
            return Err(Refusal("the authorization is incomplete".into()));
        }
        if let Some(der) = ca_der {
            transport::trust_roots(pem(der).as_bytes()).map_err(|e| Refusal(e.to_string()))?;
        }
        let issued = token_call(
            issuer,
            &key,
            &[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", redirect_uri),
                ("code_verifier", code_verifier),
                ("client_id", CLIENT_ID),
                ("resource", resource),
            ],
        )
        .map_err(|refusal| {
            Refusal(format!(
                "the authorization did not redeem: {}",
                refusal.detail()
            ))
        })?;
        let granted = issued
            .authorization_details
            .as_ref()
            .and_then(|details| details[0]["type"].as_str());
        if granted != Some(kind) {
            return Err(Refusal(format!(
                "the authorization grants {}, not {kind}",
                granted.unwrap_or("nothing named")
            )));
        }
        Ok(Arc::new(Grant {
            key,
            issuer: issuer.to_string(),
            resource: resource.trim_end_matches('/').to_string(),
            tokens: Mutex::new(Tokens::from(issued)),
        }))
    }

    /// The resource the grant's tokens are for: proofs name its origin.
    pub fn resource(&self) -> &str {
        &self.resource
    }

    /// Why the grant ended, or None while it lives.
    pub fn ended(&self) -> Option<String> {
        let tokens = self.tokens.lock().unwrap();
        match (&tokens.ended, &tokens.refresh) {
            (Some(why), _) => Some(why.clone()),
            (None, None) if Instant::now() >= tokens.expires => {
                Some("its access token expired".into())
            }
            _ => None,
        }
    }

    /// Presents this grant to `host` with a fresh proof per request.
    pub fn presenter(self: &Arc<Self>, host: &str) -> DpopCredential<Arc<Grant>> {
        DpopCredential::new(self.key.clone(), self.clone(), vec![host.to_string()])
            .proving_origin(origin_of(&self.resource))
    }

    fn refresh(&self, tokens: &mut Tokens) {
        let Some(refresh) = tokens.refresh.clone() else {
            tokens.ended = Some("the grant has no refresh token".into());
            return;
        };
        let answer = token_call(
            &self.issuer,
            &self.key,
            &[
                ("grant_type", "refresh_token"),
                ("refresh_token", &refresh),
                ("client_id", CLIENT_ID),
            ],
        );
        match answer {
            // Rotated: the old refresh token is gone for good.
            Ok(issued) => *tokens = Tokens::from(issued),
            Err(TokenRefusal::Refused { error, description }) if error == "invalid_grant" => {
                tokens.refresh = None;
                tokens.ended = Some(format!("{error}: {description:.256}"));
            }
            // No answer is not a rotation: the same token is asked again shortly, and the
            // access token serves meanwhile. Had the issuer rotated it, reuse ends the grant.
            Err(refusal) => {
                eprintln!("Hub grant refresh: {}", refusal.detail());
                tokens.renew = Instant::now() + REFRESH_RETRY;
            }
        }
    }
}

impl From<Issued> for Tokens {
    fn from(issued: Issued) -> Self {
        let now = Instant::now();
        let life = Duration::from_secs(issued.expires_in);
        Tokens {
            access: issued.access_token,
            renew: now + life / 2,
            expires: now + life,
            refresh: issued.refresh_token,
            ended: None,
        }
    }
}

impl AccessToken for Grant {
    fn current(&self) -> Option<String> {
        let mut tokens = self.tokens.lock().unwrap();
        if tokens.ended.is_none() && Instant::now() >= tokens.renew && tokens.refresh.is_some() {
            self.refresh(&mut tokens);
        }
        (tokens.ended.is_none() && Instant::now() < tokens.expires).then(|| tokens.access.clone())
    }

    fn refused(&self, token: &str) {
        let mut tokens = self.tokens.lock().unwrap();
        if tokens.access != token || tokens.ended.is_some() {
            return;
        }
        tokens.expires = Instant::now();
        if tokens.refresh.is_some() {
            self.refresh(&mut tokens);
        }
    }
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grant")
            .field("issuer", &self.issuer)
            .field("resource", &self.resource)
            .finish()
    }
}

/// The execution grants this machine holds, per signer and Hub origin, and the key that
/// redeems and proves every grant (the leaf's). Memory only: a restart forgets them, and the
/// signer's CLI grants again when a run is refused `hub_access_required`.
#[derive(Default)]
pub struct Grants {
    key: Option<Arc<DpopKey>>,
    held: Mutex<HashMap<(String, String), Arc<Grant>>>,
}

impl Grants {
    pub fn new(key: Arc<DpopKey>) -> Self {
        Self {
            key: Some(key),
            held: Mutex::default(),
        }
    }

    pub fn redeem(
        &self,
        authorization: &Authorization,
        kind: &str,
        ca_der: Option<&[u8]>,
    ) -> Result<Arc<Grant>, Refusal> {
        let key = self
            .key
            .clone()
            .ok_or_else(|| Refusal("this machine holds no key to prove a grant with".into()))?;
        Grant::redeem(authorization, kind, key, ca_der)
    }

    /// `actor`'s execution grant at `origin` from now on.
    pub fn hold(&self, actor: &str, origin: &str, grant: Arc<Grant>) {
        let key = (actor.to_string(), origin_key(origin).unwrap_or_default());
        self.held.lock().unwrap().insert(key, grant);
    }

    /// `actor`'s live execution grant at `origin`; an ended one is forgotten.
    pub fn held(&self, actor: &str, origin: &str) -> Option<Arc<Grant>> {
        let key = (actor.to_string(), origin_key(origin).unwrap_or_default());
        let mut held = self.held.lock().unwrap();
        let grant = held.get(&key)?.clone();
        if grant.ended().is_some() {
            held.remove(&key);
            return None;
        }
        Some(grant)
    }
}

/// One form POST to `issuer`'s token endpoint, proving `key`.
fn token_call(
    issuer: &str,
    key: &Arc<DpopKey>,
    form: &[(&str, &str)],
) -> Result<Issued, TokenRefusal> {
    let endpoint = format!("{issuer}/oauth2/token");
    let unanswered = |why: String| TokenRefusal::Unanswered(format!("{endpoint}: {why}"));
    let host = transport::base_host(&endpoint).map_err(|e| unanswered(e.detail))?;
    let policy = SourcePolicy {
        allowed_hosts: vec![host.clone()],
        allow_local: origin_key(origin_of(issuer)).is_some_and(|key| loopback_host(&key)),
        max_redirects: 0,
        ..Default::default()
    };
    let proof = DpopCredential::new(key.clone(), (), vec![host]);
    let (status, body) = transport::form_post(
        &endpoint,
        form,
        &policy,
        &proof,
        64 << 10,
        Deadline::after_seconds(Some(TOKEN_CALL_SECONDS)),
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
    }
    match serde_json::from_slice::<OAuthError>(&body) {
        Ok(refused) if (400..500).contains(&status) => Err(TokenRefusal::Refused {
            error: refused.error,
            description: refused.error_description,
        }),
        _ => Err(unanswered(format!("HTTP {status}"))),
    }
}

/// How a machine reads a Hub: as the signer's grant, or on a rental as the pod itself.
#[derive(Clone, Debug)]
pub enum Credential {
    Grant(Arc<Grant>),
    Worker { id: String, token: String },
}

/// Where and as whom a machine reads a Hub. On a rental, a run that names no Hub reads the
/// rental's own with the pod's worker capability (the Go agent and Python worker's default).
#[derive(Clone, Debug)]
pub struct Source {
    pub origin: String,
    pub credential: Credential,
    pub ca_der: Option<Vec<u8>>,
    pub object_hosts: Vec<String>,
}

impl Source {
    pub fn granted(
        origin: &str,
        grant: Arc<Grant>,
        ca_der: Option<Vec<u8>>,
        object_hosts: Vec<String>,
    ) -> Self {
        Self {
            origin: origin.trim_end_matches('/').to_string(),
            credential: Credential::Grant(grant),
            ca_der,
            object_hosts,
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
            credential: Credential::Worker {
                id: worker_id.into(),
                token: worker_token.into(),
            },
            ca_der,
            object_hosts,
        }
    }
    fn worker(&self, host: &str) -> ScopedHeaders {
        let headers = match &self.credential {
            Credential::Worker { id, token } => vec![
                ("x-cozy-worker-id".to_string(), id.clone()),
                ("x-cozy-worker-token".to_string(), token.clone()),
            ],
            Credential::Grant(_) => vec![],
        };
        ScopedHeaders {
            hosts: vec![host.to_string()],
            headers,
        }
    }
}

type Presenter = Box<dyn CredentialProvider + Send + Sync>;

/// Reads at a Hub with its credential, presented only to the Hub host.
pub struct Catalog {
    origin: String,
    host: String,
    credential: Presenter,
    policy: SourcePolicy,
}

impl Catalog {
    pub fn new(source: &Source) -> Result<Self, Refusal> {
        Self::presenting(source, |host| match &source.credential {
            Credential::Grant(grant) => Box::new(grant.presenter(host)),
            Credential::Worker { .. } => Box::new(source.worker(host)),
        })
    }
    fn presenting(
        source: &Source,
        credential: impl FnOnce(&str) -> Presenter,
    ) -> Result<Self, Refusal> {
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
            credential: credential(&host),
            host,
            policy,
        })
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn credential(&self) -> &dyn CredentialProvider {
        &*self.credential
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
            self.credential(),
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

/// Hub writes under a machine-publication grant, presented to the Hub host only; presigned
/// object hosts see nothing. On a rental the pod's worker capability rides beside it: the Hub
/// checks that its leaf is the grant's key.
pub struct Publishing {
    catalog: Catalog,
}

impl Publishing {
    /// `source` is where the run reads; `rental` the pod's own Hub access, if this is a rental.
    pub fn new(
        source: &Source,
        grant: &Arc<Grant>,
        rental: Option<&Source>,
    ) -> Result<Self, Refusal> {
        if let Some(why) = grant.ended() {
            return Err(Refusal(format!("the publication grant ended: {why}")));
        }
        let worker = rental.filter(|pod| origin_key(&pod.origin) == origin_key(&source.origin));
        let catalog = Catalog::presenting(source, |host| {
            let worker = worker.map_or_else(|| source.worker(host), |pod| pod.worker(host));
            Box::new(Both(grant.presenter(host), worker))
        })?;
        Ok(Self { catalog })
    }
    pub fn origin(&self) -> &str {
        self.catalog.origin()
    }
    pub fn policy(&self) -> &SourcePolicy {
        self.catalog.policy()
    }
}

impl CredentialProvider for Publishing {
    fn headers(&self, ask: &transport::Ask) -> Vec<(String, String)> {
        self.catalog.credential.headers(ask)
    }
    fn answered(&self, ask: &transport::Ask, answer: &transport::Answer) -> bool {
        self.catalog.credential.answered(ask, answer)
    }
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

    #[test]
    fn a_rental_reads_its_hub_as_the_pod() {
        let pod = Catalog::new(&Source::pod(
            "https://hub.example/",
            "wrk-1",
            "tok",
            None,
            vec!["objects.example".into()],
        ))
        .unwrap();
        assert_eq!(pod.origin(), "https://hub.example");
        assert_eq!(pod.host, "hub.example");
        assert!(pod.policy().allows_host("objects.example"));
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
