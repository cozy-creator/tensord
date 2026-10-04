//! The machine's launch contract: the environment a Hub (rental) or this computer's launcher
//! (persistent) gives it. Same names and meaning as the Go agent's grant. Unknown names are
//! reported and ignored, so a newer Hub never stops an older machine from booting.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, Read},
    net::IpAddr,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

const NAMES: &[&str] = &[
    "COZY_MACHINE_LIFETIME",
    "COZY_MACHINE_ROOT",
    "COZY_TENSORFS_ROOT",
    "COZY_LISTEN_HOST",
    "COZY_WORKER_ID",
    "COZY_WORKER_AUTH_TOKEN",
    "COZY_WORKER_INTERNAL_PORT",
    "COZY_MEDIA_INTERNAL_PORT",
    "COZY_WEBRTC_INTERNAL_PORT",
    "COZY_RECORD_OWNER_AUTH_JSON",
    "COZY_AUTHORIZED_KEYS",
    "COZY_REPO_CACHE_ROOT",
    "COZY_SSH_PUBLIC_KEY",
    RECEIPT_KEY,
    RECEIPT_KEY_FILE,
    "TENSORHUB_ORIGIN",
    "TENSORHUB_CA_DER_B64URL",
    "TENSORHUB_PUBLIC_ORIGIN",
    "TENSORHUB_OBJECT_STORAGE_HOSTS",
];
pub const RECEIPT_KEY: &str = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL";
pub const RECEIPT_KEY_FILE: &str = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// A provider allocation: Hub lease of keys, readiness receipt, idle release.
    Rental,
    /// This computer's machine: no Hub lifecycle.
    Persistent,
}

/// One Hub's worker registration of this machine.
#[derive(Clone, Debug)]
pub struct HubGrant {
    pub origin: String,
    pub public_origin: Option<String>,
    pub worker_id: String,
    pub worker_token: String,
    pub ca_der: Option<Vec<u8>>,
    pub object_hosts: Vec<String>,
}

pub struct Grant {
    pub lifetime: Lifetime,
    pub layout: Layout,
    pub listen_host: IpAddr,
    pub worker_id: String,
    pub worker_port: u16,
    /// A launcher that still grants a media port reads the receipt there.
    pub media_port: Option<u16>,
    pub webrtc_port: Option<u16>,
    pub hub: Option<HubGrant>,
    pub repo_cache_root: Option<PathBuf>,
    /// A rental's boot keys until its Hub lease answers; a computer's first authorized_keys.
    pub authorized: Vec<VerifyingKey>,
    /// The one-shot readiness key; taken by the receipt and never kept elsewhere.
    pub receipt_key: Option<Vec<u8>>,
    /// WORKER_MODE=development with COZY_SSH_PUBLIC_KEY (else PUBLIC_KEY, which a provider such as
    /// vast may own): serve SSH maintenance.
    pub developer_key: Option<String>,
    pub ignored: Vec<String>,
}

/// The image's filesystem, rooted: "/" on a pod, a directory on an owned machine. The paths
/// are the Go agent's, so a root keeps its identity whichever agent boots it.
#[derive(Clone, Debug)]
pub struct Layout {
    pub root: PathBuf,
    pub bootstrap: PathBuf,
    pub state: PathBuf,
    /// The TensorFS store this machine reads and fills: the one `COZY_TENSORFS_ROOT` names (an
    /// owned computer's box store; a worker image sets it), else the root's standard
    /// `var/lib/tensorfs`.
    pub store: PathBuf,
}
impl Layout {
    pub fn new(root: &Path, store: Option<PathBuf>) -> Self {
        Self {
            root: root.into(),
            bootstrap: root.join("run/cozy/bootstrap"),
            state: root.join("var/lib/cozy/machine"),
            store: store.unwrap_or_else(|| root.join("var/lib/tensorfs")),
        }
    }
    /// This service's execution journal and package generations: outside
    /// `var/lib/cozy/machine`, whose ownership the Go agent claims when it boots the same root.
    pub fn engine(&self) -> PathBuf {
        self.root.join("var/lib/cozy/rust-machine")
    }
}

impl Grant {
    /// A grant is present when the environment names a machine at all.
    pub fn present(env: &HashMap<String, String>) -> bool {
        env.contains_key("COZY_WORKER_ID")
    }

    pub fn read(env: &HashMap<String, String>) -> io::Result<Self> {
        let get = |name: &str| env.get(name).map(String::as_str).filter(|v| !v.is_empty());
        let mut ignored: Vec<String> = env
            .keys()
            .filter(|n| {
                (n.starts_with("COZY_") || n.starts_with("TENSORHUB_"))
                    && !NAMES.contains(&n.as_str())
            })
            .cloned()
            .collect();
        ignored.sort();
        let root = PathBuf::from(get("COZY_MACHINE_ROOT").unwrap_or("/"));
        clean_absolute(&root, "COZY_MACHINE_ROOT")?;
        let store = get("COZY_TENSORFS_ROOT").map(PathBuf::from);
        if let Some(store) = &store {
            clean_absolute(store, "COZY_TENSORFS_ROOT")?;
        }
        let listen_host: IpAddr = match get("COZY_LISTEN_HOST").unwrap_or("0.0.0.0") {
            host @ ("0.0.0.0" | "127.0.0.1") => host.parse().expect("literal address"),
            _ => return Err(invalid("COZY_LISTEN_HOST must be 0.0.0.0 or 127.0.0.1")),
        };
        let worker_id = get("COZY_WORKER_ID").unwrap_or_default().to_owned();
        if worker_id.is_empty() || worker_id.len() > 256 || worker_id.trim() != worker_id {
            return Err(invalid(
                "COZY_WORKER_ID must be one identifier of at most 256 bytes",
            ));
        }
        let lifetime = match get("COZY_MACHINE_LIFETIME").unwrap_or("rental") {
            "rental" => Lifetime::Rental,
            "persistent" => Lifetime::Persistent,
            _ => {
                return Err(invalid(
                    "COZY_MACHINE_LIFETIME must be persistent or rental",
                ))
            }
        };
        let worker_port = port(
            get("COZY_WORKER_INTERNAL_PORT"),
            "COZY_WORKER_INTERNAL_PORT",
        )?
        .ok_or_else(|| invalid("COZY_WORKER_INTERNAL_PORT is required"))?;
        // A pre-machine Hub may still grant a media port; this machine serves one endpoint.
        let media_port = port(get("COZY_MEDIA_INTERNAL_PORT"), "COZY_MEDIA_INTERNAL_PORT")?;
        if media_port == Some(worker_port) {
            return Err(invalid("the worker and media ports must differ"));
        }
        let webrtc_port = port(
            get("COZY_WEBRTC_INTERNAL_PORT"),
            "COZY_WEBRTC_INTERNAL_PORT",
        )?;
        let hub = match lifetime {
            Lifetime::Persistent => None,
            Lifetime::Rental => {
                let token = get("COZY_WORKER_AUTH_TOKEN").unwrap_or_default();
                if URL_SAFE_NO_PAD.decode(token).map(|t| t.len()) != Ok(32) {
                    return Err(invalid(
                        "COZY_WORKER_AUTH_TOKEN must be 32 bytes of unpadded base64url",
                    ));
                }
                let ca_der = match get("TENSORHUB_CA_DER_B64URL") {
                    None => None,
                    Some(text) => Some(URL_SAFE_NO_PAD.decode(text).map_err(|_| {
                        invalid("TENSORHUB_CA_DER_B64URL must be unpadded base64url DER")
                    })?),
                };
                Some(HubGrant {
                    origin: origin(get("TENSORHUB_ORIGIN"), "TENSORHUB_ORIGIN")?
                        .ok_or_else(|| invalid("TENSORHUB_ORIGIN is required for a rental"))?,
                    public_origin: origin(
                        get("TENSORHUB_PUBLIC_ORIGIN"),
                        "TENSORHUB_PUBLIC_ORIGIN",
                    )?,
                    worker_id: worker_id.clone(),
                    worker_token: token.into(),
                    ca_der,
                    object_hosts: get("TENSORHUB_OBJECT_STORAGE_HOSTS")
                        .map(|h| h.split(',').map(str::to_owned).collect())
                        .unwrap_or_default(),
                })
            }
        };
        let mut authorized = Vec::new();
        if let Some(text) = get("COZY_RECORD_OWNER_AUTH_JSON") {
            #[derive(serde::Deserialize)]
            struct OwnerAuth {
                control_public_key_ed25519_b64url: String,
            }
            let auth: OwnerAuth = serde_json::from_str(text).map_err(|e| {
                invalid(&format!(
                    "COZY_RECORD_OWNER_AUTH_JSON is not an auth document: {e}"
                ))
            })?;
            authorized.push(public_key(&auth.control_public_key_ed25519_b64url)?);
        }
        if let Some(text) = get("COZY_AUTHORIZED_KEYS") {
            authorized = text.split(',').map(public_key).collect::<io::Result<_>>()?;
        }
        // A rental's boot keys hold until its Hub lease answers (none: nobody until then). A
        // computer's machine admits its root's authorized_keys, which these keys start when absent.
        let receipt_key = match (get(RECEIPT_KEY), get(RECEIPT_KEY_FILE)) {
            (Some(_), Some(_)) => {
                return Err(invalid("specify only one bootstrap receipt key source"))
            }
            (Some(text), None) => Some(text.trim().to_owned()),
            (None, Some(path)) => Some(private_file(Path::new(path))?),
            (None, None) => None,
        };
        let receipt_key = match receipt_key {
            None => None,
            Some(text) => match URL_SAFE_NO_PAD.decode(&text) {
                Ok(key) if key.len() == 32 => Some(key),
                _ => {
                    return Err(invalid(
                        "bootstrap receipt key must be 32 bytes of unpadded base64url",
                    ))
                }
            },
        };
        let developer_key = (env.get("WORKER_MODE").map(String::as_str) == Some("development"))
            .then(|| {
                get("COZY_SSH_PUBLIC_KEY")
                    .or_else(|| get("PUBLIC_KEY"))
                    .map(str::to_owned)
            })
            .flatten();
        Ok(Self {
            lifetime,
            layout: Layout::new(&root, store),
            listen_host,
            worker_id,
            worker_port,
            media_port,
            webrtc_port,
            hub,
            repo_cache_root: get("COZY_REPO_CACHE_ROOT").map(PathBuf::from),
            authorized,
            receipt_key,
            developer_key,
            ignored,
        })
    }
}

/// Reads the grant from this process's environment and removes the one-shot key from it.
/// Call before any thread starts: no child may inherit the readiness key.
pub fn from_process() -> io::Result<Option<Grant>> {
    let env: HashMap<String, String> = std::env::vars().collect();
    if !Grant::present(&env) {
        return Ok(None);
    }
    for name in [RECEIPT_KEY, RECEIPT_KEY_FILE] {
        std::env::remove_var(name);
    }
    Grant::read(&env).map(Some)
}

fn public_key(spelled: &str) -> io::Result<VerifyingKey> {
    let raw: [u8; 32] = URL_SAFE_NO_PAD
        .decode(spelled)
        .ok()
        .and_then(|raw| raw.try_into().ok())
        .ok_or_else(|| {
            invalid(&format!(
                "{spelled:?} is not a 32-byte unpadded base64url Ed25519 key"
            ))
        })?;
    VerifyingKey::from_bytes(&raw).map_err(|e| invalid(&e.to_string()))
}

fn port(text: Option<&str>, name: &str) -> io::Result<Option<u16>> {
    match text {
        None => Ok(None),
        Some(text) => match text.parse::<u16>() {
            Ok(n) if n > 0 && n.to_string() == text => Ok(Some(n)),
            _ => Err(invalid(&format!("{name} must be one decimal TCP port"))),
        },
    }
}

fn origin(text: Option<&str>, name: &str) -> io::Result<Option<String>> {
    let Some(text) = text else { return Ok(None) };
    match text.strip_prefix("https://") {
        Some(rest) if !rest.is_empty() && !rest.contains(['/', '?', '#', '@']) => {
            Ok(Some(text.to_owned()))
        }
        _ => Err(invalid(&format!("{name} must be https://host[:port]"))),
    }
}

fn clean_absolute(path: &Path, name: &str) -> io::Result<()> {
    let clean: PathBuf = path.components().collect();
    if !path.is_absolute()
        || clean != path
        || path
            .to_str()
            .is_some_and(|p| p.len() > 1 && p.ends_with('/'))
    {
        return Err(invalid(&format!(
            "{name} must be one clean absolute directory"
        )));
    }
    Ok(())
}

fn private_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    if !path.is_absolute()
        || !meta.is_file()
        || meta.mode() & 0o077 != 0
        || fs::symlink_metadata(path)?.file_type().is_symlink()
    {
        return Err(invalid(
            "the receipt key file must be one private regular file",
        ));
    }
    let mut text = String::new();
    file.by_ref().take(256).read_to_string(&mut text)?;
    Ok(text.trim().to_owned())
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }
    const TOKEN: &str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    const KEY: &str = "O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik";

    #[test]
    fn rental_grant_matches_the_hub_environment() {
        let grant = Grant::read(&env(&[
            ("COZY_WORKER_ID", "ra-1"),
            ("COZY_WORKER_AUTH_TOKEN", TOKEN),
            ("COZY_WORKER_INTERNAL_PORT", "8443"),
            ("COZY_WEBRTC_INTERNAL_PORT", "8445"),
            ("COZY_AUTHORIZED_KEYS", KEY),
            (RECEIPT_KEY, TOKEN),
            ("TENSORHUB_ORIGIN", "https://hub.example"),
            ("COZY_FUTURE_SETTING", "x"),
            ("WORKER_MODE", "development"),
            ("PUBLIC_KEY", "ssh-ed25519 AAAA"),
        ]))
        .unwrap();
        assert_eq!(grant.lifetime, Lifetime::Rental);
        assert_eq!(grant.layout.state, Path::new("/var/lib/cozy/machine"));
        assert_eq!((grant.worker_port, grant.webrtc_port), (8443, Some(8445)));
        assert_eq!(grant.receipt_key.as_deref(), Some(&[0u8; 32][..]));
        assert_eq!(grant.hub.unwrap().origin, "https://hub.example");
        assert_eq!(grant.ignored, ["COZY_FUTURE_SETTING"]);
        assert_eq!(grant.developer_key.as_deref(), Some("ssh-ed25519 AAAA"));
    }

    #[test]
    fn the_cozy_ssh_key_wins_over_a_providers_public_key() {
        let grant = Grant::read(&env(&[
            ("COZY_WORKER_ID", "ra-1"),
            ("COZY_WORKER_AUTH_TOKEN", TOKEN),
            ("COZY_WORKER_INTERNAL_PORT", "8443"),
            ("COZY_AUTHORIZED_KEYS", KEY),
            ("TENSORHUB_ORIGIN", "https://hub.example"),
            ("WORKER_MODE", "development"),
            ("PUBLIC_KEY", "ssh-ed25519 PROVIDER"),
            ("COZY_SSH_PUBLIC_KEY", "ssh-ed25519 RENTER"),
        ]))
        .unwrap();
        assert_eq!(grant.developer_key.as_deref(), Some("ssh-ed25519 RENTER"));
        assert!(grant.ignored.is_empty());
    }

    #[test]
    fn malformed_grants_refuse_by_name() {
        let base = [
            ("COZY_WORKER_ID", "ra-1"),
            ("COZY_WORKER_INTERNAL_PORT", "8443"),
            ("COZY_MACHINE_LIFETIME", "persistent"),
            ("COZY_AUTHORIZED_KEYS", KEY),
        ];
        assert!(Grant::read(&env(&base)).is_ok());
        for (name, value) in [
            ("COZY_WORKER_INTERNAL_PORT", "08443"),
            ("COZY_MEDIA_INTERNAL_PORT", "8443"),
            ("COZY_MACHINE_ROOT", "relative"),
            ("COZY_LISTEN_HOST", "::"),
            ("COZY_AUTHORIZED_KEYS", "short"),
        ] {
            let mut pairs = env(&base);
            pairs.insert(name.into(), value.into());
            assert!(Grant::read(&pairs).is_err(), "{name}={value}");
        }
        let mut rental = env(&base);
        rental.insert("COZY_MACHINE_LIFETIME".into(), "rental".into());
        assert!(
            Grant::read(&rental).is_err(),
            "a rental needs its worker token and Hub"
        );
    }
}
