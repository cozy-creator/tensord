//! A rental's own end through its provider, for when the Hub cannot hear its idle release (RunPod
//! `RUNPOD_POD_ID` and `RUNPOD_API_KEY`, vast.ai `CONTAINER_ID` and `CONTAINER_API_KEY`).
//!
//! RunPod's key is account-wide (2026-10-06: a pod's key listed another pod), so the machine keeps
//! it to itself: at boot it removes the key from the files a provider copies the environment into
//! and re-executes itself without it, carrying it on an inherited pipe. Nothing it starts, and no
//! reader of `/proc/<pid>/environ`, sees the key afterwards. Root code that reads this process's
//! memory still could; executors running as another user close that.
use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use nix::fcntl::OFlag;
use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read, Write},
    os::fd::FromRawFd,
    path::Path,
    sync::Arc,
    time::Duration,
};
use tokio_rustls::rustls::{self, pki_types::ServerName, RootCertStore};

/// The provider API's location, when not the provider's own (a stand-in provider). Plain HTTP is
/// accepted for a loopback stand-in only.
pub const API_ORIGIN: &str = "COZY_PROVIDER_API_ORIGIN";
const CALL_BUDGET: Duration = Duration::from_secs(30);
/// The fd a re-executed machine, or an activated service, reads the credential from: open only
/// when its parent placed it there.
pub const CREDENTIAL_FD: i32 = 5;
/// The provider keys a pod's environment carries.
const KEYS: [&str; 2] = ["RUNPOD_API_KEY", "CONTAINER_API_KEY"];
/// Files, under the machine root, a provider's tooling may copy the environment into for shells.
const COPIES: &[&str] = &[
    "etc/rp_environment",
    "etc/environment",
    "etc/profile",
    "etc/bash.bashrc",
    "root/.bashrc",
    "root/.profile",
];
const COPY_DIRS: &[&str] = &["etc/profile.d"];

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProviderSelf {
    RunPod {
        pod: String,
        key: String,
        origin: String,
    },
    Vast {
        instance: String,
        key: String,
        origin: String,
    },
}

impl ProviderSelf {
    /// This machine's provider credential, kept by this process alone. With the key still in the
    /// environment (a pod's boot) it scrubs the key's copies under `root` and re-executes this
    /// program without it, and does not return unless that fails; a re-executed machine or an
    /// activated service reads it from CREDENTIAL_FD. Call before any thread starts.
    pub fn take(root: &Path) -> Option<Self> {
        if let Some(own) = inherited() {
            return Some(own);
        }
        let env: HashMap<String, String> = std::env::vars().collect();
        let secrets: Vec<&str> = KEYS
            .iter()
            .filter_map(|name| env.get(*name).map(String::as_str))
            .filter(|value| !value.is_empty())
            .collect();
        if secrets.is_empty() {
            return None;
        }
        let own = Self::from_env(&env);
        for copy in copies(root) {
            match scrub(&copy, &secrets) {
                Ok(true) => eprintln!(
                    "cozy-machine: removed the provider key from {}",
                    copy.display()
                ),
                Ok(false) => (),
                Err(error) => eprintln!(
                    "cozy-machine: cannot remove the provider key from {}: {error}",
                    copy.display()
                ),
            }
        }
        let error = reexec(own.as_ref());
        eprintln!("cozy-machine: cannot restart without the provider key in the environment ({error}); children still never inherit it");
        for name in KEYS {
            std::env::remove_var(name);
        }
        own
    }

    /// The pod's own provider credential, when its provider gave one.
    pub fn from_env(env: &HashMap<String, String>) -> Option<Self> {
        let get = |name: &str| env.get(name).filter(|v| !v.is_empty()).cloned();
        let origin = |default: &str| get(API_ORIGIN).unwrap_or_else(|| default.to_owned());
        if let (Some(pod), Some(key)) = (get("RUNPOD_POD_ID"), get("RUNPOD_API_KEY")) {
            return Some(Self::RunPod {
                pod,
                key,
                origin: origin("https://api.runpod.io"),
            });
        }
        if let (Some(instance), Some(key)) = (get("CONTAINER_ID"), get("CONTAINER_API_KEY")) {
            return Some(Self::Vast {
                instance,
                key,
                origin: origin("https://console.vast.ai"),
            });
        }
        None
    }

    /// Ends this pod at its provider, so billing stops.
    pub async fn end(&self) -> io::Result<()> {
        let (origin, method, path, key, body) = match self {
            Self::RunPod { pod, key, origin } => {
                let query = format!("mutation {{ podTerminate(input: {{podId: {pod:?}}}) }}");
                let body = serde_json::to_vec(&serde_json::json!({ "query": query }))?;
                (origin, Method::POST, "/graphql".to_owned(), key, body)
            }
            Self::Vast {
                instance,
                key,
                origin,
            } => (
                origin,
                Method::DELETE,
                format!("/api/v0/instances/{instance}/"),
                key,
                vec![],
            ),
        };
        let (status, answer) =
            tokio::time::timeout(CALL_BUDGET, call(origin, method, &path, key, body))
                .await
                .map_err(|_| {
                    io::Error::other("the provider did not answer within its call budget")
                })??;
        // RunPod answers GraphQL errors with HTTP 200 and an `errors` list.
        let refused = serde_json::from_slice::<serde_json::Value>(&answer)
            .ok()
            .and_then(|v| v.get("errors").cloned())
            .filter(|e| !e.is_null());
        match (status.is_success(), refused) {
            (true, None) => Ok(()),
            (_, Some(errors)) => Err(io::Error::other(format!(
                "the provider refused: {errors:.512}"
            ))),
            (false, None) => Err(io::Error::other(format!(
                "the provider answered HTTP {status}"
            ))),
        }
    }
}

/// Runs this program again in this process with the same arguments, the provider keys dropped
/// from its environment and the credential on CREDENTIAL_FD. Returns only on failure.
fn reexec(own: Option<&ProviderSelf>) -> io::Error {
    use std::os::unix::process::CommandExt;
    if own.is_some_and(|own| !hand_on(own)) {
        return io::Error::other("cannot place the credential");
    }
    let mut args = std::env::args_os();
    let mut command = std::process::Command::new("/proc/self/exe");
    if let Some(name) = args.next() {
        command.arg0(name);
    }
    command.args(args);
    for name in KEYS {
        command.env_remove(name);
    }
    command.exec()
}

fn pipe2_cloexec() -> io::Result<(std::os::fd::OwnedFd, std::os::fd::OwnedFd)> {
    nix::unistd::pipe2(OFlag::O_CLOEXEC).map_err(io::Error::from)
}

/// The credential a parent handed this process on CREDENTIAL_FD, read once (the fd is then
/// closed); None when it handed none.
fn inherited() -> Option<ProviderSelf> {
    // SAFETY: CREDENTIAL_FD is the parent's pipe, or not open (F_GETFD then fails).
    if unsafe { libc::fcntl(CREDENTIAL_FD, libc::F_GETFD) } < 0 {
        return None;
    }
    // SAFETY: the open fd is owned by nothing else in this process. The pipe was written and
    // closed before the exec, so a read never waits on it.
    let pipe = unsafe { File::from_raw_fd(CREDENTIAL_FD) };
    unsafe { libc::fcntl(CREDENTIAL_FD, libc::F_SETFL, libc::O_NONBLOCK) };
    let mut bytes = Vec::new();
    let _ = pipe.take(64 << 10).read_to_end(&mut bytes);
    serde_json::from_slice(&bytes).ok()
}

/// Places the credential on CREDENTIAL_FD for the program this process execs next. It fits a
/// pipe's buffer: written and closed before the exec.
pub fn hand_on(own: &ProviderSelf) -> bool {
    pipe2_cloexec()
        .and_then(|(read, write)| {
            File::from(write).write_all(&serde_json::to_vec(own)?)?;
            Ok(super::supervise::inherit(read, CREDENTIAL_FD))
        })
        .unwrap_or(false)
}

fn copies(root: &Path) -> Vec<std::path::PathBuf> {
    let mut found: Vec<_> = COPIES.iter().map(|copy| root.join(copy)).collect();
    for dir in COPY_DIRS {
        if let Ok(entries) = std::fs::read_dir(root.join(dir)) {
            found.extend(entries.flatten().map(|entry| entry.path()));
        }
    }
    found
}

/// Drops every line of `path` that names a provider key or holds one of `secrets`; true when it
/// removed any. The file keeps its mode, and is replaced in one rename.
fn scrub(path: &Path, secrets: &[&str]) -> io::Result<bool> {
    let text = match std::fs::read(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) if error.kind() == io::ErrorKind::IsADirectory => return Ok(false),
        Err(error) => return Err(error),
    };
    let names: Vec<String> = KEYS.iter().map(|name| format!("{name}=")).collect();
    let holds = |line: &[u8], needle: &[u8]| line.windows(needle.len()).any(|w| w == needle);
    let kept: Vec<&[u8]> = text
        .split_inclusive(|b| *b == b'\n')
        .filter(|line| {
            !secrets.iter().any(|secret| holds(line, secret.as_bytes()))
                && !names.iter().any(|name| holds(line, name.as_bytes()))
        })
        .collect();
    if kept.iter().map(|line| line.len()).sum::<usize>() == text.len() {
        return Ok(false);
    }
    let mode = std::fs::metadata(path)?.permissions();
    let staged = path.with_file_name(format!(
        ".{}.cozy-scrub",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    std::fs::write(&staged, kept.concat())?;
    std::fs::set_permissions(&staged, mode)?;
    std::fs::rename(&staged, path)?;
    Ok(true)
}

async fn call(
    origin: &str,
    method: Method,
    path: &str,
    key: &str,
    body: Vec<u8>,
) -> io::Result<(StatusCode, Bytes)> {
    let (tls, authority) = match (
        origin.strip_prefix("https://"),
        origin.strip_prefix("http://"),
    ) {
        (Some(authority), _) => (true, authority),
        (None, Some(authority)) if authority.starts_with("127.0.0.1:") => (false, authority),
        _ => {
            return Err(io::Error::other(format!(
                "{API_ORIGIN} must be https, or http on 127.0.0.1"
            )))
        }
    };
    let (host, port) = match authority
        .rsplit_once(':')
        .map(|(h, p)| (h, p.parse::<u16>()))
    {
        Some((host, Ok(port))) => (host.to_owned(), port),
        _ => (authority.to_owned(), 443),
    };
    let tcp = super::net::connect(&host, port).await?;
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("host", &host)
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("user-agent", "cozy-machine")
        .body(Full::new(Bytes::from(body)))
        .map_err(io::Error::other)?;
    let response = if tls {
        let roots = RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        let config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(io::Error::other)?
        .with_root_certificates(roots)
        .with_no_client_auth();
        let name = ServerName::try_from(host.clone()).map_err(io::Error::other)?;
        let stream = tokio_rustls::TlsConnector::from(Arc::new(config))
            .connect(name, tcp)
            .await?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(io::Error::other)?;
        tokio::spawn(connection);
        sender
            .send_request(request)
            .await
            .map_err(io::Error::other)?
    } else {
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tcp))
            .await
            .map_err(io::Error::other)?;
        tokio::spawn(connection);
        sender
            .send_request(request)
            .await
            .map_err(io::Error::other)?
    };
    let status = response.status();
    let answer = Limited::new(response.into_body(), 64 << 10)
        .collect()
        .await
        .map_err(io::Error::other)?
        .to_bytes();
    Ok((status, answer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pod_knows_its_provider_credential() {
        let env = |pairs: &[(&str, &str)]| {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        assert_eq!(
            ProviderSelf::from_env(&env(&[("RUNPOD_POD_ID", "p1"), ("RUNPOD_API_KEY", "k")])),
            Some(ProviderSelf::RunPod {
                pod: "p1".into(),
                key: "k".into(),
                origin: "https://api.runpod.io".into()
            })
        );
        assert_eq!(
            ProviderSelf::from_env(&env(&[
                ("CONTAINER_ID", "9"),
                ("CONTAINER_API_KEY", "k"),
                (API_ORIGIN, "http://127.0.0.1:9")
            ])),
            Some(ProviderSelf::Vast {
                instance: "9".into(),
                key: "k".into(),
                origin: "http://127.0.0.1:9".into()
            })
        );
        assert_eq!(
            ProviderSelf::from_env(&env(&[("RUNPOD_POD_ID", "p1")])),
            None
        );
    }
}
