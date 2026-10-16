//! A loopback Hub that answers by name (th-245): one package's card, each release's card, locked
//! requirements and file door, and its models' closures by name, release and lane (a hash only
//! to a credentialed caller), presign and objects, with ranges. It records every request.
#![allow(dead_code)]
use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tensorfs_core::{ids::ObjectRef, store::Store};

/// One released model: `repository` at `release` in `lane`, held in a source store.
pub struct Model {
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub manifest: ObjectRef,
}

/// One published package release: its interface, its wheel and the wheels of the other
/// packages its lock pins at this Hub (`(wheel, bytes)`, published by the same org).
pub struct Package {
    pub name: String,
    pub release: String,
    pub interface: Value,
    pub wheel: String,
    pub bytes: Vec<u8>,
    pub callees: Vec<(String, Vec<u8>)>,
    /// The lock's other rows (PyPI's), verbatim.
    pub pypi: String,
}

struct Served {
    /// Each model with the objects its closure names.
    models: Vec<(Model, Vec<ObjectRef>)>,
    bytes: HashMap<String, Vec<u8>>,
    /// Releases of one package.
    packages: Vec<Package>,
    origin: String,
    heard: Arc<Mutex<Vec<String>>>,
    /// How long each object answer takes, as a slow link's would.
    pace: std::time::Duration,
}

/// What a request was heard as: `METHOD path`, its closure ref and lane, and `+credential`
/// when it carried one.
pub type Heard = Arc<Mutex<Vec<String>>>;

pub fn serve(store: &Store, model: Model, packages: impl IntoIterator<Item = Package>) -> (String, Heard) {
    paced(store, model, packages, std::time::Duration::ZERO)
}

/// `serve`, each object answering after `pace`.
pub fn paced(
    store: &Store,
    model: Model,
    packages: impl IntoIterator<Item = Package>,
    pace: std::time::Duration,
) -> (String, Heard) {
    start(store, vec![model], packages, pace)
}

/// Several models, or several lanes of one release.
pub fn models(store: &Store, models: Vec<Model>) -> (String, Heard) {
    start(store, models, None, std::time::Duration::ZERO)
}

fn start(
    store: &Store,
    models: Vec<Model>,
    packages: impl IntoIterator<Item = Package>,
    pace: std::time::Duration,
) -> (String, Heard) {
    let mut bytes = HashMap::new();
    let models = models.into_iter().map(|model| {
        let document = store.read_manifest(&model.manifest).unwrap();
        let walked = tensorfs_core::checkpoint::walk_cozytensors(store, &document).unwrap();
        let objects: Vec<ObjectRef> =
            walked.distinct().into_iter().filter(|o| o.sha256 != model.manifest.sha256).cloned().collect();
        for object in &objects {
            bytes.insert(object.sha256.clone(), std::fs::read(store.blob_path(&object.sha256)).unwrap());
        }
        bytes.insert(model.manifest.sha256.clone(), std::fs::read(store.manifest_path(&model.manifest.sha256)).unwrap());
        (model, objects)
    }).collect();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let heard: Heard = Default::default();
    let served = Arc::new(Served {
        models,
        bytes,
        packages: packages.into_iter().collect(),
        origin: format!("http://127.0.0.1:{port}"),
        heard: heard.clone(),
        pace,
    });
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async move {
            let router = axum::Router::new().fallback(answer).with_state(served);
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, router).await.unwrap()
        })
    });
    (format!("http://127.0.0.1:{port}"), heard)
}

/// Each wheel the release's lock pins at this Hub, as the Hub renders its row (th-245): its
/// distribution, and its file door `/v1/index/<org>/<package>/<release>/<wheel>` under the
/// release that claims it (the release itself, or a callee's own). No other door answers.
fn doors(package: &Package) -> impl Iterator<Item = (String, String, &[u8])> {
    let (org, name) = package.name.split_once('/').unwrap();
    let root = (name.to_string(), package.release.clone(), &package.wheel, &package.bytes);
    let callees = package.callees.iter().map(|(wheel, bytes)| {
        let mut parts = wheel.split('-');
        let (distribution, version) = (parts.next().unwrap().replace('_', "-"), parts.next().unwrap().to_string());
        (distribution, version, wheel, bytes)
    });
    std::iter::once(root).chain(callees).map(move |(distribution, release, wheel, bytes)| {
        let door = format!("/v1/index/{org}/{distribution}/{release}/{wheel}");
        (distribution, door, bytes.as_slice())
    })
}

fn absent() -> Response {
    (StatusCode::NOT_FOUND, r#"{"error":{"code":"tensorfs.closure_denied","message":"absent"}}"#).into_response()
}

fn json(value: Value) -> Response {
    (StatusCode::OK, serde_json::to_vec(&value).unwrap()).into_response()
}

/// `bytes`, or the range `headers` ask of them.
fn ranged(bytes: &[u8], headers: &HeaderMap) -> Response {
    let asked = headers.get("range").and_then(|v| v.to_str().ok()).and_then(|v| {
        let (first, last) = v.trim().strip_prefix("bytes=")?.split_once('-')?;
        let first: usize = first.parse().ok()?;
        let last = last.parse::<usize>().unwrap_or(bytes.len() - 1).min(bytes.len() - 1);
        Some((first, last))
    });
    match asked {
        Some((first, last)) => (
            StatusCode::PARTIAL_CONTENT,
            [("content-range", format!("bytes {first}-{last}/{}", bytes.len()))],
            bytes[first..=last].to_vec(),
        )
            .into_response(),
        None => (StatusCode::OK, bytes.to_vec()).into_response(),
    }
}

async fn answer(State(served): State<Arc<Served>>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let path = uri.path().to_string();
    // Directly, or behind AuthKit's resource server, which verified it and names its owner.
    let owner = headers.get("x-verified-owner").and_then(|v| v.to_str().ok());
    let credentialed = headers.contains_key("authorization") || owner.is_some_and(|o| o != "anonymous");
    let asked: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let mut heard = format!("{method} {path}");
    if path == "/v1/tensorfs/closure" {
        heard = format!("{heard} {} {}", asked["ref"].as_str().unwrap_or_default(), asked["lane"].as_str().unwrap_or_default());
    }
    if credentialed {
        heard.push_str(" +credential");
    }
    served.heard.lock().unwrap().push(heard);
    let first = served.packages.first();
    let release_path = |p: &Package| format!("/v1/packages/{}/releases/{}", p.name, p.release);
    let release = served.packages.iter().find(|p| path == release_path(p));
    let locked = served.packages.iter().find(|p| path == format!("{}/locked-requirements", release_path(p)));
    let door = served.packages.iter().flat_map(doors).find(|(_, door, _)| *door == path);
    match (method, path.as_str()) {
        (Method::GET, p) if first.is_some_and(|package| p == format!("/v1/packages/{}", package.name)) => {
            let mut releases: Vec<Value> = served.packages.iter().map(|p| json!({"release": p.release})).collect();
            releases.extend([json!({"release": "0.9.0"}), json!({"release": "9.0.0", "yanked": true})]);
            json(json!({"package": first.unwrap().name, "releases": releases}))
        }
        (Method::GET, _) if release.is_some() => {
            let package = release.unwrap();
            json(json!({"release": {"release": package.release}, "package_interface": package.interface,
                "python_version": "3.12"}))
        }
        (Method::GET, _) if locked.is_some() => {
            let package = locked.unwrap();
            let lock: String = doors(package)
                .map(|(distribution, door, bytes)| {
                    let sha = tensorfs_core::sha256::hex_digest(bytes);
                    format!("{distribution} @ {}{door} --hash=sha256:{sha}\n", served.origin)
                })
                .collect::<String>()
                + &package.pypi;
            (StatusCode::OK, lock).into_response()
        }
        (Method::GET, _) if door.is_some() => ranged(door.unwrap().2, &headers),
        (Method::POST, "/v1/tensorfs/closure") => {
            let refspec = asked["ref"].as_str().unwrap_or_default();
            let lane = asked["lane"].as_str().unwrap_or_default();
            let mut parts = refspec.split('@');
            let repository = parts.next().unwrap_or_default();
            let (mut release, mut digest) = ("", "");
            for part in parts {
                match part.starts_with("sha256:") {
                    true => digest = part,
                    false => release = part,
                }
            }
            let named: Vec<_> = served.models.iter().filter(|(model, _)| {
                repository == model.repository
                    && (release.is_empty() || release == model.release)
                    && (lane.is_empty() || lane == model.lane)
                    && (digest.is_empty() || digest == model.manifest.id())
            }).collect();
            // No lane names a release's only one.
            let Some((model, objects)) = named.first().filter(|_| !lane.is_empty() || named.len() == 1) else {
                return absent();
            };
            // A hash alone is the owner's: answered only to a credentialed caller.
            if release.is_empty() && !digest.is_empty() && !credentialed {
                return absent();
            }
            let rows: Vec<_> = objects.iter().map(|o| json!({"length": o.length, "sha256": o.sha256})).collect();
            let (release, lane) = match release.is_empty() && !digest.is_empty() {
                true => ("", ""),
                false => (model.release.as_str(), model.lane.as_str()),
            };
            // A release lane is public; a hash alone is the owner's.
            json(json!({"complete": true, "lane": lane, "manifest": {"length": model.manifest.length,
                "sha256": model.manifest.sha256}, "model": model.repository, "objects": rows,
                "presign_max_digests": 64, "public": !release.is_empty(), "release": release, "scope": "runtime"}))
        }
        (Method::POST, "/v1/tensorfs/presign") => {
            let mut urls = serde_json::Map::new();
            for digest in asked["digests"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                if !served.bytes.contains_key(digest) {
                    return absent();
                }
                // Object hosts are not the Hub host: no credential follows the URL there.
                let url = served.origin.replace("127.0.0.1", "localhost");
                urls.insert(digest.into(), json!(format!("{url}/o/{digest}")));
            }
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            json(json!({"expires_at_unix": now + 600, "server_time_unix": now, "urls": urls}))
        }
        (Method::GET, p) if p.starts_with("/o/") => match served.bytes.get(&p[3..]) {
            Some(bytes) => {
                tokio::time::sleep(served.pace).await;
                ranged(bytes, &headers)
            }
            None => absent(),
        },
        _ => absent(),
    }
}
