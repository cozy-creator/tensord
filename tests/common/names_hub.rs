//! A loopback Hub that answers by name (th-245): one package's card, release, locked
//! requirements and file door, and one model's closure by name, release and lane (its hash only
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

/// One published package release: its interface and its one wheel.
pub struct Package {
    pub name: String,
    pub release: String,
    pub interface: Value,
    pub wheel: String,
    pub bytes: Vec<u8>,
}

struct Served {
    model: Model,
    objects: Vec<ObjectRef>,
    bytes: HashMap<String, Vec<u8>>,
    package: Option<Package>,
    origin: String,
    heard: Arc<Mutex<Vec<String>>>,
}

/// What a request was heard as: `METHOD path`, its closure ref and lane, and `+credential`
/// when it carried one.
pub type Heard = Arc<Mutex<Vec<String>>>;

pub fn serve(store: &Store, model: Model, package: Option<Package>) -> (String, Heard) {
    let document = store.read_manifest(&model.manifest).unwrap();
    let walked = tensorfs_core::checkpoint::walk_cozytensors(store, &document).unwrap();
    let objects: Vec<ObjectRef> =
        walked.distinct().into_iter().filter(|o| o.sha256 != model.manifest.sha256).cloned().collect();
    let mut bytes = HashMap::new();
    for object in &objects {
        bytes.insert(object.sha256.clone(), std::fs::read(store.blob_path(&object.sha256)).unwrap());
    }
    bytes.insert(model.manifest.sha256.clone(), std::fs::read(store.manifest_path(&model.manifest.sha256)).unwrap());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let heard: Heard = Default::default();
    let served = Arc::new(Served {
        model,
        objects,
        bytes,
        package,
        origin: format!("http://127.0.0.1:{port}"),
        heard: heard.clone(),
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
    let model = &served.model;
    let package = served.package.as_ref();
    let release_path = package.map(|p| format!("/v1/packages/{}/releases/{}", p.name, p.release)).unwrap_or_default();
    match (method, path.as_str()) {
        (Method::GET, p) if package.is_some_and(|package| p == format!("/v1/packages/{}", package.name)) => {
            let package = package.unwrap();
            json(json!({"package": package.name, "releases": [
                {"release": package.release}, {"release": "0.9.0"}, {"release": "9.0.0", "yanked": true}]}))
        }
        (Method::GET, p) if p == release_path => {
            let package = package.unwrap();
            json(json!({"release": {"release": package.release}, "package_interface": package.interface,
                "python_version": "3.12"}))
        }
        (Method::GET, p) if p == format!("{release_path}/locked-requirements") => {
            let package = package.unwrap();
            let sha = tensorfs_core::sha256::hex_digest(&package.bytes);
            let (_, distribution) = package.name.split_once('/').unwrap();
            let lock = format!("{distribution} @ {}/v1/index/files/{sha}/{} --hash=sha256:{sha}\n", served.origin, package.wheel);
            (StatusCode::OK, lock).into_response()
        }
        (Method::GET, p) if package.is_some_and(|package| p.ends_with(&format!("/{}", package.wheel))) => {
            ranged(&package.unwrap().bytes, &headers)
        }
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
            let named = repository == model.repository
                && (release.is_empty() || release == model.release)
                && (lane.is_empty() || lane == model.lane)
                && (digest.is_empty() || digest == model.manifest.id());
            // A hash alone is the owner's: answered only to a credentialed caller.
            if !named || release.is_empty() && !digest.is_empty() && !credentialed {
                return absent();
            }
            let rows: Vec<_> = served.objects.iter().map(|o| json!({"length": o.length, "sha256": o.sha256})).collect();
            let (release, lane) = match release.is_empty() && !digest.is_empty() {
                true => ("", ""),
                false => (model.release.as_str(), model.lane.as_str()),
            };
            json(json!({"complete": true, "lane": lane, "manifest": {"length": model.manifest.length,
                "sha256": model.manifest.sha256}, "model": model.repository, "objects": rows,
                "presign_max_digests": 64, "release": release, "scope": "runtime"}))
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
            Some(bytes) => ranged(bytes, &headers),
            None => absent(),
        },
        _ => absent(),
    }
}
