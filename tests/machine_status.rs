//! Status on the real binary in machine mode: anyone gets identity and the sealed receipt (the
//! bytes the Hub verifies); a machine cap gets the whole picture as it changes; an open stream is
//! not activity and only `keepalive` moves the idle deadline.
mod common;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use cozy_machine::api::{
    capability::{self, Grant},
    v1::{self, machine_client::MachineClient},
};
use ed25519_dalek::SigningKey;
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tonic::{
    metadata::MetadataValue,
    transport::{Certificate, Channel, ClientTlsConfig, Endpoint},
    Code, Request,
};

const WORKER: &str = "status-test";
const OWNER: [u8; 32] = [33; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn boot(root: &Path, port: u16, lifetime: &str) -> Machine {
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    Machine(
        Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("COZY_MACHINE_ROOT", root)
            .env("COZY_WORKER_ID", WORKER)
            .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
            .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
            .env("COZY_LISTEN_HOST", "127.0.0.1")
            .env(
                "COZY_AUTHORIZED_KEYS",
                URL_SAFE_NO_PAD.encode(owner.as_bytes()),
            )
            .env(
                "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
                URL_SAFE_NO_PAD.encode([7; 32]),
            )
            .env("TENSORHUB_ORIGIN", "https://hub.invalid")
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("COZY_MACHINE_LIFETIME", lifetime)
            .envs(webrtc(lifetime))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

/// A rental is granted a WebRTC port and its provider's mapping of it; this computer's
/// machine takes its own port.
fn webrtc(lifetime: &str) -> Vec<(String, String)> {
    if lifetime != "rental" {
        return vec![];
    }
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    [("COZY_WEBRTC_INTERNAL_PORT", port.to_string()), ("RUNPOD_PUBLIC_IP", "203.0.113.7".into()), (&format!("RUNPOD_TCP_PORT_{port}"), "30001".into())]
        .map(|(name, value)| (name.to_string(), value))
        .to_vec()
}

/// The sealed receipt, once the machine serves it on Status.
fn receipt(machine: &mut Machine, root: &Path, port: u16) -> Vec<u8> {
    let start = Instant::now();
    loop {
        assert!(
            machine.0.try_wait().unwrap().is_none(),
            "the machine exited"
        );
        if let Some(body) = common::receipt(root, port) {
            return body;
        }
        assert!(start.elapsed() < Duration::from_secs(300), "no receipt");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn cap(run: &str) -> String {
    let expires = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 600;
    let grant = Grant {
        machine: WORKER.into(),
        run: run.into(),
        action: if run.is_empty() {
            capability::MACHINE.into()
        } else {
            String::new()
        },
        expires,
        ..Default::default()
    };
    capability::mint(&SigningKey::from_bytes(&OWNER), grant)
}

fn status(keepalive: bool, cap: Option<&str>) -> Request<v1::StatusRequest> {
    let mut request = Request::new(v1::StatusRequest { keepalive });
    if let Some(cap) = cap {
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Cozy-Cap {cap}")).unwrap(),
        );
    }
    request
}

fn deadline_on_disk(root: &Path) -> i64 {
    let ledger: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("var/lib/cozy/machine/idle.json")).unwrap(),
    )
    .unwrap();
    ledger["deadline_ms"].as_i64().unwrap()
}

#[tokio::test]
async fn status_answers_identity_to_anyone_and_the_machine_to_its_owner() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-status")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    // The store holds two models before the machine starts; Status lists them.
    cozy_machine::held_models::fixture(&root.join("var/lib/tensorfs")).unwrap();
    let mut machine = boot(&root, port, "rental");
    let sealed = receipt(&mut machine, &root, port);
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let channel: Channel = Endpoint::from_shared(format!("https://127.0.0.1:{port}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(&pem))
                .domain_name("cozy-worker"),
        )
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = MachineClient::new(channel);
    let boot_id = std::fs::read_to_string(root.join("var/lib/cozy/machine/boot-id")).unwrap();

    // The readiness proof's observation can fail: a key this machine admits is not refused.
    let leaf = String::from_utf8(pem.clone()).unwrap();
    for (key, refused) in [(OWNER, false), ([9; 32], true)] {
        let key = SigningKey::from_bytes(&key);
        assert_eq!(
            cozy_machine::machine::probe::refuses_capability_of(port, &leaf, WORKER, &key).await,
            refused
        );
    }

    // No cap (the Hub's readiness read) and a run-scope cap: one identity frame, then the end.
    for cap in [None, Some(cap("some-run"))] {
        let mut frames = client
            .status(status(false, cap.as_deref()))
            .await
            .unwrap()
            .into_inner();
        let only = frames.message().await.unwrap().expect("an identity frame");
        assert_eq!(
            frames.message().await.unwrap(),
            None,
            "identity is one frame"
        );
        assert_eq!(
            (only.worker_id.as_str(), only.boot_id.as_str()),
            (WORKER, boot_id.as_str())
        );
        assert_eq!(only.receipt, sealed, "every caller reads one sealed receipt");
        assert!(only.capabilities.contains(&"update/1".to_string()));
        assert!(only.runs.is_empty() && only.gpus.is_empty() && only.idle_deadline_unix_ms == 0);
        assert!(only.models.is_empty() && only.models_bytes == 0);
    }
    let refused = client.status(status(true, None)).await.unwrap_err();
    assert_eq!(
        refused.code(),
        Code::PermissionDenied,
        "keepalive needs a machine cap"
    );

    // A machine cap streams the machine.
    let owner = cap("");
    let mut held = client
        .status(status(false, Some(&owner)))
        .await
        .unwrap()
        .into_inner();
    let first = held.message().await.unwrap().unwrap();
    assert_eq!(first.receipt, sealed);
    assert_eq!(first.phase, "ready");
    assert!(first.idle_deadline_unix_ms > 0);
    assert_eq!(
        first.platform,
        format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH)
    );
    assert!(first.started_at_unix_ms > 0);
    assert!(first
        .disk
        .is_some_and(|d| d.total_bytes > 0 && d.free_bytes <= d.total_bytes));
    assert!(first.runs.is_empty() && first.environments.is_empty());
    let models: Vec<_> = first
        .models
        .iter()
        .map(|m| (m.repository.as_str(), m.total_bytes, m.unique_bytes))
        .collect();
    assert_eq!(models, [("acme/base", 1300, 300), ("local/mine", 1050, 50)]);
    assert_eq!(first.models_bytes, 1350);
    assert_eq!(first.models[0].checkpoints[0].lane, "fp8");
    // Its player endpoint is the provider's mapping of the granted port.
    let webrtc = first.webrtc.clone().expect("a rental with a WebRTC port plays");
    assert_eq!(webrtc.addresses, ["203.0.113.7:30001"]);
    std::net::TcpStream::connect(("127.0.0.1", webrtc.port as u16)).unwrap();

    // Holding the stream is not activity: no frame, and the ledger's deadline does not move.
    let before = deadline_on_disk(&root);
    assert_eq!(before, first.idle_deadline_unix_ms);
    let quiet = tokio::time::timeout(Duration::from_millis(2500), held.message()).await;
    assert!(
        quiet.is_err(),
        "nothing changed, so nothing is sent: {quiet:?}"
    );
    assert_eq!(deadline_on_disk(&root), before);

    // One keepalive resets the deadline; its first frame carries it and the held stream sees it.
    let mut renewed = client
        .status(status(true, Some(&owner)))
        .await
        .unwrap()
        .into_inner();
    let reset = renewed.message().await.unwrap().unwrap();
    assert!(
        reset.idle_deadline_unix_ms > before,
        "{} > {before}",
        reset.idle_deadline_unix_ms
    );
    assert_eq!(deadline_on_disk(&root), reset.idle_deadline_unix_ms);
    let changed = tokio::time::timeout(Duration::from_secs(5), held.message())
        .await
        .expect("the held stream sends the change")
        .unwrap()
        .unwrap();
    assert_eq!(changed.idle_deadline_unix_ms, reset.idle_deadline_unix_ms);
    drop((held, renewed));
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}

/// A persistent machine (this computer's) never releases itself: Status names no deadline. It
/// makes its installer helper from the client it embeds.
#[tokio::test]
async fn a_persistent_machine_reports_no_idle_deadline() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-status-persistent")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    // The root's own uv, as an image or a local install provides it: the machine makes its
    // installer helper with it from the client it embeds.
    let uv = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|dir| dir.join("uv"))
        .find(|path| path.is_file())
        .expect("uv is required");
    std::fs::create_dir_all(root.join("usr/local/bin")).unwrap();
    std::os::unix::fs::symlink(&uv, root.join("usr/local/bin/uv")).unwrap();
    let mut machine = boot(&root, port, "persistent");
    receipt(&mut machine, &root, port);
    let helper = root.join("var/lib/cozy/rust-machine/helper/bin/python");
    let imported = Command::new(&helper)
        .args(["-c", "import cozy_machine_client.packages, packaging"])
        .status()
        .unwrap();
    assert!(
        imported.success(),
        "the installer helper holds the embedded client"
    );
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let channel = Endpoint::from_shared(format!("https://127.0.0.1:{port}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(pem))
                .domain_name("cozy-worker"),
        )
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = MachineClient::new(channel);
    let owner = cap("");
    for keepalive in [false, true] {
        let mut frames = client
            .status(status(keepalive, Some(&owner)))
            .await
            .unwrap()
            .into_inner();
        let frame = frames.message().await.unwrap().unwrap();
        assert_eq!(
            (frame.phase.as_str(), frame.idle_deadline_unix_ms),
            ("ready", 0)
        );
    }
    // It plays to browsers directly: Status names its WebRTC listener, an address of it that
    // accepts connections, and the certificate its DTLS presents.
    let player = |frame: v1::StatusFrame| frame.webrtc.expect("a persistent machine listens for players");
    let first = client.status(status(false, Some(&owner))).await.unwrap().into_inner().message().await;
    let webrtc = player(first.unwrap().unwrap());
    let local = format!("127.0.0.1:{}", webrtc.port);
    assert!(webrtc.addresses.contains(&local), "{:?}", webrtc.addresses);
    std::net::TcpStream::connect(&local).unwrap();
    let leaf = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let der = rustls_pemfile::certs(&mut &leaf[..]).next().unwrap().unwrap();
    assert_eq!(webrtc.fingerprint, tensorfs_core::sha256::hex_digest(&der));
    // A play link outlives a restart: the port is kept.
    drop(machine);
    let mut machine = boot(&root, port, "persistent");
    receipt(&mut machine, &root, port);
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("cozy-worker");
    let channel = Endpoint::from_shared(format!("https://127.0.0.1:{port}")).unwrap().tls_config(tls).unwrap().connect().await.unwrap();
    let again = MachineClient::new(channel).status(status(false, Some(&owner))).await.unwrap().into_inner().message().await;
    assert_eq!(player(again.unwrap().unwrap()).port, webrtc.port);
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}
