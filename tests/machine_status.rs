//! Status on the real binary in machine mode: anyone gets identity and the sealed receipt (the
//! bytes the Hub verifies); a machine cap gets the whole picture as it changes; an open stream is
//! not activity and only `keepalive` moves the idle deadline.
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
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

/// The HTTPS receipt route the Go-era Hub reads, until it answers.
fn receipt(machine: &mut Machine, root: &Path, port: u16) -> Vec<u8> {
    let start = Instant::now();
    loop {
        assert!(
            machine.0.try_wait().unwrap().is_none(),
            "the machine exited"
        );
        let output = Command::new("curl")
            .args(["-s", "-w", "\n%{http_code}", "--cacert"])
            .arg(root.join("run/cozy/bootstrap/tls.crt"))
            .args(["--resolve", &format!("cozy-worker:{port}:127.0.0.1")])
            .arg(format!("https://cozy-worker:{port}/v1/bootstrap/receipt"))
            .output()
            .unwrap();
        let text = String::from_utf8(output.stdout).unwrap();
        if let Some((body, "200")) = text.rsplit_once('\n') {
            return body.as_bytes().to_vec();
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
    let mut machine = boot(&root, port, "rental");
    let sealed = receipt(&mut machine, &root, port);
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let channel: Channel = Endpoint::from_shared(format!("https://127.0.0.1:{port}"))
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
    let boot_id = std::fs::read_to_string(root.join("var/lib/cozy/machine/boot-id")).unwrap();

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
        assert_eq!(only.receipt, sealed, "the receipt the HTTPS route serves");
        assert!(only.capabilities.contains(&"update/1".to_string()));
        assert!(only.runs.is_empty() && only.gpus.is_empty() && only.idle_deadline_unix_ms == 0);
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

/// A persistent machine (this computer's) never releases itself: Status names no deadline.
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
    let mut machine = boot(&root, port, "persistent");
    receipt(&mut machine, &root, port);
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
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}
