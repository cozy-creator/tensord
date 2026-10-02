//! The real binary in machine mode, as a pod boots it: its readiness receipt verifies under
//! the attempt key with the facts the Hub reads, and a restart keeps the boot and the bytes.
use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const KEY: [u8; 32] = [7; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn boot(root: &Path, port: u16, key: Option<&[u8]>) -> Machine {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cozy-machine"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("COZY_MACHINE_ROOT", root)
        .env("COZY_WORKER_ID", "ra-readiness-test")
        .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env(
            "COZY_AUTHORIZED_KEYS",
            "O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik",
        )
        .env("TENSORHUB_ORIGIN", "https://hub.invalid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(key) = key {
        command.env(
            "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
            URL_SAFE_NO_PAD.encode(key),
        );
    }
    Machine(command.spawn().unwrap())
}

/// The receipt over the pinned leaf, as the Hub fetches it; None while it answers 503.
fn receipt(root: &Path, port: u16) -> Option<Vec<u8>> {
    let cert = root.join("run/cozy/bootstrap/tls.crt");
    let output = Command::new("curl")
        .args(["-s", "-w", "\n%{http_code}", "--cacert"])
        .arg(&cert)
        .args(["--resolve", &format!("cozy-worker:{port}:127.0.0.1")])
        .arg(format!("https://cozy-worker:{port}/v1/bootstrap/receipt"))
        .output()
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let (body, code) = text.rsplit_once('\n')?;
    (code == "200").then(|| body.as_bytes().to_vec())
}

fn awaited(machine: &mut Machine, root: &Path, port: u16) -> Vec<u8> {
    let start = Instant::now();
    loop {
        assert!(
            machine.0.try_wait().unwrap().is_none(),
            "the machine exited before readiness"
        );
        if let Some(body) = receipt(root, port) {
            return body;
        }
        assert!(start.elapsed() < Duration::from_secs(300), "no receipt");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn receipt_verifies_and_a_restart_keeps_boot_and_bytes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-readiness")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let port = free_port();
    let mut first = boot(&root, port, Some(&KEY));
    let sealed = awaited(&mut first, &root, port);
    #[derive(serde::Deserialize)]
    struct Envelope {
        payload: String,
        hmac_sha256: String,
    }
    let envelope: Envelope = serde_json::from_slice(&sealed).unwrap();
    let payload = STANDARD.decode(&envelope.payload).unwrap();
    let mut mac = Hmac::<Sha256>::new_from_slice(&KEY).unwrap();
    mac.update(b"cozy.pod-readiness/1\0");
    mac.update(&payload);
    assert_eq!(
        envelope.hmac_sha256,
        tensorfs_core::sha256::hex(&mac.finalize().into_bytes())
    );
    let facts: serde_json::Value = serde_json::from_slice(&payload).unwrap();
    let boot_id = std::fs::read_to_string(root.join("var/lib/cozy/machine/boot-id")).unwrap();
    assert_eq!(facts["pod_boot_id"], boot_id.as_str());
    assert_eq!(URL_SAFE_NO_PAD.decode(&boot_id).unwrap().len(), 32);
    assert_eq!(facts["worker_internal_port"], port);
    assert_eq!(facts["worker_protocol"], "cozy.worker.v1");
    assert_eq!(facts["worker_listener_bound"], true);
    assert_eq!(facts["worker_foreign_credential_refused"], true);
    assert_eq!(facts["runtime_gpus"], serde_json::json!([]));
    drop(first);
    // A container restart replays the key; without one the retained envelope still serves.
    for key in [Some(&KEY[..]), None] {
        let mut again = boot(&root, port, key);
        assert_eq!(awaited(&mut again, &root, port), sealed);
    }
    // A foreign key cannot adopt this boot.
    let mut foreign = boot(&root, port, Some(&[8; 32]));
    assert!(!foreign.0.wait().unwrap().success());
    std::fs::remove_dir_all(&root).unwrap();
}
