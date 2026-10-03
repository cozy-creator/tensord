//! `runtime-update/1` on the real binary, through the routes `cozy rental update` drives: a
//! candidate Runtime wheel bundling a Rust machine is staged, verified and activated in place
//! (same boot, same receipt, the stable parent runs the new service); a candidate whose
//! service never proves readiness is rolled back.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wheel(dir: &Path, distribution: &str, version: &str, script: Option<&[u8]>) -> PathBuf {
    let path = dir.join(format!("{distribution}-{version}-py3-none-any.whl"));
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file(
        format!("{distribution}-{version}.dist-info/METADATA"),
        options,
    )
    .unwrap();
    zip.write_all(
        format!(
            "Name: {}\nVersion: {version}\n",
            distribution.replace('_', "-")
        )
        .as_bytes(),
    )
    .unwrap();
    if let Some(script) = script {
        zip.start_file(
            format!("{distribution}-{version}.data/scripts/cozy-machine"),
            options,
        )
        .unwrap();
        zip.write_all(script).unwrap();
    }
    zip.finish().unwrap();
    path
}

struct Client {
    root: PathBuf,
    port: u16,
    token: String,
}
impl Client {
    fn curl(&self, args: &[&str]) -> (String, String) {
        let output = Command::new("curl")
            .args(["-s", "-w", "\n%{http_code}", "--cacert"])
            .arg(self.root.join("run/cozy/bootstrap/tls.crt"))
            .args(["--resolve", &format!("cozy-worker:{}:127.0.0.1", self.port)])
            .args(["-H", &format!("Authorization: Cozy-Cap {}", self.token)])
            .args(args)
            .output()
            .unwrap();
        let text = String::from_utf8(output.stdout).unwrap();
        let (body, code) = text.rsplit_once('\n').unwrap_or((&text, "000"));
        (body.to_owned(), code.to_owned())
    }
    fn url(&self, path: &str) -> String {
        format!("https://cozy-worker:{}{path}", self.port)
    }
    fn state(&self) -> Option<serde_json::Value> {
        let (body, code) = self.curl(&[&self.url("/v1/machine/runtime")]);
        (code == "200").then(|| serde_json::from_str(&body).unwrap())
    }
    fn until(&self, done: impl Fn(&serde_json::Value) -> bool) -> serde_json::Value {
        let start = Instant::now();
        loop {
            if let Some(state) = self.state().filter(|s| done(s)) {
                return state;
            }
            assert!(
                start.elapsed() < Duration::from_secs(300),
                "the machine never reached the expected state"
            );
            std::thread::sleep(Duration::from_millis(300));
        }
    }
    fn stage(&self, path: &Path) -> String {
        let name = path.file_name().unwrap().to_str().unwrap();
        let (body, code) = self.curl(&[
            "-X",
            "PUT",
            "--data-binary",
            &format!("@{}", path.display()),
            &self.url(&format!("/v1/machine/runtime/wheels/{name}")),
        ]);
        assert_eq!(code, "200", "{body}");
        let staged: serde_json::Value = serde_json::from_str(&body).unwrap();
        let digest: String = Sha256::digest(std::fs::read(path).unwrap())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(staged["sha256"], digest);
        staged["sha256"].as_str().unwrap().to_owned()
    }
    fn update(&self, operation: &str, runtime: &Path, tensorfs: &Path) {
        let choice = |path: &Path| serde_json::json!({"file": path.file_name().unwrap().to_str().unwrap(), "sha256": self.stage(path)});
        let body = serde_json::json!({"operation": operation, "agent": "bundled", "pin": false, "runtime": choice(runtime), "tensorfs": choice(tensorfs)});
        let (answer, code) = self.curl(&[
            "-X",
            "POST",
            "--data-binary",
            &body.to_string(),
            &self.url("/v1/machine/runtime/update"),
        ]);
        assert_eq!(code, "202", "{answer}");
    }
}

fn service_cmdline(parent: u32) -> String {
    let children = std::fs::read_to_string(format!("/proc/{parent}/task/{parent}/children"))
        .unwrap_or_default();
    children
        .split_whitespace()
        .filter_map(|pid| std::fs::read(format!("/proc/{pid}/cmdline")).ok())
        .map(|raw| String::from_utf8_lossy(&raw).replace('\0', " "))
        .find(|line| line.contains("cozy-machine"))
        .unwrap_or_default()
}

#[test]
fn an_update_activates_in_place_and_a_failing_candidate_rolls_back() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-update")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    let (image, candidates) = (root.join("opt/cozy/wheels"), root.join("candidates"));
    std::fs::create_dir_all(&image).unwrap();
    std::fs::create_dir_all(&candidates).unwrap();
    wheel(&image, "cozy_runtime", "0.18.102", None);
    wheel(&image, "tensorfs", "0.3.93", None);
    let key = SigningKey::from_bytes(&[5; 32]);
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut machine = Machine(
        Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("COZY_MACHINE_ROOT", &root)
            .env("COZY_WORKER_ID", "ra-update-test")
            .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
            .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
            .env("COZY_LISTEN_HOST", "127.0.0.1")
            .env(
                "COZY_AUTHORIZED_KEYS",
                URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes()),
            )
            .env(
                "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
                URL_SAFE_NO_PAD.encode([7; 32]),
            )
            .env("TENSORHUB_ORIGIN", "https://hub.invalid")
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let key_id = URL_SAFE_NO_PAD.encode(&Sha256::digest(key.verifying_key().as_bytes())[..16]);
    let expires = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 3600;
    let payload = serde_json::json!({"m": "ra-update-test", "a": "runtime-update", "e": expires, "k": key_id}).to_string();
    let mut signed = b"cozy-capability/1\0".to_vec();
    signed.extend_from_slice(payload.as_bytes());
    let token = format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(&payload),
        URL_SAFE_NO_PAD.encode(key.sign(&signed).to_bytes())
    );
    let client = Client {
        root: root.clone(),
        port,
        token,
    };
    let start = Instant::now();
    while !root.join("run/cozy/bootstrap/tls.crt").exists() {
        assert!(start.elapsed() < Duration::from_secs(120), "no identity");
        std::thread::sleep(Duration::from_millis(100));
    }
    let ready = client.until(|s| s["phase"] == "ready");
    assert_eq!(
        (ready["runtime"].as_str(), ready["tensorfs"].as_str()),
        (Some("0.18.102"), Some("0.3.93"))
    );
    assert_eq!(ready["bootstrap"]["abi"], "machine-bootstrap/1");
    assert!(ready["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c == "runtime-update/1"));
    let receipt =
        || std::fs::read(root.join("run/cozy/bootstrap/readiness-envelope.json")).unwrap();
    let (boot, sealed) = (
        std::fs::read(root.join("var/lib/cozy/machine/boot-id")).unwrap(),
        receipt(),
    );
    let foreign = Client {
        root: root.clone(),
        port,
        token: "forged.token".into(),
    };
    assert_eq!(
        foreign.curl(&[&foreign.url("/v1/machine/runtime")]).1,
        "403"
    );

    // A candidate bundling this Rust machine: activated in place under the same parent.
    let machine_bytes = std::fs::read(env!("CARGO_BIN_EXE_cozy-machine")).unwrap();
    let runtime = wheel(
        &candidates,
        "cozy_runtime",
        "0.18.103",
        Some(&machine_bytes),
    );
    let tensorfs = wheel(&candidates, "tensorfs", "0.3.94", None);
    client.update("op-1", &runtime, &tensorfs);
    let done = client.until(|s| {
        s["update"]["operation"] == "op-1"
            && ["succeeded", "rolled_back", "failed"]
                .contains(&s["update"]["state"].as_str().unwrap_or(""))
    });
    assert_eq!(done["update"]["state"], "succeeded", "{done}");
    assert_eq!(
        (
            done["runtime"].as_str(),
            done["tensorfs"].as_str(),
            done["agent"]["selection"].as_str()
        ),
        (Some("0.18.103"), Some("0.3.94"), Some("bundled"))
    );
    assert!(
        service_cmdline(machine.0.id()).contains("agent/current service"),
        "the parent runs the activated binary"
    );
    assert_eq!(
        std::fs::read(root.join("var/lib/cozy/machine/boot-id")).unwrap(),
        boot
    );
    assert_eq!(receipt(), sealed);

    // A candidate whose service never proves readiness: the same parent restores the pair.
    let broken = b"#!/bin/sh\n[ \"$1\" = version ] && echo '{\"name\":\"cozy-machine\",\"implementation\":\"rust\"}' && exit 0\nexit 3\n";
    let runtime = wheel(&candidates, "cozy_runtime", "0.18.104", Some(broken));
    client.update("op-2", &runtime, &tensorfs);
    let done = client.until(|s| {
        s["update"]["operation"] == "op-2"
            && ["succeeded", "rolled_back", "failed"]
                .contains(&s["update"]["state"].as_str().unwrap_or(""))
            && s["phase"] == "ready"
    });
    assert_eq!(done["update"]["state"], "rolled_back", "{done}");
    assert_eq!(done["runtime"], "0.18.103");
    assert!(
        machine.0.try_wait().unwrap().is_none(),
        "the machine keeps running after a rollback"
    );
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}
