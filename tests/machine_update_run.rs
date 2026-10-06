//! `Run kind: update` on the real binary in machine mode: wheels the machine holds as objects are
//! staged, verified and activated in place (same boot, same receipt, the stable parent runs the
//! new service); the client attaches again across the restart and reads the outcome. A
//! candidate whose service never proves readiness ends FAILED, rolled back.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use cozy_machine::api::{
    capability::{self, Grant},
    v1::{self, machine_client::MachineClient},
};
use ed25519_dalek::SigningKey;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tonic::{
    metadata::MetadataValue,
    transport::{Certificate, ClientTlsConfig, Endpoint},
    Code, Request,
};

const WORKER: &str = "update-run-test";
const OWNER: [u8; 32] = [5; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wheel(distribution: &str, version: &str, script: Option<&[u8]>) -> (String, Vec<u8>) {
    let name = format!("{distribution}-{version}-py3-none-any.whl");
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file(
        format!("{distribution}-{version}.dist-info/METADATA"),
        options,
    )
    .unwrap();
    let metadata = format!(
        "Name: {}\nVersion: {version}\n",
        distribution.replace('_', "-")
    );
    zip.write_all(metadata.as_bytes()).unwrap();
    if let Some(script) = script {
        zip.start_file(
            format!("{distribution}-{version}.data/scripts/cozy-machine"),
            options,
        )
        .unwrap();
        zip.write_all(script).unwrap();
    }
    (name, zip.finish().unwrap().into_inner())
}

/// Puts `bytes` into the machine's store and records them as the owner's, as Write leaves an
/// object; answers its digest.
fn hold(root: &Path, bytes: &[u8]) -> String {
    let engine = root.join("var/lib/cozy/rust-machine");
    let store = tensorfs_core::store::Store::ensure(&root.join("var/lib/tensorfs")).unwrap();
    let sha256 = tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(bytes));
    let object = tensorfs_core::ids::ObjectRef {
        sha256: sha256.clone(),
        length: bytes.len() as u64,
    };
    store
        .put_stream(&mut &bytes[..], Some(&object), &Default::default())
        .unwrap();
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    cozy_machine::journal::Journal::open(&engine.join("execution"))
        .unwrap()
        .bind_object(&tensorfs_core::sha256::hex(owner.as_bytes()), &object)
        .unwrap();
    format!("sha256:{sha256}")
}

fn cap(action: &str) -> String {
    let expires = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 3600;
    let grant = Grant {
        machine: WORKER.into(),
        run: if action.is_empty() {
            "up-1".into()
        } else {
            String::new()
        },
        action: action.into(),
        expires,
        ..Default::default()
    };
    capability::mint(&SigningKey::from_bytes(&OWNER), grant)
}

fn authorized<T>(message: T, cap: &str) -> Request<T> {
    let mut request = Request::new(message);
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Cozy-Cap {cap}")).unwrap(),
    );
    request
}

/// A fresh connection, pinned to the machine's leaf, once the service answers Status.
async fn client(root: &Path, port: u16) -> MachineClient<tonic::transport::Channel> {
    let start = Instant::now();
    loop {
        if let Ok(pem) = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")) {
            let tls = ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(pem))
                .domain_name("cozy-worker");
            let endpoint = Endpoint::from_shared(format!("https://127.0.0.1:{port}"))
                .unwrap()
                .tls_config(tls)
                .unwrap();
            if let Ok(channel) = endpoint.connect().await {
                let mut client = MachineClient::new(channel);
                let ready = client
                    .status(authorized(
                        v1::StatusRequest::default(),
                        &cap(capability::MACHINE),
                    ))
                    .await;
                if let Ok(mut frames) = ready.map(|r| r.into_inner()) {
                    if let Ok(Some(frame)) = frames.message().await {
                        if frame.phase == "ready" {
                            return client;
                        }
                    }
                }
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(300),
            "the machine never answered"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn status(root: &Path, port: u16) -> v1::StatusFrame {
    let mut client = client(root, port).await;
    let request = authorized(v1::StatusRequest::default(), &cap(capability::MACHINE));
    let mut frames = client.status(request).await.unwrap().into_inner();
    frames.message().await.unwrap().unwrap()
}

/// Submits (or attaches to) update `id` and follows its log across restarts to the outcome.
async fn update(root: &Path, port: u16, id: &str, spec: Option<v1::RunSpec>) -> Vec<v1::RunEvent> {
    let (mut log, mut spec) = (Vec::<v1::RunEvent>::new(), spec);
    loop {
        let after = log.iter().map(|e| e.sequence).max().unwrap_or(0);
        let mut client = client(root, port).await;
        let request = v1::RunRequest {
            id: id.into(),
            after,
            spec: spec.take(),
        };
        let mut events = client
            .run(authorized(request, &cap(capability::MACHINE)))
            .await
            .unwrap()
            .into_inner();
        // The stream breaks when the service restarts onto the candidate; attach again.
        while let Ok(Some(event)) = events.message().await {
            let outcome = matches!(event.event, Some(v1::run_event::Event::Outcome(_)));
            log.push(event);
            if outcome {
                return log;
            }
        }
    }
}

fn outcome(log: &[v1::RunEvent]) -> v1::Outcome {
    match log.last().and_then(|e| e.event.clone()) {
        Some(v1::run_event::Event::Outcome(outcome)) => outcome,
        other => panic!("no outcome: {other:?} in {log:?}"),
    }
}

fn spec(runtime: (&str, &str), tensorfs: (&str, &str)) -> v1::RunSpec {
    let payload = serde_json::json!({
        "runtime": runtime.0, "tensorfs": tensorfs.0, "agent": "bundled", "note": "ignored",
    });
    let input = |field: &str, digest: &str| v1::InputFile {
        field: field.into(),
        digest: digest.into(),
        ..Default::default()
    };
    v1::RunSpec {
        kind: v1::RunKind::Update as i32,
        payload: serde_json::to_vec(&payload).unwrap(),
        inputs: vec![input("runtime", runtime.1), input("tensorfs", tensorfs.1)],
        ..Default::default()
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

#[tokio::test]
async fn an_update_run_activates_in_place_and_a_failing_candidate_ends_failed() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-update-run")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    let image = root.join("opt/cozy/wheels");
    std::fs::create_dir_all(&image).unwrap();
    for (name, bytes) in [
        wheel("cozy_runtime", "0.18.102", None),
        wheel("tensorfs", "0.3.93", None),
    ] {
        std::fs::write(image.join(name), bytes).unwrap();
    }
    let this_machine = std::fs::read(env!("CARGO_BIN_EXE_cozy-machine")).unwrap();
    let (runtime, runtime_bytes) = wheel("cozy_runtime", "0.18.103", Some(&this_machine));
    let (tensorfs, tensorfs_bytes) = wheel("tensorfs", "0.3.94", None);
    let broken = b"#!/bin/sh\n[ \"$1\" = version ] && echo '{\"name\":\"cozy-machine\",\"implementation\":\"rust\"}' && exit 0\nexit 3\n";
    let (failing, failing_bytes) = wheel("cozy_runtime", "0.18.104", Some(broken));
    let (runtime_digest, tensorfs_digest, failing_digest) = (
        hold(&root, &runtime_bytes),
        hold(&root, &tensorfs_bytes),
        hold(&root, &failing_bytes),
    );
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut machine = launch(&root, port, Some([7; 32]), false);
    let before = status(&root, port).await;
    assert_eq!(
        (before.runtime.as_str(), before.tensorfs.as_str()),
        ("0.18.102", "0.3.93")
    );
    assert!(before.capabilities.contains(&"update/1".to_string()));

    // A run-scope cap cannot update; a machine cap can.
    let mut refused = client(&root, port).await;
    let request = v1::RunRequest {
        id: "up-1".into(),
        after: 0,
        spec: Some(spec(
            (&runtime, &runtime_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    };
    let denied = refused
        .run(authorized(request, &cap("")))
        .await
        .unwrap_err();
    assert_eq!(denied.code(), Code::PermissionDenied);

    let log = update(
        &root,
        port,
        "up-1",
        Some(spec(
            (&runtime, &runtime_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    )
    .await;
    let done = outcome(&log);
    assert_eq!(done.status, "succeeded", "{log:?}");
    let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
    assert_eq!(
        result["to"],
        serde_json::json!({"runtime": "0.18.103", "tensorfs": "0.3.94"})
    );
    let stages: Vec<_> = log
        .iter()
        .filter_map(|e| match &e.event {
            Some(v1::run_event::Event::Progress(p)) => Some(p.stage.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        stages.contains(&"installing") && stages.contains(&"starting"),
        "{stages:?}"
    );
    let after = status(&root, port).await;
    assert_eq!(
        (after.runtime.as_str(), after.tensorfs.as_str()),
        ("0.18.103", "0.3.94")
    );
    assert_eq!(
        (after.boot_id.as_str(), after.receipt.as_slice()),
        (before.boot_id.as_str(), before.receipt.as_slice())
    );
    assert!(
        service_cmdline(machine.0.id()).contains("agent/current service"),
        "the parent runs the activated binary"
    );

    // The same id again attaches to the finished update; nothing runs twice.
    let again = update(
        &root,
        port,
        "up-1",
        Some(spec(
            (&runtime, &runtime_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    )
    .await;
    assert_eq!(outcome(&again), done);

    // A candidate whose service never proves readiness: FAILED, rolled back, still serving.
    let log = update(
        &root,
        port,
        "up-2",
        Some(spec(
            (&failing, &failing_digest),
            (&tensorfs, &tensorfs_digest),
        )),
    )
    .await;
    let failed = outcome(&log);
    assert_eq!(failed.status, "failed", "{log:?}");
    assert_eq!(failed.reason.unwrap().code, "update_rolled_back");
    assert_eq!(status(&root, port).await.runtime, "0.18.103");
    assert!(
        machine.0.try_wait().unwrap().is_none(),
        "the machine keeps running after a rollback"
    );

    // Stopped and launched again: `cozy machine stop` / `start` on this computer (a persistent
    // machine), then a rental's pod restarted. Each launch proves itself under a new key, which
    // the parent hands to the updated service it runs.
    for (persistent, key) in [(true, [8; 32]), (true, [9; 32]), (false, [10; 32])] {
        stop(&mut machine);
        std::fs::remove_file(root.join("run/cozy/bootstrap/readiness-envelope.json")).unwrap();
        machine = launch(&root, port, Some(key), persistent);
        let frame = status(&root, port).await;
        assert_eq!(frame.runtime, "0.18.103");
        assert!(
            service_cmdline(machine.0.id()).contains("agent/current service"),
            "the launch runs the activated binary"
        );
    }

    // A launch that cannot prove its boot ends, with a status, instead of serving unready.
    stop(&mut machine);
    std::fs::remove_file(root.join("run/cozy/bootstrap/readiness-envelope.json")).unwrap();
    let mut unprovable = launch(&root, port, None, true);
    let ended = unprovable.0.wait().unwrap();
    assert!(!ended.success(), "an unprovable launch ended {ended}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Starts the machine on `root` as a pod (or, persistent, as this computer's CLI) does, with this
/// launch's readiness key (none: it cannot prove readiness).
fn launch(root: &Path, port: u16, key: Option<[u8; 32]>, persistent: bool) -> Machine {
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    let mut command = Command::new(env!("CARGO_BIN_EXE_cozy-machine"));
    command
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
        .env("TENSORHUB_ORIGIN", "https://hub.invalid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if persistent {
        command.env("COZY_MACHINE_LIFETIME", "persistent");
    }
    if let Some(key) = key {
        command.env(
            "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
            URL_SAFE_NO_PAD.encode(key),
        );
    }
    Machine(command.spawn().unwrap())
}

/// Stops the machine as `cozy machine stop` does: SIGTERM to its parent, which ends cleanly.
fn stop(machine: &mut Machine) {
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(machine.0.id() as i32),
        nix::sys::signal::Signal::SIGTERM,
    )
    .unwrap();
    machine.0.wait().unwrap();
}


/// The persisted boundaries are interrupted deliberately, then a real supervisor and service
/// start over them. A partially published SDK must never be selected or committed by readiness.
#[tokio::test]
async fn startup_recovers_interrupted_publication_before_selecting_the_service() {
    use std::os::unix::fs::symlink;
    for (state, changed_links, runtime, outcome) in [
        ("installing", 0, "0.18.102", "rolled_back"),
        ("installing", 1, "0.18.102", "rolled_back"),
        ("installing", 2, "0.18.102", "rolled_back"),
        ("succeeded", 2, "0.18.103", "succeeded"),
        ("rolled_back", 2, "0.18.102", "rolled_back"),
        ("starting", 2, "0.18.102", "rolled_back"),
    ] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/machine-update-recovery")
            .join(uuid::Uuid::new_v4().to_string());
        let image = root.join("opt/cozy/wheels");
        let engine = root.join("var/lib/cozy/rust-machine");
        let candidate = engine.join("sdk/candidate");
        let updates = engine.join("update");
        for dir in [&image, &candidate, &updates, &engine.join("agent")] {
            std::fs::create_dir_all(dir).unwrap();
        }
        for (dir, version) in [(&image, "0.18.102"), (&candidate, "0.18.103")] {
            for (name, bytes) in [wheel("cozy_runtime", version, None), wheel("tensorfs", "0.3.94", None)] {
                std::fs::write(dir.join(name), bytes).unwrap();
            }
        }
        if state != "starting" {
            std::fs::copy(env!("CARGO_BIN_EXE_cozy-machine"), candidate.join("cozy-machine")).unwrap();
        }
        // `starting` exercises a dangling selected executable. It must fail exec and roll
        // back, never fall through to the image executable and falsely commit the update.
        let status = serde_json::json!({
            "operation": "interrupted", "state": state,
            "from": {"runtime": "0.18.102", "tensorfs": "0.3.94"},
            "to": {"runtime": "0.18.103", "tensorfs": "0.3.94"},
        });
        let mut pending_status = status.clone();
        pending_status["state"] = "installing".into();
        std::fs::write(updates.join("pending.json"), serde_json::to_vec(&serde_json::json!({
            "status": pending_status, "sdk_before": null, "agent_before": null,
        })).unwrap()).unwrap();
        std::fs::write(updates.join("status.json"), serde_json::to_vec(&status).unwrap()).unwrap();
        if changed_links >= 1 { symlink(&candidate, engine.join("sdk/current")).unwrap(); }
        if changed_links >= 2 { symlink(candidate.join("cozy-machine"), engine.join("agent/current")).unwrap(); }
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut machine = launch(&root, port, Some([7; 32]), true);
        let frame = self::status(&root, port).await;
        assert_eq!(frame.runtime, runtime, "{state} after {changed_links} links");
        assert!(!updates.join("pending.json").exists());
        let settled: serde_json::Value = serde_json::from_slice(&std::fs::read(updates.join("status.json")).unwrap()).unwrap();
        assert_eq!(settled["state"], outcome);
        if state == "installing" {
            assert_eq!(settled["error"], "machine_restarted: software publication did not finish");
        } else if state == "starting" {
            assert_eq!(settled["error"], "the service exited twice before readiness (1)");
        }
        stop(&mut machine);
        std::fs::remove_dir_all(root).unwrap();
    }
}
