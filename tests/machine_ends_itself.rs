//! A rental idle past its deadline whose Hub never answered through that whole idle window cannot
//! be ended by the Hub, so it ends itself at its provider and billing stops: the real binary,
//! a stand-in provider API, and a Hub that cannot be reached. The provider key it ends itself
//! with is the machine's alone: gone from every process environment and from the files the
//! provider copies it into, and still handed to an activated Runtime update's service.
mod common;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn accepted_terminal_history_prevents_unused_timeout_after_a_service_restart() {
    use cozy_machine::journal::{Invocation, Journal};

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/used-rental-history")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&root).unwrap();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (provider_port, received) = provider();
    let mut machine = launch(&root, port, provider_port);
    let start = Instant::now();
    while common::receipt(&root, port).is_none() {
        assert!(start.elapsed() < Duration::from_secs(120));
        assert!(machine.0.try_wait().unwrap().is_none());
        std::thread::sleep(Duration::from_millis(100));
    }
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(machine.0.id() as i32),
        nix::sys::signal::Signal::SIGTERM,
    ).unwrap();
    machine.0.wait().unwrap();
    // A legacy short accepted run can finish before the old one-second sampler sees it.
    let mut journal = Journal::open(&root.join("var/lib/cozy/rust-machine/execution")).unwrap();
    assert!(!journal.has_executions().unwrap());
    let execution = journal.accept("short-prior-work", Invocation {
        package: "test/accepted-work".into(), input: serde_json::json!({}), ..Default::default()
    }).unwrap();
    journal.cancel(&execution.id, "test-owner").unwrap();
    assert!(journal.has_executions().unwrap());
    drop(journal);
    let ledger = root.join("var/lib/cozy/machine/idle.json");
    let mut idle: serde_json::Value = serde_json::from_slice(&std::fs::read(&ledger).unwrap()).unwrap();
    idle["deadline_ms"] = serde_json::json!(1);
    idle["work_observed"] = serde_json::json!(false);
    std::fs::write(&ledger, serde_json::to_vec(&idle).unwrap()).unwrap();
    let mut restarted = launch(&root, port, provider_port);
    let start = Instant::now();
    while common::receipt(&root, port).is_none() {
        assert!(start.elapsed() < Duration::from_secs(120));
        assert!(restarted.0.try_wait().unwrap().is_none());
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(received.recv_timeout(Duration::from_secs(3)).is_err());
    assert!(restarted.0.try_wait().unwrap().is_none());
    let idle: serde_json::Value = serde_json::from_slice(&std::fs::read(&ledger).unwrap()).unwrap();
    assert_eq!(idle["work_observed"], true);
    assert_eq!(idle["released"], false);
    drop(restarted);
    drop(machine);
    std::fs::remove_dir_all(root).unwrap();
}

fn launch(root: &Path, port: u16, provider: u16) -> Machine {
    let owner = SigningKey::from_bytes(&[3; 32]).verifying_key();
    Machine(
        Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("COZY_MACHINE_ROOT", root)
            .env("COZY_WORKER_ID", "ends-itself")
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
            // A Hub this pod cannot reach, as when the Hub's network is down.
            .env("TENSORHUB_ORIGIN", "https://hub.invalid")
            .env("RUNPOD_POD_ID", "pod-standin")
            .env("RUNPOD_API_KEY", "key-standin")
            .env(
                "COZY_PROVIDER_API_ORIGIN",
                format!("http://127.0.0.1:{provider}"),
            )
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

/// A stand-in provider API: answers each request as RunPod does a successful terminate and
/// reports it (method, path, authorization, body).
fn provider() -> (u16, mpsc::Receiver<(String, String, String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (sender, received) = mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut parts = line.split_whitespace();
            let (method, path) = (
                parts.next().unwrap_or("").to_owned(),
                parts.next().unwrap_or("").to_owned(),
            );
            let (mut length, mut authorization) = (0usize, String::new());
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                let header = header.trim_end();
                if header.is_empty() {
                    break;
                }
                let (name, value) = header.split_once(':').unwrap();
                match name.to_ascii_lowercase().as_str() {
                    "content-length" => length = value.trim().parse().unwrap(),
                    "authorization" => authorization = value.trim().to_owned(),
                    _ => {}
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let answer = br#"{"data":{"podTerminate":null}}"#;
            let mut stream = stream;
            write!(stream, "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", answer.len()).unwrap();
            stream.write_all(answer).unwrap();
            let _ = sender.send((
                method,
                path,
                authorization,
                String::from_utf8_lossy(&body).into_owned(),
            ));
        }
    });
    (port, received)
}

#[test]
fn a_rental_the_hub_cannot_hear_ends_itself_at_its_provider() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-ends-itself")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let (provider_port, received) = provider();
    // Where RunPod's tooling copies the environment for login shells.
    std::fs::create_dir_all(root.join("etc")).unwrap();
    std::fs::create_dir_all(root.join("root")).unwrap();
    std::fs::write(
        root.join("etc/rp_environment"),
        "export RUNPOD_POD_ID=\"pod-standin\"\nexport RUNPOD_API_KEY=\"key-standin\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("root/.bashrc"),
        "source /etc/rp_environment\nexport KEY=key-standin\n",
    )
    .unwrap();

    // The first launch proves this boot; within its idle window nothing ends.
    let mut machine = launch(&root, port, provider_port);
    let envelope = root.join("run/cozy/bootstrap/readiness-envelope.json");
    let start = Instant::now();
    while !envelope.exists() {
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the machine never proved readiness"
        );
        assert!(
            machine.0.try_wait().unwrap().is_none(),
            "the machine ended before readiness"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(
        received.recv_timeout(Duration::from_secs(3)).is_err(),
        "a rental within its idle window ended itself"
    );
    let holders = environments_holding(machine.0.id(), "key-standin");
    assert!(
        holders.is_empty(),
        "processes still hold the key: {holders:?}"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("etc/rp_environment")).unwrap(),
        "export RUNPOD_POD_ID=\"pod-standin\"\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("root/.bashrc")).unwrap(),
        "source /etc/rp_environment\n"
    );
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(machine.0.id() as i32),
        nix::sys::signal::Signal::SIGTERM,
    )
    .unwrap();
    machine.0.wait().unwrap();

    // The same boot after its idle deadline, the Hub silent the whole window, now serving an
    // activated Runtime update: the parent hands the key to that service.
    let agent = root.join("var/lib/cozy/rust-machine/agent");
    std::fs::create_dir_all(&agent).unwrap();
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_cozy-machine"), agent.join("current")).unwrap();
    let ledger = root.join("var/lib/cozy/machine/idle.json");
    let mut idle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ledger).unwrap()).unwrap();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    idle["deadline_ms"] = serde_json::json!(now - 1);
    std::fs::write(&ledger, serde_json::to_vec(&idle).unwrap()).unwrap();
    let mut machine = launch(&root, port, provider_port);
    let (method, path, authorization, body) = received
        .recv_timeout(Duration::from_secs(120))
        .expect("the rental did not end itself at its provider");
    assert_eq!((method.as_str(), path.as_str()), ("POST", "/graphql"));
    assert_eq!(authorization, "Bearer key-standin");
    assert!(
        body.contains("podTerminate") && body.contains("pod-standin"),
        "{body}"
    );
    let start = Instant::now();
    let ended = loop {
        if let Some(status) = machine.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the machine kept running after ending its pod"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(ended.success(), "{ended}");
    let released: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ledger).unwrap()).unwrap();
    assert_eq!(released["released"], true);
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}

/// Each process in `pid`'s tree whose environment holds `needle`.
fn environments_holding(pid: u32, needle: &str) -> Vec<u32> {
    let mut tree = vec![pid];
    let mut held = vec![];
    while let Some(pid) = tree.pop() {
        let children =
            std::fs::read_to_string(format!("/proc/{pid}/task/{pid}/children")).unwrap_or_default();
        tree.extend(
            children
                .split_whitespace()
                .filter_map(|c| c.parse::<u32>().ok()),
        );
        let environ = std::fs::read(format!("/proc/{pid}/environ")).unwrap_or_default();
        if environ
            .windows(needle.len())
            .any(|w| w == needle.as_bytes())
        {
            held.push(pid);
        }
    }
    held
}

/// A Runtime update back to a machine that predates the credential hand-off (it does not declare
/// `provider-credential/1`) is never handed the key: it would leave the pipe open for every
/// executor it starts. It still gets this boot's readiness pipe and key.
#[test]
fn an_older_service_is_never_handed_the_provider_key() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-older-service")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    let agent = root.join("var/lib/cozy/rust-machine/agent");
    std::fs::create_dir_all(&agent).unwrap();
    let older = agent.join("older");
    std::fs::write(
        &older,
        "#!/bin/sh\n[ \"$1\" = version ] && echo '{\"name\":\"cozy-machine\",\"implementation\":\"rust\",\"capabilities\":[]}' && exit 0\nls /proc/$$/fd > \"$COZY_MACHINE_ROOT/service-fds.part\" && mv \"$COZY_MACHINE_ROOT/service-fds.part\" \"$COZY_MACHINE_ROOT/service-fds\"\nexec sleep 60\n",
    )
    .unwrap();
    std::fs::set_permissions(&older, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink(&older, agent.join("current")).unwrap();
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let (provider_port, _received) = provider();
    let machine = launch(&root, port, provider_port);
    let listing = root.join("service-fds");
    let start = Instant::now();
    while !listing.exists() {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the older service never started"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    let fds: Vec<String> = std::fs::read_to_string(&listing)
        .unwrap()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    assert!(
        fds.contains(&"3".to_string()) && fds.contains(&"4".to_string()),
        "{fds:?}"
    );
    assert!(
        !fds.contains(&"5".to_string()),
        "the older service holds fd 5: {fds:?}"
    );
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}
