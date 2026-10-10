//! A rental releases itself after its idle window with no job queued or running, used or not
//! (owner ruling 2026-10-10), and a restart neither shortens nor extends that clock. Its Hub
//! never answered through that whole idle window, so it ends itself at its provider and billing
//! stops: the real binary, a stand-in provider API, and a Hub that cannot be reached. The
//! provider key it ends itself with is the machine's alone: gone from every process environment
//! and from the files the provider copies it into, and still handed to an activated Runtime
//! update's service.
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
use cozy_machine::api::{
    capability::{self, Grant},
    v1::{self, machine_client::MachineClient},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn launch(root: &Path, port: u16, provider: u16) -> Machine {
    launch_idle(root, port, provider, 900)
}

/// `launch` with an idle window of `idle` seconds.
fn launch_idle(root: &Path, port: u16, provider: u16, idle: u32) -> Machine {
    let owner = SigningKey::from_bytes(&[3; 32]).verifying_key();
    Machine(
        Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
            .env_clear()
            .env("COZY_RENTAL_IDLE_SECONDS", idle.to_string())
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
    idle["idle_since_ms"] = serde_json::json!(now - 900_001);
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

const IDLE: u32 = 6;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64
}

fn root(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(name)
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn ready(machine: &mut Machine, root: &Path, port: u16) {
    let start = Instant::now();
    while common::receipt(root, port).is_none() {
        assert!(start.elapsed() < Duration::from_secs(120), "no receipt");
        assert!(machine.0.try_wait().unwrap().is_none(), "the machine exited");
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn stop(machine: &mut Machine) {
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(machine.0.id() as i32),
        nix::sys::signal::Signal::SIGTERM,
    )
    .unwrap();
    machine.0.wait().unwrap();
}

/// The owner's client and machine cap.
async fn owner(root: &Path, port: u16) -> (MachineClient<Channel>, String) {
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let endpoint = Endpoint::from_shared(format!("https://127.0.0.1:{port}"))
        .unwrap()
        .tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("cozy-worker"))
        .unwrap();
    let grant = Grant {
        machine: "ends-itself".into(),
        action: capability::MACHINE.into(),
        expires: now_ms() / 1000 + 600,
        ..Default::default()
    };
    let cap = capability::mint(&SigningKey::from_bytes(&[3; 32]), grant);
    (MachineClient::new(endpoint.connect().await.unwrap()), cap)
}

fn authorized<T>(message: T, cap: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request.metadata_mut().insert("authorization", format!("Cozy-Cap {cap}").parse().unwrap());
    request
}

/// The idle deadline Status reports; `keepalive` restarts the clock first.
async fn deadline(client: &mut MachineClient<Channel>, cap: &str, keepalive: bool) -> i64 {
    let mut frames = client.status(authorized(v1::StatusRequest { keepalive }, cap)).await.unwrap().into_inner();
    frames.message().await.unwrap().unwrap().idle_deadline_unix_ms
}

/// A Hub whose every request is held open, as one stalled mid-preparation, until `close`.
struct HeldHub {
    origin: String,
    close: mpsc::Sender<()>,
}
fn held_hub() -> HeldHub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let (close, closed) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut held = vec![];
        while closed.try_recv() == Err(mpsc::TryRecvError::Empty) {
            if let Ok((stream, _)) = listener.accept() {
                held.push(stream);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    HeldHub { origin, close }
}

/// Submits a call of a release that prepares from `hub`: a job queued for this rental.
async fn submit(client: &mut MachineClient<Channel>, cap: &str, id: &str, hub: &str) -> tonic::Streaming<v1::RunEvent> {
    let spec = v1::RunSpec {
        kind: v1::RunKind::Call as i32,
        source: Some(v1::run_spec::Source::Release(v1::Release { package: "acme/stalled".into(), ..Default::default() })),
        entrypoint: "generate".into(),
        hub: Some(v1::HubAccess { origin: hub.into(), ..Default::default() }),
        ..Default::default()
    };
    let request = v1::RunRequest { id: id.into(), after: 0, spec: Some(spec) };
    let mut events = client.run(authorized(request, cap)).await.unwrap().into_inner();
    assert!(events.message().await.unwrap().is_some(), "the job was accepted");
    events
}

/// The run's end (its outcome event), whenever it comes.
async fn ended(events: &mut tonic::Streaming<v1::RunEvent>) -> v1::Outcome {
    while let Some(event) = events.message().await.unwrap() {
        if let Some(v1::run_event::Event::Outcome(outcome)) = event.event {
            return outcome;
        }
    }
    panic!("the run's log ended without an outcome");
}

/// A job queued for the rental holds it past any number of idle windows; the clock starts when
/// that job ends, and the used rental releases one window later.
#[tokio::test]
async fn a_job_holds_the_rental_and_its_end_starts_the_idle_clock() {
    let root = root("idle-job");
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (provider_port, received) = provider();
    let mut machine = launch_idle(&root, port, provider_port, IDLE);
    ready(&mut machine, &root, port);
    let (mut client, cap) = owner(&root, port).await;
    let idle_ms = i64::from(IDLE) * 1000;
    let renewed = deadline(&mut client, &cap, true).await;
    let at = now_ms();
    assert!((renewed - at - idle_ms).abs() < 1_000, "keepalive answers the real deadline: {renewed} at {at}");
    let hub = held_hub();
    let mut events = submit(&mut client, &cap, "stalled-1", &hub.origin).await;
    let working = deadline(&mut client, &cap, false).await;
    assert!(working >= renewed, "a queued job holds the clock at now + the window");
    assert!(
        received.recv_timeout(Duration::from_millis(3 * idle_ms as u64)).is_err(),
        "the rental released itself while a job was queued for it"
    );
    drop(hub.close);
    let outcome = tokio::time::timeout(Duration::from_secs(120), ended(&mut events)).await.unwrap();
    let end = now_ms();
    assert_eq!(outcome.status, "failed", "{outcome:?}");
    // Status reads the jobs the machine sampled within the last second.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let due = deadline(&mut client, &cap, false).await;
    assert!(due > end - 1_000 + idle_ms - 1_000 && due <= end + idle_ms, "{due} vs end {end}");
    let (method, _, _, body) = received
        .recv_timeout(Duration::from_secs(60))
        .expect("the used rental did not release itself after its idle window");
    let released = now_ms();
    assert!(method == "POST" && body.contains("podTerminate"), "{body}");
    assert!(released >= due - 500, "released at {released}, before its deadline {due}");
    assert!(released < due + 5_000, "released at {released}, long after its deadline {due}");
    drop(machine);
    std::fs::remove_dir_all(root).unwrap();
}

/// A restart mid-job never releases early: a job preparing when the service stopped ends at the
/// restart, which starts the clock. A restart mid-idle keeps the exact deadline.
#[tokio::test]
async fn a_restart_neither_shortens_nor_extends_the_idle_clock() {
    let root = root("idle-restart");
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (provider_port, received) = provider();
    let idle_ms = 3 * i64::from(IDLE) * 1000;
    let idle = 3 * IDLE;
    let mut machine = launch_idle(&root, port, provider_port, idle);
    ready(&mut machine, &root, port);
    let (mut client, cap) = owner(&root, port).await;
    let hub = held_hub();
    let events = submit(&mut client, &cap, "stalled-2", &hub.origin).await;
    drop((events, client));
    stop(&mut machine);
    // Down longer than the whole window: the clock from before the job ran out long ago.
    std::thread::sleep(Duration::from_millis(idle_ms as u64 + 1_000));
    let relaunched = now_ms();
    let mut machine = launch_idle(&root, port, provider_port, idle);
    ready(&mut machine, &root, port);
    let (mut client, cap) = owner(&root, port).await;
    let first = deadline(&mut client, &cap, false).await;
    assert!(first >= relaunched + idle_ms, "the job ended at the restart: {first} vs {relaunched}");
    assert!(received.recv_timeout(Duration::from_millis(idle_ms as u64 / 3)).is_err(), "released early after a restart mid-job");
    drop(client);
    stop(&mut machine);
    let mut machine = launch_idle(&root, port, provider_port, idle);
    ready(&mut machine, &root, port);
    let (mut client, cap) = owner(&root, port).await;
    assert_eq!(deadline(&mut client, &cap, false).await, first, "a restart mid-idle keeps the deadline");
    let wait = (first - now_ms() + 5_000).max(0) as u64;
    received.recv_timeout(Duration::from_millis(wait)).expect("no release at the kept deadline");
    let released = now_ms();
    assert!(released >= first - 500, "released at {released}, before its deadline {first}");
    drop(hub.close);
    drop(machine);
    std::fs::remove_dir_all(root).unwrap();
}
