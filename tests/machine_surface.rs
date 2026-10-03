//! The real `cozy-machine serve` process behind its TLS/gRPC API, called as the CLI calls it.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use cozy_machine::{
    api::{auth::Authority, pb},
    journal::Journal,
};
use ed25519_dalek::{Signer, SigningKey};
use std::{
    collections::BTreeMap,
    fs,
    io::BufReader,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

struct Machine {
    child: Child,
    root: PathBuf,
    client: pb::pod_host_client::PodHostClient<Channel>,
    claim: pb::Claim,
    address: String,
}

const SIGNER: [u8; 32] = [33; 32];
const WORKER: &str = "surface-test";

impl Machine {
    async fn start() -> Self {
        Self::start_with(|_, _| ()).await
    }
    /// `prepare` sees the state root and the test actor's journal id before the machine starts.
    async fn start_with(prepare: impl FnOnce(&Path, &str)) -> Self {
        Self::start_args(prepare, &[]).await
    }
    /// As `start_with`, with more `serve` arguments.
    async fn start_args(prepare: impl FnOnce(&Path, &str), extra: &[std::ffi::OsString]) -> Self {
        let root = std::env::temp_dir().join(format!("cm-surface-{}", uuid::Uuid::new_v4()));
        let config = root.join("config");
        fs::create_dir_all(&config).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let signer = SigningKey::from_bytes(&SIGNER);
        let write = |name: &str, value: serde_json::Value| {
            let path = config.join(name);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        };
        write(
            "keys.json",
            serde_json::json!({"keys":[URL_SAFE_NO_PAD.encode(signer.verifying_key().to_bytes())]}),
        );
        write(
            "readiness.json",
            serde_json::json!({"key_b64url":URL_SAFE_NO_PAD.encode([7u8; 32])}),
        );
        write(
            "machine.json",
            serde_json::json!({"worker_id":WORKER,"identity_directory":"identity",
                "authorized_keys_file":"keys.json","readiness_hmac_key_file":"readiness.json"}),
        );
        prepare(&root.join("state"), &actor());
        let (child, client, claim, address) = launch(&root, extra).await;
        Self {
            child,
            root,
            client,
            claim,
            address,
        }
    }
    /// End the exact process this test started; its state stays for inspection.
    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
    fn store(&self) -> PathBuf {
        self.root.join("state/tensorfs")
    }
    /// One HTTPS GET with curl, pinned to the machine's leaf: (status, headers, body).
    fn get(&self, path: &str, headers: &[&str]) -> (u16, BTreeMap<String, String>, Vec<u8>) {
        let (host, port) = self.address.rsplit_once(':').unwrap();
        let ready: serde_json::Value =
            serde_json::from_slice(&fs::read(self.root.join("state/api-ready.json")).unwrap())
                .unwrap();
        let ca = self.root.join("leaf.pem");
        fs::write(&ca, ready["cert_pem"].as_str().unwrap()).unwrap();
        let mut command = Command::new("curl");
        command
            .args(["-sS", "-D", "-", "--cacert"])
            .arg(&ca)
            .args(["--resolve", &format!("localhost:{port}:{host}")]);
        for header in headers {
            command.args(["-H", header]);
        }
        let output = command
            .arg(format!("https://localhost:{port}{path}"))
            .output()
            .expect("curl runs");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let split = output
            .stdout
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .unwrap();
        let head = String::from_utf8(output.stdout[..split].to_vec()).unwrap();
        let mut lines = head.lines();
        let status = lines
            .next()
            .unwrap()
            .split(' ')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
            .collect();
        (status, headers, output.stdout[split + 4..].to_vec())
    }
    async fn machine_log(
        &mut self,
        log: pb::MachineLog,
        tail_bytes: u64,
    ) -> Result<Vec<u8>, tonic::Status> {
        let mut stream = self
            .client
            .read_machine_log(pb::MachineLogQuery {
                claim: Some(self.claim.clone()),
                log: log as i32,
                tail_bytes,
            })
            .await?
            .into_inner();
        let mut data = vec![];
        while let Some(chunk) = stream.message().await? {
            assert!(!chunk.data.is_empty() && chunk.data.len() <= 64 << 10);
            data.extend(chunk.data);
        }
        Ok(data)
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        self.stop();
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn actor() -> String {
    tensorfs_core::sha256::hex(&SigningKey::from_bytes(&SIGNER).verifying_key().to_bytes())
}

async fn launch(
    root: &Path,
    extra: &[std::ffi::OsString],
) -> (
    Child,
    pb::pod_host_client::PodHostClient<Channel>,
    pb::Claim,
    String,
) {
    let state = root.join("state");
    let _ = fs::remove_file(state.join("api-ready.json"));
    let mut child = Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
        .args(["serve", "--state"])
        .arg(&state)
        .arg("--machine-config")
        .arg(root.join("config/machine.json"))
        .args(["--listen", "127.0.0.1:0", "--host-bytes", "0"])
        .args(extra)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    // Test harness bound on a broken start; the product has no such limit.
    let until = Instant::now() + Duration::from_secs(60);
    let ready = loop {
        if let Ok(bytes) = fs::read(state.join("api-ready.json")) {
            break serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        }
        if Instant::now() >= until {
            let _ = child.kill(); // never leave a started machine behind
            panic!("machine did not publish api-ready.json");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let pem = ready["cert_pem"].as_str().unwrap().to_owned();
    let der = rustls_pemfile::certs(&mut BufReader::new(pem.as_bytes()))
        .next()
        .unwrap()
        .unwrap();
    let authority = Authority {
        worker_id: ready["worker_id"].as_str().unwrap().into(),
        boot_id: ready["boot_id"].as_str().unwrap().into(),
        leaf_digest: tensorfs_core::sha256::digest(der.as_ref()),
        keys: vec![].into(),
    };
    let claim = pb::Claim {
        worker_id: authority.worker_id.clone(),
        worker_boot_id: authority.boot_id.clone(),
        record_owner_epoch: 1,
        proof: SigningKey::from_bytes(&SIGNER)
            .sign(&authority.transcript(1).unwrap())
            .to_bytes()
            .to_vec(),
        ..Default::default()
    };
    let address = ready["address"].as_str().unwrap().to_owned();
    let channel = Endpoint::from_shared(format!("https://{address}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(pem))
                .domain_name("localhost"),
        )
        .unwrap()
        .connect()
        .await
        .unwrap();
    (
        child,
        pb::pod_host_client::PodHostClient::new(channel),
        claim,
        address,
    )
}

#[tokio::test]
async fn machine_log_reads_the_stores_transport_log_with_rotation_and_tail() {
    let mut machine = Machine::start().await;
    let transport = pb::MachineLog::TensorfsTransport;
    assert!(machine.machine_log(transport, 0).await.unwrap().is_empty());

    let logs = machine.store().join("logs");
    fs::create_dir_all(&logs).unwrap();
    fs::write(logs.join("transport.log.1"), "1.000 pull a 1 old\n").unwrap();
    let current: String = (0..3000)
        .map(|n| format!("{n}.000 hedge object-{n} 4096 grant\n"))
        .collect();
    fs::write(logs.join("transport.log"), &current).unwrap();
    let whole = machine.machine_log(transport, 0).await.unwrap();
    assert_eq!(whole, format!("1.000 pull a 1 old\n{current}").into_bytes());
    assert!(whole.len() > 64 << 10, "spans several chunks");

    let tail = String::from_utf8(machine.machine_log(transport, 100).await.unwrap()).unwrap();
    assert!(tail.len() <= 100 && tail.ends_with("2999.000 hedge object-2999 4096 grant\n"));
    // From a line start: the byte before it in the log is a newline.
    assert!(
        current.ends_with(&tail) && current.as_bytes()[current.len() - tail.len() - 1] == b'\n'
    );

    let unknown = machine
        .machine_log(pb::MachineLog::Unspecified, 0)
        .await
        .unwrap_err();
    assert_eq!(unknown.code(), tonic::Code::NotFound);
    let unauthenticated = machine
        .client
        .read_machine_log(pb::MachineLogQuery {
            claim: None,
            log: transport as i32,
            tail_bytes: 0,
        })
        .await
        .unwrap_err();
    assert_eq!(unauthenticated.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn forget_package_drops_the_owners_held_model_resolutions_for_that_package() {
    let mut machine = Machine::start_with(|state, actor| {
        let mut journal = Journal::open(&state.join("execution")).unwrap();
        journal
            .bind_resolution(actor, "sdxl-resolution", "org/sdxl", "plan-a")
            .unwrap();
        journal
            .bind_resolution(actor, "anima-resolution", "org/anima", "plan-b")
            .unwrap();
        journal
            .bind_resolution("another-actor", "other", "org/sdxl", "plan-c")
            .unwrap();
    })
    .await;
    let forget = |package: &str| pb::ForgetPackageCall {
        claim: Some(machine.claim.clone()),
        package: package.into(),
    };
    let (sdxl, nameless) = (forget("org/sdxl"), forget("sdxl"));
    machine.client.forget_package(sdxl.clone()).await.unwrap();
    machine.client.forget_package(sdxl).await.unwrap(); // repeatable
    let refused = machine.client.forget_package(nameless).await.unwrap_err();
    assert_eq!(refused.code(), tonic::Code::InvalidArgument);
    let unauthenticated = machine
        .client
        .forget_package(pb::ForgetPackageCall {
            claim: None,
            package: "org/sdxl".into(),
        })
        .await
        .unwrap_err();
    assert_eq!(unauthenticated.code(), tonic::Code::Unauthenticated);
    machine.stop();
    let journal = Journal::open(&machine.root.join("state/execution")).unwrap();
    let actor = actor();
    assert_eq!(journal.resolution(&actor, "sdxl-resolution").unwrap(), None);
    assert_eq!(
        journal
            .resolution(&actor, "anima-resolution")
            .unwrap()
            .as_deref(),
        Some("plan-b")
    );
    assert_eq!(
        journal
            .resolution("another-actor", "other")
            .unwrap()
            .as_deref(),
        Some("plan-c"),
        "only the calling owner's reads are forgotten"
    );
}

/// A run with a replaced single output and a composite list item, journaled as the device
/// executor's `publish` journals them, and its bytes in the store.
fn seed_run(state: &Path, actor: &str) {
    use cozy_machine::journal::{Invocation, ProcessBirth, SubmissionContext};
    use prost::Message;
    let store = tensorfs_core::store::Store::ensure(&state.join("tensorfs")).unwrap();
    let put = |bytes: &[u8]| -> pb::Ref {
        let object = tensorfs_core::ids::ObjectRef {
            sha256: tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(bytes)),
            length: bytes.len() as u64,
        };
        store
            .put_stream(&mut &bytes[..], Some(&object), &Default::default())
            .unwrap();
        pb::Ref {
            digest: tensorfs_core::sha256::digest(bytes).to_vec(),
            length: bytes.len() as u64,
        }
    };
    let mut journal = Journal::open(&state.join("execution")).unwrap();
    let digest = format!("sha256:{}", "a".repeat(64));
    let record = journal
        .accept_public(
            SubmissionContext {
                actor: actor.into(),
                request_id: "request-1".into(),
                submission_id: "submission-1".into(),
                expected_workspace_id: journal.workspace_id().into(),
                capture_digest: digest.clone(),
                invocation_digest: digest.clone(),
                payload_digest: digest,
                ..Default::default()
            },
            Invocation {
                package: "org/fixture".into(),
                generation: "g".repeat(32),
                module: "fixture:app".into(),
                entrypoint: "make".into(),
                input: serde_json::json!({}),
                attention_kernel: String::new(),
                inputs: Default::default(),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(record.id, "1");
    assert!(journal.claim("1").unwrap());
    // The run's executor: a process of its own (a starting machine ends orphaned executors).
    let mut executor = Command::new("sleep").arg("600").spawn().unwrap();
    let birth = cozy_machine::execution::process_birth(executor.id()).unwrap();
    journal
        .register_process("1", ProcessBirth { ..birth })
        .unwrap();
    std::thread::spawn(move || executor.wait());
    journal.running("1", None).unwrap();
    let set = |content: pb::Ref| pb::RunProduct {
        output: "preview".into(),
        op: pb::RunProductOp::Set as i32,
        content: Some(content),
        media_type: "text/plain".into(),
        ..Default::default()
    };
    for product in [
        set(put(b"preview-draft")),
        pb::RunProduct {
            output: "frames".into(),
            op: pb::RunProductOp::Append as i32,
            content: Some(pb::Ref {
                digest: tensorfs_core::sha256::digest(b"abcdefg").to_vec(),
                length: 7,
            }),
            parts: [&b"abc"[..], b"defg"]
                .into_iter()
                .map(|part| pb::RunProductPart {
                    content: Some(put(part)),
                    duration_us: 1_000_000,
                    ..Default::default()
                })
                .collect(),
            media_type: "video/mp4".into(),
            ..Default::default()
        },
        set(put(b"preview-final")),
    ] {
        journal
            .append_product("1", None, &product.encode_to_vec())
            .unwrap();
    }
}

#[tokio::test]
async fn run_outputs_over_https_need_a_capability_and_serve_revisions_ranges_and_final_digests() {
    use cozy_machine::api::capability::{mint, Grant};
    let machine = Machine::start().await;
    // Journaled while the machine runs, as its device executor journals products; a run seeded
    // before start would be settled by the machine's orphan recovery.
    seed_run(&machine.root.join("state"), &actor());
    let signer = SigningKey::from_bytes(&SIGNER);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let grant = |outputs: &[&str], machine: &str, expires: i64| {
        let token = mint(
            &signer,
            Grant {
                machine: machine.into(),
                run: "1".into(),
                outputs: outputs.iter().map(|o| o.to_string()).collect(),
                expires,
                ..Default::default()
            },
        );
        format!("Authorization: Cozy-Cap {token}")
    };
    let all = grant(&[], WORKER, now + 600);

    // The replaced single output: its current bytes and revision, not yet final.
    let (status, headers, body) = machine.get("/v1/runs/1/outputs/preview", &[all.as_str()]);
    assert_eq!((status, body.as_slice()), (200, &b"preview-final"[..]));
    assert_eq!(headers["etag"], "\"r2\"");
    assert_eq!(headers["content-type"], "text/plain");
    assert!(!headers.contains_key("repr-digest"));
    let (status, _, body) = machine.get(
        "/v1/runs/1/outputs/preview",
        &[all.as_str(), "If-None-Match: \"r2\""],
    );
    assert_eq!((status, body.len()), (304, 0));

    // A composite list item (1-based) is its parts in order; ranges cross part boundaries.
    let (status, _, body) = machine.get("/v1/runs/1/outputs/frames/1", &[all.as_str()]);
    assert_eq!((status, body.as_slice()), (200, &b"abcdefg"[..]));
    let (status, headers, body) = machine.get(
        "/v1/runs/1/outputs/frames/1",
        &[all.as_str(), "Range: bytes=2-4"],
    );
    assert_eq!((status, body.as_slice()), (206, &b"cde"[..]));
    assert_eq!(headers["content-range"], "bytes 2-4/7");
    let (status, _, body) = machine.get(
        "/v1/runs/1/outputs/frames/1",
        &[all.as_str(), "Range: bytes=-3"],
    );
    assert_eq!((status, body.as_slice()), (206, &b"efg"[..]));
    let (status, _, _) = machine.get(
        "/v1/runs/1/outputs/frames/1",
        &[all.as_str(), "Range: bytes=9-"],
    );
    assert_eq!(status, 416);

    // Absent outputs and runs, and every capability refusal.
    for path in [
        "/v1/runs/1/outputs/frames/2",
        "/v1/runs/1/outputs/other",
        "/v1/runs/1/outputs/frames/0",
    ] {
        assert_eq!(machine.get(path, &[all.as_str()]).0, 404, "{path}");
    }
    // A capability names its run: another run is out of scope before it is looked up.
    assert_eq!(
        machine.get("/v1/runs/2/outputs/preview", &[all.as_str()]).0,
        403
    );
    let scoped = grant(&["frames"], WORKER, now + 600);
    assert_eq!(
        machine
            .get("/v1/runs/1/outputs/frames/1", &[scoped.as_str()])
            .0,
        200
    );
    for headers in [
        vec![],
        vec![scoped],
        vec![grant(&[], "another-machine", now + 600)],
        vec![grant(&[], WORKER, now - 1)],
        vec!["Authorization: Cozy-Cap not-a-token".into()],
    ] {
        let headers: Vec<&str> = headers.iter().map(String::as_str).collect();
        assert_eq!(machine.get("/v1/runs/1/outputs/preview", &headers).0, 403);
    }

    // Terminal: the same bytes are final and carry their digest.
    Journal::open(&machine.root.join("state/execution"))
        .unwrap()
        .finish(
            "1",
            cozy_machine::journal::Outcome::Failed("the fixture run settled".into()),
        )
        .unwrap();
    let (status, headers, body) = machine.get("/v1/runs/1/outputs/preview", &[all.as_str()]);
    assert_eq!((status, body.as_slice()), (200, &b"preview-final"[..]));
    use base64::engine::general_purpose::STANDARD;
    assert_eq!(
        headers["repr-digest"],
        format!(
            "sha-256=:{}:",
            STANDARD.encode(tensorfs_core::sha256::digest(b"preview-final"))
        )
    );
}

/// A real uv-installed generation of `tests/fixtures/cpu_classifier`, bound as the actor's
/// installation `fixture`: what `PrepareLocalPackage` leaves behind.
fn hold_classifier(state: &Path, actor: &str) {
    const INSTALL: &str = r#"
import json, subprocess, sys
from pathlib import Path
from cozy_machine_client.packages import install
fixture, out = Path(sys.argv[1]), Path(sys.argv[2])
subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out / "client"), "."], check=True, capture_output=True)
generation = install(fixture, out / "generations", next((out / "client").glob("*.whl")), python="3.12")
print(json.dumps({"identity": generation.identity}))
"#;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::create_dir_all(state).unwrap();
    let output = Command::new("uv")
        .current_dir(repo)
        .args([
            "run", "--locked", "--extra", "test", "python", "-c", INSTALL,
        ])
        .arg(repo.join("tests/fixtures/cpu_classifier"))
        .arg(state)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(line.lines().last().unwrap()).unwrap();
    let identity = value["identity"].as_str().unwrap();
    let generation: serde_json::Value = serde_json::from_slice(
        &fs::read(
            state
                .join("generations")
                .join(identity)
                .join("generation.json"),
        )
        .unwrap(),
    )
    .unwrap();
    Journal::open(&state.join("execution"))
        .unwrap()
        .bind_installation(cozy_machine::journal::Installation {
            actor: actor.into(),
            alias: "fixture".into(),
            generation: identity.into(),
            package: generation["package"].as_str().unwrap().into(),
            release: generation["version"].as_str().unwrap().into(),
            interface: serde_json::to_vec(&generation["interface"]).unwrap(),
        })
        .unwrap();
}

mod v1_api {
    use super::*;
    use cozy_machine::api::{
        capability::{mint, Grant, MACHINE},
        v1,
    };
    use tonic::metadata::MetadataValue;

    fn cap(scope: Grant) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        mint(
            &SigningKey::from_bytes(&SIGNER),
            Grant {
                machine: WORKER.into(),
                expires: now + 600,
                ..scope
            },
        )
    }
    fn authorized<T>(message: T, cap: &str) -> tonic::Request<T> {
        let mut request = tonic::Request::new(message);
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Cozy-Cap {cap}")).unwrap(),
        );
        request
    }
    async fn client(machine: &Machine) -> v1::machine_client::MachineClient<Channel> {
        let ready: serde_json::Value =
            serde_json::from_slice(&fs::read(machine.root.join("state/api-ready.json")).unwrap())
                .unwrap();
        let channel = Endpoint::from_shared(format!("https://{}", machine.address))
            .unwrap()
            .initial_stream_window_size(16 << 20)
            .initial_connection_window_size(32 << 20)
            .tls_config(
                ClientTlsConfig::new()
                    .ca_certificate(Certificate::from_pem(ready["cert_pem"].as_str().unwrap()))
                    .domain_name("localhost"),
            )
            .unwrap()
            .connect()
            .await
            .unwrap();
        v1::machine_client::MachineClient::new(channel).max_decoding_message_size(16 << 20)
    }
    async fn collect(
        stream: tonic::Streaming<v1::RunEvent>,
    ) -> Result<Vec<v1::RunEvent>, tonic::Status> {
        let mut stream = stream;
        let mut events = vec![];
        while let Some(event) = stream.message().await? {
            events.push(event);
        }
        Ok(events)
    }
    fn spec(iterations: u64) -> v1::RunSpec {
        v1::RunSpec {
            source: Some(v1::run_spec::Source::Installation("fixture".into())),
            entrypoint: "classify".into(),
            payload: serde_json::to_vec(&serde_json::json!({
                "samples": [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 5.1, 1.6]],
                "iterations": iterations,
            }))
            .unwrap(),
            ..Default::default()
        }
    }
    async fn read(
        client: &mut v1::machine_client::MachineClient<Channel>,
        cap: &str,
        request: v1::ReadRequest,
    ) -> Result<(v1::ReadFrame, Vec<u8>), tonic::Status> {
        let mut stream = client.read(authorized(request, cap)).await?.into_inner();
        let meta = stream.message().await?.unwrap();
        let mut data = vec![];
        while let Some(frame) = stream.message().await? {
            assert!(!frame.data.is_empty() && frame.data.len() <= 1 << 20);
            data.extend(frame.data);
        }
        Ok((meta, data))
    }

    #[tokio::test]
    async fn run_streams_a_real_run_to_its_outcome_and_read_returns_its_output() {
        let machine = Machine::start_with(hold_classifier).await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let request = |spec: Option<v1::RunSpec>, after| v1::RunRequest {
            id: "run-1".into(),
            after,
            spec,
        };
        let events = collect(
            client
                .run(authorized(request(Some(spec(3)), 0), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        // The first frame is the run's state (a snapshot, sequence 0); the log ends at the outcome.
        let Some(v1::run_event::Event::State(first)) = &events[0].event else {
            panic!("{events:?}")
        };
        assert_eq!((events[0].sequence, first.id.as_str()), (0, "run-1"));
        let number = first.number;
        assert!(number > 0);
        let Some(v1::run_event::Event::Outcome(outcome)) = &events.last().unwrap().event else {
            panic!("{events:?}")
        };
        assert_eq!(outcome.status, "succeeded", "{outcome:?}");
        let result: serde_json::Value = serde_json::from_slice(&outcome.result).unwrap();
        assert_eq!(result["predictions"], serde_json::json!([0, 2]));
        assert_eq!(result["iterations"], 3);
        let report = outcome
            .outputs
            .iter()
            .find(|p| p.output == "report")
            .unwrap();
        assert_eq!((report.index, report.rev), (0, 1));
        let logged: Vec<u64> = events.iter().skip(1).map(|e| e.sequence).collect();
        assert!(logged.windows(2).all(|w| w[0] < w[1]), "{logged:?}");

        // Read: the output's bytes, digest and revision; an offset resumes.
        let target = |offset| v1::ReadRequest {
            target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                run: "run-1".into(),
                output: "report".into(),
                index: 0,
            })),
            offset,
            ..Default::default()
        };
        let (meta, bytes) = read(&mut client, &all, target(0)).await.unwrap();
        assert_eq!((meta.rev, meta.length), (1, bytes.len() as u64));
        assert_eq!(meta.length, report.length);
        let digest = format!(
            "sha256:{}",
            tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(&bytes))
        );
        assert_eq!(
            (meta.digest.as_str(), report.digest.as_str()),
            (digest.as_str(), digest.as_str())
        );
        let report_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report_json["predictions"], serde_json::json!([0, 2]));
        let (_, tail) = read(&mut client, &all, target(5)).await.unwrap();
        assert_eq!(tail, bytes[5..]);
        let stale = read(
            &mut client,
            &all,
            v1::ReadRequest {
                if_rev: 7,
                ..target(0)
            },
        )
        .await
        .unwrap_err();
        assert_eq!(stale.code(), tonic::Code::FailedPrecondition);

        // The same id and spec again is the same run; attach without a spec replays the log.
        let again = collect(
            client
                .run(authorized(request(Some(spec(3)), 0), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let Some(v1::run_event::Event::State(state)) = &again[0].event else {
            panic!()
        };
        assert_eq!(state.number, number);
        let tail_log = collect(
            client
                .run(authorized(
                    request(None, events[events.len() - 2].sequence),
                    &all,
                ))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        assert_eq!(
            tail_log.len(),
            2,
            "a snapshot, then only the outcome after the cursor"
        );

        // A run-scope cap may attach to and read its run, and nothing else.
        let scoped = cap(Grant {
            run: "run-1".into(),
            ..Default::default()
        });
        assert!(collect(
            client
                .run(authorized(request(None, 0), &scoped))
                .await
                .unwrap()
                .into_inner()
        )
        .await
        .is_ok());
        assert!(read(&mut client, &scoped, target(0)).await.is_ok());
        let submit = client
            .run(authorized(
                v1::RunRequest {
                    id: "run-2".into(),
                    after: 0,
                    spec: Some(spec(1)),
                },
                &scoped,
            ))
            .await
            .unwrap_err();
        assert_eq!(submit.code(), tonic::Code::PermissionDenied);
        let cancel = client
            .control(authorized(
                v1::ControlRequest {
                    id: "run-1".into(),
                    action: v1::Action::Cancel as i32,
                },
                &scoped,
            ))
            .await
            .unwrap_err();
        assert_eq!(cancel.code(), tonic::Code::PermissionDenied);
        let bare = client
            .run(tonic::Request::new(request(None, 0)))
            .await
            .unwrap_err();
        assert_eq!(bare.code(), tonic::Code::Unauthenticated);
    }

    #[tokio::test]
    async fn control_cancels_a_running_run_cooperatively() {
        let machine = Machine::start_with(hold_classifier).await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let mut stream = client
            .run(authorized(
                v1::RunRequest {
                    id: "long".into(),
                    after: 0,
                    spec: Some(spec(50_000_000)),
                },
                &all,
            ))
            .await
            .unwrap()
            .into_inner();
        // Cancel once the running run reports progress: it is executing package code.
        let mut running = false;
        loop {
            let event = stream
                .message()
                .await
                .unwrap()
                .expect("run ended before progress");
            match event.event {
                Some(v1::run_event::Event::State(state)) if state.state == "running" => {
                    running = true
                }
                Some(v1::run_event::Event::Progress(_)) if running => break,
                _ => {}
            }
        }
        client
            .control(authorized(
                v1::ControlRequest {
                    id: "long".into(),
                    action: v1::Action::Cancel as i32,
                },
                &all,
            ))
            .await
            .unwrap();
        let rest = collect(stream).await.unwrap();
        let Some(v1::run_event::Event::Outcome(outcome)) = &rest.last().unwrap().event else {
            panic!("{rest:?}")
        };
        assert_eq!(outcome.status, "canceled", "{outcome:?}");
        assert_eq!(outcome.reason.as_ref().unwrap().code, "canceled");
    }

    #[tokio::test]
    async fn status_lists_the_callers_live_runs_and_held_environments() {
        let machine = Machine::start_with(hold_classifier).await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let request = v1::RunRequest {
            id: "live".into(),
            after: 0,
            spec: Some(spec(50_000_000)),
        };
        let mut stream = client
            .run(authorized(request, &all))
            .await
            .unwrap()
            .into_inner();
        // Progress once it runs (a run also reports progress while it prepares).
        let mut running = false;
        loop {
            match stream.message().await.unwrap().expect("run ended").event {
                Some(v1::run_event::Event::State(state)) if state.state == "running" => {
                    running = true
                }
                Some(v1::run_event::Event::Progress(_)) if running => break,
                _ => {}
            }
        }
        let mut status = client
            .status(authorized(v1::StatusRequest { keepalive: false }, &all))
            .await
            .unwrap()
            .into_inner();
        let frame = status.message().await.unwrap().unwrap();
        let live: Vec<_> = frame
            .runs
            .iter()
            .map(|r| (r.id.as_str(), r.state.as_str()))
            .collect();
        assert_eq!(live, [("live", "running")], "{frame:?}");
        let held: Vec<_> = frame
            .environments
            .iter()
            .map(|e| e.installation.as_str())
            .collect();
        assert_eq!(held, ["fixture"], "{frame:?}");
        let cancel = v1::ControlRequest {
            id: "live".into(),
            action: v1::Action::Cancel as i32,
        };
        client.control(authorized(cancel, &all)).await.unwrap();
        collect(stream).await.unwrap();
        // The held Status stream sends the change: no live run remains.
        let until = Instant::now() + Duration::from_secs(30);
        loop {
            let next = tokio::time::timeout_at(until.into(), status.message())
                .await
                .expect("Status sent no frame without the run")
                .unwrap()
                .unwrap();
            if next.runs.is_empty() {
                break;
            }
        }
    }

    async fn write(
        client: &mut v1::machine_client::MachineClient<Channel>,
        cap: &str,
        digest: &str,
        length: u64,
        offset: u64,
        data: &[u8],
    ) -> Result<u64, tonic::Status> {
        let mut frames = vec![v1::WriteFrame {
            digest: digest.into(),
            length,
            offset,
            data: vec![],
        }];
        frames.extend(data.chunks(64 << 10).map(|chunk| v1::WriteFrame {
            data: chunk.to_vec(),
            ..Default::default()
        }));
        let request = authorized(tokio_stream::iter(frames), cap);
        Ok(client.write(request).await?.into_inner().held)
    }

    /// A serve process that installs unpublished code (the client wheel built from this repo).
    async fn installing_machine() -> (Machine, PathBuf) {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tools = std::env::temp_dir().join(format!("cm-tools-{}", uuid::Uuid::new_v4()));
        assert!(Command::new("uv")
            .current_dir(repo)
            .args(["build", "--wheel", "--out-dir"])
            .arg(&tools)
            .status()
            .unwrap()
            .success());
        let machine = Machine::start_args(|_, _| (), &installing_args(&tools)).await;
        (machine, tools)
    }

    /// `serve`'s installer arguments: the test helper Python and the client wheel in `tools`.
    fn installing_args(tools: &Path) -> Vec<std::ffi::OsString> {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let wheel = fs::read_dir(tools)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "whl"))
            .unwrap();
        let helper = Command::new("uv")
            .current_dir(repo)
            .args([
                "run",
                "--locked",
                "--extra",
                "test",
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ])
            .output()
            .unwrap();
        let helper = String::from_utf8(helper.stdout).unwrap().trim().to_string();
        vec![
            "--installer-python".into(),
            helper.into(),
            "--client-wheel".into(),
            wheel.into(),
        ]
    }

    /// A fixture package written with Write: its manifest's digest.
    async fn write_package(
        client: &mut v1::machine_client::MachineClient<Channel>,
        cap: &str,
        fixture: &str,
        package: &str,
    ) -> String {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        let mut archive = tar::Builder::new(Vec::new());
        for name in [
            "pyproject.toml",
            "package.toml",
            &format!("{fixture}/__init__.py"),
        ] {
            archive
                .append_path_with_name(directory.join(name), name)
                .unwrap();
        }
        let source = archive.into_inner().unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&source));
        let length = source.len() as u64;
        assert_eq!(
            write(client, cap, &digest, length, 0, &source)
                .await
                .unwrap(),
            length
        );
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package": package, "release": "0.1.0",
            "python_version": "3.12", "source": {"digest": digest, "length": length}}))
        .unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        let size = manifest.len() as u64;
        assert_eq!(
            write(client, cap, &manifest_digest, size, 0, &manifest)
                .await
                .unwrap(),
            size
        );
        manifest_digest
    }

    fn outcome(events: &[v1::RunEvent]) -> v1::Outcome {
        match &events.last().unwrap().event {
            Some(v1::run_event::Event::Outcome(outcome)) => outcome.clone(),
            _ => panic!("{events:?}"),
        }
    }

    /// Write and run sources on the real serve process: an object resumes from what is held,
    /// and unpublished code written with Write prepares inside its run (warm, then a call).
    #[tokio::test]
    async fn written_local_code_prepares_inside_its_run() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });

        let mut archive = tar::Builder::new(Vec::new());
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_lifecycle");
        for name in [
            "pyproject.toml",
            "package.toml",
            "cpu_lifecycle/__init__.py",
        ] {
            archive
                .append_path_with_name(fixture.join(name), name)
                .unwrap();
        }
        let source = archive.into_inner().unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&source));
        let length = source.len() as u64;
        // A probe holds nothing; half, then the rest from the held offset.
        assert_eq!(
            write(&mut client, &all, &digest, length, 0, &[])
                .await
                .unwrap(),
            0
        );
        let half = length / 2;
        assert_eq!(
            write(
                &mut client,
                &all,
                &digest,
                length,
                0,
                &source[..half as usize]
            )
            .await
            .unwrap(),
            half
        );
        let ahead = write(&mut client, &all, &digest, length, half + 1, b"x")
            .await
            .unwrap_err();
        assert_eq!(
            ahead.metadata().get("cozy-error-code").unwrap(),
            "object_offset_ahead"
        );
        assert_eq!(
            write(
                &mut client,
                &all,
                &digest,
                length,
                half,
                &source[half as usize..]
            )
            .await
            .unwrap(),
            length
        );
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package": "local/cozy-machine-cpu-lifecycle", "release": "0.1.0",
            "python_version": "3.12", "source": {"digest": digest, "length": length}}))
        .unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        let size = manifest.len() as u64;
        assert_eq!(
            write(&mut client, &all, &manifest_digest, size, 0, &manifest)
                .await
                .unwrap(),
            size
        );

        let local = |kind: v1::RunKind| v1::RunSpec {
            kind: kind as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource {
                manifest: manifest_digest.clone(),
            })),
            entrypoint: "steps".into(),
            payload: br#"{"steps":2,"seconds":0.01}"#.to_vec(),
            owner: "alice".into(),
            ..Default::default()
        };
        let run = |id: &str, spec: v1::RunSpec| v1::RunRequest {
            id: id.into(),
            after: 0,
            spec: Some(spec),
        };
        let warm = collect(
            client
                .run(authorized(run("warm-1", local(v1::RunKind::Warm)), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        assert_eq!(outcome(&warm).status, "succeeded", "{warm:?}");
        // A warm run naming no entrypoint installs the code alone (`cozy package install`).
        let mut install = local(v1::RunKind::Warm);
        install.entrypoint.clear();
        install.payload.clear();
        let installed = collect(
            client
                .run(authorized(run("warm-2", install), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let installed = outcome(&installed);
        assert_eq!(installed.status, "succeeded", "{installed:?}");
        let result: serde_json::Value = serde_json::from_slice(&installed.result).unwrap();
        assert_eq!(result["package"], "local/cozy-machine-cpu-lifecycle");
        assert_eq!(result["models"], serde_json::json!([]));
        let called = collect(
            client
                .run(authorized(run("call-1", local(v1::RunKind::Call)), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let done = outcome(&called);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["steps"], 2);
        // The id names that spec: another spec under it is refused.
        let mut other = local(v1::RunKind::Call);
        other.payload = br#"{"steps":3}"#.to_vec();
        let conflict = client
            .run(authorized(run("call-1", other), &all))
            .await
            .unwrap()
            .into_inner()
            .message()
            .await
            .unwrap_err();
        assert_eq!(conflict.code(), tonic::Code::AlreadyExists);
        let _ = fs::remove_dir_all(tools);
    }

    /// H3 long-form's shape on CPU: a job (`kind: job`) renders each segment through a child run
    /// of its own invocable, handing it the job's file input; each child's file comes back into
    /// the job's spool, and the job publishes the film after every segment.
    #[tokio::test]
    async fn a_job_renders_its_segments_through_child_runs() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &mut client,
            &all,
            "cpu_longform",
            "local/cozy-machine-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[7; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        assert_eq!(
            write(&mut client, &all, &digest, length, 0, &reference)
                .await
                .unwrap(),
            length
        );
        let segments = ["a dawn", "a storm", "a calm"];
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: serde_json::to_vec(
                &serde_json::json!({"reference": digest, "segments": segments}),
            )
            .unwrap(),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(
            client
                .run(authorized(
                    v1::RunRequest {
                        id: "film".into(),
                        after: 0,
                        spec: Some(spec.clone()),
                    },
                    &all,
                ))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["segments"], 3, "{result}");

        // Each segment saw the job's reference and continued the previous segment's context.
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let mut film = String::new();
        let mut context = String::new();
        for (index, prompt) in segments.iter().enumerate() {
            let body = format!("{index}|{prompt}|{seen}|{context}\n");
            context = tensorfs_core::sha256::hex_digest(body.as_bytes());
            film.push_str(&body);
        }
        // The film after every segment is a product of the job's `video` output.
        let products: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Product(p)) if p.output == "video" => Some(p.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            products
                .iter()
                .map(|p| p.label.as_str())
                .collect::<Vec<_>>(),
            [
                "Video (segments 1-1)",
                "Video (segments 1-2)",
                "Video (segments 1-3)"
            ],
            "{events:?}"
        );
        let (meta, bytes) = read(
            &mut client,
            &all,
            v1::ReadRequest {
                target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                    run: "film".into(),
                    output: "video".into(),
                    index: 0,
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), film, "{meta:?}");

        // Every child is a run of its own under the job, settled.
        for index in 0..3 {
            let child = collect(
                client
                    .run(authorized(
                        v1::RunRequest {
                            id: format!("film/{index}"),
                            after: 0,
                            spec: None,
                        },
                        &all,
                    ))
                    .await
                    .unwrap()
                    .into_inner(),
            )
            .await
            .unwrap();
            assert_eq!(outcome(&child).status, "succeeded", "{child:?}");
        }

        // A canceled job ends its running child with it.
        let mut held = spec.clone();
        held.payload = serde_json::to_vec(
            &serde_json::json!({"reference": digest, "segments": segments, "hold": 60.0}),
        )
        .unwrap();
        let stream = client
            .run(authorized(
                v1::RunRequest {
                    id: "held".into(),
                    after: 0,
                    spec: Some(held),
                },
                &all,
            ))
            .await
            .unwrap()
            .into_inner();
        let watch = |id: &str| v1::RunRequest {
            id: id.into(),
            after: 0,
            spec: None,
        };
        let until = Instant::now() + Duration::from_secs(60);
        loop {
            let child = client.run(authorized(watch("held/0"), &all)).await;
            let running = match child {
                Ok(child) => match child.into_inner().message().await {
                    Ok(Some(v1::RunEvent {
                        event: Some(v1::run_event::Event::State(s)),
                        ..
                    })) => s.state == "running",
                    _ => false,
                },
                Err(_) => false,
            };
            if running {
                break;
            }
            assert!(Instant::now() < until, "the first child never ran");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let cancel = v1::ControlRequest {
            id: "held".into(),
            action: v1::Action::Cancel as i32,
        };
        client.control(authorized(cancel, &all)).await.unwrap();
        let parent = collect(stream).await.unwrap();
        assert_eq!(outcome(&parent).status, "canceled", "{parent:?}");
        let child = collect(
            client
                .run(authorized(watch("held/0"), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        assert_eq!(outcome(&child).status, "canceled", "{child:?}");

        // A job's model choices address its callables (`<entrypoint>.models.<parameter>`), and
        // each child prepares with the ones addressed to it.
        let chose = |id: &str, parameter: &str| {
            let mut chosen = spec.clone();
            chosen.models = vec![v1::ModelChoice {
                parameter: parameter.into(),
                repository: "alice/model".into(),
                ..Default::default()
            }];
            v1::RunRequest {
                id: id.into(),
                after: 0,
                spec: Some(chosen),
            }
        };
        let stray = collect(
            client
                .run(authorized(chose("stray", "nope.models.base"), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let stray = outcome(&stray);
        assert_eq!(
            stray.reason.as_ref().map(|r| r.code.as_str()),
            Some("invalid_request"),
            "{stray:?}"
        );
        collect(
            client
                .run(authorized(
                    chose("chosen", "render_segment.models.base"),
                    &all,
                ))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let child = collect(
            client
                .run(authorized(watch("chosen/0"), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        // render_segment declares no model slot, so the choice it was handed is refused there.
        let refused = outcome(&child);
        assert!(
            refused.reason.as_ref().is_some_and(|r| r
                .message
                .contains("model choices name no declared model slot")),
            "{refused:?}"
        );
        let _ = fs::remove_dir_all(tools);
    }

    /// The run's state as Run's first frame (a snapshot); None while it does not exist.
    async fn snapshot(
        client: &mut v1::machine_client::MachineClient<Channel>,
        cap: &str,
        id: &str,
    ) -> Option<v1::RunState> {
        let request = v1::RunRequest {
            id: id.into(),
            after: 0,
            spec: None,
        };
        match client.run(authorized(request, cap)).await {
            Ok(stream) => match stream.into_inner().message().await {
                Ok(Some(v1::RunEvent {
                    event: Some(v1::run_event::Event::State(state)),
                    ..
                })) => Some(state),
                _ => None,
            },
            Err(_) => None,
        }
    }

    /// Polls until the run shows `wanted`.
    async fn until(
        client: &mut v1::machine_client::MachineClient<Channel>,
        cap: &str,
        id: &str,
        wanted: &str,
    ) -> v1::RunState {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let seen = snapshot(client, cap, id).await;
            if let Some(state) = seen.as_ref().filter(|s| s.state == wanted) {
                return state.clone();
            }
            assert!(Instant::now() < deadline, "{id} never showed {wanted}: {seen:?}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Pause and resume (Control): a job pauses while its second segment runs. Its root stops,
    /// the started segment runs to its end and the unstarted one waits; the job rests `paused`,
    /// keeping its scratch and checkpoint declarations, across a machine restart too. Resume
    /// replays only the root: its calls find the finished segments, so no segment runs twice,
    /// and the film is whole.
    #[tokio::test]
    async fn a_paused_job_resumes_from_its_finished_children() {
        let (mut machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &mut client,
            &all,
            "cpu_longform",
            "local/cozy-machine-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[9; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        write(&mut client, &all, &digest, length, 0, &reference)
            .await
            .unwrap();
        let segments = ["a dawn", "a storm", "a calm"];
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: serde_json::to_vec(&serde_json::json!({
                "reference": digest, "segments": segments, "hold": 4.0, "hold_at": 1}))
            .unwrap(),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let control = |action: v1::Action, id: &str| {
            authorized(
                v1::ControlRequest {
                    id: id.into(),
                    action: action as i32,
                },
                &all,
            )
        };
        let code = |status: tonic::Status| {
            status
                .metadata()
                .get("cozy-error-code")
                .map(|v| v.to_str().unwrap().to_string())
        };
        client
            .run(authorized(
                v1::RunRequest {
                    id: "film".into(),
                    after: 0,
                    spec: Some(spec.clone()),
                },
                &all,
            ))
            .await
            .unwrap();
        until(&mut client, &all, "film/1", "running").await;
        let paused = client
            .control(control(v1::Action::Pause, "film"))
            .await
            .unwrap()
            .into_inner();
        assert!(["running", "paused"].contains(&paused.state.as_str()), "{paused:?}");
        let rest = until(&mut client, &all, "film", "paused").await;
        assert_eq!(rest.attempt, 1);
        // A call does not pause, and a job still pausing does not resume.
        let call = client.control(control(v1::Action::Pause, "film/1")).await;
        assert_eq!(call.err().and_then(code).as_deref(), Some("pause_unsupported"));

        // The started segment runs to its end; the unstarted one never starts while paused.
        let second = until(&mut client, &all, "film/1", "succeeded").await;
        assert_eq!(second.attempt, 1);
        assert!(snapshot(&mut client, &all, "film/2").await.is_none());
        assert_eq!(snapshot(&mut client, &all, "film").await.unwrap().state, "paused");
        let scratch = machine
            .root
            .join("state/cpu/scratch")
            .join(rest.number.to_string());
        assert_eq!(
            fs::read_to_string(scratch.join("checkpoints/long_form/attempts")).unwrap(),
            "1"
        );

        // A paused job survives a machine restart, scratch and all.
        machine.stop();
        let extra = installing_args(&tools);
        let (child, _, _, address) = launch(&machine.root, &extra).await;
        machine.child = child;
        machine.address = address;
        let mut client = self::client(&machine).await;
        assert_eq!(snapshot(&mut client, &all, "film").await.unwrap().state, "paused");
        assert!(scratch.join("checkpoints/long_form/film-0").exists());

        let resumed = client
            .control(control(v1::Action::Resume, "film"))
            .await
            .unwrap()
            .into_inner();
        assert!(["queued", "running"].contains(&resumed.state.as_str()), "{resumed:?}");
        let events = collect(
            client
                .run(authorized(
                    v1::RunRequest {
                        id: "film".into(),
                        after: 0,
                        spec: None,
                    },
                    &all,
                ))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        // Two attempts counted in the run's scratch; the first attempt had declared the first
        // segment's checkpoint (it stopped while the second ran), so that declaration replayed.
        assert_eq!(
            (result["segments"].as_u64(), result["attempts"].as_u64(), result["replayed"].as_u64()),
            (Some(3), Some(2), Some(1)),
            "{result}"
        );
        let states: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::State(s)) => Some((s.state.clone(), s.attempt)),
                _ => None,
            })
            .collect();
        assert!(states.contains(&("running".into(), 2)), "{states:?}");

        // Each segment ran exactly once, and the film is every segment in order.
        for index in 0..3 {
            let child = snapshot(&mut client, &all, &format!("film/{index}"))
                .await
                .unwrap();
            assert_eq!((child.state.as_str(), child.attempt), ("succeeded", 1), "{child:?}");
        }
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let (mut film, mut context) = (String::new(), String::new());
        for (index, prompt) in segments.iter().enumerate() {
            let body = format!("{index}|{prompt}|{seen}|{context}\n");
            context = tensorfs_core::sha256::hex_digest(body.as_bytes());
            film.push_str(&body);
        }
        let (_, bytes) = read(
            &mut client,
            &all,
            v1::ReadRequest {
                target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                    run: "film".into(),
                    output: "video".into(),
                    index: 0,
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), film);
        // The replayed root's identical publishes add nothing to the output log.
        let labels: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Product(p)) if p.output == "video" => {
                    Some(p.label.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            labels,
            ["Video (segments 1-1)", "Video (segments 1-2)", "Video (segments 1-3)"]
        );
        // Nor do its list items: each segment is one item of `parts`, in order.
        let parts: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Product(p)) if p.output == "parts" => {
                    Some((p.index, p.label.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            parts,
            [(1, "Segment 1".into()), (2, "Segment 2".into()), (3, "Segment 3".into())]
        );
        let again = client.control(control(v1::Action::Resume, "film")).await;
        assert_eq!(again.err().and_then(code).as_deref(), Some("run_not_paused"));

        // A paused job's cancel ends it and every unfinished child, started ones too.
        let mut dropped = spec.clone();
        dropped.payload = serde_json::to_vec(&serde_json::json!({
            "reference": digest, "segments": segments, "hold": 60.0, "hold_at": 0}))
        .unwrap();
        client
            .run(authorized(
                v1::RunRequest {
                    id: "dropped".into(),
                    after: 0,
                    spec: Some(dropped),
                },
                &all,
            ))
            .await
            .unwrap();
        until(&mut client, &all, "dropped/0", "running").await;
        client.control(control(v1::Action::Pause, "dropped")).await.unwrap();
        until(&mut client, &all, "dropped", "paused").await;
        assert_eq!(snapshot(&mut client, &all, "dropped/0").await.unwrap().state, "running");
        let canceled = client
            .control(control(v1::Action::Cancel, "dropped"))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(canceled.state, "canceled");
        until(&mut client, &all, "dropped/0", "canceled").await;

        // The ended run's scratch goes with the next sweep (one runs at start).
        machine.stop();
        let (child, _, _, _) = launch(&machine.root, &extra).await;
        machine.child = child;
        assert!(!scratch.exists());
        let _ = fs::remove_dir_all(tools);
    }

    /// H3 long-form's failure contract on CPU: a segment that fails in authored code fails its
    /// child run with that reason, the job fails with `segment_failed` naming the segment, and
    /// the film it published before the failure stays readable.
    #[tokio::test]
    async fn a_failed_segment_fails_the_job_and_keeps_the_film_so_far() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &mut client,
            &all,
            "cpu_longform",
            "local/cozy-machine-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[9; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        write(&mut client, &all, &digest, length, 0, &reference)
            .await
            .unwrap();
        let segments = ["a dawn", "a storm", "a calm"];
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: serde_json::to_vec(
                &serde_json::json!({"reference": digest, "segments": segments, "fail_at": 1}),
            )
            .unwrap(),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(
            client
                .run(authorized(
                    v1::RunRequest {
                        id: "broken".into(),
                        after: 0,
                        spec: Some(spec),
                    },
                    &all,
                ))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "failed", "{done:?}");
        let reason = done.reason.clone().unwrap_or_default();
        assert!(
            reason.code.contains("segment_failed") || reason.message.contains("segment_failed"),
            "{reason:?}"
        );
        assert!(reason.message.contains("segment 2 of 3"), "{reason:?}");
        // The child that failed is a failed run with the authored reason; the one before it
        // succeeded; no third segment ran.
        let watch = |id: &str| v1::RunRequest {
            id: id.into(),
            after: 0,
            spec: None,
        };
        let first = collect(
            client
                .run(authorized(watch("broken/0"), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        assert_eq!(outcome(&first).status, "succeeded", "{first:?}");
        let second = collect(
            client
                .run(authorized(watch("broken/1"), &all))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let failed = outcome(&second);
        assert_eq!(failed.status, "failed", "{second:?}");
        assert!(
            format!("{:?}", failed.reason).contains("cannot be rendered"),
            "{failed:?}"
        );
        // No third segment ran.
        match client.run(authorized(watch("broken/2"), &all)).await {
            Err(_) => {}
            Ok(stream) => assert!(collect(stream.into_inner()).await.is_err()),
        }
        // The film through the first segment was published and stays readable.
        let labels: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Product(p)) if p.output == "video" => {
                    Some(p.label.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(labels, ["Video (segments 1-1)"], "{events:?}");
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let (meta, bytes) = read(
            &mut client,
            &all,
            v1::ReadRequest {
                target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                    run: "broken".into(),
                    output: "video".into(),
                    index: 0,
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            format!("0|a dawn|{seen}|\n"),
            "{meta:?}"
        );
        let _ = fs::remove_dir_all(tools);
    }
}
