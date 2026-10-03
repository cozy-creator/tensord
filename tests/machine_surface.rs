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
        let (child, client, claim, address) = launch(&root).await;
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
) -> (
    Child,
    pb::pod_host_client::PodHostClient<Channel>,
    pb::Claim,
    String,
) {
    let state = root.join("state");
    let _ = fs::remove_file(state.join("api-ready.json"));
    let child = Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
        .args(["serve", "--state"])
        .arg(&state)
        .arg("--machine-config")
        .arg(root.join("config/machine.json"))
        .args(["--listen", "127.0.0.1:0", "--host-bytes", "0"])
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    // Test harness bound on a broken start; the product has no such limit.
    let until = Instant::now() + Duration::from_secs(60);
    let ready = loop {
        if let Ok(bytes) = fs::read(state.join("api-ready.json")) {
            break serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        }
        assert!(
            Instant::now() < until,
            "machine did not publish api-ready.json"
        );
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
