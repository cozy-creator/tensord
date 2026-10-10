//! The real `cozy-machine serve` process behind its TLS/gRPC API, called as the CLI calls it.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use cozy_machine::journal::Journal;
use ed25519_dalek::SigningKey;
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

#[path = "common/hub_oauth.rs"]
mod hub_oauth;
#[path = "common/names_hub.rs"]
mod names_hub;

struct Machine {
    child: Child,
    root: PathBuf,
    address: String,
}

const SIGNER: [u8; 32] = [33; 32];
/// A second admitted key: another submitter on the same machine.
const OTHER: [u8; 32] = [44; 32];
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
        let keys = [SIGNER, OTHER].map(|key| {
            URL_SAFE_NO_PAD.encode(SigningKey::from_bytes(&key).verifying_key().to_bytes())
        });
        let write = |name: &str, value: serde_json::Value| {
            let path = config.join(name);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        };
        write(
            "keys.json",
            serde_json::json!({ "keys": keys }),
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
        let (child, address) = launch(&root, extra);
        Self {
            child,
            root,
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
}

impl Drop for Machine {
    fn drop(&mut self) {
        self.stop();
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn actor() -> String {
    actor_of(&SIGNER)
}
fn actor_of(key: &[u8; 32]) -> String {
    tensorfs_core::sha256::hex(&SigningKey::from_bytes(key).verifying_key().to_bytes())
}

fn launch(root: &Path, extra: &[std::ffi::OsString]) -> (Child, String) {
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
    (child, ready["address"].as_str().unwrap().to_owned())
}

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
        cap_by(&SIGNER, scope)
    }
    fn cap_by(key: &[u8; 32], scope: Grant) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        mint(
            &SigningKey::from_bytes(key),
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
    /// One Status frame as `token`'s holder sees it.
    async fn status_frame(
        client: &mut v1::machine_client::MachineClient<Channel>,
        token: &str,
    ) -> v1::StatusFrame {
        let request = authorized(v1::StatusRequest { keepalive: false }, token);
        let mut frames = client.status(request).await.unwrap().into_inner();
        frames.message().await.unwrap().unwrap()
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

    /// Read returns the store's transport log, its rotated part first, or a tail from a line
    /// start; only a machine capability reads it.
    #[tokio::test]
    async fn read_returns_the_stores_transport_log_with_rotation_and_tail() {
        let machine = Machine::start().await;
        let mut client = client(&machine).await;
        let owner = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let log = |name: &str, tail| v1::ReadRequest {
            target: Some(v1::read_request::Target::Log(name.into())),
            tail,
            ..Default::default()
        };
        let transport = |tail| log("tensorfs-transport", tail);
        assert!(read(&mut client, &owner, transport(0)).await.unwrap().1.is_empty());
        let logs = machine.store().join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(logs.join("transport.log.1"), "1.000 pull a 1 old\n").unwrap();
        let current: String = (0..3000).map(|n| format!("{n}.000 hedge object-{n} 4096 grant\n")).collect();
        fs::write(logs.join("transport.log"), &current).unwrap();
        let (_, whole) = read(&mut client, &owner, transport(0)).await.unwrap();
        assert_eq!(whole, format!("1.000 pull a 1 old\n{current}").into_bytes());
        let (_, tail) = read(&mut client, &owner, transport(100)).await.unwrap();
        let tail = String::from_utf8(tail).unwrap();
        assert!(tail.len() <= 100 && tail.ends_with("2999.000 hedge object-2999 4096 grant\n"));
        assert!(current.ends_with(&tail) && current.as_bytes()[current.len() - tail.len() - 1] == b'\n');
        let unknown = read(&mut client, &owner, log("other", 0)).await.unwrap_err();
        assert_eq!(unknown.code(), tonic::Code::NotFound);
        let run = cap(Grant { run: "1".into(), ..Default::default() });
        assert_eq!(read(&mut client, &run, transport(0)).await.unwrap_err().code(), tonic::Code::PermissionDenied);
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
        // Its own time running the callable, on the machine's clock.
        assert!(outcome.execution_ms > 0, "{outcome:?}");
        let running = events
            .iter()
            .find(|e| {
                matches!(&e.event, Some(v1::run_event::Event::State(s)) if s.state == "running")
            })
            .expect("the run is seen running");
        let finished = events.last().unwrap().at_ms;
        assert!(running.at_ms > 0 && running.at_ms <= finished, "{events:?}");
        assert!(outcome.execution_ms <= (finished - running.at_ms) as u64 + 1, "{outcome:?}");
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
                ..Default::default()
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
    async fn submitters_of_one_package_see_only_their_own_runs() {
        let machine = Machine::start_with(|state, actor| {
            hold_classifier(state, actor);
            // The other submitter holds the same package: a row of its own, the same generation.
            let mut journal = Journal::open(&state.join("execution")).unwrap();
            let held = journal.installations(actor).unwrap().remove(0);
            journal
                .bind_installation(cozy_machine::journal::Installation {
                    actor: actor_of(&OTHER),
                    ..held
                })
                .unwrap();
        })
        .await;
        let mut client = client(&machine).await;
        let whole = |key| {
            cap_by(
                key,
                Grant {
                    action: MACHINE.into(),
                    ..Default::default()
                },
            )
        };
        let (ours, theirs) = (whole(&SIGNER), whole(&OTHER));
        let request = |id: &str, spec| v1::RunRequest {
            id: id.into(),
            after: 0,
            spec,
        };
        // Both name their run "run-1": an id is its submitter's own.
        let mut settled = vec![];
        for (cap, iterations) in [(&ours, 3), (&theirs, 4)] {
            let run = request("run-1", Some(spec(iterations)));
            let events = collect(client.run(authorized(run, cap)).await.unwrap().into_inner())
                .await
                .unwrap();
            let Some(v1::run_event::Event::State(state)) = &events[0].event else {
                panic!("{events:?}")
            };
            let Some(v1::run_event::Event::Outcome(outcome)) = &events.last().unwrap().event else {
                panic!("{events:?}")
            };
            assert_eq!(outcome.status, "succeeded", "{outcome:?}");
            let result: serde_json::Value = serde_json::from_slice(&outcome.result).unwrap();
            assert_eq!(result["iterations"], iterations);
            settled.push(state.number);
        }
        let (our_run, their_run) = (settled[0], settled[1]);
        assert_ne!(our_run, their_run, "two runs, not one shared record");
        // Reading "run-1" answers each submitter with its own run's output.
        let target = v1::ReadRequest {
            target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                run: "run-1".into(),
                output: "report".into(),
                index: 0,
                ..Default::default()
            })),
            ..Default::default()
        };
        for (cap, iterations) in [(&ours, 3), (&theirs, 4)] {
            let (_, bytes) = read(&mut client, cap, target.clone()).await.unwrap();
            let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(report["iterations"], iterations);
        }
        // A run only one of them submitted does not exist for the other: attach, read, control.
        let only_ours = request("only-ours", Some(spec(2)));
        collect(
            client
                .run(authorized(only_ours, &ours))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let attach = client
            .run(authorized(request("only-ours", None), &theirs))
            .await;
        let attached = match attach {
            Ok(stream) => collect(stream.into_inner()).await.map(drop),
            Err(status) => Err(status),
        };
        assert_eq!(attached.unwrap_err().code(), tonic::Code::NotFound);
        let other = v1::ReadRequest {
            target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                run: "only-ours".into(),
                output: "report".into(),
                index: 0,
                ..Default::default()
            })),
            ..Default::default()
        };
        let refused = read(&mut client, &theirs, other).await.unwrap_err();
        assert_eq!(refused.code(), tonic::Code::NotFound);
        // Their cancel of the id fences it in their own namespace (canceled before acceptance)
        // and never reaches our run, which stays succeeded.
        let cancel = v1::ControlRequest {
            id: "only-ours".into(),
            action: v1::Action::Cancel as i32,
        };
        let fenced = client
            .control(authorized(cancel, &theirs))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(fenced.state, "canceled", "{fenced:?}");
        let ours_after = collect(
            client
                .run(authorized(request("only-ours", None), &ours))
                .await
                .unwrap()
                .into_inner(),
        )
        .await
        .unwrap();
        let Some(v1::run_event::Event::State(state)) = &ours_after[0].event else {
            panic!("{ours_after:?}")
        };
        assert_ne!(state.number, fenced.number, "their fence is a record of their own");
        assert_eq!(outcome(&ours_after).status, "succeeded", "{ours_after:?}");
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

    /// The thumbprint of the leaf the machine's API presents: what the CLI pins and names as a
    /// capability's `cnf.jkt`.
    fn leaf_jkt(machine: &Machine) -> String {
        let identity: serde_json::Value =
            serde_json::from_slice(&fs::read(machine.root.join("config/identity/identity.json")).unwrap()).unwrap();
        tensorfs_core::transport::DpopKey::from_pem(identity["private_key_pem"].as_str().unwrap())
            .unwrap()
            .thumbprint()
    }

    /// Actual installed CPU jobs retain paused work and relinquish only explicitly canceled
    /// unfinished or unadopted output custody, including across a daemon restart.
    #[tokio::test]
    async fn explicit_cancel_releases_unowned_native_outputs_and_preserves_adopted_custody() {
        use tensorfs_core::{derived, meta::Meta, store::Store};
        for phase in ["checkpoint", "commit", "adopt"] {
            let (mut machine, tools) = installing_machine().await;
            let mut peer = client(&machine).await;
            let all = cap(Grant {
                action: MACHINE.into(),
                ..Default::default()
            });
            let manifest = write_package(
                &mut peer,
                &all,
                "cpu_weights",
                "local/cozy-machine-cpu-weights",
            )
            .await;
            let mut stream = peer
                .run(authorized(
                    v1::RunRequest {
                        id: "native-cancel".into(),
                        after: 0,
                        spec: Some(v1::RunSpec {
                            kind: v1::RunKind::Job as i32,
                            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
                            entrypoint: "table".into(),
                            payload: serde_json::to_vec(
                                &serde_json::json!({"size":64,"hold":phase}),
                            )
                            .unwrap(),
                            owner: "alice".into(),
                            ..Default::default()
                        }),
                    },
                    &all,
                ))
                .await
                .unwrap()
                .into_inner();
            loop {
                let event = stream
                    .message()
                    .await
                    .unwrap()
                    .expect("job ended before native phase");
                if matches!(event.event,Some(v1::run_event::Event::Progress(ref p)) if p.stage==format!("native {phase}"))
                {
                    break;
                }
                if let Some(v1::run_event::Event::Outcome(done)) = event.event {
                    panic!("job ended before native {phase}: {done:?}");
                }
            }
            let journal_root = machine.root.join("state/execution");
            let mut journal = Journal::open(&journal_root).unwrap();
            let record = journal.get_public(&actor(), "native-cancel").unwrap();
            let bindings = journal.derived_outputs(&record.id).unwrap();
            assert_eq!(bindings.len(), 1);
            let transaction = bindings[0].transaction.clone();
            let control = |action: v1::Action| {
                authorized(
                    v1::ControlRequest {
                        id: "native-cancel".into(),
                        action: action as i32,
                    },
                    &all,
                )
            };
            if phase == "checkpoint" {
                // Pause and observer disconnect preserve progress. The explicit cancel is
                // durable while the daemon is down; startup must finish its native cleanup.
                peer.control(control(v1::Action::Pause)).await.unwrap();
                until(&mut peer, &all, "native-cancel", "paused").await;
                drop(stream);
                machine.stop();
                let store = Store::open(&machine.store()).unwrap();
                let meta = Meta::open(&store).unwrap();
                assert!(matches!(
                    derived::lookup(&store, &meta, &transaction).unwrap(),
                    derived::Lookup::Open { .. }
                ));
                assert!(journal.canceled_derived_outputs().unwrap().is_empty());
                journal.cancel(&record.id, &actor()).unwrap();
                drop(meta);
                drop(store);
                drop(journal);
                let (child, address) = launch(&machine.root, &installing_args(&tools));
                machine.child = child;
                machine.address = address;
                peer = client(&machine).await;
                until(&mut peer, &all, "native-cancel", "canceled").await;
            } else {
                peer.control(control(v1::Action::Cancel)).await.unwrap();
                let ended = collect(stream).await.unwrap();
                assert_eq!(outcome(&ended).status, "canceled", "{ended:?}");
                drop(journal);
            }
            machine.stop();
            let journal = Journal::open(&journal_root).unwrap();
            assert!(journal.derived_outputs(&record.id).unwrap().is_empty());
            let store = Store::open(&machine.store()).unwrap();
            let meta = Meta::open(&store).unwrap();
            let observed = derived::lookup(&store, &meta, &transaction).unwrap();
            match (&observed, phase) {
                (derived::Lookup::Abandoned, "checkpoint") => (),
                (derived::Lookup::Committed(result), kind) => {
                    let facts = result.disposition_value();
                    let mut fields =
                        tensorfs_core::canon::Fields::new("disposition", &facts).unwrap();
                    assert_eq!(
                        fields.req_str("kind").unwrap(),
                        if kind == "adopt" {
                            "adopted"
                        } else {
                            "released"
                        }
                    );
                }
                _ => panic!("{phase}: {observed:?}"),
            }
            tensorfs_core::gc::collect(store.root(), false).unwrap();
            if let derived::Lookup::Committed(result) = observed {
                assert_eq!(
                    store.read_manifest(&result.receipt().manifest).is_ok(),
                    phase == "adopt"
                );
                assert_eq!(
                    derived::inspect_source(
                        &store,
                        &meta,
                        result.receipt().manifest.clone(),
                        vec!["model".into()],
                        vec![]
                    )
                    .is_ok(),
                    phase == "adopt"
                );
            }
            drop(meta);
            drop(store);
            drop(journal);
            drop(machine);
            fs::remove_dir_all(tools).unwrap();
        }
    }
    /// `cozy model quantize`'s shape on CPU: a job writes its declared weights output through
    /// the machine's native writer channel and adopts it; the output is a product of its log
    /// (the manifest). A warm run keeps that held checkpoint under a local alias.
    #[tokio::test]
    async fn a_job_writes_a_weights_output_kept_under_a_local_alias() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&mut client, &all, "cpu_weights", "local/cozy-machine-cpu-weights").await;
        let job = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "table".into(),
            payload: br#"{"size":64}"#.to_vec(),
            owner: "alice".into(),
            ..Default::default()
        };
        let run = |id: &str, spec: v1::RunSpec| v1::RunRequest {
            id: id.into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(client.run(authorized(run("weights", job), &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let written = done
            .outputs
            .iter()
            .find(|p| p.output == "model")
            .unwrap_or_else(|| panic!("{done:?}"));
        assert_eq!(written.media_type, "application/vnd.cozy.model-manifest");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert!(result.to_string().contains(&written.digest), "{result} names {written:?}");
        let held = tensorfs_core::store::Store::open(&machine.store()).unwrap();
        let reference = tensorfs_core::ids::ObjectRef {
            sha256: written.digest.trim_start_matches("sha256:").into(),
            length: written.length,
        };
        held.read_manifest(&reference).unwrap();

        // The held checkpoint kept under a local alias, as `cozy model download … local/tiny`
        // keeps a model: no Hub, no publication.
        let keep = v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                manifest: written.digest.clone(),
                manifest_length: written.length,
                ..Default::default()
            }],
            weights_destination: "local/tiny".into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(client.run(authorized(run("keep", keep), &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["models"][0]["published"]["destination"], "local/tiny", "{result}");
        assert_eq!(result["models"][0]["published"]["checkpoint"], written.digest, "{result}");

        // A file that was never written is refused at submit: the run takes custody of none.
        let unwritten = v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                source: format!("object://sha256:{}/stray.safetensors", "0".repeat(64)),
                ..Default::default()
            }],
            weights_destination: "local/stray".into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let refused = match client.run(authorized(run("unwritten", unwritten), &all)).await {
            Ok(stream) => collect(stream.into_inner()).await.unwrap_err(),
            Err(status) => status,
        };
        assert!(refused.message().contains("was not written to this machine"), "{refused:?}");

        let _ = fs::remove_dir_all(tools);
    }

    /// `cozy model upload local/tiny acme/tiny` under the run's capability (th-241): the spec
    /// names the Hub and carries the owner's device-key-signed capability for this machine's
    /// leaf; the machine trades it once at the upload and AuthKit verifies every proof. With no
    /// capability, or one signed by a revoked device key, the run ends typed.
    #[tokio::test]
    #[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
    async fn a_warm_run_publishes_a_local_alias_under_its_capability() {
        use tensorfs_core::repository::{Mutation, RepositoryName};
        let (upstream, hub) = hub_oauth::test_hub();
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let mut held = None;
        let machine = Machine::start_with(|state, _| {
            let store = tensorfs_core::store::Store::ensure(&state.join("tensorfs")).unwrap();
            let manifest = hub_oauth::checkpoint(&store, &[7; 8192]);
            let replace = Mutation::ReplaceLocal {
                repo: RepositoryName::new("local", "tiny").unwrap(),
                version: manifest.sha256.clone(),
                manifest: manifest.clone(),
            };
            store.apply_repository(None, &replace, &Default::default()).unwrap();
            held = Some(manifest.id());
        })
        .await;
        let manifest = held.unwrap();
        let jkt = leaf_jkt(&machine);
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let ops = serde_json::json!([{"type": "tensorhub_model_publish", "model": "acme/tiny"}]);
        let upload = |id: &str, capability: String| v1::RunRequest {
            id: id.into(),
            after: 0,
            spec: Some(v1::RunSpec {
                kind: v1::RunKind::Warm as i32,
                models: vec![v1::ModelChoice {
                    parameter: "model".into(),
                    repository: "local/tiny".into(),
                    ..Default::default()
                }],
                weights_destination: "acme/tiny".into(),
                hub: Some(v1::HubAccess {
                    origin: authkit.hub.clone(),
                    object_hosts: vec!["localhost".into()],
                    capability,
                    token_endpoint: authkit.token_endpoint.clone(),
                    ..Default::default()
                }),
                owner: "alice".into(),
                ..Default::default()
            }),
        };
        let done = |events: Vec<v1::RunEvent>| outcome(&events);
        let refused = done(collect(client.run(authorized(upload("upload-0", String::new()), &all)).await.unwrap().into_inner()).await.unwrap());
        assert_eq!(refused.status, "failed", "{refused:?}");
        assert_eq!(refused.reason.unwrap().code, "capability_required");
        assert_eq!(authkit.exchanges(), 0);

        for id in ["upload-1", "upload-2"] {
            let signed = authkit.capability(&jkt, &ops, 600);
            let done = done(collect(client.run(authorized(upload(id, signed), &all)).await.unwrap().into_inner()).await.unwrap());
            assert_eq!(done.status, "succeeded", "{id}: {done:?}");
            let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
            assert_eq!(result["models"][0]["published"]["checkpoint"], manifest, "{result}");
        }
        {
            let hub = hub.lock().unwrap();
            assert_eq!(hub.finalized, [manifest.clone(), manifest.clone()]);
            assert_eq!(hub.credentialed_uploads, 0);
        }
        assert_eq!(authkit.exchanges(), 2, "one trade per run");
        let owner = authkit.owner();
        let verified = authkit.verified();
        assert!(!verified.is_empty() && verified.iter().all(|s| s["owner"] == owner.as_str()), "{verified:?}");

        let signed = authkit.capability(&jkt, &ops, 600);
        authkit.revoke_device_key();
        let revoked = done(collect(client.run(authorized(upload("upload-3", signed), &all)).await.unwrap().into_inner()).await.unwrap());
        assert_eq!(revoked.status, "failed", "{revoked:?}");
        assert_eq!(revoked.reason.unwrap().code, "capability_refused");
        assert_eq!(hub.lock().unwrap().finalized.len(), 2);
    }

    /// A wheel with one module and nothing else: `name` `version`.
    fn wheel(name: &str, version: &str) -> Vec<u8> {
        use std::io::Write;
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        let info = format!("{name}-{version}.dist-info");
        let files = [
            (format!("{name}/__init__.py"), "app = None\n".to_string()),
            (format!("{info}/METADATA"), format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")),
            (format!("{info}/WHEEL"), "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n".into()),
        ];
        let mut record = String::new();
        for (path, body) in &files {
            zip.start_file(path.as_str(), options).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
            record.push_str(&format!("{path},,\n"));
        }
        zip.start_file(format!("{info}/RECORD"), options).unwrap();
        zip.write_all(record.as_bytes()).unwrap();
        zip.finish().unwrap().into_inner()
    }

    /// One warm run to its outcome.
    async fn settle(
        client: &mut v1::machine_client::MachineClient<Channel>,
        cap: &str,
        id: &str,
        spec: v1::RunSpec,
    ) -> v1::Outcome {
        let request = v1::RunRequest { id: id.into(), after: 0, spec: Some(spec) };
        outcome(&collect(client.run(authorized(request, cap)).await.unwrap().into_inner()).await.unwrap())
    }

    /// th-245: a run names its release and model and the machine resolves both at its Hub by
    /// name, anonymously: the newest release's card and lock, the model's lane by one closure
    /// call, then the checkpoint pulled pinned to that lane. Under the same catalog revision a
    /// second run asks the Hub nothing; a moved revision resolves once more and moves no bytes;
    /// a hash alone is the owner's, refused without the capability. Real uv, Python and TensorFS.
    #[tokio::test]
    async fn runs_name_their_release_and_model_and_a_warm_run_asks_the_hub_nothing() {
        let source = std::env::temp_dir().join(format!("cm-names-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store::Store::ensure(&source).unwrap();
        let manifest = hub_oauth::checkpoint(&store, &[5; 8192]);
        let model = names_hub::Model {
            repository: "acme/model".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            manifest: manifest.clone(),
        };
        let package = names_hub::Package {
            name: "acme/pkg".into(),
            release: "1.0.0".into(),
            interface: serde_json::json!({"application": "pkg:app", "entrypoints": []}),
            wheel: "pkg-1.0.0-py3-none-any.whl".into(),
            bytes: wheel("pkg", "1.0.0"),
            callees: vec![],
            pypi: String::new(),
        };
        let (origin, heard) = names_hub::serve(&store, model, Some(package));
        let machine = Machine::start().await;
        let mut client = client(&machine).await;
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let spec = |release: Option<&str>, choice: v1::ModelChoice, revision: &str| v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            source: release.map(|release| {
                v1::run_spec::Source::Release(v1::Release { package: "acme/pkg".into(), release: release.into() })
            }),
            models: vec![v1::ModelChoice { parameter: "model".into(), ..choice }],
            hub: Some(v1::HubAccess {
                origin: origin.clone(),
                object_hosts: vec!["localhost".into()],
                ..Default::default()
            }),
            catalog_revision: revision.into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let named = v1::ModelChoice {
            repository: "acme/model".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            ..Default::default()
        };
        let take = || std::mem::take(&mut *heard.lock().unwrap());

        // Cold: the newest release installs by name; the model resolves and downloads by name.
        let done = settle(&mut client, &all, "names-1", spec(Some(""), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        let row = &result["models"][0];
        assert_eq!(result["release"], "1.0.0");
        assert_eq!((&row["release"], &row["lane"], &row["manifest"]), (&"1.0.0".into(), &"bf16".into(), &manifest.id().into()));
        let cold = take();
        let pinned = format!("POST /v1/tensorfs/closure acme/model@1.0.0@{} bf16", manifest.id());
        for asked in [
            "GET /v1/packages/acme/pkg",
            "GET /v1/packages/acme/pkg/releases/1.0.0",
            "GET /v1/packages/acme/pkg/releases/1.0.0/locked-requirements",
            "POST /v1/tensorfs/closure acme/model@1.0.0 bf16",
            &pinned,
            "POST /v1/tensorfs/presign",
        ] {
            assert!(cold.iter().any(|heard| heard == asked), "{asked} not in {cold:?}");
        }
        assert!(cold.iter().all(|heard| !heard.ends_with("+credential")), "{cold:?}");

        // Warm: the release's installation and the name's resolution are held, across a
        // restart too.
        let done = settle(&mut client, &all, "names-2", spec(Some("1.0.0"), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), Vec::<String>::new());
        let mut machine = machine;
        machine.stop();
        (machine.child, machine.address) = launch(&machine.root, &[]);
        let mut client = self::client(&machine).await;
        let done = settle(&mut client, &all, "names-2b", spec(Some("1.0.0"), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), Vec::<String>::new());

        // A name alone is the newest release's only lane: one closure, nothing to move.
        let bare = v1::ModelChoice { repository: "acme/model".into(), ..Default::default() };
        let done = settle(&mut client, &all, "names-3", spec(None, bare, "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!((&result["models"][0]["release"], &result["models"][0]["lane"]), (&"1.0.0".into(), &"bf16".into()));
        assert_eq!(take(), ["POST /v1/tensorfs/closure acme/model "]);

        // A revision the caller moved resolves the name once more; the bytes are held.
        let done = settle(&mut client, &all, "names-4", spec(Some("1.0.0"), named, "r2")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), ["POST /v1/tensorfs/closure acme/model@1.0.0 bf16"]);

        // A hash alone is the owner's: absent without the run's capability.
        let other = format!("sha256:{}", "ab".repeat(32));
        let hashed = v1::ModelChoice { repository: "acme/model".into(), manifest: other.clone(), manifest_length: 9, ..Default::default() };
        let done = settle(&mut client, &all, "names-5", spec(None, hashed, "r2")).await;
        assert_eq!(done.status, "failed", "{done:?}");
        assert_eq!(take(), [format!("POST /v1/tensorfs/closure acme/model@{other} ")]);
        let _ = fs::remove_dir_all(source);
    }

    /// th-245 with th-241: the owner's checkpoint by hash reads only under a run capability that
    /// names it. The machine trades it once through a real AuthKit and presents the token on the
    /// closure and presign; a capability naming another checkpoint reads anonymously and finds it
    /// absent, with no trade; one signed for another machine key is refused at submission.
    #[tokio::test]
    #[ignore = "real AuthKit: go and AUTHKIT_TEST_DATABASE_URL"]
    async fn the_owners_checkpoint_by_hash_reads_under_the_runs_capability() {
        let source = std::env::temp_dir().join(format!("cm-private-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store::Store::ensure(&source).unwrap();
        let manifest = hub_oauth::checkpoint(&store, &[9; 8192]);
        let model = names_hub::Model {
            repository: "acme/private".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            manifest: manifest.clone(),
        };
        let (upstream, heard) = names_hub::serve(&store, model, None);
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let machine = Machine::start().await;
        let mut client = client(&machine).await;
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let spec = |capability: String| v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                repository: "acme/private".into(),
                manifest: manifest.id(),
                manifest_length: manifest.length,
                ..Default::default()
            }],
            hub: Some(v1::HubAccess {
                origin: authkit.hub.clone(),
                object_hosts: vec!["localhost".into()],
                capability,
                token_endpoint: authkit.token_endpoint.clone(),
                ..Default::default()
            }),
            owner: "alice".into(),
            ..Default::default()
        };
        let read = |manifest: &str| serde_json::json!([{"type": "tensorhub_model_read", "model": "acme/private", "manifest": manifest}]);
        let jkt = leaf_jkt(&machine);

        let other = authkit.capability(&jkt, &read(&format!("sha256:{}", "ab".repeat(32))), 600);
        let done = settle(&mut client, &all, "private-1", spec(other)).await;
        assert_eq!(done.status, "failed", "{done:?}");
        assert_eq!(authkit.exchanges(), 0);

        let named = authkit.capability(&jkt, &read(&manifest.id()), 600);
        let done = settle(&mut client, &all, "private-2", spec(named)).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(authkit.exchanges(), 1);
        let verified = authkit.verified();
        assert!(verified.iter().any(|seen| seen["path"] == "/v1/tensorfs/closure"), "{verified:?}");
        let closures: Vec<_> = heard.lock().unwrap().iter().filter(|h| h.contains("closure")).cloned().collect();
        assert_eq!(closures.len(), 2, "{closures:?}");

        let forged = authkit.capability(&tensorfs_core::transport::DpopKey::generate().unwrap().thumbprint(), &read(&manifest.id()), 600);
        let request = v1::RunRequest { id: "private-3".into(), after: 0, spec: Some(spec(forged)) };
        let refused = client.run(authorized(request, &all)).await;
        let refused = match refused {
            Err(status) => status,
            Ok(stream) => collect(stream.into_inner()).await.unwrap_err(),
        };
        assert!(refused.message().contains("hub_access_invalid"), "{refused:?}");
        let _ = fs::remove_dir_all(source);
    }

    /// `cozy model upload` on the serve process: a warm run of one provider source makes it
    /// with TensorFS and puts its checkpoint in the destination under the machine's authority;
    /// every object reaches the Hub, then finalization.
    #[tokio::test]
    #[ignore = "real network: huggingface.co; real AuthKit: go and AUTHKIT_TEST_DATABASE_URL"]
    async fn a_warm_run_uploads_its_source_model_to_its_destination() {
        let (upstream, hub) = hub_oauth::test_hub();
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let machine = Machine::start().await;
        let ops = serde_json::json!([{"type": "tensorhub_model_publish", "model": "acme/tiny"}]);
        let capability = authkit.capability(&leaf_jkt(&machine), &ops, 600);
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                source: "hf://hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors".into(),
                ..Default::default()
            }],
            weights_destination: "acme/tiny".into(),
            hub: Some(v1::HubAccess {
                origin: authkit.hub.clone(),
                object_hosts: vec!["localhost".into()],
                capability,
                token_endpoint: authkit.token_endpoint.clone(),
                ..Default::default()
            }),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = v1::RunRequest {
            id: "upload-1".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(client.run(authorized(request, &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        let model = &result["models"][0];
        assert_eq!(model["published"]["destination"], "acme/tiny");
        assert_eq!(model["published"]["checkpoint"], model["manifest"]);
        let held = hub.lock().unwrap();
        assert_eq!(held.credentialed_uploads, 0);
        assert_eq!(held.finalized, [model["manifest"].as_str().unwrap()]);
        assert!(held.declared.contains(&held.finalized[0]));
        let mut uploaded: Vec<_> = held.uploaded.keys().cloned().collect();
        let mut declared = held.declared.clone();
        uploaded.sort();
        declared.sort();
        assert_eq!(uploaded, declared);
    }

    /// `cozy model upload <file>` and `cozy model download … local/name` on the serve process:
    /// a file the client wrote is made into a model with no provider and kept under a local
    /// alias. Real network for the fixture and its profile's reference configs.
    #[tokio::test]
    #[ignore = "real network: huggingface.co"]
    async fn a_written_file_is_made_and_kept_under_a_local_alias() {
        let machine = Machine::start().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let file = Command::new("curl")
            .args(["-sSfL", "https://huggingface.co/hf-internal-testing/tiny-sdxl-pipe/resolve/20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors"])
            .output()
            .unwrap();
        assert!(file.status.success());
        let sha256 = tensorfs_core::sha256::hex_digest(&file.stdout);
        let length = file.stdout.len() as u64;
        let held = write(&mut client, &all, &format!("sha256:{sha256}"), length, 0, &file.stdout)
            .await
            .unwrap();
        assert_eq!(held, length);
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                source: format!("object://sha256:{sha256}/unet.safetensors"),
                ..Default::default()
            }],
            weights_destination: "local/unet".into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = v1::RunRequest {
            id: "keep-file".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(client.run(authorized(request, &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        let model = &result["models"][0];
        assert_eq!(model["published"]["destination"], "local/unet", "{result}");
        let store = tensorfs_core::store::Store::open(&machine.store()).unwrap();
        let (manifest, _) = tensorfs_core::source_model::held(&store, "unet").unwrap().unwrap();
        assert_eq!(model["manifest"], manifest.id());
    }

    /// A job's input tree (`--asset field=dir`, a `Tree` field): each file and the tree's manifest
    /// written with Write, the manifest named as an input of the tree media type under the
    /// payload's ref; the machine materializes the directory and the callable reads it.
    #[tokio::test]
    async fn a_written_input_tree_reaches_the_callable_as_a_directory() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&mut client, &all, "cpu_tree", "local/cozy-machine-cpu-tree").await;
        let files: [(&str, &[u8]); 2] = [("a.txt", b"alpha"), ("nested/b.bin", b"\x00\x01beta")];
        let mut entries = vec![];
        for (path, body) in files {
            let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(body));
            let length = body.len() as u64;
            assert_eq!(write(&mut client, &all, &digest, length, 0, body).await.unwrap(), length);
            entries.push(serde_json::json!({"kind": "file", "path": path,
                "blob": {"sha256": digest.trim_start_matches("sha256:"), "length": length}}));
        }
        let tree = serde_json::to_vec(&serde_json::json!({"entries": entries})).unwrap();
        let tree_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&tree));
        let size = tree.len() as u64;
        assert_eq!(write(&mut client, &all, &tree_digest, size, 0, &tree).await.unwrap(), size);
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "count".into(),
            // As `cozy run --asset data=<dir>` sends it: the payload names the tree's manifest.
            payload: serde_json::to_vec(&serde_json::json!({ "data": tree_digest })).unwrap(),
            inputs: vec![v1::InputFile {
                field: "data".into(),
                digest: tree_digest.clone(),
                length: size,
                media_type: "application/vnd.cozy.tree-manifest".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let request = v1::RunRequest {
            id: "tree".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(client.run(authorized(request, &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["files"], serde_json::json!(["a.txt", "nested/b.bin"]), "{result}");
        let mut whole = b"alpha".to_vec();
        whole.extend_from_slice(b"\x00\x01beta");
        assert_eq!(result["sha256"], tensorfs_core::sha256::hex_digest(&whole));
        let _ = fs::remove_dir_all(tools);
    }

    /// An executor holds no descriptor the machine did not hand it: not one the machine itself
    /// inherited (as a service inherits its parent's readiness, key and credential pipes).
    #[tokio::test]
    async fn an_executor_inherits_no_descriptor_the_machine_did_not_hand_it() {
        use std::os::fd::AsRawFd;
        // Not close-on-exec: the machine started below inherits it, as from a parent.
        let (planted, _write) = nix::unistd::pipe().unwrap();
        let target = fs::read_link(format!("/proc/self/fd/{}", planted.as_raw_fd()))
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (machine, tools) = installing_machine().await;
        let held = fs::read_dir(format!("/proc/{}/fd", machine.child.id()))
            .unwrap()
            .filter_map(|e| fs::read_link(e.unwrap().path()).ok())
            .any(|t| t.to_string_lossy() == target);
        assert!(held, "the machine did not inherit the planted pipe {target}");
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&mut client, &all, "cpu_tree", "local/cozy-machine-cpu-tree").await;
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "descriptors".into(),
            payload: b"{}".to_vec(),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = v1::RunRequest {
            id: "descriptors".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(client.run(authorized(request, &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        let targets = result["targets"].as_array().unwrap();
        assert!(!targets.is_empty(), "{result}");
        assert!(
            !targets.iter().any(|t| t == &serde_json::json!(target)),
            "the executor inherited the machine's pipe {target}: {result}"
        );
        let _ = fs::remove_dir_all(tools);
    }

    /// A tree a child made (`out.save_tree`) reaches its parent as a directory, goes to a
    /// sibling as that one's input tree, and is the job's own output: its manifest and each
    /// member file are read back over Read.
    #[tokio::test]
    async fn a_childs_tree_is_handed_on_and_returned_as_the_jobs_output() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&mut client, &all, "cpu_tree", "local/cozy-machine-cpu-tree").await;
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "survey".into(),
            payload: serde_json::to_vec(&serde_json::json!({ "text": "verified" })).unwrap(),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = v1::RunRequest { id: "survey".into(), after: 0, spec: Some(spec) };
        let events = collect(client.run(authorized(request, &all)).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["files"], serde_json::json!(["b.txt", "empty.txt", "nested/a.txt"]), "{result}");
        let product = done.outputs.iter().find(|p| p.output == "tree").expect("the tree is an output");
        assert_eq!(product.media_type, "application/vnd.cozy.tree-manifest");
        assert_eq!(result["tree"]["digest"], product.digest, "{result}");
        let target = |member: &str| v1::ReadRequest {
            target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                run: "survey".into(),
                output: "tree".into(),
                index: 0,
                member: member.into(),
            })),
            ..Default::default()
        };
        let (meta, tree) = read(&mut client, &all, target("")).await.unwrap();
        assert_eq!(meta.digest, product.digest);
        let tree: serde_json::Value = serde_json::from_slice(&tree).unwrap();
        let paths: Vec<_> = tree["entries"].as_array().unwrap().iter().map(|e| e["path"].clone()).collect();
        assert_eq!(paths, ["b.txt", "empty.txt", "nested/a.txt"], "{tree}");
        let (meta, member) = read(&mut client, &all, target("nested/a.txt")).await.unwrap();
        assert_eq!(member, b"verified");
        assert_eq!(meta.digest, format!("sha256:{}", tensorfs_core::sha256::hex_digest(b"verified")));
        assert_eq!(read(&mut client, &all, target("empty.txt")).await.unwrap().1, b"");
        assert_eq!(read(&mut client, &all, target("absent")).await.unwrap_err().code(), tonic::Code::NotFound);
        let _ = fs::remove_dir_all(tools);
    }

    fn outcome(events: &[v1::RunEvent]) -> v1::Outcome {
        match &events.last().unwrap().event {
            Some(v1::run_event::Event::Outcome(outcome)) => outcome.clone(),
            _ => panic!("{events:?}"),
        }
    }

    /// A job's memoized call that ran is held by this machine for its signer: a later job has
    /// the same call answered from it, never run, and a new one runs. The client carries none.
    #[tokio::test]
    async fn a_memoized_call_is_answered_from_this_machines_memo() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest =
            write_package(&mut client, &all, "cpu_memo", "local/cozy-machine-cpu-memo").await;
        let counter = machine.root.join("measured");
        let survey = |id: &str, values: &[i64]| {
            let spec = v1::RunSpec {
                kind: v1::RunKind::Job as i32,
                source: Some(v1::run_spec::Source::Local(v1::LocalSource {
                    manifest: manifest.clone(),
                })),
                entrypoint: "survey".into(),
                payload: serde_json::to_vec(
                    &serde_json::json!({"values": values, "counter": counter}),
                )
                .unwrap(),
                owner: "alice".into(),
                ..Default::default()
            };
            authorized(v1::RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all)
        };
        let measured = || std::fs::read_to_string(&counter).unwrap().parse::<u32>().unwrap();
        let squares = |events: &[v1::RunEvent]| {
            let done = outcome(events);
            assert_eq!(done.status, "succeeded", "{done:?}");
            serde_json::from_slice::<serde_json::Value>(&done.result).unwrap()["squares"].clone()
        };
        let first = collect(client.run(survey("survey-1", &[3, 4])).await.unwrap().into_inner())
            .await
            .unwrap();
        assert_eq!(squares(&first), serde_json::json!([9, 16]));
        assert_eq!(measured(), 2);
        // The call for 3 is answered from the machine's memo; the one for 5 runs.
        let second = collect(client.run(survey("survey-2", &[3, 5])).await.unwrap().into_inner())
            .await
            .unwrap();
        assert_eq!(squares(&second), serde_json::json!([9, 25]));
        assert_eq!(measured(), 3, "the held call ran again");
        // Each call says how it was answered: the second job's call for 3 is memoized, with the
        // first job's computation; its call for 5 ran.
        let calls = |events: &[v1::RunEvent]| -> Vec<(bool, String)> {
            events
                .iter()
                .filter_map(|e| match &e.event {
                    Some(v1::run_event::Event::Call(call)) => Some((call.memoized, call.computation_digest.clone())),
                    _ => None,
                })
                .collect()
        };
        let (first, second) = (calls(&first), calls(&second));
        assert!(first.iter().all(|(memoized, digest)| !memoized && digest.starts_with("sha256:")), "{first:?}");
        assert_eq!(second[0], (true, first[0].1.clone()), "{second:?}");
        assert!(!second[1].0 && second[1].1.starts_with("sha256:") && second[1].1 != first[1].1, "{second:?}");
        let _ = fs::remove_dir_all(tools);
    }

    /// A job calls another package's invocable that its environment holds as a dependency, as a
    /// published package's callee is: the call is a child run of that package, counted where it
    /// truly runs.
    #[tokio::test]
    async fn a_job_calls_a_package_its_environment_depends_on() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let built = tools.join("callee-wheel");
        let status = std::process::Command::new("uv")
            .args(["build", "--wheel", "--out-dir"])
            .arg(&built)
            .arg(fixtures.join("cpu_memo"))
            .status()
            .unwrap();
        assert!(status.success());
        let wheel = fs::read_dir(&built).unwrap().next().unwrap().unwrap().path();
        let bytes = fs::read(&wheel).unwrap();
        let wheel_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes));
        write(&mut client, &all, &wheel_digest, bytes.len() as u64, 0, &bytes).await.unwrap();
        let mut archive = tar::Builder::new(Vec::new());
        for name in ["pyproject.toml", "package.toml", "cpu_caller/__init__.py"] {
            archive
                .append_path_with_name(fixtures.join("cpu_caller").join(name), name)
                .unwrap();
        }
        let source = archive.into_inner().unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&source));
        write(&mut client, &all, &digest, source.len() as u64, 0, &source).await.unwrap();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package": "local/cozy-machine-cpu-caller", "release": "0.1.0", "python_version": "3.12",
            "source": {"digest": digest, "length": source.len()},
            "wheels": [{"name": wheel.file_name().unwrap().to_str().unwrap(), "digest": wheel_digest, "length": bytes.len()}],
            "callees": {"cozy-machine-cpu-memo": "local/cozy-machine-cpu-memo"}}))
        .unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        write(&mut client, &all, &manifest_digest, manifest.len() as u64, 0, &manifest).await.unwrap();
        let counter = machine.root.join("measured");
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource {
                manifest: manifest_digest,
            })),
            entrypoint: "relay".into(),
            payload: serde_json::to_vec(&serde_json::json!({"values": [3, 4], "counter": counter}))
                .unwrap(),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = authorized(
            v1::RunRequest {
                id: "relay-1".into(),
                after: 0,
                spec: Some(spec.clone()),
            },
            &all,
        );
        let events = collect(client.run(request).await.unwrap().into_inner())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["squares"], serde_json::json!([9, 16]));
        assert_eq!(result["greeting"], "squared squared", "{result}");
        assert_eq!(fs::read_to_string(&counter).unwrap(), "2");
        // Each call is a child run of the callee's own function, and memoized as its own.
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Call(call)) => Some(call.function.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["measure", "measure", "greet"], "{events:?}");
        // H3's references (run 4830): the job shows each image another package's entrypoint
        // returns as its own output the moment it exists, and returns them all at the end.
        let mut sitting = spec.clone();
        sitting.entrypoint = "portrait".into();
        sitting.payload = serde_json::to_vec(&serde_json::json!({"names": ["Lighthouse", "Keeper"]})).unwrap();
        let events = collect(client.run(authorized(v1::RunRequest {
            id: "portrait-1".into(), after: 0, spec: Some(sitting),
        }, &all)).await.unwrap().into_inner()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Call(call)) => Some(call.function.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["paint", "paint"], "{events:?}");
        let shown: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Product(p)) if p.output == "references" => Some(p.label.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(shown, ["Reference: Lighthouse", "Reference: Keeper"], "{events:?}");
        let returned = String::from_utf8_lossy(&done.result).into_owned();
        let mut seen = Vec::new();
        for index in 1..=2u32 {
            let (_, image) = read(
                &mut client,
                &all,
                v1::ReadRequest {
                    target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                        run: "portrait-1".into(),
                        output: "references".into(),
                        index,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert!(image.starts_with(b"\x89PNG"), "reference {index} is no PNG");
            let digest = tensorfs_core::sha256::hex_digest(&image);
            assert!(returned.contains(&digest), "reference {index} ({digest}) is not the one returned: {returned}");
            seen.push(digest);
        }
        assert_ne!(seen[0], seen[1], "each reference is its own image");
        // The Runtime's own operations are callees of every environment: a job quantizes
        // another package's model with the Runtime's `quantize`, as a child run of its own.
        let mut requantize = spec.clone();
        requantize.entrypoint = "requantize".into();
        requantize.payload = b"{}".to_vec();
        let events = collect(client.run(authorized(v1::RunRequest {
            id: "requantize-1".into(), after: 0, spec: Some(requantize),
        }, &all)).await.unwrap().into_inner()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Call(call)) => Some(call.function.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["produce", "quantize"], "{events:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        let (source, quantized) = (result["source"].as_str().unwrap(), result["result"].as_str().unwrap());
        assert!(source.starts_with("sha256:") && quantized.starts_with("sha256:"), "{result}");
        assert_ne!(source, quantized, "quantize wrote a model of its own");
        // A memoized call whose result is a file is answered from this machine's own memo while
        // it holds the file, for any later job of its signer; once the file is gone, it runs.
        let noted = machine.root.join("noted");
        let annotate = |id: &str| {
            let mut spec = spec.clone();
            spec.entrypoint = "annotate".into();
            spec.payload = serde_json::to_vec(&serde_json::json!({"values": [5, 6], "counter": noted})).unwrap();
            authorized(v1::RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all)
        };
        let mut notes = vec![];
        for id in ["annotate-1", "annotate-2"] {
            let events = collect(client.run(annotate(id)).await.unwrap().into_inner()).await.unwrap();
            assert_eq!(outcome(&events).status, "succeeded", "{events:?}");
            let mut files = vec![];
            for index in 1..=2u32 {
                let target = v1::OutputTarget { run: id.into(), output: "files".into(), index, ..Default::default() };
                let request = v1::ReadRequest { target: Some(v1::read_request::Target::Output(target)), ..Default::default() };
                files.push(read(&mut client, &all, request).await.unwrap().1);
            }
            notes.push(files);
        }
        assert_eq!(notes[0], [b"note 5".to_vec(), b"note 6".to_vec()]);
        assert_eq!(notes[1], notes[0], "the reused result is the same files");
        assert_eq!(fs::read_to_string(&noted).unwrap(), "2", "the second job ran its calls again");
        // The store loses one file (as its GC does once reclaim released it): that call runs.
        let hex = tensorfs_core::sha256::hex_digest(b"note 5");
        let blob = walkdir(&machine.root.join("state/tensorfs"))
            .into_iter()
            .find(|p| {
                p.components().any(|c| c.as_os_str() == "blobs")
                    && p.file_name().is_some_and(|n| n.to_string_lossy().contains(&hex))
            })
            .expect("the note's bytes are in the store");
        fs::remove_file(blob).unwrap();
        let events = collect(client.run(annotate("annotate-3")).await.unwrap().into_inner()).await.unwrap();
        assert_eq!(outcome(&events).status, "succeeded", "{events:?}");
        assert_eq!(fs::read_to_string(&noted).unwrap(), "3", "only the call whose file is gone ran");
        let mut nested = spec.clone();
        nested.entrypoint = "relay_nested".into();
        let events = collect(client.run(authorized(v1::RunRequest {
            id: "relay-nested".into(), after: 0, spec: Some(nested),
        }, &all)).await.unwrap().into_inner()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["squares"], serde_json::json!([9, 16]));
        assert_eq!(fs::read_to_string(&counter).unwrap(), "4");
        let mut restricted = spec;
        restricted.entrypoint = "relay_restricted".into();
        let events = collect(client.run(authorized(v1::RunRequest {
            id: "relay-restricted".into(), after: 0, spec: Some(restricted),
        }, &all)).await.unwrap().into_inner()).await.unwrap();
        assert_eq!(outcome(&events).status, "failed", "{events:?}");
        assert_eq!(fs::read_to_string(&counter).unwrap(), "4", "a foreign internal export must run no effects");
        // A published release's environment is described the same way, inside it.
        let python = fs::read_dir(machine.root.join("state/generations"))
            .unwrap()
            .flatten()
            .map(|g| g.path().join("env/bin/python"))
            .find(|p| p.exists())
            .unwrap();
        let (digest, callees) = cozy_machine::published::describe_environment(
            python.to_str().unwrap(),
            "cozy-machine-cpu-caller",
            &Default::default(),
        ).unwrap();
        assert!(digest.starts_with("sha256:"), "{digest}");
        assert_eq!(callees[0].application, "cpu_memo:app", "{callees:?}");
        assert_eq!(callees[0].package, "local/cozy-machine-cpu-memo");
        // The root and callee can both be immutable wheels; no source archive is needed.
        assert!(std::process::Command::new("uv").args(["build", "--wheel", "--out-dir"])
            .arg(&built).arg(fixtures.join("cpu_caller")).status().unwrap().success());
        let root_wheel = fs::read_dir(&built).unwrap().flatten()
            .map(|entry| entry.path()).find(|path| path.file_name().unwrap().to_str().unwrap().starts_with("cozy_machine_cpu_caller-")).unwrap();
        let root_bytes = fs::read(&root_wheel).unwrap();
        let root_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&root_bytes));
        write(&mut client, &all, &root_digest, root_bytes.len() as u64, 0, &root_bytes).await.unwrap();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package":"local/cozy-machine-cpu-caller","release":"0.1.0","python_version":"3.12",
            "wheels":[
                {"name":root_wheel.file_name().unwrap().to_str().unwrap(),"digest":root_digest,"length":root_bytes.len()},
                {"name":wheel.file_name().unwrap().to_str().unwrap(),"digest":wheel_digest,"length":bytes.len()}],
            "callees":{"cozy-machine-cpu-memo":"local/cozy-machine-cpu-memo"}
        })).unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        write(&mut client, &all, &manifest_digest, manifest.len() as u64, 0, &manifest).await.unwrap();
        let wheel_counter = machine.root.join("wheel-measured");
        let events = collect(client.run(authorized(v1::RunRequest {
            id:"relay-wheels".into(),after:0,spec:Some(v1::RunSpec {
                kind:v1::RunKind::Job as i32,entrypoint:"relay".into(),owner:"alice".into(),
                source:Some(v1::run_spec::Source::Local(v1::LocalSource { manifest:manifest_digest })),
                payload:serde_json::to_vec(&serde_json::json!({"values":[3,4],"counter":wheel_counter})).unwrap(),
                ..Default::default()
            }),
        }, &all)).await.unwrap().into_inner()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(fs::read_to_string(&wheel_counter).unwrap(), "2");
        let _ = fs::remove_dir_all(tools);
    }

    /// A published release calls another package its lock pins at the Hub (long_form calling
    /// qwen-image-2's generate_image): the callee installs from the Hub's file door through
    /// TensorFS and runs as a child under its own Hub name, `<org>/<name>`, so the run's model
    /// choices addressed to it reach it. Named `local/<name>` (TensorD 0.4.1-0.5.0), a choice
    /// for generate_image never arrived: model_choice_absent (runs 5141, 5142).
    #[tokio::test]
    async fn a_published_release_calls_its_hub_callee_under_the_callees_own_name() {
        let (machine, tools) = installing_machine().await;
        let mut client = client(&machine).await;
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let built = tools.join("hub-wheels");
        for fixture in ["cpu_caller", "cpu_memo"] {
            assert!(Command::new("uv").args(["build", "--wheel", "--out-dir"]).arg(&built)
                .arg(fixtures.join(fixture)).status().unwrap().success());
        }
        let wheel = |prefix: &str| {
            let path = fs::read_dir(&built).unwrap().flatten().map(|e| e.path())
                .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with(prefix)).unwrap();
            (path.file_name().unwrap().to_str().unwrap().to_string(), fs::read(&path).unwrap())
        };
        let ((root, root_bytes), callee) = (wheel("cozy_machine_cpu_caller-"), wheel("cozy_machine_cpu_memo-"));
        let described = Command::new(installing_args(&tools)[1].clone())
            .args(["-m", "cozy_machine_client.runtime_describe"])
            .env("PYTHONPATH", Path::new(env!("CARGO_MANIFEST_DIR")).join("python"))
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
            .spawn().and_then(|mut child| {
                use std::io::Write as _;
                let project = fixtures.join("cpu_caller");
                child.stdin.take().unwrap().write_all(serde_json::json!({"kind": "describe", "project": project}).to_string().as_bytes())?;
                child.wait_with_output()
            }).unwrap();
        let described: serde_json::Value = serde_json::from_slice(&described.stdout).unwrap();
        let source = std::env::temp_dir().join(format!("cm-callee-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store::Store::ensure(&source).unwrap();
        let model = names_hub::Model {
            repository: "acme/model".into(), release: "1.0.0".into(), lane: "bf16".into(),
            manifest: hub_oauth::checkpoint(&store, &[5; 8192]),
        };
        // The rest of the lock is PyPI's, hash-pinned as the Hub's export pins it.
        let pypi = Command::new("uv").args(["pip", "compile", "--generate-hashes", "--no-header", "--quiet",
            "--python-version", "3.12", "-"])
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
            .spawn().and_then(|mut child| {
                use std::io::Write as _;
                child.stdin.take().unwrap().write_all(b"cozy-runtime>=0.19,<0.20\nmsgspec>=0.19,<1\n")?;
                child.wait_with_output()
            }).unwrap();
        assert!(pypi.status.success());
        let package = names_hub::Package {
            name: "acme/cozy-machine-cpu-caller".into(), release: "0.1.0".into(),
            interface: described["interface"].clone(), wheel: root, bytes: root_bytes, callees: vec![callee],
            pypi: String::from_utf8(pypi.stdout).unwrap(),
        };
        let (origin, _) = names_hub::serve(&store, model, Some(package));
        let counter = machine.root.join("published-measured");
        let events = collect(client.run(authorized(v1::RunRequest {
            id: "published-relay".into(), after: 0, spec: Some(v1::RunSpec {
                kind: v1::RunKind::Job as i32, entrypoint: "relay".into(), owner: "alice".into(),
                source: Some(v1::run_spec::Source::Release(v1::Release {
                    package: "acme/cozy-machine-cpu-caller".into(), release: "0.1.0".into() })),
                payload: serde_json::to_vec(&serde_json::json!({"values": [3, 4], "counter": counter})).unwrap(),
                hub: Some(v1::HubAccess { origin, object_hosts: vec!["localhost".into()], ..Default::default() }),
                catalog_revision: "r1".into(),
                ..Default::default()
            }),
        }, &all)).await.unwrap().into_inner()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = serde_json::from_slice(&done.result).unwrap();
        assert_eq!(result["squares"], serde_json::json!([9, 16]));
        let calls: Vec<_> = events.iter().filter_map(|e| match &e.event {
            Some(v1::run_event::Event::Call(call)) => Some(call.function.clone()),
            _ => None,
        }).collect();
        assert_eq!(calls, ["measure", "measure", "greet"], "{events:?}");
        let named: Vec<String> = fs::read_dir(machine.root.join("state/generations")).unwrap().flatten()
            .filter_map(|g| fs::read(g.path().join("generation.json")).ok())
            .filter_map(|raw| serde_json::from_slice::<serde_json::Value>(&raw).ok())
            .flat_map(|record| record["callees"].as_array().cloned().unwrap_or_default())
            .filter(|callee| callee["distribution"] == "cozy-machine-cpu-memo" || callee["distribution"] == "cozy_machine_cpu_memo")
            .map(|callee| callee["package"].as_str().unwrap_or_default().to_string())
            .collect();
        assert_eq!(named, ["acme/cozy-machine-cpu-memo"]);
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(tools);
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
        // A warm run's set is the caller's warm set: Status shows each member as it was sent,
        // with what it holds now. This function binds no model, so it holds its code.
        let frame = status_frame(&mut client, &all).await;
        assert!(frame.capabilities.iter().any(|c| c == "warm/2"), "{frame:?}");
        let member = |entrypoint: &str| v1::WarmItem {
            source: Some(v1::warm_item::Source::Installation(
                frame.environments[0].installation.clone(),
            )),
            entrypoint: entrypoint.into(),
            level: v1::WarmLevel::Host as i32,
            ..Default::default()
        };
        let set = |items: Vec<v1::WarmItem>| v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            set: Some(v1::WarmSet { items }),
            owner: "alice".into(),
            ..Default::default()
        };
        let kept = client.run(authorized(run("set-1", set(vec![member("steps")])), &all));
        let kept = collect(kept.await.unwrap().into_inner()).await.unwrap();
        assert_eq!(outcome(&kept).status, "succeeded", "{kept:?}");
        let result: serde_json::Value = serde_json::from_slice(&outcome(&kept).result).unwrap();
        assert_eq!(result["set"][0]["level"], "installed", "{result}");
        let frame = status_frame(&mut client, &all).await;
        let [shown] = &frame.warm[..] else {
            panic!("one member: {frame:?}")
        };
        assert_eq!(
            (shown.entrypoint.as_str(), shown.level, shown.holds.as_str()),
            ("steps", v1::WarmLevel::Host as i32, "installed")
        );
        assert!(shown.held_back.contains("binds no model"), "{shown:?}");
        assert_eq!(frame.environments[0].level, "installed");
        // A member naming no entrypoint of its package refuses the whole set: the last one stays.
        let unknown = client.run(authorized(run("set-2", set(vec![member("absent")])), &all));
        let unknown = collect(unknown.await.unwrap().into_inner()).await.unwrap();
        assert_eq!(outcome(&unknown).status, "failed", "{unknown:?}");
        assert_eq!(status_frame(&mut client, &all).await.warm.len(), 1);
        // An empty set clears it.
        let cleared = client.run(authorized(run("set-3", set(vec![])), &all));
        let cleared = collect(cleared.await.unwrap().into_inner()).await.unwrap();
        assert_eq!(outcome(&cleared).status, "succeeded", "{cleared:?}");
        assert!(status_frame(&mut client, &all).await.warm.is_empty());
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
        // What the job's executor measured comes with its outcome, as `run show` lists it: the
        // stage it bracketed and its steps, each with a count and a time.
        let measured: serde_json::Value = serde_json::from_slice(&done.measurements).unwrap();
        let assemble = &measured["attribution"]["stages"]["assemble"];
        assert_eq!(assemble["count"], 3, "{measured}");
        assert!(assemble["total_ms"].as_f64().unwrap() > 0.0, "{measured}");
        let steps = &measured["attribution"]["steps"]["segments"];
        assert_eq!(steps["count"], 3, "{measured}");
        assert_eq!(steps["series"].as_array().unwrap().len(), 3, "{measured}");

        // Each segment saw the job's reference and continued the previous segment's context.
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let mut film = String::new();
        let mut context = String::new();
        for (index, prompt) in segments.iter().enumerate() {
            let body = format!("{index}|{prompt}|{seen}|{context}\n");
            context = tensorfs_core::sha256::hex_digest(body.as_bytes());
            film.push_str(&body);
        }
        // Each child run is a settled call in the job's log, with what its executor measured.
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Call(call)) => Some(call.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls
                .iter()
                .map(|c| (c.index, c.function.as_str(), c.status.as_str()))
                .collect::<Vec<_>>(),
            (0..3)
                .map(|i| (i, "render_segment", "succeeded"))
                .collect::<Vec<_>>(),
            "{events:?}"
        );
        for call in &calls {
            assert!(call.run.ends_with(&format!("/{}", call.index)), "{call:?}");
            assert!(call.finished_at_ms >= call.called_at_ms && call.called_at_ms > 0, "{call:?}");
            assert!(call.reason.is_none(), "{call:?}");
            serde_json::from_slice::<serde_json::Value>(&call.measurements).unwrap();
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
                    ..Default::default()
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), film, "{meta:?}");
        // A child's returned file is published by the job as it is (H3 shows each reference its
        // qwen-image-2 callee generates this way): one `rendered` product per segment, its bytes.
        let rendered: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                Some(v1::run_event::Event::Product(p)) if p.output == "rendered" => Some(p.label.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            rendered,
            ["Segment 1 as rendered", "Segment 2 as rendered", "Segment 3 as rendered"],
            "{events:?}"
        );
        let bodies: Vec<&str> = film.split_inclusive('\n').collect();
        for (index, body) in bodies.iter().enumerate() {
            let (_, bytes) = read(
                &mut client,
                &all,
                v1::ReadRequest {
                    target: Some(v1::read_request::Target::Output(v1::OutputTarget {
                        run: "film".into(),
                        output: "rendered".into(),
                        // A list item's index is 1-based.
                        index: index as u32 + 1,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_eq!(String::from_utf8(bytes).unwrap(), *body, "rendered {index}");
        }

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
            let child = outcome(&child);
            assert_eq!(child.status, "succeeded", "{child:?}");
            // A call's measurements name the process that executed it.
            let measured: serde_json::Value = serde_json::from_slice(&child.measurements).unwrap();
            assert!(measured["execution"]["ranks"][0]["pid"].as_u64() > Some(0), "{measured}");
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
        // A choice naming no slot anywhere is a warning, never a refusal: the run goes on.
        let warned = stray.iter().any(|e| matches!(&e.event,
            Some(v1::run_event::Event::Log(l)) if l.level == "warning" && l.text.contains("nope.models.base")));
        let stray = outcome(&stray);
        assert_eq!(stray.status, "succeeded", "{stray:?}");
        assert!(warned, "the stray choice is not in the run's log as a warning");
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
        // render_segment declares no model slot, so the choice never reaches it: it runs.
        let child = outcome(&child);
        assert_eq!(child.status, "succeeded", "{child:?}");
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
        let (child, address) = launch(&machine.root, &extra);
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
        // Its running time sums both attempts, never the rest in between.
        assert!(done.execution_ms > 0, "{done:?}");
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
                    ..Default::default()
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
        let (child, _) = launch(&machine.root, &extra);
        machine.child = child;
        assert!(!scratch.exists());
        let _ = fs::remove_dir_all(tools);
    }

    /// A job's root sleeps while a segment runs; H3's take minutes on a GPU and say nothing
    /// between denoise steps. That stillness is its child's work, so the job is not ended as
    /// wedged (run 3858 was, 85 s into its first segment; here the root's patience is 40 s).
    #[tokio::test]
    async fn a_job_waits_through_a_segment_longer_than_its_own_patience() {
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
        let spec = v1::RunSpec {
            kind: v1::RunKind::Job as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: serde_json::to_vec(
                &serde_json::json!({"reference": digest, "segments": ["a long night"], "hold": 50.0}),
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
                        id: "patient".into(),
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
        assert_eq!(done.status, "succeeded", "{done:?}");
        drop(machine);
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
        assert_eq!(reason.code, "segment_failed", "{reason:?}");
        assert!(!reason.message.starts_with("segment_failed:"), "{reason:?}");
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
        let child_reason = failed.reason.as_ref().unwrap();
        assert_eq!(child_reason.code, "unhandled_exception");
        assert!(!child_reason.message.starts_with("unhandled_exception:"), "{child_reason:?}");
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
                    ..Default::default()
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

/// Every file under `root`.
fn walkdir(root: &Path) -> Vec<std::path::PathBuf> {
    let mut found = vec![];
    for entry in fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walkdir(&path));
        } else {
            found.push(path);
        }
    }
    found
}
