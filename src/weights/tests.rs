//! The production owner and TensorFS native socket protocol on real private stores.
use super::*;
use serde_json::{json, Value};
use std::{
    io::Write,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

struct Fixture {
    root: PathBuf,
    owner: Weights,
    grant: Grant,
    current: Arc<AtomicBool>,
    output: Frame,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("cozy-weights-{}", uuid::Uuid::new_v4()));
        Self::at(root)
    }

    fn at(root: PathBuf) -> Self {
        std::fs::create_dir_all(&root).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let current = Arc::new(AtomicBool::new(true));
        let state = current.clone();
        let grant = Grant::new(
            "owner",
            "run",
            &root,
            BTreeMap::new(),
            [("model".into(), 8)].into(),
            None,
            Arc::new(move || state.load(Ordering::SeqCst)),
        );
        let plain = tensorfs_core::registry::seeds()
            .into_iter()
            .find(|s| s.alias == "plain/1")
            .unwrap()
            .spec
            .object_id();
        let arguments = serde_json::to_vec(&json!([{}, {"model": {"add": {"weight": {
            "logical_dtype": "f32", "shape": [2], "encoding": plain,
            "parts": {"value": {"dtype": "f32", "shape": [2]}}
        }}, "drop": []}}, {}, [["model", "weight"]], 8]))
        .unwrap();
        std::fs::write(root.join("weights-model-derivation.canonical"), &arguments).unwrap();
        let output = Frame {
            operation: "output".into(),
            output_slot: "model".into(),
            length: arguments.len() as u64,
            ..Default::default()
        };
        Self {
            root,
            owner: Weights::new(store).unwrap(),
            grant,
            current,
            output,
        }
    }

    fn open(&self) -> (String, Wire) {
        let (answer, stream) = self.owner.output(&self.grant, &self.output).unwrap();
        let stream = UnixStream::from(std::os::fd::OwnedFd::from(stream));
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        (answer.transaction, Wire(stream))
    }

    fn adopt(&self, transaction: &str, facts: &Value) -> Result<(Answer, Adopted), Refusal> {
        let data = serde_json::to_vec(facts).unwrap();
        std::fs::write(
            self.root.join("weights-model-native-receipt.canonical"),
            &data,
        )
        .unwrap();
        self.owner.adopt(
            &self.grant,
            &Frame {
                operation: "adopt".into(),
                output_slot: "model".into(),
                transaction: transaction.into(),
                length: data.len() as u64,
                ..Default::default()
            },
        )
    }

    fn wait(&self, transaction: &str, predicate: impl Fn(&derived::Lookup) -> bool) {
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            let state = derived::lookup(&self.owner.store, &self.owner.meta, transaction).unwrap();
            if predicate(&state) {
                break;
            }
            assert!(Instant::now() < end, "native state: {state:?}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

struct Wire(UnixStream);
impl Wire {
    fn frame(&mut self, kind: u8, body: &[u8]) {
        self.0.write_all(&[kind]).unwrap();
        self.0
            .write_all(&(body.len() as u32).to_be_bytes())
            .unwrap();
        self.0.write_all(body).unwrap();
    }
    fn send(&mut self, value: Value) {
        self.frame(0, &serde_json::to_vec(&value).unwrap());
    }
    fn read(&mut self) -> Value {
        let mut head = [0; 5];
        self.0.read_exact(&mut head).unwrap();
        assert_eq!(head[0], 0);
        let mut data = vec![0; u32::from_be_bytes(head[1..].try_into().unwrap()) as usize];
        self.0.read_exact(&mut data).unwrap();
        serde_json::from_slice(&data).unwrap()
    }
    fn ask(&mut self, value: Value) -> Value {
        self.send(value);
        let reply = self.read();
        assert!(reply.get("error").is_none(), "{reply}");
        reply["result"].clone()
    }
    fn part(&mut self) {
        self.send(json!({"op":"part","component":"model","key":"weight","role":"value"}));
        let bytes = [1u8; 8];
        let mut offset = 0;
        loop {
            let reply = self.read();
            if let Some(size) = reply["pull"].as_u64() {
                let end = (offset + size as usize).min(bytes.len());
                self.frame(1, &bytes[offset..end]);
                offset = end;
            } else {
                assert_eq!(reply["result"]["bytes"], 8, "{reply}");
                break;
            }
        }
    }
}

#[test]
fn closed_attempt_resumes_and_lost_commit_ack_replays_without_new_bytes() {
    let mut f = Fixture::new();
    let (transaction, mut first) = f.open();
    first.part();
    assert!(!first.ask(json!({"op":"checkpoint"}))["head"].is_null());
    drop(first);
    f.wait(&transaction, |state| {
        matches!(
            state,
            derived::Lookup::Open {
                writer_session: None
            }
        )
    });
    // Reopening the actual owner and Store discards every in-memory checkpoint callback.
    f.owner = Weights::new(Arc::new(Store::open(&f.root.join("store")).unwrap())).unwrap();
    let (same, mut resumed) = f.open();
    assert_eq!(same, transaction);
    assert_eq!(
        resumed.ask(json!({"op":"completed_parts"})),
        json!([["model", "weight", "value"]])
    );
    resumed.send(json!({"op":"commit"}));
    drop(resumed); // commit before its acknowledgment
    f.wait(&transaction, |state| {
        matches!(state, derived::Lookup::Committed(_))
    });
    let (same, mut replay) = f.open();
    assert_eq!(same, transaction);
    let facts = replay.ask(json!({"op":"receipt"}));
    drop(replay);
    assert_eq!(facts["transaction_id"], transaction);
    assert!(f.adopt(&transaction, &facts).is_ok());
}

#[test]
fn adoption_uses_native_identity_and_refuses_stale_or_malformed_requests() {
    let f = Fixture::new();
    let (transaction, mut writer) = f.open();
    writer.part();
    let facts = writer.ask(json!({"op":"commit"}));
    drop(writer);
    let mut extended = facts.clone();
    extended["future_observation"] = json!({"new_field":true});
    extended["manifest"]["future_observation"] = json!(17);
    for (field, value) in [
        (
            "transaction_id",
            json!(format!("sha256:{}", "ff".repeat(32))),
        ),
        ("declaration_digest", json!(false)),
        (
            "manifest",
            json!({"sha256":facts["manifest"]["sha256"],"length":"same bytes, wrong type"}),
        ),
    ] {
        let mut bad = extended.clone();
        bad[field] = value;
        assert!(f.adopt(&transaction, &bad).is_err());
    }
    f.current.store(false, Ordering::SeqCst);
    assert!(f.adopt(&transaction, &extended).is_err());
    assert!(!f.owner.store.root().join("repos/local").exists());
    f.current.store(true, Ordering::SeqCst);
    let (answer, _) = f.adopt(&transaction, &extended).unwrap();
    assert_eq!(
        answer.manifest,
        facts["manifest"]["sha256"]
            .as_str()
            .map(|sha| format!("sha256:{sha}"))
            .unwrap()
    );
    let native = derived::lookup(&f.owner.store, &f.owner.meta, &transaction).unwrap();
    let derived::Lookup::Committed(native) = native else {
        panic!("committed result disappeared")
    };
    assert_eq!(
        answer.receipt_digest,
        format!(
            "sha256:{}",
            sha256::hex_digest(&tensorfs_core::canon::write(&native.receipt().to_value()))
        )
    );
}

#[test]
fn disposed_output_cannot_replay_or_adopt_even_while_other_consumers_retain_bytes() {
    let f = Fixture::new();
    let (transaction, mut writer) = f.open();
    writer.part();
    let facts = writer.ask(json!({"op":"commit"}));
    drop(writer);
    let manifest = ObjectRef {
        sha256: facts["manifest"]["sha256"].as_str().unwrap().into(),
        length: facts["manifest"]["length"].as_u64().unwrap(),
    };
    let repos: Vec<_> = ["consumer-a", "consumer-b"]
        .into_iter()
        .map(|name| RepositoryName::new("local", name).unwrap())
        .collect();
    for repo in &repos {
        f.owner
            .store
            .apply_repository(
                None,
                &Mutation::ReplaceLocal {
                    repo: repo.clone(),
                    manifest: manifest.clone(),
                    version: manifest.sha256.clone(),
                },
                &Fault::default(),
            )
            .unwrap();
    }
    derived::dispose(&f.owner.store, &f.owner.meta, &transaction).unwrap();
    assert_eq!(
        f.owner.output(&f.grant, &f.output).unwrap_err().0,
        "weights_transaction_closed"
    );
    assert_eq!(
        f.adopt(&transaction, &facts).err().unwrap().0,
        "weights_transaction_closed"
    );
    for (index, repo) in repos.iter().enumerate() {
        let current = std::fs::read(f.owner.store.repository_path(repo)).unwrap();
        f.owner
            .store
            .apply_repository(
                Some(&current),
                &Mutation::DeleteRepository { repo: repo.clone() },
                &Fault::default(),
            )
            .unwrap();
        tensorfs_core::gc::collect(f.owner.store.root(), false).unwrap();
        assert_eq!(
            f.owner.store.manifest_path(&manifest.sha256).exists(),
            index == 0
        );
    }
}

#[test]
fn missing_pending_root_refuses_replay_without_recreating_custody() {
    let f = Fixture::new();
    let (transaction, mut writer) = f.open();
    writer.part();
    writer.ask(json!({"op":"commit"}));
    drop(writer);
    std::fs::remove_file(f.owner.store.root().join("roots/derived").join(format!(
        "{}.json",
        transaction.trim_start_matches("sha256:")
    )))
    .unwrap();
    assert_eq!(
        f.owner.output(&f.grant, &f.output).unwrap_err().0,
        "weights_receipt_unavailable"
    );
}

#[test]
#[ignore = "subprocess fixture for owner_process_death_preserves_the_native_transaction"]
fn interrupted_owner_process() {
    let root =
        std::env::var_os("COZY_WEIGHTS_TEST_ROOT").expect("explicit subprocess fixture root");
    let f = Fixture::at(root.into());
    let (transaction, mut writer) = f.open();
    writer.part();
    writer.ask(json!({"op":"checkpoint"}));
    println!("NATIVE_READY:{transaction}");
    std::io::stdout().flush().unwrap();
    loop {
        std::thread::park();
    }
}

#[test]
fn owner_process_death_preserves_the_native_transaction() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let root = std::env::temp_dir().join(format!("cozy-weights-death-{}", uuid::Uuid::new_v4()));
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "weights::tests::interrupted_owner_process",
            "--ignored",
            "--nocapture",
        ])
        .env("COZY_WEIGHTS_TEST_ROOT", &root)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let transaction = loop {
        let mut line = String::new();
        assert!(
            output.read_line(&mut line).unwrap() > 0,
            "owner exited before checkpoint"
        );
        if let Some(id) = line.trim().strip_prefix("NATIVE_READY:") {
            break id.to_owned();
        }
    };
    child.kill().unwrap();
    child.wait().unwrap();
    let f = Fixture::at(root);
    let (resumed, mut writer) = f.open();
    assert_eq!(resumed, transaction);
    assert_eq!(
        writer.ask(json!({"op":"completed_parts"})),
        json!([["model", "weight", "value"]])
    );
    let facts = writer.ask(json!({"op":"commit"}));
    drop(writer);
    assert_eq!(facts["transaction_id"], transaction);
    assert!(f.adopt(&transaction, &facts).is_ok());
}
