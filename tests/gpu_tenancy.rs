//! Rental only (`--ignored`): who submits never decides which executor or which GPU weights
//! serve a request. Real executors and inference through the machine's own dispatch, GPU pool
//! and custody; the front door is not in this path (`machine_surface.rs` covers its privacy).
//!
//! `COZY_MACHINE_GPU_TENANCY` names a JSON plan: `state`, `store`, `generations`, `gpu_config`,
//! `entrypoint`, `input`, `output` (JSON lines) and `packages`: two published packages
//! (`{package, release}`) of different publishers that bind the same checkpoint. On a card
//! with room for both executors.
use cozy_machine::{
    gpu_service::{GpuConfig, GpuPool},
    journal::{Execution, Preparation, State, SubmissionContext},
    process::process_ended,
    resident_custody::HoldingFacts,
    service::{Call, Service},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Write,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};
use tensorfs_core::store::Store;

#[derive(Deserialize)]
struct Package {
    package: String,
    release: String,
}
#[derive(Deserialize)]
struct Plan {
    state: PathBuf,
    store: PathBuf,
    generations: PathBuf,
    gpu_config: PathBuf,
    entrypoint: String,
    input: Value,
    packages: [Package; 2],
    output: PathBuf,
}

struct Pod {
    plan: Plan,
    service: Arc<Service>,
    gpu: Arc<GpuPool>,
    output: File,
    salt: String,
    submitted: u32,
}
impl Pod {
    /// `actor` submits one request (its own prompt and seed) to a package; (plan id, record).
    fn run(&mut self, actor: &str, package: usize, input: Option<Value>) -> (String, Execution) {
        let (gpu, service, n) = (&self.gpu, &self.service, self.submitted);
        self.submitted += 1;
        let chosen = &self.plan.packages[package];
        let installed = gpu
            .published_installation(service, actor, &chosen.package, &chosen.release)
            .unwrap()
            .expect("published package mapped in the GPU config");
        let prepared = gpu
            .prepare_root(&installed, &self.plan.entrypoint, &[], &[], 1)
            .unwrap();
        let preparation = Preparation {
            actor: actor.into(),
            id: prepared.id.clone(),
            installation: installed.alias.clone(),
            document: serde_json::to_vec(&prepared).unwrap(),
        };
        service.engine.bind_preparation(preparation).unwrap();
        let input = input.unwrap_or_else(|| {
            let mut input = self.plan.input.clone();
            input["prompt"] = json!(format!("a lighthouse at dawn, study {}-{n}", self.salt));
            input["seed"] = json!(u32::from_str_radix(&self.salt[..7], 16).unwrap() + n);
            input
        });
        let digest = |value: &Value| {
            let canonical = serde_json_canonicalizer::to_vec(value).unwrap();
            format!("sha256:{}", tensorfs_core::sha256::hex_digest(&canonical))
        };
        let context = SubmissionContext {
            actor: actor.into(),
            request_id: format!("tenancy-{}-{n}", self.salt),
            submission_id: format!("tenancy-{}-{n}", self.salt),
            expected_workspace_id: service.engine.workspace_id(),
            capture_digest: digest(&json!({"n": n, "salt": self.salt})),
            invocation_digest: digest(&json!({"preparation": prepared.id})),
            payload_digest: digest(&input),
            publication_authorization_id: String::new(),
            preparation_id: prepared.id.clone(),
        };
        let call = Call {
            entrypoint: self.plan.entrypoint.clone(),
            input: input.clone(),
            ..Default::default()
        };
        let started = Instant::now();
        let accepted = service
            .submit_public(context, &installed.generation, call, "")
            .unwrap();
        let record = loop {
            let epoch = service.engine.activity_epoch();
            let record = service.engine.get(&accepted.id).unwrap();
            if record.state.terminal() {
                break record;
            }
            service
                .engine
                .wait_activity(epoch, Some(Duration::from_secs(1)));
        };
        let used = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let row = json!({"n": n, "actor": actor, "package": chosen.package, "plan": prepared.id,
            "id": record.id, "state": format!("{:?}", record.state), "failure": record.failure,
            "executor": record.process, "wall_s": started.elapsed().as_secs_f64(),
            "holdings": gpu.resident(), "gpu_used_mib": used});
        writeln!(self.output, "{row}").unwrap();
        (prepared.id, record)
    }
    fn pid(record: &Execution) -> u32 {
        record.process.as_ref().expect("the run's executor").pid
    }
    /// Custody's holdings once they are as `settled` says (a call offers after its run ended).
    fn settled(&mut self, settled: impl Fn(&[HoldingFacts]) -> bool) -> Vec<HoldingFacts> {
        // Test harness bound on a state that never comes; the product has no such limit.
        let until = Instant::now() + Duration::from_secs(120);
        loop {
            let held = self.gpu.resident();
            if settled(&held) {
                writeln!(self.output, "{}", json!({ "holdings": held })).unwrap();
                return held;
            }
            assert!(Instant::now() < until, "holdings never settled: {held:?}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    /// Bytes `plan`'s executor has uploaded to the GPU since it started, as of its last call.
    fn uploaded(&self, plan: &str) -> u64 {
        let invokes = std::fs::read_to_string(self.plan.state.join("gpu/invokes.jsonl")).unwrap();
        let mut rows = invokes
            .lines()
            .rev()
            .map(|row| serde_json::from_str::<Value>(row).unwrap());
        let last = rows.find(|row| row["plan"] == plan).expect("a call");
        last["plane"]["h2d_bytes"]
            .as_u64()
            .expect("the plane's upload count")
    }
}
fn reads(holding: &HoldingFacts, pid: u32) -> bool {
    holding.readers.iter().any(|birth| birth.pid == pid)
}
fn layouts(held: &[HoldingFacts]) -> Vec<&str> {
    held.iter().map(|h| h.key.layout.as_str()).collect()
}
fn bytes(held: &[HoldingFacts]) -> u64 {
    held.iter().map(|h| h.bytes).sum()
}

#[test]
#[ignore = "rental GPU with installed generations and cached models"]
fn submitters_share_a_packages_executor_and_packages_share_a_checkpoints_weights() {
    let path = std::env::var("COZY_MACHINE_GPU_TENANCY").expect("plan path");
    let plan: Plan = serde_json::from_reader(File::open(path).unwrap()).unwrap();
    let service = Service::open(&plan.state, &plan.generations, 1).unwrap();
    let store = Arc::new(Store::ensure(&plan.store).unwrap());
    let config = GpuConfig::load(&plan.gpu_config).unwrap();
    let gpu = GpuPool::new(&plan.state.join("gpu"), config, store).unwrap();
    service.configure_gpu(gpu.clone()).unwrap();
    let mut pod = Pod {
        output: File::create(&plan.output).unwrap(),
        plan,
        service,
        gpu,
        salt: uuid::Uuid::new_v4().simple().to_string(),
        submitted: 0,
    };

    // Two submitters, one package: one construction, one executor, one request at a time.
    let (alice_plan, alice) = pod.run("alice", 0, None);
    let (bob_plan, bob) = pod.run("bob", 0, None);
    assert_eq!(
        (alice.state, bob.state),
        (State::Completed, State::Completed)
    );
    assert_eq!(alice_plan, bob_plan);
    assert_eq!(alice.process, bob.process, "one executor served both");
    // Each sees its own run only, and each got its own result.
    let ids = |actor: &str| -> Vec<String> {
        let runs = pod.service.engine.list_actor(actor, 64).unwrap();
        runs.into_iter().map(|run| run.id).collect()
    };
    assert_eq!(
        (ids("alice"), ids("bob")),
        (vec![alice.id.clone()], vec![bob.id.clone()])
    );
    let alices = &alice.submission.as_ref().unwrap().request_id;
    assert!(pod.service.engine.get_public("bob", alices).is_err());
    assert_ne!(alice.result, bob.result);

    // A failed request on the shared executor quotes no stderr from before it began.
    let gpu_root = pod.plan.state.join("gpu");
    let logs = || -> BTreeMap<PathBuf, String> {
        let roots = std::fs::read_dir(&gpu_root).unwrap().flatten();
        roots
            .map(|root| root.path().join("stderr.log"))
            .filter_map(|path| Some((path.clone(), std::fs::read_to_string(path).ok()?)))
            .collect()
    };
    let before = logs();
    let (_, failed) = pod.run("bob", 0, Some(json!({"prompt": 7})));
    assert_eq!(failed.state, State::Failed);
    let (_, bundle) = pod
        .service
        .engine
        .triage(&failed.id)
        .unwrap()
        .expect("its triage");
    let bundle: Value = serde_json::from_slice(&bundle).unwrap();
    let quoted = bundle["executor"]["stderr_tail"].as_str().unwrap();
    let own = logs().into_iter().any(|(path, log)| {
        let from = before.get(&path).map_or(0, String::len);
        from > 0
            && log
                .get(from..)
                .is_some_and(|new| new.trim().ends_with(quoted))
    });
    writeln!(pod.output, "{}", json!({"triage": bundle})).unwrap();
    assert!(
        own,
        "the quote holds stderr from before the request: {quoted}"
    );

    // Another publisher's package binding the same checkpoint: an executor of its own. The
    // machine gives a plan's first, unmeasured call the card (the first package is unmapped
    // and its holdings revoked); its second call fills and offers.
    let (other_plan, first) = pod.run("alice", 1, None);
    let (_, second) = pod.run("bob", 1, None);
    assert_eq!(
        (first.state, second.state),
        (State::Completed, State::Completed)
    );
    assert_ne!(other_plan, alice_plan);
    assert_ne!(first.process, alice.process);
    assert_eq!(first.process, second.process);
    let theirs = Pod::pid(&second);
    let offered = pod.settled(|held| !held.is_empty() && held.iter().all(|h| reads(h, theirs)));

    // The first package again. Its executor's holdings were revoked, so it refills a private
    // copy, and at the end of the call its offer is a duplicate: it lets its copy go and
    // attaches the other package's. One weight set on the GPU for both, charged once.
    let (_, back) = pod.run("alice", 0, None);
    assert_eq!(back.state, State::Completed);
    assert_eq!(back.process, alice.process);
    let ours = Pod::pid(&back);
    let shared = pod.settled(|held| held.iter().all(|h| reads(h, theirs) && reads(h, ours)));
    assert_eq!(layouts(&shared), layouts(&offered));
    assert_eq!(bytes(&shared), bytes(&offered));

    // A replacement executor of the other package attaches at its load, uploading almost
    // nothing.
    let birth = second.process.clone().unwrap();
    assert_eq!(
        cozy_machine::execution::process_birth(birth.pid).unwrap(),
        birth
    );
    // SAFETY: the exact process this test's pool started, by its recorded birth.
    assert_eq!(unsafe { libc::kill(birth.pid as i32, libc::SIGKILL) }, 0);
    while !process_ended(&birth).unwrap() {
        std::thread::sleep(Duration::from_millis(20));
    }
    let (_, replaced) = pod.run("bob", 1, None);
    assert_eq!(replaced.state, State::Completed);
    assert_ne!(replaced.process, second.process);
    let new = Pod::pid(&replaced);
    let after = pod.settled(|held| held.iter().all(|h| reads(h, new) && reads(h, ours)));
    assert_eq!(layouts(&after), layouts(&offered));
    let fresh = pod.uploaded(&other_plan);
    assert!(fresh < bytes(&offered) / 10, "uploaded {fresh} bytes");

    pod.service.stop().unwrap();
    pod.gpu.stop().unwrap(); // every executor's exit observed before the test ends
}
