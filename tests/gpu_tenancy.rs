//! Rental only (`--ignored`): who submits never decides which executor or which GPU weights
//! serve a request. Real executors and inference through the machine's own dispatch, GPU pool
//! and custody; the front door is not in this path (`machine_surface.rs` covers its privacy).
//!
//! `COZY_MACHINE_GPU_TENANCY` names a JSON plan: `state`, `store`, `generations`, `gpu_config`,
//! `entrypoint`, `input`, `output` (JSON lines) and `packages`: two published packages
//! (`{package, release}`) of different publishers that bind the same checkpoint.
use cozy_machine::{
    gpu_service::{GpuConfig, GpuPool},
    journal::{Execution, Preparation, State, SubmissionContext},
    process::process_ended,
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

    // Another publisher's package binding the same checkpoint: an executor of its own, the
    // same GPU weights. Its first executor filled a copy before the plan was measured; the
    // replacement attaches what custody holds.
    let (other_plan, first) = pod.run("alice", 1, None);
    assert_eq!(first.state, State::Completed);
    assert_ne!(other_plan, alice_plan);
    assert_ne!(first.process, alice.process);
    let before = pod.gpu.resident();
    assert_eq!(before.len(), 1, "{before:?}");
    let birth = first.process.clone().unwrap();
    // SAFETY: the exact process this test's pool started, by its recorded birth.
    assert_eq!(
        cozy_machine::execution::process_birth(birth.pid).unwrap(),
        birth
    );
    assert_eq!(unsafe { libc::kill(birth.pid as i32, libc::SIGKILL) }, 0);
    while !process_ended(&birth).unwrap() {
        std::thread::sleep(Duration::from_millis(20));
    }
    let (_, second) = pod.run("bob", 1, None);
    assert_eq!(second.state, State::Completed);
    assert_ne!(second.process, first.process);
    let after = pod.gpu.resident();
    assert_eq!(
        after.len(),
        1,
        "one weight set for both packages: {after:?}"
    );
    assert_eq!(after[0].bytes, before[0].bytes);
    let readers: Vec<u32> = after[0].readers.iter().map(|birth| birth.pid).collect();
    assert!(readers.contains(&Pod::pid(&second)), "{readers:?}");

    pod.service.stop().unwrap();
    pod.gpu.stop().unwrap(); // every executor's exit observed before the test ends
}
