//! Rental only (`--ignored`): published models alternate on one GPU through the machine's own
//! dispatch, GPU pool and memory policy, with real executors and inference. The front door and
//! Hub are not in this path, so it is not ordinary-CLI proof.
//!
//! `COZY_MACHINE_GPU_ALTERNATION` names a JSON plan: `state`, `store`, `generations`,
//! `gpu_config`, `actor`, `models: {name: {package, release, entrypoint, input}}`, `order`
//! (model names), `output` (JSON lines), and optional `squeeze: {at, start, release}`: before
//! request `at`, run `start` (a ballast taking device memory while the call runs) and create
//! `release` once that request ends.
use cozy_machine::{
    gpu_service::{GpuConfig, GpuPool},
    journal::{Preparation, State, SubmissionContext},
    service::Service,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};
use tensorfs_core::store::Store;

#[derive(Deserialize)]
struct Model {
    package: String,
    release: String,
    entrypoint: String,
    input: Value,
}
#[derive(Deserialize)]
struct Squeeze {
    at: usize,
    start: String,
    release: PathBuf,
}
#[derive(Deserialize)]
struct Plan {
    state: PathBuf,
    store: PathBuf,
    generations: PathBuf,
    gpu_config: PathBuf,
    actor: String,
    models: BTreeMap<String, Model>,
    order: Vec<String>,
    output: PathBuf,
    squeeze: Option<Squeeze>,
}

#[test]
#[ignore = "rental GPU with installed generations and cached models"]
fn published_models_alternate_on_one_gpu_under_the_memory_policy() {
    let path = std::env::var("COZY_MACHINE_GPU_ALTERNATION").expect("plan path");
    let plan: Plan = serde_json::from_reader(File::open(path).unwrap()).unwrap();
    let service = Service::open(&plan.state, &plan.generations, 1).unwrap();
    let store = Arc::new(Store::ensure(&plan.store).unwrap());
    let gpu = GpuPool::new(
        &plan.state.join("gpu"),
        GpuConfig::load(&plan.gpu_config).unwrap(),
        store,
    )
    .unwrap();
    service.configure_gpu(gpu.clone()).unwrap();
    let mut output = File::create(&plan.output).unwrap();
    let salt = uuid::Uuid::new_v4().simple().to_string();
    let mut failed = vec![];
    for (n, name) in plan.order.iter().enumerate() {
        let model = &plan.models[name];
        let installed = gpu
            .published_installation(&service, &plan.actor, &model.package, &model.release)
            .unwrap()
            .expect("published package mapped in the GPU config");
        let prepared = gpu
            .prepare_root(&plan.actor, &installed, &model.entrypoint, &[], &[], 1)
            .unwrap();
        service
            .engine
            .bind_preparation(Preparation {
                actor: plan.actor.clone(),
                id: prepared.id.clone(),
                installation: installed.alias.clone(),
                document: serde_json::to_vec(&prepared).unwrap(),
            })
            .unwrap();
        // A unique prompt and seed per request; every other field as the plan states it.
        let mut input = model.input.clone();
        input["prompt"] = json!(format!(
            "{}, study {salt}-{n}",
            input["prompt"].as_str().unwrap_or("a lighthouse at dawn")
        ));
        input["seed"] = json!(u32::from_str_radix(&salt[..7], 16).unwrap() + n as u32);
        let digest = |value: &Value| {
            format!(
                "sha256:{}",
                tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(
                    &serde_json_canonicalizer::to_vec(value).unwrap()
                ))
            )
        };
        let context = SubmissionContext {
            actor: plan.actor.clone(),
            request_id: format!("b2-{salt}-{n}"),
            submission_id: format!("b2-{salt}-{n}"),
            expected_workspace_id: service.engine.workspace_id(),
            capture_digest: digest(&json!({"n": n, "salt": salt})),
            invocation_digest: digest(&json!({"preparation": prepared.id})),
            payload_digest: digest(&input),
            publication_authorization_id: String::new(),
            preparation_id: prepared.id.clone(),
        };
        let mut ballast = plan
            .squeeze
            .as_ref()
            .filter(|squeeze| squeeze.at == n)
            .map(|squeeze| {
                let _ = fs::remove_file(&squeeze.release);
                Command::new("sh")
                    .args(["-c", &squeeze.start])
                    .spawn()
                    .unwrap()
            });
        let started = Instant::now();
        let accepted = service
            .submit_public(
                context,
                &installed.generation,
                cozy_machine::service::Call {
                    entrypoint: model.entrypoint.clone(),
                    input: input.clone(),
                    attention_kernel: String::new(),
                    inputs: vec![],
                },
                "",
            )
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
        if let (Some(squeeze), Some(child)) = (&plan.squeeze, ballast.as_mut()) {
            File::create(&squeeze.release).unwrap();
            child.wait().unwrap();
        }
        let artifacts: Vec<_> = record
            .result
            .as_ref()
            .map(|result| {
                result
                    .artifacts
                    .iter()
                    .map(|a| plan.state.join("execution").join(&a.path))
                    .collect()
            })
            .unwrap_or_default();
        let row = json!({"n": n, "model": name, "id": record.id, "state": format!("{:?}", record.state),
            "failure": record.failure, "wall_s": started.elapsed().as_secs_f64(),
            "accepted_at_ms": record.accepted_at_ms, "finished_at_ms": record.finished_at_ms,
            "squeezed": ballast.is_some(), "input": input, "artifacts": artifacts});
        writeln!(output, "{row}").unwrap();
        if record.state != State::Completed {
            failed.push(row);
        }
    }
    service.stop().unwrap();
    gpu.stop().unwrap(); // every executor's exit observed before the test ends
    assert!(failed.is_empty(), "requests failed: {failed:?}");
}
