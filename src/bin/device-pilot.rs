//! Explicit isolated hardware pilot; validate never launches Python or initializes a device.
use cozy_machine::device_executor;

use device_executor::{
    postprocess, Answer, Baseline, Binding, Budgets, DeviceCommand, DeviceExecutor, ExecutorConfig,
    Frame, Kind, Services,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

#[derive(Deserialize)]
struct Pilot {
    python: PathBuf,
    root: PathBuf,
    socket: PathBuf,
    environment: BTreeMap<String, String>,
    package_interface: PathBuf,
    binding: Binding,
    logical_weight_bytes: u64,
    authorized_device_limit_bytes: u64,
    plane_budget_bytes: i64,
    pinned_budget_bytes: i64,
    #[serde(default)]
    generation_hold: Option<PathBuf>,
    payloads: Vec<Value>,
}

struct Turns {
    budget: i64,
    held: Vec<File>,
    events: File,
    phase: String,
}
impl Services for Turns {
    fn progress(&mut self, frame: &Frame) {
        let row = serde_json::json!({"phase":self.phase,"request_id":frame.request_id,"stage":frame.stage,"position":frame.position,"total":frame.total,"advance":frame.advance});
        let _ = writeln!(self.events, "{row}");
    }
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        let mut answer = Answer::unavailable(frame.seq);
        match frame.kind {
            Kind::BudgetCell => {
                let descriptor =
                    descriptor.ok_or_else(|| io::Error::other("budget cell omitted descriptor"))?;
                if descriptor.metadata()?.len() != 32 {
                    return Err(io::Error::other("budget cell layout differs"));
                }
                self.held.push(descriptor);
                answer.ok = true;
                answer.code.clear();
                answer.detail.clear();
            }
            Kind::StageEnter | Kind::StageExit => {
                drop(descriptor);
                answer.ok = true;
                answer.code.clear();
                answer.detail.clear();
                answer.budget_bytes = self.budget;
            }
            _ => drop(descriptor),
        }
        let row = serde_json::json!({"phase":self.phase,"exchange":format!("{:?}",frame.kind),"method":frame.method,"ok":answer.ok,"budget":answer.budget_bytes});
        writeln!(self.events, "{row}")?;
        Ok((answer, None))
    }
}

#[derive(Serialize)]
struct RunEvidence {
    id: String,
    pid: u32,
    wall_ms: f64,
    prepare_ms: f64,
    invoke_ms: f64,
    post_ms: f64,
    result: Value,
    bindings: Vec<device_executor::AssetBinding>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("device-pilot: {error}");
        std::process::exit(1);
    }
}
fn run() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let action = args
        .next()
        .ok_or_else(|| io::Error::other("usage: device-pilot validate|run CONFIG.json"))?;
    let path = PathBuf::from(
        args.next()
            .ok_or_else(|| io::Error::other("config path required"))?,
    );
    let config: Pilot = serde_json::from_slice(&fs::read(&path)?)?;
    if !config.python.is_file()
        || !config.package_interface.is_file()
        || config.binding.model_class.is_empty()
        || config.payloads.is_empty()
    {
        return Err(io::Error::other(
            "pilot requires actual interpreter/interface/model binding and unchanged requests",
        ));
    }
    let interface: Value = serde_json::from_slice(&fs::read(&config.package_interface)?)?;
    if interface["application"] != config.binding.application {
        return Err(io::Error::other("binding/interface application differs"));
    }
    if config.binding.store.is_empty() || config.binding.snapshots.is_empty() {
        return Err(io::Error::other(
            "legacy pilot requires an authoritative coherent TensorFS store/snapshot map",
        ));
    }
    if action == "validate" {
        println!(
            "{}",
            serde_json::json!({"validated":true,"mode":"legacy-stock-executor-pilot","model":config.binding.model_class,"snapshots":config.binding.snapshots,"requests":config.payloads.len(),"gpu_started":false})
        );
        return Ok(());
    }
    if action != "run" {
        return Err(io::Error::other("unknown pilot action"));
    }
    fs::create_dir_all(&config.root)?;
    let hold = config
        .generation_hold
        .map(|path| {
            let file = File::open(path)?;
            fs2::FileExt::lock_shared(&file)?;
            Ok::<_, io::Error>(Arc::new(file))
        })
        .transpose()?;
    let mut executor = DeviceExecutor::spawn(ExecutorConfig {
        python: config.python,
        root: config.root.clone(),
        socket: config.socket,
        environment: config.environment,
        generation_hold: hold,
    })?;
    let devices = executor
        .hello
        .sealed
        .get("CUDA_VISIBLE_DEVICES")
        .cloned()
        .unwrap_or_default();
    let mut turns = Turns {
        budget: config.plane_budget_bytes,
        held: Vec::new(),
        events: File::create(config.root.join("events.jsonl"))?,
        phase: "start".into(),
    };
    let mut timings = BTreeMap::new();
    let start = Instant::now();
    let started = executor.command(
        &DeviceCommand::Start {
            devices: devices.clone(),
            application: config.binding.application.clone(),
            package_interface: config.package_interface,
            sequence_parallel_degree: 1,
            import_only: false,
        },
        &mut turns,
    )?;
    timings.insert("start_ms", start.elapsed().as_secs_f64() * 1000.);
    if !started.ok {
        return Err(io::Error::other(format!(
            "start: {} {}",
            started.code, started.detail
        )));
    }
    turns.phase = "load".into();
    let start = Instant::now();
    let loaded = executor.command(
        &DeviceCommand::Load {
            construction: "pilot-model".into(),
            devices,
            sequence_parallel_degree: 1,
            binding: Box::new(config.binding),
            budgets: Budgets {
                declared_weight_bytes: config.logical_weight_bytes,
            },
            authorized_device_limit_bytes: Some(config.authorized_device_limit_bytes),
            attention_pin: String::new(),
            host_tier: false,
            stages: true,
            descriptor_sources: false,
        },
        &mut turns,
    )?;
    timings.insert("load_ms", start.elapsed().as_secs_f64() * 1000.);
    if !loaded.ok {
        return Err(io::Error::other(format!(
            "load: {} {}",
            loaded.code, loaded.detail
        )));
    }
    let budget = executor.command(
        &DeviceCommand::Budget {
            vram_bytes: config.plane_budget_bytes,
            pinned_bytes: config.pinned_budget_bytes,
        },
        &mut turns,
    )?;
    if !budget.ok {
        return Err(io::Error::other(format!(
            "budget: {} {}",
            budget.code, budget.detail
        )));
    }
    let active = executor.command(
        &DeviceCommand::Activate {
            construction: "pilot-model".into(),
        },
        &mut Baseline,
    )?;
    if !active.ok {
        return Err(io::Error::other(format!(
            "activate: {} {}",
            active.code, active.detail
        )));
    }
    let mut results = Vec::new();
    for (index, payload) in config.payloads.into_iter().enumerate() {
        let id = format!("pilot-{index}");
        turns.phase = id.clone();
        let spool = config.root.join(&id);
        fs::create_dir(&spool)?;
        let all = Instant::now();
        let start = Instant::now();
        let prepared = executor.command(
            &DeviceCommand::PrepareRequest {
                request_id: id.clone(),
                construction: "pilot-model".into(),
                entrypoint: "generate".into(),
                payload,
            },
            &mut turns,
        )?;
        let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
        if !prepared.ok {
            return Err(io::Error::other(format!(
                "prepare: {} {}",
                prepared.code, prepared.detail
            )));
        }
        let start = Instant::now();
        let reply = executor.command(
            &DeviceCommand::Invoke {
                request_id: id.clone(),
                construction: "pilot-model".into(),
                entrypoint: "generate".into(),
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: config.plane_budget_bytes,
                stages: true,
            },
            &mut turns,
        )?;
        let invoke_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let (result, bindings) = postprocess(&executor.codec(), &spool, &reply)?;
        let post_ms = start.elapsed().as_secs_f64() * 1000.;
        results.push(RunEvidence {
            id,
            pid: executor.birth.pid,
            wall_ms: all.elapsed().as_secs_f64() * 1000.,
            prepare_ms,
            invoke_ms,
            post_ms,
            result,
            bindings,
        });
        fs::write(
            config.root.join("results.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"qualification":"legacy pilot; store writer/host ownership/full machine API unqualified","timings":timings,"runs":results}),
            )?,
        )?;
    }
    executor.shutdown()
}
