//! Explicit isolated hardware pilot; validate never launches Python or initializes a device.
//! `run A.json B.json ...` runs each config's executor in turn in one process: with
//! `host_tier`, their weights come from one machine host tier that outlives each executor.
use cozy_machine::device_executor;
use cozy_machine::host_tier::{HalfOfHeadroom, HostGrant, HostTier, HostTierConfig, SealedRequest, TierLimit};
use cozy_machine::model_sources::{ModelSources, SelectedManifest};

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
    /// Auto negotiates existing stage policy; false is a fixed-allowance single-executor experiment.
    #[serde(default)]
    stages: Option<bool>,
    /// Weights from the machine's sealed host tier (`host_tiers.sealed/1`) when offered.
    #[serde(default)]
    host_tier: bool,
    /// A fixed host tier limit for this pilot (an experiment's stand-in for a smaller host);
    /// otherwise half the live headroom.
    #[serde(default)]
    host_tier_limit_bytes: Option<u64>,
    payloads: Vec<Value>,
}

struct FixedLimit(u64);
impl TierLimit for FixedLimit {
    fn limit(&self, _: &cozy_machine::host_memory::HostMemory, _: u64) -> u64 {
        self.0
    }
}

struct Turns {
    budget: i64,
    held: Vec<File>,
    events: File,
    phase: String,
    /// The host tier, this executor in it, and what it may adopt.
    host: Option<(Arc<HostTier>, u64, Vec<HostGrant>)>,
    /// Its models' headers and assets (`model_source`), with the host tier.
    sources: Option<ModelSources>,
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
        if let (Kind::SealedPrefetch, Some((tier, peer, grants))) = (frame.kind, &self.host) {
            let plans = descriptor.ok_or_else(|| io::Error::other("sealed prefetch omitted its plans"))?;
            let request = SealedRequest { sha256: &frame.sha256, length: frame.length };
            if let Err(error) = tier.prefetch(*peer, grants, request, plans) {
                answer.detail = error.to_string();
            } else {
                answer.ok = true;
                answer.code.clear();
                answer.detail.clear();
            }
            let row = serde_json::json!({"phase":self.phase,"exchange":"SealedPrefetch","ok":answer.ok,"detail":answer.detail});
            writeln!(self.events, "{row}")?;
            return Ok((answer, None));
        }
        if let (Kind::ModelSource, Some(sources)) = (frame.kind, &self.sources) {
            drop(descriptor);
            let (answer, file) = sources.serve(frame)?;
            let row = serde_json::json!({"phase":self.phase,"exchange":"ModelSource","name":frame.name,"length":answer.length});
            writeln!(self.events, "{row}")?;
            return Ok((answer, Some(file)));
        }
        if let (Kind::SealedTier, Some((tier, peer, grants))) = (frame.kind, &self.host) {
            let plan = descriptor.ok_or_else(|| io::Error::other("sealed tier omitted its plan"))?;
            let started = Instant::now();
            let request = SealedRequest { sha256: &frame.sha256, length: frame.length };
            let granted = match tier.seal(*peer, grants, request, plan) {
                Ok(granted) => granted,
                Err(error) => {
                    answer.code = "sealed_tier_refused".into();
                    answer.detail = error.to_string();
                    None
                }
            };
            if answer.code != "sealed_tier_refused" {
                (answer.ok, answer.held) = (true, granted.is_some());
                answer.code.clear();
                answer.detail.clear();
            }
            let row = serde_json::json!({"phase":self.phase,"exchange":"SealedTier","held":answer.held,"code":answer.code,"detail":answer.detail,"ms":started.elapsed().as_secs_f64()*1e3});
            writeln!(self.events, "{row}")?;
            return Ok((answer, granted));
        }
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
        let row = serde_json::json!({"phase":self.phase,"exchange":format!("{:?}",frame.kind),"method":frame.method,"components":frame.components,"ok":answer.ok,"budget":answer.budget_bytes,"yielded":frame.yielded,"passes":frame.passes,"wall_ns":frame.wall_ns,"growth_bytes":frame.growth_bytes,"stall_ns":frame.stall_ns});
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
    metrics: Option<device_executor::Metrics>,
    plane: Option<device_executor::PlaneFacts>,
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
        .ok_or_else(|| io::Error::other("usage: device-pilot validate|run CONFIG.json..."))?;
    let paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if paths.is_empty() {
        return Err(io::Error::other("config path required"));
    }
    let mut tier: Option<Arc<HostTier>> = None;
    for path in paths {
        let config: Pilot = serde_json::from_slice(&fs::read(&path)?)?;
        if config.host_tier && tier.is_none() {
            let store = Arc::new(tensorfs_core::store::Store::open(std::path::Path::new(&config.binding.store)).map_err(io::Error::other)?);
            let plans = Some(config.root.parent().unwrap_or(&config.root).join("host-plans"));
            let limit: Box<dyn TierLimit> = match config.host_tier_limit_bytes {
                Some(bytes) => Box::new(FixedLimit(bytes)),
                None => Box::new(HalfOfHeadroom),
            };
            tier = Some(HostTier::new(store, HostTierConfig { plans, ..HostTierConfig::default() }, limit)?);
        }
        pilot(&action, config, tier.as_ref())?;
    }
    Ok(())
}

fn pilot(action: &str, config: Pilot, tier: Option<&Arc<HostTier>>) -> io::Result<()> {
    let total_started = Instant::now();
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
            serde_json::json!({"validated":true,"mode":"stock-executor-pilot","requested_stages":config.stages,"model":config.binding.model_class,"snapshots":config.binding.snapshots,"requests":config.payloads.len(),"gpu_started":false})
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
    // The pilot's configured device/allocator/thread values become the executor seal.
    let configured = |name: &str| config.environment.get(name).cloned();
    let mut seal = cozy_machine::launch_identity::Seal::prepare(
        &config.root,
        None,
        "pilot",
        "pilot",
        &configured("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
    )?;
    if let Some(alloc_conf) = configured("PYTORCH_CUDA_ALLOC_CONF") {
        seal.alloc_conf = alloc_conf;
    }
    if let Some(threads) = configured("OMP_NUM_THREADS") {
        seal.threads = threads.parse().map_err(io::Error::other)?;
    }
    let spawn_started = Instant::now();
    let mut executor = DeviceExecutor::spawn(ExecutorConfig {
        python: config.python,
        root: config.root.clone(),
        socket: config.socket,
        environment: config.environment,
        seal,
        generation_hold: hold,
        identity: None,
        cgroup_namespace: None,
    })?;
    let spawn_ms = spawn_started.elapsed().as_secs_f64() * 1000.;
    let devices = executor
        .hello
        .sealed
        .get("CUDA_VISIBLE_DEVICES")
        .cloned()
        .unwrap_or_default();
    let stages = config.stages.unwrap_or(true) && executor.hello.offers("stage/1");
    let plane = executor.hello.offers("weight_plane/1");
    let sealed = config.host_tier && plane && executor.hello.offers("host_tiers.sealed/1");
    let at_load = plane && executor.hello.offers("load_pinned/1");
    fs::write(
        config.root.join("negotiation.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "runtime":executor.hello.runtime_version,
            "tensorfs":executor.hello.tensorfs_version,
            "offered_memory":executor.hello.memory,
            "stage_turns":stages,
            "requested_stages":config.stages,
            "weight_plane":plane,
            "legacy_residency":!plane,
            "sealed_tiers":sealed,
            "pinned_at_load":at_load,
            "executor_store_path":&config.binding.store
        }))?,
    )?;
    let mut turns = Turns {
        budget: config.plane_budget_bytes,
        held: Vec::new(),
        events: File::create(config.root.join("events.jsonl"))?,
        phase: "start".into(),
        host: None,
        sources: None,
    };
    if let (true, Some(tier)) = (sealed, tier) {
        let selections = selected_sources(&config.binding)?;
        let sources = ModelSources::open(&PathBuf::from(&config.binding.store), &selections)?;
        let grants = selections
            .iter()
            .map(|s| {
                Ok(HostGrant {
                    manifest: s.manifest.clone(),
                    header: sources.authorized_header(&s.manifest)?,
                    components: s.components.iter().cloned().collect(),
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        tier.prepare(grants.clone());
        turns.host = Some((tier.clone(), tier.register_peer(executor.observer_pidfd()?), grants));
        turns.sources = Some(sources);
    }
    let disk_before = disk_read_bytes();
    let mut timings = BTreeMap::new();
    timings.insert("spawn_ms", spawn_ms);
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
    let load_binding = config.binding;
    let loaded = executor.command(
        &DeviceCommand::Load {
            construction: "pilot-model".into(),
            devices,
            sequence_parallel_degree: 1,
            binding: Box::new(load_binding),
            budgets: Budgets {
                declared_weight_bytes: config.logical_weight_bytes,
            },
            models: Vec::new(),
            authorized_device_limit_bytes: Some(config.authorized_device_limit_bytes),
            attention_pin: String::new(),
            stages,
            device_weights: false,
            cap_bytes: None,
            sealed_tiers: sealed,
            model_sources: sealed,
            pinned_bytes: at_load.then_some(config.pinned_budget_bytes),
        },
        &mut turns,
    )?;
    timings.insert("load_ms", start.elapsed().as_secs_f64() * 1000.);
    let machine = serde_json::json!({
        "host_tier": tier.map(|t| t.facts()),
        "machine": cozy_machine::host_memory::process(std::process::id()).ok(),
        "executor": cozy_machine::host_memory::process(executor.birth.pid).ok(),
        "host": cozy_machine::host_memory::read(),
        "machine_disk_read_bytes": disk_read_bytes() - disk_before,
        "machine_fds": fd_count()?,
    });
    fs::write(config.root.join("load-machine.json"), serde_json::to_vec_pretty(&machine)?)?;
    if !loaded.ok {
        return Err(io::Error::other(format!(
            "load: {} {}",
            loaded.code, loaded.detail
        )));
    }
    fs::write(
        config.root.join("load-facts.json"),
        serde_json::to_vec_pretty(&loaded.facts)?,
    )?;
    if plane {
        let budget = executor.command(
            &DeviceCommand::Budget {
                vram_bytes: config.plane_budget_bytes,
                pinned_bytes: if at_load { -1 } else { config.pinned_budget_bytes },
                cap_bytes: None,
            },
            &mut turns,
        )?;
        if !budget.ok {
            return Err(io::Error::other(format!(
                "budget: {} {}",
                budget.code, budget.detail
            )));
        }
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
                attention_kernel: String::new(),
                input_metadata: Default::default(),
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
                stages,
                cap_bytes: None,
                inputs: Default::default(),
                floor_bytes: None,
                activation_bytes: Default::default(),
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
            metrics: reply.metrics,
            plane: reply.plane,
        });
        fs::write(
            config.root.join("results.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"qualification":"stock executor pilot; machine host/GPU ownership/full machine API unqualified","stage_turns":stages,"weight_plane":plane,"timings":timings,"runs":results}),
            )?,
        )?;
    }
    let shutdown_started = Instant::now();
    executor.shutdown()?;
    timings.insert(
        "shutdown_ms",
        shutdown_started.elapsed().as_secs_f64() * 1000.,
    );
    timings.insert(
        "total_wall_ms",
        total_started.elapsed().as_secs_f64() * 1000.,
    );
    fs::write(
        config.root.join("results.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"qualification":"stock executor pilot; machine host/GPU ownership/full machine API unqualified","stage_turns":stages,"weight_plane":plane,"sealed_tiers":sealed,"timings":timings,"runs":results,"host_tier":tier.map(|t| t.facts())}),
        )?,
    )?;
    Ok(())
}

fn disk_read_bytes() -> u64 {
    fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("read_bytes: ")?.trim().parse().ok()))
        .unwrap_or(0)
}

fn fd_count() -> io::Result<usize> {
    Ok(fs::read_dir("/proc/self/fd")?.count())
}

fn selected_sources(binding: &Binding) -> io::Result<Vec<SelectedManifest>> {
    let components = if binding.components.is_empty() {
        vec![binding.component.clone()]
    } else {
        binding.components.clone()
    };
    let mut grouped = BTreeMap::<String, Vec<String>>::new();
    for component in components {
        let manifest = binding
            .snapshots
            .get(&component)
            .unwrap_or(&binding.snapshot);
        if component.is_empty() || manifest.is_empty() {
            return Err(io::Error::other(
                "descriptor selection requires actual component/snapshot bindings",
            ));
        }
        let selected = grouped.entry(manifest.clone()).or_default();
        if !selected.contains(&component) {
            selected.push(component);
        }
    }
    Ok(grouped
        .into_iter()
        .map(|(manifest, components)| SelectedManifest {
            manifest,
            components,
        })
        .collect())
}
