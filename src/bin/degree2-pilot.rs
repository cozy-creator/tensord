//! Development tool (not proof, PLAN decision 5): real stock executors on one GPU sharing GPU
//! weights through the machine's real `ResidentCustody` and seam. Measures what Degree 2 buys:
//! a replacement's first image with attached weights against a private fill, the bytes two
//! live executors hold (NVML total), the release after a revoke, and output equality.
use cozy_machine::device_executor::{
    postprocess, Answer, Baseline, Binding, Budgets, DeviceCommand, DeviceExecutor, ExecutorConfig,
    Frame, Kind, Services,
};
use cozy_machine::resident_custody::{HoldingKey, Offered, Reader, ResidentCustody};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Deserialize)]
struct Config {
    python: PathBuf,
    root: PathBuf,
    environment: BTreeMap<String, String>,
    package_interface: PathBuf,
    binding: Binding,
    logical_weight_bytes: u64,
    authorized_device_limit_bytes: u64,
    plane_budget_bytes: i64,
    pinned_budget_bytes: i64,
    entrypoint: String,
    /// Unique prompt and seed each; one per timed executor below.
    payloads: Vec<Value>,
    /// Run once by an attaching and once by a private executor: outputs must be identical.
    same: Value,
}

struct Pilot {
    custody: Arc<Mutex<ResidentCustody>>,
    reader: Option<(cozy_machine::journal::ProcessBirth, File)>,
    budget: i64,
    cells: Vec<File>,
    log: File,
}
impl Services for Pilot {
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        let mut answer = Answer::unavailable(frame.seq);
        match frame.kind {
            Kind::BudgetCell => {
                self.cells.extend(descriptor);
                answer.ok = true;
            }
            Kind::StageEnter | Kind::StageExit => {
                answer.ok = true;
                answer.budget_bytes = self.budget;
            }
            _ => (),
        }
        if answer.ok {
            answer.code.clear();
            answer.detail.clear();
        }
        Ok((answer, None))
    }
    fn device_tier(
        &mut self,
        frame: &Frame,
        fds: Vec<OwnedFd>,
    ) -> io::Result<(Answer, Vec<OwnedFd>)> {
        let mut answer = Answer::unavailable(frame.seq);
        let (birth, exit) = self
            .reader
            .as_ref()
            .ok_or_else(|| io::Error::other("no reader"))?;
        let key = HoldingKey {
            device: frame.device.clone(),
            layout: frame.layout.clone(),
        };
        let (reader, lease) = Reader::lease(birth.clone(), exit.try_clone()?)?;
        answer.ok = true;
        answer.code.clear();
        answer.detail.clear();
        answer.layout = key.layout.clone();
        let mut custody = self.custody.lock().unwrap();
        let count = fds.len();
        let (out, fds) = if frame.offer {
            match custody.offer(key, &frame.name, frame.regions.clone(), fds, reader) {
                Ok(Offered::Kept { generation }) => {
                    answer.generation = generation;
                    answer.lease = true;
                    ("offer", vec![lease])
                }
                Ok(Offered::Duplicate) => {
                    answer.duplicate = true;
                    ("duplicate", Vec::new())
                }
                Err(error) => {
                    answer.ok = false;
                    answer.detail = error.to_string();
                    ("refused", Vec::new())
                }
            }
        } else {
            drop(fds);
            match custody.attach(&key, reader)? {
                Some(mut a) => {
                    answer.held = true;
                    answer.lease = true;
                    answer.generation = a.generation;
                    answer.regions = a.regions;
                    a.fds.push(lease);
                    ("attach", a.fds)
                }
                None => ("miss", Vec::new()),
            }
        };
        let row = json!({"device_tier": out, "name": frame.name, "fds_in": count, "fds_out": fds.len(), "duplicate": answer.duplicate});
        writeln!(self.log, "{row}")?;
        Ok((answer, fds))
    }
}

#[derive(Serialize, Clone)]
struct Image {
    payload: usize,
    invoke_ms: f64,
    first_image_ms: f64,
    digests: Vec<String>,
    h2d_bytes: Option<u64>,
    plane: Option<cozy_machine::device_executor::PlaneFacts>,
}

struct Exec {
    executor: DeviceExecutor,
    pilot: Pilot,
    record: Value,
}

fn used_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(0)
}

fn fail(what: &str, frame: &Frame) -> io::Error {
    io::Error::other(format!("{what}: {} {}", frame.code, frame.detail))
}

/// Spawn, start, load, activate and run `payloads`; then `share` when `device_weights`.
fn exec(
    config: &Config,
    custody: &Arc<Mutex<ResidentCustody>>,
    name: &str,
    device_weights: bool,
    payloads: &[(usize, &Value)],
) -> io::Result<Exec> {
    let root = config.root.join(name);
    fs::create_dir_all(&root)?;
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
    let began = Instant::now();
    let executor = DeviceExecutor::spawn(ExecutorConfig {
        python: config.python.clone(),
        root: root.clone(),
        socket: root.join("executor.sock"),
        environment: config.environment.clone(),
        seal,
        generation_hold: None,
        identity: None,
        cgroup_namespace: None,
    })?;
    let spawn_ms = began.elapsed().as_secs_f64() * 1000.;
    let mut executor = executor;
    let exit = executor.observer_pidfd()?;
    let mut pilot = Pilot {
        custody: custody.clone(),
        reader: Some((executor.birth.clone(), exit)),
        budget: config.plane_budget_bytes,
        cells: Vec::new(),
        log: File::create(root.join("events.jsonl"))?,
    };
    let sharing = device_weights && executor.hello.offers("weights.attach/1");
    let devices = executor
        .hello
        .sealed
        .get("CUDA_VISIBLE_DEVICES")
        .cloned()
        .unwrap_or_default();
    let t = Instant::now();
    let started = executor.command(
        &DeviceCommand::Start {
            devices: devices.clone(),
            application: config.binding.application.clone(),
            package_interface: config.package_interface.clone(),
            sequence_parallel_degree: 1,
            import_only: false,
        },
        &mut pilot,
    )?;
    if !started.ok {
        return Err(fail("start", &started));
    }
    let start_ms = t.elapsed().as_secs_f64() * 1000.;
    let t = Instant::now();
    let loaded = executor.command(
        &DeviceCommand::Load {
            construction: "pilot".into(),
            devices,
            sequence_parallel_degree: 1,
            binding: Box::new(config.binding.clone()),
            budgets: Budgets {
                declared_weight_bytes: config.logical_weight_bytes,
            },
            models: Vec::new(),
            authorized_device_limit_bytes: Some(config.authorized_device_limit_bytes),
            attention_pin: String::new(),
            stages: false,
            sealed_tiers: false,
            model_sources: false,
            staged_tiers: false,
            pinned_bytes: Some(config.pinned_budget_bytes),
            device_weights: sharing,
            cap_bytes: None,
        },
        &mut pilot,
    )?;
    if !loaded.ok {
        return Err(fail("load", &loaded));
    }
    let load_ms = t.elapsed().as_secs_f64() * 1000.;
    for command in [
        DeviceCommand::Budget {
            vram_bytes: config.plane_budget_bytes,
            pinned_bytes: config.pinned_budget_bytes,
            cap_bytes: None,
        },
        DeviceCommand::Activate {
            construction: "pilot".into(),
        },
    ] {
        let reply = executor.command(&command, &mut pilot)?;
        if !reply.ok {
            return Err(fail("budget/activate", &reply));
        }
    }
    let mut images = Vec::new();
    for (index, payload) in payloads {
        let id = format!("{name}-{index}");
        let spool = root.join(&id);
        fs::create_dir(&spool)?;
        let prepared = executor.command(
            &DeviceCommand::PrepareRequest {
                request_id: id.clone(),
                construction: "pilot".into(),
                entrypoint: config.entrypoint.clone(),
                payload: (*payload).clone(),
                attention_kernel: String::new(),
                input_metadata: Default::default(),
            },
            &mut pilot,
        )?;
        if !prepared.ok {
            return Err(fail("prepare", &prepared));
        }
        let t = Instant::now();
        let reply = executor.command(
            &DeviceCommand::Invoke {
                request_id: id.clone(),
                construction: "pilot".into(),
                entrypoint: config.entrypoint.clone(),
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: config.plane_budget_bytes,
                stages: false,
                cap_bytes: None,
                inputs: Default::default(),
                trees: Default::default(),
                floor_bytes: None,
                activation_bytes: Default::default(),
                device_weights: None,
            },
            &mut pilot,
        )?;
        let invoke_ms = t.elapsed().as_secs_f64() * 1000.;
        let outcome = reply
            .outcome
            .as_ref()
            .map(|o| o.terminal.as_str())
            .unwrap_or("");
        if outcome != "succeeded" {
            return Err(io::Error::other(format!(
                "invoke {id}: {:?}",
                reply.outcome
            )));
        }
        let (_, bindings) = postprocess(&executor.codec(), &spool, &reply)?;
        images.push(Image {
            payload: *index,
            invoke_ms,
            first_image_ms: began.elapsed().as_secs_f64() * 1000.,
            digests: bindings.iter().map(|b| b.sha256.clone()).collect(),
            h2d_bytes: reply.plane.as_ref().and_then(|p| p.h2d_bytes),
            plane: reply.plane.clone(),
        });
    }
    let mut shared_bytes = None;
    if sharing {
        let t = Instant::now();
        let reply = executor.command(&DeviceCommand::Share, &mut pilot)?;
        if !reply.ok {
            return Err(fail("share", &reply));
        }
        shared_bytes = Some((reply.shared_bytes, t.elapsed().as_secs_f64() * 1000.));
    }
    let record = json!({
        "name": name, "pid": executor.birth.pid, "device_weights": sharing,
        "spawn_ms": spawn_ms, "start_ms": start_ms, "load_ms": load_ms,
        "load_facts": loaded.facts, "images": images, "share": shared_bytes,
        "runtime": executor.hello.runtime_version, "tensorfs": executor.hello.tensorfs_version,
    });
    println!("{record}");
    Ok(Exec {
        executor,
        pilot,
        record,
    })
}

fn kill(exec: Exec) -> io::Result<()> {
    let pidfd = exec.executor.observer_pidfd()?;
    // SAFETY: signalling the exact live child we spawned, through its pidfd.
    if unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            std::os::fd::AsRawFd::as_raw_fd(&pidfd),
            libc::SIGKILL,
            0,
            0,
        )
    } < 0
    {
        return Err(io::Error::last_os_error());
    }
    let mut poll = libc::pollfd {
        fd: std::os::fd::AsRawFd::as_raw_fd(&pidfd),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live pidfd; waits for the exact process's exit, no elapsed-time policy.
    unsafe { libc::poll(&mut poll, 1, -1) };
    drop(exec);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("degree2-pilot: {error}");
        std::process::exit(1);
    }
}

fn run() -> io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: degree2-pilot CONFIG.json"))?;
    let config: Config = serde_json::from_slice(&fs::read(path)?)?;
    if config.payloads.len() < 6 {
        return Err(io::Error::other(
            "six unique payloads: one per timed executor",
        ));
    }
    fs::create_dir_all(&config.root)?;
    let custody = Arc::new(Mutex::new(ResidentCustody::default()));
    let p = |i: usize| (i, &config.payloads[i]);
    let mut out = BTreeMap::<String, Value>::new();
    let mut nvml = Vec::new();
    let mut sample = |label: &str| {
        let row = json!({"at": label, "used_mib": used_mib(), "held": custody.lock().unwrap().holdings().iter().map(|h| json!({"name": h.name, "bytes": h.bytes, "readers": h.readers.len(), "phase": h.phase})).collect::<Vec<_>>()});
        println!("{row}");
        nvml.push(row);
    };
    sample("baseline");
    // A fills privately (first executor: pays kernel compiles), then shares.
    let a = exec(&config, &custody, "a", true, &[p(0)])?;
    sample("a_shared");
    // B attaches while A lives: the weights are counted once.
    let b = exec(&config, &custody, "b", true, &[p(1)])?;
    sample("a_b_live");
    // Executor death mid-idle; the machine keeps the weights.
    kill(a)?;
    kill(b)?;
    custody.lock().unwrap().collect();
    sample("a_b_killed");
    // Replacement (attached) and private-fill executors, alternating.
    for (name, attach, payload) in [
        ("c", true, 2),
        ("d", false, 3),
        ("e", true, 4),
        ("f", false, 5),
    ] {
        let e = exec(&config, &custody, name, attach, &[p(payload)])?;
        out.insert(name.into(), e.record.clone());
        sample(&format!("{name}_live"));
        kill(e)?;
    }
    // Same request, attached vs private: identical outputs.
    let same = [(usize::MAX, &config.same)];
    let g = exec(&config, &custody, "g", true, &same)?;
    let h = exec(&config, &custody, "h", false, &same)?;
    let digests = |r: &Value| r["images"][0]["digests"].clone();
    out.insert(
        "same_output_identical".into(),
        json!(digests(&g.record) == digests(&h.record)),
    );
    kill(h)?;
    sample("g_live");
    // Revoke: G releases at its idle boundary, then custody closes the fds.
    let mut g = g;
    let held: Vec<_> = custody.lock().unwrap().holdings();
    let t = Instant::now();
    for holding in &held {
        custody
            .lock()
            .unwrap()
            .begin_revoke(&holding.key, holding.generation)?;
        let reply = g.executor.command(
            &DeviceCommand::Revoke {
                layout: holding.key.layout.clone(),
                generation: holding.generation,
            },
            &mut Baseline,
        )?;
        if reply.ok {
            custody
                .lock()
                .unwrap()
                .released(&holding.key, holding.generation, &g.executor.birth);
        }
        out.insert(
            format!("revoke_{}", holding.name),
            json!({"ok": reply.ok, "released_bytes": reply.released_bytes, "code": reply.code}),
        );
    }
    let released = custody.lock().unwrap().collect();
    out.insert("revoke_ms".into(), json!(t.elapsed().as_secs_f64() * 1000.));
    out.insert(
        "released".into(),
        json!(released
            .iter()
            .map(|(k, n)| json!({"layout": k.layout, "bytes": n}))
            .collect::<Vec<_>>()),
    );
    sample("revoked_g_live");
    g.pilot.reader = None;
    kill(g)?;
    sample("all_gone");
    let report = json!({"qualification": "development pilot through ResidentCustody and the stock executor seam; not CLI proof", "runs": out, "nvml": nvml});
    fs::write(
        config.root.join("degree2.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
