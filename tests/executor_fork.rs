//! Executors forked from an import-only executor (`fork/1`) on real Runtime processes of
//! `tests/fixtures/cpu_lifecycle`, no doubles. CUDA is hidden by the seal; the GPU half is
//! proven on a rental. `COZY_MACHINE_FORK_GENERATIONS` names installed generations whose
//! Runtime offers `fork/1`.
use cozy_machine::{
    device_executor::{
        Baseline, Binding, Budgets, DeviceCommand, DeviceExecutor, ExecutorConfig, Forked, FORK,
    },
    launch_identity::Seal,
    process::{process_ended, Exact},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct Generation {
    identity: String,
    python: PathBuf,
    application: String,
    interface: Value,
}

fn generations() -> Vec<(Generation, File)> {
    let root = PathBuf::from(
        std::env::var("COZY_MACHINE_FORK_GENERATIONS").expect("installed fork/1 generations"),
    );
    let mut found = vec![];
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        let hold = File::open(path.join(".hold")).unwrap();
        fs2::FileExt::lock_shared(&hold).unwrap();
        let generation = serde_json::from_slice(&fs::read(path.join("generation.json")).unwrap());
        found.push((generation.unwrap(), hold));
    }
    assert!(!found.is_empty());
    found
}

fn config(generation: &Generation, hold: &File, root: &Path, name: &str) -> ExecutorConfig {
    let mut seal = Seal::prepare(root, None, "fork", &generation.identity, "").unwrap();
    seal.threads = 1;
    ExecutorConfig {
        python: generation.python.clone(),
        root: root.join(name),
        socket: root.join(format!("{name}.sock")),
        environment: BTreeMap::new(),
        seal,
        generation_hold: Some(Arc::new(hold.try_clone().unwrap())),
        identity: None,
        cgroup_namespace: Some("fork".into()),
    }
}

fn start(generation: &Generation, interface: &Path, import_only: bool) -> DeviceCommand {
    DeviceCommand::Start {
        devices: String::new(),
        application: generation.application.clone(),
        package_interface: interface.to_path_buf(),
        sequence_parallel_degree: 1,
        import_only,
    }
}

fn parent(generation: &Generation, hold: &File) -> (DeviceExecutor, PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("machine-fork-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let interface = root.join("package-interface.json");
    fs::write(
        &interface,
        serde_json::to_vec(&generation.interface).unwrap(),
    )
    .unwrap();
    let mut parent = DeviceExecutor::spawn(config(generation, hold, &root, "parent")).unwrap();
    assert!(parent.hello.offers(FORK));
    let started = parent
        .command(&start(generation, &interface, true), &mut Baseline)
        .unwrap();
    assert!(started.ok, "{started:?}");
    (parent, root, interface)
}

fn fork(parent: &mut DeviceExecutor, config: ExecutorConfig) -> DeviceExecutor {
    match parent.fork(config, |_, _| Ok(())).unwrap() {
        Forked::Ready(executor) => *executor,
        Forked::Refused(_, reason) | Forked::Lost(_, reason) => panic!("fork refused: {reason}"),
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations offering fork/1"]
fn a_forked_executor_is_sealed_serves_and_is_reaped_with_its_status() {
    for (generation, hold) in generations() {
        let (mut parent, root, interface) = parent(&generation, &hold);
        let began = Instant::now();
        let mut child = fork(&mut parent, config(&generation, &hold, &root, "child"));
        let ready = began.elapsed();
        assert!(
            ready < Duration::from_secs(5),
            "forked executor ready after {ready:?}"
        );
        assert_eq!(child.hello.ppid, parent.birth.pid);
        assert_eq!(child.hello.pgid, child.birth.pid);
        assert_eq!(child.hello.sealed["CUDA_VISIBLE_DEVICES"], "");
        let status = fs::read_to_string(format!("/proc/{}/status", child.birth.pid)).unwrap();
        assert!(status.contains("NoNewPrivs:\t1"));
        for command in [
            start(&generation, &interface, false),
            DeviceCommand::Load {
                construction: "lifecycle".into(),
                devices: String::new(),
                sequence_parallel_degree: 1,
                binding: Box::new(Binding {
                    application: generation.application.clone(),
                    package_interface: interface.display().to_string(),
                    ..Binding::default()
                }),
                budgets: Budgets::default(),
                models: Vec::new(),
                authorized_device_limit_bytes: None,
                attention_pin: String::new(),
                stages: false,
                device_weights: false,
                cap_bytes: None,
                sealed_tiers: false,
                pinned_bytes: None,
            },
            DeviceCommand::Activate {
                construction: "lifecycle".into(),
            },
        ] {
            let reply = child.command(&command, &mut Baseline).unwrap();
            assert!(reply.ok, "{reply:?}");
        }
        let spool = root.join("spool");
        fs::create_dir(&spool).unwrap();
        let prepared = child
            .command(
                &DeviceCommand::PrepareRequest {
                    request_id: "forked".into(),
                    construction: "lifecycle".into(),
                    entrypoint: "steps".into(),
                    payload: json!({"steps": 2, "seconds": 0.01}),
                    attention_kernel: String::new(),
                    input_metadata: Default::default(),
                },
                &mut Baseline,
            )
            .unwrap();
        assert!(prepared.ok, "{prepared:?}");
        let reply = child
            .command(
                &DeviceCommand::Invoke {
                    request_id: "forked".into(),
                    construction: "lifecycle".into(),
                    entrypoint: "steps".into(),
                    spool,
                    deadline_s: None,
                    attention_kernel: String::new(),
                    plane_budget_bytes: -1,
                    stages: false,
                    cap_bytes: None,
                    inputs: Default::default(),
                    floor_bytes: None,
                    activation_bytes: Default::default(),
                },
                &mut Baseline,
            )
            .unwrap();
        assert_eq!(reply.outcome.unwrap().terminal, "succeeded");

        // An outside SIGKILL: the machine reads the status from its parent's zombie.
        let birth = child.birth.clone();
        Exact::open(&birth).unwrap().unwrap().kill().unwrap();
        let ended = child.terminate().unwrap();
        assert_eq!(ended.status.signal(), Some(libc::SIGKILL));

        // A second child; the parent reaps the first at this fork.
        let second = fork(&mut parent, config(&generation, &hold, &root, "second"));
        assert!(process_ended(&birth).unwrap());
        assert!(!Path::new(&format!("/proc/{}", birth.pid)).exists());
        second.shutdown().unwrap();
        parent.shutdown().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations offering fork/1"]
fn a_parent_refuses_before_its_imports_and_its_children_die_with_it() {
    for (generation, hold) in generations() {
        let root = std::env::temp_dir().join(format!("machine-fork-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&root).unwrap();
        let mut early = DeviceExecutor::spawn(config(&generation, &hold, &root, "early")).unwrap();
        let config_back = match early
            .fork(config(&generation, &hold, &root, "refused"), |_, _| Ok(()))
            .unwrap()
        {
            Forked::Refused(config, reason) => {
                assert!(reason.contains("fork_unready"), "{reason}");
                config
            }
            Forked::Ready(_) | Forked::Lost(..) => panic!("forked before its imports"),
        };
        // Refused means no process: the same configuration spawns.
        DeviceExecutor::spawn(*config_back)
            .unwrap()
            .shutdown()
            .unwrap();
        early.shutdown().unwrap();
        fs::remove_dir_all(&root).unwrap();

        let (mut parent, root, _) = parent(&generation, &hold);
        let child = fork(&mut parent, config(&generation, &hold, &root, "orphan"));
        let birth = child.birth.clone();
        let exact = Exact::open(&birth).unwrap().unwrap();
        Exact::open(&parent.birth).unwrap().unwrap().kill().unwrap();
        exact.wait().unwrap(); // parent death reaches the child (PDEATHSIG)
        drop(child);
        drop(parent);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations offering fork/1"]
fn a_dead_parent_is_lost_and_its_configuration_spawns() {
    for (generation, hold) in generations() {
        let (mut parent, root, _) = parent(&generation, &hold);
        // Killed (as the gate kills every executor): its channel fails, no process is made.
        Exact::open(&parent.birth).unwrap().unwrap().kill().unwrap();
        Exact::open(&parent.birth).unwrap().unwrap().wait().unwrap();
        let config = match parent
            .fork(config(&generation, &hold, &root, "after"), |_, _| Ok(()))
            .unwrap()
        {
            Forked::Lost(config, _) => config,
            Forked::Refused(_, reason) => panic!("a dead parent refused: {reason}"),
            Forked::Ready(_) => panic!("a dead parent forked"),
        };
        DeviceExecutor::spawn(*config).unwrap().shutdown().unwrap();
        drop(parent);
        fs::remove_dir_all(root).unwrap();
    }
}
