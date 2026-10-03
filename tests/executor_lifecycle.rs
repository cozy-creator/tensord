//! Executor lifecycle on real Runtime executors (the shipped SDK's generations of
//! `tests/fixtures/cpu_lifecycle`), no doubles. CUDA is hidden by the seal; no torch.
//! `COZY_MACHINE_LIFECYCLE_GENERATIONS` names the installed generations directory.
use cozy_machine::{
    device_executor::{
        Baseline, Binding, Budgets, DeviceCommand, DeviceExecutor, EndedBeforeStart,
        ExecutorConfig, Frame,
    },
    launch_identity::Seal,
    process::{process_ended, Exact, Liveness},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io,
    os::unix::process::ExitStatusExt,
    path::PathBuf,
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
        std::env::var("COZY_MACHINE_LIFECYCLE_GENERATIONS")
            .expect("installed cpu_lifecycle generations"),
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

fn launch(generation: &Generation, hold: &File) -> (DeviceExecutor, PathBuf, Seal) {
    let root = std::env::temp_dir().join(format!("machine-lifecycle-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let mut seal = Seal::prepare(&root, None, "lifecycle", &generation.identity, "").unwrap();
    seal.threads = 1;
    let mut executor = DeviceExecutor::spawn(ExecutorConfig {
        python: generation.python.clone(),
        root: root.join("executor"),
        socket: root.join("e.sock"),
        environment: BTreeMap::new(),
        seal: seal.clone(),
        generation_hold: Some(Arc::new(hold.try_clone().unwrap())),
        identity: None,
        cgroup_namespace: Some("lifecycle".into()),
    })
    .unwrap();
    // Shorter sampling keeps the measured-wedge tests quick; the rule is unchanged.
    executor.liveness = Liveness {
        sample: Duration::from_millis(100),
    };
    let interface = root.join("package-interface.json");
    fs::write(
        &interface,
        serde_json::to_vec(&generation.interface).unwrap(),
    )
    .unwrap();
    for command in [
        DeviceCommand::Start {
            devices: String::new(),
            application: generation.application.clone(),
            package_interface: interface.clone(),
            sequence_parallel_degree: 1,
            import_only: false,
        },
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
            model_sources: false,
            pinned_bytes: None,
        },
        DeviceCommand::Activate {
            construction: "lifecycle".into(),
        },
    ] {
        let reply = executor.command(&command, &mut Baseline).unwrap();
        assert!(reply.ok, "{reply:?}");
    }
    (executor, root, seal)
}

fn invoke(
    executor: &mut DeviceExecutor,
    root: &std::path::Path,
    id: &str,
    entrypoint: &str,
    payload: Value,
) -> io::Result<Frame> {
    let spool = root.join(id);
    fs::create_dir(&spool)?;
    let prepared = executor.command(
        &DeviceCommand::PrepareRequest {
            request_id: id.into(),
            construction: "lifecycle".into(),
            entrypoint: entrypoint.into(),
            payload,
            attention_kernel: String::new(),
            input_metadata: Default::default(),
        },
        &mut Baseline,
    )?;
    assert!(prepared.ok, "{prepared:?}");
    executor.command(
        &DeviceCommand::Invoke {
            request_id: id.into(),
            construction: "lifecycle".into(),
            entrypoint: entrypoint.into(),
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
}

fn terminal(reply: &Frame) -> &str {
    reply.outcome.as_ref().map_or("", |o| o.terminal.as_str())
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn launch_is_sealed_grouped_and_reaped_after_cooperative_shutdown() {
    for (generation, hold) in generations() {
        let (executor, root, seal) = launch(&generation, &hold);
        assert_eq!(executor.hello.ppid, std::process::id());
        assert_eq!(executor.hello.pgid, executor.birth.pid);
        for (name, value) in seal.imposed() {
            if let Some(reported) = executor.hello.sealed.get(&name) {
                assert_eq!(reported, &value, "{name}");
            }
        }
        assert_eq!(executor.hello.sealed["CUDA_VISIBLE_DEVICES"], "");
        assert!(executor.hello.sealed["COZY_HOME"]
            .ends_with(&format!("home/u{}", unsafe { libc::geteuid() })));
        let environ = fs::read(format!("/proc/{}/environ", executor.birth.pid)).unwrap();
        assert!(!String::from_utf8_lossy(&environ).contains("TOKEN"));
        let death = fs::read_to_string(format!("/proc/{}/status", executor.birth.pid)).unwrap();
        assert!(death.contains("NoNewPrivs:\t1"));
        let birth = executor.birth.clone();
        executor.shutdown().unwrap();
        assert!(process_ended(&birth).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn cooperative_cancel_stops_at_a_safe_point_and_keeps_the_executor() {
    for (generation, hold) in generations() {
        let (mut executor, root, _) = launch(&generation, &hold);
        let cancel = executor.cancellation();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300)); // let a few steps run
            cancel.cancel("long").unwrap();
        });
        let reply = invoke(
            &mut executor,
            &root,
            "long",
            "steps",
            json!({"steps": 100000, "seconds": 0.01}),
        )
        .unwrap();
        canceller.join().unwrap();
        assert_eq!(terminal(&reply), "canceled", "{reply:?}");
        let next = invoke(&mut executor, &root, "next", "steps", json!({"steps": 2})).unwrap();
        assert_eq!(terminal(&next), "succeeded", "{next:?}");
        executor.shutdown().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn slow_steady_work_is_never_killed_without_a_cancel() {
    let (generation, hold) = generations().remove(0);
    let (mut executor, root, _) = launch(&generation, &hold);
    // Each step is longer than the noise floor (0.6 s here); no cancel, so nothing judges.
    let reply = invoke(
        &mut executor,
        &root,
        "slow",
        "steps",
        json!({"steps": 2, "seconds": 1.0}),
    )
    .unwrap();
    assert_eq!(terminal(&reply), "succeeded", "{reply:?}");
    executor.shutdown().unwrap();
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn canceled_wedge_is_killed_on_measured_stillness_and_reaped() {
    for (generation, hold) in generations() {
        let (mut executor, root, _) = launch(&generation, &hold);
        let birth = executor.birth.clone();
        let cancel = executor.cancellation();
        let canceled_at = Arc::new(std::sync::Mutex::new(None));
        let mark = canceled_at.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            *mark.lock().unwrap() = Some(Instant::now());
            cancel.cancel("stuck").unwrap();
        });
        let error = invoke(&mut executor, &root, "stuck", "wedge", json!({})).unwrap_err();
        let returned = Instant::now();
        canceller.join().unwrap();
        let message = error.to_string();
        assert!(message.contains("wedged during invoke"), "{message}");
        assert!(message.contains("no measurable progress"), "{message}");
        // Patience is the 0.6 s floor here: the kill follows the cancel, never precedes it.
        let since_cancel = returned - canceled_at.lock().unwrap().unwrap();
        assert!(
            since_cancel >= Duration::from_millis(600),
            "{since_cancel:?}"
        );
        let ended = executor.terminate().unwrap();
        assert_eq!(ended.status.signal(), Some(libc::SIGKILL));
        assert!(process_ended(&birth).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn executor_killed_mid_request_returns_and_is_reaped() {
    for (generation, hold) in generations() {
        let (mut executor, root, _) = launch(&generation, &hold);
        let birth = executor.birth.clone();
        let exact = Exact::open(&birth).unwrap().unwrap();
        let killer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            exact.kill().unwrap(); // an outside SIGKILL (OOM killer, operator)
        });
        let error = invoke(
            &mut executor,
            &root,
            "killed",
            "steps",
            json!({"steps": 100000, "seconds": 0.01}),
        )
        .unwrap_err();
        killer.join().unwrap();
        assert!(error.to_string().contains("EOF"), "{error}");
        let ended = executor.terminate().unwrap();
        assert_eq!(ended.status.signal(), Some(libc::SIGKILL));
        assert!(ended.killed.is_none());
        assert!(process_ended(&birth).unwrap());

        // The package ends its own process mid-request: same contract.
        let (mut executor, root, _) = launch(&generation, &hold);
        let error = invoke(&mut executor, &root, "exit", "exit_now", json!({})).unwrap_err();
        assert!(error.to_string().contains("EOF"), "{error}");
        assert_eq!(executor.terminate().unwrap().status.code(), Some(17));
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn retained_executors_killed_while_idle_lose_their_channel_before_any_handler() {
    // The gate's kill scenario: SIGKILL the retained executor, the next request ~1 s later.
    // Each cycle's first request is the previous one's rerun, on a fresh executor.
    for (generation, hold) in generations() {
        let steps = json!({"steps": 1, "seconds": 0.01});
        for cycle in 0..20 {
            let (mut executor, root, _) = launch(&generation, &hold);
            let reply = invoke(&mut executor, &root, "first", "steps", steps.clone()).unwrap();
            assert_eq!(terminal(&reply), "succeeded", "cycle {cycle}: {reply:?}");
            Exact::open(&executor.birth).unwrap().unwrap().kill().unwrap();
            std::thread::sleep(Duration::from_secs(1));
            let error = invoke(&mut executor, &root, "next", "steps", steps.clone()).unwrap_err();
            assert!(cozy_machine::device_executor::channel_lost(&error), "cycle {cycle}: {error}");
            eprintln!("cycle {cycle}: {:?}: {error}", error.kind());
            assert_eq!(executor.terminate().unwrap().status.signal(), Some(libc::SIGKILL));
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[test]
#[ignore = "needs installed cpu_lifecycle generations"]
fn a_setsid_descendant_is_killed_with_its_executor_scope() {
    for (generation, hold) in generations() {
        let (mut executor, root, _) = launch(&generation, &hold);
        let member = fs::read_to_string(format!("/proc/{}/cgroup", executor.birth.pid)).unwrap();
        let environ = fs::read(format!("/proc/{}/environ", executor.birth.pid)).unwrap();
        let token = environ
            .split(|b| *b == 0)
            .any(|entry| entry.starts_with(b"COZY_EXECUTOR_SCOPE=lifecycle-"));
        // A delegated cgroup where the host has one, else the token its descendants inherit.
        assert!(member.contains("cozy-executor-lifecycle-") != token, "{member}");
        eprintln!("scope: {}", if token { "token" } else { "cgroup" });
        let reply = invoke(&mut executor, &root, "daemon", "daemon", json!({})).unwrap();
        assert_eq!(terminal(&reply), "succeeded", "{reply:?}");
        let spool = root.join("daemon");
        let pid = cozy_machine::device_executor::read_result(&spool, &reply).unwrap()["pid"]
            .as_u64()
            .unwrap() as u32;
        let daemon = cozy_machine::process::process_birth(pid).unwrap();
        // Its own session: a process-group kill would miss it.
        let session = |pid: u32| {
            fs::read_to_string(format!("/proc/{pid}/stat")).unwrap().rsplit_once(") ").unwrap().1
                .split_whitespace().nth(3).unwrap().to_string()
        };
        assert_ne!(session(pid), session(executor.birth.pid));
        let ended = executor.terminate().unwrap();
        assert_eq!(ended.stragglers, 1);
        assert!(process_ended(&daemon).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn pre_start_exit_is_a_typed_failure_with_its_stderr() {
    // An interpreter without cozy-runtime: the trampoline cannot even import.
    let root = std::env::temp_dir().join(format!("machine-prestart-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let seal = Seal::prepare(&root, None, "prestart", "none", "").unwrap();
    let error = DeviceExecutor::spawn(ExecutorConfig {
        python: PathBuf::from("/usr/bin/python3"),
        root: root.join("executor"),
        socket: root.join("e.sock"),
        environment: BTreeMap::new(),
        seal,
        generation_hold: None,
        identity: None,
        cgroup_namespace: Some("lifecycle".into()),
    })
    .err()
    .unwrap();
    let ended = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<EndedBeforeStart>())
        .expect("typed pre-start exit");
    assert!(!ended.status.success());
    assert!(
        ended.stderr_tail.contains("cozy_runtime"),
        "{}",
        ended.stderr_tail
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "root only: the SDK slot identity (uid 64000, gid 65533) in a rootful CPU container"]
fn foreign_identity_executor_is_contained_and_keeps_its_hold_and_caches() {
    use cozy_machine::launch_identity::LaunchIdentity;
    use std::os::unix::fs::{chown, PermissionsExt};
    assert_eq!(unsafe { libc::geteuid() }, 0, "run as root");
    let identity = LaunchIdentity {
        uid: 64000,
        gid: 65533,
    };
    let (generation, hold) = generations().remove(0);
    let hold_path = generation
        .python
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(".hold");
    let root = std::env::temp_dir().join(format!("machine-identity-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
    // The machine's private state (journal, store) is never readable by package code.
    let mut private = vec![];
    for name in ["journal", "store"] {
        let directory = root.join(name);
        fs::create_dir(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(directory.join("record"), b"machine only").unwrap();
        private.push(directory.join("record"));
    }
    let seal = Seal::prepare(
        &root.join("seal"),
        Some(identity),
        "identity",
        &generation.identity,
        "",
    )
    .unwrap();
    let mut executor = DeviceExecutor::spawn(ExecutorConfig {
        python: generation.python.clone(),
        root: root.join("executor"),
        socket: root.join("e.sock"),
        environment: BTreeMap::new(),
        seal,
        generation_hold: Some(Arc::new(hold.try_clone().unwrap())),
        identity: Some(identity),
        cgroup_namespace: Some("lifecycle".into()),
    })
    .unwrap();
    let status = fs::read_to_string(format!("/proc/{}/status", executor.birth.pid)).unwrap();
    assert!(
        status.contains("Uid:\t64000\t64000\t64000\t64000"),
        "{status}"
    );
    assert!(
        status.contains("Gid:\t65533\t65533\t65533\t65533"),
        "{status}"
    );
    assert!(status.contains("NoNewPrivs:\t1"));
    assert_eq!(executor.hello.pgid, executor.birth.pid);
    let interface = root.join("package-interface.json");
    fs::write(
        &interface,
        serde_json::to_vec(&generation.interface).unwrap(),
    )
    .unwrap();
    for command in [
        DeviceCommand::Start {
            devices: String::new(),
            application: generation.application.clone(),
            package_interface: interface.clone(),
            sequence_parallel_degree: 1,
            import_only: false,
        },
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
            model_sources: false,
            pinned_bytes: None,
        },
        DeviceCommand::Activate {
            construction: "lifecycle".into(),
        },
    ] {
        let reply = executor.command(&command, &mut Baseline).unwrap();
        assert!(reply.ok, "{reply:?}");
    }
    let spool = root.join("probe");
    fs::create_dir(&spool).unwrap();
    chown(&spool, Some(identity.uid), Some(identity.gid)).unwrap();
    fs::set_permissions(&spool, fs::Permissions::from_mode(0o700)).unwrap();
    let paths: Vec<String> = private
        .iter()
        .chain([&hold_path])
        .map(|p| p.display().to_string())
        .collect();
    let prepared = executor
        .command(
            &DeviceCommand::PrepareRequest {
                request_id: "probe".into(),
                construction: "lifecycle".into(),
                entrypoint: "probe".into(),
                payload: json!({ "paths": paths }),
                attention_kernel: String::new(),
                input_metadata: Default::default(),
            },
            &mut Baseline,
        )
        .unwrap();
    assert!(prepared.ok, "{prepared:?}");
    let reply = executor
        .command(
            &DeviceCommand::Invoke {
                request_id: "probe".into(),
                construction: "lifecycle".into(),
                entrypoint: "probe".into(),
                spool: spool.clone(),
                inputs: Default::default(),
                deadline_s: None,
                attention_kernel: String::new(),
                plane_budget_bytes: -1,
                stages: false,
                cap_bytes: None,
                floor_bytes: None,
                activation_bytes: Default::default(),
            },
            &mut Baseline,
        )
        .unwrap();
    assert_eq!(terminal(&reply), "succeeded", "{reply:?}");
    let reach = cozy_machine::device_executor::read_result(&spool, &reply).unwrap();
    println!("{}", serde_json::to_string(&reach).unwrap());
    assert_eq!(
        (reach["uid"].as_u64(), reach["gid"].as_u64()),
        (Some(64000), Some(65533))
    );
    for path in &private {
        assert_eq!(reach["reach"][path.display().to_string()], "denied");
    }
    assert_eq!(reach["home_writable"], true);
    let fds: Vec<String> = serde_json::from_value(reach["fds"].clone()).unwrap();
    assert!(
        fds.iter().any(|fd| fd == &hold_path.display().to_string()),
        "{fds:?}"
    );
    let birth = executor.birth.clone();
    executor.shutdown().unwrap();
    assert!(process_ended(&birth).unwrap());
    fs::remove_dir_all(root).unwrap();
}
