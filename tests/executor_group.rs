//! A degree-2 start on a real Runtime executor (`tests/fixtures/cpu_group`), no doubles: rank 0
//! spawns a real follower under the group seal, the follower dials back, the start is refused
//! (no torch is installed, so no device is touched) in the executor's own words, and teardown
//! leaves no member of the group. `COZY_MACHINE_GROUP_GENERATIONS` names the installed
//! generations directory.
use cozy_machine::{
    device_executor::{Baseline, DeviceCommand, DeviceExecutor, ExecutorConfig},
    journal::ProcessBirth,
    launch_identity::Seal,
    process::{group_members, process_ended, Liveness},
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

#[derive(Deserialize)]
struct Generation {
    identity: String,
    python: PathBuf,
    application: String,
    interface: Value,
}

fn generation() -> (Generation, File) {
    let root = PathBuf::from(
        std::env::var("COZY_MACHINE_GROUP_GENERATIONS").expect("installed cpu_group generation"),
    );
    let path = fs::read_dir(root).unwrap().next().unwrap().unwrap().path();
    let hold = File::open(path.join(".hold")).unwrap();
    fs2::FileExt::lock_shared(&hold).unwrap();
    let generation = serde_json::from_slice(&fs::read(path.join("generation.json")).unwrap());
    (generation.unwrap(), hold)
}

fn launch(devices: &str, group: bool) -> (DeviceExecutor, PathBuf, Generation) {
    let (generation, hold) = generation();
    let root = std::env::temp_dir().join(format!("machine-group-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let mut seal = Seal::prepare(&root, None, "group", &generation.identity, devices).unwrap();
    seal.threads = 1;
    seal.group = group;
    let mut executor = DeviceExecutor::spawn(ExecutorConfig {
        python: generation.python.clone(),
        root: root.join("executor"),
        socket: root.join("e.sock"),
        environment: BTreeMap::new(),
        seal,
        generation_hold: Some(Arc::new(hold)),
        identity: None,
    })
    .unwrap();
    executor.liveness = Liveness {
        sample: Duration::from_millis(100),
    };
    fs::write(
        root.join("package-interface.json"),
        serde_json::to_vec(&generation.interface).unwrap(),
    )
    .unwrap();
    (executor, root, generation)
}

fn start(
    degree: u32,
    devices: &str,
    root: &std::path::Path,
    generation: &Generation,
) -> DeviceCommand {
    DeviceCommand::Start {
        devices: devices.into(),
        application: generation.application.clone(),
        package_interface: root.join("package-interface.json"),
        sequence_parallel_degree: degree,
        import_only: false,
    }
}

fn environ(pid: u32) -> BTreeMap<String, String> {
    fs::read(format!("/proc/{pid}/environ"))
        .unwrap_or_default()
        .split(|b| *b == 0)
        .filter_map(|row| {
            let row = String::from_utf8_lossy(row);
            let (name, value) = row.split_once('=')?;
            Some((name.to_string(), value.to_string()))
        })
        .collect()
}

#[test]
fn a_group_start_spawns_sealed_followers_and_teardown_leaves_none() {
    let (mut executor, root, generation) = launch("0,1", true);
    let sealed = &executor.hello.sealed;
    assert_eq!(
        sealed.get("CUDA_VISIBLE_DEVICES").map(String::as_str),
        Some("0,1")
    );
    assert_eq!(
        sealed.get("NCCL_NVLS_ENABLE").map(String::as_str),
        Some("0")
    );
    assert_eq!(
        sealed.get("NCCL_P2P_LEVEL").map(String::as_str),
        Some("NVL")
    );
    // Watch the leader's group while it forms: every member, and the seal each one runs under.
    let birth = executor.birth.clone();
    let done = Arc::new(AtomicBool::new(false));
    type Seen = BTreeMap<u32, (ProcessBirth, BTreeMap<String, String>)>;
    let seen: Arc<Mutex<Seen>> = Arc::default();
    let watcher = {
        let (done, seen) = (done.clone(), seen.clone());
        std::thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                for member in group_members(&birth) {
                    let env = environ(member.pid);
                    if env.contains_key("CUDA_VISIBLE_DEVICES") {
                        seen.lock()
                            .unwrap()
                            .entry(member.pid)
                            .or_insert((member, env));
                    }
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        })
    };
    let reply = executor
        .command(&start(2, "0,1", &root, &generation), &mut Baseline)
        .unwrap();
    done.store(true, Ordering::Release);
    watcher.join().unwrap();
    // No torch here: the start is refused after the follower was spawned, in its own words.
    assert!(!reply.ok, "{reply:?}");
    assert!(!reply.code.is_empty(), "{reply:?}");
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty(), "rank 0 spawned no follower in its group");
    for (_, env) in seen.values() {
        assert_eq!(
            env.get("CUDA_VISIBLE_DEVICES").map(String::as_str),
            Some("0,1")
        );
        assert_eq!(env.get("NCCL_NVLS_ENABLE").map(String::as_str), Some("0"));
        assert_eq!(env.get("NCCL_P2P_LEVEL").map(String::as_str), Some("NVL"));
    }
    let leader = executor.birth.clone();
    let followers: Vec<ProcessBirth> = seen.values().map(|(birth, _)| birth.clone()).collect();
    executor.terminate().unwrap();
    assert!(
        group_members(&leader).is_empty(),
        "a member outlived teardown"
    );
    for follower in followers {
        assert!(
            process_ended(&follower).unwrap(),
            "{follower:?} outlived teardown"
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_degree_the_seal_does_not_name_is_refused_before_any_follower() {
    let (mut executor, root, generation) = launch("0,1", true);
    let reply = executor
        .command(&start(3, "0,1", &root, &generation), &mut Baseline)
        .unwrap();
    assert_eq!(reply.code, "sequence_parallel_degree_mismatch", "{reply:?}");
    assert!(group_members(&executor.birth).is_empty());
    executor.terminate().unwrap();
    fs::remove_dir_all(root).unwrap();
}
