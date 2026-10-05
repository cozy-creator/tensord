//! The host-pressure feedback rule on a real kernel (MEM/HOST-PRESSURE.md): a PSI trigger on a
//! memory-limited scope, "warm" processes holding memory, one ended per rung the rule gives.
//! Against a hog beside them, giving back helps and the hog finishes; against a stall in a
//! scope of its own (its own memory.high, so not ours to fix), giving stops after a rung or two
//! and the warm set stays.
use cozy_machine::host_pressure::{Feedback, HostMode, Pressure};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const MIB: usize = 1 << 20;

/// A process holding `mib` of touched anonymous memory until it is ended.
fn warm(mib: usize) -> Child {
    let mut child = python(&format!(
        "b = bytearray({}); b[::4096] = b'\\1' * len(b[::4096]); print('held', flush=True)\nimport time; time.sleep(600)",
        mib * MIB
    ));
    let mut line = String::new();
    BufReader::new(child.stdout.as_mut().unwrap()).read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "held");
    child
}

/// Allocates, touches and frees `mib` again and again (`passes`, or forever): above its
/// cgroup's memory.high with no swap, every allocation stalls.
fn churn(mib: usize, passes: Option<usize>) -> String {
    let count = passes.map_or("iter(int, 1)".to_string(), |n| format!("range({n})"));
    format!(
        "for _ in {count}:\n    b = bytearray({}); b[::4096] = b'\\1' * len(b[::4096]); del b",
        mib * MIB
    )
}

fn python(script: &str) -> Child {
    Command::new("python3")
        .args(["-c", script])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap()
}

fn cgroup_of(pid: u32) -> PathBuf {
    let text = fs::read_to_string(format!("/proc/{pid}/cgroup")).unwrap();
    let relative = text.lines().find_map(|l| l.strip_prefix("0::")).unwrap();
    PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/'))
}

fn scope(memory: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = ["--user", "--scope", "--quiet", "--collect", "-p", "MemorySwapMax=0"]
        .map(String::from)
        .to_vec();
    for p in memory {
        args.extend(["-p".into(), p.to_string()]);
    }
    args
}

fn scopes() -> bool {
    Command::new("systemd-run")
        .args(scope(&[]))
        .arg("true")
        .status()
        .is_ok_and(|s| s.success())
}

/// One rung per event the rule gives: the last warm process still held is ended. A thread that
/// runs until the test process ends; `given` counts (events, rungs).
fn give_back(pressure: Pressure, held: Arc<Mutex<Vec<Child>>>, given: Arc<Mutex<(usize, usize)>>) {
    std::thread::spawn(move || {
        let mut feedback = Feedback::new(pressure.stalled_us().unwrap(), Instant::now());
        while pressure.wait().is_ok() {
            let give = feedback.give(pressure.stalled_us().unwrap(), Instant::now());
            let mut counts = given.lock().unwrap();
            counts.0 += 1;
            if give {
                match held.lock().unwrap().pop() {
                    Some(mut child) => {
                        child.kill().unwrap();
                        child.wait().unwrap();
                        counts.1 += 1;
                    }
                    None => feedback.exhausted(),
                }
            }
        }
    });
}

#[test]
fn giving_back_helps_against_a_hog_beside_the_warm_set() {
    if !scopes() {
        eprintln!("this session cannot create a memory-limited scope");
        return;
    }
    let inner = Command::new("systemd-run")
        .args(scope(&["MemoryHigh=192M", "MemoryMax=1G"]))
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", "inside_a_scope_with_a_hog", "--ignored", "--nocapture"])
        .status()
        .unwrap();
    assert!(inner.success());
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_a_scope_with_a_hog() {
    let held = Arc::new(Mutex::new((0..3).map(|_| warm(48)).collect::<Vec<_>>()));
    let given = Arc::new(Mutex::new((0, 0)));
    // A rented pod's mode: this cgroup's full stalls and its memory.events.
    give_back(Pressure::arm(HostMode::Dedicated).unwrap(), held.clone(), given.clone());
    // 144 MiB held and 96 MiB churned in a 192 MiB scope: the hog stalls until room comes back.
    let mut hog = python(&churn(96, Some(60)));
    let deadline = Instant::now() + Duration::from_secs(300);
    while hog.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "the hog never got room: {:?}", given.lock().unwrap());
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(hog.wait().unwrap().success());
    let (events, rungs) = *given.lock().unwrap();
    assert!(rungs >= 1, "{events} events, {rungs} rungs");
    for mut child in held.lock().unwrap().drain(..) {
        child.kill().unwrap();
        child.wait().unwrap();
    }
}

#[test]
fn a_stall_in_a_scope_of_its_own_leaves_the_warm_set() {
    if !scopes() {
        eprintln!("this session cannot create a memory-limited scope");
        return;
    }
    // Someone else's stall: a churn above its own scope's memory.high, which no memory given
    // back here can relieve.
    let mut other = Command::new("systemd-run")
        .args(scope(&["MemoryHigh=64M", "MemoryMax=1G"]))
        .args(["python3", "-c", &churn(96, None)])
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(500));
    let theirs = cgroup_of(other.id());
    let held = Arc::new(Mutex::new((0..10).map(|_| warm(8)).collect::<Vec<_>>()));
    let given = Arc::new(Mutex::new((0, 0)));
    give_back(Pressure::arm_at(theirs.join("memory.pressure"), "some").unwrap(), held.clone(), given.clone());
    let deadline = Instant::now() + Duration::from_secs(120);
    while given.lock().unwrap().0 < 12 {
        assert!(Instant::now() < deadline, "too few stall events: {:?}", given.lock().unwrap());
        std::thread::sleep(Duration::from_millis(200));
    }
    other.kill().unwrap();
    other.wait().unwrap();
    let (events, rungs) = *given.lock().unwrap();
    // Giving back cannot lower this stall: a rung or two for the noise (simulated: 9 or more
    // in 12 events once in ten thousand), never the warm set.
    assert!(rungs <= 8, "{events} events shed {rungs} of 10 warm processes");
    for mut child in held.lock().unwrap().drain(..) {
        child.kill().unwrap();
        child.wait().unwrap();
    }
}
