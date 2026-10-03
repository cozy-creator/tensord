//! Real installed SDK trampoline CPU proof; no executor/model/GPU imports.
use cozy_machine::launch_identity::{trampoline, LaunchIdentity};
use serde_json::Value;
use std::{fs, path::PathBuf};

#[test]
#[ignore = "requires an explicitly configured installed SDK interpreter; no GPU imports"]
fn optional_same_uid_sdk_launch_seals_process_and_preserves_default_identity() {
    let python = PathBuf::from(
        std::env::var("COZY_MACHINE_CPU_TEST_PYTHON")
            .expect("installed SDK interpreter configuration"),
    );
    let identity = LaunchIdentity {
        uid: unsafe { libc::geteuid() },
        gid: unsafe { libc::getegid() },
    };
    for selected in [None, Some(identity)] {
        let output = trampoline(&python, selected, None).unwrap()
            .args(["-I","-c",r#"import os,json,ctypes,pathlib
status=pathlib.Path('/proc/self/status').read_text()
libc=ctypes.CDLL(None);death=ctypes.c_int();assert libc.prctl(2,ctypes.byref(death),0,0,0)==0
maps=pathlib.Path('/proc/self/maps').read_text()
print(json.dumps({'uid':os.geteuid(),'gid':os.getegid(),'pid':os.getpid(),'parent':os.getppid(),'pgid':os.getpgrp(),'no_new_privs':next(r for r in status.splitlines() if r.startswith('NoNewPrivs:')).split(':')[1].strip(),'oom':pathlib.Path('/proc/self/oom_score_adj').read_text().strip(),'death':death.value,'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'gpu':any(s in maps for s in ('libcuda.so','libnvidia-ml.so'))}))"#])
            .env_clear().env("PATH","/usr/bin:/bin").output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["uid"], identity.uid);
        assert_eq!(value["gid"], identity.gid);
        assert_eq!(value["parent"], std::process::id());
        assert_eq!(value["no_new_privs"], "1");
        assert_eq!(value["oom"], "1000");
        assert_eq!(value["death"], 9);
        assert_eq!(value["gpu"], false);
        assert_eq!(
            value["cgroup"],
            fs::read_to_string("/proc/self/cgroup").unwrap()
        );
        // Every launch leads its own group: kills reach its descendants, and a terminal
        // signal to the machine's group never reaches it directly.
        assert_eq!(value["pgid"], value["pid"]);
    }
}

#[test]
#[ignore = "installed SDK CPU fault probe; proves Linux creator-thread lifetime"]
fn retained_child_is_killed_when_the_temporary_creator_thread_exits() {
    use std::{
        io::{BufRead, BufReader},
        os::{fd::FromRawFd, unix::process::ExitStatusExt},
    };
    let python = PathBuf::from(std::env::var("COZY_MACHINE_CPU_TEST_PYTHON").unwrap());
    let mut child = std::thread::spawn(move || {
        let mut child = trampoline(&python, None, None)
            .unwrap()
            .args([
                "-I",
                "-c",
                "import signal; print('armed',flush=True); signal.pause()",
            ])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        assert_eq!(ready, "armed\n");
        child // moved to live parent process; only creating THREAD exits
    })
    .join()
    .unwrap();
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, child.id(), 0) } as i32;
    assert!(raw >= 0);
    let _fd = unsafe { std::fs::File::from_raw_fd(raw) };
    let mut p = libc::pollfd {
        fd: raw,
        events: libc::POLLIN,
        revents: 0,
    };
    let observed = unsafe { libc::poll(&mut p, 1, 5000) };
    if observed <= 0 {
        // Explicit teardown of this owned CPU fault probe; never a product timeout policy.
        child.kill().unwrap();
        child.wait().unwrap();
        panic!("creator-thread exit did not signal observed fixture child");
    }
    assert_eq!(child.wait().unwrap().signal(), Some(9));
}

#[test]
#[ignore = "installed SDK CPU positive proof for retained-child spawn ownership"]
fn pool_launcher_keeps_retained_child_usable_after_requesting_thread_exits() {
    use std::{
        io::{BufRead, BufReader, Write},
        sync::Arc,
    };
    let python = PathBuf::from(std::env::var("COZY_MACHINE_CPU_TEST_PYTHON").unwrap());
    let launcher = Arc::new(cozy_machine::child_launcher::ChildLauncher::new().unwrap());
    let owner = launcher.clone();
    let (mut child,mut stdout)=std::thread::spawn(move || {
        let mut command=trampoline(&python, None, None).unwrap();
        command.args(["-I","-c","import sys;print('armed',flush=True)\nfor line in sys.stdin: print(line.strip(),flush=True)"])
            .env_clear().env("PATH","/usr/bin:/bin").stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped());
        let mut child=owner.spawn(command).unwrap();
        let mut stdout=BufReader::new(child.stdout.take().unwrap());
        let mut line=String::new();stdout.read_line(&mut line).unwrap();assert_eq!(line,"armed\n");
        (child,stdout)
    }).join().unwrap();
    assert!(child.try_wait().unwrap().is_none());
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"usable\n")
        .unwrap();
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    assert_eq!(line, "usable\n");
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    drop(launcher); // explicit finish preceded launcher-owner lifetime end
}
