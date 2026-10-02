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
        let output = trampoline(&python,selected).unwrap()
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
        if selected.is_some() {
            assert_eq!(value["pgid"], value["pid"]);
        } else {
            assert_eq!(value["pgid"], unsafe { libc::getpgrp() });
        }
    }
}
