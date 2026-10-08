//! CPU birth/journal proof only: this does not qualify a GPU or ordinary CLI inference.
use cozy_machine::{
    execution::process_birth,
    journal::{
        Installation, Invocation, Journal, Outcome, Preparation, ProcessBirth, ResultRecord,
        SubmissionContext,
    },
    service::Service,
};
use serde_json::json;
use std::{
    fs,
    process::{Command, Stdio},
};

#[test]
fn restart_ends_a_retained_gpu_birth_before_admitting_gpu_work() {
    let root = std::env::temp_dir().join(format!("machine-gpu-fence-{}", uuid::Uuid::new_v4()));
    let mut child = Command::new("/usr/bin/python3")
        .args(["-I", "-c", "import sys; sys.stdin.buffer.read()"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let birth = process_birth(child.id()).unwrap();
    let state = root.join("state");
    let mut journal = Journal::open(&state.join("execution")).unwrap();
    let generation = "a".repeat(32);
    journal
        .bind_installation(Installation {
            actor: "actor".into(),
            alias: "published".into(),
            generation: generation.clone(),
            package: "fixture".into(),
            release: "1".into(),
            interface: b"{}".to_vec(),
        })
        .unwrap();
    journal
        .bind_preparation(Preparation {
            actor: "actor".into(),
            id: "gpu-plan".into(),
            installation: "published".into(),
            document: b"{}".to_vec(),
        })
        .unwrap();
    // Cross a journal page boundary and exercise repeated requests on one retained process.
    for n in 0..258 {
        complete(&mut journal, n, &generation, birth.clone(), "gpu-plan");
    }
    let first = journal.gpu_births_after(0, 256).unwrap();
    assert_eq!(first.len(), 256);
    assert_eq!(
        journal
            .gpu_births_after(first.last().unwrap().0, 256)
            .unwrap()
            .len(),
        2
    );
    // Same PID with a different birth is already ended, and an ordinary CPU process
    // is not a GPU reservation merely because it is still alive.
    let mut obsolete = birth.clone();
    obsolete.start_ticks += 1;
    complete(&mut journal, 258, &generation, obsolete, "gpu-plan");
    complete(&mut journal, 259, &generation, birth.clone(), "");
    drop(journal);
    let service = Service::open(&state, &root.join("generations"), 1).unwrap();
    assert!(service.idle().unwrap()); // terminal results never authorize freeing a context
                                      // The prior machine's retained executor is killed, never adopted or waited on forever.
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
    assert_eq!(service.gpu_startup_fences(), 0);
    assert!(service.stop().unwrap());
    // Only this exited test's data is removed, never a task branch or worktree.
    fs::remove_dir_all(root).unwrap();
}

fn complete(journal: &mut Journal, n: usize, generation: &str, birth: ProcessBirth, preparation: &str) {
    complete_device(journal, n, generation, birth, preparation, false);
}

fn complete_device(journal: &mut Journal, n: usize, generation: &str, birth: ProcessBirth, preparation: &str, accelerator: bool) {
    let context = SubmissionContext {
        actor: "actor".into(),
        request_id: format!("request-{n}"),
        submission_id: format!("submission-{n}"),
        expected_workspace_id: journal.workspace_id().into(),
        preparation_id: preparation.into(),
        ..Default::default()
    };
    let record = journal
        .accept_public(
            context,
            Invocation {
                attention_kernel: String::new(),
                inputs: vec![],
                package: "fixture".into(),
                generation: generation.into(),
                module: "fixture:app".into(),
                entrypoint: "infer".into(),
                job: accelerator,
                accelerator,
                input: json!({"n":n}),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(journal.claim(&record.id).unwrap());
    journal.register_process(&record.id, birth).unwrap();
    journal.running(&record.id, None).unwrap();
    journal
        .finish(
            &record.id,
            Outcome::Completed(ResultRecord {
                value: json!(n),
                artifacts: vec![],
                asset_bindings: vec![],
            }),
        )
        .unwrap();
}

#[test]
fn restart_ends_processes_left_in_earlier_executor_scopes() {
    use cozy_machine::scope::{namespace, Scope};
    use std::io::{Read, Write};
    let root = std::env::temp_dir().join(format!("machine-scope-sweep-{}", uuid::Uuid::new_v4()));
    let state = root.join("state");
    fs::create_dir_all(&state).unwrap();
    let scope = Scope::create(&namespace(&state)).unwrap();
    // A daemon an earlier executor left behind, in its own session inside its scope.
    let mut command = Command::new("/bin/sh");
    command
        .args(["-c", "read go; setsid sleep 1000 </dev/null >/dev/null 2>&1 & echo $!"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    if let Some((name, value)) = scope.environment() {
        command.env(name, value);
    }
    let mut leader = command.spawn().unwrap();
    scope.adopt(leader.id()).unwrap();
    leader.stdin.take().unwrap().write_all(b"go\n").unwrap();
    let mut daemon = String::new();
    leader.stdout.take().unwrap().read_to_string(&mut daemon).unwrap();
    assert!(leader.wait().unwrap().success());
    let daemon = cozy_machine::process::process_birth(daemon.trim().parse().unwrap()).unwrap();
    assert_eq!(scope.processes().unwrap(), 1);
    // That machine is gone: nothing in this process claims the scope any more.
    let cgroup = scope.cgroup_relative().map(|relative| format!("/sys/fs/cgroup{relative}"));
    drop(scope);
    let service = Service::open(&state, &root.join("generations"), 1).unwrap();
    assert!(cozy_machine::process::process_ended(&daemon).unwrap());
    if let Some(path) = cgroup {
        assert!(!std::path::Path::new(&path).exists());
    }
    assert!(service.stop().unwrap());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn restart_fences_a_terminal_inline_gpu_job_without_a_model_preparation() {
    let root = std::env::temp_dir().join(format!("machine-job-gpu-fence-{}", uuid::Uuid::new_v4()));
    let mut child = Command::new("/usr/bin/python3")
        .args(["-I", "-c", "import sys; sys.stdin.buffer.read()"])
        .stdin(Stdio::piped()).spawn().unwrap();
    let birth = process_birth(child.id()).unwrap();
    let state = root.join("state");
    let mut journal = Journal::open(&state.join("execution")).unwrap();
    complete_device(&mut journal, 1, &"a".repeat(32), birth.clone(), "", true);
    complete_device(&mut journal, 2, &"a".repeat(32), birth, "", false);
    assert_eq!(journal.gpu_births_after(0, 256).unwrap().len(), 1);
    drop(journal);
    let service = Service::open(&state, &root.join("generations"), 1).unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
    assert_eq!(service.gpu_startup_fences(), 0);
    assert!(service.stop().unwrap());
    fs::remove_dir_all(root).unwrap();
}
