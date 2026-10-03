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
        complete(&mut journal, n, &generation, birth.clone(), true);
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
    complete(&mut journal, 258, &generation, obsolete, true);
    complete(&mut journal, 259, &generation, birth.clone(), false);
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

fn complete(journal: &mut Journal, n: usize, generation: &str, birth: ProcessBirth, gpu: bool) {
    let context = SubmissionContext {
        actor: "actor".into(),
        request_id: format!("request-{n}"),
        submission_id: format!("submission-{n}"),
        expected_workspace_id: journal.workspace_id().into(),
        preparation_id: if gpu {
            "gpu-plan".into()
        } else {
            String::new()
        },
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
                input: json!({"n":n}),
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
