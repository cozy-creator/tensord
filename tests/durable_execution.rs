//! Real process/socket/filesystem checks, not full Runtime/CLI qualification.
#[path = "../src/execution.rs"]
mod execution;
#[path = "../src/journal.rs"]
mod journal;

use execution::{process_birth, Engine, RunnerConfig};
use journal::{Invocation, Journal, State};
use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    engine: std::sync::Arc<Engine>,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "cozy-machine-durable-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("runner_fixture.py"), RUNNER).unwrap();
        fs::write(root.join("cpu_package.py"), PACKAGE).unwrap();
        let engine = Engine::open(&root.join("state")).unwrap();
        Self { root, engine }
    }
    fn config(&self) -> RunnerConfig {
        RunnerConfig {
            python: "/usr/bin/python3".into(),
            module: "runner_fixture".into(),
            import_paths: vec![self.root.clone()],
            generation_hold: None,
        }
    }
    fn invocation(&self, mode: &str) -> Invocation {
        Invocation {
            package: "fixture-cpu".into(),
            generation: "installed-cpu-v1".into(),
            module: "cpu_package".into(),
            entrypoint: "infer".into(),
            input: json!({"mode":mode,"side_effect":self.root.join("effect"),"release":self.root.join("release"),"advance":self.root.join("advance")}),
        }
    }
    fn submit(&self, mode: &str) -> String {
        let record = self.engine.submit(mode, self.invocation(mode)).unwrap();
        assert!(self.engine.dispatch(&record.id, self.config()).unwrap());
        record.id
    }
    fn wait(
        &self,
        id: &str,
        predicate: impl Fn(&journal::Execution) -> bool,
    ) -> journal::Execution {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let record = self.engine.get(id).unwrap();
            if predicate(&record) {
                return record;
            }
            assert!(
                Instant::now() < deadline,
                "test observation deadline, record={record:?}"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::write(self.root.join("release"), b"fixture cleanup");
        if let Ok(records) = self.engine.list() {
            for record in records {
                if !record.state.terminal() {
                    let _ = self.engine.cancel(&record.id, "test-fixture-teardown");
                }
            }
        }
        // An observation deadline does not authorize a kill or deleting a live generation.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let _ = self.engine.reconcile();
            let settled = self
                .engine
                .list()
                .map(|records| {
                    records
                        .iter()
                        .all(|record| record.state.terminal() || record.process.is_none())
                })
                .unwrap_or(false);
            if settled && self.engine.supervising() == 0 {
                let _ = fs::remove_dir_all(&self.root);
                break;
            }
            if Instant::now() > deadline {
                eprintln!("preserving live fixture {}", self.root.display());
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

struct OwnedFaultProcess(std::process::Child);
impl Drop for OwnedFaultProcess {
    fn drop(&mut self) {
        // Explicit teardown of the test-owned fault driver, never a product kill policy.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn actual_inference_has_one_durable_result_and_survives_observer_loss() {
    let fixture = Fixture::new();
    let id = fixture.submit("wait");
    assert_eq!(fixture.engine.list().unwrap().len(), 1);
    fixture.wait(&id, |record| record.completed_units == 1);
    // Observation has no owned handle and dropping every observer changes nothing.
    drop(fixture.engine.get(&id).unwrap());
    assert_eq!(fixture.engine.get(&id).unwrap().state, State::Running);
    let duplicate = fixture
        .engine
        .submit("wait", fixture.invocation("wait"))
        .unwrap();
    assert_eq!(duplicate.id, id);
    assert!(!fixture.engine.dispatch(&id, fixture.config()).unwrap());
    fs::write(fixture.root.join("release"), b"continue").unwrap();
    let result = fixture.wait(&id, |record| record.state.terminal());
    assert_eq!(result.state, State::Completed);
    assert_eq!(result.result.unwrap().value, json!({"prediction": [17,39]}));
    let mut bytes = String::new();
    fixture
        .engine
        .open_result(&id, 0)
        .unwrap()
        .read_to_string(&mut bytes)
        .unwrap();
    assert_eq!(bytes, "17,39\n");
    assert_eq!(
        fs::read_to_string(fixture.root.join("effect")).unwrap(),
        "once\n"
    );
    drop(fixture.engine.get(&id).unwrap());
    let reopened = Engine::open(&fixture.root.join("state")).unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Completed);
    assert_eq!(
        reopened
            .submit("wait", fixture.invocation("wait"))
            .unwrap()
            .id,
        id
    );
}

#[test]
fn started_executor_crash_never_reexecutes_package_effects() {
    let fixture = Fixture::new();
    let id = fixture.submit("wait");
    let running = fixture.wait(&id, |record| record.completed_units == 1);
    let pid = running.process.unwrap().pid;
    // Explicit test fault injection into this owned process, never an elapsed-time policy.
    assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
    assert_eq!(
        fixture.wait(&id, |record| record.state.terminal()).state,
        State::Failed
    );
    assert!(!fixture.engine.dispatch(&id, fixture.config()).unwrap());
    assert_eq!(
        fixture
            .engine
            .submit("wait", fixture.invocation("wait"))
            .unwrap()
            .state,
        State::Failed
    );
    assert_eq!(
        fs::read_to_string(fixture.root.join("effect")).unwrap(),
        "once\n"
    );
}

#[test]
fn cooperative_cancel_is_actor_attributed_and_not_a_timer_kill() {
    let fixture = Fixture::new();
    let id = fixture.submit("wait");
    fixture.wait(&id, |record| record.completed_units == 1);
    assert!(fixture.engine.cancel(&id, "").is_err());
    fixture.engine.cancel(&id, "test-controller").unwrap();
    let result = fixture.wait(&id, |record| record.state.terminal());
    assert_eq!(result.state, State::Canceled);
    assert_eq!(result.cancel_actor.as_deref(), Some("test-controller"));
    assert!(result.result.is_none());
    assert_eq!(result.completed_units, 1);
    assert_eq!(
        Journal::open(&fixture.root.join("state"))
            .unwrap()
            .get(&id)
            .unwrap()
            .completed_units,
        1
    );
}

#[test]
fn progress_is_coalesced_without_wal_writes_and_terminal_preserves_latest() {
    let fixture = Fixture::new();
    let id = fixture.submit("progress");
    let first = fixture.wait(&id, |record| record.completed_units == 1);
    let independent = Journal::open(&fixture.root.join("state")).unwrap();
    let start = independent.get(&id).unwrap();
    assert_eq!(start.completed_units, 0);
    assert!(first.revision > start.revision);
    let wal = fixture.root.join("state/executions.sqlite3-wal");
    let durable_bytes = fs::metadata(&wal).unwrap().len();
    fs::write(fixture.root.join("advance"), b"produce actual progress").unwrap();
    let latest = fixture.wait(&id, |record| record.completed_units == 257);
    assert!(latest.revision > first.revision);
    assert_eq!(fixture.engine.list().unwrap()[0].completed_units, 257);
    assert_eq!(
        fixture
            .engine
            .submit("progress", fixture.invocation("progress"))
            .unwrap()
            .completed_units,
        257
    );
    assert_eq!(independent.get(&id).unwrap().revision, start.revision);
    assert_eq!(independent.get(&id).unwrap().completed_units, 0);
    assert_eq!(fs::metadata(&wal).unwrap().len(), durable_bytes);
    fs::write(fixture.root.join("release"), b"complete").unwrap();
    let completed = fixture.wait(&id, |record| record.state.terminal());
    assert_eq!(completed.state, State::Completed);
    assert_eq!(completed.completed_units, 257);
    assert!(completed.revision > latest.revision);
    let persisted = independent.get(&id).unwrap();
    assert_eq!(persisted.completed_units, 257);
    assert_eq!(persisted.revision, completed.revision);
}

#[test]
fn completion_wakes_owner_scheduler_to_dispatch_next_queued_package() {
    let fixture = Fixture::new();
    let first = fixture.submit("wait");
    fixture.wait(&first, |record| record.completed_units == 1);
    let next = fixture
        .engine
        .submit("queued-next", fixture.invocation("infer"))
        .unwrap();
    assert_eq!(fixture.engine.ready(8).unwrap()[0].id, next.id);
    assert_eq!(fixture.engine.active(8).unwrap()[0].id, first);
    assert_eq!(fixture.engine.nonterminal(1).unwrap().len(), 1);
    let observed = fixture.engine.activity_epoch();
    assert_eq!(
        fixture.engine.wait_activity(observed, Some(Duration::ZERO)),
        observed
    );
    assert_eq!(fixture.engine.get(&first).unwrap().state, State::Running);
    let engine = fixture.engine.clone();
    let config = fixture.config();
    let first_for_scheduler = first.clone();
    let (sent, received) = std::sync::mpsc::channel();
    let scheduler = thread::spawn(move || {
        let mut epoch = observed;
        loop {
            epoch = engine.wait_activity(epoch, None);
            if !engine.get(&first_for_scheduler).unwrap().state.terminal() {
                continue;
            }
            let ready = engine.ready(1).unwrap();
            assert_eq!(ready.len(), 1);
            assert!(engine.dispatch(&ready[0].id, config).unwrap());
            sent.send(epoch).unwrap();
            break;
        }
    });
    fs::write(fixture.root.join("release"), b"complete first execution").unwrap();
    assert!(received.recv_timeout(Duration::from_secs(15)).unwrap() > observed);
    scheduler.join().unwrap();
    assert_eq!(
        fixture
            .wait(&next.id, |record| record.state.terminal())
            .state,
        State::Completed
    );
    assert_eq!(fixture.engine.get(&first).unwrap().state, State::Completed);
    assert!(fixture.engine.nonterminal(8).unwrap().is_empty());
    assert!(fixture.engine.ready(8).unwrap().is_empty());
    assert_eq!(fixture.engine.list().unwrap().len(), 2);
}

fn public_context(
    engine: &Engine,
    actor: &str,
    request: &str,
    submission: &str,
) -> journal::SubmissionContext {
    journal::SubmissionContext {
        actor: actor.into(),
        request_id: request.into(),
        submission_id: submission.into(),
        expected_workspace_id: engine.workspace_id(),
        capture_digest: "sha256:authored-capture".into(),
        invocation_digest: "sha256:consumed-invocation".into(),
        payload_digest: "sha256:payload".into(),
        publication_authorization_id: "publication-authority-id".into(),
    }
}

fn admission_kind(error: &std::io::Error) -> &journal::AdmissionError {
    error
        .get_ref()
        .unwrap()
        .downcast_ref::<journal::AdmissionError>()
        .unwrap()
}

#[test]
fn public_workspace_and_actor_request_binding_survive_reopen() {
    let fixture = Fixture::new();
    let workspace = fixture.engine.workspace_id();
    assert_eq!(
        uuid::Uuid::parse_str(&workspace).unwrap().get_version_num(),
        4
    );
    assert_eq!(
        Engine::open(&fixture.root.join("state"))
            .unwrap()
            .workspace_id(),
        workspace
    );
    let context = public_context(
        &fixture.engine,
        "ed25519:key-a",
        "same-request",
        "same-submission",
    );
    let record = fixture
        .engine
        .submit_public(context.clone(), fixture.invocation("infer"))
        .unwrap();
    assert_eq!(
        fixture
            .engine
            .submit_public(context.clone(), fixture.invocation("infer"))
            .unwrap()
            .id,
        record.id
    );
    let mut changed = context.clone();
    changed.payload_digest = "sha256:different-payload".into();
    assert_eq!(
        *admission_kind(
            &fixture
                .engine
                .submit_public(changed, fixture.invocation("infer"))
                .unwrap_err()
        ),
        journal::AdmissionError::BindingConflict
    );
    let other = public_context(
        &fixture.engine,
        "ed25519:key-b",
        "same-request",
        "same-submission",
    );
    let other_record = fixture
        .engine
        .submit_public(other, fixture.invocation("infer"))
        .unwrap();
    assert_ne!(other_record.id, record.id);
    assert_eq!(
        fixture
            .engine
            .get_public("ed25519:key-a", "same-request")
            .unwrap()
            .id,
        record.id
    );
    assert_eq!(
        fixture
            .engine
            .get_public("record-owner-label", "same-request")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::NotFound
    );
    assert_eq!(
        fixture.engine.list_actor("ed25519:key-a", 8).unwrap().len(),
        1
    );
    let reopened = Engine::open(&fixture.root.join("state")).unwrap();
    assert_eq!(
        reopened
            .get_public("ed25519:key-a", "same-request")
            .unwrap()
            .submission
            .as_ref(),
        Some(&context)
    );
    let mut wrong_workspace = context;
    wrong_workspace.expected_workspace_id = uuid::Uuid::new_v4().to_string();
    assert_eq!(
        *admission_kind(
            &fixture
                .engine
                .submit_public(wrong_workspace, fixture.invocation("infer"))
                .unwrap_err()
        ),
        journal::AdmissionError::WorkspaceMismatch
    );
    // A malformed public operation does not refuse or cancel the private baseline.
    let private = fixture.submit("infer");
    assert_eq!(
        fixture
            .wait(&private, |record| record.state.terminal())
            .state,
        State::Completed
    );
}

#[test]
fn public_close_blocks_late_acceptance_but_never_cancels_accepted_inference() {
    let fixture = Fixture::new();
    let closed = public_context(
        &fixture.engine,
        "ed25519:key-a",
        "never-accepted",
        "closed-submission",
    );
    assert!(fixture
        .engine
        .close_submission(
            &closed.actor,
            &closed.submission_id,
            &closed.request_id,
            &closed.expected_workspace_id
        )
        .unwrap()
        .is_none());
    let reopened = Engine::open(&fixture.root.join("state")).unwrap();
    assert_eq!(
        *admission_kind(
            &reopened
                .submit_public(closed, fixture.invocation("infer"))
                .unwrap_err()
        ),
        journal::AdmissionError::SubmissionClosed
    );
    let context = public_context(
        &fixture.engine,
        "ed25519:key-a",
        "active-request",
        "active-submission",
    );
    let record = fixture
        .engine
        .submit_public(context.clone(), fixture.invocation("wait"))
        .unwrap();
    fixture
        .engine
        .dispatch(&record.id, fixture.config())
        .unwrap();
    fixture.wait(&record.id, |record| record.completed_units == 1);
    let receipt = fixture
        .engine
        .close_submission(
            &context.actor,
            &context.submission_id,
            &context.request_id,
            &context.expected_workspace_id,
        )
        .unwrap()
        .unwrap();
    assert_eq!(receipt.id, record.id);
    assert_eq!(receipt.state, State::Running);
    assert!(receipt.cancel_actor.is_none());
    assert_eq!(
        fixture
            .engine
            .submit_public(context, fixture.invocation("wait"))
            .unwrap()
            .id,
        record.id
    );
    fs::write(fixture.root.join("release"), b"finish authorized work").unwrap();
    assert_eq!(
        fixture
            .wait(&record.id, |record| record.state.terminal())
            .state,
        State::Completed
    );
}

#[test]
fn public_accept_close_race_has_one_atomic_known_outcome() {
    let fixture = Fixture::new();
    for attempt in 0..16 {
        let context = public_context(
            &fixture.engine,
            "ed25519:key-a",
            &format!("request-{attempt}"),
            &format!("submission-{attempt}"),
        );
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let accepting = fixture.engine.clone();
        let closing = fixture.engine.clone();
        let accept_context = context.clone();
        let close_context = context.clone();
        let accept_barrier = barrier.clone();
        let close_barrier = barrier.clone();
        let invocation = fixture.invocation("infer");
        let accepted = thread::spawn(move || {
            accept_barrier.wait();
            accepting.submit_public(accept_context, invocation)
        });
        let closed = thread::spawn(move || {
            close_barrier.wait();
            closing.close_submission(
                &close_context.actor,
                &close_context.submission_id,
                &close_context.request_id,
                &close_context.expected_workspace_id,
            )
        });
        barrier.wait();
        let accepted = accepted.join().unwrap();
        let closed = closed.join().unwrap().unwrap();
        match accepted {
            Ok(record) => {
                assert_eq!(closed.unwrap().id, record.id);
                assert_eq!(record.state, State::Queued);
            }
            Err(error) => {
                assert_eq!(
                    *admission_kind(&error),
                    journal::AdmissionError::SubmissionClosed
                );
                assert!(closed.is_none());
            }
        }
    }
}

#[test]
fn cancel_before_dispatch_is_durable_and_never_executes_package() {
    let fixture = Fixture::new();
    let mut journal = Journal::open(&fixture.root.join("queued-state")).unwrap();
    let record = journal
        .accept("cancel-queued", fixture.invocation("infer"))
        .unwrap();
    assert_eq!(
        journal
            .cancel(&record.id, "explicit-test-controller")
            .unwrap()
            .state,
        State::Canceled
    );
    drop(journal);
    let reopened = Engine::open(&fixture.root.join("queued-state")).unwrap();
    assert_eq!(reopened.get(&record.id).unwrap().state, State::Canceled);
    assert!(!reopened.dispatch(&record.id, fixture.config()).unwrap());
    assert!(!fixture.root.join("effect").exists());
}

#[test]
fn cursor_reservation_renewal_never_reuses_old_cursors_and_accepts_older_records() {
    let fixture = Fixture::new();
    let root = fixture.root.join("cursor-state");
    let mut journal = Journal::open(&root).unwrap();
    let id = journal
        .accept("cursor-allocation", fixture.invocation("infer"))
        .unwrap()
        .id;
    assert!(journal.claim(&id).unwrap());
    let mut child = Command::new("/usr/bin/python3")
        .arg("-c")
        .arg("import sys;sys.stdin.read()")
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    journal
        .register_process(&id, process_birth(child.id()).unwrap())
        .unwrap();
    let first = journal.running(&id).unwrap();
    let old_edge = first.revision_ceiling;
    // Exercise the allocator's boundary directly, without publishing billions of frames.
    let snapshot = journal::ProgressSnapshot {
        completed_units: 7,
        detail: "seven completed units".into(),
        revision: old_edge,
    };
    let renewed = journal.reserve_observations(&id, Some(&snapshot)).unwrap();
    assert!(renewed.revision > old_edge);
    assert!(renewed.revision_ceiling > renewed.revision);
    assert_eq!(renewed.completed_units, 7);
    let mut older = serde_json::to_value(&renewed).unwrap();
    older.as_object_mut().unwrap().remove("revision_ceiling");
    let decoded: journal::Execution = serde_json::from_value(older).unwrap();
    assert_eq!(decoded.revision_ceiling, 0);
    drop(child.stdin.take());
    child.wait().unwrap();
    let finished = journal
        .finish(
            &id,
            journal::Outcome::Failed("owned cursor driver ended".into()),
        )
        .unwrap();
    assert!(finished.revision > renewed.revision_ceiling);
    drop(journal);
    assert_eq!(
        Journal::open(&root).unwrap().get(&id).unwrap().revision,
        finished.revision
    );
}

#[test]
fn launch_failure_is_a_visible_wait_without_retry_churn() {
    let fixture = Fixture::new();
    let record = fixture
        .engine
        .submit("unlaunchable", fixture.invocation("infer"))
        .unwrap();
    let mut config = fixture.config();
    config.python = fixture.root.join("absent-python");
    fixture.engine.dispatch(&record.id, config).unwrap();
    let result = fixture.wait(&record.id, |record| record.waiting_reason.is_some());
    assert_eq!(result.state, State::Queued);
    assert_eq!(result.attempt, 1);
    let duplicate = fixture
        .engine
        .submit("unlaunchable", fixture.invocation("infer"))
        .unwrap();
    assert_eq!(duplicate.attempt, 1);
    // Correct interpreter is the observed changed condition permitting an explicit retry.
    fixture
        .engine
        .dispatch(&record.id, fixture.config())
        .unwrap();
    let result = fixture.wait(&record.id, |record| record.state.terminal());
    assert_eq!(result.state, State::Completed);
    assert_eq!(result.attempt, 2);
}

#[test]
fn durable_custody_rejects_symlink_escape_and_detects_mutation() {
    let fixture = Fixture::new();
    let unsafe_id = fixture.submit("symlink");
    assert_eq!(
        fixture
            .wait(&unsafe_id, |record| record.state.terminal())
            .state,
        State::Failed
    );
    let fifo_id = fixture.submit("fifo");
    assert_eq!(
        fixture
            .wait(&fifo_id, |record| record.state.terminal())
            .state,
        State::Failed
    );
    let id = fixture.submit("infer");
    let record = fixture.wait(&id, |record| record.state.terminal());
    let path = fixture
        .root
        .join("state")
        .join(&record.result.unwrap().artifacts[0].path);
    // Same-UID packages are not sandboxed; detect rather than silently serving altered bytes.
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(path, b"corrupted").unwrap();
    assert_eq!(
        fixture.engine.open_result(&id, 0).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData
    );
}

#[test]
fn restart_settlement_waits_for_exact_process_birth_without_adoption() {
    let fixture = Fixture::new();
    let journal_root = fixture.root.join("restart-state");
    let mut journal = Journal::open(&journal_root).unwrap();
    let id = journal
        .accept("orphan", fixture.invocation("infer"))
        .unwrap()
        .id;
    assert!(journal.claim(&id).unwrap());
    // Owned process holds actual execution lifetime until stdin EOF.
    let mut child = Command::new("/usr/bin/python3")
        .arg("-c")
        .arg("import sys;sys.stdin.read()")
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    journal
        .register_process(&id, process_birth(child.id()).unwrap())
        .unwrap();
    journal.running(&id).unwrap();
    drop(journal);
    let reopened = Engine::open(&journal_root).unwrap();
    reopened.reconcile().unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Running);
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    reopened.reconcile().unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Failed);
    assert!(!reopened.dispatch(&id, fixture.config()).unwrap());
}

#[test]
fn never_authorized_restart_can_retry_only_after_exact_birth_ends() {
    let fixture = Fixture::new();
    let journal_root = fixture.root.join("restart-state");
    let mut journal = Journal::open(&journal_root).unwrap();
    let id = journal
        .accept("never-invoked", fixture.invocation("infer"))
        .unwrap()
        .id;
    assert!(journal.claim(&id).unwrap());
    let mut child = Command::new("/usr/bin/python3")
        .arg("-c")
        .arg("import sys;sys.stdin.read()")
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    journal
        .register_process(&id, process_birth(child.id()).unwrap())
        .unwrap();
    drop(journal);
    let reopened = Engine::open(&journal_root).unwrap();
    reopened.reconcile().unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Starting);
    drop(child.stdin.take());
    child.wait().unwrap();
    reopened.reconcile().unwrap();
    let record = reopened.get(&id).unwrap();
    assert_eq!(record.state, State::Queued);
    assert!(record.waiting_reason.is_some());
    assert!(reopened.dispatch(&id, fixture.config()).unwrap());
    let deadline = Instant::now() + Duration::from_secs(15);
    while !reopened.get(&id).unwrap().state.terminal() {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(reopened.get(&id).unwrap().state, State::Completed);
}

#[test]
fn idempotency_compares_semantics_and_additive_runner_fields_are_tolerated() {
    let fixture = Fixture::new();
    fixture
        .engine
        .submit("stable", fixture.invocation("infer"))
        .unwrap();
    assert_eq!(
        fixture
            .engine
            .submit("stable", fixture.invocation("wait"))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    let id = fixture.submit("infer");
    assert_eq!(
        fixture.wait(&id, |record| record.state.terminal()).state,
        State::Completed
    );
}

#[test]
fn actual_owner_death_waits_for_orphan_then_fails_without_repeating_effects() {
    let fixture = Fixture::new();
    let mut owner = OwnedFaultProcess(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "owner_process_fixture",
                "--nocapture",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let config = json!({"state":fixture.root.join("state"),"imports":fixture.root,"invocation":fixture.invocation("wait")});
    writeln!(owner.0.stdin.as_mut().unwrap(), "{config}").unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    let record = loop {
        if let Ok(record) = fixture.engine.get("1") {
            if record.state == State::Running
                && fixture.root.join("owner-observation.json").exists()
            {
                break record;
            }
        }
        assert!(
            Instant::now() < deadline,
            "child owner failed to start its package"
        );
        thread::sleep(Duration::from_millis(5));
    };
    let observed: journal::Execution =
        serde_json::from_slice(&fs::read(fixture.root.join("owner-observation.json")).unwrap())
            .unwrap();
    assert_eq!(observed.completed_units, 1);
    let birth = record.process.unwrap();
    owner.0.kill().unwrap();
    owner.0.wait().unwrap();
    fixture.engine.reconcile().unwrap();
    assert_eq!(
        fixture.engine.get(&record.id).unwrap().state,
        State::Running
    );
    assert_eq!(fixture.engine.get(&record.id).unwrap().completed_units, 0);
    assert!(fixture.engine.get(&record.id).unwrap().revision > observed.revision);
    assert!(!execution::process_ended(&birth).unwrap());
    fs::write(fixture.root.join("release"), b"continue").unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while !execution::process_ended(&birth).unwrap() {
        assert!(
            Instant::now() < deadline,
            "orphan has not completed the authored package"
        );
        thread::sleep(Duration::from_millis(5));
    }
    fixture.engine.reconcile().unwrap();
    assert_eq!(fixture.engine.get(&record.id).unwrap().state, State::Failed);
    let persisted = Journal::open(&fixture.root.join("state"))
        .unwrap()
        .get(&record.id)
        .unwrap();
    assert!(persisted.revision > observed.revision);
    assert!(persisted.revision > observed.revision_ceiling);
    assert!(!fixture
        .engine
        .dispatch(&record.id, fixture.config())
        .unwrap());
    assert_eq!(
        fs::read_to_string(fixture.root.join("effect")).unwrap(),
        "once\n"
    );
}

/// Explicitly selected child-process fixture; its input is typed stdin configuration.
#[test]
#[ignore = "launched only by actual_owner_death component test"]
fn owner_process_fixture() {
    use std::io::BufRead;
    let mut input = String::new();
    std::io::stdin().lock().read_line(&mut input).unwrap();
    let config: serde_json::Value = serde_json::from_str(&input).unwrap();
    let engine = Engine::open(std::path::Path::new(config["state"].as_str().unwrap())).unwrap();
    let invocation: Invocation = serde_json::from_value(config["invocation"].clone()).unwrap();
    let record = engine.submit("owner-death", invocation).unwrap();
    engine
        .dispatch(
            &record.id,
            RunnerConfig {
                python: "/usr/bin/python3".into(),
                module: "runner_fixture".into(),
                import_paths: vec![PathBuf::from(config["imports"].as_str().unwrap())],
                generation_hold: None,
            },
        )
        .unwrap();
    loop {
        let snapshot = engine.get(&record.id).unwrap();
        if snapshot.completed_units == 1 {
            let path =
                PathBuf::from(config["imports"].as_str().unwrap()).join("owner-observation.json");
            let temporary = path.with_extension("pending");
            fs::write(&temporary, serde_json::to_vec(&snapshot).unwrap()).unwrap();
            fs::rename(temporary, path).unwrap();
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    // Remain the actual owner until explicitly killed by the parent fault experiment.
    let mut control = String::new();
    std::io::stdin().read_to_string(&mut control).unwrap();
}

const RUNNER: &str = r#"
import argparse, importlib, json, os, socket, struct, threading
p=argparse.ArgumentParser();p.add_argument('--execution-fd',type=int,required=True);a=p.parse_args()
s=socket.socket(fileno=a.execution_fd);lock=threading.Lock();canceled=threading.Event()
def send(value):
    data=json.dumps(value).encode()
    with lock:s.sendall(struct.pack('!I',len(data))+data)
def read():
    h=s.recv(4)
    if not h:return None
    while len(h)<4:h+=s.recv(4-len(h))
    n=struct.unpack('!I',h)[0];b=b''
    while len(b)<n:
        piece=s.recv(n-len(b))
        if not piece:return None
        b+=piece
    return json.loads(b)
send({'kind':'ready','pid':os.getpid(),'capabilities':['runtime.author-cpu/1'],'future_field':'ignored'})
command=read()
if command is None:raise SystemExit(0)
eid=command['execution_id']
def controls():
    while True:
        c=read()
        if c is None:return
        if c['kind']=='cancel' and c['execution_id']==eid:canceled.set()
threading.Thread(target=controls,daemon=True).start()
try:
    module=importlib.import_module(command['module'])
    def progress(units):send({'kind':'progress','execution_id':eid,'completed_units':units,'detail':'matrix rows completed'})
    value,artifacts=getattr(module,command['entrypoint'])(command['input'],command['output_root'],canceled,progress)
    send({'kind':'canceled','execution_id':eid} if canceled.is_set() else {'kind':'result','execution_id':eid,'value':value,'artifacts':artifacts,'extra_result_field':True})
except Exception as error:send({'kind':'failed','execution_id':eid,'code':type(error).__name__,'detail':str(error)})
"#;

const PACKAGE: &str = r#"
from pathlib import Path
def infer(inputs, output_root, canceled, progress):
    with open(inputs['side_effect'],'a') as stream:stream.write('once\n')
    progress(1)
    if inputs['mode']=='progress':
        while not Path(inputs['advance']).exists() and not canceled.wait(0.01):pass
        for units in range(2,258):progress(units)
        progress(257);progress(7)
    if inputs['mode'] in ('wait','progress'):
        while not Path(inputs['release']).exists() and not canceled.wait(0.01):pass
    if canceled.is_set():return None,[]
    # Actual CPU inference against fixed linear weights, with an independently checked result.
    weights=[[1,2],[3,4]];vector=[5,6]
    prediction=[sum(w*x for w,x in zip(row,vector)) for row in weights]
    root=Path(output_root)
    if inputs['mode']=='symlink':
        (root/'prediction').symlink_to('/etc/passwd')
    elif inputs['mode']=='fifo':
        import os
        os.mkfifo(root/'prediction')
    else:(root/'prediction').write_text(','.join(map(str,prediction))+'\n')
    return {'prediction':prediction},['prediction']
"#;
