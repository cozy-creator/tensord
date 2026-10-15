//! The engine's durable execution on real processes, sockets and files. Calls run on the
//! machine's own pool (a deviceless one, `tests/fixtures/cpu_lifecycle` installed); journal
//! checks drive the journal with stand-in processes. Not CLI qualification.
use tensord::{
    execution::{process_birth, Engine},
    gpu_service::{GpuConfig, Pool},
    journal::{self, Invocation, Journal, State},
    service::Service,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, OnceLock},
    thread,
    time::{Duration, Instant},
};
use tensorfs_core::store::Store;

/// A journal over a fresh root, for checks that run no package.
struct Fixture {
    root: PathBuf,
    engine: Arc<Engine>,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("tensord-durable-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let engine = Engine::open(&root.join("state")).unwrap();
        Self { root, engine }
    }
    fn invocation(&self, mode: &str) -> Invocation {
        Invocation {
            package: "fixture-cpu".into(),
            generation: "installed-cpu-v1".into(),
            module: "cpu_package".into(),
            entrypoint: "infer".into(),
            input: json!({"mode": mode}),
            ..Default::default()
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// `cpu_lifecycle` installed once for this test binary: its generations directory and identity.
fn lifecycle() -> &'static (PathBuf, String) {
    static INSTALLED: OnceLock<(PathBuf, String)> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        const INSTALL: &str = r#"
import json, subprocess, sys
from pathlib import Path
from tensord_client.packages import install
fixture, out = Path(sys.argv[1]), Path(sys.argv[2])
subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out / "client"), "."], check=True, capture_output=True)
generation = install(fixture, out / "generations", next((out / "client").glob("*.whl")), python="3.12")
print(json.dumps({"identity": generation.identity}))
"#;
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = std::env::temp_dir().join(format!("tensord-durable-lifecycle-{}", uuid::Uuid::new_v4()));
        let output = Command::new("uv")
            .current_dir(repo)
            .args(["run", "--locked", "--extra", "test", "python", "-c", INSTALL])
            .arg(repo.join("tests/fixtures/cpu_lifecycle"))
            .arg(&out)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let line = String::from_utf8(output.stdout).unwrap();
        let installed: Value = serde_json::from_str(line.lines().last().unwrap()).unwrap();
        (out.join("generations"), installed["identity"].as_str().unwrap().into())
    })
}

/// A machine's service, one call at a time, on a deviceless pool.
struct Machine {
    root: PathBuf,
    service: Arc<Service>,
}
impl Machine {
    fn open() -> Self {
        let root = std::env::temp_dir().join(format!("tensord-durable-pool-{}", uuid::Uuid::new_v4()));
        let (generations, _) = lifecycle();
        let service = Service::open(&root.join("state"), generations, 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let config = GpuConfig::deviceless(&root.join("home")).unwrap();
        service.configure_pool(Pool::new(&root.join("pool"), config, store).unwrap()).unwrap();
        Self { root, service }
    }
    fn engine(&self) -> &Arc<Engine> {
        &self.service.engine
    }
    fn submit(&self, key: &str, entrypoint: &str, input: Value) -> String {
        self.service.submit(key, &lifecycle().1, entrypoint, input).unwrap().id
    }
    fn wait(&self, id: &str, predicate: impl Fn(&journal::Execution) -> bool) -> journal::Execution {
        // Test harness bound on a broken run; the product has no such limit.
        let deadline = Instant::now() + Duration::from_secs(120);
        let mut seen = self.engine().activity_epoch();
        loop {
            let record = self.engine().get(id).unwrap();
            if predicate(&record) {
                return record;
            }
            assert!(Instant::now() < deadline, "test observation deadline, record={record:?}");
            seen = self.engine().wait_activity(seen, Some(Duration::from_millis(50)));
        }
    }
    /// One call of `entrypoint` to its end.
    fn run(&self, key: &str, entrypoint: &str, input: Value) -> journal::Execution {
        let id = self.submit(key, entrypoint, input);
        self.wait(&id, |record| record.state.terminal())
    }
}
impl Drop for Machine {
    fn drop(&mut self) {
        for record in self.engine().list().unwrap_or_default() {
            if !record.state.terminal() {
                let _ = self.engine().cancel(&record.id, "test-fixture-teardown");
            }
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.service.stop().unwrap_or(true) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// A call's success is one durable result: a second submission of the same request names the
/// same run, another request under its key is refused, and a reopened journal still holds it.
#[test]
fn a_call_has_one_durable_result_that_survives_reopen() {
    let machine = Machine::open();
    let input = json!({"steps": 2, "seconds": 0.01});
    let done = machine.run("steps", "steps", input.clone());
    assert_eq!(done.state, State::Completed, "{done:?}");
    assert_eq!(done.result.as_ref().unwrap().value["steps"], 2);
    assert_eq!(machine.service.submit("steps", &lifecycle().1, "steps", input.clone()).unwrap().id, done.id);
    let other = machine.service.submit("steps", &lifecycle().1, "steps", json!({"steps": 3}));
    assert_eq!(other.unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
    let reopened = Engine::open(&machine.root.join("state/execution")).unwrap();
    assert_eq!(reopened.get(&done.id).unwrap().state, State::Completed);
    assert_eq!(reopened.get(&done.id).unwrap().attempt, 1);
}

/// An executor that ends in the middle of a call fails that run once: it is never started
/// again, and submitting it again answers the failure.
#[test]
fn an_executor_that_ends_mid_call_fails_its_run_once() {
    let machine = Machine::open();
    let failed = machine.run("exit", "exit_now", json!({}));
    assert_eq!((failed.state, failed.attempt), (State::Failed, 1), "{failed:?}");
    machine.engine().notify_activity();
    let again = machine.service.submit("exit", &lifecycle().1, "exit_now", json!({})).unwrap();
    assert_eq!((again.id, again.state, again.attempt), (failed.id, State::Failed, 1));
}

/// A cancel needs an actor, stops the call at its next safe point, and keeps the progress it
/// had made; nothing is killed for time.
#[test]
fn cooperative_cancel_is_actor_attributed_and_not_a_timer_kill() {
    let machine = Machine::open();
    let id = machine.submit("long", "steps", json!({"steps": 400, "seconds": 0.05}));
    machine.wait(&id, |record| record.completed_units >= 1);
    assert!(machine.engine().cancel(&id, "").is_err());
    machine.engine().cancel(&id, "test-controller").unwrap();
    let ended = machine.wait(&id, |record| record.state.terminal());
    assert_eq!(ended.state, State::Canceled);
    assert_eq!(ended.cancel_actor.as_deref(), Some("test-controller"));
    assert!(ended.result.is_none() && ended.completed_units >= 1 && ended.completed_units < 400);
    let persisted = Journal::open(&machine.root.join("state/execution")).unwrap().get(&id).unwrap();
    assert_eq!(persisted.completed_units, ended.completed_units);
}

/// A call's progress is observed in memory, never a journal write per step; its last value is
/// durable with its end.
#[test]
fn progress_is_coalesced_without_wal_writes_and_terminal_preserves_latest() {
    let machine = Machine::open();
    let id = machine.submit("progress", "steps", json!({"steps": 60, "seconds": 0.05}));
    let first = machine.wait(&id, |record| record.completed_units >= 2);
    let independent = Journal::open(&machine.root.join("state/execution")).unwrap();
    let start = independent.get(&id).unwrap();
    assert!(start.completed_units < first.completed_units);
    let wal = machine.root.join("state/execution/executions.sqlite3-wal");
    let durable = fs::metadata(&wal).unwrap().len();
    let later = machine.wait(&id, |record| record.completed_units >= 20);
    assert!(later.revision > first.revision);
    assert_eq!(fs::metadata(&wal).unwrap().len(), durable, "progress wrote the journal");
    assert_eq!(independent.get(&id).unwrap().completed_units, start.completed_units);
    let done = machine.wait(&id, |record| record.state.terminal());
    assert_eq!(done.state, State::Completed);
    assert!((60..=61).contains(&done.completed_units), "a position, never a sum: {done:?}");
    assert_eq!(independent.get(&id).unwrap().completed_units, done.completed_units);
}

/// With room for one call, a queued call runs once the running one has ended.
#[test]
fn a_queued_call_runs_once_the_running_one_ends() {
    let machine = Machine::open();
    let first = machine.submit("first", "steps", json!({"steps": 20, "seconds": 0.05}));
    machine.wait(&first, |record| record.state == State::Running);
    let next = machine.submit("next", "steps", json!({"steps": 1}));
    assert_eq!(machine.engine().ready().unwrap()[0].id, next);
    let next = machine.wait(&next, |record| record.state.terminal());
    let first = machine.engine().get(&first).unwrap();
    assert_eq!((first.state, next.state), (State::Completed, State::Completed));
    assert!(next.started_at_ms >= first.finished_at_ms, "{next:?} began before {first:?} ended");
}

/// An environment whose interpreter is gone fails its run once with the reason: nothing
/// authored ran, and nothing waits in the queue to fail again.
#[test]
fn deterministic_launch_failure_fails_once_with_its_reason() {
    let machine = Machine::open();
    let (generations, identity) = lifecycle();
    let broken = format!("{}0", &identity[..identity.len() - 1]);
    let directory = generations.join(&broken);
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join(".hold"), b"").unwrap();
    // Its interpreter is a file the kernel cannot execute.
    fs::create_dir_all(directory.join("env/bin")).unwrap();
    fs::write(directory.join("env/bin/python"), b"not an interpreter").unwrap();
    let mut record: Value = serde_json::from_slice(&fs::read(generations.join(identity).join("generation.json")).unwrap()).unwrap();
    record["identity"] = broken.clone().into();
    record["python"] = directory.join("env/bin/python").to_string_lossy().into_owned().into();
    fs::write(directory.join("generation.json"), serde_json::to_vec(&record).unwrap()).unwrap();
    let id = machine.service.submit("unlaunchable", &broken, "steps", json!({})).unwrap().id;
    let failed = machine.wait(&id, |record| record.state.terminal());
    assert_eq!((failed.state, failed.attempt), (State::Failed, 1), "{failed:?}");
    assert!(failed.failure.as_deref().unwrap_or_default().contains("did not start"), "{failed:?}");
    let again = machine.service.submit("unlaunchable", &broken, "steps", json!({})).unwrap();
    assert_eq!((again.state, again.attempt), (State::Failed, 1));
    let _ = fs::remove_dir_all(directory);
}

/// A failed call keeps one triage bundle: its own terminal and its executor's stderr.
#[test]
fn a_failed_call_keeps_a_triage_bundle_with_its_stderr() {
    let machine = Machine::open();
    let failed = machine.run("note", "note", json!({"text": "crashing on purpose", "fail": true}));
    assert_eq!(failed.state, State::Failed, "{failed:?}");
    let (reference, bytes) = machine.engine().triage(&failed.id).unwrap().unwrap();
    assert_eq!(reference.length, bytes.len() as u64);
    let bundle: Value = serde_json::from_slice(&bytes).unwrap();
    let stderr = bundle["executor"]["stderr_tail"].as_str().unwrap();
    assert!(stderr.contains("crashing on purpose"), "{bundle}");
    assert!(bundle["executor"]["pid"].as_u64().unwrap() > 0);
    assert_eq!(bundle["software"]["tensord"], env!("CARGO_PKG_VERSION"));
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
        preparation_id: String::new(),
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
}

#[test]
fn cancel_before_dispatch_is_durable_and_never_executes_package() {
    let root = std::env::temp_dir().join(format!("tensord-durable-canceled-{}", uuid::Uuid::new_v4()));
    let mut journal = Journal::open(&root.join("state/execution")).unwrap();
    let invocation = Invocation { generation: lifecycle().1.clone(), module: "cpu_lifecycle:app".into(),
        entrypoint: "steps".into(), input: json!({}), ..Default::default() };
    let record = journal.accept("cancel-queued", invocation).unwrap();
    assert_eq!(journal.cancel(&record.id, "explicit-test-controller").unwrap().state, State::Canceled);
    drop(journal);
    // The machine that opens it next never starts it.
    let service = Service::open(&root.join("state"), &lifecycle().0, 1).unwrap();
    let config = GpuConfig::deviceless(&root.join("home")).unwrap();
    let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
    service.configure_pool(Pool::new(&root.join("pool"), config, store).unwrap()).unwrap();
    service.engine.notify_activity();
    let ended = service.engine.get(&record.id).unwrap();
    assert_eq!((ended.state, ended.attempt, ended.process), (State::Canceled, 0, None));
    assert!(service.stop().unwrap());
    let _ = fs::remove_dir_all(root);
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
    let first = journal.running(&id, None).unwrap();
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
            journal::Outcome::Failed(journal::Failure::abandoned("owned cursor driver ended")),
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
    journal.running(&id, None).unwrap();
    drop(journal);
    let reopened = Engine::open(&journal_root).unwrap();
    reopened.reconcile().unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Running);
    drop(child.stdin.take());
    assert!(child.wait().unwrap().success());
    reopened.reconcile().unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Failed);
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
    assert!(record.waiting_reason.is_none(), "dispatched again at once, never parked");
    assert_eq!(reopened.ready().unwrap()[0].id, id);
}

#[test]
fn an_authorized_attempt_that_never_reached_a_handler_starts_again_under_its_claim() {
    let fixture = Fixture::new();
    let mut journal = Journal::open(&fixture.root.join("undelivered-state")).unwrap();
    let executor = || {
        Command::new("/usr/bin/python3")
            .arg("-c")
            .arg("import sys;sys.stdin.read()")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let authorize = |journal: &mut Journal, key: &str| {
        let id = journal.accept(key, fixture.invocation("infer")).unwrap().id;
        assert!(journal.claim(&id).unwrap());
        let mut retained = executor();
        journal
            .register_process(&id, process_birth(retained.id()).unwrap())
            .unwrap();
        journal.running(&id, None).unwrap();
        retained.kill().unwrap();
        retained.wait().unwrap();
        id
    };
    let id = authorize(&mut journal, "undelivered");
    let record = journal.redeliver(&id).unwrap();
    assert_eq!(record.state, State::Starting);
    assert!(record.process.is_none() && record.executor.is_none() && record.started_at_ms == 0);
    assert!(record.waiting_reason.is_none());
    // Still its dispatcher's: no other claims it, and it takes a fresh executor.
    assert!(!journal.claim(&id).unwrap());
    let mut fresh = executor();
    journal
        .register_process(&id, process_birth(fresh.id()).unwrap())
        .unwrap();
    assert_eq!(journal.running(&id, None).unwrap().state, State::Running);
    drop(fresh.stdin.take());
    fresh.wait().unwrap();
    let canceled = authorize(&mut journal, "undelivered-canceled");
    journal.cancel(&canceled, "owner").unwrap();
    assert_eq!(journal.redeliver(&canceled).unwrap().state, State::Canceled);
    assert!(journal.redeliver(&canceled).is_err());
}

#[test]
fn a_newer_or_damaged_row_never_hides_the_rest_of_the_journal() {
    let fixture = Fixture::new();
    let kept = fixture
        .engine
        .submit("kept", fixture.invocation("infer"))
        .unwrap();
    let newer = fixture
        .engine
        .submit("newer", fixture.invocation("infer"))
        .unwrap();
    let damaged = fixture
        .engine
        .submit("damaged", fixture.invocation("infer"))
        .unwrap();
    let database =
        rusqlite::Connection::open(fixture.root.join("state/executions.sqlite3")).unwrap();
    database
        .execute(
            "UPDATE executions SET record=json_set(record,'$.state','paused_by_newer_machine'), state='paused_by_newer_machine' WHERE id=?1",
            [&newer.id],
        )
        .unwrap();
    database
        .execute(
            "UPDATE executions SET record=json_object('id',id,'layout','from a newer machine') WHERE id=?1",
            [&damaged.id],
        )
        .unwrap();
    let journal = Journal::open(&fixture.root.join("state")).unwrap();
    let listed = journal.list().unwrap();
    assert_eq!(listed.len(), 2, "{listed:?}");
    assert!(listed
        .iter()
        .any(|r| r.id == kept.id && r.state == State::Queued));
    let unknown = listed.iter().find(|r| r.id == newer.id).unwrap();
    assert_eq!(unknown.state, State::Unknown);
    assert!(!journal.ready().unwrap().iter().any(|r| r.id == newer.id));
    // A newer machine's state is never overwritten by this one.
    let mut journal = journal;
    assert!(journal.cancel(&newer.id, "actor-a").is_err());
    let format: String = database
        .query_row(
            "SELECT value FROM machine_metadata WHERE key='journal_format'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(format, journal::JOURNAL_FORMAT.to_string());
}

fn age(path: &std::path::Path, days: u64) {
    let when = std::time::SystemTime::now() - Duration::from_secs(days * 24 * 60 * 60);
    let times = fs::FileTimes::new().set_accessed(when).set_modified(when);
    fs::File::open(path).unwrap().set_times(times).unwrap();
}

#[test]
fn caches_expire_but_held_named_and_uncollected_work_stays() {
    use tensord::{catalog::Catalog, reclaim};
    let fixture = Machine::open();
    // Two completed runs: one collected by its client, one not.
    let collected = fixture.run("collected", "steps", json!({"steps": 1, "seconds": 0.0}));
    let owed = fixture.run("owed", "steps", json!({"steps": 1, "seconds": 0.0}));
    assert_eq!((collected.state, owed.state), (State::Completed, State::Completed));
    let (collected, owed) = (collected.id, owed.id);
    fixture.engine().acknowledge_collection(&collected).unwrap();
    let state = fixture.root.join("state/execution");
    for id in [&collected, &owed] {
        age(&state.join("results").join(id), 8);
    }
    // A crash leftover spool of a settled run, and one of a run still preparing.
    let invocation = Invocation { package: "fixture".into(), input: json!({}), ..Default::default() };
    let (queued, _) = fixture.engine().accept_run("alice", "queued", "queued", invocation).unwrap();
    let queued = queued.id;
    for id in [&collected, &queued] {
        fs::create_dir_all(state.join("staging").join(id)).unwrap();
    }
    // Generations: unused for 8 days (unbound, held, named) and one used today.
    let generations = fixture.root.join("generations");
    let catalog = Catalog::new(&generations).unwrap();
    let make = |name: &str, days: u64| {
        let directory = generations.join(name);
        fs::create_dir_all(directory.join("env")).unwrap();
        fs::write(directory.join(".hold"), b"").unwrap();
        age(&directory.join(".hold"), days);
        directory
    };
    let unused = make(&"a".repeat(32), 8);
    let held = make(&"b".repeat(32), 8);
    let named = make(&"c".repeat(32), 8);
    let recent = make(&"d".repeat(32), 0);
    let lease = fs::File::open(held.join(".hold")).unwrap();
    fs2::FileExt::lock_shared(&lease).unwrap(); // a run or executor holding it
    let bound = std::collections::HashSet::from(["c".repeat(32)]);
    let kernels = fixture.root.join("kernels");
    for (namespace, entry, days) in [("u1", "old", 9), ("u1", "new", 0), ("u2", "busy", 9)] {
        let directory = kernels.join(namespace).join("triton").join(entry);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("kernel.cubin"), b"compiled").unwrap();
        age(&directory.join("kernel.cubin"), days);
        age(&directory, days);
    }
    let caches = reclaim::KernelCaches {
        root: kernels.clone(),
    };
    // A live executor holds its namespace.
    let executor = reclaim::kernel_hold(&kernels, "u2").unwrap();
    fs2::FileExt::lock_shared(&executor).unwrap();
    let swept = reclaim::sweep(&reclaim::Caches {
        engine: fixture.engine(),
        catalog: &catalog,
        bound: &bound,
        kernels: Some(&caches),
        memo: None,
        uv_cache: None,
        store: None,
        disk: &|| reclaim::measure(&fixture.root),
    })
    .unwrap();
    assert_eq!(
        (swept.staging, swept.results, swept.generations),
        (1, 1, 1),
        "{swept:?}"
    );
    // A compiled kernel unused for the TTL goes; a recent one and a held namespace's stay.
    assert_eq!(swept.kernels, 1, "{swept:?}");
    assert!(!kernels.join("u1/triton/old").exists());
    assert!(kernels.join("u1/triton/new").exists());
    assert!(kernels.join("u2/triton/busy").exists());
    drop(executor);
    assert!(!state.join("staging").join(&collected).exists());
    assert!(state.join("staging").join(&queued).exists());
    assert!(!state.join("results").join(&collected).exists());
    assert!(state.join("results").join(&owed).exists()); // a durable output still owed
    assert!(!unused.exists());
    assert!(held.exists() && named.exists() && recent.exists());
    drop(lease);
    // The journal keeps every row.
    assert_eq!(fixture.engine().list().unwrap().len(), 3);
}
