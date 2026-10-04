//! CPU/storage qualification of the sole-store-owner and durable bare-input boundaries.
use cozy_machine::{
    execution::Engine,
    journal::{InputFile, Invocation, State},
    objects::Objects,
    owner::Owner,
};
use std::{fs, path::PathBuf, time::Duration};
use tensorfs_core::{ids::ObjectRef, object_roots};

struct Area(PathBuf);
impl Area {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("cm-custody-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Area {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn write(objects: &Objects, bytes: &[u8]) -> ObjectRef {
    let object = ObjectRef::of(bytes);
    let mut writer = objects
        .begin("alice", &object.id(), object.length, 0)
        .unwrap();
    writer.append(bytes).unwrap();
    writer.finish().unwrap();
    object
}
fn invocation(object: &ObjectRef) -> Invocation {
    Invocation {
        package: "audit/package".into(),
        input: serde_json::json!({}),
        inputs: vec![InputFile {
            input_id: "file".into(),
            digest: object.id(),
            length: object.length,
            media_type: "application/octet-stream".into(),
            order: 0,
        }],
        ..Default::default()
    }
}

#[test]
fn accepted_input_survives_gc_and_restart() {
    let area = Area::new();
    let state = area.0.join("state");
    let owner = Owner::new(&state, &area.0.join("store"), 0, Duration::from_secs(60)).unwrap();
    let store = owner.lock().unwrap().store();
    let engine = Engine::open(&state).unwrap();
    let objects = Objects::new(&area.0.join("writes"), store.clone(), engine.clone()).unwrap();
    let object = write(&objects, b"accepted run input");
    let run = engine.submit("queued", invocation(&object)).unwrap();
    assert_eq!(run.state, State::Queued);
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 0);
    tensorfs_core::gc::collect(store.root(), false).unwrap();
    assert!(store.object_path(&object.sha256).is_file());
    drop(objects);
    drop(engine);
    drop(owner);
    // Native root custody is independently durable, before Engine or Objects restores.
    tensorfs_core::gc::collect(store.root(), false).unwrap();
    let engine = Engine::open(&state).unwrap();
    let objects = Objects::new(&area.0.join("writes"), store.clone(), engine).unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 0);
    assert!(objects.path("alice", &object.id()).unwrap().is_some());
}

#[test]
fn paused_unknown_and_adopted_parent_inputs_survive_until_terminal() {
    let area = Area::new();
    let state = area.0.join("state");
    let owner = Owner::new(&state, &area.0.join("store"), 0, Duration::from_secs(60)).unwrap();
    let store = owner.lock().unwrap().store();
    let engine = Engine::open(&state).unwrap();
    let objects = Objects::new(&area.0.join("writes"), store.clone(), engine.clone()).unwrap();
    let object = write(&objects, b"paused dependency");
    let paused = engine.submit("paused", invocation(&object)).unwrap();
    engine
        .with_journal(|j| j.pause(&paused.id, "alice", false))
        .unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 0);
    // Unknown/newer state is conservatively nonterminal, not expiration authority.
    let db = rusqlite::Connection::open(state.join("executions.sqlite3")).unwrap();
    db.execute(
        "UPDATE executions SET state='future-state' WHERE id=?1",
        [&paused.id],
    )
    .unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 0);
    db.execute(
        "UPDATE executions SET state='completed',updated_ms=0 WHERE id=?1",
        [&paused.id],
    )
    .unwrap();
    db.execute("UPDATE object_uses SET used_ms=0", []).unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 1);

    let parent = engine
        .submit(
            "parent",
            Invocation {
                package: "audit/job".into(),
                input: serde_json::json!({}),
                job: true,
                ..Default::default()
            },
        )
        .unwrap();
    let output = b"child result needed after parent resumes";
    let adopted = ObjectRef::of(output);
    let file = area.0.join("child-output");
    fs::write(&file, output).unwrap();
    objects.adopt("alice", &parent.id, &file, &adopted).unwrap();
    engine
        .with_journal(|j| j.pause(&parent.id, "alice", false))
        .unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 0);
    tensorfs_core::gc::collect(store.root(), false).unwrap();
    assert!(store.object_path(&adopted.sha256).is_file());
    engine
        .with_journal(|j| j.cancel(&parent.id, "alice"))
        .unwrap();
    db.execute(
        "UPDATE executions SET updated_ms=0 WHERE id=?1",
        [&parent.id],
    )
    .unwrap();
    db.execute("UPDATE object_uses SET used_ms=0", []).unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 1);
    tensorfs_core::gc::collect(store.root(), false).unwrap();
    assert!(!store.object_path(&adopted.sha256).exists());
}

#[test]
fn one_store_has_one_owner_across_state_roots_and_path_aliases() {
    const AREA: &str = "COZY_STORE_CUSTODY_TEST_AREA";
    if let Some(area) = std::env::var_os(AREA) {
        let area = PathBuf::from(area);
        let second = Owner::new(
            &area.join("state-b"),
            &area.join("alias/store"),
            0,
            Duration::from_secs(60),
        );
        assert!(matches!(second, Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists));
        return;
    }
    let area = Area::new();
    fs::create_dir(area.0.join("real")).unwrap();
    std::os::unix::fs::symlink(area.0.join("real"), area.0.join("alias")).unwrap();
    let first = Owner::new(
        &area.0.join("state-a"),
        &area.0.join("real/store"),
        0,
        Duration::from_secs(60),
    )
    .unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "one_store_has_one_owner_across_state_roots_and_path_aliases",
        ])
        .env(AREA, &area.0)
        .status()
        .unwrap();
    assert!(status.success());
    drop(first);
    assert!(Owner::new(
        &area.0.join("state-b"),
        &area.0.join("alias/store"),
        0,
        Duration::from_secs(60)
    )
    .is_ok());
}

#[test]
fn unaccepted_and_crash_orphan_roots_expire_without_retaining_forever() {
    let area = Area::new();
    let state = area.0.join("state");
    let owner = Owner::new(&state, &area.0.join("store"), 0, Duration::from_secs(60)).unwrap();
    let store = owner.lock().unwrap().store();
    let engine = Engine::open(&state).unwrap();
    let objects = Objects::new(&area.0.join("writes"), store.clone(), engine).unwrap();
    let object = write(&objects, b"write never submitted");
    assert_eq!(
        objects.sweep(Duration::from_secs(7 * 24 * 3600)).unwrap(),
        0
    );
    let orphan = store
        .put_stream(
            &mut &b"death between root and journal"[..],
            None,
            &Default::default(),
        )
        .unwrap()
        .obj;
    object_roots::retain(&store, &orphan).unwrap();
    rusqlite::Connection::open(state.join("executions.sqlite3"))
        .unwrap()
        .execute("UPDATE object_uses SET used_ms=0", [])
        .unwrap();
    assert_eq!(objects.sweep(Duration::ZERO).unwrap(), 2);
    tensorfs_core::gc::collect(store.root(), false).unwrap();
    assert!(!store.object_path(&object.sha256).exists());
    assert!(!store.object_path(&orphan.sha256).exists());
}
