//! Write objects keep durable TensorFS roots while an unfinished run names them; the store's
//! GC takes them only once released.
use cozy_machine::{
    execution::Engine,
    journal::{InputFile, Invocation, State},
    objects::Objects,
};
use std::{fs, path::PathBuf, sync::Arc, time::Duration};
use tensorfs_core::{ids::ObjectRef, object_roots, store::Store};

struct Area {
    root: PathBuf,
    store: Arc<Store>,
    engine: Arc<Engine>,
    objects: Objects,
}
impl Area {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("cm-custody-{}", uuid::Uuid::new_v4()));
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let engine = Engine::open(&root.join("state")).unwrap();
        let objects = Objects::new(&root.join("writes"), store.clone(), engine.clone()).unwrap();
        Self {
            root,
            store,
            engine,
            objects,
        }
    }
    fn write(&self, bytes: &[u8]) -> ObjectRef {
        let object = ObjectRef::of(bytes);
        let mut writer = self
            .objects
            .begin("alice", &object.id(), object.length, 0)
            .unwrap();
        writer.append(bytes).unwrap();
        writer.finish().unwrap();
        object
    }
    fn present(&self, object: &ObjectRef) -> bool {
        self.store.object_path(&object.sha256).is_file()
    }
    /// The store's first GC tier, then a release of every root nothing needs now.
    fn collect(&self) -> usize {
        std::thread::sleep(Duration::from_millis(5));
        let released = self.objects.release(Duration::ZERO).unwrap();
        tensorfs_core::gc::collect(self.store.root(), false).unwrap();
        released
    }
}
impl Drop for Area {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
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
fn an_acknowledged_write_survives_gc_until_its_run_ends() {
    let area = Area::new();
    let object = area.write(b"accepted run input");
    // Rooted before the journal row: the store's GC alone never takes it.
    tensorfs_core::gc::collect(area.store.root(), false).unwrap();
    assert!(area.present(&object));
    let run = area.engine.submit("queued", invocation(&object)).unwrap();
    assert_eq!(run.state, State::Queued);
    assert_eq!(area.collect(), 0);
    area.engine.cancel(&run.id, "alice").unwrap();
    assert_eq!(area.collect(), 1);
    assert!(!area.present(&object));
}

#[test]
fn paused_unknown_and_adopted_dependencies_stay_until_their_run_ends() {
    let area = Area::new();
    let input = area.write(b"paused dependency");
    let paused = area.engine.submit("paused", invocation(&input)).unwrap();
    area.engine.pause(&paused.id, "alice", false).unwrap();
    assert_eq!(area.collect(), 0);
    // A state a newer machine wrote is unfinished to this one.
    let db = rusqlite::Connection::open(area.root.join("state/executions.sqlite3")).unwrap();
    db.execute(
        "UPDATE executions SET state='future-state' WHERE id=?1",
        [&paused.id],
    )
    .unwrap();
    assert_eq!(area.collect(), 0);
    assert!(area.present(&input));

    let parent = area
        .engine
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
    let output = b"child result needed after the parent resumes";
    let adopted = ObjectRef::of(output);
    let file = area.root.join("child-output");
    fs::write(&file, output).unwrap();
    area.objects
        .adopt("alice", &parent.id, &file, &adopted)
        .unwrap();
    area.engine.pause(&parent.id, "alice", false).unwrap();
    assert_eq!(area.collect(), 0);
    assert!(area.present(&adopted));
    area.engine.cancel(&parent.id, "alice").unwrap();
    assert_eq!(area.collect(), 1);
    assert!(!area.present(&adopted) && area.present(&input));
}

#[test]
fn unsubmitted_writes_and_crash_orphans_age_out() {
    let area = Area::new();
    let unsubmitted = area.write(b"write never submitted");
    // Within the TTL an acknowledged Write waits for its run.
    assert_eq!(area.objects.release(cozy_machine::reclaim::TTL).unwrap(), 0);
    // A root written just before a crash, with no journal row.
    let orphan = area
        .store
        .put_stream(
            &mut &b"death between root and journal"[..],
            None,
            &Default::default(),
        )
        .unwrap()
        .obj;
    object_roots::retain(&area.store, &orphan).unwrap();
    assert_eq!(
        area.objects.release(cozy_machine::reclaim::IDLE).unwrap(),
        0
    );
    assert_eq!(area.collect(), 2);
    assert!(!area.present(&unsubmitted) && !area.present(&orphan));
}
