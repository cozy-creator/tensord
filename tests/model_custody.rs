//! CPU/storage proofs only: native model roots across independent GC callers and restart.
use cozy_machine::{
    device_executor::Binding,
    execution::Engine,
    gpu_service::{GpuPlan, PlanSlot},
    journal::{Installation, Invocation, Preparation, State, SubmissionContext},
    owner::Owner,
};
use std::{fs, path::PathBuf, sync::Arc, time::Duration};
use tensorfs_core::{
    dtype::Dtype,
    header::{Header, Part, Tensor},
    ids::ObjectRef,
    manifest::{Draft, Entry},
    repository::{Mutation, RepositoryName},
    store::{Fault, Store},
};

struct Fixture {
    root: PathBuf,
    _owner: cozy_machine::owner::Shared,
    store: Arc<Store>,
    engine: Arc<Engine>,
    manifest: ObjectRef,
    data: ObjectRef,
}
impl Fixture {
    fn new(configure: bool) -> Self {
        let root = std::env::temp_dir().join(format!("cm-model-custody-{}", uuid::Uuid::new_v4()));
        let owner = Owner::new(
            &root.join("state"),
            &root.join("store"),
            0,
            Duration::from_secs(60),
        )
        .unwrap();
        let store = owner.lock().unwrap().store();
        let engine = Engine::open(&root.join("state")).unwrap();
        let bytes = vec![0x31; 2048];
        let data = store
            .put_stream(&mut bytes.as_slice(), None, &Fault::default())
            .unwrap()
            .obj;
        let plain = tensorfs_core::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: vec![],
            assets: vec![],
            encodings: vec![plain.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![2048],
                        encoding: plain.object_id(),
                        parts: vec![("value".into(), Part::plan(Dtype::U8, vec![2048], &bytes))],
                    },
                )],
            )],
        };
        let header = store
            .put_stream(
                &mut header.canonical_bytes().unwrap().as_slice(),
                None,
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let optional = store
            .put_stream(&mut &vec![0x55; 8192][..], None, &Fault::default())
            .unwrap()
            .obj;
        let manifest = store
            .put_manifest(
                &Draft {
                    entries: vec![
                        ("cozytensors".into(), Entry::CozyTensors(header)),
                        ("optional.txt".into(), Entry::File(optional)),
                        ("payload".into(), Entry::File(data.clone())),
                    ],
                }
                .seal()
                .unwrap(),
            )
            .unwrap()
            .obj;
        store
            .apply_cached_repository(
                None,
                &[Mutation::PutCheckpoint {
                    repo: RepositoryName::new("models", "fixture").unwrap(),
                    manifest: manifest.clone(),
                }],
                &Fault::default(),
            )
            .unwrap();
        if configure {
            engine.configure_model_custody(store.clone()).unwrap();
        }
        let fixture = Self {
            root,
            _owner: owner,
            store,
            engine,
            manifest,
            data,
        };
        fixture.prepare();
        fixture
    }
    fn prepare(&self) {
        self.engine
            .bind_installation(Installation {
                actor: "alice".into(),
                alias: "published".into(),
                generation: "a".repeat(32),
                package: "audit/package".into(),
                release: "1".into(),
                interface: b"{}".to_vec(),
            })
            .unwrap();
        let plan = GpuPlan {
            id: "gpu-custody".into(),
            installation: "published".into(),
            generation: "a".repeat(32),
            entrypoint: "infer".into(),
            slots: vec![PlanSlot {
                binding: Binding {
                    snapshot: self.manifest.id(),
                    model: "models/fixture@1".into(),
                    components: vec!["model".into()],
                    ..Default::default()
                },
                selected_encoded_bytes: self.data.length,
            }],
            degree: 1,
        };
        self.engine
            .bind_preparation(Preparation {
                actor: "alice".into(),
                id: plan.id.clone(),
                installation: plan.installation.clone(),
                document: serde_json::to_vec(&plan).unwrap(),
            })
            .unwrap();
    }
    fn accept(&self, request: &str) -> String {
        self.engine
            .submit_public(
                SubmissionContext {
                    actor: "alice".into(),
                    request_id: request.into(),
                    submission_id: request.into(),
                    expected_workspace_id: self.engine.workspace_id(),
                    invocation_digest: request.into(),
                    preparation_id: "gpu-custody".into(),
                    ..Default::default()
                },
                Invocation {
                    package: "audit/package".into(),
                    generation: "a".repeat(32),
                    entrypoint: "infer".into(),
                    input: serde_json::json!({}),
                    ..Default::default()
                },
            )
            .unwrap()
            .id
    }
    fn gc(&self) {
        // A source/ingest caller has no knowledge of Publisher::protected's keep snapshot.
        tensorfs_core::gc::collect_cached_for(self.store.root(), &[], 1).unwrap();
        let repo=RepositoryName::new("models","fixture").unwrap();
        if let Ok(body)=fs::read(self.store.repository_path(&repo)) {
            self.store.apply_repository(Some(&body),&Mutation::DeleteRepository {repo},&Fault::default()).unwrap();
        }
        tensorfs_core::gc::collect(self.store.root(), false).unwrap();
    }
    fn age(&self) {
        rusqlite::Connection::open(self.root.join("state/executions.sqlite3"))
            .unwrap()
            .execute("UPDATE model_roots SET used_ms=0", [])
            .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
#[ignore = "negative baseline reproduction; run explicitly on a machine without model custody"]
fn caller_local_keep_lists_do_not_protect_an_accepted_model_from_another_gc_caller() {
    let fixture = Fixture::new(false);
    fixture.accept("baseline");
    fixture.gc();
    assert!(
        fixture
            .store
            .manifest_path(&fixture.manifest.sha256)
            .is_file(),
        "independent native GC removed the accepted model"
    );
}

#[test]
fn queued_paused_unknown_and_restarted_models_survive_independent_gc() {
    let mut fixture = Fixture::new(true);
    let id = fixture.accept("accepted");
    fixture.gc();
    assert!(fixture
        .store
        .manifest_path(&fixture.manifest.sha256)
        .is_file());
    assert!(fixture.store.object_path(&fixture.data.sha256).is_file());
    // The optional documentation can go with the cache repository; runtime custody remains.
    assert!(!fixture
        .store
        .repository_path(&RepositoryName::new("models", "fixture").unwrap())
        .exists());
    fixture
        .engine
        .with_journal(|j| j.pause(&id, "alice", false))
        .unwrap();
    fixture.age();
    assert_eq!(fixture.engine.sweep_models().unwrap(), 0);
    fixture.gc();
    let db = rusqlite::Connection::open(fixture.root.join("state/executions.sqlite3")).unwrap();
    db.execute("UPDATE executions SET state='future-state',record=json_set(record,'$.state','future-state') WHERE id=?1",[&id]).unwrap();
    let old = std::mem::replace(
        &mut fixture.engine,
        Engine::open(&fixture.root.join("placeholder")).unwrap(),
    );
    drop(old);
    let reopened = Engine::open(&fixture.root.join("state")).unwrap();
    reopened
        .configure_model_custody(fixture.store.clone())
        .unwrap();
    assert_eq!(reopened.get(&id).unwrap().state, State::Unknown);
    assert_eq!(reopened.sweep_models().unwrap(), 0);
    fixture.engine = reopened;
    fixture.gc();
    assert!(fixture.store.object_path(&fixture.data.sha256).is_file());
}

#[test]
fn one_native_root_serves_two_runs_and_releases_only_after_both_settle() {
    let fixture = Fixture::new(true);
    let first = fixture.accept("first");
    let second = fixture.accept("second");
    fixture.age();
    let db = rusqlite::Connection::open(fixture.root.join("state/executions.sqlite3")).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM model_roots", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    fixture
        .engine
        .with_journal(|j| j.cancel(&first, "alice"))
        .unwrap();
    assert_eq!(fixture.engine.sweep_models().unwrap(), 0);
    fixture.gc();
    fixture
        .engine
        .with_journal(|j| j.cancel(&second, "alice"))
        .unwrap();
    assert_eq!(fixture.engine.sweep_models().unwrap(), 1);
    tensorfs_core::gc::collect(fixture.store.root(), false).unwrap();
    assert!(!fixture.store.object_path(&fixture.data.sha256).exists());
}

#[test]
fn unreadable_model_obligation_disables_native_gc_but_other_cpu_acceptance_works() {
    let mut fixture = Fixture::new(true);
    fixture.accept("accepted");
    let db = rusqlite::Connection::open(fixture.root.join("state/executions.sqlite3")).unwrap();
    db.execute("UPDATE preparations SET record='broken'", [])
        .unwrap();
    let old = std::mem::replace(
        &mut fixture.engine,
        Engine::open(&fixture.root.join("placeholder")).unwrap(),
    );
    drop(old);
    let reopened = Engine::open(&fixture.root.join("state")).unwrap();
    reopened
        .configure_model_custody(fixture.store.clone())
        .unwrap();
    assert_eq!(
        tensorfs_core::gc::collect_cached_for(fixture.store.root(), &[], 1)
            .unwrap_err()
            .code,
        tensorfs_core::err::Code::STORE_BUSY
    );
    assert!(reopened
        .submit(
            "other-cpu",
            Invocation {
                package: "audit/cpu".into(),
                input: serde_json::json!({}),
                ..Default::default()
            }
        )
        .is_ok());
    assert!(fixture.store.object_path(&fixture.data.sha256).is_file());
    fixture.engine = reopened;
}
