//! Objects an unfinished run still needs survive the store's collection, and a restart. A job
//! paused after one finished child and a job that never started each hold a Write input (and
//! the paused one its children's results); the first tier of every pressured sweep
//! (`gc::collect`, run here in the process that owns the store, as the machine runs it) then
//! collects whatever nothing holds. Both runs are resumed and must finish whole.
use cozy_machine::{
    api::install::InstallerConfig,
    execution::Engine,
    jobs::Jobs,
    journal::{Execution, InputFile, State},
    local_source::LocalSources,
    objects::Objects,
    runs::{Runs, Source, Spec},
    service::Service,
};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};
use tensorfs_core::{sha256, store::Store};

const ACTOR: &str = "alice";

struct Machine {
    service: Arc<Service>,
    store: Arc<Store>,
    objects: Arc<Objects>,
    runs: Arc<Runs>,
}

/// The installer's helper Python and the client wheel built from this repository.
struct Tools {
    helper: PathBuf,
    client: PathBuf,
}

fn tools(root: &Path) -> Tools {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(Command::new("uv")
        .current_dir(repo)
        .args(["build", "--wheel", "--out-dir"])
        .arg(root.join("client"))
        .status()
        .unwrap()
        .success());
    let helper = Command::new("uv")
        .current_dir(repo)
        .args(["run", "--locked", "--extra", "test", "python", "-c", "import sys; print(sys.executable)"])
        .output()
        .unwrap();
    let client = fs::read_dir(root.join("client"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "whl"))
        .unwrap();
    Tools {
        helper: String::from_utf8(helper.stdout).unwrap().trim().into(),
        client,
    }
}

/// The machine's run surface over `root`, owning its store as the machine process does.
fn open(root: &Path, tools: &Tools) -> Machine {
    let state = root.join("state");
    let service = Service::open(&state, &root.join("generations"), 1).unwrap();
    let store = Arc::new(Store::ensure(&state.join("tensorfs")).unwrap());
    tensorfs_core::meta::own(&store);
    let objects =
        Arc::new(Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap());
    let local = LocalSources::new(
        objects.clone(),
        InstallerConfig {
            helper_python: tools.helper.clone(),
            python: "3.12".into(),
            generations: root.join("generations"),
            client_wheel: tools.client.clone(),
            staging_root: root.join("staging"),
            sdk: vec![],
            uv: "uv".into(),
        },
        store.clone(),
    );
    let runs = Arc::new(Runs {
        service: service.clone(),
        objects: objects.clone(),
        publisher: None,
        local: Some(Arc::new(local)),
        own_hub: None,
        jobs: Default::default(),
    });
    Jobs::configure(&service, store.clone(), Some(&runs)).unwrap();
    Machine {
        service,
        store,
        objects,
        runs,
    }
}

fn write(objects: &Objects, bytes: &[u8]) -> (String, u64) {
    let digest = format!("sha256:{}", sha256::hex_digest(bytes));
    let mut writer = objects.begin(ACTOR, &digest, bytes.len() as u64, 0).unwrap();
    writer.append(bytes).unwrap();
    writer.finish().unwrap();
    (digest, bytes.len() as u64)
}

fn png(shade: u8) -> Vec<u8> {
    let mut bytes = vec![];
    let mut encoder = png::Encoder::new(&mut bytes, 2, 2);
    encoder.set_color(png::ColorType::Rgb);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[shade; 12])
        .unwrap();
    bytes
}

/// The long-form job fixture written with Write: its manifest's digest.
fn write_longform(objects: &Objects) -> String {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_longform");
    let mut archive = tar::Builder::new(Vec::new());
    for name in ["pyproject.toml", "package.toml", "cpu_longform/__init__.py"] {
        archive.append_path_with_name(fixture.join(name), name).unwrap();
    }
    let (source, length) = write(objects, &archive.into_inner().unwrap());
    let manifest = json!({"package": "local/cozy-machine-cpu-longform", "release": "0.1.0",
        "python_version": "3.12", "source": {"digest": source, "length": length}});
    write(objects, manifest.to_string().as_bytes()).0
}

fn film(manifest: &str, reference: &(String, u64), segments: &[&str], hold_at: i64) -> Spec {
    Spec {
        warm: false,
        job: true,
        parent: String::new(),
        source: Source::Local(manifest.into()),
        entrypoint: "long_form".into(),
        input: json!({"reference": reference.0, "segments": segments, "hold": 3.0, "hold_at": hold_at}),
        inputs: vec![InputFile {
            input_id: "reference".into(),
            digest: reference.0.clone(),
            length: reference.1,
            media_type: "image/png".into(),
            order: 0,
        }],
        models: vec![],
        binding_revision: String::new(),
        attention_kernel: String::new(),
        hub: None,
        providers: Default::default(),
        weights_destination: String::new(),
        publication: String::new(),
        owner: ACTOR.into(),
        digest: format!("{}-{hold_at}", reference.0),
    }
}

/// Waits until `done` holds for the run (test harness bound; the product has no such limit).
fn until(engine: &Engine, id: &str, what: &str, done: impl Fn(&Execution) -> bool) -> Execution {
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut seen = engine.activity_epoch();
    loop {
        let record = engine.get(id).unwrap();
        if done(&record) {
            return record;
        }
        assert!(Instant::now() < deadline, "{id} never {what}: {:?} {:?}", record.state, record.failure);
        seen = engine.wait_activity(seen, Some(Duration::from_millis(200)));
    }
}

/// The first tier of a pressured sweep: whatever nothing holds goes.
fn collect(store: &Store) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match tensorfs_core::gc::collect(store.root(), false) {
            Ok(_) => return,
            Err(busy) if busy.code == tensorfs_core::err::Code::STORE_BUSY && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100))
            }
            Err(error) => panic!("gc: {error}"),
        }
    }
}

/// A job paused after one finished child, and one paused before it started; the store's
/// collection runs (after a restart, when `restart`); both resume and finish whole.
fn paused_and_unstarted_runs_keep_their_objects(restart: bool) {
    let root = std::env::temp_dir().join(format!("cm-journal-roots-{}", uuid::Uuid::new_v4()));
    let tools = tools(&root);
    let mut machine = open(&root, &tools);
    let manifest = write_longform(&machine.objects);
    let (first, second) = (write(&machine.objects, &png(9)), write(&machine.objects, &png(77)));
    let engine = machine.service.engine.clone();
    let jobs = machine.service.jobs().unwrap();

    // `paused`: its second segment holds; it pauses there, so one child has finished.
    let paused = machine
        .runs
        .submit(ACTOR, "paused", film(&manifest, &first, &["a dawn", "a storm", "a calm"], 1))
        .unwrap();
    until(&engine, &paused.id, "ran its second segment", |_| {
        engine.children(&paused.id).unwrap().len() == 2
    });
    jobs.pause(&engine.get(&paused.id).unwrap(), ACTOR).unwrap();
    until(&engine, &paused.id, "rested paused", |r| r.state == State::Paused);
    let children = engine.children(&paused.id).unwrap();
    for child in &children {
        until(&engine, &child.id, "finished", |r| r.state.terminal());
    }
    assert!(engine.children(&paused.id).unwrap().iter().any(|c| c.state == State::Completed));
    // `unstarted`: accepted and paused before its root ever ran.
    let unstarted = machine
        .runs
        .submit(ACTOR, "unstarted", film(&manifest, &second, &["a field"], -1))
        .unwrap();
    jobs.pause(&engine.get(&unstarted.id).unwrap(), ACTOR).unwrap();
    until(&engine, &unstarted.id, "rested paused", |r| r.state == State::Paused);
    assert!(engine.children(&unstarted.id).unwrap().is_empty());

    if restart {
        assert!(machine.service.stop().unwrap(), "the machine was not idle");
        drop(jobs);
        drop(machine);
        machine = open(&root, &tools);
    }
    collect(&machine.store);
    for (input, run) in [(&first, "paused"), (&second, "unstarted")] {
        let hex = input.0.trim_start_matches("sha256:");
        assert!(machine.store.object_path(hex).is_file(), "the collection took {run}'s input");
    }

    let engine = machine.service.engine.clone();
    let jobs = machine.service.jobs().unwrap();
    for id in [&paused.id, &unstarted.id] {
        jobs.resume(&engine.get(id).unwrap()).unwrap();
    }
    let film = until(&engine, &paused.id, "settled", |r| r.state.terminal());
    assert_eq!(film.state, State::Completed, "{:?}", film.failure);
    let value = &film.result.unwrap().value;
    assert_eq!((value["segments"].as_u64(), value["attempts"].as_u64()), (Some(3), Some(2)), "{value}");
    let single = until(&engine, &unstarted.id, "settled", |r| r.state.terminal());
    assert_eq!(single.state, State::Completed, "{:?}", single.failure);
    assert!(machine.service.stop().unwrap());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_sweep_keeps_what_paused_and_unstarted_runs_need() {
    paused_and_unstarted_runs_keep_their_objects(false);
}

#[test]
fn a_sweep_after_a_restart_keeps_what_paused_and_unstarted_runs_need() {
    paused_and_unstarted_runs_keep_their_objects(true);
}
