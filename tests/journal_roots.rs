//! A run canceled after one finished child, submitted again, skips that child: its memoized
//! success was held when it completed, whatever its root then did. Between the two runs the
//! first tier of every pressured sweep (`gc::collect`, run here in the process that owns the
//! store, as the machine runs it), after a restart too, keeps what the memo and the second run
//! need.
use tensord::{
    api::install::InstallerConfig,
    calls::Calls,
    execution::Engine,
    gpu_service::{GpuConfig, Pool},
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
use tensorfs_core::{err::Code, sha256, store::Store};

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
    let store = Arc::new(Store::ensure_owned(&state.join("tensorfs")).unwrap());
    let objects =
        Arc::new(Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap());
    service.engine.hold_memos(&objects);
    let local = LocalSources::new(
        objects.clone(),
        InstallerConfig {
            helper_python: tools.helper.clone(),
            python: "3.12".into(),
            generations: root.join("generations"),
            client_wheel: tools.client.clone(),
            staging_root: root.join("staging"),
            // A Runtime wheel not yet published, installed as this machine's own (a dev build).
            sdk: std::env::var_os("COZY_TEST_RUNTIME_WHEEL").map(PathBuf::from).into_iter().collect(),
            uv: "uv".into(),
            seed_cache: None,
        },
        store.clone(),
    );
    let runs = Arc::new(Runs {
        service: service.clone(),
        objects: objects.clone(),
        publisher: None,
        local: Some(Arc::new(local)),
        own_hub: None,
        leaf: None,
        contexts: Default::default(),
    });
    let config = GpuConfig::deviceless(&root.join("home")).unwrap();
    service.configure_pool(Pool::new(&root.join("pool"), config, store.clone()).unwrap()).unwrap();
    Calls::configure(&service, store.clone(), Some(&runs)).unwrap();
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
    let manifest = json!({"package": "local/tensord-cpu-longform", "release": "0.1.0",
        "python_version": "3.12", "source": {"digest": source, "length": length}});
    write(objects, manifest.to_string().as_bytes()).0
}

/// A film whose segment `hold_at` takes 15 s: long enough to cancel it inside that segment.
fn film(manifest: &str, reference: &(String, u64), segments: &[&str], hold_at: i64) -> Spec {
    Spec {
        warm: false,
        remove: false,
        set: None,
        parent: String::new(),
        source: Source::Local(manifest.into()),
        entrypoint: "long_form".into(),
        input: json!({"reference": reference.0, "segments": segments, "hold": 15.0, "hold_at": hold_at}),
        inputs: vec![InputFile {
            input_id: "reference".into(),
            digest: reference.0.clone(),
            length: reference.1,
            media_type: "image/png".into(),
            order: 0,
        }],
        models: vec![],
        attention_kernel: String::new(),
        gpus: 0,
        hub: None,
        providers: Default::default(),
        weights_destination: String::new(),
        owner: ACTOR.into(),
        catalog_revision: String::new(),
        application: String::new(),
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

/// Every child of `parent`, ended or not.
fn children(engine: &Engine, parent: &str) -> Vec<Execution> {
    let all = engine.list().unwrap().into_iter();
    all.filter(|record| record.invocation.parent == parent).collect()
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

/// A film canceled inside its second segment, then submitted again (after a restart, when
/// `restart`), with the store's collection in between: its first segment is answered from the
/// memo, nothing of it runs; the rest run, and the film is whole.
fn a_canceled_root_resubmitted_skips_its_finished_steps(restart: bool) {
    let root = std::env::temp_dir().join(format!("cm-journal-roots-{}", uuid::Uuid::new_v4()));
    let tools = tools(&root);
    let mut machine = open(&root, &tools);
    let manifest = write_longform(&machine.objects);
    let reference = write(&machine.objects, &png(9));
    let engine = machine.service.engine.clone();
    let segments = ["a dawn", "a storm", "a calm"];
    let first = machine.runs.submit(ACTOR, "first", film(&manifest, &reference, &segments, 1)).unwrap();
    until(&engine, &first.id, "ran its second segment", |_| {
        let children = children(&engine, &first.id);
        children.len() == 2 && children[0].state == State::Completed && children[1].state == State::Running
    });
    let canceled = engine.cancel(&first.id, ACTOR).unwrap();
    machine.service.calls().unwrap().canceled(&canceled);
    until(&engine, &first.id, "ended", |r| r.state.terminal());
    let ended = children(&engine, &first.id);
    for child in &ended {
        until(&engine, &child.id, "ended", |r| r.state.terminal());
    }
    let (done, stopped) = (engine.get(&ended[0].id).unwrap(), engine.get(&ended[1].id).unwrap());
    assert_eq!((done.state, stopped.state), (State::Completed, State::Canceled));
    assert!(!done.memo.computation.is_empty() && !done.memo.memoized, "{:?}", done.memo);

    if restart {
        assert!(machine.service.stop().unwrap(), "the machine was not idle");
        let store_root = machine.store.root().to_path_buf();
        drop(machine);
        // stop() requests dispatch shutdown. Its thread may still retain the broker and the
        // Store until it observes that request; an in-process restart must wait for that
        // custody to end just as a process restart waits for the old process to exit. A
        // successful claim observes the lock's release, including completion of destructors.
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match Store::ensure_owned(&store_root) {
                Ok(store) => {
                    drop(store);
                    break;
                }
                Err(error) if error.code == Code::STORE_BUSY => {
                    assert!(Instant::now() < deadline, "the stopped dispatcher retains its Store: {error}");
                    std::thread::yield_now();
                }
                Err(error) => panic!("reopen the stopped Store: {error}"),
            }
        }
        machine = open(&root, &tools);
    }
    collect(&machine.store);
    let engine = machine.service.engine.clone();
    let mut again = film(&manifest, &reference, &segments, 1);
    again.digest.push_str("-again");
    let second = machine.runs.submit(ACTOR, "second", again).unwrap();
    let film = until(&engine, &second.id, "settled", |r| r.state.terminal());
    assert_eq!(film.state, State::Completed, "{:?}", film.failure);
    assert_eq!(film.result.unwrap().value["segments"].as_u64(), Some(3));
    let steps = children(&engine, &second.id);
    let memoized: Vec<bool> = steps.iter().map(|step| step.memo.memoized).collect();
    assert_eq!(memoized, [true, false, false], "only the finished segment is answered");
    assert_eq!(steps[0].memo.computation, done.memo.computation);
    assert_eq!(steps[0].started_at_ms, 0, "the answered segment never ran");
    assert!(machine.service.stop().unwrap());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_canceled_root_resubmitted_skips_its_finished_memoized_steps() {
    a_canceled_root_resubmitted_skips_its_finished_steps(false);
}

#[test]
fn after_a_restart_a_resubmitted_root_skips_its_finished_memoized_steps() {
    a_canceled_root_resubmitted_skips_its_finished_steps(true);
}
