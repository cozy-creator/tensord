//! Real source owner and two SDK ranks; requires the deployed plane-only Python wheel.
use cozy_machine::{
    device_executor::{read_frame, write_answer, Answer, Event, Kind},
    execution::Engine,
    host_memory::HostMemory,
    host_tier::{HostGrant, HostTier, HostTierConfig, SealedRequest, TierLimit},
    journal::Invocation,
    model_sources::{ModelSources, SelectedManifest},
    process::{self, Exact},
    protocol,
    scope::Scope,
};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::{net::UnixListener, process::CommandExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tensorfs_core::{
    dtype::Dtype,
    header::{Asset, Header, Part, Tensor},
    ids::ObjectRef,
    manifest::{Draft, Entry},
    repository::{Mutation, RepositoryName},
    store::{Fault, Store},
};

struct Limit;
impl TierLimit for Limit {
    fn limit(&self, _: &HostMemory, _: u64) -> u64 {
        64 << 20
    }
}

struct Fixture {
    root: PathBuf,
    store: Arc<Store>,
    manifest: ObjectRef,
    objects: Vec<ObjectRef>,
    header: Header,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("rank-source-{}", uuid::Uuid::new_v4()));
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let plain = tensorfs_core::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let mut objects = Vec::new();
        let mut tensors = Vec::new();
        for (key, byte, length) in [("first", 0x31, 2048), ("second", 0x42, 4096)] {
            let bytes = vec![byte; length];
            objects.push(
                store
                    .put_stream(&mut bytes.as_slice(), None, &Fault::default())
                    .unwrap()
                    .obj,
            );
            tensors.push((
                key.into(),
                Tensor {
                    dtype: Dtype::U8,
                    shape: vec![length as u64],
                    encoding: plain.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part::plan(Dtype::U8, vec![length as u64], &bytes),
                    )],
                },
            ));
        }
        let asset = store
            .put_stream(&mut &b"static tokenizer asset"[..], None, &Fault::default())
            .unwrap()
            .obj;
        objects.push(asset.clone());
        let denied_bytes = vec![0x77; 1024];
        objects.push(
            store
                .put_stream(&mut denied_bytes.as_slice(), None, &Fault::default())
                .unwrap()
                .obj,
        );
        let header = Header {
            configs: vec![(
                "model".into(),
                br#"{"channels":2,"fixture":"rank-source"}"#.to_vec(),
            )],
            assets: vec![(
                "tokenizer/vocab.json".into(),
                Asset {
                    logical_sha256: asset.sha256.clone(),
                    logical_length: asset.length,
                    media_type: "application/json".into(),
                    segments: vec![asset],
                },
            )],
            encodings: vec![plain.clone()],
            components: vec![
                ("allowed".into(), tensors),
                (
                    "denied".into(),
                    vec![(
                        "weight".into(),
                        Tensor {
                            dtype: Dtype::U8,
                            shape: vec![1024],
                            encoding: plain.object_id(),
                            parts: vec![(
                                "value".into(),
                                Part::plan(Dtype::U8, vec![1024], &denied_bytes),
                            )],
                        },
                    )],
                ),
            ],
        };
        let bytes = header.canonical_bytes().unwrap();
        let header_ref = store
            .put_stream(&mut bytes.as_slice(), None, &Fault::default())
            .unwrap()
            .obj;
        let manifest = store
            .put_manifest(
                &Draft {
                    entries: vec![("model".into(), Entry::CozyTensors(header_ref))],
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
                    repo: RepositoryName::new("models", "ranks").unwrap(),
                    manifest: manifest.clone(),
                }],
                &Fault::default(),
            )
            .unwrap();
        Self {
            root,
            store,
            manifest,
            objects,
            header,
        }
    }
    fn gc(&self) {
        tensorfs_core::gc::collect_cached_for(self.store.root(), &[], 1).unwrap();
        tensorfs_core::gc::collect(self.store.root(), false).unwrap();
    }
    fn retained(&self) {
        assert!(self.store.manifest_path(&self.manifest.sha256).exists());
        for object in &self.objects {
            assert!(self.store.object_path(&object.sha256).exists());
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

struct Cleanup {
    scope: Arc<Scope>,
    births: Mutex<Vec<cozy_machine::journal::ProcessBirth>>,
}
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = self.scope.end();
        for birth in self.births.get_mut().unwrap() {
            if let Ok(Some(exact)) = Exact::open(birth) {
                let _ = exact.kill();
                let _ = exact.wait();
            }
        }
    }
}

fn wait_for(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "test observation deadline: {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
#[ignore = "requires COZY_TEST_SDK_PYTHON selecting an installed plane-only SDK; CPU only"]
fn two_plane_only_ranks_relay_native_sources_and_keep_a_scope_after_leader_exit() {
    let python = std::env::var_os("COZY_TEST_SDK_PYTHON")
        .expect("select the exact plane-only SDK interpreter");
    let fixture = Fixture::new();
    let engine = Engine::open(&fixture.root.join("state")).unwrap();
    engine
        .configure_model_custody(fixture.store.clone())
        .unwrap();
    let run = engine
        .accept_run("alice", "rank-source", "rank-source", Invocation::default())
        .unwrap()
        .0;
    engine
        .retain_model(Some(&run.id), "models/ranks", &fixture.manifest)
        .unwrap();
    let socket = fixture.root.join("source.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let fifo = fixture.root.join("release-follower");
    nix::unistd::mkfifo(
        &fifo,
        nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
    )
    .unwrap();
    let scope = Arc::new(Scope::create(&format!("rank{}", uuid::Uuid::new_v4().simple())).unwrap());
    let cleanup = Cleanup {
        scope: scope.clone(),
        births: Mutex::default(),
    };
    let mut command = Command::new(python);
    command
        .args([
            "-I",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/rank_model_sources.py"
            ),
            "--root",
        ])
        .arg(&fixture.root)
        .arg("--socket")
        .arg(&socket)
        .arg("--manifest")
        .arg(fixture.manifest.id())
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .process_group(0);
    if let Some((name, value)) = scope.environment() {
        command.env(name, value);
    }
    let mut receiver = command.spawn().unwrap();
    scope.adopt(receiver.id()).unwrap();
    let birth = process::process_birth(receiver.id()).unwrap();
    cleanup.births.lock().unwrap().push(birth.clone());
    let exact = Exact::open(&birth).unwrap().unwrap();
    let recovery = scope.recovery(unsafe { libc::geteuid() });
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let (mut channel, _) = loop {
        match listener.accept() {
            Ok(connected) => break connected,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    receiver.try_wait().unwrap().is_none(),
                    "SDK receiver exited before connecting"
                );
                assert!(
                    Instant::now() < deadline,
                    "test observation deadline: SDK receiver did not connect"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("SDK receiver socket: {error}"),
        }
    };
    assert_eq!(
        read_frame(&mut channel).unwrap().unwrap().event,
        Some(Event::Unknown)
    );
    let selected = [SelectedManifest {
        manifest: fixture.manifest.id(),
        components: vec!["allowed".into()],
    }];
    let sources = ModelSources::open_session_shared(
        fixture.store.clone(),
        &selected,
        &fixture.root.join("source-holds"),
        birth.clone(),
        recovery.clone(),
    )
    .unwrap();
    let host = HostTier::new(
        fixture.store.clone(),
        HostTierConfig {
            fill_threads: 2,
            ..Default::default()
        },
        Box::new(Limit),
    )
    .unwrap();
    let peer = host.register_peer(exact.as_file().try_clone().unwrap(), true);
    let grants = [HostGrant {
        manifest: fixture.manifest.id(),
        header: fixture.header.clone(),
        components: BTreeSet::from(["allowed".into()]),
    }];
    let mut go = Answer::unavailable(0);
    go.ok = true;
    write_answer(&mut channel, &go).unwrap();
    let mut model_reads = 0;
    let mut refusals = 0;
    while let Some(frame) = read_frame(&mut channel).unwrap() {
        assert_eq!(frame.event, Some(Event::Request));
        let mut answer = Answer::unavailable(frame.seq);
        let mut files = Vec::<File>::new();
        let plan = frame
            .descriptor
            .then(|| File::from(protocol::recv_fd(&channel).unwrap()));
        let request = SealedRequest {
            sha256: &frame.sha256,
            length: frame.length,
            stage: frame.stage_regions.as_deref(),
        };
        match frame.kind {
            Kind::ModelSource => match sources.serve(&frame) {
                Ok((accepted, file)) => {
                    answer = accepted;
                    answer.descriptor = true;
                    files.push(file);
                    model_reads += 1;
                }
                Err(error) => {
                    answer.code = "model_source_refused".into();
                    answer.detail = error.to_string();
                    refusals += 1;
                }
            },
            Kind::SealedTier => {
                let file = host
                    .seal(peer, &grants, request, plan.unwrap())
                    .unwrap()
                    .unwrap();
                answer.ok = true;
                answer.held = true;
                answer.descriptor = true;
                files.push(file);
            }
            Kind::SealedStage => {
                answer.staged = host.stage(peer, &grants, request, plan.unwrap()).unwrap();
                answer.ok = true;
            }
            Kind::ObjectFiles => match host.object_files(peer, &grants, request, plan.unwrap()) {
                Ok(objects) => {
                    answer.ok = true;
                    answer.objects_sha256 = cozy_machine::host_tier::objects_digest(&objects);
                    answer.descriptors = objects.len() as u64;
                    files.extend(objects.into_iter().map(|(_, file)| file));
                }
                Err(error) => {
                    answer.code = "object_files_refused".into();
                    answer.detail = error.to_string();
                    refusals += 1;
                }
            },
            _ => panic!("unexpected source exchange: {:?}", frame.kind),
        }
        write_answer(&mut channel, &answer).unwrap();
        for file in files {
            protocol::send_fd(&channel, &file).unwrap();
        }
    }
    assert!(receiver.wait().unwrap().success());
    assert_eq!(model_reads, 4);
    assert_eq!(refusals, 6);
    assert_eq!(host.facts().ledger.object_file_grants, 4);
    assert_eq!(
        host.facts().entries,
        2,
        "both ranks share each native layout"
    );
    let facts: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("rank-facts.json")).unwrap()).unwrap();
    let follower = facts["follower_pid"].as_u64().unwrap() as u32;
    cleanup
        .births
        .lock()
        .unwrap()
        .push(process::process_birth(follower).unwrap());
    assert_eq!(facts["leader"]["store"], false);
    assert_eq!(facts["follower"]["store"], false);
    assert_eq!(facts["follower"]["rank"], 1);
    assert!(
        process::group_ended(&birth).unwrap(),
        "setsid follower is outside the original group"
    );
    assert!(
        !recovery.empty().unwrap(),
        "the actual owned scope still holds a reader"
    );
    drop(sources);
    engine.cancel(&run.id, "alice").unwrap();
    rusqlite::Connection::open(fixture.root.join("state/executions.sqlite3"))
        .unwrap()
        .execute("UPDATE model_roots SET used_ms=0", [])
        .unwrap();
    assert_eq!(engine.sweep_models().unwrap(), 1);
    let repo = RepositoryName::new("models", "ranks").unwrap();
    let previous = fs::read(fixture.store.repository_path(&repo)).unwrap();
    fixture
        .store
        .apply_repository(
            Some(&previous),
            &Mutation::DeleteRepository { repo },
            &Fault::default(),
        )
        .unwrap();
    assert_eq!(
        ModelSources::recover_sessions(fixture.store.clone(), &fixture.root.join("source-holds"))
            .unwrap(),
        0
    );
    fixture.gc();
    fixture.retained();
    let mut release = OpenOptions::new().write(true).open(&fifo).unwrap();
    release.write_all(b"release").unwrap();
    drop(release);
    wait_for(&fixture.root.join("follower-completed.json"));
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match recovery.empty() {
            Ok(true) => break,
            Ok(false) => (),
            // Exit can race a strict /proc read. Unknown is not release authority;
            // the fixture keeps the native record until a later complete proof.
            Err(error) => eprintln!("source scope exit remains unproved: {error}"),
        }
        assert!(
            Instant::now() < deadline,
            "test observation deadline: reader scope still live"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    loop {
        match ModelSources::recover_sessions(
            fixture.store.clone(),
            &fixture.root.join("source-holds"),
        ) {
            Ok(released) => {
                assert_eq!(released, 1);
                break;
            }
            Err(error) => {
                eprintln!("source recovery exit remains unproved: {error}");
                assert!(
                    Instant::now() < deadline,
                    "test observation deadline: source recovery unavailable"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
    fixture.gc();
    assert!(!fixture
        .store
        .manifest_path(&fixture.manifest.sha256)
        .exists());
    for object in &fixture.objects {
        assert!(!fixture.store.object_path(&object.sha256).exists());
    }
}
