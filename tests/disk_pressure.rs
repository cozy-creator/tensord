//! A low disk (no more than the TensorFS reserve free) drops a covering plan or nothing: only
//! bytes an unlink frees count, local caches go before models, compiled kernels last.
//! The disk's free space is each test's own figure (it cannot fill a real filesystem); what
//! is deleted, and what that frees, is real.
use cozy_machine::{
    catalog::Catalog,
    execution::Engine,
    journal::{Invocation, Outcome},
    launch_identity::Seal,
    reclaim::{self, Caches, Disk, KernelCaches, StoreCaches, Swept},
};
use std::{
    collections::HashSet,
    fs, io,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};
use tensorfs_core::{
    ids::ObjectRef,
    manifest::{Draft, Entry},
    repository::{Mutation, RepositoryName},
    store::{Fault, Store},
};

const MIB: u64 = 1 << 20;
const CAPACITY: u64 = 100 << 30;

struct Fixture {
    root: PathBuf,
    engine: Arc<Engine>,
    catalog: Catalog,
    kernels: KernelCaches,
    store: Store,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("cm-disk-pressure-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root.join("state")).unwrap();
        let catalog = Catalog::new(&root.join("generations")).unwrap();
        let kernels = KernelCaches {
            root: root.join("kernels"),
        };
        let store = Store::ensure(&root.join("store")).unwrap();
        Self {
            root,
            engine,
            catalog,
            kernels,
            store,
        }
    }
    fn kernel(&self, namespace: &str, name: &str) -> PathBuf {
        let path = self.kernels.root.join(namespace).join("triton").join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("kernel"), vec![1u8; MIB as usize]).unwrap();
        path
    }
    /// An unbound generation last used just over one sweep period ago.
    fn generation(&self, name: char) -> PathBuf {
        let path = self.catalog.root().join(name.to_string().repeat(32));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join(".hold"), b"").unwrap();
        let used = SystemTime::now() - reclaim::IDLE - std::time::Duration::from_secs(1);
        let hold = fs::File::open(path.join(".hold")).unwrap();
        hold.set_modified(used).unwrap();
        fs::write(path.join("weights"), vec![2u8; MIB as usize]).unwrap();
        path
    }
    /// A settled run whose client collected its result, with a result copy and a log.
    fn collected(&self, id: &str) -> (PathBuf, PathBuf) {
        let invocation = Invocation {
            package: "audit/package".into(),
            input: serde_json::json!({}),
            ..Default::default()
        };
        let (run, _) = self.engine.accept_run("alice", id, id, invocation).unwrap();
        let ended = Outcome::Failed(cozy_machine::journal::Failure::machine(
            "test_failed",
            "settled",
        ));
        self.engine.end_preparation(&run.id, ended).unwrap();
        self.engine.acknowledge_collection(&run.id).unwrap();
        let result = self.engine.root.join("results").join(&run.id);
        fs::create_dir_all(&result).unwrap();
        fs::write(result.join("body"), vec![3u8; MIB as usize]).unwrap();
        let log = self
            .engine
            .root
            .join("logs")
            .join(format!("{}.stderr.log", run.id));
        fs::create_dir_all(log.parent().unwrap()).unwrap();
        fs::write(&log, vec![4u8; MIB as usize]).unwrap();
        (result, log)
    }
    /// A model the store cached from a Hub: one object under one repository.
    fn model(&self, name: &str, mib: u64) -> (ObjectRef, String) {
        let body = vec![name.as_bytes()[0]; (mib * MIB) as usize];
        let object = self
            .store
            .put_stream(&mut body.as_slice(), None, &Fault::default())
            .unwrap()
            .obj;
        let entries = vec![("model.bin".to_string(), Entry::File(object.clone()))];
        let manifest = self
            .store
            .put_manifest(&Draft { entries }.seal().unwrap())
            .unwrap()
            .obj;
        let repo = RepositoryName::new("acme", name).unwrap();
        let put = Mutation::PutCheckpoint {
            repo,
            manifest: manifest.clone(),
        };
        self.store
            .apply_repository(None, &put, &Fault::default())
            .unwrap();
        // The transport's receipt makes a repository a cache root. No Hub serves this test,
        // so the receipt (the repository document's digest) is written here.
        let document = fs::read(
            self.store
                .root()
                .join("repos/acme")
                .join(format!("{name}.json")),
        )
        .unwrap();
        let receipt = self.store.root().join("tmp/cache-roots/acme");
        fs::create_dir_all(&receipt).unwrap();
        fs::write(
            receipt.join(name),
            tensorfs_core::sha256::hex_digest(&document),
        )
        .unwrap();
        (object, manifest.id())
    }
    /// A disk `short` bytes from leaving its reserve, gaining whatever this fixture frees.
    fn low(&self, short: u64) -> impl Fn() -> io::Result<Disk> + '_ {
        let before = allocated(&self.root, &mut HashSet::new());
        move || {
            let freed = before.saturating_sub(allocated(&self.root, &mut HashSet::new()));
            Ok(disk((reserve() + 1 + freed).saturating_sub(short)))
        }
    }
    /// A wheel of `name` installed with real uv (offline, from a local wheel file) into
    /// `target`: uv unpacks it into the fixture's cache and hard-links the files from there.
    fn install(&self, name: &str, target: &Path) {
        let wheel = self.root.join(format!("{name}-0.1-py3-none-any.whl"));
        if !wheel.exists() {
            let info = format!("{name}-0.1.dist-info");
            let mut zip = zip::ZipWriter::new(fs::File::create(&wheel).unwrap());
            let body = vec![b'#'; 256 << 10];
            let metadata = format!("Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n");
            let tag =
                "Wheel-Version: 1.0\nGenerator: st\nRoot-Is-Purelib: true\nTag: py3-none-any\n";
            let files = [
                (format!("{name}/__init__.py"), body.as_slice()),
                (format!("{info}/METADATA"), metadata.as_bytes()),
                (format!("{info}/WHEEL"), tag.as_bytes()),
                (format!("{info}/RECORD"), b"".as_slice()),
            ];
            for (path, bytes) in files {
                zip.start_file(path, zip::write::SimpleFileOptions::default())
                    .unwrap();
                std::io::Write::write_all(&mut zip, bytes).unwrap();
            }
            zip.finish().unwrap();
        }
        let installed = std::process::Command::new("uv")
            .args([
                "pip",
                "install",
                "--quiet",
                "--offline",
                "--no-config",
                "--target",
            ])
            .arg(target)
            .arg(&wheel)
            .env("UV_CACHE_DIR", self.root.join("uv-cache"))
            .status()
            .unwrap();
        assert!(installed.success());
    }
    fn sweep(&self, keep: Option<&[String]>, disk: &dyn Fn() -> io::Result<Disk>) -> Swept {
        reclaim::sweep(&Caches {
            engine: &self.engine,
            catalog: &self.catalog,
            bound: &self.engine.bound_generations().unwrap(),
            kernels: Some(&self.kernels),
            memo: None,
            uv_cache: Some(&self.root.join("uv-cache")),
            store: keep.map(|keep| StoreCaches {
                store: &self.store,
                keep: keep.to_vec(),
            }),
            disk,
        })
        .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn disk(available: u64) -> Disk {
    Disk {
        capacity: CAPACITY,
        available,
        inodes: 0,
        available_inodes: 0,
    }
}

fn reserve() -> u64 {
    disk(0).reserve()
}

/// Bytes on disk under `path`, each inode once.
fn allocated(path: &Path, seen: &mut HashSet<u64>) -> u64 {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return 0;
    };
    let own = if seen.insert(metadata.ino()) {
        metadata.blocks() * 512
    } else {
        0
    };
    let children = if metadata.is_dir() {
        fs::read_dir(path).ok()
    } else {
        None
    };
    own + children
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| allocated(&e.path(), seen))
        .sum::<u64>()
}

#[test]
fn a_disk_above_its_reserve_and_a_plan_that_cannot_cover_drop_nothing() {
    let fixture = Fixture::new();
    let (result, log) = fixture.collected("done");
    let (generation, kernel) = (fixture.generation('a'), fixture.kernel("u1", "valuable"));
    // A fortieth free is not low: the reserve is the store's (2% within 1 to 10 GiB).
    assert_eq!(disk(CAPACITY / 40).short(), None);
    assert_eq!(
        fixture.sweep(None, &|| Ok(disk(CAPACITY / 40))),
        Swept::default()
    );
    // Low by a gibibyte, with four mebibytes droppable: nothing covers it.
    assert_eq!(fixture.sweep(None, &fixture.low(1 << 30)), Swept::default());
    assert!(result.exists() && log.exists() && generation.exists() && kernel.exists());
}

#[test]
fn held_linked_and_open_bytes_do_not_make_a_plan_cover() {
    let fixture = Fixture::new();
    // A generation a run holds, and one whose payload a cache outside it also links.
    let (held, linked) = (fixture.generation('a'), fixture.generation('b'));
    let run = fs::File::open(held.join(".hold")).unwrap();
    fs2::FileExt::lock_shared(&run).unwrap();
    fs::hard_link(linked.join("weights"), fixture.root.join("uv-cache")).unwrap();
    // A collected result this process is still serving.
    let (result, log) = fixture.collected("served");
    fs::remove_file(&log).unwrap();
    let serving = fs::File::open(result.join("body")).unwrap();
    // A kernel namespace a live executor holds, and one reached through a link.
    let kernel = fixture.kernel("u1", "in-use");
    let executor = reclaim::kernel_hold(&fixture.kernels.root, "u1").unwrap();
    fs2::FileExt::lock_shared(&executor).unwrap();
    let outside = fixture.root.join("outside/triton/foreign");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("kernel"), vec![1u8; MIB as usize]).unwrap();
    std::os::unix::fs::symlink(
        fixture.root.join("outside"),
        fixture.kernels.root.join("u2"),
    )
    .unwrap();
    // Counted by size alone these would cover half a mebibyte many times over.
    assert_eq!(fixture.sweep(None, &fixture.low(MIB / 2)), Swept::default());
    assert!(held.exists() && linked.exists() && result.exists() && kernel.exists());
    assert!(outside.join("kernel").exists());
    drop((run, serving, executor));
}

#[test]
fn a_covering_plan_drops_results_logs_generations_then_kernels() {
    let fixture = Fixture::new();
    let (result, log) = fixture.collected("done");
    let (generation, kernel) = (fixture.generation('a'), fixture.kernel("u1", "compiled"));
    let swept = fixture.sweep(None, &fixture.low(MIB / 2));
    assert_eq!(
        (swept.results, swept.logs, swept.generations, swept.kernels),
        (1, 0, 0, 0)
    );
    assert!(!result.exists() && log.exists() && generation.exists() && kernel.exists());
    let swept = fixture.sweep(None, &fixture.low(MIB + MIB / 2));
    assert_eq!(
        (swept.logs, swept.generations, swept.kernels),
        (1, 1, 0),
        "{swept:?}"
    );
    assert!(!log.exists() && !generation.exists() && kernel.exists());
    // Kernels alone must cover what is missing, or they stay.
    assert_eq!(fixture.sweep(None, &fixture.low(2 * MIB)), Swept::default());
    assert!(kernel.exists());
    assert_eq!(fixture.sweep(None, &fixture.low(MIB / 2)).kernels, 1);
    assert!(!kernel.exists());
}

#[test]
fn models_go_after_local_caches_and_before_kernels() {
    let fixture = Fixture::new();
    let present = |object: &ObjectRef| fixture.store.object_path(&object.sha256).is_file();
    let garbage = fixture
        .store
        .put_stream(&mut &[9u8; 4096][..], None, &Fault::default())
        .unwrap()
        .obj;
    let (model, manifest) = fixture.model("unused", 4);
    let (generation, kernel) = (fixture.generation('a'), fixture.kernel("u1", "compiled"));
    // A model an unfinished run names is kept, and what remains cannot cover three
    // mebibytes: only the store's garbage goes.
    let swept = fixture.sweep(Some(&[manifest]), &fixture.low(3 * MIB));
    assert_eq!((swept.generations, swept.kernels), (0, 0), "{swept:?}");
    assert!(!present(&garbage) && present(&model) && generation.exists() && kernel.exists());
    // Unnamed, it goes once the generation has, and the kernel is not needed.
    let swept = fixture.sweep(Some(&[]), &fixture.low(3 * MIB));
    assert_eq!((swept.generations, swept.kernels), (1, 0), "{swept:?}");
    assert!(swept.store_bytes >= 4 * MIB, "{swept:?}");
    assert!(!present(&model) && !generation.exists() && kernel.exists());
}

#[test]
fn a_live_seal_holds_its_kernel_namespace() {
    let fixture = Fixture::new();
    let seal = Seal::prepare(&fixture.root, None, "incarnation", "generation", "").unwrap();
    fs::create_dir_all(seal.kernels.join("triton/compiled")).unwrap();
    fs::write(
        seal.kernels.join("triton/compiled/kernel"),
        vec![1u8; MIB as usize],
    )
    .unwrap();
    let kernel = seal.kernels.join("triton/compiled");
    assert_eq!(fixture.sweep(None, &fixture.low(MIB / 2)), Swept::default());
    assert!(kernel.exists());
    drop(seal);
    assert!(fixture.sweep(None, &fixture.low(MIB / 2)).kernels >= 1);
    assert!(!kernel.exists());
}

#[test]
fn paused_and_unknown_runs_keep_the_generations_they_name() {
    let fixture = Fixture::new();
    let (paused, unknown) = (fixture.generation('c'), fixture.generation('d'));
    let invoke = |path: &Path| Invocation {
        generation: path.file_name().unwrap().to_string_lossy().into_owned(),
        package: "audit/package".into(),
        input: serde_json::json!({}),
        ..Default::default()
    };
    let (run, _) = fixture
        .engine
        .accept_run("alice", "paused", "paused", invoke(&paused))
        .unwrap();
    fixture.engine.pause(&run.id, "alice", true).unwrap();
    let (other, _) = fixture
        .engine
        .accept_run("alice", "unknown", "unknown", invoke(&unknown))
        .unwrap();
    // A state a newer machine wrote is unfinished to this one.
    let journal =
        rusqlite::Connection::open(fixture.engine.root.join("executions.sqlite3")).unwrap();
    journal
        .execute(
            "UPDATE executions SET state='future-state' WHERE id=?1",
            [&other.id],
        )
        .unwrap();
    assert_eq!(fixture.sweep(None, &fixture.low(MIB / 2)), Swept::default());
    assert!(paused.exists() && unknown.exists());
}

#[test]
fn the_uv_cache_drops_wheels_no_environment_links_before_generations_and_never_mid_install() {
    let fixture = Fixture::new();
    let (live, collected) = (
        fixture.root.join("env-live"),
        fixture.root.join("env-collected"),
    );
    fixture.install("alpha", &live);
    fixture.install("beta", &collected);
    fs::remove_dir_all(&collected).unwrap(); // its generation was collected
    let generation = fixture.generation('a');
    let module = live.join("alpha/__init__.py");
    let unpacked = || {
        fs::read_dir(fixture.root.join("uv-cache/archive-v0"))
            .unwrap()
            .count()
    };
    assert_eq!(unpacked(), 2);
    // Fresh, it outlives the TTL pass.
    assert_eq!(
        fixture.sweep(None, &|| Ok(disk(CAPACITY / 40))),
        Swept::default()
    );
    // Low by an eighth of a mebibyte: beta's unpacked wheel covers it, before the generation.
    // Alpha's is a live environment's (its files are linked twice): it frees nothing, stays.
    let swept = fixture.sweep(None, &fixture.low(MIB / 8));
    assert_eq!((swept.uv_cache, swept.generations), (1, 0), "{swept:?}");
    assert_eq!(unpacked(), 1);
    assert!(generation.exists() && module.exists());
    assert_eq!(fs::metadata(&module).unwrap().nlink(), 2);
    // A removed entry is a cache miss: uv unpacks the wheel again.
    let again = fixture.root.join("env-again");
    fixture.install("beta", &again);
    assert!(again.join("beta/__init__.py").exists());
    fs::remove_dir_all(&again).unwrap();
    // An install holds the cache shared, as uv does for each command and the publisher for a
    // whole environment build: nothing of it goes, so the generation is what covers.
    let install = fs::File::open(fixture.root.join("uv-cache/.lock")).unwrap();
    fs2::FileExt::lock_shared(&install).unwrap();
    let swept = fixture.sweep(None, &fixture.low(MIB / 8));
    assert_eq!((swept.uv_cache, swept.generations), (0, 1), "{swept:?}");
    assert_eq!(unpacked(), 2);
}
