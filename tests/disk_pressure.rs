//! CPU proofs for reserve thresholds and conservative cache pressure plans.
use cozy_machine::{
    catalog::Catalog,
    execution::Engine,
    reclaim::{self, Disk, KernelCaches},
};
use fs2::FileExt;
use std::{
    collections::HashSet,
    fs,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Arc,
};

struct Fixture {
    root: PathBuf,
    engine: Arc<Engine>,
    catalog: Catalog,
    kernels: KernelCaches,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("cm-disk-pressure-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root.join("state")).unwrap();
        let catalog = Catalog::new(&root.join("generations")).unwrap();
        let kernels = KernelCaches {
            root: root.join("kernels"),
            busy: HashSet::new(),
        };
        Self {
            root,
            engine,
            catalog,
            kernels,
        }
    }
    fn kernel(&self, name: &str, bytes: usize) -> PathBuf {
        let path = self.kernels.root.join("u1/triton").join(name);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("kernel"), vec![1u8; bytes]).unwrap();
        path
    }
    fn generation(&self, name: char) -> PathBuf {
        let path = self.catalog.root().join(name.to_string().repeat(32));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join(".hold"), b"").unwrap();
        fs::File::open(path.join(".hold"))
            .unwrap()
            .set_modified(
                std::time::SystemTime::now() - reclaim::IDLE - std::time::Duration::from_secs(1),
            )
            .unwrap();
        fs::write(path.join("weights"), vec![2u8; 1 << 20]).unwrap();
        path
    }
    fn sweep(&self, disk: &dyn Fn() -> std::io::Result<Disk>) -> reclaim::Swept {
        reclaim::sweep_with_disk(
            &self.engine,
            &self.catalog,
            &HashSet::new(),
            Some(&self.kernels),
            disk,
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn allocated(path: &Path) -> u64 {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return 0;
    };
    metadata.blocks() * 512
        + if metadata.is_dir() {
            fs::read_dir(path)
                .unwrap()
                .flatten()
                .map(|e| allocated(&e.path()))
                .sum()
        } else {
            0
        }
}

#[test]
fn only_the_reserve_is_low_disk_and_unrelievable_foreign_pressure_keeps_kernels() {
    let fixture = Fixture::new();
    let kernel = fixture.kernel("valuable", 1 << 20);
    let capacity = 100u64 << 30;
    let reserve = capacity / 50;
    let roomy = || {
        Ok(Disk {
            capacity,
            available: capacity / 40,
        })
    };
    assert!(!roomy().unwrap().pressure());
    assert_eq!(fixture.sweep(&roomy), reclaim::Swept::default());
    let full_of_foreign_data = || {
        Ok(Disk {
            capacity,
            available: 1 << 30,
        })
    };
    assert!(full_of_foreign_data().unwrap().pressure());
    assert_eq!(
        fixture.sweep(&full_of_foreign_data),
        reclaim::Swept::default()
    );
    assert!(kernel.exists());
    assert!(
        Disk {
            capacity,
            available: reserve
        }
        .pressure()
    );
    assert!(
        Disk {
            capacity,
            available: reserve + 1
        }
        .relieved()
    );
}

#[test]
fn held_generations_and_external_hardlinks_cannot_make_a_plan_look_covering() {
    let fixture = Fixture::new();
    let held = fixture.generation('a');
    let linked = fixture.generation('b');
    let hold = fs::File::open(held.join(".hold")).unwrap();
    FileExt::lock_shared(&hold).unwrap();
    let external = fixture.root.join("external-cache");
    fs::hard_link(linked.join("weights"), &external).unwrap();
    let kernel = fixture.kernel("keep", 4096);
    // Without eligibility/physical-link accounting, the two 1MiB generations appear to cover this.
    let disk = || {
        Ok(Disk {
            capacity: 100 << 30,
            available: (2 << 30) - (512 << 10),
        })
    };
    assert_eq!(fixture.sweep(&disk), reclaim::Swept::default());
    assert!(held.exists() && linked.exists() && kernel.exists() && external.exists());
}

#[test]
fn a_covering_local_plan_takes_generation_before_compiled_kernels() {
    let fixture = Fixture::new();
    let generation = fixture.generation('a');
    let kernel = fixture.kernel("keep", 1 << 20);
    let before = allocated(&fixture.root);
    let capacity = 100u64 << 30;
    let reserve = capacity / 50;
    let disk = || {
        Ok(Disk {
            capacity,
            available: reserve - (512 << 10) + (before - allocated(&fixture.root)),
        })
    };
    let swept = fixture.sweep(&disk);
    assert_eq!((swept.generations, swept.kernels), (1, 0));
    assert!(!generation.exists() && kernel.exists());
    assert!(disk().unwrap().relieved());
}

#[test]
fn a_kernel_namespace_snapshot_does_not_grant_deletion_authority() {
    let mut fixture = Fixture::new();
    let old = fixture.kernel("expired", 8192);
    let at = std::time::SystemTime::now() - std::time::Duration::from_secs(9 * 24 * 3600);
    fs::File::open(old.join("kernel"))
        .unwrap()
        .set_modified(at)
        .unwrap();
    fs::File::open(&old).unwrap().set_modified(at).unwrap();
    fixture.kernels.busy.insert("u2".into());
    let busy = fixture.kernels.root.join("u2/triton/keep");
    fs::create_dir_all(&busy).unwrap();
    fs::write(busy.join("kernel"), b"needed").unwrap();
    fs::File::open(busy.join("kernel"))
        .unwrap()
        .set_modified(at)
        .unwrap();
    fs::File::open(&busy).unwrap().set_modified(at).unwrap();
    let swept = fixture.sweep(&|| {
        Ok(Disk {
            capacity: 100 << 30,
            available: 50 << 30,
        })
    });
    assert_eq!(swept.kernels, 0);
    assert!(old.exists() && busy.exists());
}

#[test]
fn kernel_namespace_symlinks_never_authorize_foreign_cache_deletion() {
    let fixture = Fixture::new();
    let outside = fixture.root.join("outside");
    fs::create_dir_all(outside.join("triton/foreign")).unwrap();
    fs::write(outside.join("triton/foreign/kernel"), vec![1; 1 << 20]).unwrap();
    fs::create_dir_all(&fixture.kernels.root).unwrap();
    std::os::unix::fs::symlink(&outside, fixture.kernels.root.join("u1")).unwrap();
    let short = || {
        Ok(Disk {
            capacity: 100 << 30,
            available: (2 << 30) - (512 << 10),
        })
    };
    assert_eq!(fixture.sweep(&short), reclaim::Swept::default());
    assert!(outside.join("triton/foreign/kernel").exists());
}

#[test]
fn paused_and_unknown_runs_keep_their_named_generations_without_a_live_handle() {
    let fixture = Fixture::new();
    let paused = fixture.generation('c');
    let unknown = fixture.generation('d');
    let invoke = |path: &Path| cozy_machine::journal::Invocation {
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
    let database =
        rusqlite::Connection::open(fixture.engine.root.join("executions.sqlite3")).unwrap();
    database
        .execute(
            "UPDATE executions SET state='future-state' WHERE id=?1",
            [&other.id],
        )
        .unwrap();
    let bound = fixture.engine.bound_generations().unwrap();
    assert!(bound.contains(paused.file_name().unwrap().to_str().unwrap()));
    assert!(bound.contains(unknown.file_name().unwrap().to_str().unwrap()));
    let disk = || {
        Ok(Disk {
            capacity: 100 << 30,
            available: (2 << 30) - (512 << 10),
        })
    };
    let swept =
        reclaim::sweep_with_disk(&fixture.engine, &fixture.catalog, &bound, None, &disk).unwrap();
    assert_eq!(swept.generations, 0);
    assert!(paused.exists() && unknown.exists());
    database
        .execute(
            "UPDATE executions SET invocation='broken' WHERE id=?1",
            [&other.id],
        )
        .unwrap();
    assert!(
        fixture.engine.bound_generations().is_err(),
        "unreadable obligations cannot supply a destructive keep set"
    );
}

#[test]
fn a_collected_output_with_a_live_http_body_cannot_cover_kernel_pressure() {
    let fixture = Fixture::new();
    let (run, _) = fixture
        .engine
        .accept_run(
            "alice",
            "body",
            "body",
            cozy_machine::journal::Invocation {
                package: "audit/package".into(),
                input: serde_json::json!({}),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .engine
        .end_preparation(
            &run.id,
            cozy_machine::journal::Outcome::Failed("fixture settled".into()),
        )
        .unwrap();
    fixture.engine.acknowledge_collection(&run.id).unwrap();
    let path = fixture.engine.root.join("results").join(&run.id);
    fs::create_dir_all(&path).unwrap();
    fs::write(path.join("body"), vec![0x42; 1 << 20]).unwrap();
    let snapshot = cozy_machine::api::backend::OutputSnapshot {
        parts: vec![(fs::File::open(path.join("body")).unwrap(), 1 << 20)],
        length: 1 << 20,
        rev: 1,
        media_type: "application/octet-stream".into(),
        sha256: None,
    };
    let kernel = fixture.kernel("valuable", 4096);
    let disk = || {
        Ok(Disk {
            capacity: 100 << 30,
            available: (2 << 30) - (512 << 10),
        })
    };
    assert_eq!(fixture.sweep(&disk), reclaim::Swept::default());
    assert!(path.exists() && kernel.exists());
    assert_eq!(snapshot.parts[0].0.metadata().unwrap().len(), 1 << 20);
}
