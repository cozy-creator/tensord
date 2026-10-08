use super::image_sdk;
use cozy_machine::machine::update::Paths;
use std::{fs, os::unix::fs::symlink, path::Path};

const RUNTIME: &str = "cozy_runtime-0.21.1-py3-none-any.whl";
const TENSORFS: &str = "tensorfs-0.6.1-cp312-abi3-linux_x86_64.whl";

fn pair(directory: &Path, runtime: &[u8]) {
    fs::create_dir_all(directory).unwrap();
    fs::write(directory.join(RUNTIME), runtime).unwrap();
    fs::write(directory.join(TENSORFS), b"same TensorFS wheel").unwrap();
}

#[test]
fn an_activated_sdk_selection_survives_a_same_filename_link_switch() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/sdk-selection")
        .join(uuid::Uuid::new_v4().to_string());
    let engine = root.join("var/lib/cozy/rust-machine");
    let first = engine.join("sdk/first");
    let second = engine.join("sdk/second");
    pair(&first, b"first Runtime build");
    pair(&second, b"second Runtime build");
    let link = engine.join("sdk/current");
    symlink(&first, &link).unwrap();
    let paths = Paths::new(&engine, &root);
    let before = image_sdk(&paths.sdk(), &root, "uv".into());
    let before_runtime = before
        .requirements
        .iter()
        .find(|p| p.ends_with(RUNTIME))
        .unwrap();
    assert_eq!(fs::read(before_runtime).unwrap(), b"first Runtime build");

    // The real update path atomically selects a new immutable candidate. Filenames
    // and version labels are unchanged, as in a rebuilt unpublished candidate.
    let replacement = engine.join("sdk/next");
    symlink(&second, &replacement).unwrap();
    fs::rename(replacement, &link).unwrap();
    let after = image_sdk(&paths.sdk(), &root, "uv".into());
    // Both the local installer and published installer consume these exact paths;
    // their cache keys must describe the bytes the helper will later open.
    assert_ne!(
        before.requirements, after.requirements,
        "same filenames reused the old SDK cache selection"
    );
    assert_ne!(before.find_links, after.find_links);
    assert_eq!(
        fs::read(before_runtime).unwrap(),
        b"first Runtime build",
        "old selection followed the mutable link"
    );
    let after_runtime = after
        .requirements
        .iter()
        .find(|p| p.ends_with(RUNTIME))
        .unwrap();
    assert_eq!(fs::read(after_runtime).unwrap(), b"second Runtime build");
    assert_eq!(before.find_links, Some(first.canonicalize().unwrap()));
    assert_eq!(after.find_links, Some(second.canonicalize().unwrap()));
    let repeated = image_sdk(&paths.sdk(), &root, "uv".into());
    assert_eq!(repeated.requirements, after.requirements);
    assert_eq!(repeated.find_links, after.find_links);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn image_sdk_selection_preserves_baked_and_absent_pair_behavior() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/sdk-selection")
        .join(uuid::Uuid::new_v4().to_string());
    let engine = root.join("var/lib/cozy/rust-machine");
    let baked = root.join("opt/cozy/machine/wheels");
    let missing = image_sdk(&baked, &root, "uv".into());
    assert!(missing.requirements.is_empty());
    assert!(missing.find_links.is_none());
    pair(&baked, b"baked Runtime");
    let paths = Paths::new(&engine, &root);
    let selected = image_sdk(&paths.sdk(), &root, "uv".into());
    assert_eq!(selected.find_links, Some(baked.canonicalize().unwrap()));
    assert_eq!(selected.requirements.len(), 2);
    fs::remove_file(baked.join(TENSORFS)).unwrap();
    let incomplete = image_sdk(&paths.sdk(), &root, "uv".into());
    assert!(incomplete.requirements.is_empty());
    assert!(incomplete.find_links.is_none());
    fs::remove_dir_all(root).unwrap();
}

/// Real local-package installation against two previously built SDK wheels with the
/// same filename/version. Explicit local artifacts and UV_OFFLINE keep this CPU-only.
#[test]
#[ignore = "requires COZY_SDK_TEST_RUNTIME_A/B, COZY_SDK_TEST_TENSORFS, COZY_SDK_TEST_REPORT and cached offline uv dependencies"]
fn same_filename_activation_installs_new_runtime_and_retains_old_generation() {
    use cozy_machine::{
        api::install::InstallerConfig, local_source::LocalSources, machine::client,
        objects::Objects, service::Service,
    };
    use std::{path::PathBuf, process::Command, sync::Arc};
    use tensorfs_core::{sha256, store::Store};
    assert_eq!(std::env::var("UV_OFFLINE").unwrap(), "1");
    let artifact = |name: &str| PathBuf::from(std::env::var(name).expect(name));
    let (a, b, tensorfs, report) = (
        artifact("COZY_SDK_TEST_RUNTIME_A"),
        artifact("COZY_SDK_TEST_RUNTIME_B"),
        artifact("COZY_SDK_TEST_TENSORFS"),
        artifact("COZY_SDK_TEST_REPORT"),
    );
    assert_eq!(a.file_name(), b.file_name());
    assert_ne!(
        sha256::hex_digest(&fs::read(&a).unwrap()),
        sha256::hex_digest(&fs::read(&b).unwrap())
    );
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = repo
        .join("target/sdk-installation")
        .join(uuid::Uuid::new_v4().to_string());
    let engine = root.join("var/lib/cozy/rust-machine");
    let generations = root.join("generations");
    let first = engine.join("sdk/first");
    let second = engine.join("sdk/second");
    for (dir, runtime) in [(&first, &a), (&second, &b)] {
        fs::create_dir_all(dir).unwrap();
        fs::copy(runtime, dir.join(runtime.file_name().unwrap())).unwrap();
        fs::copy(&tensorfs, dir.join(tensorfs.file_name().unwrap())).unwrap();
    }
    let link = engine.join("sdk/current");
    symlink(&first, &link).unwrap();
    let paths = Paths::new(&engine, &root);
    let client = client::wheel(&engine).unwrap();
    let helper = client::helper(&engine, Path::new("uv"), &client).unwrap();
    let service = Service::open(&engine, &generations, 1).unwrap();
    let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
    let objects = Arc::new(
        Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap(),
    );
    let write = |bytes: &[u8]| {
        let digest = format!("sha256:{}", sha256::hex_digest(bytes));
        let mut writer = objects
            .begin("alice", &digest, bytes.len() as u64, 0)
            .unwrap();
        writer.append(bytes).unwrap();
        writer.finish().unwrap();
        serde_json::json!({"digest":digest,"length":bytes.len()})
    };
    let mut archive = tar::Builder::new(Vec::new());
    let fixture = repo.join("tests/fixtures/cpu_lifecycle");
    for name in [
        "pyproject.toml",
        "package.toml",
        "cpu_lifecycle/__init__.py",
    ] {
        archive
            .append_path_with_name(fixture.join(name), name)
            .unwrap();
    }
    let source = write(&archive.into_inner().unwrap());
    let manifest = write(
        serde_json::json!({"package":"local/cozy-machine-cpu-lifecycle",
        "release":"0.1.0","python_version":"3.12","source":source})
        .to_string()
        .as_bytes(),
    );
    let manifest = manifest["digest"].as_str().unwrap();
    let selected = || image_sdk(&paths.sdk(), &root, "uv".into());
    let local = |sdk: &cozy_machine::published::PackageSdk| {
        LocalSources::new(
            objects.clone(),
            InstallerConfig {
                helper_python: helper.clone(),
                python: "3.12".into(),
                generations: generations.clone(),
                client_wheel: client.clone(),
                staging_root: root.join("staging"),
                sdk: sdk.requirements.iter().map(PathBuf::from).collect(),
                uv: "uv".into(),
            },
            store.clone(),
        )
    };
    let provenance = |generation: &str| {
        let held = service.catalog.resolve(generation).unwrap();
        let result = Command::new(&held.record.python).args(["-c",
            "import importlib.metadata as m;from cozy_runtime._build_provenance import COMMIT;print(m.version('cozy-runtime'),m.version('tensorfs'),COMMIT)"])
            .output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap().trim().to_string()
    };
    let old_sdk = selected();
    let old_sources = local(&old_sdk);
    let old = old_sources.install(&service, "alice", manifest).unwrap();
    let old_provenance = provenance(&old.generation);
    assert_eq!(
        old_sources
            .install(&service, "alice", manifest)
            .unwrap()
            .generation,
        old.generation
    );
    let next = engine.join("sdk/next");
    symlink(&second, &next).unwrap();
    fs::rename(next, &link).unwrap();
    let new_sdk = selected();
    let new_sources = local(&new_sdk);
    let new = new_sources.install(&service, "alice", manifest).unwrap();
    assert_ne!(
        old.alias, new.alias,
        "new SDK activation reopened the old local alias"
    );
    assert_ne!(old.generation, new.generation);
    let new_provenance = provenance(&new.generation);
    assert_ne!(
        old_provenance, new_provenance,
        "new activation kept the old Runtime bytes"
    );
    assert_eq!(
        old_provenance
            .split_whitespace()
            .take(2)
            .collect::<Vec<_>>(),
        new_provenance
            .split_whitespace()
            .take(2)
            .collect::<Vec<_>>(),
        "the proof must keep SDK versions fixed"
    );
    assert_eq!(
        new_sources
            .install(&service, "alice", manifest)
            .unwrap()
            .generation,
        new.generation
    );
    assert_eq!(
        provenance(&old.generation),
        old_provenance,
        "old accepted generation was changed"
    );
    fs::write(
        report,
        serde_json::to_vec_pretty(&serde_json::json!({"root":root,"engine":engine,
        "generations":generations,"store":store.root(),"client":client,
        "old_sdk":old_sdk.requirements,"old_links":old_sdk.find_links,
        "new_sdk":new_sdk.requirements,"new_links":new_sdk.find_links,
        "old_installation":old,"new_installation":new,"old_provenance":old_provenance,
        "new_provenance":new_provenance,"offline":true,"old_generation_preserved":true,
        "unchanged_selection_reused":true}))
        .unwrap(),
    )
    .unwrap();
}
