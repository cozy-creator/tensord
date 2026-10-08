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
