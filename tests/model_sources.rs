#[path = "../src/model_sources.rs"]
mod model_sources;
#[allow(dead_code)]
#[path = "../src/os.rs"]
mod os;

use model_sources::{ModelSources, SelectedManifest, SourceRequest, SourceRole};
use std::{
    fs,
    io::{self, Read},
    os::unix::fs::FileExt,
};
use tensorfs_core::{
    dtype::Dtype,
    header::{Asset, Header, Part, Tensor},
    ids::ObjectRef,
    manifest::{Draft, Entry},
    registry,
    store::{Fault, Store},
};

fn fixture() -> (std::path::PathBuf, String, Vec<u8>, ObjectRef, ObjectRef) {
    let root = std::env::temp_dir().join(format!(
        "machine-source-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let plain = registry::seeds()
        .into_iter()
        .find(|s| s.alias == "plain/1")
        .unwrap()
        .spec;
    let data: Vec<u8> = (0..65540).map(|n| (n % 251) as u8).collect();
    let other = vec![9; 65540];
    let object = ObjectRef::of(&data);
    let denied = ObjectRef::of(&other);
    store
        .put_stream(&mut data.as_slice(), Some(&object), &Fault::default())
        .unwrap();
    store
        .put_stream(&mut other.as_slice(), Some(&denied), &Fault::default())
        .unwrap();
    let asset_bytes = b"static tokenizer asset";
    let asset_ref = ObjectRef::of(asset_bytes);
    store
        .put_stream(
            &mut asset_bytes.as_slice(),
            Some(&asset_ref),
            &Fault::default(),
        )
        .unwrap();
    let header = Header {
        configs: vec![],
        assets: vec![(
            "tokenizer/vocab.json".into(),
            Asset {
                logical_sha256: asset_ref.sha256.clone(),
                logical_length: asset_ref.length,
                media_type: "application/json".into(),
                segments: vec![asset_ref],
            },
        )],
        encodings: vec![plain.clone()],
        components: vec![
            (
                "allowed".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![data.len() as u64],
                        encoding: plain.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part::plan(Dtype::U8, vec![data.len() as u64], &data),
                        )],
                    },
                )],
            ),
            (
                "denied".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![other.len() as u64],
                        encoding: plain.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part::plan(Dtype::U8, vec![other.len() as u64], &other),
                        )],
                    },
                )],
            ),
        ],
    };
    let bytes = header.canonical_bytes().unwrap();
    let header_ref = ObjectRef::of(&bytes);
    store
        .put_stream(&mut bytes.as_slice(), Some(&header_ref), &Fault::default())
        .unwrap();
    let manifest = Draft {
        entries: vec![("model".into(), Entry::CozyTensors(header_ref))],
    }
    .seal()
    .unwrap();
    store.put_manifest(&manifest).unwrap();
    (root, manifest.manifest_id(), data, object, denied)
}

#[test]
fn source_role_exports_only_trusted_manifest_components_with_exact_bytes() {
    let (root, manifest, data, object, denied) = fixture();
    let mut broker = ModelSources::open(
        &root,
        &[SelectedManifest {
            manifest: manifest.clone(),
            components: vec!["allowed".into()],
        }],
    )
    .unwrap();
    let request = SourceRequest {
        manifest: manifest.clone(),
        role: SourceRole::Header,
        name: String::new(),
        length: 0,
    };
    let mut header = broker.read(&request).unwrap();
    let mut bytes = Vec::new();
    header.file.read_to_end(&mut bytes).unwrap();
    assert_eq!(tensorfs_core::sha256::hex_digest(&bytes), header.sha256);
    assert_eq!(bytes.len() as u64, header.length);
    Header::parse(&bytes).unwrap();
    let mut wanted = SourceRequest {
        role: SourceRole::Object,
        name: object.id(),
        length: object.length,
        ..request.clone()
    };
    let grant = broker.read(&wanted).unwrap();
    let mut part = vec![0; 97];
    grant.file.read_exact_at(&mut part, 17).unwrap();
    assert_eq!(part, data[17..114]);
    assert_eq!(grant.sha256, object.sha256);
    // Verification never changes identity or delegates another component.
    wanted.name = denied.id();
    wanted.length = denied.length;
    assert_eq!(
        broker.read(&wanted).err().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    wanted.name = object.id();
    wanted.length += 1;
    assert_eq!(
        broker.read(&wanted).err().unwrap().kind(),
        io::ErrorKind::InvalidInput
    );
    wanted.manifest = "sha256:".to_owned() + &"0".repeat(64);
    assert_eq!(
        broker.read(&wanted).err().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    wanted.manifest = manifest;
    wanted.role = SourceRole::Asset;
    wanted.name = "undeclared".into();
    assert_eq!(
        broker.read(&wanted).err().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    wanted.name = "tokenizer/vocab.json".into();
    let mut asset = broker.read(&wanted).unwrap();
    let mut bytes = Vec::new();
    asset.file.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"static tokenizer asset");
    assert_eq!(tensorfs_core::sha256::hex_digest(&bytes), asset.sha256);
    assert_eq!(os::seals(&asset.file).unwrap(), os::FULL_SEALS);
    drop(asset);
    drop(broker);
    drop(grant);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn copied_closure_is_verified_by_linked_core_without_foreign_catalog() {
    let (source, manifest, data, object, denied) = fixture();
    let selection = SelectedManifest {
        manifest: manifest.clone(),
        components: vec!["allowed".into()],
    };
    let original = ModelSources::open(&source, std::slice::from_ref(&selection)).unwrap();
    let target = source.with_extension("target");
    let store = Store::ensure(&target).unwrap();
    for path in original.closure_paths(&manifest).unwrap() {
        let copied = target.join(path.strip_prefix(&source).unwrap());
        fs::create_dir_all(copied.parent().unwrap()).unwrap();
        fs::copy(path, copied).unwrap();
    }
    assert!(!store.blob_path(&denied.sha256).exists());
    drop(original);
    drop(store);
    let mut broker = ModelSources::open(&target, &[selection]).unwrap();
    let (count, bytes) = broker.verify_selected().unwrap();
    assert_eq!(count, 3); // Selected tensor, header, declared tokenizer asset.
    assert!(bytes > object.length);
    let grant = broker
        .read(&SourceRequest {
            manifest,
            role: SourceRole::Object,
            name: object.id(),
            length: object.length,
        })
        .unwrap();
    let mut read = vec![0; data.len()];
    grant.file.read_exact_at(&mut read, 0).unwrap();
    assert_eq!(read, data);
    drop(grant);
    drop(broker);
    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(target).unwrap();
}

#[test]
fn transferred_object_corruption_fails_native_admission() {
    let (source, manifest, _, object, _) = fixture();
    let selection = SelectedManifest {
        manifest: manifest.clone(),
        components: vec!["allowed".into()],
    };
    let original = ModelSources::open(&source, std::slice::from_ref(&selection)).unwrap();
    let target = source.with_extension("corrupt-target");
    let store = Store::ensure(&target).unwrap();
    for path in original.closure_paths(&manifest).unwrap() {
        let copied = target.join(path.strip_prefix(&source).unwrap());
        fs::create_dir_all(copied.parent().unwrap()).unwrap();
        fs::copy(path, copied).unwrap();
    }
    let path = store.blob_path(&object.sha256);
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
    fs::set_permissions(&path, permissions).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .write_at(&[255], 0)
        .unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o444);
    fs::set_permissions(&path, permissions).unwrap();
    drop(original);
    drop(store);
    let mut broker = ModelSources::open(&target, &[selection]).unwrap();
    assert!(broker.verify_selected().is_err());
    drop(broker);
    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(target).unwrap();
}

#[test]
fn empty_selection_never_grants_store_presence_as_authority() {
    let (root, manifest, _, _, _) = fixture();
    let mut broker = ModelSources::open(&root, &[]).unwrap();
    let request = SourceRequest {
        manifest,
        role: SourceRole::Header,
        name: String::new(),
        length: 0,
    };
    assert_eq!(
        broker.read(&request).err().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    let unknown: SourceRequest = serde_json::from_value(serde_json::json!({
        "manifest": request.manifest,
        "role": "future_source",
        "future_field": {"harmless": true}
    }))
    .unwrap();
    assert_eq!(unknown.role, SourceRole::Unknown);
    assert!(ModelSources::open(
        &root,
        &[SelectedManifest {
            manifest: request.manifest.clone(),
            components: vec!["absent".into()]
        }]
    )
    .is_err());
    drop(broker);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn readonly_source_fd_can_outlive_owner_broker() {
    let (root, manifest, data, object, _) = fixture();
    let mut broker = ModelSources::open(
        &root,
        &[SelectedManifest {
            manifest: manifest.clone(),
            components: vec!["allowed".into()],
        }],
    )
    .unwrap();
    let grant = broker
        .read(&SourceRequest {
            manifest,
            role: SourceRole::Object,
            name: object.id(),
            length: object.length,
        })
        .unwrap();
    drop(broker);
    let mut got = vec![0; data.len()];
    grant.file.read_exact_at(&mut got, 0).unwrap();
    assert_eq!(got, data);
    // SAFETY: pure flag query on the live granted descriptor.
    let flags = unsafe { libc::fcntl(std::os::fd::AsRawFd::as_raw_fd(&grant.file), libc::F_GETFL) };
    assert_eq!(flags & libc::O_ACCMODE, libc::O_RDONLY);
    // SAFETY: descriptor flag query, no external mutation or device access.
    let flags = unsafe { libc::fcntl(std::os::fd::AsRawFd::as_raw_fd(&grant.file), libc::F_GETFD) };
    assert_ne!(flags & libc::FD_CLOEXEC, 0);
    drop(grant);
    fs::remove_dir_all(root).unwrap();
}
