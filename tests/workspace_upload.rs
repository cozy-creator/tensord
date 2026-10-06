use cozy_machine::api::{auth::VerifiedActor, domain, workspaces::WorkspaceUploads};
use std::{fs, io::Read, path::PathBuf, sync::Arc};
use tensorfs_core::store::Store;

struct Area(PathBuf);
impl Area {
    fn new() -> Self {
        let uuid = fs::read_to_string("/proc/sys/kernel/random/uuid").unwrap();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/workspace-tests")
            .join(uuid.trim());
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn uploads(&self) -> Arc<WorkspaceUploads> {
        WorkspaceUploads::open(
            &self.0.join("uploads"),
            Arc::new(Store::ensure(&self.0.join("store")).unwrap()),
        )
        .unwrap()
    }
}
impl Drop for Area {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn actor(key: u8) -> VerifiedActor {
    VerifiedActor {
        public_key: [key; 32],
    }
}
fn make_header(operation: &str, name: &str, bytes: &[u8]) -> domain::LocalPackageUploadHeader {
    domain::LocalPackageUploadHeader {
        operation_id: operation.into(),
        file: Some(domain::LocalPackageFileRef {
            filename: name.into(),
            length: bytes.len() as u64,
            digest: if name == "source.tar" {
                vec![]
            } else {
                tensorfs_core::sha256::digest(bytes).to_vec()
            },
        }),
    }
}
fn selected(header: &domain::LocalPackageUploadHeader) -> domain::DesiredLocalPackageSet {
    domain::DesiredLocalPackageSet {
        operation_id: header.operation_id.clone(),
        package: Some(domain::DevelopmentPackage {
            package: "local/fixture".into(),
            release: "0.0.1".into(),
            installation_id: "install-fixture".into(),
        }),
        files: vec![header.file.clone().unwrap()],
        source_archive: if header.file.as_ref().unwrap().filename == "source.tar" {
            "source.tar".into()
        } else {
            String::new()
        },
        ..Default::default()
    }
}
fn archive(extra: &str, kind: tar::EntryType) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut out);
        for (name, bytes, ty) in [
            (
                "pyproject.toml",
                b"[project]\nname='fixture'\nversion='0.0.1'\n".as_slice(),
                tar::EntryType::file(),
            ),
            (
                "uv.lock",
                b"version = 1\n".as_slice(),
                tar::EntryType::file(),
            ),
            (
                extra,
                b"raise RuntimeError('must never import for description')\n".as_slice(),
                kind,
            ),
        ] {
            let mut h = tar::Header::new_gnu();
            h.set_size(bytes.len() as u64);
            h.set_mode(0o644);
            h.set_entry_type(ty);
            h.set_cksum();
            builder.append_data(&mut h, name, bytes).unwrap();
        }
        builder.finish().unwrap();
    }
    out
}

#[test]
fn resumed_prefix_is_durable_owner_scoped_and_root_descriptor_immutable() {
    let area = Area::new();
    let uploads = area.uploads();
    let bytes = b"a real checksummed carrier, not executed by ingress";
    let header = make_header("resume", "fixture-0.0.1-py3-none-any.whl", bytes);
    let mut session = uploads.begin(actor(1), &header).unwrap();
    session
        .append(domain::LocalPackageUploadChunk {
            offset: 0,
            data: bytes[..8].to_vec(),
        })
        .unwrap();
    assert_eq!(session.received(), 8);
    assert!(uploads.begin(actor(1), &header).is_err());
    assert_eq!(uploads.begin(actor(2), &header).unwrap().received(), 0);
    drop(session);
    drop(uploads);
    let uploads = area.uploads();
    let mut session = uploads.begin(actor(1), &header).unwrap();
    assert_eq!(session.received(), 8);
    session
        .append(domain::LocalPackageUploadChunk {
            offset: 8,
            data: bytes[8..].to_vec(),
        })
        .unwrap();
    assert!(session.verified());
    drop(session);
    let replay = uploads.begin(actor(1), &header).unwrap();
    assert!(replay.verified());
    drop(replay);
    let package = uploads
        .package(actor(1), &selected(&header))
        .unwrap()
        .unwrap();
    let mut read = vec![];
    package.files[0]
        .open()
        .unwrap()
        .read_to_end(&mut read)
        .unwrap();
    assert_eq!(read, bytes);
    assert!(uploads
        .package(actor(2), &selected(&header))
        .unwrap()
        .is_none());
    let mut changed = selected(&header);
    changed.package.as_mut().unwrap().installation_id = "other-generation".into();
    assert!(uploads.package(actor(1), &changed).is_err());
    let mut changed = header.clone();
    changed.file.as_mut().unwrap().length += 1;
    assert!(uploads.begin(actor(1), &changed).is_err());
}

#[test]
fn source_archive_is_statically_inspected_and_imported_into_tensorfs() {
    let area = Area::new();
    let uploads = area.uploads();
    let bytes = archive("fixture.py", tar::EntryType::file());
    let header = make_header("static-source", "source.tar", &bytes);
    let mut session = uploads.begin(actor(1), &header).unwrap();
    session
        .append(domain::LocalPackageUploadChunk {
            offset: 0,
            data: bytes.clone(),
        })
        .unwrap();
    assert!(session.verified());
    drop(session);
    let package = uploads
        .package(actor(1), &selected(&header))
        .unwrap()
        .unwrap();
    let mut read = Vec::new();
    package.files[0]
        .open()
        .unwrap()
        .read_to_end(&mut read)
        .unwrap();
    assert_eq!(read, bytes);
    let native = Store::open(&area.0.join("store")).unwrap();
    assert!(native.record_valid(&package.files[0].object.sha256).is_ok());
    assert_eq!(package.root.source_archive, "source.tar");
    uploads
        .release_after_install(actor(1), "static-source")
        .unwrap();
    assert!(!area
        .0
        .join("uploads")
        .join(tensorfs_core::sha256::hex(&actor(1).public_key))
        .join("static-source")
        .exists());
}

#[test]
fn corrupt_and_unsafe_carriers_never_acknowledge_verified_custody() {
    let area = Area::new();
    let uploads = area.uploads();
    let good = b"trusted wheel bytes";
    let header = make_header("bad-digest", "fixture-0.0.1-py3-none-any.whl", good);
    let mut session = uploads.begin(actor(1), &header).unwrap();
    assert!(session
        .append(domain::LocalPackageUploadChunk {
            offset: 0,
            data: vec![0; good.len()]
        })
        .is_err());
    assert!(!session.verified());
    drop(session);
    assert!(uploads
        .package(actor(1), &selected(&header))
        .unwrap()
        .is_none());
    for (index, bytes) in [
        archive("uv.lock", tar::EntryType::file()),
        archive("link", tar::EntryType::symlink()),
    ]
    .into_iter()
    .enumerate()
    {
        let header = make_header(&format!("unsafe-{index}"), "source.tar", &bytes);
        let mut session = uploads.begin(actor(1), &header).unwrap();
        assert!(session
            .append(domain::LocalPackageUploadChunk {
                offset: 0,
                data: bytes
            })
            .is_err());
        assert!(!session.verified());
    }
    let mut invalid = header.clone();
    invalid.operation_id = "../escape".into();
    assert!(uploads.begin(actor(1), &invalid).is_err());
    invalid.operation_id = "filename".into();
    invalid.file.as_mut().unwrap().filename = "../escape.whl".into();
    assert!(uploads.begin(actor(1), &invalid).is_err());
}
