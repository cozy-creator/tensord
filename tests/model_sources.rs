//! A session's model selection: its authority, its headers and its encoded size, and the
//! header and assets it serves an executor that reads no store.
use tensord::model_sources::{ModelSources, SelectedManifest};
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    os::{fd::AsRawFd, unix::fs::FileExt},
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
fn a_selection_grants_its_headers_and_sizes_and_nothing_else() {
    let (root, manifest, data, _, _) = fixture();
    let selected = SelectedManifest {
        manifest: manifest.clone(),
        components: vec!["allowed".into()],
    };
    let sources = ModelSources::open(&root, std::slice::from_ref(&selected)).unwrap();
    let (components, encoded, _) = sources.selected_facts(&manifest).unwrap();
    assert_eq!(components, vec!["allowed".to_string()]);
    assert!(encoded >= data.len() as u64, "{encoded}");
    assert!(sources
        .authorized_header(&manifest)
        .unwrap()
        .components
        .iter()
        .any(|(c, _)| c == "allowed"));
    // Store presence is no authority: an unselected manifest, an absent component.
    let none = ModelSources::open(&root, &[]).unwrap();
    assert_eq!(
        none.authorized_header(&manifest).err().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    let absent = SelectedManifest {
        manifest,
        components: vec!["absent".into()],
    };
    assert!(ModelSources::open(&root, &[absent]).is_err());
    fs::remove_dir_all(root).unwrap();
}

fn frame(manifest: &str, name: &str) -> tensord::device_executor::Frame {
    serde_json::from_value(serde_json::json!({
        "event": "request", "seq": 7, "kind": "model_source", "manifest": manifest, "name": name,
    }))
    .unwrap()
}

/// What `model_source` hands an executor: the header and each asset, verified, in a sealed
/// read-only memfd with its digest; nothing outside the selection.
#[test]
fn the_selection_serves_its_header_and_assets_sealed_and_nothing_else() {
    let (root, manifest, _, _, _) = fixture();
    let selected = SelectedManifest {
        manifest: manifest.clone(),
        components: vec!["allowed".into()],
    };
    let sources = ModelSources::open(&root, std::slice::from_ref(&selected)).unwrap();
    for (name, want) in [
        ("", None),
        ("tokenizer/vocab.json", Some(&b"static tokenizer asset"[..])),
    ] {
        let (answer, file) = sources.serve(&frame(&manifest, name)).unwrap();
        let mut got = vec![0; answer.length as usize];
        file.read_exact_at(&mut got, 0).unwrap();
        assert!(answer.ok && answer.held);
        assert_eq!(answer.sha256, format!("{:x}", Sha256::digest(&got)));
        match want {
            Some(bytes) => assert_eq!(got, bytes),
            None => assert!(
                tensorfs_core::header::Header::parse(&got).is_ok(),
                "the header"
            ),
        }
        // SAFETY: scalar fcntl queries on a descriptor we hold.
        unsafe {
            assert_eq!(
                libc::fcntl(file.as_raw_fd(), libc::F_GETFL) & libc::O_ACCMODE,
                libc::O_RDONLY
            );
            assert_eq!(
                libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) & 0x8,
                0x8,
                "write-sealed"
            );
        }
    }
    assert!(
        sources.serve(&frame(&manifest, "../tfs.sqlite")).is_err(),
        "only declared assets"
    );
    let other = "sha256:".to_string() + &"0".repeat(64);
    assert!(
        sources.serve(&frame(&other, "")).is_err(),
        "only selected manifests"
    );
    fs::remove_dir_all(root).unwrap();
}
