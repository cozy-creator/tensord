//! The models this machine's store holds, for Status: each repository with what it weighs and
//! its retained checkpoints, read from the store's own documents (TensorFS `storage`). A
//! listing is read again only when a repository document changed.
use crate::api::v1;
use serde::Deserialize;
use std::{
    fs, io,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tensorfs_core::storage;

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Listing {
    pub models: Vec<v1::Model>,
    /// Every held model's bytes together, shared content counted once.
    pub bytes: u64,
}

/// The repository documents a listing was read from: path, modification time and length.
type Seen = Vec<(PathBuf, i64, i64, u64)>;

static LAST: Mutex<Option<(PathBuf, Seen, Listing)>> = Mutex::new(None);

/// The store at `root`'s held models. A store whose documents cannot be read just now (a
/// repository being written) lists nothing until they change, and says why once.
pub fn listing(root: &Path) -> Listing {
    let Ok(seen) = seen(root) else {
        return Listing::default();
    };
    let mut last = LAST.lock().unwrap();
    if let Some((at, was, listing)) = last.as_ref() {
        if at == root && *was == seen {
            return listing.clone();
        }
    }
    let listing = match seen.is_empty() {
        true => Listing::default(),
        false => read(root).unwrap_or_else(|error| {
            eprintln!("tensord: held models unreadable: {error}");
            Listing::default()
        }),
    };
    *last = Some((root.into(), seen, listing.clone()));
    listing
}

fn seen(root: &Path) -> io::Result<Seen> {
    let mut seen = Seen::new();
    let orgs = match fs::read_dir(root.join("repos")) {
        Ok(orgs) => orgs,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(seen),
        Err(error) => return Err(error),
    };
    for org in orgs {
        for document in fs::read_dir(org?.path())? {
            let document = document?;
            let meta = document.metadata()?;
            seen.push((document.path(), meta.mtime(), meta.mtime_nsec(), meta.len()));
        }
    }
    seen.sort();
    Ok(seen)
}

fn read(root: &Path) -> io::Result<Listing> {
    #[derive(Deserialize)]
    struct Row {
        org: String,
        name: String,
        #[serde(default)]
        version: String,
        #[serde(default)]
        lane: String,
        manifest_sha256: String,
        #[serde(default)]
        release_yanked: bool,
    }
    // Usage needs the blob inventory (a walk of the store's blobs); so only on a change.
    let usage = storage::Census::open(root)
        .and_then(|census| census.usage())
        .map_err(io::Error::other)?;
    let mut models: Vec<v1::Model> = usage
        .repos
        .iter()
        .map(|repo| v1::Model {
            repository: format!("{}/{}", repo.org, repo.name),
            total_bytes: repo.bytes_total,
            unique_bytes: repo.bytes_unique,
            checkpoints: vec![],
        })
        .collect();
    for line in storage::repository_release_lines(root).map_err(io::Error::other)? {
        let row: Row = serde_json::from_slice(&line)?;
        let repository = format!("{}/{}", row.org, row.name);
        if let Some(model) = models.iter_mut().find(|m| m.repository == repository) {
            model.checkpoints.push(v1::Checkpoint {
                release: row.version, // empty for a local alias
                lane: row.lane,
                manifest: format!("sha256:{}", row.manifest_sha256),
                yanked: row.release_yanked,
            });
        }
    }
    Ok(Listing {
        models,
        bytes: usage.bytes_total,
    })
}

/// A store with two models that share one blob: `acme/base` (release 1.0.0, lane fp8) and the
/// local alias `local/mine`. Tests of the listing and of Status build on it.
pub fn fixture(root: &Path) -> io::Result<()> {
    use tensorfs_core::{
        ids::{Doc, ObjectRef},
        manifest::Manifest,
        repository::{Mutation, ReleaseLane, RepositoryName},
        store::{Fault, Store},
    };
    let store = Store::init(root).map_err(io::Error::other)?;
    let blob = |body: &[u8]| -> io::Result<ObjectRef> {
        let want = ObjectRef::of(body);
        store
            .put_stream(&mut &body[..], Some(&want), &Fault::default())
            .map_err(io::Error::other)?;
        Ok(want)
    };
    let (shared, base, mine) = (blob(&[1; 1000])?, blob(&[2; 300])?, blob(&[3; 50])?);
    let manifest = |files: Vec<(String, ObjectRef)>| -> io::Result<ObjectRef> {
        let manifest = Manifest::from_files(files).map_err(io::Error::other)?;
        Ok(store.put_manifest(&manifest).map_err(io::Error::other)?.obj)
    };
    let acme = RepositoryName::new("acme", "base").map_err(io::Error::other)?;
    let checkpoint = manifest(vec![("a".into(), shared.clone()), ("b".into(), base)])?;
    let put = Mutation::PutCheckpoint {
        repo: acme.clone(),
        manifest: checkpoint.clone(),
    };
    let held = store
        .apply_repository(None, &put, &Fault::default())
        .map_err(io::Error::other)?
        .expect("a repository");
    let release = Mutation::UpdateRelease {
        expected_revision: 0,
        repo: acme,
        remove: vec![],
        set: vec![ReleaseLane {
            lane: "fp8".into(),
            manifest: checkpoint,
            extra: Default::default(),
        }],
        version: "1.0.0".into(),
    };
    store
        .apply_repository(Some(&held.canonical_bytes()), &release, &Fault::default())
        .map_err(io::Error::other)?;
    let local = Mutation::ReplaceLocal {
        repo: RepositoryName::new("local", "mine").map_err(io::Error::other)?,
        manifest: manifest(vec![("a".into(), shared), ("c".into(), mine)])?,
        version: tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(b"selection")),
    };
    store
        .apply_repository(None, &local, &Fault::default())
        .map_err(io::Error::other)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_each_model_once_and_reads_again_only_on_change() {
        let root = std::env::temp_dir().join(format!("held-models-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        assert_eq!(listing(&root), Listing::default(), "no store, no models");
        fixture(&root).unwrap();
        let held = listing(&root);
        let rows: Vec<_> = held
            .models
            .iter()
            .map(|m| (m.repository.as_str(), m.total_bytes, m.unique_bytes))
            .collect();
        assert_eq!(rows, [("acme/base", 1300, 300), ("local/mine", 1050, 50)]);
        assert_eq!(held.bytes, 1350, "the shared blob counts once");
        let release = &held.models[0].checkpoints[0];
        assert_eq!(
            (release.release.as_str(), release.lane.as_str()),
            ("1.0.0", "fp8")
        );
        assert!(release.manifest.starts_with("sha256:") && !release.yanked);
        assert_eq!(
            held.models[1].checkpoints[0].release, "",
            "a local alias has no release"
        );
        // Unchanged documents answer from the last read; a removed repository is seen.
        assert_eq!(listing(&root), held);
        fs::remove_file(root.join("repos/local/mine.json")).unwrap();
        assert_eq!(listing(&root).models.len(), 1);
        fs::remove_dir_all(&root).unwrap();
    }
}
