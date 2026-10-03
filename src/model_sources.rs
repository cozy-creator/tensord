//! A session's selected models: each manifest's verified header and the selected components'
//! encoded size. Selection grants authority; the weights come from the sealed host tier.
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Read},
    path::Path,
    sync::Arc,
};
use tensorfs_core::{header::Header, ids::ObjectRef, read, store::Store};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SelectedManifest {
    pub manifest: String,
    pub components: Vec<String>,
}

struct Selection {
    header: Header,
    components: Vec<String>,
    encoded_bytes: u64,
    manifest_length: u64,
}

pub struct ModelSources {
    selected: BTreeMap<String, Selection>,
}

fn failure(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

fn digest(value: &str) -> io::Result<&str> {
    let raw = value.strip_prefix("sha256:").unwrap_or(value);
    if raw.len() != 64
        || !raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model identity must be SHA-256",
        ));
    }
    Ok(raw)
}

impl ModelSources {
    pub fn open(root: &Path, selections: &[SelectedManifest]) -> io::Result<Self> {
        Self::open_shared(Arc::new(Store::open(root).map_err(failure)?), selections)
    }

    /// Public service and its upload/custody paths use the same owned store.
    pub fn open_shared(store: Arc<Store>, selections: &[SelectedManifest]) -> io::Result<Self> {
        let mut selected = BTreeMap::new();
        for selection in selections {
            let manifest = digest(&selection.manifest)?.to_owned();
            if selection.components.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "model selection names components",
                ));
            }
            let length = fs::metadata(store.manifest_path(&manifest))?.len();
            let snapshot = store
                .read_manifest(&ObjectRef {
                    sha256: manifest.clone(),
                    length,
                })
                .map_err(failure)?;
            let header_ref = snapshot
                .header()
                .ok_or_else(|| failure("selected snapshot has no model header"))?
                .clone();
            let mut header_bytes = Vec::new();
            store
                .open_verified(&header_ref.sha256)
                .map_err(failure)?
                .into_file()
                .read_to_end(&mut header_bytes)?;
            if header_bytes.len() as u64 != header_ref.length {
                return Err(failure("selected header length differs from its manifest"));
            }
            let header = Header::parse(&header_bytes).map_err(failure)?;
            let wanted: BTreeSet<&str> = selection.components.iter().map(String::as_str).collect();
            let mut traversal = Vec::new();
            for component in &wanted {
                let tensors = header
                    .components
                    .iter()
                    .find(|(name, _)| name == component)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            format!("selected component {component} is absent"),
                        )
                    })?;
                traversal.extend(
                    tensors
                        .1
                        .iter()
                        .map(|(key, _)| ((*component).to_owned(), key.clone())),
                );
            }
            let components: Vec<String> = wanted.into_iter().map(str::to_owned).collect();
            let encoded_bytes = read::plan_for_traversal(&header, &traversal, &components, 4 << 20)
                .map_err(failure)?
                .bytes;
            if selected
                .insert(
                    manifest,
                    Selection {
                        header,
                        components,
                        encoded_bytes,
                        manifest_length: length,
                    },
                )
                .is_some()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate manifest selection",
                ));
            }
        }
        Ok(Self { selected })
    }

    /// Selected encoded source size, not GPU-resident/allocator memory. The SDK
    /// measures its real device footprint; callers cannot infer a fit from this.
    pub fn selected_facts(&self, manifest: &str) -> io::Result<(Vec<String>, u64, u64)> {
        let selected = self.selected.get(digest(manifest)?).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "manifest is outside selection",
            )
        })?;
        Ok((
            selected.components.clone(),
            selected.encoded_bytes,
            selected.manifest_length,
        ))
    }

    pub fn authorized_header(&self, manifest: &str) -> io::Result<Header> {
        self.selected
            .get(digest(manifest)?)
            .map(|selection| selection.header.clone())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "manifest is outside selection",
                )
            })
    }
}
