//! Stored metadata and assets for a session's selected models. TensorD supplies each verified
//! header, declared asset and selected component's encoded byte count through TensorFS.
//! `ModelSource` returns sealed read-only descriptors; HostTier supplies CPU weight buffers
//! and object descriptors. Runtime parses the metadata, constructs models and performs GPU
//! transfers/mappings with its own TensorFS plane. Selection remains the access boundary.
use crate::device_executor::{Answer, Frame};
use crate::os;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read, Write},
    os::fd::AsRawFd,
    path::Path,
    sync::Arc,
};
use tensorfs_core::{header::Header, ids::ObjectRef, meta::Meta, read, store::Store};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SelectedManifest {
    pub manifest: String,
    pub components: Vec<String>,
}

struct Selection {
    header: Header,
    header_bytes: Vec<u8>,
    components: Vec<String>,
    encoded_bytes: u64,
    manifest_length: u64,
}

pub struct ModelSources {
    store: Arc<Store>,
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
                        header_bytes,
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
        Ok(Self { store, selected })
    }

    /// Selected encoded source size, separate from prepared CPU-buffer and GPU residency.
    /// Runtime measures its real device footprint; callers cannot infer a fit from this.
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

    /// One selected model's header (`name` empty) or one header-declared asset, verified:
    /// assets through a read lease over their own objects only.
    pub fn source(&self, manifest: &str, name: &str) -> io::Result<Vec<u8>> {
        let manifest = digest(manifest)?;
        let selection = self.selected.get(manifest).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "manifest is outside selection",
            )
        })?;
        if name.is_empty() {
            return Ok(selection.header_bytes.clone());
        }
        let (_, asset) = selection
            .header
            .assets
            .iter()
            .find(|(asset, _)| asset == name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "asset is outside the selected manifest",
                )
            })?;
        let meta = Meta::open(&self.store).map_err(failure)?;
        let (lease, _) =
            read::acquire(&self.store, &meta, manifest, asset.segments.clone()).map_err(failure)?;
        let bytes = read::read_asset(&lease, name, asset, asset.logical_length);
        lease.release(&meta).map_err(failure)?;
        bytes.map_err(failure)
    }

    /// Answer a `model_source` request: the bytes in a sealed memfd, with their digest.
    pub fn serve(&self, frame: &Frame) -> io::Result<(Answer, File)> {
        let bytes = self.source(&frame.manifest, &frame.name)?;
        let mut file = os::memfd()?;
        file.write_all(&bytes)?;
        os::seal(&file)?;
        let file = File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
        let mut answer = Answer::unavailable(frame.seq);
        (answer.ok, answer.held) = (true, true);
        answer.code.clear();
        answer.detail.clear();
        answer.sha256 = format!("{:x}", Sha256::digest(&bytes));
        answer.length = bytes.len() as u64;
        Ok((answer, file))
    }
}
