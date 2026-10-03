//! Machine-owned selected model byte exports. Selection grants authority; file identity
//! and native TensorFS planning grant neither another manifest nor another component.
use crate::os;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read, Write},
    os::fd::AsRawFd,
    path::Path,
    sync::Arc,
};
use tensorfs_core::{
    header::Header,
    ids::ObjectRef,
    meta::{Hold, Meta},
    read::{self, Source},
    sha256,
    store::Store,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SelectedManifest {
    pub manifest: String,
    pub components: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceRole {
    Header,
    Asset,
    Object,
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceRequest {
    pub manifest: String,
    pub role: SourceRole,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub length: u64,
}

pub struct SourceGrant {
    pub sha256: String,
    pub length: u64,
    pub file: File,
}

struct Selection {
    header_ref: ObjectRef,
    header: Header,
    allowed_objects: BTreeMap<String, ObjectRef>,
    retained_objects: Vec<ObjectRef>,
    /// The GC hold while the executor reads: two descriptors, never one per object.
    hold: Option<Hold>,
    components: Vec<String>,
    encoded_bytes: u64,
    manifest_length: u64,
}

pub struct ModelSources {
    store: Arc<Store>,
    meta: Meta,
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
        let meta = Meta::open(&store).map_err(failure)?;
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
            let plan = read::plan_for_traversal(&header, &traversal, &components, 4 << 20)
                .map_err(failure)?;
            let encoded_bytes = plan.bytes;
            let mut allowed_objects = BTreeMap::new();
            for item in plan.items {
                if let Source::Object(range) = item.source {
                    allowed_objects.insert(range.obj.sha256.clone(), range.obj);
                }
            }
            // Assets are declared by the authorized manifest; ordinary snapshot siblings
            // do not become model byte authority.
            let mut retained_objects = allowed_objects.clone();
            retained_objects.insert(header_ref.sha256.clone(), header_ref.clone());
            for (_, asset) in &header.assets {
                for object in &asset.segments {
                    retained_objects.insert(object.sha256.clone(), object.clone());
                }
            }
            if selected
                .insert(
                    manifest,
                    Selection {
                        header_ref,
                        header,
                        allowed_objects,
                        retained_objects: retained_objects.into_values().collect(),
                        hold: None,
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
        Ok(Self {
            store,
            meta,
            selected,
        })
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

    /// Native paths of the explicitly selected snapshot/model closure, for owned transfer.
    /// The caller does not reconstruct TensorFS's shard layout or acquire another model.
    pub fn closure_paths(&self, manifest: &str) -> io::Result<Vec<std::path::PathBuf>> {
        let manifest = digest(manifest)?;
        let selection = self.selected.get(manifest).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "manifest is outside pilot selection",
            )
        })?;
        let mut paths = vec![self.store.manifest_path(manifest)];
        paths.extend(
            selection
                .retained_objects
                .iter()
                .map(|object| self.store.blob_path(&object.sha256)),
        );
        Ok(paths)
    }

    /// Verify/admit all selected immutable blobs in one linked-core process. Useful after
    /// an owned store was initialized and exact CAS paths copied without a foreign catalog.
    pub fn verify_selected(&mut self) -> io::Result<(usize, u64)> {
        let mut objects = BTreeMap::new();
        for selection in self.selected.values() {
            for object in &selection.retained_objects {
                objects.insert(object.sha256.clone(), object.clone());
            }
        }
        let bytes = objects.values().map(|object| object.length).sum();
        for (manifest, selection) in &mut self.selected {
            if selection.hold.is_none() {
                // Verify every object once; keep only the hold, not a descriptor per object.
                let (lease, _) = read::acquire(
                    &self.store,
                    &self.meta,
                    manifest,
                    selection.retained_objects.clone(),
                )
                .map_err(failure)?;
                selection.hold = Some(self.meta.acquire_hold("read").map_err(failure)?);
                lease.release(&self.meta).map_err(failure)?;
            }
        }
        Ok((objects.len(), bytes))
    }

    pub fn read(&mut self, request: &SourceRequest) -> io::Result<SourceGrant> {
        let manifest = digest(&request.manifest)?;
        let selection = self.selected.get_mut(manifest).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "manifest is outside pilot selection",
            )
        })?;
        if matches!(request.role, SourceRole::Unknown) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "unknown model source role",
            ));
        }
        if request.role == SourceRole::Header {
            let file = self
                .store
                .open_verified(&selection.header_ref.sha256)
                .map_err(failure)?
                .into_file();
            return Ok(SourceGrant {
                sha256: selection.header_ref.sha256.clone(),
                length: selection.header_ref.length,
                file,
            });
        }
        // Verify authority before expensive lease acquisition, even when bytes are cached.
        let object = if request.role == SourceRole::Object {
            let id = digest(&request.name)?;
            let object = selection.allowed_objects.get(id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "object is outside selected components",
                )
            })?;
            if request.length != object.length {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "object length differs from selection",
                ));
            }
            Some(object.clone())
        } else {
            if !selection
                .header
                .assets
                .iter()
                .any(|(name, _)| name == &request.name)
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "asset is outside selected manifest",
                ));
            }
            None
        };
        if selection.hold.is_none() {
            selection.hold = Some(self.meta.acquire_hold("read").map_err(failure)?);
        }
        if let Some(object) = object {
            let file = self
                .store
                .open_verified(&object.sha256)
                .map_err(failure)?
                .into_file();
            return Ok(SourceGrant {
                sha256: object.sha256,
                length: object.length,
                file,
            });
        }
        let asset = selection
            .header
            .assets
            .iter()
            .find(|(name, _)| name == &request.name)
            .unwrap();
        // A lease over this asset's own segments, for this read only.
        let (lease, _) = read::acquire(&self.store, &self.meta, manifest, asset.1.segments.clone())
            .map_err(failure)?;
        let bytes = read::read_asset(&lease, &request.name, &asset.1, asset.1.logical_length);
        lease.release(&self.meta).map_err(failure)?;
        let bytes = bytes.map_err(failure)?;
        let mut writable = os::memfd()?;
        writable.write_all(&bytes)?;
        os::seal(&writable)?;
        let file = File::open(format!("/proc/self/fd/{}", writable.as_raw_fd()))?;
        Ok(SourceGrant {
            sha256: sha256::hex_digest(&bytes),
            length: bytes.len() as u64,
            file,
        })
    }
}

impl Drop for ModelSources {
    fn drop(&mut self) {
        for selection in self.selected.values_mut() {
            if let Some(hold) = selection.hold.take() {
                if let Err(error) = hold.release(&self.meta) {
                    eprintln!("model source hold release: {error}");
                }
            }
        }
    }
}
