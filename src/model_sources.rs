//! A session's selected models: each manifest's verified header and the selected components'
//! encoded size. Selection grants authority; the weights come from the sealed host tier, and
//! the header and model assets from here (`ModelSource`): the executor reads no store.
use crate::device_executor::{Answer, Frame};
use crate::os;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read, Write},
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    sync::Arc,
};
use tensorfs_core::{
    catalog::WriterGuard, checkpoint_root, header::Header, ids::ObjectRef, meta::Meta, read,
    store::Store,
};

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
    _custody: Option<SessionCustody>,
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
        Ok(Self {
            store,
            selected,
            _custody: None,
        })
    }

    /// A live executor's selected source closure, independently retained until exact
    /// receiver and supported owned-scope exit. Caller retains the returned Arc in its
    /// DeviceExecutor exact-exit/quarantine resources before exposing any source/fds.
    pub fn open_session_shared(
        store: Arc<Store>,
        selections: &[SelectedManifest],
        directory: &Path,
        receiver: crate::journal::ProcessBirth,
        scope: crate::scope::Recovery,
    ) -> io::Result<Self> {
        Self::open_session_observed(store, selections, directory, receiver, scope, &|_| {})
    }

    fn open_session_observed(
        store: Arc<Store>,
        selections: &[SelectedManifest],
        directory: &Path,
        receiver: crate::journal::ProcessBirth,
        scope: crate::scope::Recovery,
        observe: &dyn Fn(ReaderStage),
    ) -> io::Result<Self> {
        let _writer = WriterGuard::acquire(store.root()).map_err(failure)?;
        let mut sources = Self::open_shared(store.clone(), selections)?;
        let id = tensorfs_core::sha256::hex_digest(uuid::Uuid::new_v4().as_bytes());
        let roots = sources
            .selected
            .iter()
            .map(|(sha256, selection)| ReaderRoot {
                owner: format!(
                    "sha256:{}",
                    tensorfs_core::sha256::hex_digest(format!("{id}:{sha256}").as_bytes())
                ),
                sha256: sha256.clone(),
                length: selection.manifest_length,
            })
            .collect();
        let record = ReaderRecord {
            id,
            receiver,
            scope,
            roots,
        };
        let path = write_reader_record(directory, &record)?;
        sources._custody = Some(SessionCustody {
            store: store.clone(),
            path,
            record,
            drop_release: true,
        });
        observe(ReaderStage::Recorded);
        for root in &sources._custody.as_ref().unwrap().record.roots {
            checkpoint_root::retain_materialized(
                &store,
                &root.owner,
                READER_SOURCE,
                root.reference(),
            )
            .map_err(failure)?;
        }
        observe(ReaderStage::Rooted);
        Ok(sources)
    }

    /// Recover only recorded source readers with strict exact-birth and whole supported
    /// scope exit proof. A corrupt/unreadable obligation is not permission to drop roots.
    pub fn recover_sessions(store: Arc<Store>, directory: &Path) -> io::Result<usize> {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let _writer = WriterGuard::acquire(store.root()).map_err(failure)?;
        let mut released = 0;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(id) = name.to_str().and_then(|name| name.strip_suffix(".json")) else {
                continue;
            };
            tensorfs_core::ids::hex64("source reader record", id).map_err(failure)?;
            let mut file = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(entry.path())?;
            let mut bytes = Vec::new();
            Read::by_ref(&mut file)
                .take(MAX_READER_RECORD as u64)
                .read_to_end(&mut bytes)?;
            if bytes.len() >= MAX_READER_RECORD {
                return Err(failure("source reader record exceeds its bound"));
            }
            let record: ReaderRecord = serde_json::from_slice(&bytes).map_err(failure)?;
            if record.id != id {
                return Err(failure("source reader record names another owner"));
            }
            for root in &record.roots {
                tensorfs_core::ids::prefixed("source reader owner", &root.owner)
                    .map_err(failure)?;
                tensorfs_core::ids::hex64("source reader manifest", &root.sha256)
                    .map_err(failure)?;
                if root.length == 0 {
                    return Err(failure("source reader manifest is empty"));
                }
            }
            let custody = SessionCustody {
                store: store.clone(),
                path: entry.path(),
                record,
                drop_release: false,
            };
            if custody.release_if_ended()? {
                released += 1;
            } else {
                // An owner died between record persistence and root installation. Repair
                // the missing no-gap handoff while the native writer fence is still held.
                for root in &custody.record.roots {
                    checkpoint_root::retain_materialized(
                        &store,
                        &root.owner,
                        READER_SOURCE,
                        root.reference(),
                    )
                    .map_err(failure)?;
                }
            }
            // Explicit recovery does not use the normal last-Arc destructor path.
        }
        Ok(released)
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

#[derive(Clone, Copy)]
enum ReaderStage {
    Recorded,
    Rooted,
}

const MAX_READER_RECORD: usize = 16 << 20;
const READER_SOURCE: &str = "local/machine-reader";

#[derive(Serialize, Deserialize)]
struct ReaderRoot {
    owner: String,
    sha256: String,
    length: u64,
}
impl ReaderRoot {
    fn reference(&self) -> ObjectRef {
        ObjectRef {
            sha256: self.sha256.clone(),
            length: self.length,
        }
    }
}
#[derive(Serialize, Deserialize)]
struct ReaderRecord {
    id: String,
    receiver: crate::journal::ProcessBirth,
    scope: crate::scope::Recovery,
    roots: Vec<ReaderRoot>,
}
impl ReaderRecord {
    fn ended(&self) -> io::Result<bool> {
        if crate::process::process_birth(std::process::id())?.boot_id != self.receiver.boot_id {
            return Ok(true); // Every process from the recorded boot is gone.
        }
        if !crate::process::group_ended(&self.receiver)? {
            return Ok(false);
        }
        self.scope.empty()
    }
}

use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
fn write_reader_record(directory: &Path, record: &ReaderRecord) -> io::Result<PathBuf> {
    fs::create_dir_all(directory)?;
    fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    let path = directory.join(format!("{}.json", record.id));
    let temporary = directory.join(format!("{}.tmp", record.id));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temporary)?;
    let bytes = serde_json::to_vec(record).map_err(failure)?;
    if bytes.len() >= MAX_READER_RECORD {
        return Err(failure("source reader record exceeds its bound"));
    }
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, &path)?;
    File::open(directory)?.sync_all()?;
    Ok(path)
}

struct SessionCustody {
    store: Arc<Store>,
    path: PathBuf,
    record: ReaderRecord,
    drop_release: bool,
}
impl SessionCustody {
    fn release_if_ended(&self) -> io::Result<bool> {
        let _writer = WriterGuard::acquire(self.store.root()).map_err(failure)?;
        if !self.record.ended()? {
            return Ok(false);
        }
        for root in &self.record.roots {
            checkpoint_root::release(&self.store, &root.owner, READER_SOURCE, root.reference())
                .map_err(failure)?;
        }
        match fs::remove_file(&self.path) {
            Ok(()) => File::open(self.path.parent().unwrap())?.sync_all()?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error),
        }
        Ok(true)
    }
}
impl Drop for SessionCustody {
    fn drop(&mut self) {
        if !self.drop_release {
            return;
        }
        if let Err(error) = self.release_if_ended() {
            eprintln!("model source reader remains recorded and rooted: {error}");
        }
    }
}

#[cfg(test)]
mod reader_recovery_tests {
    use super::*;
    use tensorfs_core::{
        dtype::Dtype,
        header::{Part, Tensor},
        manifest::{Draft, Entry},
        store::Fault,
    };

    struct Fixture {
        root: PathBuf,
        store: Arc<Store>,
        manifest: String,
        data: ObjectRef,
    }
    impl Fixture {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("source-recovery-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&root).unwrap();
            let store = Arc::new(Store::init(&root.join("store")).unwrap());
            let bytes = vec![0x33; 2048];
            let data = store
                .put_stream(&mut bytes.as_slice(), None, &Fault::default())
                .unwrap()
                .obj;
            let plain = tensorfs_core::registry::seeds()
                .into_iter()
                .find(|row| row.alias == "plain/1")
                .unwrap()
                .spec;
            let header = Header {
                configs: vec![],
                assets: vec![],
                encodings: vec![plain.clone()],
                components: vec![(
                    "model".into(),
                    vec![(
                        "weight".into(),
                        Tensor {
                            dtype: Dtype::U8,
                            shape: vec![2048],
                            encoding: plain.object_id(),
                            parts: vec![(
                                "value".into(),
                                Part::plan(Dtype::U8, vec![2048], &bytes),
                            )],
                        },
                    )],
                )],
            };
            let header = store
                .put_stream(
                    &mut header.canonical_bytes().unwrap().as_slice(),
                    None,
                    &Fault::default(),
                )
                .unwrap()
                .obj;
            let manifest = store
                .put_manifest(
                    &Draft {
                        entries: vec![("model".into(), Entry::CozyTensors(header))],
                    }
                    .seal()
                    .unwrap(),
                )
                .unwrap()
                .obj
                .id();
            Self {
                root,
                store,
                manifest,
                data,
            }
        }
        fn directory(&self) -> PathBuf {
            self.root.join("readers")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    #[ignore = "subprocess constructor crash-stage fixture"]
    fn inside_recorded_source_creator() {
        let root = PathBuf::from(std::env::var_os("COZY_SOURCE_FIXTURE_ROOT").unwrap());
        let store = Arc::new(Store::open(&root.join("store")).unwrap());
        let manifest = std::env::var("COZY_SOURCE_FIXTURE_MANIFEST").unwrap();
        let receiver =
            serde_json::from_str(&std::env::var("COZY_SOURCE_FIXTURE_RECEIVER").unwrap()).unwrap();
        let scope =
            serde_json::from_str(&std::env::var("COZY_SOURCE_FIXTURE_SCOPE").unwrap()).unwrap();
        let stage = std::env::var("COZY_SOURCE_FIXTURE_STAGE").unwrap();
        let _sources = ModelSources::open_session_observed(
            store,
            &[SelectedManifest {
                manifest,
                components: vec!["model".into()],
            }],
            &root.join("readers"),
            receiver,
            scope,
            &|seen| {
                let exit = matches!(
                    (stage.as_str(), seen),
                    ("recorded", ReaderStage::Recorded) | ("rooted", ReaderStage::Rooted)
                );
                if exit {
                    std::process::exit(0);
                }
            },
        )
        .unwrap();
        panic!("fixture did not exit at its requested constructor stage");
    }

    #[test]
    fn owner_death_after_record_or_roots_repairs_custody_before_independent_gc() {
        for stage in ["recorded", "rooted"] {
            let fixture = Fixture::new();
            let scope =
                crate::scope::Scope::create(&format!("source{}", uuid::Uuid::new_v4().simple()))
                    .unwrap();
            let mut command = std::process::Command::new("sleep");
            command.arg("1000");
            if let Some((name, value)) = scope.environment() {
                command.env(name, value);
            }
            let mut receiver = command.spawn().unwrap();
            scope.adopt(receiver.id()).unwrap();
            let birth = crate::process::process_birth(receiver.id()).unwrap();
            struct EndReceiver(crate::journal::ProcessBirth);
            impl Drop for EndReceiver {
                fn drop(&mut self) {
                    if let Ok(Some(exact)) = crate::process::Exact::open(&self.0) {
                        let _ = exact.kill();
                        let _ = exact.wait();
                    }
                }
            }
            let _cleanup = EndReceiver(birth.clone());
            let recovery = scope.recovery(unsafe { libc::geteuid() });
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "model_sources::reader_recovery_tests::inside_recorded_source_creator",
                    "--ignored",
                ])
                .env("COZY_SOURCE_FIXTURE_ROOT", &fixture.root)
                .env("COZY_SOURCE_FIXTURE_MANIFEST", &fixture.manifest)
                .env(
                    "COZY_SOURCE_FIXTURE_RECEIVER",
                    serde_json::to_string(&birth).unwrap(),
                )
                .env(
                    "COZY_SOURCE_FIXTURE_SCOPE",
                    serde_json::to_string(&recovery).unwrap(),
                )
                .env("COZY_SOURCE_FIXTURE_STAGE", stage)
                .status()
                .unwrap();
            assert!(status.success(), "{stage}");
            assert_eq!(
                ModelSources::recover_sessions(fixture.store.clone(), &fixture.directory())
                    .unwrap(),
                0
            );
            tensorfs_core::gc::collect(fixture.store.root(), false).unwrap();
            assert!(
                fixture.store.object_path(&fixture.data.sha256).exists(),
                "{stage}"
            );
            receiver.kill().unwrap();
            receiver.wait().unwrap();
            scope.end().unwrap();
            match ModelSources::recover_sessions(fixture.store.clone(), &fixture.directory()) {
                Ok(released) => {
                    assert_eq!(released, 1, "{stage}");
                    tensorfs_core::gc::collect(fixture.store.root(), false).unwrap();
                    assert!(
                        !fixture.store.object_path(&fixture.data.sha256).exists(),
                        "{stage}"
                    );
                }
                Err(error) => {
                    assert!(
                        std::env::var_os("COZY_REQUIRE_STRICT_SOURCE_EXIT").is_none(),
                        "{error}"
                    );
                    eprintln!("{stage}: strict exit unavailable; recovery preserves root: {error}");
                }
            }
        }
    }

    #[test]
    fn corrupt_source_recovery_record_is_not_release_authority() {
        let fixture = Fixture::new();
        fs::create_dir_all(fixture.directory()).unwrap();
        fs::write(
            fixture.directory().join(format!("{}.json", "a".repeat(64))),
            b"broken",
        )
        .unwrap();
        assert!(
            ModelSources::recover_sessions(fixture.store.clone(), &fixture.directory()).is_err()
        );
    }
}
