//! Native input custody over released TensorFS and the service's sole journal.
use crate::api::{auth::VerifiedActor, backend::InputTreeReceiver, pb};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};
use tensorfs_core::{
    canon,
    catalog::WriterGuard,
    ids::{ObjectRef, StoredDoc},
    manifest::Manifest,
    sha256, source_artifact,
    store::{Fault, Store},
};
use tonic::Status;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct IntakeSpec {
    pub actor: String,
    pub request_id: String,
    pub input_id: String,
    pub manifest_sha256: String,
    pub manifest_length: u64,
    pub manifest_canonical_bytes: Vec<u8>,
    pub content_bytes: u64,
    pub retention_id: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IntakeState {
    pub spec: IntakeSpec,
    pub released: bool,
    pub receipt: Option<Vec<u8>>,
}
/// Engine implements these in its existing SQLite journal. Finish registers the
/// exact NativeByteRetentionRequest in the same transaction as the result receipt.
pub trait IntakeJournal: Send + Sync {
    fn begin_intake(&self, spec: IntakeSpec) -> io::Result<IntakeState>;
    fn finish_intake(
        &self,
        actor: &str,
        retention: &str,
        receipt: Vec<u8>,
    ) -> io::Result<IntakeState>;
    fn abort_intake(&self, actor: &str, retention: &str) -> io::Result<IntakeState>;
}
struct Partial {
    reference: ObjectRef,
    path: PathBuf,
    file: File,
    received: u64,
}
pub struct SourceIntake {
    store: Arc<Store>,
    journal: Arc<dyn IntakeJournal>,
    state: IntakeState,
    manifest: Manifest,
    directory: PathBuf,
    objects: BTreeMap<String, ObjectRef>,
    partials: BTreeMap<String, Partial>,
    _writer: WriterGuard,
}
impl Drop for SourceIntake {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

impl SourceIntake {
    pub fn begin(
        store: Arc<Store>,
        journal: Arc<dyn IntakeJournal>,
        staging: &Path,
        workspace: &str,
        actor: VerifiedActor,
        header: pb::InputTreeImportHeader,
    ) -> Result<Box<dyn InputTreeReceiver>, Status> {
        if header.request_id.is_empty()
            || header.request_id.len() > 128
            || !header
                .request_id
                .bytes()
                .all(|b| (0x20..=0x7e).contains(&b))
            || header.input_id.is_empty()
            || header.input_id.len() > 1024
            || !header.input_id.bytes().all(|b| (0x20..=0x7e).contains(&b))
        {
            return Err(Status::invalid_argument(
                "input subject is outside its typed identifier bound",
            ));
        }
        let reference = header
            .manifest
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("input manifest reference is absent"))?;
        if reference.digest.len() != 32
            || reference.length != header.manifest_canonical_bytes.len() as u64
            || header.manifest_canonical_bytes.is_empty()
            || header.manifest_canonical_bytes.len() > 1 << 20
            || sha256::digest(&header.manifest_canonical_bytes).as_slice() != reference.digest
        {
            return Err(Status::invalid_argument(
                "input manifest bytes do not verify",
            ));
        }
        let manifest = Manifest::parse(&header.manifest_canonical_bytes).map_err(native)?;
        if manifest.header().is_some()
            || manifest
                .entries()
                .iter()
                .any(|(_, entry)| entry.kind() != "file")
        {
            return Err(Status::invalid_argument(
                "input tree must contain ordinary files",
            ));
        }
        let content = manifest
            .entries()
            .iter()
            .try_fold(0u64, |sum, (_, entry)| sum.checked_add(entry.blob().length))
            .ok_or_else(|| Status::invalid_argument("input content length overflow"))?;
        if content != header.content_bytes {
            return Err(Status::invalid_argument(
                "input content count differs from manifest",
            ));
        }
        let mut objects = BTreeMap::new();
        for (_, entry) in manifest.entries() {
            let reference = entry.blob();
            if objects
                .insert(reference.sha256.clone(), reference.clone())
                .is_some_and(|prior: ObjectRef| prior.length != reference.length)
            {
                return Err(Status::invalid_argument(
                    "one input object cannot declare multiple lengths",
                ));
            }
        }
        let actor = sha256::hex(&actor.public_key);
        let retention_id = format!(
            "sha256:{}",
            sha256::hex_digest(&canon::write(&canon::Value::obj(vec![
                ("format", canon::Value::str("cozy.machine.input-intake/1")),
                ("workspace", canon::Value::str(workspace)),
                ("actor", canon::Value::str(&actor)),
                ("request", canon::Value::str(&header.request_id)),
                ("input", canon::Value::str(&header.input_id))
            ])))
        );
        let spec = IntakeSpec {
            actor,
            request_id: header.request_id,
            input_id: header.input_id,
            manifest_sha256: sha256::hex(&reference.digest),
            manifest_length: reference.length,
            manifest_canonical_bytes: header.manifest_canonical_bytes,
            content_bytes: header.content_bytes,
            retention_id,
        };
        let state = journal.begin_intake(spec.clone()).map_err(database)?;
        if state.spec != spec {
            return Err(Status::failed_precondition(
                "input intake changed its immutable subject",
            ));
        }
        // Acquire before trusting existing catalog objects, keeping GC excluded for
        // the whole transfer. Source-artifact import adds its own durable liveness.
        let writer = WriterGuard::acquire(store.root()).map_err(native)?;
        fs::create_dir_all(staging).map_err(database)?;
        let directory = staging.join(
            fs::read_to_string("/proc/sys/kernel/random/uuid")
                .map_err(database)?
                .trim(),
        );
        fs::create_dir(&directory).map_err(database)?;
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).map_err(database)?;
        Ok(Box::new(Self {
            store,
            journal,
            state,
            manifest,
            directory,
            objects,
            partials: BTreeMap::new(),
            _writer: writer,
        }))
    }
    fn result(&self, released: bool) -> Result<pb::NativeByteRetentionResult, Status> {
        let root =
            source_artifact::read(&self.store, &self.state.spec.retention_id).map_err(native)?;
        let source = match root {
            Some(root) if root.complete => Some(pb::NativeByteTreeRef {
                producer_root_id: root.producer.clone(),
                receipt_digest: sha256::digest(&root.receipt().map_err(native)?).to_vec(),
                manifest: Some(pb::Ref {
                    digest: decode_hex(&root.manifest.sha256)?,
                    length: root.manifest.length,
                }),
                content_bytes: self.state.spec.content_bytes,
            }),
            _ => None,
        };
        Ok(pb::NativeByteRetentionResult {
            source,
            retention_id: self.state.spec.retention_id.clone(),
            released,
        })
    }
    fn release_native(&self) -> Result<(), Status> {
        if source_artifact::read(&self.store, &self.state.spec.retention_id)
            .map_err(native)?
            .is_some()
        {
            source_artifact::release(&self.store, &self.state.spec.retention_id).map_err(native)?;
        }
        Ok(())
    }
}
impl InputTreeReceiver for SourceIntake {
    fn blob(&mut self, blob: pb::InputTreeImportBlob) -> Result<(), Status> {
        if self.state.released {
            return Err(Status::failed_precondition(
                "input intake is permanently released",
            ));
        }
        let object = blob
            .object
            .ok_or_else(|| Status::invalid_argument("input object reference absent"))?;
        if object.digest.len() != 32 {
            return Err(Status::invalid_argument("input object digest is invalid"));
        }
        let hash = sha256::hex(&object.digest);
        let expected = self
            .objects
            .get(&hash)
            .ok_or_else(|| Status::invalid_argument("input object is outside declared manifest"))?;
        if object.length != expected.length || blob.data.len() > 1 << 20 {
            return Err(Status::invalid_argument(
                "input object identity or chunk bound differs",
            ));
        }
        let partial = match self.partials.entry(hash.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let path = self.directory.join(&hash);
                let file = OpenOptions::new()
                    .create_new(true)
                    .read(true)
                    .write(true)
                    .mode(0o600)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&path)
                    .map_err(database)?;
                entry.insert(Partial {
                    reference: expected.clone(),
                    path,
                    file,
                    received: 0,
                })
            }
        };
        if blob.offset != partial.received
            || blob.data.len() as u64 > partial.reference.length - partial.received
            || (blob.data.is_empty() && partial.reference.length != 0)
        {
            return Err(Status::failed_precondition(
                "input object chunks must be contiguous within one stream",
            ));
        }
        partial
            .file
            .seek(SeekFrom::Start(partial.received))
            .map_err(database)?;
        partial.file.write_all(&blob.data).map_err(database)?;
        partial.received += blob.data.len() as u64;
        Ok(())
    }
    fn commit(
        self: Box<Self>,
        commit: pb::InputTreeImportCommit,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        if commit.abort || self.state.released {
            self.journal
                .abort_intake(&self.state.spec.actor, &self.state.spec.retention_id)
                .map_err(database)?;
            let result = self.result(true)?;
            self.release_native()?;
            return Ok(result);
        }
        for reference in self.objects.values() {
            if matches!(self.store.record_valid(&reference.sha256),Ok(row) if row.length==reference.length)
            {
                continue;
            }
            let partial = self
                .partials
                .get(&reference.sha256)
                .ok_or_else(|| Status::failed_precondition("input object is incomplete"))?;
            if partial.received != reference.length {
                return Err(Status::failed_precondition("input object is incomplete"));
            }
            partial.file.sync_all().map_err(database)?;
        }
        let files: Vec<_> = self
            .manifest
            .entries()
            .iter()
            .map(|(path, entry)| {
                (
                    path.clone(),
                    self.partials
                        .get(&entry.blob().sha256)
                        .map(|partial| partial.path.clone())
                        .unwrap_or_else(|| self.directory.join(&entry.blob().sha256)),
                )
            })
            .collect();
        source_artifact::import_tree(
            &self.store,
            &self.state.spec.retention_id,
            &self.manifest,
            &files,
            &Fault::default(),
        )
        .map_err(native)?;
        let result = self.result(false)?;
        let committed = self
            .journal
            .finish_intake(
                &self.state.spec.actor,
                &self.state.spec.retention_id,
                result.encode_to_vec(),
            )
            .map_err(database)?;
        if committed.released {
            self.release_native()?;
            return Ok(pb::NativeByteRetentionResult {
                released: true,
                ..result
            });
        }
        if let Some(receipt) = committed.receipt {
            return pb::NativeByteRetentionResult::decode(receipt.as_slice())
                .map_err(|_| Status::data_loss("committed intake receipt is invalid"));
        }
        Err(Status::data_loss(
            "journal acknowledged input without durable receipt",
        ))
    }
}
fn decode_hex(spelling: &str) -> Result<Vec<u8>, Status> {
    (0..32)
        .map(|index| {
            u8::from_str_radix(&spelling[index * 2..index * 2 + 2], 16)
                .map_err(|_| Status::data_loss("native digest is invalid"))
        })
        .collect()
}
fn native(error: impl std::fmt::Display) -> Status {
    Status::failed_precondition(format!("native input custody: {error}"))
}
fn database(error: io::Error) -> Status {
    match error.kind() {
        io::ErrorKind::AlreadyExists => Status::already_exists(error.to_string()),
        io::ErrorKind::PermissionDenied => Status::permission_denied(error.to_string()),
        io::ErrorKind::InvalidInput => Status::invalid_argument(error.to_string()),
        _ => Status::unavailable(error.to_string()),
    }
}
