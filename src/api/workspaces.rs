//! Deployed resumable package carriers, scoped by verified owner key. No execution journal.
use super::{auth::VerifiedActor, pb};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};
use tensorfs_core::{
    ids::ObjectRef,
    manifest::{Draft, Entry},
    sha256, source_artifact,
    store::{Fault, Store, VerifiedFile},
};
use tonic::Status;

const MAX_FILES: usize = 129;
const MAX_SET_BYTES: u64 = 1 << 30;
const MAX_WHEEL_BYTES: u64 = 512 << 20;
const MAX_CHUNK: usize = 1 << 20;

pub struct WorkspaceUploads {
    root: PathBuf,
    store: Arc<Store>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Carrier {
    filename: String,
    length: u64,
    #[serde(default)]
    digest: Vec<u8>,
}
impl From<&pb::LocalPackageFileRef> for Carrier {
    fn from(file: &pb::LocalPackageFileRef) -> Self {
        Self {
            filename: file.filename.clone(),
            length: file.length,
            digest: file.digest.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    carrier: Carrier,
    object: Option<StoredObject>,
    retention: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredObject {
    pub sha256: String,
    pub length: u64,
}
impl StoredObject {
    pub fn reference(&self) -> ObjectRef {
        ObjectRef {
            sha256: self.sha256.clone(),
            length: self.length,
        }
    }
}

pub struct UploadedFile {
    pub filename: String,
    pub object: StoredObject,
    store: Arc<Store>,
}
impl UploadedFile {
    /// Verify and hold the exact inode through a descriptor, never resolve a caller path.
    pub fn open(&self) -> Result<VerifiedFile, Status> {
        self.store
            .open_verified(&self.object.sha256)
            .map_err(storage)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootSet {
    pub operation_id: String,
    pub package: String,
    pub release: String,
    pub installation_id: String,
    #[serde(default)]
    pub python_requires: String,
    #[serde(default)]
    pub python_version: String,
    #[serde(default)]
    pub source_archive: String,
    #[serde(default)]
    pub dependency_requirements: Vec<u8>,
    files: Vec<Carrier>,
}
pub struct UploadedPackage {
    pub root: RootSet,
    pub files: Vec<UploadedFile>,
}

pub struct UploadSession {
    uploads: Arc<WorkspaceUploads>,
    directory: PathBuf,
    record: Record,
    record_path: PathBuf,
    payload: File,
    received: u64,
    _operation_lock: File,
}

impl WorkspaceUploads {
    pub fn open(root: &Path, store: Arc<Store>) -> Result<Arc<Self>, Status> {
        fs::create_dir_all(root).map_err(disk)?;
        let metadata = fs::symlink_metadata(root).map_err(disk)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(Status::failed_precondition(
                "upload root must be a real owned directory",
            ));
        }
        fs::set_permissions(root, fs::Permissions::from_mode(0o700)).map_err(disk)?;
        sync_dir(root)?;
        Ok(Arc::new(Self {
            root: root.to_owned(),
            store,
        }))
    }

    pub fn begin(
        self: &Arc<Self>,
        actor: VerifiedActor,
        header: &pb::LocalPackageUploadHeader,
    ) -> Result<UploadSession, Status> {
        valid_operation(&header.operation_id)?;
        let carrier = Carrier::from(
            header
                .file
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("upload requires one file header"))?,
        );
        valid_carrier(&carrier)?;
        let directory = self.directory(actor, &header.operation_id)?;
        let operation_lock = lock(&directory)?;
        let record_path = directory.join(format!("{}.json", carrier.filename));
        let record = match read_record(&record_path)? {
            Some(record) if record.carrier == carrier => record,
            Some(_) => {
                return Err(Status::failed_precondition(
                    "upload identity changed under this operation and owner",
                ))
            }
            None => {
                let prior = records(&directory)?;
                let total = prior
                    .iter()
                    .try_fold(carrier.length, |sum, r| sum.checked_add(r.carrier.length));
                if prior.len() >= MAX_FILES || total.is_none_or(|sum| sum > MAX_SET_BYTES) {
                    return Err(Status::invalid_argument(
                        "captured package roster exceeds the deployed transfer bound",
                    ));
                }
                // A lifecycle hold name, never a source fingerprint or peer equality gate.
                let retention = format!(
                    "sha256:{}",
                    sha256::hex_digest(
                        format!(
                            "cozy.machine.package-carrier/1\0{}\0{}\0{}",
                            sha256::hex(&actor.public_key),
                            header.operation_id,
                            carrier.filename
                        )
                        .as_bytes()
                    )
                );
                let record = Record {
                    carrier,
                    object: None,
                    retention,
                };
                atomic_record(&record_path, &record)?;
                record
            }
        };
        let payload = match &record.object {
            Some(object) => self.store.open_nofollow(&object.sha256).map_err(storage)?,
            None => OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(directory.join(format!("{}.partial", record.carrier.filename)))
                .map_err(disk)?,
        };
        if !payload.metadata().map_err(disk)?.is_file() {
            return Err(Status::failed_precondition(
                "upload payload must be a regular file",
            ));
        }
        let received = payload.metadata().map_err(disk)?.len();
        if received > record.carrier.length {
            return Err(Status::failed_precondition(
                "retained upload prefix exceeds its accepted length",
            ));
        }
        payload.sync_all().map_err(disk)?;
        sync_dir(&directory)?;
        let mut session = UploadSession {
            uploads: self.clone(),
            directory,
            record,
            record_path,
            payload,
            received,
            _operation_lock: operation_lock,
        };
        if session.record.object.is_some() {
            session.verify_retained()?;
        } else if received == session.record.carrier.length {
            session.finish()?;
        }
        Ok(session)
    }

    pub fn package(
        &self,
        actor: VerifiedActor,
        selected: &pb::DesiredLocalPackageSet,
    ) -> Result<Option<UploadedPackage>, Status> {
        let root = root_set(selected)?;
        let directory = self.directory(actor, &root.operation_id)?;
        let _lock = lock(&directory)?;
        let mut files = Vec::new();
        for carrier in &root.files {
            let Some(record) = read_record(&directory.join(format!("{}.json", carrier.filename)))?
            else {
                return Ok(None);
            };
            if record.carrier != *carrier {
                return Err(Status::failed_precondition(
                    "local package selection differs from accepted upload",
                ));
            }
            let Some(object) = record.object else {
                return Ok(None);
            };
            let verified = self.store.open_verified(&object.sha256).map_err(storage)?;
            if verified.len() != object.length {
                return Err(Status::data_loss("uploaded package object length changed"));
            }
            files.push(UploadedFile {
                filename: carrier.filename.clone(),
                object,
                store: self.store.clone(),
            });
        }
        let path = directory.join("root-set.json");
        match read_json::<RootSet>(&path)? {
            Some(prior) if prior != root => {
                return Err(Status::failed_precondition(
                    "package metadata changed under this operation and owner",
                ))
            }
            Some(_) => (),
            None => atomic_record(&path, &root)?,
        }
        Ok(Some(UploadedPackage { root, files }))
    }

    /// Installer lifecycle cleanup, only after its immutable generation owns the
    /// materialized carriers. Accepted/running environments remain installer-owned.
    pub fn release_after_install(
        &self,
        actor: VerifiedActor,
        operation: &str,
    ) -> Result<(), Status> {
        valid_operation(operation)?;
        let directory = self.directory(actor, operation)?;
        let _lock = lock(&directory)?;
        for record in records(&directory)? {
            source_artifact::release(&self.store, &record.retention).map_err(storage)?;
        }
        fs::remove_dir_all(&directory).map_err(disk)?;
        sync_dir(directory.parent().expect("owned operation parent"))
    }

    fn directory(&self, actor: VerifiedActor, operation: &str) -> Result<PathBuf, Status> {
        let owner = self.root.join(sha256::hex(&actor.public_key));
        let operation = owner.join(operation);
        for path in [&owner, &operation] {
            fs::create_dir_all(path).map_err(disk)?;
            let metadata = fs::symlink_metadata(path).map_err(disk)?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                return Err(Status::failed_precondition(
                    "upload namespace must be a real directory",
                ));
            }
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(disk)?;
            sync_dir(path.parent().expect("owned namespace parent"))?;
        }
        Ok(operation)
    }
}

impl UploadSession {
    pub fn received(&self) -> u64 {
        if self.record.object.is_some() {
            self.record.carrier.length
        } else {
            self.received
        }
    }
    pub fn verified(&self) -> bool {
        self.record.object.is_some()
    }
    pub fn append(&mut self, chunk: pb::LocalPackageUploadChunk) -> Result<(), Status> {
        if self.verified()
            || chunk.data.is_empty()
            || chunk.data.len() > MAX_CHUNK
            || chunk.offset != self.received
            || chunk.data.len() as u64 > self.record.carrier.length - self.received
        {
            return Err(Status::failed_precondition(
                "upload chunk must continue the exact durable prefix",
            ));
        }
        self.payload
            .seek(SeekFrom::Start(self.received))
            .map_err(disk)?;
        self.payload.write_all(&chunk.data).map_err(disk)?;
        self.payload.sync_all().map_err(disk)?;
        self.received += chunk.data.len() as u64;
        if self.received == self.record.carrier.length {
            self.finish()?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), Status> {
        if self.record.carrier.filename == "source.tar" {
            validate_archive(&self.payload)?;
        }
        self.payload.seek(SeekFrom::Start(0)).map_err(disk)?;
        let mut hash = sha256::Sha256::new();
        let mut buffer = [0; 64 << 10];
        loop {
            let count = self.payload.read(&mut buffer).map_err(disk)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        let digest = hash.finish();
        if !self.record.carrier.digest.is_empty() && self.record.carrier.digest != digest {
            return Err(Status::failed_precondition(
                "captured wheel digest does not verify",
            ));
        }
        let object = ObjectRef {
            sha256: sha256::hex(&digest),
            length: self.record.carrier.length,
        };
        let tree = Draft {
            entries: vec![(
                self.record.carrier.filename.clone(),
                Entry::File(object.clone()),
            )],
        }
        .seal()
        .map_err(storage)?;
        source_artifact::import_tree(
            &self.uploads.store,
            &self.record.retention,
            &tree,
            &[(
                self.record.carrier.filename.clone(),
                self.directory
                    .join(format!("{}.partial", self.record.carrier.filename)),
            )],
            &Fault::default(),
        )
        .map_err(storage)?;
        self.record.object = Some(StoredObject {
            sha256: object.sha256,
            length: object.length,
        });
        atomic_record(&self.record_path, &self.record)?;
        fs::remove_file(
            self.directory
                .join(format!("{}.partial", self.record.carrier.filename)),
        )
        .map_err(disk)?;
        sync_dir(&self.directory)?;
        Ok(())
    }
    fn verify_retained(&self) -> Result<(), Status> {
        let object = self.record.object.as_ref().expect("completed carrier");
        let file = self
            .uploads
            .store
            .open_verified(&object.sha256)
            .map_err(storage)?;
        if file.len() != self.record.carrier.length {
            return Err(Status::data_loss(
                "retained carrier length differs from accepted upload",
            ));
        }
        Ok(())
    }
}

fn valid_operation(value: &str) -> Result<(), Status> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphanumeric())
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(Status::invalid_argument(
            "invalid package upload operation ID",
        ));
    }
    Ok(())
}
fn valid_carrier(file: &Carrier) -> Result<(), Status> {
    let source =
        file.filename == "source.tar" && file.digest.is_empty() && file.length <= MAX_SET_BYTES;
    let wheel = file.filename.len() <= 255
        && file.filename.ends_with(".whl")
        && file
            .filename
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphanumeric())
        && file
            .filename
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._+-".contains(&b))
        && file.digest.len() == 32
        && file.length <= MAX_WHEEL_BYTES;
    if file.length == 0 || !(source || wheel) {
        return Err(Status::invalid_argument(
            "package carrier must be bounded source.tar or a checksummed wheel",
        ));
    }
    Ok(())
}
fn root_set(selected: &pb::DesiredLocalPackageSet) -> Result<RootSet, Status> {
    valid_operation(&selected.operation_id)?;
    let package = selected
        .package
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("local package metadata is required"))?;
    if [&package.package, &package.release, &package.installation_id]
        .iter()
        .any(|s| s.is_empty() || s.len() > 256)
        || selected.files.is_empty()
        || selected.files.len() > MAX_FILES
        || selected.dependency_requirements.len() > MAX_CHUNK
        || selected.python_requires.len() > 4096
        || selected.python_version.len() > 64
    {
        return Err(Status::invalid_argument(
            "incomplete or unbounded local package selection",
        ));
    }
    let files: Vec<_> = selected.files.iter().map(Carrier::from).collect();
    let mut total = 0;
    for (index, file) in files.iter().enumerate() {
        valid_carrier(file)?;
        total += file.length;
        if total > MAX_SET_BYTES || (index > 0 && files[index - 1].filename >= file.filename) {
            return Err(Status::invalid_argument(
                "package files must be sorted, unique and within the deployed aggregate bound",
            ));
        }
    }
    if !selected.source_archive.is_empty()
        && (selected.source_archive != "source.tar"
            || !files.iter().any(|f| f.filename == selected.source_archive))
    {
        return Err(Status::invalid_argument(
            "source archive must name the uploaded source.tar",
        ));
    }
    Ok(RootSet {
        operation_id: selected.operation_id.clone(),
        package: package.package.clone(),
        release: package.release.clone(),
        installation_id: package.installation_id.clone(),
        python_requires: selected.python_requires.clone(),
        python_version: selected.python_version.clone(),
        source_archive: selected.source_archive.clone(),
        dependency_requirements: selected.dependency_requirements.clone(),
        files,
    })
}

fn validate_archive(file: &File) -> Result<(), Status> {
    // Bound extension allocations in a raw standard-library pass before allowing
    // the tar crate to resolve PAX/GNU names. No archive format implementation here.
    let mut input = file.try_clone().map_err(disk)?;
    input.seek(SeekFrom::Start(0)).map_err(disk)?;
    let mut archive = tar::Archive::new(input);
    for entry in archive.entries().map_err(disk)?.raw(true) {
        let entry = entry.map_err(disk)?;
        let kind = entry.header().entry_type();
        if !(kind.is_file()
            || kind.is_dir()
            || kind.is_gnu_longname()
            || kind.is_pax_local_extensions())
            || ((kind.is_gnu_longname() || kind.is_pax_local_extensions())
                && entry.size() > 64 << 10)
        {
            return Err(Status::invalid_argument(
                "unsupported or unbounded source archive member",
            ));
        }
    }
    let mut input = file.try_clone().map_err(disk)?;
    input.seek(SeekFrom::Start(0)).map_err(disk)?;
    let mut archive = tar::Archive::new(input);
    let mut seen = HashSet::new();
    let mut total = 0u64;
    let (mut project_file, mut lock_file) = (false, false);
    for entry in archive.entries().map_err(disk)? {
        let mut entry = entry.map_err(disk)?;
        let raw = entry.path_bytes();
        let name = std::str::from_utf8(&raw)
            .map_err(|_| Status::invalid_argument("source member path is not UTF-8"))?
            .trim_end_matches('/')
            .to_owned();
        let kind = entry.header().entry_type();
        project_file |= kind.is_file() && name == "pyproject.toml";
        lock_file |= kind.is_file() && name == "uv.lock";
        if name.is_empty()
            || name.contains('\\')
            || name.starts_with('/')
            || name
                .split('/')
                .any(|p| p.is_empty() || p == "." || p == "..")
            || !seen.insert(name)
            || !(kind.is_file() || kind.is_dir())
        {
            return Err(Status::invalid_argument(
                "unsafe or repeated source archive member",
            ));
        }
        total = total
            .checked_add(entry.size())
            .ok_or_else(|| Status::invalid_argument("source archive expansion overflow"))?;
        if seen.len() > 100_000 || total > MAX_SET_BYTES {
            return Err(Status::invalid_argument(
                "source inventory exceeds the deployed archive bound",
            ));
        }
        let count = std::io::copy(&mut entry, &mut std::io::sink()).map_err(disk)?;
        if count != entry.size() {
            return Err(Status::invalid_argument(
                "source archive member was truncated",
            ));
        }
    }
    if !project_file || !lock_file {
        return Err(Status::invalid_argument(
            "source archive requires pyproject.toml and uv.lock",
        ));
    }
    Ok(())
}

fn lock(directory: &Path) -> Result<File, Status> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(directory.join("operation.lock"))
        .map_err(disk)?;
    file.try_lock_exclusive()
        .map_err(|_| Status::aborted("this owner operation already has a live uploader"))?;
    Ok(file)
}
fn read_record(path: &Path) -> Result<Option<Record>, Status> {
    read_json(path)
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>, Status> {
    match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => {
            if !file.metadata().map_err(disk)?.is_file() {
                return Err(Status::data_loss("upload record is not a regular file"));
            }
            serde_json::from_reader(file.take(8 << 20))
                .map(Some)
                .map_err(|_| Status::data_loss("retained upload record is invalid"))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(disk(error)),
    }
}
fn records(directory: &Path) -> Result<Vec<Record>, Status> {
    let mut records = Vec::new();
    for entry in fs::read_dir(directory).map_err(disk)? {
        let path = entry.map_err(disk)?.path();
        if path.extension().is_some_and(|e| e == "json")
            && path.file_name().is_none_or(|n| n != "root-set.json")
        {
            if let Some(record) = read_record(&path)? {
                records.push(record);
            }
        }
    }
    Ok(records)
}
fn atomic_record<T: Serialize>(path: &Path, record: &T) -> Result<(), Status> {
    let temporary = path.with_extension("pending");
    let bytes = serde_json::to_vec(record)
        .map_err(|_| Status::internal("cannot encode typed upload record"))?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temporary)
        .map_err(disk)?;
    file.write_all(&bytes).map_err(disk)?;
    file.sync_all().map_err(disk)?;
    fs::rename(temporary, path).map_err(disk)?;
    sync_dir(path.parent().expect("owned record parent"))
}
fn sync_dir(path: &Path) -> Result<(), Status> {
    File::open(path).and_then(|f| f.sync_all()).map_err(disk)
}
fn disk(error: std::io::Error) -> Status {
    Status::unavailable(format!("package upload storage unavailable: {error}"))
}
fn storage(error: tensorfs_core::err::Refusal) -> Status {
    Status::failed_precondition(format!("package upload TensorFS custody: {error}"))
}
