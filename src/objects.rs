//! Write: content-addressed objects a signer uploads (run inputs, local package sources),
//! resumable from the length already held. Bytes stage per signer and enter the TensorFS
//! store only once their digest verifies. A run names only objects its own signer wrote, so
//! another signer's object is neither readable nor reported as held.
use crate::execution::Engine;
use fs2::FileExt;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, MutexGuard},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tensorfs_core::{catalog::WriterGuard, ids::ObjectRef, object_roots, sha256, store::Fault, store::Store};

pub struct Objects {
    root: PathBuf,
    store: Arc<Store>,
    engine: Arc<Engine>,
}

/// Why a write was refused, as the API's typed reason.
#[derive(Debug)]
pub struct Refused {
    pub code: &'static str,
    pub message: String,
}
fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}
impl From<io::Error> for Refused {
    fn from(error: io::Error) -> Self {
        refused("object_storage_failed", error.to_string())
    }
}

/// One object's upload in progress: bytes append from the held length.
pub struct Writer<'a> {
    objects: &'a Objects,
    actor: String,
    object: ObjectRef,
    part: PathBuf,
    file: Option<File>,
    held: u64,
    skip: u64,
}

impl Objects {
    pub fn new(root: &Path, store: Arc<Store>, engine: Arc<Engine>) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        let objects = Self {
            root: root.to_path_buf(),
            store,
            engine,
        };
        // Migration/restoration happens before any publisher or background GC is wired.
        // Current writes are already durably rooted; old admitted rows gain that custody.
        let _custody = objects.guard();
        let _writer = objects.writer_guard()?;
        for object in objects.engine.with_journal(|j| j.written_objects())? {
            if objects.store.object_path(&object.sha256).is_file() {
                object_roots::retain(&objects.store, &object).map_err(io::Error::other)?;
            }
        }
        drop(_writer);
        drop(_custody);
        Ok(objects)
    }

    pub(crate) fn guard(&self) -> MutexGuard<'_, ()> {
        self.engine.object_custody.lock().unwrap()
    }
    pub(crate) fn writer_guard(&self) -> io::Result<WriterGuard> {
        WriterGuard::acquire(self.store.root()).map_err(io::Error::other)
    }
    /// Called with custody and native writer exclusion held, before acceptance.
    pub(crate) fn retain(&self, objects: &[ObjectRef]) -> io::Result<()> {
        for object in objects {
            object_roots::retain(&self.store, object).map_err(io::Error::other)?;
        }
        Ok(())
    }

    /// Unreferenced writes expire after the configured cache TTL. Native roots survive
    /// a crash before the journal bind too: root mtime gives those orphans the same TTL.
    pub fn sweep(&self, ttl: Duration) -> io::Result<usize> {
        let _custody = self.guard();
        let before = SystemTime::now().checked_sub(ttl).unwrap_or(UNIX_EPOCH);
        let before_ms = before.duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64;
        let mut removed = 0;
        for object in object_roots::list(&self.store).map_err(io::Error::other)? {
            let path = self.store.root().join("roots/objects").join(format!("{}.json",object.sha256));
            if fs::metadata(path)?.modified()? >= before { continue; }
            if self.engine.with_journal(|j| j.object_releasable(&object.sha256, before_ms))?
                && object_roots::remove(&self.store, &object.sha256).map_err(io::Error::other)? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// The stored object this signer wrote, or None.
    pub fn path(&self, actor: &str, digest: &str) -> io::Result<Option<(PathBuf, u64)>> {
        let Some(hex) = hex(digest) else {
            return Ok(None);
        };
        let Some(length) = self.engine.with_journal(|j| j.object(actor, &hex))? else {
            return Ok(None);
        };
        let path = self.store.object_path(&hex);
        Ok(path.is_file().then_some((path, length)))
    }

    /// A file this machine produced for the signer (a child run's result) as its object, so a
    /// later run of that signer may be handed it.
    pub fn adopt(&self, actor: &str, parent: &str, file: &Path, object: &ObjectRef) -> io::Result<()> {
        let _custody = self.guard();
        let _writer = self.writer_guard()?;
        if self.path(actor, &format!("sha256:{}", object.sha256))?.is_none() {
            self.store
                .put_file(file, Some(object), &Fault::default())
                .map_err(io::Error::other)?;
        }
        self.retain(std::slice::from_ref(object))?;
        self.engine.with_journal(|j| j.adopt_object(actor, parent, object))?;
        Ok(())
    }

    /// Opens `digest` (`sha256:<hex>`) of `length` for this signer, its bytes continuing at
    /// `offset`. Bytes before the held length are skipped; a gap beyond it is refused.
    pub fn begin(
        &self,
        actor: &str,
        digest: &str,
        length: u64,
        offset: u64,
    ) -> Result<Writer<'_>, Refused> {
        let hex = hex(digest).ok_or_else(|| {
            refused(
                "invalid_request",
                "an object is named by sha256:<64 lowercase hex>",
            )
        })?;
        let object = ObjectRef {
            sha256: hex.clone(),
            length,
        };
        let dir = self.root.join(&sha256::hex_digest(actor.as_bytes())[..32]);
        let part = dir.join(format!("{hex}.part"));
        let mut writer = Writer {
            objects: self,
            actor: actor.to_string(),
            object,
            part,
            file: None,
            held: 0,
            skip: 0,
        };
        if self.path(actor, digest)?.is_some_and(|(_, l)| l == length) {
            writer.held = length;
            return Ok(writer);
        }
        fs::create_dir_all(&dir)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&writer.part)?;
        file.try_lock_exclusive().map_err(|_| {
            refused(
                "object_write_busy",
                "another write of this object is in progress",
            )
        })?;
        writer.held = file.metadata()?.len();
        if writer.held > length {
            file.set_len(0)?;
            writer.held = 0;
        }
        if offset > writer.held {
            return Err(refused(
                "object_offset_ahead",
                format!("this machine holds {} bytes of the object", writer.held),
            ));
        }
        self.store
            .admits(length - writer.held)
            .map_err(|e| refused("machine_disk_full", format!("no room for the object: {e}")))?;
        writer.file = Some(file);
        writer.skip = writer.held - offset;
        Ok(writer)
    }
}

impl Writer<'_> {
    pub fn held(&self) -> u64 {
        self.held
    }

    /// The next bytes of the stream, in order from the frame's offset.
    pub fn append(&mut self, mut data: &[u8]) -> Result<(), Refused> {
        let skip = (self.skip as usize).min(data.len());
        self.skip -= skip as u64;
        data = &data[skip..];
        if data.is_empty() {
            return Ok(());
        }
        let Some(file) = self.file.as_mut() else {
            return Ok(()); // already written: the bytes are known
        };
        if self.held + data.len() as u64 > self.object.length {
            return Err(refused(
                "object_length_exceeded",
                "more bytes than the object's declared length",
            ));
        }
        file.seek(SeekFrom::End(0))?;
        file.write_all(data)?;
        self.held += data.len() as u64;
        Ok(())
    }

    /// Ends this attempt: a complete object is verified and stored for this signer; a
    /// partial one stays staged for the next attempt. Answers the bytes held.
    pub fn finish(mut self) -> Result<u64, Refused> {
        let Some(file) = self.file.take() else {
            let _custody = self.objects.guard();
            let _writer = self.objects.writer_guard()?;
            self.objects.retain(std::slice::from_ref(&self.object))?;
            self.objects.engine.with_journal(|j| j.bind_object(&self.actor, &self.object))?;
            return Ok(self.held);
        };
        file.sync_data()?;
        if self.held < self.object.length {
            return Ok(self.held);
        }
        let _custody = self.objects.guard();
        let _writer = self.objects.writer_guard()?;
        let stored = self
            .objects
            .store
            .put_file(&self.part, Some(&self.object), &Fault::default());
        let _ = fs::remove_file(&self.part);
        stored.map_err(|e| {
            refused(
                match e.code {
                    tensorfs_core::err::Code::CAPACITY_EXHAUSTED => "machine_disk_full",
                    tensorfs_core::err::Code::OBJECT_ID_MISMATCH | tensorfs_core::err::Code::LENGTH_MISMATCH => "object_digest_mismatch",
                    _ => "object_storage_failed",
                },
                e.to_string(),
            )
        })?;
        self.objects.retain(std::slice::from_ref(&self.object))?;
        let (actor, object) = (self.actor.clone(), self.object.clone());
        self.objects
            .engine
            .with_journal(|j| j.bind_object(&actor, &object))?;
        Ok(self.held)
    }
}

fn hex(digest: &str) -> Option<String> {
    digest
        .strip_prefix("sha256:")
        .filter(|h| h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_resumes_verifies_and_belongs_to_its_signer() {
        let root = std::env::temp_dir().join(format!("cm-objects-{}", uuid::Uuid::new_v4()));
        let service =
            crate::service::Service::open(&root.join("state"), &root.join("g"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let objects = Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap();
        let bytes: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let digest = format!("sha256:{}", sha256::hex_digest(&bytes));
        let length = bytes.len() as u64;

        // A first attempt stops partway; a probe then reports what is held.
        let mut first = objects.begin("alice", &digest, length, 0).unwrap();
        first.append(&bytes[..100_000]).unwrap();
        assert_eq!(first.finish().unwrap(), 100_000);
        assert_eq!(
            objects
                .begin("alice", &digest, length, 0)
                .unwrap()
                .finish()
                .unwrap(),
            100_000
        );
        assert!(objects.path("alice", &digest).unwrap().is_none());
        assert_eq!(
            objects
                .begin("alice", &digest, length, 200_000)
                .err()
                .unwrap()
                .code,
            "object_offset_ahead"
        );
        // A resend overlapping the held prefix skips it; the rest completes the object.
        let mut second = objects.begin("alice", &digest, length, 50_000).unwrap();
        second.append(&bytes[50_000..]).unwrap();
        assert_eq!(second.finish().unwrap(), length);
        let (path, held) = objects.path("alice", &digest).unwrap().unwrap();
        assert_eq!((fs::read(path).unwrap(), held), (bytes.clone(), length));
        assert_eq!(
            objects.begin("alice", &digest, length, 0).unwrap().held(),
            length
        );

        // Another signer learns nothing of it and must send the bytes itself.
        assert!(objects.path("bob", &digest).unwrap().is_none());
        assert_eq!(objects.begin("bob", &digest, length, 0).unwrap().held(), 0);

        // Bytes that differ from their digest are dropped, not stored.
        let wrong = format!("sha256:{}", "0".repeat(64));
        let mut bad = objects.begin("alice", &wrong, 3, 0).unwrap();
        bad.append(b"abc").unwrap();
        assert_eq!(bad.finish().err().unwrap().code, "object_digest_mismatch");
        assert_eq!(objects.begin("alice", &wrong, 3, 0).unwrap().held(), 0);
        let _ = fs::remove_dir_all(root);
    }
}
