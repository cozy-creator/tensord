//! A job's weights: the executor's `weights_writer` requests answered over TensorFS's native
//! channels (`tensorfs_core::derived::channel`). A source is granted when it is one of the job's
//! model inputs, an output when the job declares it. An adopted output is held in a local
//! repository, becomes a product of the run's log (its digest the output's manifest) and, with
//! a weights destination, is published there before the adopt is answered.
use crate::{
    device_executor::{Answer, Frame},
    hub,
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tensorfs_core::{
    canon::Fields,
    derived::{self, channel},
    ids::ObjectRef,
    meta::Meta,
    repository::{Mutation, RepositoryName},
    sha256,
    store::{Fault, Store},
};

/// The media type of a weights output's product: its bytes are the output's manifest.
pub const MANIFEST_MEDIA: &str = "application/vnd.cozy.model-manifest";
/// Executor-written exchange files stay this small (TensorFS bounds its own documents).
const EXCHANGE_MAX: u64 = 64 << 20;

type BindOutput = Arc<dyn Fn(&str, &str) -> io::Result<()> + Send + Sync>;

/// Execution-owner hooks: current attempt authority and durable output association.
pub struct GrantAuthority {
    pub current: Arc<dyn Fn() -> bool + Send + Sync>,
    pub bind_output: BindOutput,
}

/// Where a job's outputs are published: the repository and its Hub, written under the run's
/// capability there.
#[derive(Clone)]
pub struct Destination {
    pub repository: String,
    pub hub: hub::Source,
}

/// One job attempt's weights grant.
pub struct Grant {
    pub actor: String,
    pub run: String,
    pub spool: PathBuf,
    /// Its model inputs' manifests ("sha256:<hex>") and lengths.
    pub sources: BTreeMap<String, u64>,
    /// Its declared outputs and their byte bounds.
    pub outputs: BTreeMap<String, u64>,
    pub destination: Option<Destination>,
    /// Refuses once the attempt is no longer current (canceled, paused, ended).
    pub current: Arc<dyn Fn() -> bool + Send + Sync>,
    /// Persist the run/output association before any native transaction can begin.
    bind_output: BindOutput,
    opened: Mutex<BTreeMap<String, String>>,
    /// Publications attempted and not yet in the run's log.
    uploads: Mutex<Vec<Upload>>,
}

/// One output's publication into its destination: when it ran, and its refusal if any.
pub struct Upload {
    pub output: String,
    pub transaction: String,
    pub destination: String,
    pub called_ms: i64,
    pub finished_ms: i64,
    pub error: Option<String>,
}

impl Grant {
    pub fn new(
        actor: &str,
        run: &str,
        spool: &Path,
        sources: BTreeMap<String, u64>,
        outputs: BTreeMap<String, u64>,
        destination: Option<Destination>,
        authority: GrantAuthority,
    ) -> Self {
        let GrantAuthority {
            current,
            bind_output,
        } = authority;
        Self {
            actor: actor.into(),
            run: run.into(),
            spool: spool.into(),
            sources,
            outputs,
            destination,
            current,
            bind_output,
            opened: Mutex::new(BTreeMap::new()),
            uploads: Mutex::new(Vec::new()),
        }
    }

    /// The publications attempted since the last call, for the run's log.
    pub fn take_uploads(&self) -> Vec<Upload> {
        std::mem::take(&mut *self.uploads.lock().unwrap())
    }
}

/// An adopted output, as the run's log records it.
pub struct Adopted {
    pub output: String,
    pub manifest: ObjectRef,
}

pub struct Weights {
    store: Arc<Store>,
    meta: Arc<Meta>,
    /// Local custody transitions only. Native payload IO and publication never hold this.
    custody: Mutex<()>,
}

type Refusal = (&'static str, String);

impl Weights {
    pub fn new(store: Arc<Store>) -> io::Result<Self> {
        let meta = Meta::open(&store).map_err(io::Error::other)?;
        Ok(Self {
            store,
            meta: Arc::new(meta),
            custody: Mutex::new(()),
        })
    }

    /// One `weights_writer` request: its answer, the channel end that follows it, and the
    /// output it adopted.
    pub fn answer(&self, grant: &Grant, frame: &Frame) -> (Answer, Option<File>, Option<Adopted>) {
        let answered = match frame.operation.as_str() {
            "source" => self.source(grant, frame).map(|(a, f)| (a, Some(f), None)),
            "output" => self.output(grant, frame).map(|(a, f)| (a, Some(f), None)),
            "adopt" => self
                .adopt(grant, frame)
                .map(|(a, adopted)| (a, None, Some(adopted))),
            other => Err((
                "weights_writer_refused",
                format!("unknown weights operation {other:?}"),
            )),
        };
        answered
            .unwrap_or_else(|(code, detail)| (Answer::refused(frame.seq, code, detail), None, None))
    }

    fn source(&self, grant: &Grant, frame: &Frame) -> Result<(Answer, File), Refusal> {
        let length = grant.sources.get(&frame.manifest).copied().ok_or((
            "weights_source_ungranted",
            "source is outside the invocation".to_string(),
        ))?;
        let source = reference(&frame.manifest, length)?;
        let (client, server) = UnixStream::pair().map_err(io_refusal)?;
        let (store, meta, current) = (self.store.clone(), self.meta.clone(), grant.current.clone());
        std::thread::Builder::new()
            .name("weights-source".into())
            .spawn(move || {
                let check = move || current_or_refuse(&*current);
                if let Err(refusal) = channel::serve_source(&store, &meta, &source, server, &check)
                {
                    eprintln!("weights source {}: {}", source.id(), refusal.detail);
                }
            })
            .map_err(io_refusal)?;
        let mut answer = Answer::ok(frame.seq);
        (answer.descriptor, answer.length) = (true, length);
        Ok((answer, File::from(std::os::fd::OwnedFd::from(client))))
    }

    fn output(&self, grant: &Grant, frame: &Frame) -> Result<(Answer, File), Refusal> {
        let slot = &frame.output_slot;
        let bound = grant.outputs.get(slot).copied().ok_or((
            "weights_output_ungranted",
            format!("{slot:?} is not a declared weights output"),
        ))?;
        if !(grant.current)() {
            return Err((
                "weights_writer_closed",
                "output attempt is not running".into(),
            ));
        }
        let raw = exchange(&grant.spool, slot, "derivation", frame.length)?;
        let mut declaration = channel::declaration_from_arguments(&raw).map_err(tfs)?;
        if declaration.max_new_bytes != bound {
            return Err((
                "weights_writer_ungranted",
                "output exceeds invocation grant".into(),
            ));
        }
        if declaration
            .sources
            .iter()
            .any(|s| grant.sources.get(&s.manifest.id()) != Some(&s.manifest.length))
        {
            return Err((
                "weights_source_ungranted",
                "source is outside invocation grant".into(),
            ));
        }
        // The output's work is its run, its slot and what it declares: a resumed attempt that
        // declares the same resumes the same transaction.
        let work = sha256::hex_digest(&raw);
        let identity = format!("{}\0{}\0{slot}\0{work}", grant.actor, grant.run);
        let transaction = format!("sha256:{}", sha256::hex_digest(identity.as_bytes()));
        declaration.work_fingerprint = Some(format!("sha256:{work}"));
        (grant.bind_output)(&transaction, slot).map_err(io_refusal)?;
        grant
            .opened
            .lock()
            .unwrap()
            .insert(transaction.clone(), slot.clone());
        let (client, server) = UnixStream::pair().map_err(io_refusal)?;
        let hooks_current = grant.current.clone();
        let (store, meta) = (self.store.clone(), self.meta.clone());
        let (operation, output) = (grant.run.clone(), slot.clone());
        let committed = match derived::lookup(&store, &meta, &transaction).map_err(tfs)? {
            derived::Lookup::Committed(result) => Some(self.retained_receipt(&result)?),
            _ => None,
        };
        let writer = match committed {
            Some(_) => None,
            None => Some(
                channel::Writer::begin_next(&store, meta.clone(), &transaction, declaration, None)
                    .map_err(tfs)?,
            ),
        };
        std::thread::Builder::new()
            .name("weights-output".into())
            .spawn(move || {
                let check = move || current_or_refuse(&*hooks_current);
                let hooks = channel::Hooks {
                    check_current: &check,
                    record_checkpoint: &|_: &tensorfs_core::jcs::Json| (),
                    record_receipt: &|_: &derived::ReceiptFacts| (),
                };
                let served = match (&writer, &committed) {
                    (Some(writer), _) => {
                        channel::serve_derived(writer, server, &operation, &output, None, &hooks)
                    }
                    (None, Some(receipt)) => channel::serve_derived_replay(receipt, server, &hooks),
                    (None, None) => Ok(()),
                };
                if let Err(refusal) = served {
                    eprintln!("weights output {output}: {}", refusal.detail);
                }
            })
            .map_err(io_refusal)?;
        let mut answer = Answer::ok(frame.seq);
        (answer.descriptor, answer.transaction) = (true, transaction);
        Ok((answer, File::from(std::os::fd::OwnedFd::from(client))))
    }

    fn adopt(&self, grant: &Grant, frame: &Frame) -> Result<(Answer, Adopted), Refusal> {
        current_or_refuse(&*grant.current).map_err(tfs)?;
        let slot = &frame.output_slot;
        let opened = grant
            .opened
            .lock()
            .unwrap()
            .get(&frame.transaction)
            .cloned();
        if opened.as_ref() != Some(slot) {
            return Err((
                "weights_receipt_mismatch",
                "receipt has no opened output".into(),
            ));
        }
        let relayed = exchange(&grant.spool, slot, "native-receipt", frame.length)?;
        let custody = self.custody.lock().unwrap();
        let receipt =
            match derived::lookup(&self.store, &self.meta, &frame.transaction).map_err(tfs)? {
                derived::Lookup::Committed(result) => self.retained_receipt(&result)?,
                _ => {
                    return Err((
                        "weights_receipt_mismatch",
                        "the output is not committed".into(),
                    ))
                }
            };
        let facts = tensorfs_core::canon::write(&receipt.to_value());
        let relayed = tensorfs_core::canon::parse(&relayed, EXCHANGE_MAX as usize).map_err(tfs)?;
        if !same_receipt(&relayed, &receipt).map_err(tfs)? {
            return Err((
                "weights_receipt_mismatch",
                "receipt differs from native custody".into(),
            ));
        }
        // The output stays reachable in its own local repository, whatever happens to the run.
        let name = format!(
            "output-{}",
            &sha256::hex_digest(format!("{}\0{}\0{slot}", grant.actor, grant.run).as_bytes())[..40]
        );
        let repo = RepositoryName::new("local", &name).map_err(tfs)?;
        let current = std::fs::read(self.store.repository_path(&repo)).ok();
        let mutation = Mutation::ReplaceLocal {
            repo,
            version: frame.transaction.trim_start_matches("sha256:").into(),
            manifest: receipt.manifest.clone(),
        };
        current_or_refuse(&*grant.current).map_err(tfs)?;
        self.store
            .apply_repository(current.as_deref(), &mutation, &Fault::default())
            .map_err(tfs)?;
        // This local repository is durable result custody even if its acknowledgment or
        // subsequent publication is lost. Do not replace another consumer's private root.
        if let derived::Lookup::Committed(result) =
            derived::lookup(&self.store, &self.meta, &frame.transaction).map_err(tfs)?
        {
            let disposition = result.disposition_value();
            if Fields::new("derived disposition", &disposition)
                .map_err(tfs)?
                .req_str("kind")
                .map_err(tfs)?
                == "pending"
            {
                let retained =
                    format!("output-{}", frame.transaction.trim_start_matches("sha256:"));
                derived::adopt(&self.store, &self.meta, &frame.transaction, &retained)
                    .map_err(tfs)?;
            }
        }
        drop(custody);
        if let Some(destination) = &grant.destination {
            current_or_refuse(&*grant.current).map_err(tfs)?;
            let called_ms = crate::machine::lifecycle::now_ms();
            let published = self.publish(grant, destination, slot, &receipt.manifest);
            grant.uploads.lock().unwrap().push(Upload {
                output: slot.clone(),
                transaction: frame.transaction.clone(),
                destination: destination.repository.trim_start_matches("model://").into(),
                called_ms,
                finished_ms: crate::machine::lifecycle::now_ms(),
                error: published.as_ref().err().map(|(_, message)| message.clone()),
            });
            published?;
        }
        let mut answer = Answer::ok(frame.seq);
        current_or_refuse(&*grant.current).map_err(tfs)?;
        answer.request_id = grant.run.clone();
        answer.manifest = receipt.manifest.id();
        answer.manifest_length = receipt.manifest.length;
        answer.receipt_digest = format!("sha256:{}", sha256::hex_digest(&facts));
        Ok((
            answer,
            Adopted {
                output: slot.clone(),
                manifest: receipt.manifest,
            },
        ))
    }

    /// Explicit run cancellation relinquishes only its unfinished/pending native claim.
    /// A committed adopted result and independent consumer roots survive.
    pub fn cancel_output(&self, transaction: &str) -> io::Result<()> {
        let _custody = self.custody.lock().unwrap();
        if let derived::Lookup::Open {
            writer_session: Some(session),
        } = derived::lookup(&self.store, &self.meta, transaction).map_err(io::Error::other)?
        {
            derived::fence(&self.meta, transaction, session).map_err(io::Error::other)?;
        }
        let committed =
            derived::abandon(&self.store, &self.meta, transaction).map_err(io::Error::other)?;
        if committed.is_some() {
            if let derived::Lookup::Committed(result) =
                derived::lookup(&self.store, &self.meta, transaction).map_err(io::Error::other)?
            {
                let disposition = result.disposition_value();
                let kind = Fields::new("derived disposition", &disposition)
                    .and_then(|mut fields| fields.req_str("kind"))
                    .map_err(io::Error::other)?;
                match kind {
                    "pending" | "released" => {
                        derived::dispose(&self.store, &self.meta, transaction)
                            .map_err(io::Error::other)?;
                    }
                    "adopted" => (),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            "unknown derived result disposition",
                        ))
                    }
                }
            }
        }
        Ok(())
    }

    /// Receipt metadata outlives its payload custody. Replay and adoption require both.
    fn retained_receipt(
        &self,
        result: &derived::DispositionResult,
    ) -> Result<derived::ReceiptFacts, Refusal> {
        let disposition = result.disposition_value();
        let mut fields = Fields::new("derived disposition", &disposition).map_err(tfs)?;
        if !matches!(fields.req_str("kind").map_err(tfs)?, "pending" | "adopted") {
            return Err((
                "weights_transaction_closed",
                "output is no longer retained".into(),
            ));
        }
        let receipt = result.receipt();
        let components = receipt
            .declaration
            .components
            .iter()
            .map(|row| row.target.clone())
            .collect();
        let configs = receipt
            .declaration
            .configs
            .iter()
            .map(|row| match row {
                derived::ConfigDeclaration::Add { target }
                | derived::ConfigDeclaration::Copy { target, .. }
                | derived::ConfigDeclaration::Derive { target, .. } => target.clone(),
            })
            .collect();
        derived::inspect_source(
            &self.store,
            &self.meta,
            receipt.manifest.clone(),
            components,
            configs,
        )
        .map_err(|refusal| {
            (
                "weights_receipt_unavailable",
                format!(
                    "completed output is not retained and complete: {}",
                    refusal.detail
                ),
            )
        })?;
        Ok(receipt.clone())
    }

    /// One adopted output into its destination, under the run's capability.
    fn publish(
        &self,
        grant: &Grant,
        destination: &Destination,
        slot: &str,
        manifest: &ObjectRef,
    ) -> Result<(), Refusal> {
        let repository = destination.repository.trim_start_matches("model://");
        let publishing = hub::Catalog::for_op(
            &destination.hub,
            hub::Op::Publish { model: repository },
        )?;
        let operation = format!(
            "output-{}",
            &sha256::hex_digest(format!("{}\0{}\0{slot}", grant.actor, grant.run).as_bytes())[..40]
        );
        tensorfs_core::transport::publish(&tensorfs_core::transport::Publication {
            store: &self.store,
            hub: publishing.origin(),
            destination: repository,
            manifest,
            operation: &operation,
            credential: publishing.credential(),
            policy: publishing.policy(),
            progress: &|_, _| (),
            streams: 8,
        })
        .map(drop)
        .map_err(|e| {
            let message = if e.code == tensorfs_core::err::Code::HUB_REFUSED {
                format!("publication was refused before its commit: {}", e.detail)
            } else {
                format!("publication failed before its commit: {e}")
            };
            publishing.reason("weights_publication_failed", message)
        })
    }
}

/// The executor relays identity, while adoption records the owner's native facts. Future
/// receipt observations do not change transaction, declaration or immutable result identity.
fn same_receipt(
    relayed: &tensorfs_core::canon::Value,
    native: &derived::ReceiptFacts,
) -> tensorfs_core::err::Result<bool> {
    let mut receipt = Fields::new("derived receipt", relayed)?;
    let mut manifest = Fields::new("derived receipt manifest", receipt.req("manifest")?)?;
    let native_value = native.to_value();
    let mut native_fields = Fields::new("native derived receipt", &native_value)?;
    Ok(receipt.req_str("transaction_id")? == native.transaction
        && receipt.req_str("declaration_digest")? == native_fields.req_str("declaration_digest")?
        && manifest.req_str("sha256")? == native.manifest.sha256
        && manifest.req_uint("length")? == native.manifest.length)
}

/// An executor-written exchange file in the attempt's spool, exactly `length` bytes.
fn exchange(spool: &Path, slot: &str, kind: &str, length: u64) -> Result<Vec<u8>, Refusal> {
    let valid = !slot.is_empty()
        && slot.len() <= 64
        && slot
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !valid || length > EXCHANGE_MAX {
        return Err((
            "weights_host_identity",
            "weights exchange identity is invalid".into(),
        ));
    }
    let path = spool.join(format!("weights-{slot}-{kind}.canonical"));
    let mut file = File::open(&path).map_err(io_refusal)?;
    let mut bytes = Vec::with_capacity(length as usize);
    file.by_ref()
        .take(length + 1)
        .read_to_end(&mut bytes)
        .map_err(io_refusal)?;
    if bytes.len() as u64 != length {
        return Err((
            "weights_writer_length",
            "writer bytes differ from their declared length".into(),
        ));
    }
    Ok(bytes)
}

fn reference(id: &str, length: u64) -> Result<ObjectRef, Refusal> {
    let sha = id.strip_prefix("sha256:").unwrap_or(id);
    tensorfs_core::ids::hex64("manifest", sha).map_err(tfs)?;
    Ok(ObjectRef {
        sha256: sha.into(),
        length,
    })
}

fn current_or_refuse(current: &(dyn Fn() -> bool + Send + Sync)) -> tensorfs_core::err::Result<()> {
    match current() {
        true => Ok(()),
        false => tensorfs_core::err::refuse(
            tensorfs_core::err::Code::LEASE_REVOKED,
            "the writer's attempt is no longer current",
        ),
    }
}

fn tfs(refusal: tensorfs_core::err::Refusal) -> Refusal {
    (
        "weights_writer_refused",
        format!("{}: {}", refusal.code.as_str(), refusal.detail),
    )
}

fn io_refusal(error: io::Error) -> Refusal {
    ("weights_writer_refused", error.to_string())
}

#[cfg(test)]
mod tests;
