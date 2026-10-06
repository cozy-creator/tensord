//! The run output log's products: what a run has made so far, as it makes it (wire 65).
//! `Outputs.publish` commits the asset's bytes into native custody, binds them to the run's
//! actor and journals one `product` on the run's log, in that order, so every product a
//! client reads can be fetched with ReadByteTreeObject. A single output's product is replaced
//! by each publish (SET); a list output grows (APPEND). The returned result publishes only
//! what the log does not already show.
use crate::{
    api::domain,
    device_executor::{Answer, Frame},
    execution::Engine,
    journal::StoredProduct,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::{Component, Path, PathBuf},
};
use tensorfs_core::{
    ids::{ObjectRef, StoredDoc},
    manifest::{Draft, Entry},
    sha256, source_artifact,
    store::{Fault, Store},
};

const MAX_PARTS: usize = 128;

fn storage(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

/// Retain one byte string as a sealed one-file native tree named by `owner`, bound to `actor`
/// so a reader with that actor's capability can fetch it. Idempotent per owner.
pub fn retain(
    store: &Store,
    engine: &Engine,
    actor: &str,
    owner: &str,
    source: &mut File,
    object: &ObjectRef,
) -> io::Result<domain::NativeByteRetentionRequest> {
    let tree = Draft {
        entries: vec![("payload".into(), Entry::File(object.clone()))],
    }
    .seal()
    .map_err(storage)?;
    // The writer's shared guard excludes GC until its standing root records the member.
    let mut writer =
        source_artifact::Writer::open(store, owner, tree.object_ref().map_err(storage)?)
            .map_err(storage)?;
    if !writer.completed().map_err(storage)? {
        store
            .put_stream(source, Some(object), &Fault::default())
            .map_err(storage)?;
        writer.landed(object).map_err(storage)?;
    }
    let root = writer.finish(&tree).map_err(storage)?;
    let receipt = root.receipt().map_err(storage)?;
    let native = domain::NativeByteRetentionRequest {
        source: Some(domain::NativeByteTreeRef {
            producer_root_id: root.producer,
            receipt_digest: sha256::digest(&receipt).to_vec(),
            manifest: Some(domain::Ref {
                digest: unhex(&root.manifest.sha256)?,
                length: root.manifest.length,
            }),
            content_bytes: object.length,
        }),
        retention_id: owner.into(),
    };
    engine.bind_native_output(actor, owner, &crate::archive::encode_native_byte_retention_request(&native))?;
    Ok(native)
}

fn unhex(text: &str) -> io::Result<Vec<u8>> {
    if text.len() != 64 {
        return Err(storage("digest length differs"));
    }
    (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(storage))
        .collect()
}

/// A published asset's bytes: a regular file of the attempt's spool, at most one directory
/// down (a join's parts). Never a symlink or a path outside the spool.
fn spool_file(spool: &Path, name: &str) -> io::Result<PathBuf> {
    let refuse = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the published asset names no file in this attempt's spool",
        )
    };
    let relative = Path::new(name);
    let steps: Vec<_> = relative.components().collect();
    if steps.is_empty()
        || steps.len() > 2
        || steps.iter().any(|step| {
            !matches!(step, Component::Normal(part) if !part.to_string_lossy().starts_with('.'))
        })
    {
        return Err(refuse());
    }
    let path = spool.join(relative);
    if steps.len() == 2
        && spool
            .join(steps[0])
            .symlink_metadata()?
            .file_type()
            .is_symlink()
    {
        return Err(refuse());
    }
    let metadata = path.symlink_metadata().map_err(|_| refuse())?;
    if !metadata.file_type().is_file() {
        return Err(refuse());
    }
    Ok(path)
}

/// A saved asset's spool file, named as the SDK's post phase names it:
/// `attempt:<request>/<kind>/<n>` lives at `<kind>-<n>`. A file a child returned is named by
/// its digest, `sha256:<hex>`, and linked into its parent's spool as `sha256-<hex>`
/// (`jobs::grant`), so a job can publish it as it is.
fn spool_name(asset_ref: &str) -> String {
    if let Some(hex) = asset_ref
        .strip_prefix("sha256:")
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return format!("sha256-{hex}");
    }
    let tail: Vec<_> = asset_ref.rsplitn(3, '/').collect();
    if tail.len() == 3 {
        format!("{}-{}", tail[1], tail[0])
    } else {
        String::new()
    }
}

fn hash_file(path: &Path) -> io::Result<ObjectRef> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    let mut length = 0u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        length += count as u64;
    }
    Ok(ObjectRef {
        sha256: sha256::hex(&hash.finalize()),
        length,
    })
}

/// How a declared result field takes products: a single asset is SET, a list of assets grows
/// by APPEND. Anything else is not a publishable output.
fn output_op(result: &Value, output: &str) -> Option<domain::RunProductOp> {
    let mut schema = result;
    for name in output.split('.') {
        schema = schema
            .get("fields")?
            .as_array()?
            .iter()
            .find(|field| field.get("name").and_then(Value::as_str) == Some(name))?
            .get("type")?;
    }
    if schema.get("asset").is_some() {
        Some(domain::RunProductOp::Set)
    } else if schema.get("list")?.get("asset").is_some() {
        Some(domain::RunProductOp::Append)
    } else {
        None
    }
}

pub fn decode(stored: &StoredProduct) -> io::Result<domain::RunProduct> {
    crate::archive::decode_run_product(stored.product.as_slice())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "journaled product is corrupt"))
}

/// The SDK's `publish` request during a running attempt: answers `Published`.
pub fn publish(store: &Store, engine: &Engine, id: &str, spool: &Path, frame: &Frame) -> Answer {
    publish_in(store, engine, id, spool, frame, None)
}

/// `publish` for an attempt that may replay an earlier one's (a resumed job): `appended` counts
/// this attempt's list items per output, and an item an earlier attempt already published at
/// that position, with the same bytes, adds nothing.
pub fn publish_replayed(
    store: &Store,
    engine: &Engine,
    id: &str,
    spool: &Path,
    frame: &Frame,
    appended: &mut std::collections::HashMap<String, usize>,
) -> Answer {
    publish_in(store, engine, id, spool, frame, Some(appended))
}

fn publish_in(
    store: &Store,
    engine: &Engine,
    id: &str,
    spool: &Path,
    frame: &Frame,
    appended: Option<&mut std::collections::HashMap<String, usize>>,
) -> Answer {
    let mut answer = Answer::unavailable(frame.seq);
    match commit(store, engine, id, spool, frame, appended) {
        Ok((content, sequence)) => {
            answer.ok = true;
            answer.code.clear();
            answer.detail.clear();
            answer.digest = format!("sha256:{}", sha256::hex(&content.digest));
            answer.length = content.length;
            answer.sequence = sequence;
        }
        Err(error) => {
            answer.code = match error.kind() {
                io::ErrorKind::InvalidInput => "output_undeclared",
                _ => "publish_refused",
            }
            .into();
            answer.detail = error.to_string();
        }
    }
    answer
}

fn commit(
    store: &Store,
    engine: &Engine,
    id: &str,
    spool: &Path,
    frame: &Frame,
    appended: Option<&mut std::collections::HashMap<String, usize>>,
) -> io::Result<(domain::Ref, u64)> {
    let record = engine.get(id)?;
    let context = record
        .submission
        .as_ref()
        .ok_or_else(|| io::Error::other("only a public execution has an output log"))?;
    let installed = engine
        .installation_for_generation(&context.actor, &record.invocation.generation)?
        .ok_or_else(|| io::Error::other("held installation interface absent"))?;
    let interface: Value = serde_json::from_slice(&installed.interface).map_err(storage)?;
    let section = match record.invocation.job {
        true => "jobs",
        false => "entrypoints",
    };
    let result = interface
        .get(section)
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter().find(|row| {
                row.get("name").and_then(Value::as_str) == Some(&record.invocation.entrypoint)
            })
        })
        .and_then(|entry| entry.get("result"))
        .ok_or_else(|| io::Error::other("declared result schema absent"))?;
    let op = output_op(result, &frame.output).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{:?} is not an asset field of this function's result",
                frame.output
            ),
        )
    })?;
    if frame.parts.len() > MAX_PARTS {
        return Err(storage(format!("a product has at most {MAX_PARTS} parts")));
    }
    let retain_file = |path: &Path| -> io::Result<(domain::Ref, domain::NativeByteRetentionRequest)> {
        let object = hash_file(path)?;
        // One hold per distinct byte string of the run; TensorFS names owners by digest.
        let named = format!(
            "cozy.product/1\0{}\0{}\0{}",
            engine.workspace_id(),
            context.request_id,
            object.sha256
        );
        let owner = format!("sha256:{}", sha256::hex(&sha256::digest(named.as_bytes())));
        let source = retain(
            store,
            engine,
            &context.actor,
            &owner,
            &mut File::open(path)?,
            &object,
        )?;
        Ok((
            domain::Ref {
                digest: unhex(&object.sha256)?,
                length: object.length,
            },
            source,
        ))
    };
    let mut product = domain::RunProduct {
        output: frame.output.clone(),
        op: op as i32,
        media_type: frame.media_type.clone(),
        label: frame.label.clone(),
        ..Default::default()
    };
    if frame.parts.is_empty() {
        let path = spool_file(spool, &spool_name(&frame.asset_ref))?;
        let (content, source) = retain_file(&path)?;
        product.content = Some(content);
        product.source = Some(source);
    } else {
        let mut whole = Sha256::new();
        let mut length = 0;
        for part in &frame.parts {
            let path = spool_file(spool, &part.local)?;
            let mut file = File::open(&path)?;
            length += io::copy(&mut file, &mut whole)?;
            let (content, source) = retain_file(&path)?;
            product.parts.push(domain::RunProductPart {
                content: Some(content),
                source: Some(source),
                duration_us: part.duration_us,
            });
        }
        product.content = Some(domain::Ref {
            digest: whole.finalize().to_vec(),
            length,
        });
    }
    let content = product.content.clone().unwrap_or_default();
    let sequence = engine.append_product(id, |prior| {
        let prior = prior.iter().map(decode).collect::<io::Result<Vec<_>>>()?;
        if op == domain::RunProductOp::Append {
            let items: Vec<_> = prior
                .iter()
                .filter(|p| p.output == product.output && p.op == op as i32)
                .collect();
            if let Some(appended) = appended {
                let position = appended.entry(product.output.clone()).or_default();
                let replayed = items
                    .get(*position)
                    .is_some_and(|item| item.content == product.content);
                *position += 1;
                if replayed {
                    return Ok(None);
                }
            }
            product.index = items.len() as u32;
        } else if prior
            .iter()
            .rev()
            .find(|p| p.output == product.output && p.op == op as i32)
            .is_some_and(|current| current.content == product.content)
        {
            return Ok(None); // the log already shows these bytes
        }
        Ok(Some(encode(&product)))
    })?;
    Ok((content, sequence))
}

/// A job's adopted weights output as a product of its run's log: its bytes are the output's
/// manifest (`weights::MANIFEST_MEDIA`), held in the machine's store; a resumed attempt that
/// adopts the same manifest adds nothing.
pub fn record_manifest(engine: &Engine, id: &str, output: &str, manifest: &tensorfs_core::ids::ObjectRef) -> io::Result<u64> {
    let content = domain::Ref {
        digest: unhex(&manifest.sha256)?,
        length: manifest.length,
    };
    let product = domain::RunProduct {
        output: output.into(),
        op: domain::RunProductOp::Set as i32,
        media_type: crate::weights::MANIFEST_MEDIA.into(),
        label: output.into(),
        content: Some(content.clone()),
        ..Default::default()
    };
    engine.append_product(id, |prior| {
        let prior = prior.iter().map(decode).collect::<io::Result<Vec<_>>>()?;
        let held = prior
            .iter()
            .rev()
            .find(|p| p.output == product.output)
            .is_some_and(|current| current.content.as_ref() == Some(&content));
        Ok((!held).then(|| encode(&product)))
    })
}

/// The canonical body of one product event.
pub fn document(product: &domain::RunProduct) -> io::Result<Value> {
    let missing = || io::Error::new(io::ErrorKind::InvalidData, "product reference absent");
    let reference = |content: &domain::Ref| json!({"digest":format!("sha256:{}",sha256::hex(&content.digest)),"length":content.length});
    let source = |source: &domain::NativeByteRetentionRequest| -> io::Result<Value> {
        let tree = source.source.as_ref().ok_or_else(missing)?;
        let manifest = tree.manifest.as_ref().ok_or_else(missing)?;
        Ok(
            json!({"retention_id":source.retention_id,"source":{"producer_root_id":tree.producer_root_id,"receipt_digest":format!("sha256:{}",sha256::hex(&tree.receipt_digest)),"manifest":reference(manifest),"content_bytes":tree.content_bytes}}),
        )
    };
    let mut document = json!({"format":"cozy.machine.product/1","output":product.output,"op":product.op,
        "content":reference(product.content.as_ref().ok_or_else(missing)?),"media_type":product.media_type});
    if let Some(held) = &product.source {
        document["source"] = source(held)?;
    }
    if product.index != 0 {
        document["index"] = json!(product.index);
    }
    if !product.label.is_empty() {
        document["label"] = json!(product.label);
    }
    if !product.parts.is_empty() {
        document["parts"] = product
            .parts
            .iter()
            .map(|part| {
                let mut value =
                    json!({"content":reference(part.content.as_ref().ok_or_else(missing)?)});
                if let Some(held) = &part.source {
                    value["source"] = source(held)?;
                }
                if part.duration_us != 0 {
                    value["duration_us"] = json!(part.duration_us);
                }
                Ok(value)
            })
            .collect::<io::Result<Vec<_>>>()?
            .into();
    }
    Ok(document)
}

/// Whether the log already shows a returned product: the same bytes as the last SET of that
/// output, or an APPEND at an index the log already holds.
pub fn shown(log: &[domain::RunProduct], product: &domain::RunProduct) -> bool {
    if product.op == domain::RunProductOp::Append as i32 {
        log.iter()
            .any(|p| p.output == product.output && p.op == product.op && p.index == product.index)
    } else {
        log.iter()
            .rev()
            .find(|p| p.output == product.output && p.op == product.op)
            .is_some_and(|current| current.content == product.content)
    }
}

/// The product bytes held by the execution journal.
pub fn encode(product: &domain::RunProduct) -> Vec<u8> { crate::archive::encode_run_product(product) }
