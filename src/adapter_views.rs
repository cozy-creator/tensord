//! Caller LoRA adapters as one zero-copy TensorFS derivation over the base checkpoint (the
//! Python worker's adapter view, `lora_composition.prepare`): base `<path>.weight/.bias`
//! become `<path>.base_layer.*`, factors graft in as `<path>.lora_{A,B}.adapter_<i>.weight`,
//! and the `model_adapters` config holds the ordered graph the executor binds with PEFT.
//! No byte is copied or computed. Factors must already be canonical PEFT keys
//! (`<path>.lora_A.weight`, `<path>.lora_B.weight`, optional scalar `<path>.alpha`): foreign
//! layouts are normalized when a model is ingested, never here.
use serde::Serialize;
use std::{collections::BTreeMap, io, io::Read, time::SystemTime};
use tensorfs_core::{
    checkpoint::load_header,
    derived::{
        self, ComponentDeclaration, ConfigDeclaration, Declaration, Lookup, PartDeclaration,
        PartSource, Source, TensorDeclaration,
    },
    dtype::Dtype,
    header::{Body, Header, Tensor},
    ids::{Doc, ObjectRef},
    meta::Meta,
    repository::{Mutation, Repository, RepositoryName},
    store::{Fault, Store},
};

pub const FORMAT: &str = "cozy.model.lora/1";
pub const GRAPH_CONFIG: &str = "model_adapters";
const VIEW_PREFIX: &str = "adapters-";
const PLAIN: &str = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f";

/// One caller adapter, in order. An empty `component` is inferred when exactly one base
/// component fits every factor pair; an empty `source_component` is the adapter's component
/// named like `component`, else `adapter`, else its only one.
#[derive(Clone, Debug)]
pub struct Selection {
    pub manifest: ObjectRef,
    pub component: String,
    pub source_component: String,
    pub strength: f64,
}

/// The composed checkpoint: a local repository the slot's snapshots point at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Composed {
    pub repository: String,
    pub manifest: ObjectRef,
}

fn refuse(code: &str, detail: impl Into<String>) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("adapter_{code}: {}", detail.into()),
    )
}
fn native(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
fn id(object: &ObjectRef) -> String {
    format!("sha256:{}", object.sha256)
}

#[derive(Serialize)]
struct Linear {
    component: String,
    target: String,
    adapter: String,
    rank: u64,
    alpha: f64,
    strength: f64,
    dtype: &'static str,
}
#[derive(Serialize)]
struct AdapterRef {
    #[serde(rename = "ref")]
    reference: String,
    scale: f64,
    kind: &'static str,
    component: String,
    source_component: String,
    family: &'static str,
}
#[derive(Serialize)]
struct Graph {
    format: &'static str,
    layers: Vec<Linear>,
    adapters: Vec<AdapterRef>,
}

fn header(store: &Store, manifest: &ObjectRef) -> io::Result<Header> {
    let manifest = store.read_manifest(manifest).map_err(native)?;
    let reference = manifest
        .header()
        .ok_or_else(|| refuse("source", "checkpoint has no tensor header"))?;
    load_header(store, reference).map_err(native)
}

fn tensors<'h>(header: &'h Header, component: &str) -> Option<&'h [(String, Tensor)]> {
    header
        .components
        .iter()
        .find(|(name, _)| name == component)
        .map(|(_, t)| t.as_slice())
}

fn lookup<'t>(tensors: &'t [(String, Tensor)], key: &str) -> Option<&'t Tensor> {
    tensors.iter().find(|(name, _)| name == key).map(|(_, t)| t)
}

/// Every factor value finite (zero-strength inputs too), read in bounded pieces; a
/// scalar's value is returned.
fn validate(store: &Store, component: &str, key: &str, tensor: &Tensor) -> io::Result<Option<f64>> {
    let plain = |part: &tensorfs_core::header::Part| {
        part.dtype == tensor.dtype && part.shape == tensor.shape
    };
    let width = match tensor.dtype {
        Dtype::F16 | Dtype::Bf16 => 2,
        Dtype::F32 => 4,
        _ => 0,
    };
    let part = match tensor.parts.as_slice() {
        [(role, part)]
            if role == "value" && tensor.encoding == PLAIN && width > 0 && plain(part) =>
        {
            part
        }
        _ => {
            return Err(refuse(
                "tensor",
                format!("{component}.{key}: plain f16, bf16 or f32 factors required"),
            ))
        }
    };
    let decode = |chunk: &[u8]| -> Vec<f64> {
        chunk
            .chunks_exact(width)
            .map(|b| match tensor.dtype {
                Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                Dtype::Bf16 => {
                    f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64
                }
                _ => half(u16::from_le_bytes([b[0], b[1]])),
            })
            .collect()
    };
    let mut first = None;
    let mut check = |values: Vec<f64>| -> io::Result<()> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(refuse(
                "nonfinite",
                format!("{component}.{key}: factor contains NaN or infinity"),
            ));
        }
        first = first.or(values.first().copied());
        Ok(())
    };
    match &part.body {
        Body::Inline(bytes) => check(decode(bytes))?,
        Body::Segments(objects) => {
            let mut carry = Vec::new();
            for object in objects {
                let mut file = store
                    .open_verified(&object.sha256)
                    .map_err(native)?
                    .into_file();
                let mut buffer = vec![0u8; 1 << 20];
                loop {
                    let count = file.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    carry.extend_from_slice(&buffer[..count]);
                    let whole = carry.len() / width * width;
                    check(decode(&carry[..whole]))?;
                    carry.drain(..whole);
                }
            }
        }
    }
    Ok(if tensor.shape.is_empty() { first } else { None })
}

fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f64;
    match exponent {
        0 => sign * fraction * 2f64.powi(-24),
        31 if fraction == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    }
}

fn graft(
    key: &str,
    tensor: &Tensor,
    source: &str,
    component: &str,
    from: &str,
) -> TensorDeclaration {
    TensorDeclaration {
        key: key.into(),
        dtype: tensor.dtype,
        shape: tensor.shape.clone(),
        encoding: tensor.encoding.clone(),
        parts: tensor
            .parts
            .iter()
            .map(|(role, part)| PartDeclaration {
                role: role.clone(),
                dtype: part.dtype,
                shape: part.shape.clone(),
                source: Some(PartSource {
                    source: source.into(),
                    component: component.into(),
                    tensor: from.into(),
                    role: role.clone(),
                }),
            })
            .collect(),
    }
}

/// Factor pairs by target path: (A key, A, B key, B, alpha).
type Pairs<'h> = BTreeMap<String, BTreeMap<&'static str, (String, &'h Tensor)>>;

fn pairs<'h>(factors: &'h [(String, Tensor)]) -> io::Result<Pairs<'h>> {
    let mut pairs: Pairs = BTreeMap::new();
    for (key, tensor) in factors {
        let (path, role) = [
            (".lora_A.weight", "a"),
            (".lora_B.weight", "b"),
            (".alpha", "alpha"),
        ]
        .into_iter()
        .find_map(|(suffix, role)| key.strip_suffix(suffix).map(|path| (path, role)))
        .ok_or_else(|| refuse("key", format!("unsupported adapter tensor {key:?}")))?;
        if path.is_empty() {
            return Err(refuse(
                "target",
                "an adapter target must name a child linear layer",
            ));
        }
        pairs
            .entry(path.into())
            .or_default()
            .insert(role, (key.clone(), tensor));
    }
    Ok(pairs)
}

fn infer_component(base: &Header, factors: &[(String, Tensor)]) -> io::Result<String> {
    let pairs = pairs(factors)?;
    let fits = |tensors: &[(String, Tensor)]| {
        !pairs.is_empty()
            && pairs.iter().all(|(path, pair)| {
                match (
                    pair.get("a"),
                    pair.get("b"),
                    lookup(tensors, &format!("{path}.weight")),
                ) {
                    (Some((_, a)), Some((_, b)), Some(weight)) => {
                        a.shape.len() == 2
                            && b.shape.len() == 2
                            && [b.shape[0], a.shape[1]] == weight.shape[..]
                    }
                    _ => false,
                }
            })
    };
    let matches: Vec<_> = base
        .components
        .iter()
        .filter(|(_, t)| fits(t))
        .map(|(n, _)| n.clone())
        .collect();
    match matches.as_slice() {
        [one] => Ok(one.clone()),
        _ => Err(refuse(
            "component",
            format!(
                "select one adapter component; compatible choices: {}",
                if matches.is_empty() {
                    "none".into()
                } else {
                    matches.join(", ")
                }
            ),
        )),
    }
}

/// The derivation and its graph config, refusing unsupported keys before anything is written.
fn prepare(
    store: &Store,
    base: (&ObjectRef, &Header),
    adapters: &[(Selection, Header)],
) -> io::Result<(Declaration, Vec<u8>)> {
    let (base_ref, base) = base;
    if base.configs.iter().any(|(name, _)| name == GRAPH_CONFIG) {
        return Err(refuse(
            "nested",
            "select the original base and one complete ordered adapter stack",
        ));
    }
    let mut sources = vec![Source {
        alias: "base".into(),
        manifest: base_ref.clone(),
    }];
    let mut drops: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut additions: BTreeMap<String, Vec<TensorDeclaration>> = BTreeMap::new();
    let mut graph = Graph {
        format: FORMAT,
        layers: vec![],
        adapters: vec![],
    };
    for (index, (selection, adapter)) in adapters.iter().enumerate() {
        if !selection.strength.is_finite() {
            return Err(refuse("scale", "adapter strength must be finite"));
        }
        // The factor component: the one the selection names; else the one named like the
        // selected base component (TensorFS `sdxl.lora/1` names a LoRA's components after
        // the base's: `unet`, `text_encoder`, `text_encoder_2`), else `adapter`, else the
        // adapter's only component. Other components are other selections' to read.
        let named = |name: &str| adapter.components.iter().find(|(n, _)| n == name);
        let found = if selection.source_component.is_empty() {
            named(&selection.component).or_else(|| named("adapter")).or(
                match adapter.components.as_slice() {
                    [one] => Some(one),
                    _ => None,
                },
            )
        } else {
            named(&selection.source_component)
        };
        let Some((source_component, factors)) = found else {
            let has: Vec<_> = adapter.components.iter().map(|(n, _)| n.as_str()).collect();
            return Err(refuse(
                "components",
                format!(
                    "the adapter has no factor component for this selection; it has: {}",
                    has.join(", ")
                ),
            ));
        };
        let source_component = source_component.as_str();
        if adapter
            .configs
            .iter()
            .any(|(name, _)| name != "normalization")
        {
            return Err(refuse(
                "contract",
                "adapter introduces a separate inference configuration",
            ));
        }
        if factors.is_empty() {
            return Err(refuse("empty", "adapter contains no factors"));
        }
        let component = if selection.component.is_empty() {
            infer_component(base, factors)?
        } else {
            selection.component.clone()
        };
        let base_tensors = tensors(base, &component)
            .ok_or_else(|| refuse("component", format!("base has no component {component:?}")))?;
        let mut scalars = BTreeMap::new();
        for (key, tensor) in factors {
            if let Some(value) = validate(store, source_component, key, tensor)? {
                if let Some(path) = key.strip_suffix(".alpha") {
                    scalars.insert(path.to_string(), value);
                }
            } else if key.ends_with(".alpha") {
                return Err(refuse("alpha", format!("{key}: alpha must be a scalar")));
            }
        }
        graph.adapters.push(AdapterRef {
            reference: id(&selection.manifest),
            scale: selection.strength,
            kind: "lora",
            component: component.clone(),
            source_component: source_component.into(),
            family: "",
        });
        let alias = format!("adapter_{index}");
        for (path, pair) in pairs(factors)? {
            let (Some((a_key, a)), Some((b_key, b))) = (pair.get("a"), pair.get("b")) else {
                return Err(refuse(
                    "pair",
                    format!("{component}.{path}: incomplete A/B pair"),
                ));
            };
            let weight = lookup(base_tensors, &format!("{path}.weight"));
            if weight.is_some_and(|w| w.shape.len() != 2) {
                return Err(refuse(
                    "target_type",
                    format!("{component}.{path}: only ordinary linear LoRA targets are supported"),
                ));
            }
            let fits = weight.is_some_and(|w| {
                matches!(w.dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32)
                    && a.shape.len() == 2
                    && b.shape.len() == 2
                    && a.shape[0] > 0
                    && a.shape[0] == b.shape[1]
                    && [b.shape[0], a.shape[1]] == w.shape[..]
                    && a.dtype == b.dtype
            });
            if !fits {
                return Err(refuse(
                    "shape",
                    format!("{component}.{path}: factors do not fit the base projection"),
                ));
            }
            let rank = a.shape[0];
            let alpha = scalars.get(&path).copied().unwrap_or(rank as f64);
            if !(alpha / rank as f64 * selection.strength).is_finite() {
                return Err(refuse(
                    "scale",
                    format!("{component}.{path}: effective scale is not finite"),
                ));
            }
            if selection.strength == 0.0 {
                continue;
            }
            if !sources.iter().any(|s| s.alias == alias) {
                sources.push(Source {
                    alias: alias.clone(),
                    manifest: selection.manifest.clone(),
                });
            }
            for suffix in ["weight", "bias"] {
                let key = format!("{path}.{suffix}");
                if let Some(tensor) = lookup(base_tensors, &key) {
                    let dropped = drops.entry(component.clone()).or_default();
                    if !dropped.contains(&key) {
                        dropped.push(key.clone());
                        additions.entry(component.clone()).or_default().push(graft(
                            &format!("{path}.base_layer.{suffix}"),
                            tensor,
                            "base",
                            &component,
                            &key,
                        ));
                    }
                }
            }
            for (role, key, tensor) in [("A", a_key, a), ("B", b_key, b)] {
                additions.entry(component.clone()).or_default().push(graft(
                    &format!("{path}.lora_{role}.{alias}.weight"),
                    tensor,
                    &alias,
                    source_component,
                    key,
                ));
            }
            graph.layers.push(Linear {
                component: component.clone(),
                target: path.clone(),
                adapter: alias.clone(),
                rank,
                alpha,
                strength: selection.strength,
                dtype: a.dtype.name(),
            });
        }
    }
    let mut components = vec![];
    let mut order = vec![];
    for (name, tensors) in &base.components {
        let dropped = drops.remove(name).unwrap_or_default();
        let add = additions.remove(name).unwrap_or_default();
        order.extend(
            tensors
                .iter()
                .filter(|(k, _)| !dropped.contains(k))
                .map(|(k, _)| (name.clone(), k.clone())),
        );
        order.extend(add.iter().map(|t| (name.clone(), t.key.clone())));
        let mut drop = dropped;
        drop.sort();
        components.push(ComponentDeclaration {
            target: name.clone(),
            source: Some("base".into()),
            source_component: Some(name.clone()),
            drop,
            add,
        });
    }
    let mut configs: Vec<_> = base
        .configs
        .iter()
        .map(|(name, _)| ConfigDeclaration::Copy {
            target: name.clone(),
            source: "base".into(),
            source_config: name.clone(),
        })
        .collect();
    configs.push(ConfigDeclaration::Add {
        target: GRAPH_CONFIG.into(),
    });
    let config = serde_json_canonicalizer::to_vec(&graph).map_err(native)?;
    let declaration = Declaration {
        work_fingerprint: None,
        sources,
        components,
        configs,
        files: vec![],
        objects: vec![],
        order,
        max_new_bytes: config.len() as u64,
    };
    Ok((declaration, config))
}

/// The adapter view of `base` with `selections`, kept as the local repository
/// `local/adapters-<id>`: the repository is its only root (the composition's derived root
/// is released once it lands) and `evict` reclaims it. A held view is reused.
pub fn compose(store: &Store, base: &ObjectRef, selections: &[Selection]) -> io::Result<Composed> {
    let rows: Vec<_> = selections
        .iter()
        .map(|s| {
            serde_json::json!([
                id(&s.manifest),
                s.component,
                s.source_component,
                s.strength.to_string()
            ])
        })
        .collect();
    let identity = tensorfs_core::sha256::hex_digest(
        &serde_json_canonicalizer::to_vec(&serde_json::json!([FORMAT, id(base), rows]))
            .map_err(native)?,
    );
    let name = format!("{VIEW_PREFIX}{}", &identity[..40]);
    let meta = Meta::open(store).map_err(native)?;
    // A transaction id never reopens once committed or abandoned, so a view whose bytes were
    // collected (or whose composition died mid-way) is composed again under the next one.
    let mut generation = 0u64;
    let (transaction, manifest) = loop {
        let transaction = match generation {
            0 => format!("sha256:{identity}"),
            n => format!(
                "sha256:{}",
                tensorfs_core::sha256::hex_digest(format!("{identity}/{n}").as_bytes())
            ),
        };
        generation += 1;
        match derived::lookup(store, &meta, &transaction).map_err(native)? {
            Lookup::Committed(done) => {
                let manifest = done.receipt().manifest.clone();
                if store.manifest_path(&manifest.sha256).is_file() {
                    break (transaction, manifest);
                }
            }
            Lookup::Open { writer_session } => {
                if let Some(session) = writer_session {
                    derived::fence(&meta, &transaction, session).map_err(native)?;
                }
                derived::abandon(store, &meta, &transaction).map_err(native)?;
            }
            Lookup::Abandoned => {}
            Lookup::Absent => {
                let manifest = derive(store, &meta, &transaction, base, selections)?;
                break (transaction, manifest);
            }
        }
    };
    let repo = RepositoryName::new("local", &name).map_err(native)?;
    let path = store.repository_path(&repo);
    let replace = Mutation::ReplaceLocal {
        repo,
        version: identity,
        manifest: manifest.clone(),
    };
    // A concurrent `evict` can change the repository between the read and the write.
    let mut attempts = 0;
    while let Err(error) = store.apply_repository(
        std::fs::read(&path).ok().as_deref(),
        &replace,
        &Fault::default(),
    ) {
        attempts += 1;
        if error.code != tensorfs_core::err::Code::REPOSITORY_CONFLICT || attempts == 3 {
            return Err(native(error));
        }
    }
    // Last use, for `evict`.
    std::fs::File::options()
        .write(true)
        .open(&path)
        .and_then(|file| file.set_modified(SystemTime::now()))?;
    if let Err(error) = derived::dispose(store, &meta, &transaction) {
        eprintln!("adapter view {name}: composition root not released: {error}");
    }
    Ok(Composed {
        repository: format!("local/{name}"),
        manifest,
    })
}

/// One composition through the derived writer: no byte copied, one config added.
fn derive(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    base: &ObjectRef,
    selections: &[Selection],
) -> io::Result<ObjectRef> {
    let base_header = header(store, base)?;
    let adapters = selections
        .iter()
        .map(|s| Ok((s.clone(), header(store, &s.manifest)?)))
        .collect::<io::Result<Vec<_>>>()?;
    let (mut declaration, config) = prepare(store, (base, &base_header), &adapters)?;
    declaration.work_fingerprint = Some(transaction.to_string());
    let session = (tensorfs_core::meta::now_nanos_unique() / 1_000_000) as u64 & ((1 << 53) - 1);
    let begun =
        derived::begin(store, meta, transaction, session, declaration, None).map_err(native)?;
    let written = derived::add_config(
        meta,
        transaction,
        session,
        GRAPH_CONFIG,
        &mut config.as_slice(),
    )
    .and_then(|()| derived::commit(store, meta, transaction, session));
    for (_, lease) in begun.source_leases {
        let _ = lease.release(meta);
    }
    let _ = begun.writer_hold.release(meta);
    match written {
        Ok(receipt) => Ok(receipt.manifest),
        Err(error) => {
            let _ = derived::fence(meta, transaction, session);
            let _ = derived::abandon(store, meta, transaction);
            Err(native(error))
        }
    }
}

/// Adapter views manage themselves like the machine's other caches (`reclaim`): a view
/// unused for the TTL loses its repository, and GC then takes its header and manifest. A
/// view a live executor, an unfinished run or a preparation names (`keep`) is never
/// touched. Returns the views removed.
pub fn evict(store: &Store, keep: &[String]) -> io::Result<usize> {
    let mut removed = 0;
    let directory = store.root().join("repos").join("local");
    let entries = match std::fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        let Some(name) = file
            .strip_suffix(".json")
            .filter(|name| name.starts_with(VIEW_PREFIX))
        else {
            continue;
        };
        let age = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .unwrap_or_default();
        if age <= crate::reclaim::TTL {
            continue;
        }
        let Ok(body) = std::fs::read(entry.path()) else {
            continue;
        };
        let repository = Repository::parse(&body).map_err(native)?;
        let manifest = &repository.local_checkpoint().map_err(native)?.manifest;
        if keep
            .iter()
            .any(|k| k.trim_start_matches("sha256:") == manifest.sha256)
        {
            continue;
        }
        let repo = RepositoryName::new("local", name).map_err(native)?;
        match store.apply_repository(
            Some(&body),
            &Mutation::DeleteRepository { repo },
            &Fault::default(),
        ) {
            Ok(_) => removed += 1,
            // Composed again since it was read: it is in use.
            Err(error) if error.code == tensorfs_core::err::Code::REPOSITORY_CONFLICT => {}
            Err(error) => return Err(native(error)),
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    type Tensors<'a> = [(&'a str, Vec<u64>, Vec<u8>)];

    /// A real one-component checkpoint written through the TensorFS derived writer.
    fn checkpoint(
        store: &Store,
        meta: &Meta,
        component: &str,
        tensors: &Tensors,
        tag: u8,
    ) -> ObjectRef {
        model(store, meta, &[(component, tensors)], tag)
    }

    /// A real checkpoint of several components (no sources).
    fn model(store: &Store, meta: &Meta, components: &[(&str, &Tensors)], tag: u8) -> ObjectRef {
        let transaction = format!("sha256:{}", format!("{tag:02x}").repeat(32));
        let all = || {
            components
                .iter()
                .flat_map(|(component, tensors)| tensors.iter().map(move |t| (*component, t)))
        };
        let declaration = Declaration {
            work_fingerprint: Some(transaction.clone()),
            sources: vec![],
            components: components
                .iter()
                .map(|(component, tensors)| ComponentDeclaration {
                    target: (*component).into(),
                    source: None,
                    source_component: None,
                    drop: vec![],
                    add: tensors
                        .iter()
                        .map(|(key, shape, _)| TensorDeclaration {
                            key: (*key).into(),
                            dtype: Dtype::F32,
                            shape: shape.clone(),
                            encoding: PLAIN.into(),
                            parts: vec![PartDeclaration {
                                role: "value".into(),
                                dtype: Dtype::F32,
                                shape: shape.clone(),
                                source: None,
                            }],
                        })
                        .collect(),
                })
                .collect(),
            configs: vec![],
            files: vec![],
            objects: vec![],
            order: all()
                .map(|(component, (key, _, _))| (component.to_string(), key.to_string()))
                .collect(),
            max_new_bytes: all().map(|(_, t)| t.2.len() as u64).sum(),
        };
        let begun = derived::begin(store, meta, &transaction, 1, declaration, None).unwrap();
        for (component, (key, _, bytes)) in all() {
            derived::add_part(
                store,
                meta,
                &transaction,
                1,
                (component, key, "value"),
                &mut bytes.as_slice(),
            )
            .unwrap();
        }
        let receipt = derived::commit(store, meta, &transaction, 1).unwrap();
        let _ = begun.writer_hold.release(meta);
        receipt.manifest
    }

    /// Back-date a repository's last use.
    fn unused_for(store: &Store, repository: &str, age: std::time::Duration) {
        let (org, name) = repository.split_once('/').unwrap();
        std::fs::File::options()
            .write(true)
            .open(store.repository_path(&RepositoryName::new(org, name).unwrap()))
            .unwrap()
            .set_modified(SystemTime::now() - age)
            .unwrap();
    }

    fn put_local(store: &Store, name: &str, manifest: &ObjectRef) {
        store
            .apply_repository(
                None,
                &Mutation::ReplaceLocal {
                    repo: RepositoryName::new("local", name).unwrap(),
                    version: manifest.sha256.clone(),
                    manifest: manifest.clone(),
                },
                &Fault::default(),
            )
            .unwrap();
    }

    #[test]
    fn a_view_is_rooted_only_by_its_repository_which_evicts_itself() {
        let root = std::env::temp_dir().join(format!("adapter-views-{}", uuid::Uuid::new_v4()));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let base = checkpoint(
            &store,
            &meta,
            "unet",
            &[("proj.weight", vec![4, 3], f32s(&[0.5; 12]))],
            0xb1,
        );
        let lora = checkpoint(
            &store,
            &meta,
            "adapter",
            &[
                ("proj.lora_A.weight", vec![2, 3], f32s(&[0.25; 6])),
                ("proj.lora_B.weight", vec![4, 2], f32s(&[0.75; 8])),
            ],
            0xa1,
        );
        // Base and adapter are held by their repositories alone, as downloads are.
        put_local(&store, "base", &base);
        put_local(&store, "lora", &lora);
        for tag in [0xb1u8, 0xa1] {
            derived::dispose(
                &store,
                &meta,
                &format!("sha256:{}", format!("{tag:02x}").repeat(32)),
            )
            .unwrap();
        }
        let selection = [Selection {
            manifest: lora.clone(),
            component: String::new(),
            source_component: String::new(),
            strength: 1.0,
        }];
        let held = |m: &ObjectRef| store.manifest_path(&m.sha256).is_file();
        let collect = || tensorfs_core::gc::collect(store.root(), false).unwrap();
        let view = compose(&store, &base, &selection).unwrap();
        let hour = std::time::Duration::from_secs(3600);

        // Young, or in use: kept.
        unused_for(&store, &view.repository, hour);
        assert_eq!(evict(&store, &[]).unwrap(), 0);
        unused_for(&store, &view.repository, crate::reclaim::TTL + hour);
        assert_eq!(evict(&store, &[id(&view.manifest)]).unwrap(), 0);
        // Unused past the TTL: its repository goes, then GC takes the view and nothing else.
        assert_eq!(evict(&store, &[]).unwrap(), 1);
        collect();
        assert!(!held(&view.manifest), "the view outlived its repository");
        assert!(held(&base) && held(&lora));

        // Composed again after collection, under the next transaction.
        let again = compose(&store, &base, &selection).unwrap();
        assert_eq!(again, view);
        assert!(held(&again.manifest));

        // The view pins no source: once the adapter's repository is gone, GC takes the
        // adapter's manifest while the factors the view grafts stay.
        store
            .apply_repository(
                Some(
                    &std::fs::read(
                        store.repository_path(&RepositoryName::new("local", "lora").unwrap()),
                    )
                    .unwrap(),
                ),
                &Mutation::DeleteRepository {
                    repo: RepositoryName::new("local", "lora").unwrap(),
                },
                &Fault::default(),
            )
            .unwrap();
        collect();
        assert!(
            !held(&lora),
            "the view's composition still roots its adapter"
        );
        assert!(held(&again.manifest) && held(&base));
        let grafted = header(&store, &again.manifest).unwrap();
        for (_, tensor) in tensors(&grafted, "unet").unwrap() {
            for (_, part) in &tensor.parts {
                if let Body::Segments(objects) = &part.body {
                    assert!(objects
                        .iter()
                        .all(|o| store.object_path(&o.sha256).is_file()));
                }
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// The machine's periodic sweep reaches the store: a view unused for the TTL goes, and
    /// TensorFS's GC takes it and what nothing references.
    #[test]
    fn the_machine_sweep_evicts_views_and_collects_them() {
        let root = std::env::temp_dir().join(format!("adapter-sweep-{}", uuid::Uuid::new_v4()));
        let state = root.join("state");
        let service = crate::service::Service::open(&state, &root.join("generations"), 1).unwrap();
        let store = std::sync::Arc::new(Store::ensure(&state.join("tensorfs")).unwrap());
        service.configure_publisher(
            crate::published::Publisher::new(
                &state.join("published"),
                Default::default(),
                store.clone(),
            )
            .unwrap(),
        );
        let meta = Meta::open(&store).unwrap();
        let base = checkpoint(
            &store,
            &meta,
            "unet",
            &[("proj.weight", vec![4, 3], f32s(&[0.5; 12]))],
            0xb2,
        );
        let lora = checkpoint(
            &store,
            &meta,
            "adapter",
            &[
                ("proj.lora_A.weight", vec![2, 3], f32s(&[0.25; 6])),
                ("proj.lora_B.weight", vec![4, 2], f32s(&[0.75; 8])),
            ],
            0xa2,
        );
        put_local(&store, "base", &base);
        put_local(&store, "lora", &lora);
        for tag in [0xb2u8, 0xa2] {
            derived::dispose(
                &store,
                &meta,
                &format!("sha256:{}", format!("{tag:02x}").repeat(32)),
            )
            .unwrap();
        }
        let view = compose(
            &store,
            &base,
            &[Selection {
                manifest: lora,
                component: String::new(),
                source_component: String::new(),
                strength: 1.0,
            }],
        )
        .unwrap();
        unused_for(
            &store,
            &view.repository,
            crate::reclaim::TTL + std::time::Duration::from_secs(60),
        );
        let garbage = vec![7u8; 1 << 20];
        let garbage = store
            .put_stream(&mut garbage.as_slice(), None, &Fault::default())
            .unwrap()
            .obj;
        let swept = service.reclaim();
        assert_eq!(swept.adapter_views, 1, "{swept:?}");
        assert!(swept.store_bytes >= garbage.length, "{swept:?}");
        assert!(!store.object_path(&garbage.sha256).exists());
        assert!(!store.manifest_path(&view.manifest.sha256).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn plain_is_the_registry_plain_encoding() {
        let plain = tensorfs_core::registry::seeds()
            .into_iter()
            .find(|s| s.alias == "plain/1")
            .unwrap()
            .spec
            .object_id();
        assert_eq!(plain, PLAIN);
    }

    #[test]
    fn adapters_graft_onto_the_base_by_reference_with_their_ordered_graph() {
        let root = std::env::temp_dir().join(format!("adapter-views-{}", uuid::Uuid::new_v4()));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let base = checkpoint(
            &store,
            &meta,
            "unet",
            &[
                ("proj.weight", vec![4, 3], f32s(&[0.5; 12])),
                ("proj.bias", vec![4], f32s(&[0.1; 4])),
                ("norm.weight", vec![4], f32s(&[1.0; 4])),
            ],
            0xb0,
        );
        let lora = checkpoint(
            &store,
            &meta,
            "adapter",
            &[
                ("proj.lora_A.weight", vec![2, 3], f32s(&[0.25; 6])),
                ("proj.lora_B.weight", vec![4, 2], f32s(&[0.75; 8])),
                ("proj.alpha", vec![], f32s(&[4.0])),
            ],
            0xa0,
        );
        let select = |strength: f64| Selection {
            manifest: lora.clone(),
            component: String::new(),
            source_component: String::new(),
            strength,
        };
        let composed = compose(&store, &base, &[select(0.8)]).unwrap();
        assert!(composed.repository.starts_with("local/adapters-"));
        assert_eq!(
            compose(&store, &base, &[select(0.8)]).unwrap(),
            composed,
            "a held view is reused"
        );

        let view = header(&store, &composed.manifest).unwrap();
        let unet = tensors(&view, "unet").unwrap();
        let keys: Vec<_> = unet.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "norm.weight",
                "proj.base_layer.weight",
                "proj.base_layer.bias",
                "proj.lora_A.adapter_0.weight",
                "proj.lora_B.adapter_0.weight"
            ]
        );
        let original = header(&store, &base).unwrap();
        let segments = |h: &Header, k: &str| {
            lookup(tensors(h, "unet").unwrap(), k).unwrap().parts[0]
                .1
                .body
                .clone()
        };
        assert_eq!(
            segments(&view, "proj.base_layer.weight"),
            segments(&original, "proj.weight"),
            "no byte copied"
        );
        let graph: serde_json::Value = serde_json::from_slice(
            &view
                .configs
                .iter()
                .find(|(n, _)| n == GRAPH_CONFIG)
                .unwrap()
                .1,
        )
        .unwrap();
        assert_eq!(graph["format"], FORMAT);
        assert_eq!(
            graph["layers"][0],
            serde_json::json!({"component":"unet","target":"proj","adapter":"adapter_0","rank":2,"alpha":4,"strength":0.8,"dtype":"f32"})
        );
        assert_eq!(graph["adapters"][0]["ref"], id(&lora));
        assert!(std::fs::metadata(
            store.repository_path(
                &RepositoryName::new("local", composed.repository.trim_start_matches("local/"))
                    .unwrap()
            )
        )
        .is_ok());

        // A zero strength keeps its reference but grafts nothing; a foreign layout is refused.
        let quiet = header(
            &store,
            &compose(&store, &base, &[select(0.0)]).unwrap().manifest,
        )
        .unwrap();
        assert!(lookup(tensors(&quiet, "unet").unwrap(), "proj.weight").is_some());
        let kohya = checkpoint(
            &store,
            &meta,
            "adapter",
            &[(
                "lora_unet_proj.lora_down.weight",
                vec![2, 3],
                f32s(&[0.0; 6]),
            )],
            0xc0,
        );
        let refused = compose(
            &store,
            &base,
            &[Selection {
                manifest: kohya,
                ..select(1.0)
            }],
        )
        .unwrap_err();
        assert!(refused.to_string().starts_with("adapter_key"), "{refused}");
        let _ = std::fs::remove_dir_all(root);
    }
    #[test]
    fn a_lora_whose_components_are_named_after_the_base_grafts_each_onto_its_own() {
        // TensorFS `sdxl.lora/1` names a LoRA's components `unet`, `text_encoder`, ...: there
        // is no `adapter` component, and a LoRA with text towers has several.
        let root = std::env::temp_dir().join(format!("adapter-views-{}", uuid::Uuid::new_v4()));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let unet: &Tensors = &[("proj.weight", vec![4, 3], f32s(&[0.5; 12]))];
        let tower: &Tensors = &[("fc.weight", vec![2, 5], f32s(&[0.5; 10]))];
        let base = model(
            &store,
            &meta,
            &[("unet", unet), ("text_encoder", tower)],
            0xb1,
        );
        let unet_factors: &Tensors = &[
            ("proj.lora_A.weight", vec![2, 3], f32s(&[0.25; 6])),
            ("proj.lora_B.weight", vec![4, 2], f32s(&[0.75; 8])),
        ];
        let tower_factors: &Tensors = &[
            ("fc.lora_A.weight", vec![1, 5], f32s(&[0.25; 5])),
            ("fc.lora_B.weight", vec![2, 1], f32s(&[0.75; 2])),
        ];
        let both = model(
            &store,
            &meta,
            &[("unet", unet_factors), ("text_encoder", tower_factors)],
            0xa1,
        );
        let on = |manifest: &ObjectRef, component: &str| Selection {
            manifest: manifest.clone(),
            component: component.into(),
            source_component: String::new(),
            strength: 1.0,
        };
        let composed = compose(
            &store,
            &base,
            &[on(&both, "unet"), on(&both, "text_encoder")],
        )
        .unwrap();
        let view = header(&store, &composed.manifest).unwrap();
        let keys = |component: &str| -> Vec<String> {
            tensors(&view, component)
                .unwrap()
                .iter()
                .map(|(k, _)| k.clone())
                .collect()
        };
        assert_eq!(
            keys("unet"),
            [
                "proj.base_layer.weight",
                "proj.lora_A.adapter_0.weight",
                "proj.lora_B.adapter_0.weight"
            ]
        );
        assert_eq!(
            keys("text_encoder"),
            [
                "fc.base_layer.weight",
                "fc.lora_A.adapter_1.weight",
                "fc.lora_B.adapter_1.weight"
            ]
        );
        // A one-component LoRA named after the base needs no component at all.
        let only = checkpoint(&store, &meta, "unet", unet_factors, 0xa2);
        let inferred = compose(&store, &base, &[on(&only, "")]).unwrap();
        let view = header(&store, &inferred.manifest).unwrap();
        assert!(lookup(
            tensors(&view, "unet").unwrap(),
            "proj.lora_A.adapter_0.weight"
        )
        .is_some());
        // A selection the adapter has no component for names what it has.
        let refused = compose(&store, &base, &[on(&both, "vae")]).unwrap_err();
        assert!(
            refused.to_string().contains("it has: unet, text_encoder"),
            "{refused}"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
