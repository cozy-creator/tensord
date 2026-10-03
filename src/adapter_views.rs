//! Caller LoRA adapters as one zero-copy TensorFS derivation over the base checkpoint (the
//! Python worker's adapter view, `lora_composition.prepare`): base `<path>.weight/.bias`
//! become `<path>.base_layer.*`, factors graft in as `<path>.lora_{A,B}.adapter_<i>.weight`,
//! and the `model_adapters` config holds the ordered graph the executor binds with PEFT.
//! No byte is copied or computed. Factors must already be canonical PEFT keys
//! (`<path>.lora_A.weight`, `<path>.lora_B.weight`, optional scalar `<path>.alpha`): foreign
//! layouts are normalized when a model is ingested, never here.
use serde::Serialize;
use std::{collections::BTreeMap, io, io::Read};
use tensorfs_core::{
    checkpoint::load_header,
    derived::{
        self, ComponentDeclaration, ConfigDeclaration, Declaration, Lookup, PartDeclaration,
        PartSource, Source, TensorDeclaration,
    },
    dtype::Dtype,
    header::{Body, Header, Tensor},
    ids::ObjectRef,
    meta::Meta,
    repository::{Mutation, RepositoryName},
    store::{Fault, Store},
};

pub const FORMAT: &str = "cozy.model.lora/1";
pub const GRAPH_CONFIG: &str = "model_adapters";
const PLAIN: &str = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f";

/// One caller adapter, in order. An empty `component` is inferred when exactly one base
/// component fits every factor pair.
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
        let source_component = if selection.source_component.is_empty() {
            "adapter"
        } else {
            &selection.source_component
        };
        if adapter.components.len() != 1 || adapter.components[0].0 != source_component {
            return Err(refuse(
                "components",
                "an adapter must contain only its selected factor component",
            ));
        }
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
        let factors = &adapter.components[0].1;
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

/// The adapter view of `base` with `selections`, composed once and kept as the local
/// repository `local/adapters-<id>`; a held view is reused.
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
    let transaction = format!("sha256:{identity}");
    let name = format!("adapters-{}", &identity[..40]);
    let meta = Meta::open(store).map_err(native)?;
    let manifest = match derived::lookup(store, &meta, &transaction).map_err(native)? {
        Lookup::Committed(done) => done.receipt().manifest.clone(),
        previous => {
            if let Lookup::Open { writer_session } = previous {
                if let Some(session) = writer_session {
                    derived::fence(&meta, &transaction, session).map_err(native)?;
                }
                derived::abandon(store, &meta, &transaction).map_err(native)?;
            }
            let base_header = header(store, base)?;
            let adapters = selections
                .iter()
                .map(|s| Ok((s.clone(), header(store, &s.manifest)?)))
                .collect::<io::Result<Vec<_>>>()?;
            let (mut declaration, config) = prepare(store, (base, &base_header), &adapters)?;
            declaration.work_fingerprint = Some(transaction.clone());
            let session =
                (tensorfs_core::meta::now_nanos_unique() / 1_000_000) as u64 & ((1 << 53) - 1);
            let begun = derived::begin(store, &meta, &transaction, session, declaration, None)
                .map_err(native)?;
            let written = derived::add_config(
                &meta,
                &transaction,
                session,
                GRAPH_CONFIG,
                &mut config.as_slice(),
            )
            .and_then(|()| derived::commit(store, &meta, &transaction, session));
            for (_, lease) in begun.source_leases {
                let _ = lease.release(&meta);
            }
            let _ = begun.writer_hold.release(&meta);
            match written {
                Ok(receipt) => receipt.manifest,
                Err(error) => {
                    let _ = derived::fence(&meta, &transaction, session);
                    let _ = derived::abandon(store, &meta, &transaction);
                    return Err(native(error));
                }
            }
        }
    };
    let repo = RepositoryName::new("local", &name).map_err(native)?;
    let current = std::fs::read(store.repository_path(&repo)).ok();
    store
        .apply_repository(
            current.as_deref(),
            &Mutation::ReplaceLocal {
                repo,
                version: identity,
                manifest: manifest.clone(),
            },
            &Fault::default(),
        )
        .map_err(native)?;
    Ok(Composed {
        repository: format!("local/{name}"),
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A real checkpoint written through the TensorFS derived writer (no sources).
    fn checkpoint(
        store: &Store,
        meta: &Meta,
        component: &str,
        tensors: &[(&str, Vec<u64>, Vec<u8>)],
        tag: u8,
    ) -> ObjectRef {
        let transaction = format!("sha256:{}", format!("{tag:02x}").repeat(32));
        let declaration = Declaration {
            work_fingerprint: Some(transaction.clone()),
            sources: vec![],
            components: vec![ComponentDeclaration {
                target: component.into(),
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
            }],
            configs: vec![],
            files: vec![],
            objects: vec![],
            order: tensors
                .iter()
                .map(|(key, _, _)| (component.to_string(), key.to_string()))
                .collect(),
            max_new_bytes: tensors.iter().map(|t| t.2.len() as u64).sum(),
        };
        let begun = derived::begin(store, meta, &transaction, 1, declaration, None).unwrap();
        for (key, _, bytes) in tensors {
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
}
