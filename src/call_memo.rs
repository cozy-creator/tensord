//! The machine memo of memoized callables (`memoize=True`). A run of one, top-level or a
//! child call, is answered from the success this machine holds for its signer under the same
//! computation, and starts nothing; a success is held (`Objects::hold_memo`). A computation is
//! the callable's operation identity, its request as given (files by content), its attention
//! pin and bits and the models it runs with. An author's condition (`memo_requires`) names the
//! parameters a request must give for it to be memoizable at all.
use crate::{api::domain::ModelChoice, catalog::Generation, gpu_service::Plan, journal::InputFile};
use serde_json::{json, Value};
use tensorfs_core::sha256;

pub enum Memo {
    /// Not memoizable: not declared so, or nothing identifies its code.
    Off,
    /// Memoizable, but the request does not meet its author's condition: why.
    Unmet(String),
    /// Its computation (`sha256:<hex>`).
    Key(String),
}

/// The memo of a call of `invocable` (its declaration) in App `application` of `generation`.
pub fn of(
    generation: &Generation,
    application: &str,
    invocable: &Value,
    request: &Value,
    files: &[InputFile],
    (attention, bits): (&str, u32),
    models: Value,
) -> Memo {
    if invocable["memoize"].as_bool() != Some(true) {
        return Memo::Off;
    }
    let missing: Vec<_> = invocable["memo_requires"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|name| request.get(*name).is_none_or(Value::is_null))
        .collect();
    if !missing.is_empty() {
        return Memo::Unmet(format!("{} not given", missing.join(", ")));
    }
    let Some(operation) = operation(generation, application, invocable) else {
        return Memo::Off;
    };
    let mut files: Vec<_> = files.iter().map(|f| json!([f.input_id, f.digest, f.media_type])).collect();
    files.sort_by_key(|file| file.to_string());
    let mut computation = json!({"operation": operation, "request": checkpoints(request), "files": files,
        "attention": attention, "models": models});
    // At 16 bits every attention site is exact, whatever lane served.
    if bits != 0 {
        computation["bits"] = bits.into();
    }
    Memo::Key(format!("sha256:{}", sha256::hex_digest(&crate::boundary_json::exact(&computation))))
}

/// The request with each model artifact it carries as its checkpoint's manifest: the run that
/// produced a checkpoint is provenance, never part of a consumer's computation.
fn checkpoints(value: &Value) -> Value {
    const ARTIFACT: [&str; 4] = ["manifest", "output_slot", "producer_request_id", "tensorfs_receipt_digest"];
    match value {
        Value::Object(fields) if fields.len() == ARTIFACT.len() && ARTIFACT.iter().all(|k| fields.contains_key(*k)) => {
            json!({"manifest": fields["manifest"]})
        }
        Value::Object(fields) => Value::Object(fields.iter().map(|(k, v)| (k.clone(), checkpoints(v))).collect()),
        Value::Array(items) => Value::Array(items.iter().map(checkpoints).collect()),
        other => other.clone(),
    }
}

/// The models a call runs with: each slot's checkpoints and adapters once resolved (its raw
/// sources too), else the choices it carries.
pub fn models(plan: Option<&Plan>, choices: &[ModelChoice]) -> Value {
    match plan {
        Some(plan) => plan
            .slots
            .iter()
            .map(|slot| json!({"slot": slot.binding.model_binding_path,
                "snapshots": slot.binding.snapshots, "adapters": slot.binding.adapters}))
            .chain(plan.sources.iter().map(|source| json!({"source": source.parameter, "manifest": source.manifest})))
            .collect(),
        None => {
            let mut chosen: Vec<_> = choices
                .iter()
                .map(|choice| sha256::hex_digest(&crate::archive::encode_model_choice(choice)))
                .collect();
            chosen.sort();
            json!(chosen)
        }
    }
}

/// A memoized function's operation identity, the same on every machine: a callee's own
/// function by its package and the identity this machine read from its installed source, else
/// its package's installed files; with the module and the export. The root's own description
/// is its caller's, so its files key it. A package with neither memoizes nothing.
pub(crate) fn operation(generation: &Generation, application: &str, invocable: &Value) -> Option<String> {
    let (module, export) = (invocable["module"].as_str()?, invocable["export"].as_str()?);
    let identity = invocable["operation_identity"].as_str().filter(|identity| {
        identity
            .strip_prefix("sha256:")
            .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
    });
    let base = match application.is_empty() || application == generation.application {
        true => generation.source_digest.clone(),
        false => {
            let callee = generation.callees.iter().find(|c| c.application == application)?;
            match identity {
                Some(identity) => format!("{}\0{identity}", callee.package),
                None => callee.source_digest.clone(),
            }
        }
    };
    (!base.is_empty()).then(|| format!("sha256:{}", sha256::hex_digest(format!("{base}\0{module}\0{export}").as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A callee's memoized function is keyed by its package and the identity this machine read
    /// from its installed source: a new release that leaves it unchanged keeps its memos, one
    /// that changes it does not. Without an identity, and always for the root's own functions
    /// (whose description is its caller's), the package's installed files key it.
    #[test]
    fn a_callees_unchanged_function_keeps_its_memo_key_across_releases() {
        let invocable = |module: &str, export: &str, identity: Option<&str>| {
            let mut invocable = json!({"module": module, "export": export, "memoize": true});
            if let Some(identity) = identity {
                invocable["operation_identity"] = identity.into();
            }
            invocable
        };
        let (same, changed) = (format!("sha256:{}", "a".repeat(64)), format!("sha256:{}", "b".repeat(64)));
        let generation = |files: &str, root: &str| -> Generation {
            serde_json::from_value(json!({
                "identity": "g", "package": "local/root", "version": "1", "application": "root:app",
                "python": "/p", "source_digest": root, "interface": {},
                "callees": [{"distribution": "lib", "version": "1", "application": "lib:app",
                    "source_digest": files, "package": "local/lib", "interface": {}}],
            }))
            .unwrap()
        };
        let key = |generation: &Generation, application: &str, invocable: &Value| {
            match of(generation, application, invocable, &json!({"value": 3}), &[], ("", 0), json!([])) {
                Memo::Key(key) => key,
                _ => panic!("not memoizable"),
            }
        };
        let (before, after) = (generation("sha256:one", "sha256:root"), generation("sha256:two", "sha256:root"));
        let first = invocable("lib", "first", Some(&same));
        assert_eq!(key(&before, "lib:app", &first), key(&after, "lib:app", &first), "an unchanged function keeps its key");
        assert_ne!(
            key(&before, "lib:app", &invocable("lib", "second", Some(&same))),
            key(&after, "lib:app", &invocable("lib", "second", Some(&changed))),
            "a changed function does not"
        );
        let third = invocable("lib", "third", None);
        assert_ne!(key(&before, "lib:app", &third), key(&after, "lib:app", &third), "without an identity, the files key it");
        // The root's declared identity is not this machine's reading: its files key it.
        let own = invocable("root", "own", Some(&same));
        assert_ne!(key(&before, "", &own), key(&generation("sha256:one", "sha256:moved"), "", &own));
        assert!(matches!(of(&generation("sha256:one", ""), "", &own, &json!({}), &[], ("", 0), json!([])), Memo::Off));
    }

    /// An author's condition names parameters a request must give: absent or null, the call is
    /// not memoizable and says why; given, its value keys it. Files are keyed by their bytes.
    #[test]
    fn a_condition_unmet_is_not_memoizable_and_a_given_seed_keys_the_call() {
        let generation: Generation = serde_json::from_value(json!({"identity": "g", "package": "local/p",
            "version": "1", "application": "p:app", "python": "/p", "source_digest": "sha256:p", "interface": {}}))
            .unwrap();
        let seeded = json!({"module": "p", "export": "paint", "memoize": true, "memo_requires": ["seed", "steps"]});
        let memo = |request: Value, files: &[InputFile]| of(&generation, "", &seeded, &request, files, ("", 0), json!([]));
        for request in [json!({"prompt": "fox", "steps": 4}), json!({"prompt": "fox", "seed": null, "steps": 4})] {
            assert!(matches!(memo(request, &[]), Memo::Unmet(why) if why == "seed not given"));
        }
        assert!(matches!(memo(json!({"prompt": "fox"}), &[]), Memo::Unmet(why) if why == "seed, steps not given"));
        let key = |request: Value, files: &[InputFile]| match memo(request, files) {
            Memo::Key(key) => key,
            _ => panic!("not memoizable"),
        };
        let fox = json!({"prompt": "fox", "seed": 7, "steps": 4});
        assert_eq!(key(fox.clone(), &[]), key(fox.clone(), &[]));
        assert_ne!(key(fox.clone(), &[]), key(json!({"prompt": "fox", "seed": 8, "steps": 4}), &[]));
        assert_ne!(key(fox.clone(), &[]), key(json!({"prompt": "owl", "seed": 7, "steps": 4}), &[]));
        let file = |digest: &str| InputFile { input_id: "image".into(), digest: digest.into(), length: 1,
            media_type: "image/png".into(), order: 0 };
        assert_eq!(key(fox.clone(), &[file("sha256:a")]), key(fox.clone(), &[file("sha256:a")]));
        assert_ne!(key(fox.clone(), &[file("sha256:a")]), key(fox, &[file("sha256:b")]));
        let plain = json!({"module": "p", "export": "paint"});
        assert!(matches!(of(&generation, "", &plain, &json!({}), &[], ("", 0), json!([])), Memo::Off));
    }
}
