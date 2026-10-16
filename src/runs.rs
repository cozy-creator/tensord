//! Run sources and preparation inside a run (`cozy.machine.v1` Run). A run is accepted at
//! once and prepares inside itself: its code installs, its models resolve at the run's Hub and
//! download, and each stage is the run's progress. The Hub token lives only in memory, for that
//! preparation and the run's life (its children prepare with it); a restart before it
//! completes ends the run FAILED (`journal::PREPARING`).
use crate::{
    api::domain,
    call_memo::{self, Memo},
    execution::Stoppable,
    gpu_service::{without_gpus, KeptMember, Level, Member},
    hub,
    journal::{CallMemo, Execution, Failure, InputFile, Installation, Outcome, Preparation, ResultRecord},
    local_source::LocalSources,
    objects::{Objects, Refused},
    published::{declares_models, Providers, Publisher, Request},
    service::{Call, Service},
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use tensorfs_core::{ids::ObjectRef, transport::PullCancellation};
use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
};

pub enum Source {
    /// A published release by name (th-245); empty: the Hub's newest.
    Release { package: String, release: String },
    /// An installation this machine already holds for the signer.
    Installation(String),
    /// Unpublished code written with Write: its manifest's digest.
    Local(String),
    /// No code: a warm run that only makes its model choices (`cozy model upload`).
    Models,
}

/// A published release a run installs: its package and release.
type Published = (String, String);

/// One member of the warm set a warm run carries (`RunSpec.set`).
pub struct SetItem {
    pub source: Source,
    pub entrypoint: String,
    pub models: Vec<domain::ModelChoice>,
    pub level: Level,
    /// The item as the caller sent it, for Status.
    pub sent: Vec<u8>,
}

pub struct Spec {
    /// Prepare only (`kind: warm`): install and download, then succeed.
    pub warm: bool,
    /// `kind: remove`: the caller's installations of the source leave the machine.
    pub remove: bool,
    /// A warm run's whole warm set for its caller, replacing the previous one.
    pub set: Option<Vec<SetItem>>,
    /// A child run's parent execution (a call through its parent's seam).
    pub parent: String,
    pub source: Source,
    pub entrypoint: String,
    pub input: Value,
    pub inputs: Vec<InputFile>,
    pub models: Vec<domain::ModelChoice>,
    pub attention_kernel: String,
    pub gpus: u32,
    /// The Hub the run reads and publishes to, under this machine's authority there.
    pub hub: Option<hub::Source>,
    /// Provider tokens for source models, held in memory only.
    pub providers: Providers,
    /// `org/name`: the requester's repository for the run's weights outputs, or where a
    /// model-only warm run puts the source model it made.
    pub weights_destination: String,
    pub owner: String,
    /// The caller's catalog revision: held resolutions are reused only under it (th-245).
    pub catalog_revision: String,
    /// A child run of a callee: the other package's App its parent's environment holds.
    pub application: String,
    /// The spec's identity (its access excluded): a resubmitted id must carry the same.
    pub digest: String,
}

pub struct Runs {
    pub service: Arc<Service>,
    pub objects: Arc<Objects>,
    pub publisher: Option<Arc<Publisher>>,
    pub local: Option<Arc<LocalSources>>,
    /// On a rental: its own Hub, where a run that names none reads.
    pub own_hub: Option<hub::Source>,
    /// This machine's TLS leaf key: it trades a run's capability for its token.
    pub leaf: Option<Arc<hub::Leaf>>,
    /// Unfinished runs' preparation: their children prepare with it, and their weights are
    /// fetched and published with its access. Tokens live only here.
    pub contexts: Mutex<HashMap<String, RunContext>>,
}

/// What a run's children prepare with: its installation, Hub access, owner, catalog revision,
/// attention pin and the model choices addressed to its callables (`<entrypoint>.models.<p>`).
#[derive(Clone, Default)]
pub struct RunContext {
    pub(crate) installation: String,
    pub(crate) hub: Option<hub::Source>,
    pub(crate) providers: Providers,
    pub(crate) owner: String,
    pub(crate) catalog_revision: String,
    pub(crate) attention_kernel: String,
    pub(crate) gpus: u32,
    pub(crate) models: Vec<domain::ModelChoice>,
    pub(crate) weights_destination: String,
    /// What its weights come under before its call's executor asks (`Fetch::ahead`).
    pub(crate) ahead: Option<Arc<crate::access::Fetch>>,
}

/// A choice for the run's own slots: a bare parameter, or one under its own name.
fn own_choice(choice: &domain::ModelChoice, entrypoint: &str) -> bool {
    match choice.parameter.split_once(".models.") {
        None => true,
        Some((callable, _)) => callable == entrypoint,
    }
}

/// The slot of `package`'s `interface` a choice's parameter addresses: its full slot path
/// (`<callable>.models.<parameter>`), or that path behind the package's name, as the CLI names
/// another package's callable (`<org>/<name>/<callable>.models.<parameter>`).
fn addressed(parameter: &str, package: &str, interface: &Value) -> Option<String> {
    let path = parameter
        .strip_prefix(package)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(parameter);
    interface["entrypoints"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|entry| entry["models"].as_array().into_iter().flatten().any(|slot| slot["path"] == path))
        .then(|| path.to_string())
}

/// Whether `installation` declares a public entrypoint `name` for a warm set.
fn declares(installation: &Installation, name: &str) -> Result<bool, Refused> {
    let interface: Value = serde_json::from_slice(&installation.interface)
        .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
    let entry = interface["entrypoints"].as_array()
        .and_then(|rows| rows.iter().find(|row| row["name"] == name));
    if let Some(entry) = entry {
        public_root(entry)?;
    }
    Ok(entry.is_some())
}
fn public_root(entry: &Value) -> Result<(), Refused> {
    if entry["internal"].as_bool() == Some(true) {
        return Err(refused(
            "callable_internal",
            format!("{} is an internal package function", entry["name"].as_str().unwrap_or_default()),
        ));
    }
    Ok(())
}
fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

impl Runs {
    /// The run this id already names, if its spec is `digest`; another spec is a conflict.
    pub(crate) fn existing(&self, actor: &str, id: &str, digest: &str) -> Result<Option<Execution>, Refused> {
        match self.service.engine.get_public(actor, id) {
            Ok(run) if run.canceled_before_acceptance && run.state == crate::journal::State::Canceled => Ok(Some(run)),
            Ok(run) if run.submission.as_ref().map(|s| s.invocation_digest.as_str()) == Some(digest) => Ok(Some(run)),
            Ok(_) => Err(refused("run_id_conflict", "this run id already names another run spec")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Accepts the run (or answers the one this id already names) and starts its preparation.
    /// An accepted run is answered before its inputs are checked: they may be gone since.
    pub fn submit(self: &Arc<Self>, actor: &str, id: &str, spec: Spec) -> Result<Execution, Refused> {
        if let Some(run) = self.existing(actor, id, &spec.digest)? {
            return Ok(run);
        }
        if !spec.input.is_object() {
            return Err(refused(
                "invalid_request",
                "the payload is a JSON object of the function's parameters",
            ));
        }
        // Store GC and root release stay out from validation through acceptance.
        let custody = self.objects.guard();
        let writer = self.objects.writer_guard()?;
        let mut references = Vec::new();
        let unwritten = |input: &str| refused("input_unwritten", format!("input {input} was not written to this machine"));
        for input in &spec.inputs {
            let path = match self.objects.path(actor, &input.digest)? {
                Some((path, length)) if length == input.length => path,
                _ => return Err(unwritten(&input.input_id)),
            };
            references.push(ObjectRef {
                sha256: input.digest.trim_start_matches("sha256:").into(),
                length: input.length,
            });
            if input.media_type != crate::gpu_service::TREE_MEDIA {
                continue;
            }
            // A tree's files were written too: each its manifest names.
            let manifest: Value = serde_json::from_slice(&std::fs::read(path)?)
                .map_err(|_| refused("invalid_request", format!("input tree {} has no readable manifest", input.input_id)))?;
            for entry in manifest["entries"].as_array().into_iter().flatten() {
                let (Some(sha), Some(length)) = (entry["blob"]["sha256"].as_str(), entry["blob"]["length"].as_u64()) else {
                    return Err(refused("invalid_request", format!("input tree {} names a file without its blob", input.input_id)));
                };
                if self.objects.path(actor, &format!("sha256:{sha}"))?.is_none_or(|(_, held)| held != length) {
                    return Err(unwritten(&format!("{}/{}", input.input_id, entry["path"].as_str().unwrap_or_default())));
                }
                references.push(ObjectRef {
                    sha256: sha.into(),
                    length,
                });
            }
        }
        // A local source is refused here, not accepted to fail: the run takes custody of its
        // objects now, and a refused id is free for the client to write them and send again.
        if let Source::Local(digest) = &spec.source {
            let local = self.local.as_ref().ok_or_else(|| {
                refused("capability_unavailable", "this machine installs no local packages")
            })?;
            references.extend(local.written(actor, digest)?);
        }
        // A written file a warm run makes into a model (`object://sha256:<hex>/<file>`).
        for choice in &spec.models {
            let Some(hex) = choice.source.strip_prefix("object://").and_then(|rest| rest.split('/').next()) else {
                continue;
            };
            let sha256 = hex.trim_start_matches("sha256:").to_string();
            match self.objects.path(actor, &format!("sha256:{sha256}"))? {
                Some((_, length)) => references.push(ObjectRef { sha256, length }),
                None => return Err(unwritten(&choice.source)),
            }
        }
        self.objects.retain(&references)?;
        let package = match &spec.source {
            Source::Release { package, .. } => package.clone(),
            Source::Installation(alias) => alias.clone(),
            Source::Local(digest) => format!("local:{digest}"),
            Source::Models => "models".into(),
        };
        let draft = crate::journal::Invocation {
            package,
            generation: String::new(),
            module: String::new(),
            entrypoint: spec.entrypoint.clone(),
            input: spec.input.clone(),
            attention_kernel: spec.attention_kernel.clone(),
            gpus: spec.gpus,
            inputs: spec.inputs.clone(),
            parent: spec.parent.clone(),
            ..Default::default() // root and family: stamped at acceptance
        };
        let (record, new) = self
            .service
            .engine
            .accept_run_objects(actor, id, &spec.digest, draft, &references)
            .map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => refused("run_id_conflict", e.to_string()),
                _ => Refused::from(e),
            })?;
        drop(writer);
        drop(custody);
        if new {
            let (this, actor, run) = (self.clone(), actor.to_string(), record.id.clone());
            std::thread::Builder::new()
                .name(format!("prepare-{run}"))
                .spawn(move || {
                    let stoppable = this.service.engine.stoppable(&run);
                    // A panic (a thread the container's task cap refused) ends the run FAILED:
                    // no preparation outlives the thread that would end it.
                    let prepared = std::panic::AssertUnwindSafe(|| this.prepare(&actor, &run, spec, &stoppable));
                    let outcome = std::panic::catch_unwind(prepared).unwrap_or_else(|panic| {
                        let text = panic.downcast_ref::<String>().cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()));
                        Err(refused("preparation_failed", text.unwrap_or_else(|| "the preparation panicked".into())))
                    });
                    let ended = match outcome {
                        Ok(None) => Ok(()),
                        Ok(Some(result)) => this
                            .service
                            .engine
                            .end_preparation(&run, Outcome::Completed(result))
                            .map(drop),
                        Err(refusal) => {
                            let failure = Failure {
                                status: 3,
                                cause: 7,
                                origin: if refusal.code.starts_with("invalid")
                                    || refusal.code.ends_with("_absent")
                                {
                                    6
                                } else {
                                    3
                                },
                                code: refusal.code.into(),
                                message: refusal.message,
                            };
                            this.service
                                .engine
                                .end_preparation(&run, Outcome::Failed(failure))
                                .map(drop)
                        }
                    };
                    if let Err(error) = ended {
                        eprintln!("run {run} preparation: {error}");
                    }
                })
                .map_err(Refused::from)?;
        }
        Ok(record)
    }

    /// A warm run's model choices made present when it names no entrypoint (`cozy package
    /// install`, `cozy model download`) or no code (`cozy model upload`, `cozy run upload`): a
    /// Hub checkpoint downloaded, a provider source or a written file (`object://`) made, a
    /// checkpoint this machine holds (a run's weights output, a `local/` alias) found. With a
    /// weights destination the one choice is put there: a `local/` alias held by name, or a
    /// Hub repository under the run's capability.
    fn warm_models(
        &self,
        actor: &str,
        id: &str,
        spec: &Spec,
        observe: &(dyn Fn(&str, u64, u64) + Sync),
        stoppable: &Stoppable,
    ) -> Result<Vec<Value>, Refused> {
        let stop = Some(&stoppable.stop);
        let destination = spec.weights_destination.trim_start_matches("model://");
        if !destination.is_empty() && spec.models.len() != 1 {
            return Err(refused("invalid_request", "a weights destination takes one model"));
        }
        if spec.models.is_empty() {
            return Ok(vec![]);
        }
        let publisher = self.publisher.as_ref().ok_or_else(|| {
            refused("capability_unavailable", "this machine prepares no models")
        })?;
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let hub_access = || {
            hub.as_ref().ok_or_else(|| {
                refused("hub_access_absent", "this needs the run's Hub access, and it carries none")
            })
        };
        let mut models = vec![];
        for choice in &spec.models {
            let (manifest, mut row) = if !choice.source.is_empty() {
                let made = publisher.make_source(&choice.source, &choice.profiles, &spec.providers, stop, observe)?;
                let row = json!({"parameter": choice.parameter, "source": choice.source,
                    "resolved": made.resolved, "profiles": made.profiles,
                    "repository": made.repository, "manifest": made.manifest.id()});
                (made.manifest, row)
            } else if let (Some(name), None) = (choice.repository.strip_prefix("local/"), &choice.manifest) {
                // A local alias: the model this machine's store holds under that name.
                let (manifest, _) = tensorfs_core::source_model::held(publisher.store(), name)
                    .map_err(|e| refused("checkpoint_absent", e.to_string()))?
                    .ok_or_else(|| refused("checkpoint_absent", format!("this machine holds no local/{name}")))?;
                let row = json!({"parameter": choice.parameter,
                    "repository": choice.repository, "manifest": manifest.id()});
                (manifest, row)
            } else if choice.repository.is_empty() {
                // A checkpoint this machine already holds (a run's output).
                let manifest = choice.manifest.as_ref().ok_or_else(|| {
                    refused("invalid_request", format!("model choice {:?} names no model", choice.parameter))
                })?;
                let sha256 = tensorfs_core::sha256::hex(&manifest.digest);
                let length = std::fs::metadata(publisher.store().manifest_path(&sha256))
                    .map_err(|_| refused("checkpoint_absent", format!("this machine holds no checkpoint sha256:{sha256}")))?;
                let row = json!({"parameter": choice.parameter, "manifest": format!("sha256:{sha256}")});
                (ObjectRef { sha256, length: length.len() }, row)
            } else {
                // A Hub checkpoint by name, or the owner's by hash.
                let (gpu, width) = crate::published::machine_gpu(&self.service);
                let named = crate::published::choose(choice, &choice.parameter, (&gpu, width))
                    .map_err(|(code, message)| refused(code, message))?;
                let stage = format!("downloading {}", choice.repository);
                let (named, (hub, volume)) = publisher.download(&self.service, hub_access()?, named, &spec.catalog_revision, stop, &|done, total| {
                    observe(&stage, done, total)
                })?;
                let sha256 = named.manifest.trim_start_matches("sha256:").to_string();
                let length = std::fs::metadata(publisher.store().manifest_path(&sha256))?.len();
                let row = json!({"parameter": choice.parameter, "repository": named.repository,
                    "release": named.release, "lane": named.lane, "manifest": named.manifest,
                    "bytes": {"hub": hub, "volume": volume}});
                (ObjectRef { sha256, length }, row)
            };
            if let Some(name) = destination.strip_prefix("local/") {
                // A local alias (`cozy model download … local/name`): the model, held by name.
                let repo = tensorfs_core::repository::RepositoryName::new("local", name)
                    .map_err(|e| refused("invalid_request", e.to_string()))?;
                let current = std::fs::read(publisher.store().repository_path(&repo)).ok();
                let mutation = tensorfs_core::repository::Mutation::ReplaceLocal {
                    repo,
                    version: manifest.sha256.clone(),
                    manifest: manifest.clone(),
                };
                publisher
                    .store()
                    .apply_repository(current.as_deref(), &mutation, &Default::default())
                    .map_err(|e| refused("local_alias_refused", e.to_string()))?;
                row["published"] = json!({"destination": destination, "checkpoint": manifest.id(), "converged": false});
            } else if !destination.is_empty() {
                let publishing = hub::Catalog::for_op(
                    hub_access()?,
                    hub::Op::Publish { model: destination },
                )
                .map_err(|(code, message)| refused(code, message))?;
                let operation = format!(
                    "upload-{}",
                    &tensorfs_core::sha256::hex_digest(format!("{actor}\0{id}").as_bytes())[..40]
                );
                let stage = format!("uploading to {destination}");
                // A cancel stops the push; only finalizing, which publishes, cannot be taken back.
                let published = tensorfs_core::transport::publish(&tensorfs_core::transport::Publication {
                    store: publisher.store(),
                    hub: publishing.origin(),
                    destination,
                    manifest: &manifest,
                    operation: &operation,
                    credential: publishing.credential(),
                    policy: publishing.policy(),
                    progress: &|done, total| observe(&stage, done, total),
                    streams: 8,
                    stop,
                    commit: &|| stoppable.commit(),
                })
                .map_err(|e| match stoppable.stop.is_cancelled() {
                    true => refused(crate::published::CANCELED,
                        "the run was canceled while it uploaded: nothing was published, and the Hub keeps what landed for an upload asked again"),
                    false => {
                        let (code, message) = publishing.reason("weights_publication_failed", e);
                        refused(code, message)
                    }
                })?;
                row["published"] = json!({"destination": destination,
                    "checkpoint": published.checkpoint, "converged": published.converged});
            }
            models.push(row);
        }
        Ok(models)
    }

    /// The run's code and models made ready: Ok(None) once it is dispatchable, Ok(Some) for a
    /// warm run's result.
    fn prepare(&self, actor: &str, id: &str, spec: Spec, stoppable: &Stoppable) -> Result<Option<ResultRecord>, Refused> {
        let engine = self.service.engine.clone();
        let run = id.to_string();
        // Byte counts arrive per chunk: a new stage shows at once, bytes at most 4 times a second.
        let last = std::sync::Mutex::new((String::new(), std::time::Instant::now()));
        let observe = Arc::new(move |stage: &str, done: u64, total: u64| {
            let mut last = last.lock().unwrap();
            if last.0 == stage && last.1.elapsed() < std::time::Duration::from_millis(250) && done < total {
                return;
            }
            *last = (stage.to_string(), std::time::Instant::now());
            let detail = json!({"stage": stage, "bytes_done": done, "bytes_total": total});
            let _ = engine.observe_progress(&run, 0, detail.to_string());
        });
        observe("preparing", 0, 0);
        let warm_result = |installed: Value, models: Vec<Value>| -> Result<Option<ResultRecord>, Refused> {
            let mut value = installed;
            value["models"] = Value::Array(models);
            Ok(Some(ResultRecord {
                value,
                artifacts: vec![],
                asset_bindings: vec![],
            }))
        };
        if spec.remove {
            return self.remove(actor, &spec).map(Some);
        }
        if let Some(items) = &spec.set {
            return self.warm_set(actor, &spec, items, &observe, stoppable);
        }
        if let Source::Models = spec.source {
            let models = self.warm_models(actor, id, &spec, &*observe, stoppable)?;
            return warm_result(json!({}), models);
        }
        // A warm run naming no entrypoint installs its code, then makes its choices present.
        let install_only = spec.warm && spec.entrypoint.is_empty();
        let held = match &spec.source {
            Source::Release { .. } | Source::Models => None,
            Source::Installation(alias) => Some(
                self.service
                    .engine
                    .installation(actor, alias)?
                    .ok_or_else(|| {
                        refused(
                            "installation_absent",
                            "this machine holds no such installation for this signer",
                        )
                    })?,
            ),
            Source::Local(digest) => {
                observe("installing the local package", 0, 0);
                let local = self.local.as_ref().ok_or_else(|| {
                    refused("capability_unavailable", "this machine installs no local packages")
                })?;
                let installed = local.install(&self.service, actor, digest, Some(&stoppable.stop))?;
                self.supersede(actor, &installed)?;
                local.collect(actor, &installed.package);
                // A call's imports start as soon as its code is installed, as a release's do.
                if let (false, Some(gpu)) = (spec.warm, self.service.gpu()) {
                    gpu.prespawn(self.service.catalog.resolve(&installed.generation)?);
                }
                Some(installed)
            }
        };
        let held = match held {
            Some(installed) => Some(self.callee_view(installed, &spec.application)?),
            None => None,
        };
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let release = match &spec.source {
            Source::Release { package, release } => Some((package.clone(), release.clone())),
            _ => None,
        };
        // Its own slots' choices resolve here; the rest address its callables, whose calls
        // resolve them.
        let own: Vec<_> = match install_only {
            true => vec![],
            false => spec.models.iter().filter(|c| own_choice(c, &spec.entrypoint)).cloned().collect(),
        };
        let (installation, plan) = self.resolve(
            actor,
            held,
            release,
            hub.clone(),
            &spec,
            &spec.entrypoint,
            &own,
            Some(Box::new({
                let observe = observe.clone();
                move |stage: &str, done: u64, total: u64| observe(stage, done, total)
            })),
            Some(&stoppable.stop),
        )?;
        let interface: Value = serde_json::from_slice(&installation.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        let generation = self.service.catalog.resolve(&installation.generation)?;
        // A call's weights come from here on, in the Start pass's order, until its executor
        // asks for them itself; its imports start now unless they already did at install. A
        // warm run imports only when its level asks (`downloaded` never does).
        let mut ahead = None;
        if let (false, Some(gpu)) = (spec.warm, self.service.gpu()) {
            ahead = plan.as_ref().zip(self.publisher.as_ref()).map(|(plan, publisher)| {
                let order = self.service.engine.get(id).map_or(u64::MAX, |record| record.root_seq());
                let fetch = crate::access::Fetch::new(publisher.store().clone(), hub.clone(), plan.manifests(), order);
                if let Err(error) = fetch.ahead(&plan.upfront()) {
                    eprintln!("run {id}: its weights' early fetch: {error}");
                }
                fetch
            });
            gpu.prespawn(generation.clone());
        }
        if !generation.record.sdk_fallback.is_empty() {
            let warning = format!("{} {}: {}", installation.package, installation.release, generation.record.sdk_fallback);
            if let Err(error) = self.service.engine.append_log(id, "warning", &warning) {
                eprintln!("run {id}: {warning} ({error})");
            }
        }
        let declared = interface["entrypoints"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == spec.entrypoint.as_str()));
        if declared.is_none() && !install_only {
            return Err(refused(
                "invalid_entrypoint",
                format!("{} declares no entrypoint {:?}", installation.package, spec.entrypoint),
            ));
        }
        if spec.parent.is_empty() {
            if let Some(entry) = declared {
                public_root(entry)?;
            }
        }
        // A choice for another callable goes to the call that names it: one of this package's,
        // or a slot a callee App of its environment declares. One naming neither is a warning.
        for choice in spec.models.iter().filter(|c| !install_only && !own_choice(c, &spec.entrypoint)) {
            let declared = addressed(&choice.parameter, &installation.package, &interface).is_some()
                || generation.record.callees.iter().any(|callee| {
                    addressed(&choice.parameter, &callee.package, &callee.interface).is_some()
                });
            if !declared {
                let warning = format!(
                    "model choice {:?} names no model slot of this function or the packages it calls; it is ignored",
                    choice.parameter
                );
                if let Err(error) = self.service.engine.append_log(id, "warning", &warning) {
                    eprintln!("run {id}: {warning} ({error})");
                }
            }
        }
        if spec.warm {
            let models = match install_only {
                true => self.warm_models(actor, id, &spec, &*observe, stoppable)?,
                false => vec![],
            };
            // A function's warm run ends holding what its constructions read (`downloaded`).
            let mut moved = (0, 0);
            if let (Some(plan), Some(publisher)) = (&plan, &self.publisher) {
                let fetch = crate::access::Fetch::new(publisher.store().clone(), hub.clone(), plan.manifests(), u64::MAX);
                let report = |done, total| observe("downloading its weights", done, total);
                moved = fetch.download(&plan.upfront(), &stoppable.stop, &report).map_err(|e| match stoppable.stop.is_cancelled() {
                    true => refused(crate::published::CANCELED, "the run was canceled while it downloaded"),
                    false => {
                        let text = e.to_string();
                        refused(if text.starts_with("machine_disk_full") { "machine_disk_full" } else { "model_fetch_failed" }, text)
                    }
                })?;
            }
            // The installed release and its interface, in the warm run's result.
            let installed = json!({"package": installation.package, "release": installation.release,
                "interface": interface, "bytes": {"hub": moved.0, "volume": moved.1}});
            return warm_result(installed, models);
        }
        let entry = declared.expect("checked above");
        let degree = match (&plan, self.service.pool()) {
            (Some(plan), _) => plan.degree,
            (None, Some(pool)) => pool.degree(entry, spec.gpus),
            (None, None) => 0,
        };
        // A memoized callable's run is answered from the success this machine holds for its
        // signer under the same computation, and starts nothing. A writer (weights outputs,
        // published media) has an effect beyond its result: it always runs.
        let writer = entry["weights_outputs"].as_array().is_some_and(|rows| !rows.is_empty())
            || entry["publishes"].as_bool() == Some(true)
            || !spec.weights_destination.is_empty();
        let memo = match writer {
            false => call_memo::of(
                &generation.record,
                &spec.application,
                &entry["invocable"],
                &spec.input,
                &spec.inputs,
                &spec.attention_kernel,
                call_memo::models(plan.as_ref(), &own),
            ),
            true => Memo::Off,
        };
        let (memo, answer) = match memo {
            Memo::Key(computation) => {
                let answer = self.objects.recall(actor, &computation, id)?;
                (CallMemo { memoized: answer.is_some(), computation, ..Default::default() }, answer)
            }
            Memo::Unmet(why) => (CallMemo { unmemoized: why, ..Default::default() }, None),
            Memo::Off => (CallMemo::default(), None),
        };
        // Its children prepare with its context, and its weights travel under its access.
        if answer.is_none() {
            let context = RunContext {
                installation: installation.alias.clone(),
                hub,
                providers: spec.providers.clone(),
                owner: spec.owner.clone(),
                catalog_revision: spec.catalog_revision.clone(),
                attention_kernel: spec.attention_kernel.clone(),
                gpus: spec.gpus,
                models: spec.models.clone(),
                weights_destination: spec.weights_destination.clone(),
                ahead,
            };
            self.contexts.lock().unwrap().insert(id.into(), context);
        }
        let bound = self.service.bind_prepared(
            id,
            &installation.generation,
            Call {
                entrypoint: spec.entrypoint,
                input: spec.input,
                attention_kernel: spec.attention_kernel,
                gpus: spec.gpus,
                inputs: spec.inputs,
                degree,
                parent: spec.parent,
                application: spec.application,
            },
            plan.as_ref().map(|plan| plan.id.as_str()).unwrap_or_default(),
            memo,
            answer,
        );
        if bound.is_err() {
            self.end(id);
        }
        bound?;
        Ok(None)
    }

    /// A warm run's set: every member installed and its weights downloaded as a warm run of it
    /// alone does, the set kept in the journal, and one pass to bring the members up. The
    /// result lists each member's `level` and why it is lower than asked (`held_back`).
    fn warm_set(
        &self,
        actor: &str,
        spec: &Spec,
        items: &[SetItem],
        observe: &Arc<impl Fn(&str, u64, u64) + Send + Sync + 'static>,
        stoppable: &Stoppable,
    ) -> Result<Option<ResultRecord>, Refused> {
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let (mut members, mut kept, mut rows) = (vec![], vec![], vec![]);
        for item in items {
            let (held, release) = match &item.source {
                Source::Installation(alias) => {
                    let held = self.service.engine.installation(actor, alias)?;
                    let absent = || refused("installation_absent", "this machine holds no such installation for this signer");
                    (Some(held.ok_or_else(absent)?), None)
                }
                Source::Release { package, release } => (None, Some((package.clone(), release.clone()))),
                _ => return Err(refused("invalid_request", "a warm set member names a release or an installation")),
            };
            // `installed` stops at the code: nothing of the function is resolved or downloaded.
            let (entrypoint, choices) = match item.level {
                Level::Installed => ("", &[][..]),
                _ => (item.entrypoint.as_str(), &item.models[..]),
            };
            let report = observe.clone();
            let (installation, plan) = self.resolve(
                actor,
                held,
                release,
                hub.clone(),
                spec,
                entrypoint,
                choices,
                Some(Box::new(move |stage: &str, done: u64, total: u64| report(stage, done, total))),
                Some(&stoppable.stop),
            )?;
            if item.level > Level::Installed && !declares(&installation, &item.entrypoint)? {
                return Err(refused(
                    "invalid_entrypoint",
                    format!("{} declares no entrypoint {:?}", installation.package, item.entrypoint),
                ));
            }
            kept.push(serde_json::to_string(&KeptMember {
                installation: installation.alias.clone(),
                level: item.level.name().into(),
                preparation: plan.as_ref().map(|plan| plan.id.clone()).unwrap_or_default(),
                item: STANDARD.encode(&item.sent),
            })
            .map_err(io::Error::other)?);
            rows.push(json!({"package": installation.package, "release": installation.release,
                "entrypoint": item.entrypoint}));
            let held = self.service.catalog.resolve(&installation.generation)?;
            // A member's weights come under the warm run's access, after every run's.
            let fetch = self.publisher.as_ref().zip(plan.as_ref()).map(|(publisher, plan)| {
                crate::access::Fetch::new(publisher.store().clone(), hub.clone(), plan.manifests(), u64::MAX)
            });
            members.push(Member::new(held, plan, item.level, fetch));
        }
        self.service
            .engine
            .with_journal(|journal| journal.replace_warm_set(actor, &kept))?;
        observe("warming", 0, 0);
        let holds = match self.service.gpu() {
            Some(gpu) => {
                gpu.set_members(actor, members);
                gpu.keep(&self.service.engine, actor);
                gpu.members(actor)
            }
            None => members.iter().map(|member| without_gpus(member.level)).collect(),
        };
        for (row, (level, held_back)) in rows.iter_mut().zip(holds) {
            row["level"] = level.name().into();
            if !held_back.is_empty() {
                row["held_back"] = held_back.into();
            }
        }
        Ok(Some(ResultRecord {
            value: json!({"set": rows}),
            artifacts: vec![],
            asset_bindings: vec![],
        }))
    }

    /// A remove run: the caller's installations its source names leave the journal, their
    /// warm set members and the GPU pool, and each environment no installation or unfinished
    /// run names any more is deleted unless a process still holds it (reclaim deletes it once
    /// that ends). A release names its package at the run's Hub (any Hub for local/ code, or
    /// for an installation recorded before installations named theirs).
    fn remove(&self, actor: &str, spec: &Spec) -> Result<ResultRecord, Refused> {
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let origin = hub.and_then(|hub| hub::origin_key(&hub.origin)).unwrap_or_default();
        let named = |installed: &Installation| match &spec.source {
            Source::Installation(alias) => &installed.alias == alias,
            Source::Release { package, release } => {
                &installed.package == package
                    && (release.is_empty() || &installed.release == release)
                    && (package.starts_with("local/") || installed.hub.is_empty() || installed.hub == origin)
            }
            Source::Local(_) | Source::Models => false,
        };
        let (removed, reclaimed, retained) = self.uninstall(actor, named)?;
        if let Some(local) = &self.local {
            for gone in removed.iter().filter(|i| i.package.starts_with("local/")) {
                local.collect(actor, &gone.package);
            }
        }
        let rows: Vec<Value> = removed
            .iter()
            .map(|i| json!({"package": i.package, "release": i.release, "installation": i.alias, "hub": i.hub}))
            .collect();
        Ok(ResultRecord {
            value: json!({"removed": rows, "reclaimed_bytes": reclaimed, "retained": retained}),
            artifacts: vec![],
            asset_bindings: vec![],
        })
    }

    /// `actor`'s installations `named` leave the journal, their warm set members and the GPU
    /// pool. Each environment no installation or unfinished run names any more is deleted,
    /// its idle executors ended first; one a process still holds is `retained` for reclaim.
    fn uninstall(
        &self,
        actor: &str,
        named: impl Fn(&Installation) -> bool,
    ) -> Result<(Vec<Installation>, u64, Vec<String>), Refused> {
        let engine = &self.service.engine;
        let mut removed = vec![];
        for installed in engine.installations(actor)?.into_iter().filter(|i| named(i)) {
            if let Some(gone) = engine.with_journal(|j| j.remove_installation(actor, &installed.alias))? {
                removed.push(gone);
            }
        }
        let generations: std::collections::BTreeSet<String> = removed.iter().map(|i| i.generation.clone()).collect();
        if let Some(gpu) = self.service.gpu() {
            gpu.drop_members(actor, &generations);
        }
        let bound = engine.with_journal(|j| j.bound_generations())?;
        let (mut reclaimed, mut retained) = (0u64, vec![]);
        for generation in generations.iter().filter(|g| !bound.contains(*g)) {
            if let Some(gpu) = self.service.gpu() {
                gpu.end_generation(generation);
            }
            match crate::reclaim::remove_generation(self.service.catalog.root(), generation)? {
                Some(bytes) => reclaimed += bytes,
                None => retained.push(generation.clone()),
            }
        }
        Ok((removed, reclaimed, retained))
    }

    /// A local package is one installation: the state just installed replaces its earlier
    /// ones, whose executors end and environments' records leave. One an unfinished run names
    /// stays until a later install finds it idle.
    fn supersede(&self, actor: &str, current: &Installation) -> Result<(), Refused> {
        let running = self.service.engine.with_journal(|j| j.running_generations())?;
        self.uninstall(actor, |prior| {
            prior.package == current.package
                && prior.alias != current.alias
                && prior.alias.starts_with("local-")
                && !running.contains(&prior.generation)
        })?;
        Ok(())
    }

    /// The run's installation and model plan: through the Hub when it needs one (a release,
    /// or held code declaring models), else held code with the operator's configured grants.
    /// Unobserved, it is a hint's: its downloads go behind the ones calls wait on.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        actor: &str,
        held: Option<Installation>,
        release: Option<Published>,
        hub: Option<hub::Source>,
        spec: &Spec,
        entrypoint: &str,
        choices: &[domain::ModelChoice],
        observe: Option<crate::published::Observer>,
        stop: Option<&PullCancellation>,
    ) -> Result<(Installation, Option<crate::gpu_service::Plan>), Refused> {
        let needs_hub = held
            .as_ref()
            .is_none_or(|installed| declares_models(installed, entrypoint));
        match (hub, &self.publisher) {
            (Some(source), Some(publisher)) if needs_hub => {
                let (package, release) = release.unwrap_or_default();
                let request = Request {
                    source,
                    package,
                    release,
                    installed: held,
                    owner: spec.owner.clone(),
                    catalog_revision: spec.catalog_revision.clone(),
                    providers: spec.providers.clone(),
                    entrypoint: entrypoint.into(),
                    choices: choices.to_vec(),
                    gpus: spec.gpus,
                    imports: !spec.warm,
                };
                let prepared = publisher
                    .prepare_now(&self.service, actor, &request, observe, stop)
                    .map_err(|(code, message)| refused(code, message))?;
                Ok((prepared.installation.clone(), prepared.plan.clone()))
            }
            _ => {
                let installed = held.ok_or_else(|| {
                    refused(
                        "hub_access_absent",
                        "a release runs with the run's Hub access, and this run carries none",
                    )
                })?;
                let plan = self.configured(actor, &installed, entrypoint, choices, spec.gpus)?;
                Ok((installed, plan))
            }
        }
    }

    /// A child call of a running attempt: a run `<request>` under its parent's signer,
    /// idempotent on `intent`, preparing inside itself with its parent's context and the
    /// choices addressed to its callable.
    #[allow(clippy::too_many_arguments)]
    pub fn child(
        self: &Arc<Self>,
        parent: &Execution,
        request: &str,
        intent: &str,
        application: &str,
        entrypoint: &str,
        input: Value,
        inputs: Vec<InputFile>,
        passed: Vec<domain::ModelChoice>,
    ) -> Result<Execution, Refused> {
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        // A call the same attempt made already is its run: nothing prepares.
        if let Some(existing) = self.existing(&actor, request, intent)? {
            return Ok(existing);
        }
        let spec =
            self.child_spec(&actor, &parent.id, application, entrypoint, input, inputs, intent, &passed)?;
        self.submit(&actor, request, spec)
    }

    /// The installation as a callee's child runs see it: the callee's package, release and
    /// interface (its weights and bindings are its own), in the caller's environment.
    fn callee_view(&self, installed: Installation, application: &str) -> Result<Installation, Refused> {
        if application.is_empty() {
            return Ok(installed);
        }
        let held = self.service.catalog.resolve(&installed.generation)?;
        if application == held.record.application {
            return Ok(installed);
        }
        let (package, release, interface) = held.record.app(application).ok_or_else(|| {
            refused("child_undeclared", format!("this environment holds no App {application}"))
        })?;
        Ok(Installation {
            package: package.into(),
            release: release.into(),
            interface: serde_json::to_vec(interface).map_err(io::Error::other)?,
            ..installed
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn child_spec(
        &self,
        actor: &str,
        parent: &str,
        application: &str,
        entrypoint: &str,
        input: Value,
        inputs: Vec<InputFile>,
        digest: &str,
        passed: &[domain::ModelChoice],
    ) -> Result<Spec, Refused> {
        let context = self.contexts.lock().unwrap().get(parent).cloned().ok_or_else(|| {
            refused("child_call_refused", "the caller's preparation context is gone")
        })?;
        let installed = self.service.engine.installation(actor, &context.installation)?
            .ok_or_else(|| refused("child_call_refused", "the caller's installation is gone"))?;
        let installed = self.callee_view(installed, application)?;
        let callee_interface: Value = serde_json::from_slice(&installed.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        let prefix = format!("{entrypoint}.");
        let mut spec = Spec {
            remove: false,
            warm: false,
            set: None,
            parent: parent.into(),
            source: Source::Installation(context.installation),
            entrypoint: entrypoint.into(),
            input,
            inputs,
            // A child takes the caller's choices addressed to a slot its own interface declares
            // (a callee's included), named by that slot's path; any other was a warning at the
            // caller's preparation.
            models: context
                .models
                .into_iter()
                .filter_map(|choice| {
                    let path = addressed(&choice.parameter, &installed.package, &callee_interface)
                        .filter(|path| path.starts_with(&prefix))?;
                    Some(domain::ModelChoice { parameter: path, ..choice })
                })
                .collect(),
            attention_kernel: context.attention_kernel,
            // The caller's count caps a child's group; the child's own declaration picks within it.
            gpus: context.gpus,
            hub: context.hub,
            providers: context.providers,
            weights_destination: String::new(),
            owner: context.owner,
            catalog_revision: context.catalog_revision,
            application: application.into(),
            digest: digest.into(),
        };
        // Both demand and prefetch overlay explicit artifacts on the same inherited
        // choices. Declaration and data-access validation stay in the existing resolver.
        for choice in passed {
            let path = addressed(&choice.parameter, &installed.package, &callee_interface)
                .filter(|path| path.starts_with(&prefix))
                .ok_or_else(|| refused("invalid_request", "model choice does not name a declared callee slot"))?;
            let mut choice = choice.clone();
            choice.parameter = path;
            spec.models.retain(|prior| prior.parameter != choice.parameter);
            spec.models.push(choice);
        }
        Ok(spec)
    }

    /// `model_prefetch`: an acknowledged hint about a future child call. TensorD resolves
    /// its models, downloading behind every download a call waits on, and asks the GPU pool
    /// to prewarm a Runtime construction when admitted.
    pub fn prefetch(
        self: &Arc<Self>, parent: &Execution, application: &str, entrypoint: &str,
        passed: Vec<domain::ModelChoice>,
    ) -> Result<(), Refused> {
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        let spec = self.child_spec(&actor, &parent.id, application, entrypoint,
            json!({}), vec![], "", &passed)?;
        let Some(gpu) = self.service.gpu() else {
            return Ok(());
        };
        // Downloads may outlast the hint's GPU phase; the GPU load waits for its root's lane.
        let (root, family) = (parent.root_seq(), parent.family().map(String::from));
        let owner = parent.id.clone();
        let (runs, entrypoint) = (self.clone(), entrypoint.to_string());
        let application = application.to_string();
        let started = std::thread::Builder::new().name("child-prefetch".into()).spawn(move || {
            let prepared = (|| {
                    let Source::Installation(alias) = &spec.source else { unreachable!() };
                    let held = match runs.service.engine.installation(&actor, alias)? {
                        Some(installed) => Some(runs.callee_view(installed, &spec.application)?),
                        None => None,
                    };
                    let (hub, models) = (spec.hub.clone(), spec.models.clone());
                    runs.resolve(&actor, held, None, hub, &spec, &spec.entrypoint, &models, None, None)
                })();
            match prepared {
                Ok((installation, Some(plan))) if plan.degree > 0 && !plan.slots.is_empty() => {
                    match runs.service.catalog.resolve(&installation.generation)
                        .and_then(|held| held.application(&application)) {
                        Ok(held) => {
                            // Its weights come after its root's own, under the child's access.
                            let hub = spec.hub.clone().or_else(|| runs.own_hub.clone());
                            let fetch = crate::access::Fetch::new(gpu.store().clone(), hub, plan.manifests(), root);
                            gpu.prefetch(&runs.service.engine, held, plan, (root, family.clone()), owner.clone(), fetch)
                        }
                        Err(error) => eprintln!("prefetch of {entrypoint}: {error}"),
                    }
                }
                Ok(_) => (),
                Err(refusal) => {
                    eprintln!("prefetch of {entrypoint}: {}: {}", refusal.code, refusal.message)
                }
            }
        });
        if let Err(error) = started {
            eprintln!("prefetch of a child: {error}");
        }
        Ok(())
    }

    /// Where a run's weights outputs go: the requester's repository, under its capability.
    pub fn destination(&self, id: &str) -> io::Result<Option<crate::weights::Destination>> {
        let repository = self.contexts.lock().unwrap().get(id).map(|c| c.weights_destination.clone()).unwrap_or_default();
        match (repository.is_empty(), self.hub(id)) {
            (true, _) => Ok(None),
            (false, Some(hub)) => Ok(Some(crate::weights::Destination { repository, hub })),
            (false, None) => Err(io::Error::other("a weights destination needs the run's Hub access, and the run has none")),
        }
    }

    /// The fetches a run's preparation started, for its call to end once its executor asks.
    pub fn ahead(&self, id: &str) -> Option<Arc<crate::access::Fetch>> {
        self.contexts.lock().unwrap().get_mut(id).and_then(|context| context.ahead.take())
    }

    /// The Hub a run reads and writes: its own access, else the machine's.
    pub fn hub(&self, id: &str) -> Option<hub::Source> {
        let context = self.contexts.lock().unwrap().get(id).and_then(|c| c.hub.clone());
        context.or_else(|| self.own_hub.clone())
    }

    /// A run ended: its children prepare no more, and its access is forgotten.
    pub fn end(&self, id: &str) {
        self.contexts.lock().unwrap().remove(id);
    }

    /// Without a Hub, held code's models come from the operator's configured grants.
    fn configured(
        &self,
        actor: &str,
        installed: &Installation,
        entrypoint: &str,
        choices: &[domain::ModelChoice],
        gpus: u32,
    ) -> Result<Option<crate::gpu_service::Plan>, Refused> {
        if !declares_models(installed, entrypoint) {
            if !choices.is_empty() {
                return Err(refused(
                    "invalid_request",
                    "model choices name no declared model slot",
                ));
            }
            return Ok(None);
        }
        let pool = self.service.pool().ok_or_else(|| {
            refused("capability_unavailable", "this machine runs no functions")
        })?;
        let plan = pool
            .prepare_root(installed, entrypoint, choices, &[], gpus)
            .map_err(|e| refused("model_preparation_failed", e.to_string()))?;
        self.service.engine.bind_preparation(Preparation {
            actor: actor.into(),
            id: plan.id.clone(),
            installation: installed.alias.clone(),
            document: serde_json::to_vec(&plan).map_err(io::Error::other)?,
        })?;
        Ok(Some(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{api::install::InstallerConfig, execution::Engine, journal::State};
    use std::{fs, path::Path, process::Command, time::Duration};
    use tensorfs_core::{sha256, store::Store};

    /// The store keeps every checkpoint an unfinished run's plan reads (its raw sources too);
    /// a plan that cannot be read stops store GC for that pass.
    #[test]
    fn unfinished_runs_keep_their_checkpoints_and_an_unreadable_plan_stops_store_gc_for_a_pass() {
        let root = std::env::temp_dir().join(format!("cm-keep-list-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let publisher = Publisher::new(&root.join("published"), Default::default(), store.clone()).unwrap();
        service.configure_publisher(publisher.clone());
        drop(store);
        service.engine.bind_installation(Installation {
            actor: "alice".into(), alias: "installed".into(), generation: "g".into(),
            package: "audit/derive".into(), release: "1".into(), interface: b"{}".to_vec(), hub: String::new(),
        }).unwrap();
        let kept = |model: &str| format!("sha256:{}", model.repeat(64));
        let run = |name: &str, model: &str| {
            let plan = crate::gpu_service::Plan {
                id: format!("plan-{model}"), construction: "c".into(), installation: "installed".into(),
                generation: "g".into(), entrypoint: "derive".into(), slots: vec![], degree: 0,
                sources: vec![crate::gpu_service::PlanSource { parameter: "source".into(), manifest: kept(model),
                    length: 1, repository: String::new(), release: String::new(), lane: String::new() }],
            };
            service.engine.bind_preparation(Preparation { actor: "alice".into(), id: plan.id.clone(),
                installation: "installed".into(), document: serde_json::to_vec(&plan).unwrap() }).unwrap();
            let invocation = crate::journal::Invocation { package: "audit/derive".into(), input: json!({}), ..Default::default() };
            let (run, _) = service.engine.accept_run("alice", name, name, invocation.clone()).unwrap();
            service.engine.bind_prepared(&run.id, invocation, &plan.id, Default::default(), None).unwrap();
            run.id
        };
        let (queued, unknown, ended) = (run("queued", "a"), run("unknown", "b"), run("ended", "c"));
        // A state a newer machine wrote is unfinished to this one.
        let journal = rusqlite::Connection::open(root.join("state/execution/executions.sqlite3")).unwrap();
        journal.execute("UPDATE executions SET state='future-state' WHERE id=?1", [&unknown]).unwrap();
        service.engine.cancel(&ended, "alice").unwrap();
        assert_eq!(publisher.caches(&service).unwrap().keep, [kept("a"), kept("b")]);
        // An unreadable plan is no list at all, for this pass only.
        let plan = journal.query_row("SELECT record FROM preparations WHERE id='plan-a'", [], |row| row.get::<_, String>(0)).unwrap();
        journal.execute("UPDATE preparations SET record='not a plan' WHERE id='plan-a'", []).unwrap();
        assert!(publisher.caches(&service).is_err());
        assert_eq!(service.reclaim(), crate::reclaim::Swept::default());
        journal.execute("UPDATE preparations SET record=?1 WHERE id='plan-a'", [plan]).unwrap();
        assert_eq!(publisher.caches(&service).unwrap().keep, [kept("a"), kept("b")]);
        drop(queued);
        std::fs::remove_dir_all(root).unwrap();
    }

    fn write(objects: &Objects, actor: &str, bytes: &[u8]) -> (String, u64) {
        let digest = format!("sha256:{}", sha256::hex_digest(bytes));
        let mut writer = objects.begin(actor, &digest, bytes.len() as u64, 0).unwrap();
        writer.append(bytes).unwrap();
        writer.finish().unwrap();
        (digest, bytes.len() as u64)
    }

    fn settled(engine: &Engine, id: &str) -> Execution {
        let mut seen = engine.activity_epoch();
        loop {
            let record = engine.get(id).unwrap();
            if record.state.terminal() {
                return record;
            }
            seen = engine.wait_activity(seen, Some(Duration::from_secs(5)));
        }
    }

    fn spec(source: Source, warm: bool, digest: &str) -> Spec {
        Spec {
            remove: false,
            warm,
            set: None,
            parent: String::new(),
            source,
            entrypoint: "steps".into(),
            input: json!({"steps": 2, "seconds": 0.01}),
            inputs: vec![],
            models: vec![],
            attention_kernel: String::new(),
            gpus: 0,
            hub: None,
            providers: Default::default(),
            weights_destination: String::new(),
            owner: "alice".into(),
            catalog_revision: String::new(),
            application: String::new(),
            digest: digest.into(),
        }
    }

    #[test]
    fn resolved_internal_callables_refuse_roots_but_allow_managed_children() {
        let root = std::env::temp_dir().join(format!("cm-private-call-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let generation = "a".repeat(32);
        let directory = service.catalog.root().join(&generation);
        fs::create_dir_all(directory.join("env/bin")).unwrap();
        std::os::unix::fs::symlink("/usr/bin/python3", directory.join("env/bin/python")).unwrap();
        fs::write(directory.join(".hold"), "").unwrap();
        let interface = json!({
            "application":"fixture:app",
            "entrypoints":[{"name":"steps","internal":true}]
        });
        fs::write(directory.join("generation.json"), json!({
            "identity":generation,"package":"fixture","version":"2.0.0",
            "application":"fixture:app","python":directory.join("env/bin/python"),
            "interface":interface
        }).to_string()).unwrap();
        service.engine.bind_installation(Installation {
            actor:"alice".into(),alias:"installed".into(),generation,
            package:"alice/fixture".into(),release:"2.0.0".into(),
            interface:serde_json::to_vec(&interface).unwrap(),
            hub: String::new(),
        }).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap());
        let runs = Runs { service: service.clone(), objects, publisher: None, local: None,
            own_hub: None, leaf: None, contexts: Default::default() };
        let root_call = spec(Source::Installation("installed".into()), true, "root");
        let refused = runs.prepare("alice", "root", root_call, &runs.service.engine.stoppable("root")).err().unwrap();
        assert_eq!(refused.code, "callable_internal");
        let mut child = spec(Source::Installation("installed".into()), true, "child");
        child.parent = "managed-parent".into();
        let prepared = runs.prepare("alice", "child", child, &runs.service.engine.stoppable("child")).unwrap().unwrap();
        assert_eq!(prepared.value["release"], "2.0.0");
        let mut warm = spec(Source::Models, true, "set");
        warm.set = Some(vec![SetItem {
            source: Source::Installation("installed".into()), entrypoint: "steps".into(),
            models: vec![], level: Level::Host, sent: vec![],
        }]);
        assert_eq!(runs.prepare("alice", "set", warm, &runs.service.engine.stoppable("set")).err().unwrap().code, "callable_internal");
        let _ = fs::remove_dir_all(root);
    }

    /// Unpublished code written with Write runs through a run that prepares inside itself,
    /// on the real installer and a deviceless executor; a warm run only prepares.
    #[test]
    fn a_local_source_prepares_inside_its_run_and_runs() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = std::env::temp_dir().join(format!("cm-runs-{}", uuid::Uuid::new_v4()));
        let (runs, objects) = local_runs(&root);
        let service = runs.service.clone();
        let mut archive = tar::Builder::new(Vec::new());
        let fixture = repo.join("tests/fixtures/cpu_lifecycle");
        for name in ["pyproject.toml", "package.toml", "cpu_lifecycle/__init__.py"] {
            archive.append_path_with_name(fixture.join(name), name).unwrap();
        }
        let (source, length) = write(&objects, "alice", &archive.into_inner().unwrap());
        let manifest = json!({"package": "local/tensord-cpu-lifecycle", "release": "0.1.0",
            "python_version": "3.12", "source": {"digest": source, "length": length}});
        let (manifest, _) = write(&objects, "alice", manifest.to_string().as_bytes());

        let accepted = runs
            .submit("alice", "run-1", spec(Source::Local(manifest.clone()), false, "d1"))
            .unwrap();
        assert_eq!(accepted.state, State::Queued);
        assert_eq!(accepted.waiting_reason.as_deref(), Some(crate::journal::PREPARING));
        let done = settled(&service.engine, &accepted.id);
        assert_eq!(done.state, State::Completed, "{:?}", done.failure);
        assert_eq!(done.result.unwrap().value["steps"], 2);
        assert_eq!(done.invocation.entrypoint, "steps");

        // The id is idempotent on its spec; another spec under it is refused.
        let again = runs
            .submit("alice", "run-1", spec(Source::Local(manifest.clone()), false, "d1"))
            .unwrap();
        assert_eq!(again.id, accepted.id);
        let conflict = runs
            .submit("alice", "run-1", spec(Source::Local(manifest.clone()), false, "d2"))
            .err()
            .unwrap();
        assert_eq!(conflict.code, "run_id_conflict");

        // A warm run prepares (the held installation) and succeeds without running.
        let warm = runs
            .submit("alice", "warm-1", spec(Source::Local(manifest.clone()), true, "w1"))
            .unwrap();
        let warm = settled(&service.engine, &warm.id);
        assert_eq!(warm.state, State::Completed, "{:?}", warm.failure);
        assert_eq!(warm.result.unwrap().value["package"], "local/tensord-cpu-lifecycle");

        // A warm run's set is kept in the journal as sent, replacing the caller's last one.
        // This function binds no model, so its member holds its code and says why no more.
        let alias = service.engine.installations("alice").unwrap()[0].alias.clone();
        let set = |digest: &str, items: Vec<SetItem>| Spec {
            remove: false,
            set: Some(items),
            ..spec(Source::Models, true, digest)
        };
        let member = SetItem {
            source: Source::Installation(alias.clone()),
            entrypoint: "steps".into(),
            models: vec![],
            level: Level::Imported,
            sent: b"as sent".to_vec(),
        };
        let kept = runs.submit("alice", "set-1", set("s1", vec![member])).unwrap();
        let kept = settled(&service.engine, &kept.id);
        assert_eq!(kept.state, State::Completed, "{:?}", kept.failure);
        let row = kept.result.unwrap().value["set"][0].clone();
        assert_eq!(row["entrypoint"], "steps");
        assert_eq!(row["level"], "installed");
        assert!(row["held_back"].as_str().unwrap().contains("binds no model"), "{row}");
        let stored = || service.engine.with_journal(|journal| journal.warm_sets()).unwrap();
        let members = stored();
        let [(actor, record)] = &members[..] else {
            panic!("one member kept")
        };
        let record: KeptMember = serde_json::from_str(record).unwrap();
        assert_eq!((actor.as_str(), record.installation, record.level.as_str()), ("alice", alias.clone(), "imported"));
        assert_eq!(STANDARD.decode(record.item).unwrap(), b"as sent");
        // Its empty set clears it.
        let cleared = runs.submit("alice", "set-2", set("s2", vec![])).unwrap();
        assert_eq!(settled(&service.engine, &cleared.id).state, State::Completed);
        assert!(stored().is_empty());

        // Another signer cannot run code it did not write: refused, naming what is missing,
        // and nothing is journaled under its id.
        let theirs = runs
            .submit(
                "bob",
                "run-1",
                spec(Source::Local(manifest.clone()), false, "d1"),
            )
            .err()
            .unwrap();
        assert_eq!(theirs.code, "local_source_incomplete");
        assert!(theirs.message.contains(&manifest), "{}", theirs.message);
        assert!(service.engine.get_public("bob", "run-1").is_err());

        // A remove run takes the installation away and deletes its environment; another
        // finds nothing to remove; the written code installs again.
        let generation = service.engine.installations("alice").unwrap()[0].generation.clone();
        let package = || Source::Release { package: "local/tensord-cpu-lifecycle".into(), release: String::new() };
        let remove = |id: &str| {
            let spec = Spec { remove: true, ..spec(package(), false, id) };
            settled(&service.engine, &runs.submit("alice", id, spec).unwrap().id)
        };
        let removed = remove("remove-1");
        assert_eq!(removed.state, State::Completed, "{:?}", removed.failure);
        let value = removed.result.unwrap().value;
        assert_eq!(value["removed"][0]["installation"], alias.as_str(), "{value}");
        assert!(service.engine.installations("alice").unwrap().is_empty());
        // Deleted, unless the finished run's process had not let it go yet.
        let retained = value["retained"] == json!([generation]);
        assert_eq!(root.join("generations").join(&generation).exists(), retained, "{value}");
        assert!(retained || value["reclaimed_bytes"].as_u64().unwrap() > 0, "{value}");
        let none = remove("remove-2").result.unwrap().value;
        assert_eq!(none["removed"], json!([]), "{none}");
        let back = runs
            .submit("alice", "warm-2", spec(Source::Local(manifest.clone()), true, "w2"))
            .unwrap();
        assert_eq!(settled(&service.engine, &back.id).state, State::Completed);
        assert_eq!(service.engine.installations("alice").unwrap().len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    /// Runs that install local code with this repository's client and the real uv installer.
    fn local_runs(root: &Path) -> (Arc<Runs>, Arc<Objects>) {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        fs::create_dir_all(root).unwrap();
        assert!(Command::new("uv")
            .current_dir(repo)
            .args(["build", "--wheel", "--out-dir"])
            .arg(root.join("client"))
            .status()
            .unwrap()
            .success());
        let helper = Command::new("uv")
            .current_dir(repo)
            .args(["run", "--locked", "--extra", "test", "python", "-c", "import sys; print(sys.executable)"])
            .output()
            .unwrap();
        let client = fs::read_dir(root.join("client"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "whl"))
            .unwrap();
        let state = root.join("state");
        let service = Service::open(&state, &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&state.join("tensorfs")).unwrap());
        let objects =
            Arc::new(Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap());
        let local = LocalSources::new(
            objects.clone(),
            InstallerConfig {
                helper_python: String::from_utf8(helper.stdout).unwrap().trim().into(),
                python: "3.12".into(),
                generations: root.join("generations"),
                client_wheel: client,
                staging_root: root.join("staging"),
                sdk: vec![],
                uv: "uv".into(),
                seed_cache: None,
            },
            store.clone(),
        );
        let runs = Arc::new(Runs {
            service: service.clone(),
            objects: objects.clone(),
            publisher: None,
            local: Some(Arc::new(local)),
            own_hub: None,
            leaf: None,
            contexts: Default::default(),
        });
        let config = crate::gpu_service::GpuConfig::deviceless(&root.join("home")).unwrap();
        service.configure_pool(crate::gpu_service::Pool::new(&root.join("pool"), config, store.clone()).unwrap()).unwrap();
        crate::calls::Calls::configure(&service, store, Some(&runs)).unwrap();
        (runs, objects)
    }

    /// A local package whose build never returns (its backend sleeps, moving no byte and no
    /// CPU), written for `alice`; the backend writes its PID to `pid`.
    fn hanging_package(objects: &Objects, pid: &Path, editable: bool) -> String {
        let backend = format!(
            "import os, time\ndef hang(*args, **kwargs):\n    open({pid:?}, 'w').write(str(os.getpid()))\n    time.sleep(3600)\n\
             get_requires_for_build_wheel = prepare_metadata_for_build_wheel = build_wheel = hang\n\
             get_requires_for_build_editable = prepare_metadata_for_build_editable = build_editable = hang\n"
        );
        let pyproject = "[build-system]\nrequires = []\nbuild-backend = \"hang\"\nbackend-path = [\".\"]\n\n\
            [project]\nname = \"hanging-build\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n\
            [project.entry-points.\"cozy.application\"]\ndefault = \"hang:app\"\n";
        let mut archive = tar::Builder::new(Vec::new());
        let mut files = vec![];
        for (name, body) in [("pyproject.toml", pyproject.to_string()), ("hang.py", backend), ("package.toml", "[application]\nobject = \"hang:app\"\n".into())] {
            if editable {
                let (digest, length) = write(objects, "alice", body.as_bytes());
                files.push(json!({"name":name, "digest":digest, "length":length}));
                continue;
            }
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, body.as_bytes()).unwrap();
        }
        let mut manifest = json!({"package": "local/hanging-build", "release": "0.1.0", "python_version": "3.12"});
        if editable {
            manifest["files"] = files.into();
        } else {
            let (source, length) = write(objects, "alice", &archive.into_inner().unwrap());
            manifest["source"] = json!({"digest":source, "length":length});
        }
        write(objects, "alice", manifest.to_string().as_bytes()).0
    }

    /// The hanging backend's PID once it runs; a run that ends first is the test's failure.
    fn backend_pid(engine: &Engine, id: &str, pid: &Path) -> u32 {
        loop {
            if let Some(pid) = fs::read_to_string(pid).ok().and_then(|text| text.trim().parse().ok()) {
                return pid;
            }
            let record = engine.get(id).unwrap();
            assert!(!record.state.terminal(), "the run ended before its build: {:?}", record.failure);
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    fn gone(pid: u32) -> bool {
        // A zombie is gone too: its parent (uv) has not reaped it yet.
        fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| stat.contains(") Z "))
    }

    /// A canceled preparation stops its work: the install's process group is killed at once,
    /// not left running behind a run that says canceled.
    #[test]
    fn a_canceled_preparation_kills_its_install() {
        for editable in [false, true] {
            let root = std::env::temp_dir().join(format!("cm-cancel-{editable}-{}", uuid::Uuid::new_v4()));
            let (runs, objects) = local_runs(&root);
            let manifest = hanging_package(&objects, &root.join("backend.pid"), editable);
            let accepted = runs.submit("alice", "warm-1", spec(Source::Local(manifest), true, "w1")).unwrap();
            let backend = backend_pid(&runs.service.engine, &accepted.id, &root.join("backend.pid"));
            let canceled = runs.service.engine.cancel_run("alice", "warm-1").unwrap();
            assert_eq!(canceled.state, State::Canceled);
            while !gone(backend) {
                std::thread::sleep(Duration::from_millis(100));
            }
            assert_eq!(settled(&runs.service.engine, &accepted.id).state, State::Canceled);
            assert!(runs.service.engine.installations("alice").unwrap().is_empty(), "a canceled install binds no generation");
            runs.local.as_ref().unwrap().collect("alice", "local/hanging-build");
            assert!(!root.join("editable").join(&tensorfs_core::sha256::hex_digest(b"alice")[..16]).join("hanging-build").exists());
            let _ = fs::remove_dir_all(root);
        }
    }

    /// An install that stops moving fails on that measurement, never hanging its run.
    #[test]
    fn a_wedged_install_fails_its_run() {
        let root = std::env::temp_dir().join(format!("cm-wedge-{}", uuid::Uuid::new_v4()));
        let (runs, objects) = local_runs(&root);
        let manifest = hanging_package(&objects, &root.join("backend.pid"), false);
        let accepted = runs.submit("alice", "warm-1", spec(Source::Local(manifest), true, "w1")).unwrap();
        let backend = backend_pid(&runs.service.engine, &accepted.id, &root.join("backend.pid"));
        let failed = settled(&runs.service.engine, &accepted.id);
        assert_eq!(failed.state, State::Failed);
        let failure = failed.failure.unwrap();
        assert!(failure.contains("package_installation_stalled") && failure.contains("no measurable progress"), "{failure}");
        assert!(gone(backend));
        let _ = fs::remove_dir_all(root);
    }

    /// A callee's child runs with its own model slots in its caller's environment, and takes
    /// the choices its caller carries for them.
    #[test]
    fn a_callee_uses_its_own_model_slots_in_the_callers_environment() {
        let root = std::env::temp_dir().join(format!("cm-callee-model-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let generation = "a".repeat(32);
        let directory = service.catalog.root().join(&generation);
        fs::create_dir_all(directory.join("env/bin")).unwrap();
        std::os::unix::fs::symlink("/usr/bin/python3", directory.join("env/bin/python")).unwrap();
        fs::write(directory.join(".hold"), "").unwrap();
        let callee = json!({"application":"callee:app","entrypoints":[{
            "name":"render","models":[{"path":"render.models.network","class":"callee.Model"}]
        }]});
        let caller = json!({"application":"caller:app","entrypoints":[{
            "name":"render","models":[{"path":"render.models.network","class":"caller.Model"}]
        }]});
        fs::write(directory.join("generation.json"), json!({
            "identity":generation,"package":"caller","version":"1.0.0","application":"caller:app",
            "python":directory.join("env/bin/python"),"interface":caller,
            "callees":[{"distribution":"callee","package":"second/callee","version":"2.0.0",
                "application":"callee:app","interface":callee}]
        }).to_string()).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap());
        let runs = Runs { service: service.clone(), objects, publisher: None, local: None,
            own_hub: None, leaf: None, contexts: Default::default() };
        let installed = service.engine.bind_installation(Installation {
            actor:"alice".into(),alias:"caller".into(),generation:generation.clone(),
            package:"first/caller".into(),release:"1.0.0".into(),interface:serde_json::to_vec(&caller).unwrap(),
            hub: String::new(),
        }).unwrap();
        let selected = runs.callee_view(installed, "callee:app").unwrap();
        assert_eq!((&selected.alias, &selected.generation), (&"caller".to_string(), &generation));
        assert_eq!((&selected.package, &selected.release), (&"second/callee".to_string(), &"2.0.0".to_string()));
        let interface: Value = serde_json::from_slice(&selected.interface).unwrap();
        let slot = &interface["entrypoints"][0]["models"][0];
        assert_eq!(slot["class"], "callee.Model");
        let projected = service.catalog.resolve(&generation).unwrap().application("callee:app").unwrap();
        assert_eq!(projected.record.application, "callee:app");
        assert_eq!(projected.record.interface, callee);
        assert_eq!(service.catalog.resolve(&generation).unwrap().record.interface, caller);
        let legacy_id = "b".repeat(32);
        let legacy = service.catalog.root().join(&legacy_id);
        fs::create_dir_all(legacy.join("env/bin")).unwrap();
        std::os::unix::fs::symlink("/usr/bin/python3",legacy.join("env/bin/python")).unwrap();
        fs::write(legacy.join(".hold"),"").unwrap();
        fs::write(legacy.join("generation.json"),json!({
            "identity":legacy_id,"package":"caller","version":"1.0.0","application":"caller:app",
            "python":legacy.join("env/bin/python"),"interface":caller,"callees":[{
                "distribution":"CALLEE__Lib.Name","version":"2.0.0","application":"callee:app","interface":callee
            }]
        }).to_string()).unwrap();
        let legacy = service.catalog.resolve(&legacy_id).unwrap();
        assert_eq!(legacy.record.app("caller:app").unwrap().0,"caller");
        assert_eq!(legacy.record.app("callee:app").unwrap().0,"local/callee-lib-name");
        assert_eq!(legacy.application("callee:app").unwrap().record.package,"local/callee-lib-name");
        assert!(legacy.record.owns("callee:app","CALLEE__Lib.Name"));
        assert!(!legacy.record.owns("callee:app","unrelated"));
        assert!(!legacy.record.owns("callee:app","other/callee-lib-name"));

        let source = hub::Source::new("https://hub.example", None, vec![], None).unwrap();
        runs.contexts.lock().unwrap().insert("7".into(), RunContext {
            installation:"caller".into(),hub:Some(source),owner:"alice".into(),gpus:2,
            models:vec![
                domain::ModelChoice { parameter:"second/callee/render.models.network".into(),repository:"second/chosen".into(),..Default::default() },
                domain::ModelChoice { parameter:"render.models.absent".into(),repository:"second/nowhere".into(),..Default::default() },
            ],
            ..Default::default()
        });
        let child = runs.child_spec("alice","7","callee:app","render",json!({}),vec![],"child",&[]).unwrap();
        assert_eq!(child.gpus, 2);
        let chosen: Vec<_> = child.models.iter().map(|c| (c.parameter.as_str(), c.repository.as_str())).collect();
        assert_eq!(chosen, [("render.models.network", "second/chosen")],
            "a choice addressed to the callee's package reaches its declared slot, by that slot's path");
        // Prefetch and demand call this same selector: an explicit B replaces inherited A,
        // without mutating the parent's choices. Omitted choices leave A and its defaults.
        let chosen = domain::ModelChoice {
            parameter: "render.models.network".into(),
            manifest: Some(domain::Ref { digest: vec![0xbb; 32], length: 321 }),
            ..Default::default()
        };
        let hinted = runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", std::slice::from_ref(&chosen)).unwrap();
        let demanded = runs.child_spec("alice", "7", "callee:app", "render", json!({"prompt":"later"}), vec![], "call", std::slice::from_ref(&chosen)).unwrap();
        assert_eq!((hinted.gpus, demanded.gpus), (2, 2));
        assert_eq!(hinted.models, vec![chosen]);
        assert_eq!(hinted.models, demanded.models);
        assert!(hinted.models[0].repository.is_empty(), "B must not remain constrained to A's repository");
        assert_eq!(runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", &[]).unwrap().models[0].repository, "second/chosen");
        let bad = domain::ModelChoice { parameter: "render.models.absent".into(), ..Default::default() };
        assert_eq!(runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", &[bad]).err().unwrap().code, "invalid_request");
        runs.contexts.lock().unwrap().get_mut("7").unwrap().models.clear();
        assert!(runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", &[]).unwrap().models.is_empty(),
            "without inherited/explicit choices the normal resolver owns captured defaults");
        fs::remove_dir_all(root).unwrap();
    }

    /// A preparing run's stages are observed like a running attempt's.
    #[test]
    fn a_preparing_run_shows_its_progress() {
        let root = std::env::temp_dir().join(format!("cm-observed-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let draft = crate::journal::Invocation {
            package: "org/pkg".into(),
            generation: String::new(),
            module: String::new(),
            entrypoint: "run".into(),
            input: json!({}),
            attention_kernel: String::new(),
            inputs: vec![],
            ..Default::default()
        };
        let accepted = engine.accept_run("alice", "run-1", "d", draft).unwrap().0;
        engine
            .observe_progress(&accepted.id, 0, r#"{"stage":"downloading"}"#.into())
            .unwrap();
        let seen = engine.get(&accepted.id).unwrap();
        assert!(seen.revision > accepted.revision);
        assert_eq!(seen.progress.as_deref(), Some(r#"{"stage":"downloading"}"#));
        let _ = fs::remove_dir_all(root);
    }

    /// No preparation survives a restart: the token is gone, so the run ends FAILED.
    #[test]
    fn a_restart_ends_a_preparing_run_failed() {
        let root = std::env::temp_dir().join(format!("cm-restart-{}", uuid::Uuid::new_v4()));
        let draft = crate::journal::Invocation {
            package: "org/pkg".into(),
            generation: String::new(),
            module: String::new(),
            entrypoint: "run".into(),
            input: json!({}),
            attention_kernel: String::new(),
            inputs: vec![],
            ..Default::default()
        };
        let id = {
            let engine = Engine::open(&root).unwrap();
            engine.accept_run("alice", "run-1", "d", draft).unwrap().0.id
        };
        let record = Engine::open(&root).unwrap().get(&id).unwrap();
        assert_eq!(record.state, State::Failed);
        assert!(record.failure.unwrap().contains("preparation_interrupted"));
        let _ = fs::remove_dir_all(root);
    }
}
