//! Run sources and preparation inside a run (`cozy.machine.v1` Run). A run is accepted at
//! once and prepares inside itself: its code installs, its models resolve at the run's Hub and
//! download, and each stage is the run's progress. The Hub token lives only in memory, for that
//! preparation (and a job's life, which its children prepare with); a restart before it
//! completes ends the run FAILED (`journal::PREPARING`).
use crate::{
    api::domain,
    gpu_service::{without_gpus, KeptMember, Level, Member},
    hub,
    journal::{Execution, Failure, InputFile, Installation, Outcome, Preparation, ResultRecord},
    local_source::LocalSources,
    objects::{Objects, Refused},
    published::{declares_models, Providers, Publisher, Request},
    service::{Call, Service},
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use tensorfs_core::ids::ObjectRef;
use std::{
    collections::{BTreeMap, HashMap},
    io,
    sync::{Arc, Mutex},
};

pub enum Source {
    Release { package: String, release: String },
    /// An installation this machine already holds for the signer.
    Installation(String),
    /// Unpublished code written with Write: its manifest's digest.
    Local(String),
    /// No code: a warm run that only makes its model choices (`cozy model upload`).
    Models,
}

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
    /// A warm run's whole warm set for its caller, replacing the previous one.
    pub set: Option<Vec<SetItem>>,
    /// `kind: job`: `entrypoint` names an `@app.job`; its installed declaration selects its device.
    pub job: bool,
    /// A child run's parent execution (a job's call through its seam).
    pub parent: String,
    pub source: Source,
    pub entrypoint: String,
    pub input: Value,
    pub inputs: Vec<InputFile>,
    pub models: Vec<domain::ModelChoice>,
    pub binding_revision: String,
    pub attention_kernel: String,
    /// The run's Hub access; its grant is held in memory only.
    pub hub: Option<hub::Source>,
    /// Provider tokens for source models, held the same way.
    pub providers: Providers,
    /// `org/name`: a model-only warm run puts the source model it made there.
    pub weights_destination: String,
    /// The machine-publication grant the weights destination is written under.
    pub publication: Option<Arc<hub::Grant>>,
    pub owner: String,
    /// A child's memoized result this machine holds: its answer.
    pub held: Option<ResultRecord>,
    /// A child run of a callee: the other package's App its parent's environment holds.
    pub application: String,
    /// The spec's identity (its token excluded): a resubmitted id must carry the same.
    pub digest: String,
}

pub struct Runs {
    pub service: Arc<Service>,
    pub objects: Arc<Objects>,
    pub publisher: Option<Arc<Publisher>>,
    pub local: Option<Arc<LocalSources>>,
    /// On a rental: its own Hub, read with the pod's worker capability.
    pub own_hub: Option<hub::Source>,
    /// Signers' execution grants, and the key that redeems and proves them.
    pub grants: hub::Grants,
    /// Unfinished jobs' preparation: their children prepare with it. The tokens live only
    /// here; the rest is journaled too (`Durable`), for a paused job resumed after a restart.
    pub jobs: Mutex<HashMap<String, JobContext>>,
}

/// What a job's children prepare with: its installation, Hub access, owner, binding revision,
/// attention pin and the model choices addressed to its callables (`<entrypoint>.models.<p>`).
#[derive(Clone)]
pub struct JobContext {
    installation: String,
    application: String,
    hub: Option<hub::Source>,
    providers: Providers,
    owner: String,
    binding_revision: String,
    attention_kernel: String,
    models: Vec<domain::ModelChoice>,
    /// The job's own model inputs, made present: parameter -> (class, manifest).
    inputs: BTreeMap<String, (String, ObjectRef)>,
    weights_destination: String,
    publication: Option<Arc<hub::Grant>>,
}

/// What a job's weights grant reads of its context (`jobs.rs`).
pub struct JobWeights {
    pub inputs: BTreeMap<String, (String, ObjectRef)>,
    pub destination: Option<crate::weights::Destination>,
}

/// A job context's journaled part: everything but its tokens (choices prost-encoded).
#[derive(serde::Serialize, serde::Deserialize)]
struct Durable {
    installation: String,
    #[serde(default)]
    application: String,
    owner: String,
    binding_revision: String,
    attention_kernel: String,
    models: Vec<Vec<u8>>,
    #[serde(default)]
    inputs: BTreeMap<String, (String, String, u64)>,
    #[serde(default)]
    weights_destination: String,
}
impl Durable {
    fn of(context: &JobContext) -> Self {
        Self {
            installation: context.installation.clone(),
            application: context.application.clone(),
            owner: context.owner.clone(),
            binding_revision: context.binding_revision.clone(),
            attention_kernel: context.attention_kernel.clone(),
            models: context.models.iter().map(crate::archive::encode_model_choice).collect(),
            inputs: context
                .inputs
                .iter()
                .map(|(p, (class, m))| (p.clone(), (class.clone(), m.sha256.clone(), m.length)))
                .collect(),
            weights_destination: context.weights_destination.clone(),
        }
    }
    /// Without the run's tokens: its children prepare with the machine's own Hub, if any.
    fn context(self) -> io::Result<JobContext> {
        Ok(JobContext {
            installation: self.installation,
            application: self.application,
            hub: None,
            providers: Providers::default(),
            owner: self.owner,
            binding_revision: self.binding_revision,
            attention_kernel: self.attention_kernel,
            models: self
                .models
                .iter()
                .map(|m| crate::archive::decode_model_choice(m.as_slice()))
                .collect::<Result<_, _>>()
                .map_err(io::Error::other)?,
            inputs: self
                .inputs
                .into_iter()
                .map(|(p, (class, sha256, length))| (p, (class, ObjectRef { sha256, length })))
                .collect(),
            weights_destination: self.weights_destination,
            publication: None,
        })
    }
}

/// The manifests of a journaled job context's model inputs (`sha256:<hex>`): the store keeps
/// them while the job is unfinished.
pub(crate) fn job_models(journaled: &[u8]) -> io::Result<Vec<String>> {
    let context: Durable = serde_json::from_slice(journaled).map_err(io::Error::other)?;
    let manifests = context.inputs.into_values();
    Ok(manifests.map(|(_, sha256, _)| format!("sha256:{sha256}")).collect())
}

/// A job's own model input: a bare parameter, or one under the job's own name.
fn own_input(choice: &domain::ModelChoice, job: &str) -> bool {
    match choice.parameter.split_once(".models.") {
        None => true,
        Some((callable, _)) => callable == job,
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
    ["jobs", "entrypoints"]
        .iter()
        .any(|section| {
            interface[section].as_array().into_iter().flatten().any(|entry| {
                entry["models"].as_array().into_iter().flatten().any(|slot| slot["path"] == path)
            })
        })
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

/// A child's kind is whatever its held package declares it as.
fn captured_child_job(interface: &[u8], name: &str) -> Result<bool, Refused> {
    let interface: Value = serde_json::from_slice(interface)
        .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
    let declares = |rows: &str| {
        interface[rows].as_array().into_iter().flatten().any(|row| row["name"] == name)
    };
    match (declares("jobs"), declares("entrypoints")) {
        (true, false) => Ok(true),
        (false, true) => Ok(false),
        _ => Err(refused("invalid_entrypoint", format!("the package declares {name:?} as neither one job nor one entrypoint"))),
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
            references.extend(crate::local_source::written(&self.objects, actor, digest)?);
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
            inputs: spec.inputs.clone(),
            job: spec.job,
            accelerator: false, // known only after the actual installation is prepared
            parent: spec.parent.clone(),
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
                    let outcome = this.prepare(&actor, &run, spec);
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
                                message: format!("{}: {}", refusal.code, refusal.message),
                            };
                            this.service
                                .engine
                                .end_preparation(&run, Outcome::Failed(failure.encode()))
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
    /// Hub repository under the run's machine-publication grant.
    fn warm_models(
        &self,
        actor: &str,
        id: &str,
        spec: &Spec,
        observe: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<Vec<Value>, Refused> {
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
                let made = publisher.make_source(&choice.source, &choice.profiles, &spec.providers, observe)?;
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
            } else {
                let digest = choice.manifest.as_ref().ok_or_else(|| {
                    refused(
                        "invalid_request",
                        format!("model choice {:?} names no exact manifest or provider source", choice.parameter),
                    )
                })?;
                let sha256 = tensorfs_core::sha256::hex(&digest.digest);
                let manifest = format!("sha256:{sha256}");
                if !choice.repository.is_empty() {
                    let stage = format!("downloading {}", choice.repository);
                    publisher.download(&self.service, hub_access()?, &choice.repository, &manifest, &|done, total| {
                        observe(&stage, done, total)
                    })?;
                }
                // No repository: a checkpoint this machine already holds (a run's output).
                let length = std::fs::metadata(publisher.store().manifest_path(&sha256)).map_err(|_| {
                    refused("checkpoint_absent", format!("this machine holds no checkpoint {manifest}"))
                })?;
                let row = json!({"parameter": choice.parameter,
                    "repository": choice.repository, "manifest": manifest});
                (ObjectRef { sha256, length: length.len() }, row)
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
                let Some(publication) = &spec.publication else {
                    return Err(refused(
                        "publication_unauthorized",
                        "a weights destination needs the run's publication grant",
                    ));
                };
                let publishing = hub::Publishing::new(hub_access()?, publication, self.own_hub.as_ref())
                    .map_err(|e| refused("publication_unauthorized", e.0))?;
                let operation = format!(
                    "upload-{}",
                    &tensorfs_core::sha256::hex_digest(format!("{actor}\0{id}").as_bytes())[..40]
                );
                let stage = format!("uploading to {destination}");
                let published = tensorfs_core::transport::publish(&tensorfs_core::transport::Publication {
                    store: publisher.store(),
                    hub: publishing.origin(),
                    destination,
                    manifest: &manifest,
                    operation: &operation,
                    credential: &publishing,
                    policy: publishing.policy(),
                    progress: &|done, total| observe(&stage, done, total),
                    streams: 8,
                })
                .map_err(|e| refused("weights_publication_failed", e.to_string()))?;
                row["published"] = json!({"destination": destination,
                    "checkpoint": published.checkpoint, "converged": published.converged});
            }
            models.push(row);
        }
        Ok(models)
    }

    /// The run's code and models made ready: Ok(None) once it is dispatchable, Ok(Some) for a
    /// warm run's result.
    fn prepare(&self, actor: &str, id: &str, spec: Spec) -> Result<Option<ResultRecord>, Refused> {
        // A memoized call answered from a result this machine holds runs nothing.
        if let (false, Some(held)) = (spec.parent.is_empty(), &spec.held) {
            return Ok(Some(held.clone()));
        }
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
        if let Some(items) = &spec.set {
            return self.warm_set(actor, &spec, items, &observe);
        }
        if let Source::Models = spec.source {
            let models = self.warm_models(actor, id, &spec, &*observe)?;
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
                Some(local.install(&self.service, actor, digest)?)
            }
        };
        let held = match held {
            Some(installed) => Some(self.callee_view(installed, &spec.application)?),
            None => None,
        };
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let release = match &spec.source {
            // Naming no release asks for the newest at this machine's own Hub (describe/1).
            Source::Release { package, release } if release.is_empty() => {
                let publisher = self.publisher.as_ref().ok_or_else(|| {
                    refused("capability_unavailable", "this machine prepares no releases")
                })?;
                let source = hub.as_ref().ok_or_else(|| {
                    refused("hub_access_absent", "a release runs with the run's Hub access, and this run carries none")
                })?;
                let newest = publisher.newest_release(source, package).map_err(|(code, message)| refused(code, message))?;
                Some((package.clone(), newest))
            }
            Source::Release { package, release } => Some((package.clone(), release.clone())),
            _ => None,
        };
        // A job's choices address its callables; its children resolve them.
        let choices = if spec.job || install_only { &[][..] } else { &spec.models[..] };
        let (installation, plan) = self.resolve(
            actor,
            held,
            release,
            hub.clone(),
            &spec,
            &spec.entrypoint,
            choices,
            Box::new({
                let observe = observe.clone();
                move |stage: &str, done: u64, total: u64| observe(stage, done, total)
            }),
        )?;
        let interface: Value = serde_json::from_slice(&installation.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        if let Ok(held) = self.service.catalog.resolve(&installation.generation) {
            if !held.record.sdk_fallback.is_empty() {
                let warning = format!("{} {}: {}", installation.package, installation.release, held.record.sdk_fallback);
                if let Err(error) = self.service.engine.append_log(id, "warning", &warning) {
                    eprintln!("run {id}: {warning} ({error})");
                }
            }
        }
        let (rows, noun) = match spec.job {
            true => ("jobs", "job"),
            false => ("entrypoints", "entrypoint"),
        };
        let declared = interface[rows]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == spec.entrypoint.as_str()));
        if declared.is_none() && !install_only {
            return Err(refused(
                "invalid_entrypoint",
                format!("{} declares no {noun} {:?}", installation.package, spec.entrypoint),
            ));
        }
        if spec.parent.is_empty() {
            if let Some(entry) = declared {
                public_root(entry)?;
            }
        }
        if spec.job {
            // A bare parameter, or `<job>.models.<parameter>`, is the job's own model input.
            // Another goes to the child it names: one of this package's callables, or a slot a
            // callee App of its environment declares. One naming neither is only a warning.
            let callees = self.service.catalog.resolve(&installation.generation).map(|held| held.record.callees)?;
            for choice in spec.models.iter().filter(|c| !own_input(c, &spec.entrypoint)) {
                let declared = addressed(&choice.parameter, &installation.package, &interface).is_some()
                    || callees.iter().any(|callee| {
                        addressed(&choice.parameter, &callee.package, &callee.interface).is_some()
                    });
                if !declared {
                    let warning = format!(
                        "model choice {:?} names no model slot of this job or the packages it calls; it is ignored",
                        choice.parameter
                    );
                    if let Err(error) = self.service.engine.append_log(id, "warning", &warning) {
                        eprintln!("run {id}: {warning} ({error})");
                    }
                }
            }
        }
        if spec.warm {
            let models = match install_only {
                true => self.warm_models(actor, id, &spec, &*observe)?,
                false => vec![],
            };
            // The installed release and its interface: a client that never read the Hub
            // types its calls with it (describe/1).
            let installed = json!({"package": installation.package, "release": installation.release,
                "interface": interface});
            return warm_result(installed, models);
        }
        if spec.job {
            let inputs = self.job_inputs(actor, &spec, &installation, &interface, hub.as_ref(), &*observe)?;
            let context = JobContext {
                installation: installation.alias.clone(),
                application: spec.application.clone(),
                hub,
                providers: spec.providers.clone(),
                owner: spec.owner.clone(),
                binding_revision: spec.binding_revision.clone(),
                attention_kernel: spec.attention_kernel.clone(),
                models: spec.models.clone(),
                inputs,
                weights_destination: spec.weights_destination.clone(),
                publication: spec.publication.clone(),
            };
            let durable = serde_json::to_vec(&Durable::of(&context)).map_err(io::Error::other)?;
            self.service
                .engine
                .with_journal(|journal| journal.bind_job_context(id, &durable))?;
            self.jobs.lock().unwrap().insert(id.into(), context);
        }
        self.service.bind_prepared(
            id,
            &installation.generation,
            Call {
                entrypoint: spec.entrypoint,
                input: spec.input,
                attention_kernel: spec.attention_kernel,
                inputs: spec.inputs,
                job: spec.job,
                parent: spec.parent,
                application: spec.application,
            },
            plan.as_ref().map(|plan| plan.id.as_str()).unwrap_or_default(),
        )?;
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
                Box::new(move |stage: &str, done: u64, total: u64| report(stage, done, total)),
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
            members.push(Member::new(held, plan, item.level));
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

    /// The run's installation and model plan: through the Hub when it needs one (a release,
    /// or held code declaring models), else held code with the operator's configured grants.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        actor: &str,
        held: Option<Installation>,
        release: Option<(String, String)>,
        hub: Option<hub::Source>,
        spec: &Spec,
        entrypoint: &str,
        choices: &[domain::ModelChoice],
        observe: crate::published::Observer,
    ) -> Result<(Installation, Option<crate::gpu_service::GpuPlan>), Refused> {
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
                    binding_revision: spec.binding_revision.clone(),
                    providers: spec.providers.clone(),
                    entrypoint: entrypoint.into(),
                    choices: choices.to_vec(),
                };
                let prepared = publisher
                    .prepare_now(&self.service, actor, &request, observe)
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
                let plan = self.configured(actor, &installed, entrypoint, choices)?;
                Ok((installed, plan))
            }
        }
    }

    /// A child call of a running job: a run `<request>` under the job's signer, idempotent on
    /// `intent`, preparing inside itself with the job's context and the choices addressed to
    /// its callable. `held`, a memoized call's result this machine holds, completes it without
    /// running.
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
        held: Option<ResultRecord>,
        passed: Vec<domain::ModelChoice>,
    ) -> Result<Execution, Refused> {
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        // An earlier attempt's call is already its run (a resumed job): nothing prepares.
        if let Some(existing) = self.existing(&actor, request, intent)? {
            return Ok(existing);
        }
        let mut spec =
            self.child_spec(&actor, &parent.id, application, entrypoint, input, inputs, intent, &passed)?;
        spec.held = held;
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
        let held = self.jobs.lock().unwrap().get(parent).cloned();
        let context = match held {
            Some(context) => context,
            None => {
                let journaled = self
                    .service
                    .engine
                    .with_journal(|journal| journal.job_context(parent))?
                    .ok_or_else(|| {
                        refused("child_call_refused", "the job's preparation context is gone")
                    })?;
                serde_json::from_slice::<Durable>(&journaled)
                    .map_err(io::Error::other)?
                    .context()?
            }
        };
        let installed = self.service.engine.installation(actor, &context.installation)?
            .ok_or_else(|| refused("child_call_refused", "the job's installation is gone"))?;
        let installed = self.callee_view(installed, application)?;
        let job = captured_child_job(&installed.interface, entrypoint)?;
        let callee_interface: Value = serde_json::from_slice(&installed.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        let prefix = format!("{entrypoint}.");
        let mut spec = Spec {
            warm: false,
            set: None,
            job,
            parent: parent.into(),
            source: Source::Installation(context.installation),
            entrypoint: entrypoint.into(),
            input,
            inputs,
            // A child takes the caller's choices addressed to a slot its own interface declares
            // (a callee's included), named by that slot's path; any other was a warning at the
            // job's preparation.
            models: context
                .models
                .into_iter()
                .filter_map(|choice| {
                    let path = addressed(&choice.parameter, &installed.package, &callee_interface)
                        .filter(|path| path.starts_with(&prefix))?;
                    Some(domain::ModelChoice { parameter: path, ..choice })
                })
                .collect(),
            // The owner's binding revision covers every package: a callee's rebinding is seen too.
            binding_revision: context.binding_revision,
            attention_kernel: context.attention_kernel,
            hub: context.hub,
            providers: context.providers,
            weights_destination: String::new(),
            publication: None,
            owner: context.owner,
            held: None,
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
    /// its models and asks the GPU pool to prewarm a Runtime construction when admitted.
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
        let family = crate::gpu_reservation::family(parent, |id| self.service.engine.get(id))?;
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
                    runs.resolve(&actor, held, None, hub, &spec, &spec.entrypoint, &models, Box::new(|_, _, _| ()))
                })();
            match prepared {
                Ok((installation, Some(plan))) => {
                    match runs.service.catalog.resolve(&installation.generation)
                        .and_then(|held| held.application(&application)) {
                        Ok(held) => gpu.prefetch(&runs.service.engine, held, plan, family.clone()),
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

    /// A job ended: its children prepare no more.
    /// A job's model inputs and where its weights outputs go, for its weights grant.
    pub fn job_weights(&self, id: &str) -> io::Result<Option<JobWeights>> {
        let context = match self.jobs.lock().unwrap().get(id).cloned() {
            Some(context) => context,
            None => match self.service.engine.with_journal(|journal| journal.job_context(id))? {
                Some(journaled) => serde_json::from_slice::<Durable>(&journaled)
                    .map_err(io::Error::other)?
                    .context()?,
                None => return Ok(None),
            },
        };
        let hub = context.hub.clone().or_else(|| self.own_hub.clone());
        let destination = match (context.weights_destination.is_empty(), hub) {
            (true, _) => None,
            (false, Some(hub)) => Some(crate::weights::Destination {
                repository: context.weights_destination.clone(),
                hub,
                publication: context.publication.clone(),
                rental: self.own_hub.clone(),
            }),
            (false, None) => {
                return Err(io::Error::other(
                    "a weights destination needs the run's Hub access, and the job has none",
                ))
            }
        };
        Ok(Some(JobWeights {
            inputs: context.inputs,
            destination,
        }))
    }

    /// The job's own model inputs (`<job>.models.<parameter>`), each made present here: a
    /// Hub checkpoint downloaded by its exact manifest, a provider source made.
    fn job_inputs(
        &self,
        actor: &str,
        spec: &Spec,
        installation: &Installation,
        interface: &Value,
        hub: Option<&hub::Source>,
        observe: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<BTreeMap<String, (String, ObjectRef)>, Refused> {
        let declared = interface["jobs"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == spec.entrypoint.as_str()))
            .and_then(|row| row["models"].as_array().cloned())
            .unwrap_or_default();
        let mut inputs = BTreeMap::new();
        for row in declared {
            let path = row["path"].as_str().unwrap_or_default();
            let parameter = path.rsplit_once(".models.").map_or(path, |(_, p)| p).to_string();
            let class = row["class"].as_str().unwrap_or_default().to_string();
            let choice = spec
                .models
                .iter()
                .find(|c| c.parameter == parameter || c.parameter == path);
            let publisher = self.publisher.as_ref().ok_or_else(|| {
                refused("capability_unavailable", "this machine prepares no models")
            })?;
            // A model passed by value (a `ModelArtifact` another run here produced) is the call's
            // own argument: its exact checkpoint, as this machine holds it.
            let passed = spec.input.get(&parameter).and_then(|value| {
                let digest = value["manifest"]["digest"].as_str()?.strip_prefix("sha256:")?;
                Some((digest.to_string(), value["manifest"]["length"].as_u64()?))
            });
            let manifest = if let Some((sha256, length)) = passed {
                let held = std::fs::metadata(publisher.store().manifest_path(&sha256)).map(|m| m.len());
                if held.ok() != Some(length) {
                    return Err(refused(
                        "checkpoint_absent",
                        format!("the model passed to {parameter:?} (sha256:{sha256}) is not held on this machine"),
                    ));
                }
                ObjectRef { sha256, length }
            } else if let Some(choice) = choice.filter(|c| !c.source.is_empty()) {
                publisher
                    .make_source(&choice.source, &choice.profiles, &spec.providers, observe)?
                    .manifest
            } else {
                // An exact checkpoint, a repository selector or nothing chosen: resolved at the
                // run's Hub the way a direct run's slot is.
                let hub = hub.ok_or_else(|| {
                    refused("hub_access_absent", "a Hub model downloads with the run's Hub access, and this run carries none")
                })?;
                // Resolved once per installation, slot, choice, owner, binding revision and GPU,
                // then kept: a warm run asks the Hub nothing; `cozy package bind` moves the
                // revision, and so the key.
                let (gpu, width) = crate::published::machine_gpu(&self.service);
                let key = serde_json::to_string(&json!(["job-input/1", installation.alias, path, choice.map(|c| json!([
                    c.repository, c.release, c.lane, c.manifest.as_ref().map(|m| (tensorfs_core::sha256::hex(&m.digest), m.length))
                ])), spec.owner, spec.binding_revision, gpu, width]))
                .map_err(io::Error::other)?;
                let kept = self.service.engine.with_journal(|j| j.job_input(actor, &key))?;
                let (repository, manifest) = match kept {
                    Some(kept) => kept,
                    None => {
                        let catalog = hub::Catalog::new(hub).map_err(|e| refused("catalog_read_failed", e.0))?;
                        observe(&format!("resolving the model for {path}"), 0, 0);
                        let resolved = crate::published::job_input(
                            &catalog,
                            installation,
                            &row,
                            choice,
                            &spec.owner,
                            (&gpu, width),
                        )
                        .map_err(|(code, message)| refused(code, message))?;
                        self.service
                            .engine
                            .with_journal(|j| j.bind_job_input(actor, &key, &resolved.0, &resolved.1))?;
                        resolved
                    }
                };
                let stage = format!("downloading {repository}");
                publisher.download(&self.service, hub, &repository, &manifest, &|done, total| {
                    observe(&stage, done, total)
                })?;
                let sha256 = manifest.trim_start_matches("sha256:").to_string();
                let length = std::fs::metadata(publisher.store().manifest_path(&sha256))?.len();
                ObjectRef { sha256, length }
            };
            inputs.insert(parameter, (class, manifest));
        }
        Ok(inputs)
    }

    pub fn end_job(&self, id: &str) {
        self.jobs.lock().unwrap().remove(id);
        if let Err(error) = self
            .service
            .engine
            .with_journal(|journal| journal.forget_job_context(id))
        {
            eprintln!("job {id}: its journaled context remains: {error}");
        }
    }

    /// Without a Hub, held code's models come from the operator's configured grants.
    fn configured(
        &self,
        actor: &str,
        installed: &Installation,
        entrypoint: &str,
        choices: &[domain::ModelChoice],
    ) -> Result<Option<crate::gpu_service::GpuPlan>, Refused> {
        if !declares_models(installed, entrypoint) {
            if !choices.is_empty() {
                return Err(refused(
                    "invalid_request",
                    "model choices name no declared model slot",
                ));
            }
            return Ok(None);
        }
        let gpu = self.service.gpu().ok_or_else(|| {
            refused(
                "capability_unavailable",
                "this callable needs a GPU and this machine has none configured",
            )
        })?;
        let plan = gpu
            .prepare_root(installed, entrypoint, choices, &[], 0)
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

    #[test]
    fn unfinished_jobs_keep_their_model_inputs_and_an_unreadable_one_stops_store_gc_for_a_pass() {
        let root = std::env::temp_dir().join(format!("cm-keep-list-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let publisher = Publisher::new(&root.join("published"), Default::default(), store).unwrap();
        service.configure_publisher(publisher.clone());
        let context = |model: &str| Durable {
            installation: String::new(),
            application: String::new(),
            owner: "alice".into(),
            binding_revision: String::new(),
            attention_kernel: String::new(),
            models: vec![],
            inputs: BTreeMap::from([("model".into(), ("Model".into(), model.repeat(64), 1))]),
            weights_destination: String::new(),
        };
        let bind = |id: &str, context: &[u8]| {
            let bound = service.engine.with_journal(|journal| journal.bind_job_context(id, context));
            bound.unwrap()
        };
        let job = |name: &str, model: &str| {
            let invocation = crate::journal::Invocation {
                package: "audit/job".into(),
                input: json!({}),
                job: true,
                ..Default::default()
            };
            let (run, _) = service.engine.accept_run("alice", name, name, invocation).unwrap();
            bind(&run.id, &serde_json::to_vec(&context(model)).unwrap());
            run.id
        };
        let (paused, unknown, ended) = (job("paused", "a"), job("unknown", "b"), job("ended", "c"));
        service.engine.pause(&paused, "alice", true).unwrap();
        // A state a newer machine wrote is unfinished to this one.
        let journal = rusqlite::Connection::open(root.join("state/execution/executions.sqlite3")).unwrap();
        let future = "UPDATE executions SET state='future-state' WHERE id=?1";
        journal.execute(future, [&unknown]).unwrap();
        service.engine.end_preparation(&ended, Outcome::Failed("ended".into())).unwrap();
        let kept = |model: &str| format!("sha256:{}", model.repeat(64));
        assert_eq!(publisher.caches(&service).unwrap().keep, [kept("a"), kept("b")]);
        // An unreadable context is no list at all, for this pass only.
        bind(&paused, b"not a context");
        assert!(publisher.caches(&service).is_err());
        assert_eq!(service.reclaim(), crate::reclaim::Swept::default());
        bind(&paused, &serde_json::to_vec(&context("a")).unwrap());
        assert_eq!(publisher.caches(&service).unwrap().keep, [kept("a"), kept("b")]);
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
            warm,
            set: None,
            job: false,
            parent: String::new(),
            source,
            entrypoint: "steps".into(),
            input: json!({"steps": 2, "seconds": 0.01}),
            inputs: vec![],
            models: vec![],
            binding_revision: String::new(),
            attention_kernel: String::new(),
            hub: None,
            providers: Default::default(),
            weights_destination: String::new(),
            publication: None,
            owner: "alice".into(),
            held: None,
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
            "entrypoints":[{"name":"steps","internal":true}],
            "jobs":[{"name":"steps","internal":true}]
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
        }).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap());
        let runs = Runs { service: service.clone(), objects, publisher: None, local: None,
            own_hub: None, grants: Default::default(), jobs: Default::default() };
        for job in [false, true] {
            let mut root_call = spec(Source::Installation("installed".into()), true, "root");
            root_call.job = job;
            let refused = runs.prepare("alice", "root", root_call).err().unwrap();
            assert_eq!(refused.code, "callable_internal");
            let mut child = spec(Source::Installation("installed".into()), true, "child");
            child.job = job;
            child.parent = "managed-parent".into();
            let prepared = runs.prepare("alice", "child", child).unwrap().unwrap();
            assert_eq!(prepared.value["release"], "2.0.0");
        }
        let mut warm = spec(Source::Models, true, "set");
        warm.set = Some(vec![SetItem {
            source: Source::Installation("installed".into()), entrypoint: "steps".into(),
            models: vec![], level: Level::Host, sent: vec![],
        }]);
        assert_eq!(runs.prepare("alice", "set", warm).err().unwrap().code, "callable_internal");
        let _ = fs::remove_dir_all(root);
    }

    /// Unpublished code written with Write runs through a run that prepares inside itself,
    /// on the real installer and CPU runner; a warm run only prepares.
    #[test]
    fn a_local_source_prepares_inside_its_run_and_runs() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = std::env::temp_dir().join(format!("cm-runs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
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
            },
            store.clone(),
        );
        let runs = Arc::new(Runs {
            service: service.clone(),
            objects: objects.clone(),
            publisher: None,
            local: Some(Arc::new(local)),
            own_hub: None,
            grants: Default::default(),
            jobs: Default::default(),
        });
        crate::jobs::Jobs::configure(&service, store.clone(), Some(&runs)).unwrap();
        let mut archive = tar::Builder::new(Vec::new());
        let fixture = repo.join("tests/fixtures/cpu_lifecycle");
        for name in ["pyproject.toml", "package.toml", "cpu_lifecycle/__init__.py"] {
            archive.append_path_with_name(fixture.join(name), name).unwrap();
        }
        let (source, length) = write(&objects, "alice", &archive.into_inner().unwrap());
        let manifest = json!({"package": "local/cozy-machine-cpu-lifecycle", "release": "0.1.0",
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
        assert_eq!(warm.result.unwrap().value["package"], "local/cozy-machine-cpu-lifecycle");

        // A warm run's set is kept in the journal as sent, replacing the caller's last one.
        // This function binds no model, so its member holds its code and says why no more.
        let alias = service.engine.installations("alice").unwrap()[0].alias.clone();
        let set = |digest: &str, items: Vec<SetItem>| Spec {
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
        assert_eq!((actor.as_str(), record.installation, record.level.as_str()), ("alice", alias, "imported"));
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
        let _ = fs::remove_dir_all(root);
    }

    /// A model passed by value to a job's model slot (a `ModelArtifact` a sibling produced
    /// here) is that exact checkpoint; one this machine does not hold is refused by name.
    #[test]
    fn a_model_passed_by_value_is_its_exact_checkpoint() {
        let root = std::env::temp_dir().join(format!("cm-by-value-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let publisher = Publisher::new(&root.join("published"), Default::default(), store.clone()).unwrap();
        let objects = Arc::new(Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap());
        let runs = Runs { service: service.clone(), objects, publisher: Some(publisher), local: None,
            own_hub: None, grants: Default::default(), jobs: Default::default() };
        let held = "ab".repeat(32);
        let path = store.manifest_path(&held);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"manifest").unwrap();
        let interface = json!({"jobs":[{"name":"compute","models":[{"path":"compute.models.source","class":"Source"}]}]});
        let artifact = |hex: &str| json!({"producer_request_id":"job-1/0","output_slot":"weights",
            "manifest":{"digest":format!("sha256:{hex}"),"length":8},"tensorfs_receipt_digest":format!("sha256:{hex}")});
        let mut spec = spec(Source::Models, false, "d");
        (spec.job, spec.entrypoint) = (true, "compute".into());
        spec.input = json!({"source": artifact(&held), "factor": 2});
        let installation = Installation { actor: "alice".into(), alias: "pkg".into(), generation: String::new(),
            package: "local/pkg".into(), release: "0.1.0".into(), interface: vec![] };
        let inputs = runs.job_inputs("owner", &spec, &installation, &interface, None, &|_, _, _| ()).unwrap();
        assert_eq!(inputs["source"], ("Source".to_string(), ObjectRef { sha256: held, length: 8 }));
        spec.input = json!({"source": artifact(&"cd".repeat(32)), "factor": 2});
        let absent = runs.job_inputs("owner", &spec, &installation, &interface, None, &|_, _, _| ()).unwrap_err();
        assert_eq!(absent.code, "checkpoint_absent", "{}", absent.message);
        fs::remove_dir_all(root).unwrap();
    }

    /// A job's child that is itself a job runs as one, as its held package declares it.
    #[test]
    fn a_callee_uses_its_own_model_slots_and_hub_bindings_in_the_callers_environment() {
        use std::io::{Read, Write};
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
            own_hub: None, grants: Default::default(), jobs: Default::default() };
        let installed = service.engine.bind_installation(Installation {
            actor:"alice".into(),alias:"caller".into(),generation:generation.clone(),
            package:"first/caller".into(),release:"1.0.0".into(),interface:serde_json::to_vec(&caller).unwrap(),
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

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let source = hub::Source::pod(&format!("http://{}", listener.local_addr().unwrap()), "wrk", "callee-test", None, vec![]);
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut bytes = Vec::new();
            while !bytes.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                assert_eq!(stream.read(&mut byte).unwrap(), 1);
                bytes.push(byte[0]);
            }
            let request = String::from_utf8(bytes).unwrap();
            assert!(request.starts_with("GET /v1/packages/second/callee/bindings "), "{request}");
            assert!(request.to_ascii_lowercase().contains("x-cozy-worker-token: callee-test"));
            let body = json!({"bindings":[{"slot":"render.models.network","model":"second/weights",
                "release":"3.0.0","ladder":[{"gpu":"*","lane":"bf16","gpus":1}]}]}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        let catalog = hub::Catalog::new(&source).unwrap();
        let mut cached = None;
        let bound = crate::published::model_binding(&catalog, &selected, slot, &mut cached).unwrap();
        assert_eq!(bound["model"], "second/weights");
        assert_eq!(crate::published::model_binding(&catalog, &selected, slot, &mut cached).unwrap(), bound);
        server.join().unwrap();
        runs.jobs.lock().unwrap().insert("7".into(), JobContext {
            installation:"caller".into(),application:String::new(),hub:Some(source),providers:Default::default(),
            owner:"alice".into(),binding_revision:"caller-binding".into(),attention_kernel:String::new(),
            models:vec![
                domain::ModelChoice { parameter:"second/callee/render.models.network".into(),repository:"second/chosen".into(),..Default::default() },
                domain::ModelChoice { parameter:"render.models.absent".into(),repository:"second/nowhere".into(),..Default::default() },
            ],
            inputs:Default::default(),weights_destination:String::new(),publication:None,
        });
        let child = runs.child_spec("alice","7","callee:app","render",json!({}),vec![],"child",&[]).unwrap();
        let chosen: Vec<_> = child.models.iter().map(|c| (c.parameter.as_str(), c.repository.as_str())).collect();
        assert_eq!(chosen, [("render.models.network", "second/chosen")],
            "a choice addressed to the callee's package reaches its declared slot, by that slot's path");
        assert_eq!(child.binding_revision, "caller-binding", "a callee sees the owner's rebindings");
        // Prefetch and demand call this same selector: an explicit B replaces inherited A,
        // without mutating the parent's choices. Omitted choices leave A and its defaults.
        let chosen = domain::ModelChoice {
            parameter: "render.models.network".into(),
            manifest: Some(domain::Ref { digest: vec![0xbb; 32], length: 321 }),
            ..Default::default()
        };
        let hinted = runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", std::slice::from_ref(&chosen)).unwrap();
        let demanded = runs.child_spec("alice", "7", "callee:app", "render", json!({"prompt":"later"}), vec![], "call", std::slice::from_ref(&chosen)).unwrap();
        assert_eq!(hinted.models, vec![chosen]);
        assert_eq!(hinted.models, demanded.models);
        assert!(hinted.models[0].repository.is_empty(), "B must not remain constrained to A's repository");
        assert_eq!(runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", &[]).unwrap().models[0].repository, "second/chosen");
        let bad = domain::ModelChoice { parameter: "render.models.absent".into(), ..Default::default() };
        assert_eq!(runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", &[bad]).err().unwrap().code, "invalid_request");
        runs.jobs.lock().unwrap().get_mut("7").unwrap().models.clear();
        assert!(runs.child_spec("alice", "7", "callee:app", "render", json!({}), vec![], "", &[]).unwrap().models.is_empty(),
            "without inherited/explicit choices the normal resolver owns captured defaults");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_child_takes_its_kind_from_the_held_package() {
        let root = std::env::temp_dir().join(format!("cm-child-kind-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap());
        let runs = Runs { service: service.clone(), objects, publisher: None, local: None, own_hub: None, grants: Default::default(), jobs: Default::default() };
        let interface = json!({
            "jobs": [{"name": "pipeline"}, {"name": "leaf-job"}, {"name": "both"}],
            "entrypoints": [{"name": "leaf-call"}, {"name": "both"}],
        });
        service.engine.bind_installation(crate::journal::Installation {
            actor: "alice".into(), alias: "pkg".into(), generation: "g1".into(), package: "org/pkg".into(),
            release: "1.0.0".into(), interface: serde_json::to_vec(&interface).unwrap(),
        }).unwrap();
        runs.jobs.lock().unwrap().insert("7".into(), JobContext {
            installation: "pkg".into(), hub: None, providers: Default::default(), owner: "alice".into(),
            application: String::new(),
            binding_revision: String::new(), attention_kernel: String::new(), models: vec![],
            inputs: Default::default(), weights_destination: String::new(), publication: None,
        });
        let child = |name: &str| runs.child_spec("alice", "7", "", name, json!({}), vec![], "d", &[]);
        assert!(child("leaf-job").unwrap().job);
        assert!(!child("leaf-call").unwrap().job);
        assert_eq!(child("missing").err().unwrap().code, "invalid_entrypoint");
        assert_eq!(child("both").err().unwrap().code, "invalid_entrypoint");
        let _ = fs::remove_dir_all(root);
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
