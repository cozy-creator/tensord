//! Run sources and preparation inside a run (`cozy.machine.v1` Run). A run is accepted at
//! once and prepares inside itself: its code installs, its models resolve at the run's Hub and
//! download, and each stage is the run's progress. The Hub token lives only in memory, for that
//! preparation (and a job's life, which its children prepare with); a restart before it
//! completes ends the run FAILED (`journal::PREPARING`).
use crate::{
    api::pb,
    hub,
    journal::{Execution, Failure, InputFile, Installation, Outcome, Preparation, ResultRecord},
    local_source::LocalSources,
    objects::{Objects, Refused},
    published::{declares_models, Providers, Publisher, Request},
    service::{Call, Service},
};
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

pub struct Spec {
    /// Prepare only (`kind: warm`): install and download, then succeed.
    pub warm: bool,
    /// `kind: job`: `entrypoint` names an `@app.job`, run in a deviceless executor.
    pub job: bool,
    /// A child run's parent execution (a job's call through its seam).
    pub parent: String,
    pub source: Source,
    pub entrypoint: String,
    pub input: Value,
    pub inputs: Vec<InputFile>,
    pub models: Vec<pb::ModelChoice>,
    pub binding_revision: String,
    pub attention_kernel: String,
    /// The run's Hub access, held in memory for this preparation only.
    pub hub: Option<hub::Source>,
    /// Provider tokens for source models, held the same way.
    pub providers: Providers,
    /// `org/name`: a model-only warm run puts the source model it made there.
    pub weights_destination: String,
    /// The machine-publication authorization the weights destination is written under.
    pub publication: String,
    pub owner: String,
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
    /// Unfinished jobs' preparation: their children prepare with it. The tokens live only
    /// here; the rest is journaled too (`Durable`), for a paused job resumed after a restart.
    pub jobs: Mutex<HashMap<String, JobContext>>,
}

/// What a job's children prepare with: its installation, Hub access, owner, binding revision,
/// attention pin and the model choices addressed to its callables (`<entrypoint>.models.<p>`).
#[derive(Clone)]
pub struct JobContext {
    installation: String,
    hub: Option<hub::Source>,
    providers: Providers,
    owner: String,
    binding_revision: String,
    attention_kernel: String,
    models: Vec<pb::ModelChoice>,
    /// The job's own model inputs, made present: parameter -> (class, manifest).
    inputs: BTreeMap<String, (String, ObjectRef)>,
    weights_destination: String,
    publication: String,
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
    owner: String,
    binding_revision: String,
    attention_kernel: String,
    models: Vec<Vec<u8>>,
    #[serde(default)]
    inputs: BTreeMap<String, (String, String, u64)>,
    #[serde(default)]
    weights_destination: String,
    #[serde(default)]
    publication: String,
}
impl Durable {
    fn of(context: &JobContext) -> Self {
        Self {
            installation: context.installation.clone(),
            owner: context.owner.clone(),
            binding_revision: context.binding_revision.clone(),
            attention_kernel: context.attention_kernel.clone(),
            models: context.models.iter().map(prost::Message::encode_to_vec).collect(),
            inputs: context
                .inputs
                .iter()
                .map(|(p, (class, m))| (p.clone(), (class.clone(), m.sha256.clone(), m.length)))
                .collect(),
            weights_destination: context.weights_destination.clone(),
            publication: context.publication.clone(),
        }
    }
    /// Without the run's tokens: its children prepare with the machine's own Hub, if any.
    fn context(self) -> io::Result<JobContext> {
        Ok(JobContext {
            installation: self.installation,
            hub: None,
            providers: Providers::default(),
            owner: self.owner,
            binding_revision: self.binding_revision,
            attention_kernel: self.attention_kernel,
            models: self
                .models
                .iter()
                .map(|m| <pb::ModelChoice as prost::Message>::decode(m.as_slice()))
                .collect::<Result<_, _>>()
                .map_err(io::Error::other)?,
            inputs: self
                .inputs
                .into_iter()
                .map(|(p, (class, sha256, length))| (p, (class, ObjectRef { sha256, length })))
                .collect(),
            weights_destination: self.weights_destination,
            publication: self.publication,
        })
    }
}

/// A job's own model input: a bare parameter, or one under the job's own name.
fn own_input(choice: &pb::ModelChoice, job: &str) -> bool {
    match choice.parameter.split_once(".models.") {
        None => true,
        Some((callable, _)) => callable == job,
    }
}

fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

/// The held package capture owns callable kind; a child frame owns no placement authority.
fn captured_child_job(interface: &[u8], name: &str) -> Result<bool, Refused> {
    let interface: Value = serde_json::from_slice(interface)
        .map_err(|_| refused("interface_invalid", "held interface is corrupt"))?;
    let contains = |section: &str| interface[section].as_array().into_iter().flatten()
        .any(|row| row["name"].as_str() == Some(name) && row["invocable"].is_object());
    match (contains("jobs"), contains("entrypoints")) {
        (true, false) => Ok(true),
        (false, true) => Ok(false),
        _ => Err(refused("child_undeclared", "child is absent or ambiguous in the held callable capture")),
    }
}

impl Runs {
    /// Reattachment observes an accepted obligation before consulting mutable input caches.
    pub(crate) fn existing(&self, actor: &str, id: &str, digest: &str) -> Result<Option<Execution>, Refused> {
        match self.service.engine.get_public(actor,id) {
            Ok(record) if record.submission.as_ref().map(|s|s.invocation_digest.as_str())==Some(digest) => Ok(Some(record)),
            Ok(_) => Err(refused("run_id_conflict","this run id already names another run spec")),
            Err(error) if error.kind()==io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Refused::from(error)),
        }
    }

    /// Accepts the run (or answers the one this id already names) and starts its preparation.
    pub fn submit(self: &Arc<Self>, actor: &str, id: &str, spec: Spec) -> Result<Execution, Refused> {
        if let Some(record)=self.existing(actor,id,&spec.digest)? { return Ok(record); }
        if !spec.input.is_object() {
            return Err(refused(
                "invalid_request",
                "the payload is a JSON object of the function's parameters",
            ));
        }
        // Native GC exclusion covers validation -> durable roots -> journal acceptance.
        // The machine custody gate also excludes TTL release of those same roots.
        let custody = self.objects.guard();
        let writer = self.objects.writer_guard()?;
        let mut references = Vec::new();
        let unwritten = |input: &str| refused("input_unwritten", format!("input {input} was not written to this machine"));
        for input in &spec.inputs {
            let path = match self.objects.path(actor, &input.digest)? {
                Some((path, length)) if length == input.length => path,
                _ => return Err(unwritten(&input.input_id)),
            };
            references.push(ObjectRef { sha256: input.digest.trim_start_matches("sha256:").into(), length: input.length });
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
                references.push(ObjectRef { sha256: sha.into(), length });
            }
        }
        if let Source::Local(digest) = &spec.source {
            let (path, length) = self.objects.path(actor, digest)?.ok_or_else(|| unwritten("local package manifest"))?;
            if length > 1 << 20 { return Err(refused("local_source_invalid", "a local package manifest is at most 1 MiB")); }
            references.push(ObjectRef { sha256: digest.trim_start_matches("sha256:").into(), length });
            let manifest: crate::local_source::Manifest = serde_json::from_slice(&std::fs::read(path)?)
                .map_err(|e| refused("local_source_invalid", e.to_string()))?;
            for member in manifest.source.iter().chain(&manifest.wheels).chain(&manifest.requirements) {
                match self.objects.path(actor, &member.digest)? {
                    Some((_, length)) if length == member.length => references.push(ObjectRef {
                        sha256: member.digest.trim_start_matches("sha256:").into(), length,
                    }),
                    _ => return Err(unwritten(&member.name)),
                }
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
            let launched = std::thread::Builder::new()
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
                .map(drop);
            preparation_launched(&self.service.engine, &record.id, launched)?;
        }
        Ok(record)
    }

    /// A warm run's model choices made present when it names no entrypoint (`cozy package
    /// install`, `cozy model download`) or no code (`cozy model upload`, `cozy run upload`): a
    /// Hub checkpoint downloaded, a provider source made, a checkpoint this machine holds (a
    /// run's weights output) found; with a weights destination, the one choice is put there
    /// under the run's machine-publication authorization.
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
            } else {
                let stage=format!("downloading {}",choice.repository);
                let (repository,manifest)=publisher.download_choice(&self.service,hub.as_ref(),choice,
                    &|done,total|observe(&stage,done,total),id)?;
                let row=json!({"parameter":choice.parameter,"repository":repository,"manifest":manifest.id()});
                (manifest,row)
            };
            self.service.engine.retain_model(Some(id),row["repository"].as_str().unwrap_or_default(),&manifest)?;
            if !destination.is_empty() {
                if spec.publication.is_empty() {
                    return Err(refused(
                        "publication_unauthorized",
                        "a weights destination needs the run's publication authorization",
                    ));
                }
                let publishing = hub::Publishing::new(hub_access()?, &spec.publication)
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
        let release = match &spec.source {
            Source::Release { package, release } => Some((package.clone(), release.clone())),
            _ => None,
        };
        // A job's choices address its callables; its children resolve them.
        let choices = if spec.job || install_only { &[][..] } else { &spec.models[..] };
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let (installation, plan) = self.resolve(
            actor,
            id,
            held,
            release,
            hub.clone(),
            &spec,
            choices,
            Box::new({
                let observe = observe.clone();
                move |stage: &str, done: u64, total: u64| observe(stage, done, total)
            }),
        )?;
        let interface: Value = serde_json::from_slice(&installation.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        let (rows, noun) = match spec.job {
            true => ("jobs", "job"),
            false => ("entrypoints", "entrypoint"),
        };
        let declared = interface[rows]
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["name"] == spec.entrypoint.as_str()));
        if !declared && !install_only {
            return Err(refused(
                "invalid_entrypoint",
                format!("{} declares no {noun} {:?}", installation.package, spec.entrypoint),
            ));
        }
        if spec.job {
            // A bare parameter, or `<job>.models.<parameter>`, is the job's own model input.
            for choice in spec.models.iter().filter(|c| !own_input(c, &spec.entrypoint)) {
                let callable = choice.parameter.split_once(".models.").map(|(name, _)| name);
                let declared = interface["entrypoints"].as_array().is_some_and(|rows| {
                    rows.iter()
                        .any(|row| Some(row["name"].as_str().unwrap_or_default()) == callable)
                });
                if !declared {
                    return Err(refused(
                        "invalid_request",
                        format!(
                            "model choice {:?} names no callable of this job's package \
                             (<entrypoint>.models.<parameter>)",
                            choice.parameter
                        ),
                    ));
                }
            }
        }
        if spec.warm {
            let models = match install_only {
                true => self.warm_models(actor, id, &spec, &*observe)?,
                false => vec![],
            };
            let installed = json!({"package": installation.package, "release": installation.release});
            return warm_result(installed, models);
        }
        if spec.job {
            let inputs = self.job_inputs(id, &spec, &interface, hub.as_ref(), &*observe)?;
            let context = JobContext {
                installation: installation.alias.clone(),
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
            },
            plan.as_ref().map(|plan| plan.id.as_str()).unwrap_or_default(),
        )?;
        Ok(None)
    }

    /// The run's installation and model plan: through the Hub when it needs one (a release,
    /// or held code declaring models), else held code with the operator's configured grants.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        actor: &str,
        execution: &str,
        held: Option<Installation>,
        release: Option<(String, String)>,
        hub: Option<hub::Source>,
        spec: &Spec,
        choices: &[pb::ModelChoice],
        observe: crate::published::Observer,
    ) -> Result<(Installation, Option<crate::gpu_service::GpuPlan>), Refused> {
        let needs_hub = held
            .as_ref()
            .is_none_or(|installed| declares_models(installed, &spec.entrypoint));
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
                    entrypoint: spec.entrypoint.clone(),
                    choices: choices.to_vec(),
                };
                let prepared = publisher
                    .prepare_now(&self.service, actor, &request, observe, execution)
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
                let plan = self.configured(actor, execution, &installed, &spec.entrypoint, choices)?;
                Ok((installed, plan))
            }
        }
    }

    /// A child call of a running job: a run `<request>` under the job's signer, idempotent on
    /// `intent`, preparing inside itself with the job's context and the choices addressed to
    /// its callable.
    pub fn child(
        self: &Arc<Self>,
        parent: &Execution,
        request: &str,
        intent: &str,
        entrypoint: &str,
        input: Value,
        inputs: Vec<InputFile>,
    ) -> Result<Execution, Refused> {
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        // An earlier attempt's call is already its run (a resumed job): nothing prepares.
        match self.service.engine.get_public(&actor, request) {
            Ok(existing) => {
                let same = existing.submission.as_ref().map(|s| s.invocation_digest.as_str());
                if same != Some(intent) {
                    return Err(refused("run_id_conflict", "this run id already names another run spec"));
                }
                return Ok(existing);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
        let spec = self.child_spec(&parent.id, entrypoint, input, inputs, intent)?;
        self.submit(&actor, request, spec)
    }

    fn child_spec(
        &self,
        parent: &str,
        entrypoint: &str,
        input: Value,
        inputs: Vec<InputFile>,
        digest: &str,
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
        let actor = self.service.engine.get(parent)?.submission
            .ok_or_else(|| refused("child_call_refused", "parent has no submission actor"))?.actor;
        let installed = self.service.engine.installation(&actor, &context.installation)?
            .ok_or_else(|| refused("installation_absent", "parent's held installation is absent"))?;
        let job = captured_child_job(&installed.interface, entrypoint)?;
        let prefix = format!("{entrypoint}.");
        Ok(Spec {
            warm: false,
            job,
            parent: parent.into(),
            source: Source::Installation(context.installation),
            entrypoint: entrypoint.into(),
            input,
            inputs,
            models: context
                .models
                .into_iter()
                .filter(|choice| choice.parameter.starts_with(&prefix))
                .collect(),
            binding_revision: context.binding_revision,
            attention_kernel: context.attention_kernel,
            hub: context.hub,
            providers: context.providers,
            weights_destination: String::new(),
            publication: String::new(),
            owner: context.owner,
            digest: digest.into(),
        })
    }

    /// `model_prefetch`: the job will call `entrypoint` next, so its models prepare and load
    /// now, beside whatever the GPU already holds.
    pub fn prefetch(self: &Arc<Self>, parent: &Execution, entrypoint: &str) {
        let Some(gpu) = self.service.gpu() else {
            return;
        };
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        let (runs, parent, entrypoint) = (self.clone(), parent.id.clone(), entrypoint.to_string());
        let started = std::thread::Builder::new().name("child-prefetch".into()).spawn(move || {
            let prepared = runs
                .child_spec(&parent, &entrypoint, json!({}), vec![], "")
                .and_then(|spec| {
                    let Source::Installation(alias) = &spec.source else { unreachable!() };
                    let held = runs.service.engine.installation(&actor, alias)?;
                    let (hub, models) = (spec.hub.clone(), spec.models.clone());
                    runs.resolve(&actor, &parent, held, None, hub, &spec, &models, Box::new(|_, _, _| ()))
                });
            match prepared {
                Ok((installation, Some(plan))) => {
                    match runs.service.catalog.resolve(&installation.generation) {
                        Ok(held) => gpu.prefetch(&runs.service.engine, held, plan),
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
    /// Logical catalog selector resolved and downloaded, an exact checkpoint held, or a provider source made.
    fn job_inputs(
        &self,
        id: &str,
        spec: &Spec,
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
                .find(|c| c.parameter == parameter || c.parameter == path)
                .ok_or_else(|| {
                    refused("model_input_absent", format!("the job's model input {parameter:?} is not chosen"))
                })?;
            let publisher = self.publisher.as_ref().ok_or_else(|| {
                refused("capability_unavailable", "this machine prepares no models")
            })?;
            let manifest = if !choice.source.is_empty() {
                publisher
                    .make_source(&choice.source, &choice.profiles, &spec.providers, observe)?
                    .manifest
            } else {
                let stage=format!("downloading {}",choice.repository);
                let (_,manifest)=publisher.download_choice(&self.service,hub,choice,
                    &|done,total|observe(&stage,done,total),id)?;
                manifest
            };
            self.service.engine.retain_model(Some(id),&choice.repository,&manifest)?;
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
        execution: &str,
        installed: &Installation,
        entrypoint: &str,
        choices: &[pb::ModelChoice],
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
        let _reader=self.objects.writer_guard()?;
        let plan = gpu
            .prepare_root(installed, entrypoint, choices, &[], 0)
            .map_err(|e| refused("model_preparation_failed", e.to_string()))?;
        let preparation=Preparation {
            actor: actor.into(),
            id: plan.id.clone(),
            installation: installed.alias.clone(),
            document: serde_json::to_vec(&plan).map_err(io::Error::other)?,
        };
        self.service.engine.bind_preparation(preparation.clone())?;
        self.service.engine.retain_preparation(Some(execution),&preparation)?;
        Ok(Some(plan))
    }
}

/// Acceptance already committed: a failed launcher must settle that durable obligation
/// before answering a refusal. Reattachment observes the failure and never silently waits.
fn preparation_launched(engine: &crate::execution::Engine, id: &str, launched: io::Result<()>) -> Result<(), Refused> {
    if let Err(error) = launched {
        let message = format!("preparation_launch_failed: {error}");
        engine.end_preparation(id, Outcome::Failed(Failure {
            status: 3, cause: 7, origin: 3, message: message.clone(),
        }.encode()))?;
        return Err(refused("preparation_launch_failed", message));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_acceptance_is_found_before_mutable_input_validation() {
        let root=std::env::temp_dir().join(format!("cm-existing-run-{}",uuid::Uuid::new_v4()));
        let service=Service::open(&root.join("state"),&root.join("generations"),1).unwrap();
        let store=Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
        let objects=Arc::new(Objects::new(&root.join("writes"),store,service.engine.clone()).unwrap());
        let runs=Runs {service:service.clone(),objects,publisher:None,local:None,own_hub:None,jobs:Default::default()};
        let (accepted,_)=service.engine.accept_run("alice","accepted","same-intent",crate::journal::Invocation {
            package:"audit/package".into(),input:json!({}),..Default::default()
        }).unwrap();
        service.engine.end_preparation(&accepted.id,Outcome::Failed("finished without input retention".into())).unwrap();
        assert_eq!(runs.existing("alice","accepted","same-intent").unwrap().unwrap().id,accepted.id);
        assert_eq!(runs.existing("alice","accepted","other-intent").unwrap_err().code,"run_id_conflict");
        assert!(runs.existing("alice","missing","same-intent").unwrap().is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn captured_children_use_the_held_callable_kind() {
        let interface = serde_json::to_vec(&json!({
            "jobs": [{"name":"leaf-job", "invocable":{"module":"package", "export":"leaf_job"}}],
            "entrypoints": [{"name":"leaf-call", "invocable":{"module":"package", "export":"leaf_call"}}]
        })).unwrap();
        assert!(captured_child_job(&interface,"leaf-job").unwrap());
        assert!(!captured_child_job(&interface,"leaf-call").unwrap());
        assert_eq!(captured_child_job(&interface,"missing").unwrap_err().code,"child_undeclared");
        assert_eq!(captured_child_job(b"broken","leaf-job").unwrap_err().code,"interface_invalid");
        let ambiguous = serde_json::to_vec(&json!({
            "jobs":[{"name":"duplicate","invocable":{}}],
            "entrypoints":[{"name":"duplicate","invocable":{}}]
        })).unwrap();
        assert_eq!(captured_child_job(&ambiguous,"duplicate").unwrap_err().code,"child_undeclared");
    }

    #[test]
    fn failed_preparation_launcher_settles_acceptance_and_reattach_observes_failure() {
        let root = std::env::temp_dir().join(format!("cm-prepare-launch-{}", uuid::Uuid::new_v4()));
        let engine = crate::execution::Engine::open(&root).unwrap();
        let invocation = crate::journal::Invocation { package: "audit/package".into(), input: json!({}), ..Default::default() };
        let (run, new) = engine.accept_run("alice", "same-request", "same-intent", invocation.clone()).unwrap();
        assert!(new);
        let error = io::Error::new(io::ErrorKind::WouldBlock, "process resources exhausted");
        assert_eq!(preparation_launched(&engine, &run.id, Err(error)).unwrap_err().code, "preparation_launch_failed");
        let (reattached, new) = engine.accept_run("alice", "same-request", "same-intent", invocation).unwrap();
        assert!(!new);
        assert_eq!(reattached.state, crate::journal::State::Failed);
        assert!(reattached.failure.unwrap().contains("preparation_launch_failed"));
        std::fs::remove_dir_all(root).unwrap();
    }
    use crate::{api::install::InstallerConfig, execution::Engine, journal::State};
    use std::{fs, path::Path, process::Command, time::Duration};
    use tensorfs_core::{sha256, store::Store};

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
            publication: String::new(),
            owner: "alice".into(),
            digest: digest.into(),
        }
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

        // Another signer cannot run code it did not write.
        let refusal = runs
            .submit("bob", "run-1", spec(Source::Local(manifest), false, "d1"))
            .unwrap_err();
        assert_eq!(refusal.code, "input_unwritten");
        assert_eq!(service.engine.get_public("bob", "run-1").unwrap_err().kind(), io::ErrorKind::NotFound);
        let _ = fs::remove_dir_all(root);
    }

    /// A real accepted CPU job resolves a logical catalog selector at the Hub and consumes
    /// the exact native source channel. No client-side manifest substitution or GPU load.
    #[test]
    fn an_accepted_cpu_job_resolves_and_reads_its_logical_model_selector() {
        use std::{io::{Read,Write},net::TcpListener,sync::atomic::{AtomicUsize,Ordering}};
        use tensorfs_core::{dtype::Dtype,header::{Header,Part,Tensor},manifest::{Draft,Entry},store::Fault};
        let repo=Path::new(env!("CARGO_MANIFEST_DIR"));
        let root=std::env::temp_dir().join(format!("cm-logical-model-job-{}",uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let service=Service::open(&root.join("state"),&root.join("generations"),1).unwrap();
        let store=Arc::new(Store::ensure(&root.join("store")).unwrap());
        service.engine.configure_model_custody(store.clone()).unwrap();
        let payload=vec![0x31;2048];
        let data=store.put_stream(&mut payload.as_slice(),None,&Fault::default()).unwrap().obj;
        let plain=tensorfs_core::registry::seeds().into_iter().find(|seed|seed.alias=="plain/1").unwrap().spec;
        let header=Header {configs:vec![],assets:vec![],encodings:vec![plain.clone()],components:vec![("model".into(),vec![("weight".into(),Tensor {
            dtype:Dtype::U8,shape:vec![2048],encoding:plain.object_id(),parts:vec![("value".into(),Part::plan(Dtype::U8,vec![2048],&payload))],
        })])]};
        let header=store.put_stream(&mut header.canonical_bytes().unwrap().as_slice(),None,&Fault::default()).unwrap().obj;
        let manifest=store.put_manifest(&Draft {entries:vec![("model".into(),Entry::CozyTensors(header.clone()))]}.seal().unwrap()).unwrap().obj;
        let listener=TcpListener::bind("127.0.0.1:0").unwrap();
        let origin=format!("http://{}",listener.local_addr().unwrap());
        let resolved=Arc::new(AtomicUsize::new(0));
        let reads=resolved.clone(); let checkpoint=manifest.clone();
        // Read-only loopback transport fixture, real native ensure/closure/custody paths.
        let hub=std::thread::spawn(move || {
            for socket in listener.incoming() {
                let mut socket=socket.unwrap();
                let mut bytes=Vec::new(); let mut buffer=[0u8;1024];
                loop {
                    let size=socket.read(&mut buffer).unwrap(); if size==0 {break;}
                    bytes.extend_from_slice(&buffer[..size]);
                    if bytes.windows(4).any(|part|part==b"\r\n\r\n") {break;}
                }
                let text=String::from_utf8_lossy(&bytes); let path=text.split_whitespace().nth(1).unwrap_or("");
                if path=="/stop" {break;}
                assert!(text.to_ascii_lowercase().contains("authorization: bearer cpu-model-source"));
                let value=if path.starts_with("/v1/models/resolve?") {
                    assert!(path.contains("models%2Ffixture%401.0.0"),"{path}");
                    reads.fetch_add(1,Ordering::Relaxed);
                    json!({"model":"models/fixture","release":"1.0.0","lane":"bf16","manifest_id":checkpoint.id(),"components":["model"]})
                } else {
                    assert_eq!(path,"/v1/tensorfs/closure");
                    json!({"complete":true,"scope":"runtime","model":"models/fixture","release":"1.0.0","lane":"bf16",
                        "manifest":{"sha256":checkpoint.sha256,"length":checkpoint.length},
                        "objects":[{"sha256":header.sha256,"length":header.length},{"sha256":data.sha256,"length":data.length}],"presign_max_digests":1024})
                };
                let body=serde_json::to_vec(&value).unwrap();
                write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
        });
        assert!(Command::new("uv").current_dir(repo).args(["build","--wheel","--out-dir"]).arg(root.join("client")).status().unwrap().success());
        let helper_python=match std::env::var_os("COZY_TEST_SDK_PYTHON") {
            Some(path)=>std::path::PathBuf::from(path),
            None=> {
                let helper=Command::new("uv").current_dir(repo).args(["run","--locked","--extra","test","python","-c","import sys; print(sys.executable)"]).output().unwrap();
                assert!(helper.status.success(),"test SDK helper resolution failed");
                String::from_utf8(helper.stdout).unwrap().trim().into()
            }
        };
        let sdk=std::env::var("COZY_TEST_SDK_REQUIREMENTS").map(|value|
            serde_json::from_str::<Vec<String>>(&value).expect("test SDK requirements must be a JSON string array")
        ).unwrap_or_default();
        let client=fs::read_dir(root.join("client")).unwrap().map(|entry|entry.unwrap().path()).find(|path|path.extension().is_some_and(|value|value=="whl")).unwrap();
        let objects=Arc::new(Objects::new(&root.join("writes"),store.clone(),service.engine.clone()).unwrap());
        let local=LocalSources::new(objects.clone(),InstallerConfig {
            helper_python,python:"3.12".into(),generations:root.join("generations"),client_wheel:client,
            staging_root:root.join("staging"),sdk,uv:"uv".into(),
        },store.clone());
        let publisher=crate::published::Publisher::new(&root.join("published"),crate::published::PackageSdk::default(),store.clone()).unwrap();
        let runs=Arc::new(Runs {service:service.clone(),objects:objects.clone(),publisher:Some(publisher),local:Some(Arc::new(local)),own_hub:None,jobs:Default::default()});
        crate::jobs::Jobs::configure(&service,store.clone(),Some(&runs)).unwrap();
        let mut archive=tar::Builder::new(Vec::new());
        let fixture=repo.join("tests/fixtures/cpu_model_source");
        for name in ["pyproject.toml","package.toml","cpu_model_source/__init__.py"] {archive.append_path_with_name(fixture.join(name),name).unwrap();}
        let (source,length)=write(&objects,"alice",&archive.into_inner().unwrap());
        let local_manifest=json!({"package":"local/cozy-machine-cpu-model-source","release":"0.1.0","python_version":"3.12","source":{"digest":source,"length":length}});
        let (local_manifest,_)=write(&objects,"alice",local_manifest.to_string().as_bytes());
        let mut request=spec(Source::Local(local_manifest),false,"model-selector-job");
        request.job=true; request.entrypoint="metadata".into(); request.input=json!({});
        request.hub=Some(hub::Source {origin:origin.clone(),credential:"bearer cpu-model-source".into(),ca_der:None,object_hosts:vec![]});
        request.models=vec![pb::ModelChoice {parameter:"source".into(),repository:"models/fixture".into(),release:"1.0.0".into(),..Default::default()}];
        let accepted=runs.submit("alice","logical-model-job",request).unwrap();
        assert_eq!(accepted.waiting_reason.as_deref(),Some(crate::journal::PREPARING));
        let done=settled(&service.engine,&accepted.id);
        let mut stop=std::net::TcpStream::connect(origin.trim_start_matches("http://")).unwrap();
        stop.write_all(b"GET /stop HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap(); hub.join().unwrap();
        assert_eq!(done.state,State::Completed,"{:?}",done.failure);
        assert_eq!(done.result.unwrap().value["manifest"],manifest.id());
        assert_eq!(resolved.load(Ordering::Relaxed),1);
        tensorfs_core::gc::collect_cached_for(store.root(),&[],1).unwrap();
        assert!(store.manifest_path(&manifest.sha256).exists());
        let _=fs::remove_dir_all(root);
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
