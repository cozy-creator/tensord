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
use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
};

pub enum Source {
    Release { package: String, release: String },
    /// An installation this machine already holds for the signer.
    Installation(String),
    /// Unpublished code written with Write: its manifest's digest.
    Local(String),
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
    /// Running jobs' preparation, in memory only (the token is never journaled): their
    /// children prepare with it.
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
}

fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

impl Runs {
    /// Accepts the run (or answers the one this id already names) and starts its preparation.
    pub fn submit(self: &Arc<Self>, actor: &str, id: &str, spec: Spec) -> Result<Execution, Refused> {
        if !spec.input.is_object() {
            return Err(refused(
                "invalid_request",
                "the payload is a JSON object of the function's parameters",
            ));
        }
        for input in &spec.inputs {
            match self.objects.path(actor, &input.digest)? {
                Some((_, length)) if length == input.length => {}
                _ => {
                    return Err(refused(
                        "input_unwritten",
                        format!("input {} was not written to this machine", input.input_id),
                    ))
                }
            }
        }
        let package = match &spec.source {
            Source::Release { package, .. } => package.clone(),
            Source::Installation(alias) => alias.clone(),
            Source::Local(digest) => format!("local:{digest}"),
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
            .accept_run(actor, id, &spec.digest, draft)
            .map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => refused("run_id_conflict", e.to_string()),
                _ => Refused::from(e),
            })?;
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

    /// The run's code and models made ready: Ok(None) once it is dispatchable, Ok(Some) for a
    /// warm run's result.
    fn prepare(&self, actor: &str, id: &str, spec: Spec) -> Result<Option<ResultRecord>, Refused> {
        let engine = self.service.engine.clone();
        let run = id.to_string();
        // Byte counts arrive per chunk: a new stage shows at once, bytes at most 4 times a second.
        let last = std::sync::Mutex::new((String::new(), std::time::Instant::now()));
        let observe = move |stage: &str, done: u64, total: u64| {
            let mut last = last.lock().unwrap();
            if last.0 == stage && last.1.elapsed() < std::time::Duration::from_millis(250) && done < total {
                return;
            }
            *last = (stage.to_string(), std::time::Instant::now());
            let detail = json!({"stage": stage, "bytes_done": done, "bytes_total": total});
            let _ = engine.observe_progress(&run, 0, detail.to_string());
        };
        observe("preparing", 0, 0);
        let held = match &spec.source {
            Source::Release { .. } => None,
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
        let choices = if spec.job { &[][..] } else { &spec.models[..] };
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let (installation, plan) = self.resolve(
            actor,
            held,
            release,
            hub.clone(),
            &spec,
            choices,
            Box::new(observe),
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
        if !declared {
            return Err(refused(
                "invalid_entrypoint",
                format!("{} declares no {noun} {:?}", installation.package, spec.entrypoint),
            ));
        }
        if spec.job {
            for choice in &spec.models {
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
        // Device executors take file inputs; a job and its children run in them.
        if !spec.inputs.is_empty() && plan.is_none() && !spec.job && spec.parent.is_empty() {
            return Err(refused(
                "invalid_request",
                "file inputs reach device executors only; this CPU callable takes none",
            ));
        }
        if spec.warm {
            return Ok(Some(ResultRecord {
                value: json!({"package": installation.package, "release": installation.release}),
                artifacts: vec![],
                asset_bindings: vec![],
            }));
        }
        if spec.job {
            self.jobs.lock().unwrap().insert(
                id.into(),
                JobContext {
                    installation: installation.alias.clone(),
                    hub,
                    providers: spec.providers.clone(),
                    owner: spec.owner.clone(),
                    binding_revision: spec.binding_revision.clone(),
                    attention_kernel: spec.attention_kernel.clone(),
                    models: spec.models.clone(),
                },
            );
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
                let plan = self.configured(actor, &installed, &spec.entrypoint, choices)?;
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
        let context = self.jobs.lock().unwrap().get(parent).cloned().ok_or_else(|| {
            refused("child_call_refused", "the job's preparation context is gone")
        })?;
        let prefix = format!("{entrypoint}.");
        Ok(Spec {
            warm: false,
            job: false,
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
                    runs.resolve(&actor, held, None, hub, &spec, &models, Box::new(|_, _, _| ()))
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
    pub fn end_job(&self, id: &str) {
        self.jobs.lock().unwrap().remove(id);
    }

    /// Without a Hub, held code's models come from the operator's configured grants.
    fn configured(
        &self,
        actor: &str,
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
        let plan = gpu
            .prepare_root(actor, installed, entrypoint, choices, &[], 0)
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
            },
            store,
        );
        let runs = Arc::new(Runs {
            service: service.clone(),
            objects: objects.clone(),
            publisher: None,
            local: Some(Arc::new(local)),
            own_hub: None,
            jobs: Default::default(),
        });
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
        let theirs = runs
            .submit("bob", "run-1", spec(Source::Local(manifest), false, "d1"))
            .unwrap();
        let theirs = settled(&service.engine, &theirs.id);
        assert_eq!(theirs.state, State::Failed);
        assert!(theirs.failure.unwrap().contains("local_source_incomplete"));
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
