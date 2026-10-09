//! Published packages and their models, prepared from a Hub under the owner's access and
//! reused: a run whose installation and model resolution are held reads nothing from the
//! Hub. Port of the Python worker's release-root preparation (package set from the
//! release's locked requirements with uv; model by owner binding ladder, then a TensorFS
//! download) with its `release_root_preparing` progress answers.
use crate::{
    api::domain,
    gpu_service::{GpuPlan, GpuPool, ModelGrant},
    hub::{self, Catalog},
    journal::{Installation, Preparation},
    objects::Refused,
    service::Service,
};
use fs2::FileExt;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};
use tensorfs_core::{
    err::{Code, Refusal},
    fetch::{DeliveryGrant, FetchPlan},
    ids::ObjectRef,
    sha256,
    store::Store,
    transport::{self, Anonymous, Deadline, Ledger, Ranged, SourcePolicy},
};
use crate::catalog::normalized;

/// How each published environment gets its Runtime SDK. Empty `requirements` keeps the
/// release's own locked SDK rows.
#[derive(Clone, Debug, Default)]
pub struct PackageSdk {
    pub uv: PathBuf,
    pub python: String,
    pub requirements: Vec<String>,
    pub find_links: Option<PathBuf>,
    /// The machine's CPU runner client, so a published CPU package can run here too.
    pub client_wheel: Option<PathBuf>,
    /// An image's baked uv cache (expanded wheels, index records), linked rather than copied.
    pub seed_cache: Option<PathBuf>,
}

pub struct Prepared {
    pub installation: Installation,
    pub plan: Option<GpuPlan>,
}

#[derive(Clone)]
pub enum Progress {
    Preparing {
        stage: String,
        moved: u64,
        total: u64,
    },
    Ready(Arc<Prepared>),
    Failed(&'static str, String),
}

pub struct Request {
    pub source: hub::Source,
    pub package: String,
    pub release: String,
    /// The release's card as the run carries it; installing needs it.
    pub card: Option<Card>,
    /// Unpublished code this machine already holds, in place of the release.
    pub installed: Option<Installation>,
    /// The account an unpublished package's org-relative model names belong to.
    pub owner: String,
    /// Provider tokens for source models (memory only).
    pub providers: Providers,
    pub entrypoint: String,
    pub choices: Vec<domain::ModelChoice>,
}

/// A release as its Hub publishes it, carried by the run (th-241): the machine installs it
/// reading no Hub.
#[derive(Clone, Debug)]
pub struct Card {
    pub interface: Value,
    pub python_version: String,
    /// The locked requirements' text.
    pub lock: String,
}

/// The caller's provider tokens; empty reads public sources anonymously.
#[derive(Clone, Debug, Default)]
pub struct Providers {
    pub huggingface: String,
    pub civitai: String,
}

/// Where a preparation's stage and bytes are reported as they change.
pub type Observer = Box<dyn Fn(&str, u64, u64) + Send + Sync>;

#[derive(Default)]
struct Job {
    progress: Mutex<Progress>,
    changed: Condvar,
    observer: Option<Observer>,
}
impl Default for Progress {
    fn default() -> Self {
        Progress::Preparing {
            stage: String::new(),
            moved: 0,
            total: 0,
        }
    }
}
impl Job {
    fn set(&self, progress: Progress) {
        if let (Some(observe), Progress::Preparing { stage, moved, total }) =
            (&self.observer, &progress)
        {
            observe(stage, *moved, *total);
        }
        *self.progress.lock().unwrap() = progress;
        self.changed.notify_all();
    }
    fn stage(&self, stage: String) {
        self.set(Progress::Preparing {
            stage,
            moved: 0,
            total: 0,
        });
    }
    fn bytes(&self, moved: u64, total: u64) {
        if let Progress::Preparing {
            stage,
            moved: m,
            total: t,
        } = &mut *self.progress.lock().unwrap()
        {
            (*m, *t) = (moved, total);
            if let Some(observe) = &self.observer {
                observe(stage, moved, total);
            }
        }
    }
}

type Failure = (&'static str, String);

/// What one environment generation is built from: the release's locked requirements, its
/// Python, its package interface and its version.
#[derive(Clone, Copy)]
struct Environment<'a> {
    split: &'a Lock,
    python: &'a str,
    interface: &'a Value,
    release: &'a str,
    source: &'a hub::Source,
}

pub struct Publisher {
    root: PathBuf,
    sdk: PackageSdk,
    store: Arc<Store>,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    /// Manifests preparations are fetching, counted per preparation (`Fetching`).
    fetching: Mutex<HashMap<String, usize>>,
    /// Checkpoints a run started on part of: the rest downloads behind it (`complete_behind`).
    remainders: Mutex<HashMap<String, Remainder>>,
}

/// The rest of a checkpoint whose declared components a run downloaded first.
struct Remainder {
    repository: String,
    source: hub::Source,
    /// What the parts landed, held from GC until the whole checkpoint is recorded.
    holds: Vec<tensorfs_core::ensure::PartHold>,
    started: bool,
}

/// Streams a remainder downloads with while no preparation waits on it: well under a
/// foreground pull's, so the running request keeps the link and the disk.
const BEHIND_STREAMS: usize = 64;

/// One preparation's manifests in `Publisher::fetching`, released when it ends.
struct Fetching<'a> {
    publisher: &'a Publisher,
    manifests: Vec<String>,
}
impl Drop for Fetching<'_> {
    fn drop(&mut self) {
        let mut fetching = self.publisher.fetching.lock().unwrap();
        for manifest in &self.manifests {
            if let Some(count) = fetching.get_mut(manifest) {
                *count -= 1;
                if *count == 0 {
                    fetching.remove(manifest);
                }
            }
        }
    }
}

impl Publisher {
    pub fn new(root: &Path, sdk: PackageSdk, store: Arc<Store>) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            sdk,
            store,
            jobs: Mutex::new(HashMap::new()),
            fetching: Mutex::new(HashMap::new()),
            remainders: Mutex::new(HashMap::new()),
        }))
    }

    /// What one submission's preparation has reached. A held installation and resolution
    /// answer at once; otherwise preparation runs in the background and this waits for its
    /// next stage, its end, or a short while, so the client can show progress and ask again.
    pub fn prepare(
        self: &Arc<Self>,
        service: &Arc<Service>,
        actor: &str,
        submission: &str,
        request: Request,
    ) -> Progress {
        if let Ok(Some(prepared)) = self.held(service, actor, &request) {
            return Progress::Ready(Arc::new(prepared));
        }
        let key = format!("{actor}\0{submission}");
        let job = {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.entry(key.clone())
                .or_insert_with(|| {
                    let job = Arc::new(Job {
                        progress: Mutex::new(Progress::Preparing {
                            stage: match &request.installed {
                                Some(installed) => format!("preparing {}", installed.package),
                                None => format!("preparing {}@{}", request.package, request.release),
                            },
                            moved: 0,
                            total: 0,
                        }),
                        ..Default::default()
                    });
                    let (this, service, actor, worker) = (
                        self.clone(),
                        service.clone(),
                        actor.to_string(),
                        job.clone(),
                    );
                    std::thread::spawn(move || {
                        // A rental is not idle while it installs or downloads for a run.
                        let _preparing = service.preparing();
                        let result = this.work(&service, &actor, &request, &worker);
                        worker.set(match result {
                            Ok(prepared) => Progress::Ready(Arc::new(prepared)),
                            Err((code, detail)) => Progress::Failed(code, detail),
                        });
                    });
                    job
                })
                .clone()
        };
        let started = Instant::now();
        let mut progress = job.progress.lock().unwrap();
        let first = stage_of(&progress);
        while matches!(&*progress, Progress::Preparing { .. })
            && stage_of(&progress) == first
            && started.elapsed() < Duration::from_secs(20)
        {
            progress = job
                .changed
                .wait_timeout(progress, Duration::from_secs(2))
                .unwrap()
                .0;
        }
        let answer = progress.clone();
        drop(progress);
        if !matches!(answer, Progress::Preparing { .. }) {
            self.jobs.lock().unwrap().remove(&key);
        }
        answer
    }

    /// One run's preparation on the calling thread, reported through `observe`: a held
    /// installation and resolution answer at once.
    pub fn prepare_now(
        &self,
        service: &Arc<Service>,
        actor: &str,
        request: &Request,
        observe: Observer,
    ) -> Result<Prepared, Failure> {
        if let Ok(Some(prepared)) = self.held(service, actor, request) {
            return Ok(prepared);
        }
        let job = Job {
            observer: Some(observe),
            ..Default::default()
        };
        self.work(service, actor, request, &job)
    }

    /// A slot's model made from its provider source (TensorFS `source_model`), kept as the
    /// local repository named by the source and its profiles.
    fn source_grant(
        &self,
        installation: &Installation,
        path: &str,
        choice: &domain::ModelChoice,
        providers: &Providers,
        job: &Job,
    ) -> Result<ModelGrant, Failure> {
        if !choice.repository.is_empty() || choice.manifest.is_some() {
            return Err((
                "model_override_invalid",
                format!("{path} names a source and a catalog checkpoint"),
            ));
        }
        let made = make_source(&self.store, &choice.source, &choice.profiles, providers, &|stage, moved, total| {
            job.set(Progress::Preparing {
                stage: format!("{stage} for {path}"),
                moved,
                total,
            })
        })?;
        let manifest = self
            .store
            .read_manifest(&made.manifest)
            .map_err(|e| ("model_source_failed", e.to_string()))?;
        let header = manifest
            .header()
            .map(|reference| tensorfs_core::checkpoint::load_header(&self.store, reference))
            .transpose()
            .map_err(|e| ("model_source_failed", e.to_string()))?;
        Ok(ModelGrant {
            package: installation.package.clone(),
            slot: path.to_string(),
            repository: made.repository,
            release: String::new(),
            lane: String::new(),
            manifest: format!("sha256:{}", made.manifest.sha256),
            components: header
                .map(|h| h.components.into_iter().map(|(name, _)| name).collect())
                .unwrap_or_default(),
        })
    }

    /// The component names a held checkpoint's header declares.
    fn components(&self, manifest: &str) -> Result<Vec<String>, Failure> {
        let sha = manifest.trim_start_matches("sha256:");
        let absent = |e: String| ("checkpoint_absent", format!("{manifest}: {e}"));
        let length = fs::metadata(self.store.manifest_path(sha)).map_err(|e| absent(e.to_string()))?.len();
        let held = self
            .store
            .read_manifest(&tensorfs_core::ids::ObjectRef { sha256: sha.to_string(), length })
            .map_err(|e| absent(e.to_string()))?;
        let header = held
            .header()
            .map(|reference| tensorfs_core::checkpoint::load_header(&self.store, reference))
            .transpose()
            .map_err(|e| absent(e.to_string()))?;
        Ok(header.map(|h| h.components.into_iter().map(|(name, _)| name).collect()).unwrap_or_default())
    }

    /// A slot's model this machine's store holds, by its exact manifest.
    fn held_grant(&self, installation: &Installation, path: &str, reference: &domain::Ref) -> Result<ModelGrant, Failure> {
        let sha = sha256::hex(&reference.digest);
        let manifest = self.store.read_manifest(&tensorfs_core::ids::ObjectRef { sha256: sha.clone(), length: reference.length })
            .map_err(|_| ("checkpoint_absent", format!("{path} names checkpoint sha256:{sha}, which this machine does not hold")))?;
        let header = manifest
            .header()
            .map(|reference| tensorfs_core::checkpoint::load_header(&self.store, reference))
            .transpose()
            .map_err(|e| ("checkpoint_absent", e.to_string()))?;
        Ok(ModelGrant {
            package: installation.package.clone(),
            slot: path.to_string(),
            repository: String::new(),
            release: String::new(),
            lane: String::new(),
            manifest: format!("sha256:{sha}"),
            components: header
                .map(|h| h.components.into_iter().map(|(name, _)| name).collect())
                .unwrap_or_default(),
        })
    }

    /// The store models are made and held in.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// A provider source made into a local model (`make_source`), as a run's refusal.
    pub fn make_source(
        &self,
        source: &str,
        profiles: &[String],
        providers: &Providers,
        progress: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<tensorfs_core::source_model::Made, Refused> {
        make_source(&self.store, source, profiles, providers, progress)
            .map_err(|(code, message)| Refused { code, message })
    }

    /// A Hub checkpoint downloaded into the store (a warm run's model choice), held from
    /// eviction while it downloads.
    pub fn download(
        &self,
        service: &Service,
        source: &hub::Source,
        repository: &str,
        manifest: &str,
        bytes: &(dyn Fn(u64, u64) + Sync),
    ) -> Result<(), Refused> {
        // A checkpoint this store already holds whole, in that repository, asks the Hub nothing.
        if let Some(sha256) = manifest.strip_prefix("sha256:") {
            if let Ok(meta) = std::fs::metadata(self.store.manifest_path(sha256)) {
                let held = tensorfs_core::ids::ObjectRef { sha256: sha256.to_string(), length: meta.len() };
                if tensorfs_core::checkpoint_root::check_source(&self.store, repository, &held).is_ok() {
                    return Ok(());
                }
            }
        }
        let _fetching = self.fetch(vec![manifest.to_string()]);
        let keep = self.protected(service).map_err(io_failure).map_err(|(code, message)| Refused { code, message })?;
        ensure(&self.store, source, repository, manifest, &keep, bytes)
            .map_err(|(code, message)| Refused { code, message })
    }

    /// Downloads the rest of every checkpoint `run`'s preparation took part of, once the run
    /// is on its GPU. A remainder yields while a preparation fetches another checkpoint, and
    /// speeds up for one waiting on it; its parts stay held from GC until it is recorded.
    pub fn complete_behind(self: &Arc<Self>, service: &Arc<Service>, run: &str) {
        let pending: Vec<String> = {
            let mut remainders = self.remainders.lock().unwrap();
            remainders
                .iter_mut()
                .filter(|(_, remainder)| !remainder.started)
                .map(|(manifest, remainder)| {
                    remainder.started = true;
                    manifest.clone()
                })
                .collect()
        };
        for manifest in pending {
            let (this, service, run) = (self.clone(), service.clone(), run.to_string());
            let spawned = std::thread::Builder::new()
                .name("remainder".into())
                .spawn(move || this.remainder(&service, &run, &manifest));
            if let Err(error) = spawned {
                eprintln!("checkpoint remainder: {error}");
            }
        }
    }

    fn remainder(&self, service: &Service, run: &str, manifest: &str) {
        let tick = std::time::Duration::from_secs(1);
        while matches!(
            service.engine.get(run).map(|record| record.state),
            Ok(crate::journal::State::Queued | crate::journal::State::Starting)
        ) {
            std::thread::sleep(tick);
        }
        let Some((repository, source)) = self
            .remainders
            .lock()
            .unwrap()
            .get(manifest)
            .map(|r| (r.repository.clone(), r.source.clone()))
        else {
            return;
        };
        let others = || self.fetching.lock().unwrap().keys().any(|m| m != manifest);
        let waited_on = || self.fetching.lock().unwrap().contains_key(manifest);
        let outcome = loop {
            while others() {
                std::thread::sleep(tick);
            }
            let keep = match self.protected(service) {
                Ok(keep) => keep,
                Err(e) => break Err(io_failure(e)),
            };
            let urgent = waited_on();
            let cancellation = tensorfs_core::transport::PullCancellation::default();
            let pull = Pull {
                components: Vec::new(),
                streams: (!urgent).then_some(BEHIND_STREAMS),
                cancellation: Some(cancellation.clone()),
            };
            let done = std::sync::atomic::AtomicBool::new(false);
            let result = std::thread::scope(|scope| {
                scope.spawn(|| {
                    while !done.load(std::sync::atomic::Ordering::Acquire) {
                        std::thread::park_timeout(tick);
                        if others() || (!urgent && waited_on()) {
                            cancellation.cancel();
                            return;
                        }
                    }
                });
                let result = ensure_with(&self.store, &source, &repository, manifest, &keep, &pull, &|_, _| ());
                done.store(true, std::sync::atomic::Ordering::Release);
                result
            });
            match result {
                Ok(_) => break Ok(()),
                Err(_) if cancellation.is_cancelled() => continue,
                Err(failure) => break Err(failure),
            }
        };
        if let Err((code, detail)) = outcome {
            eprintln!("checkpoint {manifest}: the rest did not download ({code}: {detail})");
        }
        self.remainders.lock().unwrap().remove(manifest);
    }

    fn fetch(&self, manifests: Vec<String>) -> Fetching<'_> {
        let mut fetching = self.fetching.lock().unwrap();
        for manifest in &manifests {
            *fetching.entry(manifest.clone()).or_default() += 1;
        }
        Fetching {
            publisher: self,
            manifests,
        }
    }

    /// The store's share of the machine's self-managing caches (`reclaim`), with what it
    /// must keep (`protected`).
    pub fn caches(&self, service: &Service) -> io::Result<crate::reclaim::StoreCaches<'_>> {
        Ok(crate::reclaim::StoreCaches {
            store: &self.store,
            keep: self.protected(service)?,
        })
    }

    /// What no store GC may evict: what live executors read, what preparations are fetching
    /// (this one's included), and the prepared models and job model inputs of every
    /// unfinished run, paused and unknown states included. A journal or plan that cannot be
    /// read is an error, never a shorter list: the caller then runs no store GC this pass.
    fn protected(&self, service: &Service) -> io::Result<Vec<String>> {
        let gpu = service.gpu();
        let mut keep = gpu.as_ref().map(|gpu| gpu.serving()).unwrap_or_default();
        keep.extend(self.fetching.lock().unwrap().keys().cloned());
        keep.extend(self.remainders.lock().unwrap().keys().cloned());
        for record in service.engine.with_journal(|journal| journal.unfinished())? {
            if record.invocation.job {
                let context = service.engine.with_journal(|journal| journal.job_context(&record.id))?;
                if let Some(context) = context {
                    keep.extend(crate::runs::job_models(&context)?);
                }
            }
            let prepared = record.submission.filter(|s| !s.preparation_id.is_empty());
            let (Some(submission), Some(gpu)) = (prepared, &gpu) else {
                continue;
            };
            let preparation = service
                .engine
                .preparation(&submission.actor, &submission.preparation_id)?;
            if let Some(preparation) = preparation {
                let selections = gpu.plan(&preparation)?.selections();
                keep.extend(selections.into_iter().map(|s| s.manifest));
            }
        }
        keep.sort();
        keep.dedup();
        Ok(keep)
    }

    fn alias(&self, request: &Request) -> String {
        if let Some(installed) = &request.installed {
            return installed.alias.clone();
        }
        let origin = hub::origin_key(&request.source.origin).unwrap_or_default();
        let sdk = format!(
            "{:?}{:?}{:?}",
            self.sdk.requirements, self.sdk.find_links, self.sdk.client_wheel
        );
        let key = format!("{origin}\0{}\0{}\0{sdk}", request.package, request.release);
        format!("hub-{}", &sha256::hex_digest(key.as_bytes())[..32])
    }

    fn resolution_key(alias: &str, request: &Request, gpu: &str) -> String {
        let choices: Vec<_> = request
            .choices
            .iter()
            .map(|c| {
                json!([
                    c.parameter,
                    c.repository,
                    c.release,
                    c.lane,
                    c.manifest
                        .as_ref()
                        .map(|m| (sha256::hex(&m.digest), m.length)),
                    c.source,
                    c.profiles,
                    // Adapter order and scale are part of what the slot runs.
                    c.adapters
                        .iter()
                        .map(|a| [
                            &a.model,
                            &a.release,
                            &a.lane,
                            &a.manifest,
                            &a.component,
                            &a.source_component,
                            &a.scale,
                            &a.source
                        ])
                        .collect::<Vec<_>>()
                ])
            })
            .collect();
        let origin = hub::origin_key(&request.source.origin).unwrap_or_default();
        let package = request.installed.as_ref().map(|i| i.package.as_str()).unwrap_or(&request.package);
        let release = request.installed.as_ref().map(|i| i.release.as_str()).unwrap_or(&request.release);
        let key = json!({"installation":alias,"package":package,"release":release,"hub":origin,"owner":request.owner,"entrypoint":request.entrypoint,"choices":choices,"gpu":gpu});
        format!("hub-{}", sha256::hex_digest(key.to_string().as_bytes()))
    }

    /// The held installation and model plan, with no Hub read, or None.
    fn held(
        &self,
        service: &Service,
        actor: &str,
        request: &Request,
    ) -> io::Result<Option<Prepared>> {
        let held = match &request.installed {
            Some(installed) => Some(installed.clone()),
            None => service.engine.installation(actor, &self.alias(request))?,
        };
        let Some(installation) = held else {
            return Ok(None);
        };
        if service.catalog.resolve(&installation.generation).is_err() {
            return Ok(None);
        }
        if !declares_models(&installation, &request.entrypoint) {
            return Ok(Some(Prepared {
                installation,
                plan: None,
            }));
        }
        let Some(gpu) = service.gpu() else {
            return Ok(None);
        };
        let key = Self::resolution_key(
            &installation.alias,
            request,
            &format!("{}x{}", gpu.width(), gpu_name(&gpu.config().envelope()[0])),
        );
        let Some(id) = service.engine.with_journal(|j| j.resolution(actor, &key))? else {
            return Ok(None);
        };
        let Some(preparation) = service.engine.preparation(actor, &id)? else {
            return Ok(None);
        };
        let plan = gpu.plan(&preparation)?;
        // The bytes may have been reclaimed since; then the model is fetched again.
        if gpu.source_facts(&plan.selections()).is_err() {
            return Ok(None);
        }
        Ok(Some(Prepared {
            installation,
            plan: Some(plan),
        }))
    }

    fn work(
        &self,
        service: &Arc<Service>,
        actor: &str,
        request: &Request,
        job: &Job,
    ) -> Result<Prepared, Failure> {
        let held = match &request.installed {
            Some(installed) => Some(installed.clone()),
            None => service
                .engine
                .installation(actor, &self.alias(request))
                .map_err(io_failure)?,
        };
        let installation = match held {
            Some(held) if service.catalog.resolve(&held.generation).is_ok() => held,
            _ if request.installed.is_some() => {
                return Err((
                    "release_root_installation_absent",
                    "this owner has not prepared the named installation".into(),
                ))
            }
            _ => self.install(service, actor, request, job)?,
        };
        if !declares_models(&installation, &request.entrypoint) {
            return Ok(Prepared {
                installation,
                plan: None,
            });
        }
        let gpu = service.gpu().ok_or((
            "capability_unavailable",
            "this callable needs a GPU and this machine has none configured".to_string(),
        ))?;
        let plan = self.model(service, &gpu, actor, &installation, request, job)?;
        Ok(Prepared {
            installation,
            plan: Some(plan),
        })
    }

    fn install(
        &self,
        service: &Service,
        actor: &str,
        request: &Request,
        job: &Job,
    ) -> Result<Installation, Failure> {
        job.stage(format!(
            "installing {}@{}",
            request.package, request.release
        ));
        let (_, name) = request.package.split_once('/').ok_or((
            "release_root_invalid",
            "package must be org/name".to_string(),
        ))?;
        let card = request.card.as_ref().ok_or((
            "release_card_absent",
            format!("the run carries no card for {}@{}, which this signer has not installed", request.package, request.release),
        ))?;
        if !card.interface.is_object() {
            return Err((
                "package_prepare_interface_missing",
                "the release card carries no package interface".to_string(),
            ));
        }
        let interface = card.interface.clone();
        let python = Some(card.python_version.as_str())
            .filter(|v| !v.is_empty())
            .unwrap_or(&self.sdk.python)
            .to_string();
        let lock = card.lock.clone();
        let split = split_lock(
            &lock,
            name,
            &request.release,
            !self.sdk.requirements.is_empty(),
        )?;
        let identity = sha256::hex_digest(json!({"lock":lock,"python":python,"hub":hub::origin_key(&request.source.origin),"sdk":self.sdk.requirements,"links":self.sdk.find_links,"client":self.sdk.client_wheel}).to_string().as_bytes())[..32].to_string();
        let wanted = Environment {
            split: &split,
            python: &python,
            interface: &interface,
            release: &request.release,
            source: &request.source,
        };
        self.generation(service.catalog.root(), &identity, &wanted, job)?;
        let held = service.catalog.resolve(&identity).map_err(io_failure)?;
        if let Some(gpu) = service.gpu() {
            // Its kernel compiles and its imports overlap the model download.
            gpu.kernel_boot(held.clone());
            gpu.prespawn(held.clone());
        }
        service
            .engine
            .bind_installation(Installation {
                actor: actor.into(),
                alias: self.alias(request),
                generation: identity,
                package: request.package.clone(),
                release: request.release.clone(),
                interface: serde_json::to_vec(&held.record.interface)
                    .map_err(|e| io_failure(io::Error::other(e)))?,
            })
            .map_err(io_failure)
    }

    /// Builds one immutable environment at its final path (its interpreter embeds the path);
    /// only the atomic `generation.json` makes it resolvable. Concurrent preparations of the
    /// same identity wait for one another.
    fn generation(
        &self,
        generations: &Path,
        identity: &str,
        wanted: &Environment,
        job: &Job,
    ) -> Result<(), Failure> {
        let Environment {
            split,
            python,
            interface,
            release,
            source,
        } = *wanted;
        let locks = generations.join(".locks");
        fs::create_dir_all(&locks).map_err(io_failure)?;
        let lock = File::create(locks.join(format!("{identity}.lock"))).map_err(io_failure)?;
        lock.lock_exclusive().map_err(io_failure)?;
        let dir = generations.join(identity);
        if dir.join("generation.json").is_file() {
            return Ok(());
        }
        let _cache = self.uv_cache_hold().map_err(io_failure)?;
        if dir.exists() {
            fs::remove_dir_all(&dir).map_err(io_failure)?; // an interrupted, never-published build
        }
        fs::create_dir(&dir).map_err(io_failure)?;
        let write = |name: &str, body: &str| fs::write(dir.join(name), body).map_err(io_failure);
        let exact = self.hub_files(&split.exact, source, &dir, job)?;
        write("requirements.txt", &exact)?;
        write("constraints.txt", &split.constraints)?;
        let mut sdk_choice = if self.sdk.requirements.is_empty() {
            "locked"
        } else {
            "machine"
        };
        let env = dir.join("env");
        let interpreter = env.join("bin/python");
        let py = interpreter.to_string_lossy().to_string();
        job.stage(format!("creating its Python {python} environment"));
        self.uv(&[
            "venv",
            "--no-project",
            "--no-config",
            "--python",
            python,
            &env.to_string_lossy(),
        ])?;
        let requirements = dir.join("requirements.txt").to_string_lossy().to_string();
        // The last install compiles the whole environment's bytecode: executors run `-I` from
        // an environment they may not write, so uncompiled modules compile again in every
        // executor (SDXL's imports: 10.7 s uncompiled, 4.5 s compiled; J/BREAKDOWN.md).
        let sdk = !self.sdk.requirements.is_empty();
        let client = self.sdk.client_wheel.is_some();
        let compile = |last: bool| {
            if last {
                "--compile-bytecode"
            } else {
                "--no-compile-bytecode"
            }
        };
        let packages = split
            .exact
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with([' ', '\t', '#', '-']))
            .count();
        job.stage(format!("installing {packages} locked packages"));
        self.uv_observed(
            &[
                "pip",
                "install",
                "--no-config",
                "--python",
                &py,
                compile(!sdk && !client),
                "--require-hashes",
                "--no-deps",
                "--requirements",
                &requirements,
            ],
            &|read| job.bytes(read, 0),
        )?;
        let mut sdk_fallback = String::new();
        if sdk {
            job.stage("installing the machine's Runtime and TensorFS".into());
            sdk_fallback = self.install_sdk(&py, &dir, &split.sdk, compile(!client))?;
            if !sdk_fallback.is_empty() {
                sdk_choice = "locked";
                eprintln!("package {} {release}: {sdk_fallback}", split.distribution);
                job.stage(format!("warning: {sdk_fallback}"));
            }
        }
        if let Some(wheel) = &self.sdk.client_wheel {
            let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
            let wheel = wheel.to_string_lossy().to_string();
            self.uv(&[
                "pip",
                "install",
                "--no-config",
                "--python",
                &py,
                compile(true),
                "--constraints",
                &constraints,
                &wheel,
            ])?;
        }
        let application = interface
            .get("application")
            .and_then(Value::as_str)
            .ok_or((
                "package_prepare_interface_invalid",
                "the package interface names no application".to_string(),
            ))?;
        File::create(dir.join(".hold")).map_err(io_failure)?;
        let (source_digest, callees) = match &self.sdk.client_wheel {
            Some(_) => describe_environment(&py, &split.distribution, &source.origin)?,
            None => (String::new(), vec![]),
        };
        let record = json!({"identity":identity,"package":split.distribution,"version":release,"application":application,"python":interpreter,"dependencies":installed_sdk(&env),"interface":interface,"sdk":sdk_choice,"sdk_fallback":sdk_fallback,"source_digest":source_digest,"callees":callees});
        let staged = dir.join(".generation.json.new");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staged)
            .map_err(io_failure)?;
        file.write_all(record.to_string().as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(io_failure)?;
        fs::rename(&staged, dir.join("generation.json")).map_err(io_failure)?;
        File::open(&dir)
            .and_then(|d| d.sync_all())
            .map_err(io_failure)?;
        File::open(generations)
            .and_then(|d| d.sync_all())
            .map_err(io_failure)
    }

    /// The lock's rows its Hub publishes, through TensorFS's downloader instead of uv's: 1 MiB
    /// ranged parts side by side, a late one asked again, verified by the lock's sha256 into
    /// the store, then installed by uv from that file. uv asks one GET per wheel and waits out
    /// every stall of it (run 5107, oczy: the install sat in two ~147 s rounds behind the
    /// Hub's 302 to R2, while TensorFS moved 51 GiB from the same R2 in 91 s). PyPI rows stay
    /// uv's and the image's seeded cache's.
    fn hub_files(&self, exact: &str, source: &hub::Source, dir: &Path, job: &Job) -> Result<String, Failure> {
        let files: Vec<HubFile> = exact.lines().filter_map(|line| hub_file(line, source)).collect();
        if files.is_empty() {
            return Ok(exact.to_string());
        }
        job.stage(format!("downloading {} package files", files.len()));
        let policy = hub::files_policy(source).map_err(|e| ("catalog_read_failed", e.0))?;
        let wheels = dir.join("wheels");
        fs::create_dir(&wheels).map_err(io_failure)?;
        let fetched: Vec<Result<PathBuf, Failure>> = std::thread::scope(|scope| {
            let handles: Vec<_> = files
                .iter()
                .map(|file| scope.spawn(|| self.hub_wheel(file, &policy, &wheels)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or_else(|_| Err(io_failure(io::Error::other("a package file download panicked")))))
                .collect()
        });
        let mut local = HashMap::new();
        for (file, path) in files.iter().zip(fetched) {
            local.insert(file.url.as_str(), format!("file://{}", path?.display()));
        }
        Ok(exact
            .lines()
            .map(|line| match hub_file(line, source) {
                Some(file) => line.replacen(&file.url, &local[file.url.as_str()], 1),
                None => line.to_string(),
            })
            .map(|line| line + "\n")
            .collect())
    }

    fn hub_wheel(&self, file: &HubFile, policy: &SourcePolicy, wheels: &Path) -> Result<PathBuf, Failure> {
        let failed = |e: Refusal| ("package_file_download_failed", format!("{}: {e}", file.url));
        if !self.store.contains(&file.sha256) {
            let ledger = Ledger::new();
            let object = ObjectRef {
                sha256: file.sha256.clone(),
                length: probe(&file.url, policy, &ledger).map_err(failed)?,
            };
            let (plan, _) = FetchPlan::of_objects(&self.store, "package-files", std::slice::from_ref(&object))
                .map_err(failed)?;
            if !plan.wanted.is_empty() {
                let grant = DeliveryGrant::mint(&plan, &object).map_err(failed)?;
                let ranged = Ranged { streams: transport::STREAMS, part: 1 << 20 };
                transport::fetch_ranged(&self.store, &grant, &file.url, policy, &Anonymous, Deadline::none(), &ledger, ranged, None)
                    .map_err(failed)?;
            }
        }
        let blob = self.store.blob_path(&file.sha256);
        let path = wheels.join(&file.name);
        fs::hard_link(&blob, &path)
            .or_else(|_| fs::copy(&blob, &path).map(drop))
            .map_err(io_failure)?;
        Ok(path)
    }

    /// This machine's own uv cache, which `reclaim` manages; None when an image's seeded cache
    /// serves installs (environments symlink into it, so nothing there is ever removed).
    pub fn uv_cache(&self) -> Option<PathBuf> {
        self.sdk
            .seed_cache
            .is_none()
            .then(|| self.root.join("uv-cache"))
    }

    /// Held shared through one environment build, so `reclaim` removes nothing from the cache
    /// between uv's own per-command holds either.
    fn uv_cache_hold(&self) -> io::Result<Option<File>> {
        let Some(cache) = self.uv_cache() else {
            return Ok(None);
        };
        fs::create_dir_all(&cache)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(cache.join(".lock"))?;
        lock.lock_shared()?;
        Ok(Some(lock))
    }

    /// This machine's own Runtime and TensorFS in an environment holding a release's locked
    /// requirements, where the package's bounds admit them (uv's check of every installed
    /// requirement); else the release's locked SDK. A package's bounds are never overridden.
    /// Returns "" for the machine's pair, else why it fell back and to what: never silent.
    fn install_sdk(&self, py: &str, dir: &Path, locked: &str, compile: &str) -> Result<String, Failure> {
        let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
        let mut args = vec!["pip", "install", "--no-config", "--python", py, compile, "--constraints", &constraints];
        let links = self.sdk.find_links.as_ref().map(|p| p.to_string_lossy().to_string());
        if let Some(links) = &links {
            args.extend(["--find-links", links]);
        }
        args.extend(self.sdk.requirements.iter().map(String::as_str));
        // uv's failure names its step (`uv pip install` or `uv pip check`) and its output.
        let own = self.uv(&args).and_then(|()| self.uv(&["pip", "check", "--no-config", "--python", py]));
        let Err((code, detail)) = own else {
            return Ok(String::new());
        };
        let rows: Vec<&str> = locked
            .lines()
            .filter(|l| !l.starts_with("--"))
            .map(|l| l.split_whitespace().next().unwrap_or(l))
            .collect();
        if rows.is_empty() {
            return Err((code, detail));
        }
        fs::write(dir.join("sdk-requirements.txt"), locked).map_err(io_failure)?;
        let sdk = dir.join("sdk-requirements.txt").to_string_lossy().to_string();
        self.uv(&["pip", "install", "--no-config", "--python", py, compile, "--require-hashes", "--no-deps", "--requirements", &sdk])?;
        let detail: Vec<&str> = detail.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let detail = detail[detail.len().saturating_sub(12)..].join(" | ");
        Ok(format!(
            "this machine's own Runtime and TensorFS did not install ({code}: {}); it runs the release's locked {}",
            detail.chars().take(2000).collect::<String>(),
            rows.join(", ")
        ))
    }

    fn uv(&self, args: &[&str]) -> Result<(), Failure> {
        self.uv_observed(args, &|_| ())
    }

    /// `uv` with `read` told, about once a second, how many bytes the process has read so far
    /// (its downloads and the cache it copies from), so a long install shows it moves.
    fn uv_observed(&self, args: &[&str], read: &(dyn Fn(u64) + Sync)) -> Result<(), Failure> {
        let mut command = Command::new(&self.sdk.uv);
        command.args(args).env_clear().stdin(Stdio::null());
        for name in ["PATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "LANG"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .env("HOME", self.root.join("home"))
            .env("UV_PYTHON_INSTALL_DIR", self.root.join("python"));
        // As the Python worker: an image's seeded cache, symlinked so overlayfs never copies
        // the baked Torch payload into each environment; else this machine's own cache.
        match &self.sdk.seed_cache {
            Some(seed) => command
                .env("UV_CACHE_DIR", seed)
                .env("UV_LINK_MODE", "symlink"),
            None => command.env("UV_CACHE_DIR", self.root.join("uv-cache")),
        };
        let child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                (
                    "package_installation_uv_absent",
                    format!("cannot run uv: {e}"),
                )
            })?;
        let io = format!("/proc/{}/io", child.id());
        let ended = std::sync::atomic::AtomicBool::new(false);
        let output = std::thread::scope(|scope| {
            let sampler = scope.spawn(|| {
                while !ended.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::park_timeout(std::time::Duration::from_secs(1));
                    let rchar = fs::read_to_string(&io).ok().and_then(|text| {
                        text.lines()
                            .find_map(|l| l.strip_prefix("rchar: "))
                            .and_then(|v| v.trim().parse().ok())
                    });
                    if let (Some(rchar), false) =
                        (rchar, ended.load(std::sync::atomic::Ordering::Acquire))
                    {
                        read(rchar);
                    }
                }
            });
            let output = child.wait_with_output();
            ended.store(true, std::sync::atomic::Ordering::Release);
            sampler.thread().unpark();
            output
        })
        .map_err(|e| ("package_installation_uv_failed", format!("uv: {e}")))?;
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .chars()
            .rev()
            .take(2000)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        Err((
            "package_installation_uv_failed",
            format!(
                "uv {}: {}",
                args[..2.min(args.len())].join(" "),
                tail.trim()
            ),
        ))
    }

    /// The run's model grants, every Hub slot exact as its choice names it (th-241): a
    /// pinned checkpoint, or the widest rung of its binding this machine's GPUs fit. The
    /// widest fitting rung's GPU count is the group's width; 0 lets each slot's declared
    /// degrees decide. A slot with no choice is refused: the machine resolves nothing.
    fn model(
        &self,
        service: &Service,
        gpu: &GpuPool,
        actor: &str,
        installation: &Installation,
        request: &Request,
        job: &Job,
    ) -> Result<GpuPlan, Failure> {
        let interface: Value = serde_json::from_slice(&installation.interface).map_err(|_| {
            (
                "package_prepare_interface_invalid",
                "held interface is corrupt".to_string(),
            )
        })?;
        let slots = model_slots(&interface, &request.entrypoint)
            .filter(|slots| !slots.is_empty())
            .ok_or((
                "package_prepare_interface_invalid",
                "declared model slot absent".to_string(),
            ))?;
        let gpu_model = gpu_name(&gpu.config().envelope()[0]);
        let width = gpu.width();
        let mut grants = vec![];
        let mut degree = 0u32;
        for slot in &slots {
            let path = slot
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let parameter = path.rsplit('.').next().unwrap_or_default().to_string();
            job.stage(format!("choosing the model for {path}"));
            let choice = request
                .choices
                .iter()
                .find(|c| c.parameter == path || c.parameter == parameter)
                .ok_or(("model_choice_absent", format!("the run names no model for {path}")))?;
            if !choice.source.is_empty() {
                grants.push(self.source_grant(installation, &path, choice, &request.providers, job)?);
                continue;
            }
            if let (Some(reference), true) = (
                choice.manifest.as_ref().filter(|m| m.digest.len() == 32),
                choice.repository.is_empty(),
            ) {
                // A model passed by value (a ModelArtifact a run here produced): held here.
                grants.push(self.held_grant(installation, &path, reference)?);
                continue;
            }
            let exact = exact(choice, &path, (&gpu_model, width))?;
            degree = degree.max(exact.gpus);
            grants.push(ModelGrant {
                package: installation.package.clone(),
                slot: path,
                repository: exact.repository,
                release: exact.release,
                lane: exact.lane,
                manifest: exact.manifest,
                components: vec![],
            });
        }
        let _fetching = self.fetch(grants.iter().map(|g| g.manifest.clone()).collect());
        let keep = self.protected(service).map_err(io_failure)?;
        // One download per checkpoint (a source model is already here), all at once: a LoRA
        // never waits behind its 100 GB base.
        // A slot that declares the components its callable uses has those downloaded first;
        // the rest of its checkpoint follows once the run is on its GPU (`complete_behind`).
        let declared: HashMap<&str, Vec<&str>> = slots
            .iter()
            .filter_map(|slot| {
                let path = slot.get("path")?.as_str()?;
                let components = slot.get("components")?.as_array()?;
                Some((path, components.iter().filter_map(Value::as_str).collect()))
            })
            .collect();
        let mut parts: HashMap<&str, Option<Vec<String>>> = HashMap::new();
        for grant in &grants {
            // An exact choice names no components before its header is here: the slot's
            // declared ones are its part (a checkpoint lacking one downloads whole).
            let part: Option<Vec<String>> =
                declared.get(grant.slot.as_str()).map(|names| names.iter().map(|n| n.to_string()).collect());
            let part = part.filter(|part| !part.is_empty());
            // A checkpoint two slots share downloads whole if either needs all of it.
            let merged = match (parts.remove(grant.manifest.as_str()), part) {
                (Some(Some(mut a)), Some(b)) => {
                    a.extend(b);
                    a.sort();
                    a.dedup();
                    Some(a)
                }
                (None, part) => part,
                _ => None,
            };
            parts.insert(grant.manifest.as_str(), merged);
        }
        let mut fetched = std::collections::BTreeSet::new();
        let downloads: Vec<(&ModelGrant, Vec<String>)> = grants
            .iter()
            .filter(|g| fetched.insert(g.manifest.clone()) && !g.repository.is_empty() && !g.repository.starts_with("local/"))
            .map(|g| (g, parts.get(g.manifest.as_str()).cloned().flatten().unwrap_or_default()))
            .collect();
        let landed = download_all(&self.store, &request.source, &downloads, &keep, job)?;
        let mut remainders = self.remainders.lock().unwrap();
        for (grant, hold) in landed {
            remainders
                .entry(grant.manifest.clone())
                .or_insert_with(|| Remainder {
                    repository: grant.repository.clone(),
                    source: request.source.clone(),
                    holds: Vec::new(),
                    started: false,
                })
                .holds
                .push(hold);
        }
        drop(remainders);
        for grant in grants.iter_mut().filter(|g| g.components.is_empty() && !g.repository.is_empty()) {
            grant.components = self.components(&grant.manifest)?;
        }
        for grant in &mut grants {
            let parameter = grant
                .slot
                .rsplit('.')
                .next()
                .unwrap_or_default()
                .to_string();
            if let Some(choice) = request
                .choices
                .iter()
                .find(|c| c.parameter == grant.slot || c.parameter == parameter)
            {
                apply_adapters(&self.store, &request.source, choice, grant, &keep, &request.providers, job)?;
            }
        }
        job.stage(format!("preparing {}", request.package));
        let plan = gpu
            .prepare_root(
                installation,
                &request.entrypoint,
                &request.choices,
                &grants,
                degree,
            )
            .map_err(|e| ("model_preparation_failed", e.to_string()))?;
        service
            .engine
            .bind_preparation(Preparation {
                actor: actor.into(),
                id: plan.id.clone(),
                installation: installation.alias.clone(),
                document: serde_json::to_vec(&plan).map_err(|e| io_failure(io::Error::other(e)))?,
            })
            .map_err(io_failure)?;
        let key = Self::resolution_key(
            &installation.alias,
            request,
            &format!("{width}x{gpu_model}"),
        );
        service
            .engine
            .with_journal(|j| j.bind_resolution(actor, &key, &installation.package, &plan.id))
            .map_err(io_failure)?;
        Ok(plan)
    }
}

/// The GPU this machine's ladders are read for, and how many it groups (none: "", 0).
pub(crate) fn machine_gpu(service: &Service) -> (String, usize) {
    service.gpu().map_or((String::new(), 0), |gpu| {
        (gpu_name(&gpu.config().envelope()[0]), gpu.width())
    })
}

/// A Hub model slot's exact checkpoint as its choice names it (th-241: the caller resolved
/// it): the pinned manifest, or the widest rung of its binding this machine's GPUs fit.
pub(crate) struct Exact {
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub manifest: String,
    /// The rung's GPU count; 0 for a pinned checkpoint.
    pub gpus: u32,
}

pub(crate) fn exact(choice: &domain::ModelChoice, path: &str, (gpu, width): (&str, usize)) -> Result<Exact, Failure> {
    if choice.repository.is_empty() {
        return Err(("model_choice_inexact", format!("{path}'s choice names no model repository")));
    }
    let (lane, manifest, gpus) = match &choice.manifest {
        Some(reference) if reference.digest.len() == 32 => (choice.lane.clone(), &reference.digest, 0),
        _ if choice.rungs.is_empty() => {
            return Err(("model_choice_inexact", format!("{path}'s choice names no exact checkpoint")));
        }
        _ => {
            let rung = rung(&choice.rungs, gpu, width).ok_or((
                "model_binding_absent",
                format!("no rung of {path}'s binding fits {width}x {gpu:?}"),
            ))?;
            (rung.lane.clone(), &rung.manifest.digest, rung.gpus)
        }
    };
    Ok(Exact {
        repository: choice.repository.clone(),
        release: choice.release.clone(),
        lane,
        manifest: format!("sha256:{}", sha256::hex(manifest)),
        gpus,
    })
}


fn make_source(
    store: &Store,
    source: &str,
    profiles: &[String],
    providers: &Providers,
    progress: &(dyn Fn(&str, u64, u64) + Sync),
) -> Result<tensorfs_core::source_model::Made, Failure> {
    let mut sorted = profiles.to_vec();
    sorted.sort();
    let identity =
        serde_json_canonicalizer::to_vec(&json!(["cozy.machine-source-model/1", source, sorted]))
            .map_err(|e| io_failure(io::Error::other(e)))?;
    let name = format!("source-{}", &sha256::hex_digest(&identity)[..40]);
    let token = if source.starts_with("civitai://") {
        &providers.civitai
    } else {
        &providers.huggingface
    };
    let access = tensorfs_core::source_model::Access {
        credential: if token.is_empty() {
            String::new()
        } else {
            format!("bearer {token}")
        },
        endpoints: Default::default(),
    };
    tensorfs_core::source_model::make(
        store,
        &tensorfs_core::source_model::Request {
            source,
            profiles,
            name: &name,
            access: &access,
            registry: None,
            progress,
            cancellation: None,
        },
    )
    .map_err(|e| refused("model_source_failed", e))
}

/// Download one exact checkpoint, anonymously unless the run's capability names it (an
/// owner's unpublished checkpoint); `keep` names what its GC must not evict (`protected`).
fn ensure(
    store: &Store,
    source: &hub::Source,
    repository: &str,
    manifest: &str,
    keep: &[String],
    bytes: &(dyn Fn(u64, u64) + Sync),
) -> Result<(), Failure> {
    ensure_with(store, source, repository, manifest, keep, &Pull::default(), bytes).map(drop)
}

/// How one ensure pulls: some components only (held until dropped), its streams, its stop.
#[derive(Default)]
struct Pull {
    components: Vec<String>,
    streams: Option<usize>,
    cancellation: Option<tensorfs_core::transport::PullCancellation>,
}

fn ensure_with(
    store: &Store,
    source: &hub::Source,
    repository: &str,
    manifest: &str,
    keep: &[String],
    pull: &Pull,
    bytes: &(dyn Fn(u64, u64) + Sync),
) -> Result<Option<tensorfs_core::ensure::PartHold>, Failure> {
    let catalog = Catalog::for_op(source, hub::Op::Read { model: repository, manifest })?;
    let credential = catalog.credential();
    let refspec = format!("{repository}@{manifest}");
    let mut keep = keep.to_vec();
    keep.push(manifest.to_string());
    let on_event = |event: &tensorfs_core::ensure::Event| bytes(event.bytes_done, event.bytes_total);
    let mut request = tensorfs_core::ensure::Request::new(
        store,
        catalog.origin(),
        &refspec,
        credential,
        catalog.policy(),
    );
    request.keep = &keep;
    request.on_event = Some(&on_event);
    request.cancellation = pull.cancellation.clone();
    if let Some(streams) = pull.streams {
        request.streams = streams;
    }
    let refusal = |e| {
        let (code, message) = refused("model_download_failed", e);
        catalog.reason(code, message)
    };
    if !pull.components.is_empty() {
        request.components = &pull.components;
        match tensorfs_core::ensure::ensure_part(&request) {
            Ok((_, hold)) => return Ok(Some(hold)),
            // A component the slot declares that this checkpoint lacks: the whole of it, as
            // for a slot that declares none.
            Err(e) if e.code == tensorfs_core::err::Code::NOT_CONTAINED => request.components = &[],
            Err(e) => return Err(refusal(e)),
        }
    }
    tensorfs_core::ensure::ensure(&request).map(|_| None).map_err(refusal)
}

/// Every checkpoint a preparation needs, downloaded at once under one stage whose bytes are
/// their sum; a checkpoint with a part named downloads that part only, and answers its hold.
/// The first refusal is the answer, once every download has ended.
fn download_all<'g>(
    store: &Store,
    source: &hub::Source,
    grants: &[(&'g ModelGrant, Vec<String>)],
    keep: &[String],
    job: &Job,
) -> Result<Vec<(&'g ModelGrant, tensorfs_core::ensure::PartHold)>, Failure> {
    if grants.is_empty() {
        return Ok(Vec::new());
    }
    let names: Vec<String> = grants
        .iter()
        .map(|(g, part)| match part.is_empty() {
            true => format!("{}@{} {}", g.repository, g.release, g.lane),
            false => format!("{}@{} {} ({})", g.repository, g.release, g.lane, part.join(", ")),
        })
        .collect();
    job.stage(format!("downloading {}", names.join(" and ")));
    let moved = Mutex::new(vec![(0u64, 0u64); grants.len()]);
    let results: Vec<Result<Option<tensorfs_core::ensure::PartHold>, Failure>> = std::thread::scope(|scope| {
        let handles: Vec<_> = grants
            .iter()
            .enumerate()
            .map(|(at, (grant, part))| {
                let moved = &moved;
                scope.spawn(move || {
                    let pull = Pull { components: part.clone(), ..Pull::default() };
                    ensure_with(store, source, &grant.repository, &grant.manifest, keep, &pull, &|done, total| {
                        let (done, total) = {
                            let mut moved = moved.lock().unwrap();
                            moved[at] = (done, total);
                            moved.iter().fold((0, 0), |(d, t), (dd, tt)| (d + dd, t + tt))
                        };
                        job.bytes(done, total);
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(("model_download_failed", "a download thread panicked".to_string()))))
            .collect()
    });
    let mut held = Vec::new();
    for ((grant, _), result) in grants.iter().zip(results) {
        if let Some(hold) = result? {
            held.push((*grant, hold));
        }
    }
    Ok(held)
}

/// A TensorFS refusal as a run's reason; a download the disk cannot fit is its own.
fn refused(code: &'static str, refusal: tensorfs_core::err::Refusal) -> Failure {
    match refusal.code {
        tensorfs_core::err::Code::CAPACITY_EXHAUSTED => ("machine_disk_full", refusal.to_string()),
        _ => (code, refusal.to_string()),
    }
}

/// A slot's caller adapters applied to its resolved base: each adapter downloaded, then one
/// adapter view composed (`adapter_views`); the grant then names the view.
fn apply_adapters(
    store: &Store,
    source: &hub::Source,
    choice: &domain::ModelChoice,
    grant: &mut ModelGrant,
    keep: &[String],
    providers: &Providers,
    job: &Job,
) -> Result<(), Failure> {
    let mut keep = keep.to_vec();
    if choice.adapters.is_empty() {
        return Ok(());
    }
    job.stage(format!("preparing model adapters for {}", grant.slot));
    let object = |manifest: &str| -> Result<tensorfs_core::ids::ObjectRef, Failure> {
        let hex = manifest.trim_start_matches("sha256:");
        // A held manifest's file is its exact canonical bytes.
        let length = fs::metadata(store.manifest_path(hex))
            .map_err(|e| {
                (
                    "model_download_failed",
                    format!("manifest {hex} is not held: {e}"),
                )
            })?
            .len();
        Ok(tensorfs_core::ids::ObjectRef {
            sha256: hex.to_string(),
            length,
        })
    };
    let mut selections = vec![];
    for adapter in &choice.adapters {
        let manifest = if adapter.source.is_empty() {
            // Exact as the caller resolved it (th-241): the machine reads no model card.
            if adapter.model.is_empty() || !adapter.manifest.starts_with("sha256:") {
                return Err((
                    "adapter_inexact",
                    format!("an adapter of {} names no exact checkpoint", grant.slot),
                ));
            }
            ensure(store, source, &adapter.model, &adapter.manifest, &keep, &|d, t| job.bytes(d, t))?;
            adapter.manifest.clone()
        } else {
            // A provider-source LoRA (civitai://, hf://) is made here, normalized at ingest.
            let made = make_source(store, &adapter.source, &adapter.profiles, providers, &|stage, moved, total| {
                job.set(Progress::Preparing {
                    stage: format!("{stage} for an adapter of {}", grant.slot),
                    moved,
                    total,
                })
            })?;
            format!("sha256:{}", made.manifest.sha256)
        };
        keep.push(manifest.clone());
        let strength = if adapter.scale.is_empty() {
            1.0
        } else {
            adapter.scale.parse::<f64>().map_err(|_| {
                (
                    "model_override_invalid",
                    format!("adapter scale {:?} is not a decimal", adapter.scale),
                )
            })?
        };
        selections.push(crate::adapter_views::Selection {
            manifest: object(&manifest)?,
            component: adapter.component.clone(),
            source_component: adapter.source_component.clone(),
            strength,
        });
    }
    let composed = crate::adapter_views::compose(store, &object(&grant.manifest)?, &selections)
        .map_err(|e| ("model_adapter_refused", e.to_string()))?;
    grant.repository = composed.repository;
    grant.manifest = format!("sha256:{}", composed.manifest.sha256);
    Ok(())
}

fn stage_of(progress: &Progress) -> String {
    match progress {
        Progress::Preparing { stage, .. } => stage.clone(),
        _ => String::new(),
    }
}

/// A lock row the run's Hub publishes: `name @ <url>/<file>.whl ... --hash=sha256:<hex>`.
struct HubFile {
    url: String,
    sha256: String,
    name: String,
}

fn hub_file(line: &str, source: &hub::Source) -> Option<HubFile> {
    let mut words = line.split_whitespace();
    let (_, at, url) = (words.next()?, words.next()?, words.next()?);
    let sha256 = words.find_map(|word| word.strip_prefix("--hash=sha256:"))?;
    let name = url.rsplit('/').next()?;
    let plain = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+');
    (at == "@"
        && name.ends_with(".whl")
        && !name.starts_with('.')
        && name.chars().all(plain)
        && sha256.len() == 64
        && sha256.chars().all(|c| c.is_ascii_hexdigit())
        && hub::publishes(source, url))
    .then(|| HubFile {
        url: url.to_string(),
        sha256: sha256.to_ascii_lowercase(),
        name: name.to_string(),
    })
}

/// A file's length, asked again on a fresh connection while an ask stays silent past the
/// transport's noise floor, as its downloader asks a late part again.
fn probe(url: &str, policy: &SourcePolicy, ledger: &Ledger) -> Result<u64, Refusal> {
    let mut attempt = 1;
    loop {
        let deadline = Deadline::after_seconds(Some(ledger.floor().as_secs_f64()));
        match transport::probe_length(url, policy, &Anonymous, deadline, ledger) {
            Err(e) if e.code == Code::DEADLINE_EXCEEDED && attempt < transport::FETCH_ATTEMPTS => attempt += 1,
            answer => return answer,
        }
    }
}

fn io_failure(error: io::Error) -> Failure {
    ("release_root_preparation_failed", error.to_string())
}

fn model_slots(interface: &Value, entrypoint: &str) -> Option<Vec<Value>> {
    interface
        .get("entrypoints")?
        .as_array()?
        .iter()
        .find(|e| e.get("name").and_then(Value::as_str) == Some(entrypoint))?
        .get("models")?
        .as_array()
        .cloned()
}

pub fn declares_models(installation: &Installation, entrypoint: &str) -> bool {
    serde_json::from_slice(&installation.interface)
        .ok()
        .and_then(|i: Value| model_slots(&i, entrypoint))
        .is_some_and(|m| !m.is_empty())
}

/// The widest rung this machine holds (its GPU pattern fits the device and it asks for at
/// most `available` of them; the first among equals): its lane and GPU count (0 unstated).
/// The rung `available` GPUs named `gpu` run: the widest that fits, the owner's first among
/// equals. A rung's `gpu` is "*" (or empty), or tokens the GPU's name holds in order.
fn rung<'a>(rungs: &'a [domain::ModelRung], gpu: &str, available: usize) -> Option<&'a domain::ModelRung> {
    let fits = |pattern: &str| {
        if pattern.is_empty() || pattern == "*" {
            return true;
        }
        let have: Vec<String> = gpu
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect();
        let mut rest = have.iter();
        pattern
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .all(|token| rest.any(|t| t == token))
    };
    rungs
        .iter()
        .rev()
        .filter(|r| r.gpus as usize <= available)
        .filter(|r| fits(&r.gpu))
        .max_by_key(|r| r.gpus)
}

/// The configured device's model name from the NVIDIA driver's proc files (no CUDA/NVML).
fn gpu_name(device: &str) -> String {
    let mut rows: Vec<(String, String)> = fs::read_dir("/proc/driver/nvidia/gpus")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let text = fs::read_to_string(entry.path().join("information")).ok()?;
            let field = |name: &str| {
                text.lines()
                    .find_map(|l| l.strip_prefix(name))
                    .map(|v| v.trim_start_matches(':').trim().to_string())
            };
            Some((field("GPU UUID")?, field("Model")?))
        })
        .collect();
    rows.sort();
    match device.parse::<usize>() {
        Ok(index) => rows.get(index).map(|r| r.1.clone()),
        Err(_) => rows.iter().find(|r| r.0 == device).map(|r| r.1.clone()),
    }
    .unwrap_or_default()
}

struct Lock {
    exact: String,
    constraints: String,
    distribution: String,
    /// The lock's own SDK rows (with its index options), held back for an own-SDK machine.
    sdk: String,
}

/// The environment's root digest and the other Apps it holds, described inside it by the
/// machine's client without importing them (`runtime_describe`). A failed description fails
/// this preparation; it never publishes an environment with silently missing callees.
pub fn describe_environment(python: &str, root: &str, hub_origin: &str) -> Result<(String, Vec<crate::catalog::Callee>), Failure> {
    #[derive(serde::Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Reply {
        DescribedEnvironment { source_digest: String, callees: Vec<crate::catalog::Callee> },
        DescribeFailed { code: String, detail: String },
    }
    // -S keeps authored .pth startup code out; only the venv's own site-packages are added.
    const BOOTSTRAP: &str = "import pathlib,runpy,sys;\
        root=pathlib.Path(sys.executable).absolute().parent.parent;\
        sys.path[:0]=[str(p) for p in (root/'lib').glob('python*/site-packages')];\
        runpy.run_module('cozy_machine_client.runtime_describe',run_name='__main__')";
    let described = Command::new(python)
        .args(["-I", "-S", "-c", BOOTSTRAP])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .and_then(|mut child| {
            let request = json!({"kind": "describe_environment", "root": root, "hub_origin":hub_origin}).to_string();
            child.stdin.take().expect("piped").write_all(request.as_bytes())?;
            child.wait_with_output()
        });
    let output = described.map_err(io_failure)?;
    let reply = serde_json::from_slice::<Reply>(&output.stdout).map_err(|error| {
        ("package_prepare_description_failed", format!("environment description failed: {error}"))
    })?;
    match reply {
        Reply::DescribedEnvironment { source_digest, callees } if output.status.success() => Ok((source_digest, callees)),
        Reply::DescribeFailed { code, detail } => Err(("package_prepare_description_failed", format!("{code}: {detail}"))),
        _ => Err(("package_prepare_description_failed", "environment description did not complete".into())),
    }
}

/// The Runtime and TensorFS an environment holds, by their installed distribution names.
fn installed_sdk(env: &Path) -> Vec<Value> {
    let mut found: Vec<Value> = fs::read_dir(env.join("lib"))
        .into_iter()
        .flatten()
        .flatten()
        .flat_map(|python| fs::read_dir(python.path().join("site-packages")).into_iter().flatten().flatten())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let (distribution, version) = name.strip_suffix(".dist-info")?.split_once('-')?;
            let distribution = normalized(distribution);
            matches!(distribution.as_str(), "cozy-runtime" | "tensorfs")
                .then(|| json!({"name": distribution, "version": version}))
        })
        .collect();
    found.sort_by_key(|d| d["name"].as_str().unwrap_or_default().to_string());
    found
}

/// The release's lock as uv input: index lines and every row, minus the SDK rows when the
/// machine supplies its own SDK, whose resolution the other pins then constrain.
fn split_lock(lock: &str, name: &str, release: &str, own_sdk: bool) -> Result<Lock, Failure> {
    let (mut exact, mut constraints, mut distribution, mut sdk) =
        (String::new(), String::new(), None, String::new());
    // One requirement per line: `uv export` continues a requirement's `--hash` options on
    // the lines after it (`\` at a line's end).
    let lock = lock.replace("\\\r\n", " ").replace("\\\n", " ");
    for line in lock
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        if line.starts_with("--") {
            exact.push_str(line);
            exact.push('\n');
            sdk.push_str(line);
            sdk.push('\n');
            continue;
        }
        let end = line
            .find(|c: char| !(c.is_alphanumeric() || matches!(c, '-' | '_' | '.')))
            .unwrap_or(line.len());
        let row = normalized(&line[..end]);
        let pin = line[end..].trim_start().strip_prefix("==").map(|v| {
            v.split(|c: char| c.is_whitespace() || c == ';')
                .next()
                .unwrap_or_default()
        });
        if row == normalized(name)
            && (pin == Some(release) || line[end..].trim_start().starts_with('@'))
        {
            distribution = Some(line[..end].to_string());
        }
        if own_sdk && matches!(row.as_str(), "cozy-runtime" | "tensorfs") {
            sdk.push_str(line);
            sdk.push('\n');
            continue;
        }
        exact.push_str(line);
        exact.push('\n');
        if let Some(version) = pin {
            let marker = line
                .split_once(';')
                .map(|(_, m)| m.split(" --hash").next().unwrap_or_default().trim())
                .unwrap_or_default();
            constraints.push_str(&format!(
                "{}=={version}{}\n",
                &line[..end],
                if marker.is_empty() {
                    String::new()
                } else {
                    format!(" ; {marker}")
                }
            ));
        }
    }
    let distribution = distribution.ok_or((
        "package_prepare_project_pin_missing",
        format!("the lock does not pin {name}=={release}"),
    ))?;
    Ok(Lock {
        exact,
        constraints,
        distribution,
        sdk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wheel with nothing but its metadata: `name` `version`, requiring `requires`.
    fn tiny_wheel(dir: &Path, name: &str, version: &str, requires: &[&str]) -> PathBuf {
        use std::io::Write;
        let module = name.replace('-', "_");
        let info = format!("{module}-{version}.dist-info");
        let path = dir.join(format!("{module}-{version}-py3-none-any.whl"));
        let mut zip = zip::ZipWriter::new(File::create(&path).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        let mut metadata = format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n");
        for requirement in requires {
            metadata.push_str(&format!("Requires-Dist: {requirement}\n"));
        }
        let files = [
            (format!("{module}/__init__.py"), String::new()),
            (format!("{info}/METADATA"), metadata),
            (format!("{info}/WHEEL"), "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n".into()),
        ];
        let mut record = String::new();
        for (name, body) in &files {
            zip.start_file(name.as_str(), options).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
            record.push_str(&format!("{name},,\n"));
        }
        record.push_str(&format!("{info}/RECORD,,\n"));
        zip.start_file(format!("{info}/RECORD"), options).unwrap();
        zip.write_all(record.as_bytes()).unwrap();
        zip.finish().unwrap();
        path
    }

    /// A loopback Hub whose file door redirects to a loopback object store answering ranges,
    /// as the Hub's 302 to R2 does. Answers every request on its own connection.
    fn file_door(objects: HashMap<String, Vec<u8>>, asks: Arc<Mutex<Vec<String>>>, stall: Option<usize>) -> String {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let base = origin.clone();
        let stalled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (objects, asks, base, stalled) = (objects.clone(), asks.clone(), base.clone(), stalled.clone());
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let (mut range, mut header) = (None, String::new());
                    while reader.read_line(&mut header).unwrap_or(0) > 2 {
                        if let Some((name, value)) = header.split_once(':') {
                            if name.eq_ignore_ascii_case("range") {
                                let (first, last) = value.trim().trim_start_matches("bytes=").split_once('-').unwrap();
                                range = Some((first.parse::<usize>().unwrap(), last.parse::<usize>().unwrap()));
                            }
                        }
                        header.clear();
                    }
                    let mut words = line.split_whitespace();
                    let (method, path) = (words.next().unwrap().to_string(), words.next().unwrap().to_string());
                    asks.lock().unwrap().push(format!("{method} {path} {range:?}"));
                    // The first ask of the stalled part answers nothing, as a cold R2 object can.
                    if stall.is_some() && range.map(|(first, _)| first) == stall
                        && !stalled.swap(true, std::sync::atomic::Ordering::SeqCst)
                    {
                        std::thread::sleep(Duration::from_secs(120));
                        return;
                    }
                    let mut out = stream;
                    let answer = match path.split('/').collect::<Vec<_>>().as_slice() {
                        ["", "v1", "index", "acme", "files", sha, _] => format!(
                            "HTTP/1.1 302 Found\r\nLocation: {base}/objects/{sha}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .into_bytes(),
                        ["", "objects", sha] => {
                            let body = &objects[*sha];
                            let (status, first, last) = match range {
                                Some((first, last)) => ("206 Partial Content", first, last.min(body.len() - 1)),
                                None => ("200 OK", 0, body.len() - 1),
                            };
                            let mut answer = format!(
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Range: bytes {first}-{last}/{}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                last + 1 - first,
                                body.len()
                            )
                            .into_bytes();
                            if method == "GET" {
                                answer.extend_from_slice(&body[first..=last]);
                            }
                            answer
                        }
                        _ => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                    };
                    let _ = out.write_all(&answer);
                });
            }
        });
        origin
    }

    /// The lock's rows its Hub publishes come through TensorFS in ranged parts after the file
    /// door's redirect, land verified, and uv installs them from disk; a PyPI row is left to
    /// uv, and bytes that do not hash to the lock's sha256 are refused. Real uv, no network.
    #[test]
    fn hub_published_rows_download_ranged_and_install_from_disk() {
        use std::io::Write as _;
        let root = std::env::temp_dir().join(format!("cm-hub-files-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let wheel = root.join("acme_pkg-1.0-py3-none-any.whl");
        let mut zip = zip::ZipWriter::new(File::create(&wheel).unwrap());
        let stored = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        let payload: Vec<u8> = (0..(5u32 << 19)).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let info = "acme_pkg-1.0.dist-info";
        for (name, body) in [
            ("acme_pkg/__init__.py".to_string(), b"VALUE = 7\n".to_vec()),
            ("acme_pkg/blob.bin".into(), payload),
            (format!("{info}/METADATA"), b"Metadata-Version: 2.1\nName: acme-pkg\nVersion: 1.0\n".to_vec()),
            (format!("{info}/WHEEL"), b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n".to_vec()),
            (format!("{info}/RECORD"), format!("acme_pkg/__init__.py,,\nacme_pkg/blob.bin,,\n{info}/METADATA,,\n{info}/WHEEL,,\n{info}/RECORD,,\n").into_bytes()),
        ] {
            zip.start_file(name, stored).unwrap();
            zip.write_all(&body).unwrap();
        }
        zip.finish().unwrap();
        let bytes = fs::read(&wheel).unwrap();
        let good = sha256::hex_digest(&bytes);
        let bad = sha256::hex_digest(b"other bytes");
        let asks = Arc::new(Mutex::new(vec![]));
        let origin = file_door(HashMap::from([(good.clone(), bytes.clone()), (bad.clone(), bytes)]), asks.clone(), None);
        let source = hub::Source::new(&origin, None, vec![], None).unwrap();
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), PackageSdk::default(), store).unwrap();
        let pypi = "six @ https://files.pythonhosted.org/packages/six-1.17.0-py2.py3-none-any.whl --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274";
        let exact = format!(
            "--index-url https://pypi.org/simple\n{pypi}\nacme-pkg @ {origin}/v1/index/acme/files/{good}/acme_pkg-1.0-py3-none-any.whl --hash=sha256:{good}\n"
        );
        let dir = root.join("generation");
        fs::create_dir_all(&dir).unwrap();
        let rewritten = publisher.hub_files(&exact, &source, &dir, &Job::default()).unwrap();
        let local = dir.join("wheels/acme_pkg-1.0-py3-none-any.whl");
        assert_eq!(
            rewritten,
            format!("--index-url https://pypi.org/simple\n{pypi}\nacme-pkg @ file://{} --hash=sha256:{good}\n", local.display())
        );
        assert_eq!(sha256::hex_digest(&fs::read(&local).unwrap()), good);
        let parts = asks.lock().unwrap().iter().filter(|ask| ask.starts_with("GET /objects/") && ask.contains("Some") && !ask.contains("Some((0, 0))")).count();
        assert!(parts >= 3, "a 2.5 MiB wheel comes in 1 MiB parts: {:?}", asks.lock().unwrap());

        let row = rewritten.lines().last().unwrap();
        fs::write(dir.join("requirements.txt"), format!("{row}\n")).unwrap();
        let env = dir.join("env");
        let python = env.join("bin/python");
        for args in [
            vec!["venv", "-q", "--python", "3.12", env.to_str().unwrap()],
            vec!["pip", "install", "-q", "--offline", "--no-config", "--python", python.to_str().unwrap(),
                "--require-hashes", "--no-deps", "--requirements", dir.join("requirements.txt").to_str().unwrap()],
        ] {
            assert!(Command::new("uv").args(&args).env("UV_CACHE_DIR", root.join("uv-cache")).status().unwrap().success(), "{args:?}");
        }
        let value = Command::new(&python).args(["-c", "import acme_pkg; print(acme_pkg.VALUE)"]).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&value.stdout).trim(), "7");

        let tampered = format!("acme-pkg @ {origin}/v1/index/acme/files/{bad}/acme_pkg-1.0-py3-none-any.whl --hash=sha256:{bad}\n");
        let other = root.join("tampered");
        fs::create_dir_all(&other).unwrap();
        let (code, why) = publisher.hub_files(&tampered, &source, &other, &Job::default()).unwrap_err();
        assert_eq!(code, "package_file_download_failed", "{why}");
        assert!(!publisher.store.contains(&bad));
        let _ = fs::remove_dir_all(root);
    }

    /// A part whose first ask never answers is asked again once its peers are home, so the
    /// file lands in seconds where one GET (uv's) would wait out its read timeout.
    #[test]
    fn a_stalled_part_of_a_hub_file_is_asked_again() {
        let root = std::env::temp_dir().join(format!("cm-hub-stall-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("generation")).unwrap();
        let bytes: Vec<u8> = (0..(3u32 << 20)).map(|i| (i.wrapping_mul(2246822519) >> 11) as u8).collect();
        let sha = sha256::hex_digest(&bytes);
        let asks = Arc::new(Mutex::new(vec![]));
        let origin = file_door(HashMap::from([(sha.clone(), bytes)]), asks.clone(), Some(1 << 20));
        let source = hub::Source::new(&origin, None, vec![], None).unwrap();
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), PackageSdk::default(), store).unwrap();
        let exact = format!("acme-data @ {origin}/v1/index/acme/files/{sha}/acme_data-1.0-py3-none-any.whl --hash=sha256:{sha}\n");
        let began = Instant::now();
        publisher.hub_files(&exact, &source, &root.join("generation"), &Job::default()).unwrap();
        let took = began.elapsed();
        let stalled = asks.lock().unwrap().iter().filter(|ask| ask.ends_with("Some((1048576, 2097151))")).count();
        assert_eq!(stalled, 2, "{:?}", asks.lock().unwrap());
        assert!(took < Duration::from_secs(30), "took {took:?}");
        assert_eq!(sha256::hex_digest(&fs::read(root.join("generation/wheels/acme_data-1.0-py3-none-any.whl")).unwrap()), sha);
        let _ = fs::remove_dir_all(root);
    }

    /// The machine's own Runtime goes into an environment whose packages admit it; one whose
    /// bounds refuse it runs the release's locked SDK, and the generation says why (uv's own
    /// words) and which. Real uv and wheels, no network.
    #[test]
    fn a_refused_machine_sdk_falls_back_to_the_locked_one_and_says_so() {
        let root = std::env::temp_dir().join(format!("cm-sdk-{}", uuid::Uuid::new_v4()));
        let wheels = root.join("wheels");
        fs::create_dir_all(&wheels).unwrap();
        let own = tiny_wheel(&wheels, "cozy-runtime", "9.0", &[]);
        let locked = tiny_wheel(&root, "cozy-runtime", "1.0", &[]);
        let bounded = tiny_wheel(&root, "needs-old", "1.0", &["cozy-runtime<2"]);
        let open = tiny_wheel(&root, "takes-any", "1.0", &["cozy-runtime"]);
        let sdk = PackageSdk {
            uv: PathBuf::from("uv"),
            python: "3.12".into(),
            requirements: vec![own.to_string_lossy().into()],
            ..Default::default()
        };
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), sdk, store).unwrap();
        let lock = format!(
            "--no-index\n--find-links {}\ncozy-runtime==1.0 --hash=sha256:{}\n",
            root.display(),
            sha256::hex_digest(&fs::read(&locked).unwrap())
        );
        let environment = |name: &str, package: &Path| {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("constraints.txt"), "").unwrap();
            let env = dir.join("env");
            for args in [
                vec!["venv".to_string(), "-q".into(), "--python".into(), "3.12".into(), env.to_string_lossy().into()],
                vec!["pip".into(), "install".into(), "-q".into(), "--no-deps".into(), "--python".into(),
                    env.join("bin/python").to_string_lossy().into(), package.to_string_lossy().into()],
            ] {
                assert!(Command::new("uv").args(&args).status().unwrap().success(), "{args:?}");
            }
            (dir, env)
        };
        let (dir, env) = environment("admits", &open);
        let py = env.join("bin/python").to_string_lossy().to_string();
        assert_eq!(publisher.install_sdk(&py, &dir, &lock, "--no-compile-bytecode").unwrap(), "");
        assert_eq!(installed_sdk(&env), vec![json!({"name": "cozy-runtime", "version": "9.0"})]);

        let (dir, env) = environment("refuses", &bounded);
        let py = env.join("bin/python").to_string_lossy().to_string();
        let why = publisher.install_sdk(&py, &dir, &lock, "--no-compile-bytecode").unwrap();
        assert!(why.contains("uv pip check") && why.contains("needs-old"), "{why}");
        assert!(why.ends_with("it runs the release's locked cozy-runtime==1.0"), "{why}");
        assert_eq!(installed_sdk(&env), vec![json!({"name": "cozy-runtime", "version": "1.0"})]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lock_keeps_indexes_pins_and_drops_sdk_rows_only_for_an_own_sdk() {
        let lock = "--index-url https://pypi.org/simple\n--extra-index-url https://hub/v1/index/o/simple/\ncozy-runtime==0.18.67 ; sys_platform == 'linux' --hash=sha256:aa\nSDXL==2.3.24 --hash=sha256:bb\ntorch==2.14.0 ; platform_machine == 'x86_64' --hash=sha256:cc\n";
        let own = split_lock(lock, "sdxl", "2.3.24", true).unwrap();
        assert_eq!(own.distribution, "SDXL");
        assert!(
            own.exact.contains("--extra-index-url")
                && own.exact.contains("torch==2.14.0")
                && !own.exact.contains("cozy-runtime")
        );
        // The locked SDK stays available for a package whose bounds exclude the machine's.
        assert!(
            own.sdk.contains("cozy-runtime==0.18.67")
                && own.sdk.contains("--index-url")
                && !own.sdk.contains("torch")
        );
        assert!(
            own.constraints
                .contains("torch==2.14.0 ; platform_machine == 'x86_64'\n")
                && !own.constraints.contains("hash")
        );
        assert!(split_lock(lock, "sdxl", "2.3.24", false)
            .unwrap()
            .exact
            .contains("cozy-runtime"));
        assert_eq!(
            split_lock(lock, "sdxl", "9.9.9", true).err().unwrap().0,
            "package_prepare_project_pin_missing"
        );
    }

    /// `uv export` writes each requirement's hashes on continuation lines: they stay with it.
    #[test]
    fn a_requirement_continued_over_lines_keeps_its_hashes() {
        let lock = "cozy-runtime==0.18.102 \\\n    --hash=sha256:aa \\\n    --hash=sha256:bb\n    # via probe\nmsgspec==0.22.0 ; python_full_version >= '3.12' \\\n    --hash=sha256:cc\nprobe @ http://files/probe-1.0.0-py3-none-any.whl --hash=sha256:dd\n";
        let split = split_lock(lock, "probe", "1.0.0", true).unwrap();
        assert!(split
            .exact
            .lines()
            .chain(split.sdk.lines())
            .all(|l| !l.starts_with("--hash")));
        assert!(split.sdk.contains("cozy-runtime==0.18.102") && split.sdk.contains("sha256:bb"));
        assert!(split.exact.contains("msgspec==0.22.0") && split.exact.contains("sha256:cc"));
        assert_eq!(
            split.constraints,
            "msgspec==0.22.0 ; python_full_version >= '3.12'\n"
        );
    }

    fn rungs(ladder: &[(&str, u32, &str)]) -> Vec<domain::ModelRung> {
        ladder
            .iter()
            .enumerate()
            .map(|(i, (gpu, gpus, lane))| domain::ModelRung {
                gpu: gpu.to_string(),
                gpus: *gpus,
                lane: lane.to_string(),
                manifest: domain::Ref { digest: vec![i as u8; 32], length: 1 },
            })
            .collect()
    }

    #[test]
    fn rung_is_the_widest_group_the_machine_holds() {
        let lane = |ladder: &[domain::ModelRung], gpu, width| rung(ladder, gpu, width).map(|r| (r.lane.clone(), r.gpus));
        let ladder = rungs(&[("*", 0, "bf16"), ("rtx 4090", 0, "fp8"), ("h100", 4, "bf16")]);
        assert_eq!(lane(&ladder, "NVIDIA GeForce RTX 4090", 1), Some(("bf16".into(), 0)));
        let ladder = rungs(&[("rtx 4090", 0, "fp8")]);
        assert_eq!(lane(&ladder, "NVIDIA GeForce RTX 4090", 1), Some(("fp8".into(), 0)));
        assert_eq!(lane(&ladder, "NVIDIA A40", 2), None);
        let h3 = rungs(&[("H100", 2, "fp8-pruned"), ("H100", 4, "fp8-pruned"), ("H200", 1, "fp8-pruned")]);
        assert_eq!(lane(&h3, "NVIDIA H100 80GB HBM3", 1), None);
        assert_eq!(lane(&h3, "NVIDIA H100 80GB HBM3", 2), Some(("fp8-pruned".into(), 2)));
        assert_eq!(lane(&h3, "NVIDIA H100 80GB HBM3", 8), Some(("fp8-pruned".into(), 4)));
    }

    /// A Hub slot's checkpoint is exact as its choice names it, a pinned manifest or the rung
    /// this machine fits; anything else is refused, never resolved at a Hub.
    #[test]
    fn a_hub_choice_is_exact_or_refused() {
        let pinned = domain::ModelChoice {
            repository: "proof/probe".into(),
            lane: "bf16".into(),
            manifest: Some(domain::Ref { digest: vec![7; 32], length: 9 }),
            ..Default::default()
        };
        let got = self::exact(&pinned, "touch.models.source", ("", 0)).unwrap();
        assert_eq!((got.manifest, got.gpus), (format!("sha256:{}", "07".repeat(32)), 0));
        let laddered = domain::ModelChoice {
            repository: "proof/probe".into(),
            release: "1.0.0".into(),
            rungs: rungs(&[("h100", 2, "fp8"), ("*", 0, "bf16")]),
            ..Default::default()
        };
        let got = self::exact(&laddered, "touch.models.source", ("NVIDIA H100 80GB HBM3", 2)).unwrap();
        assert_eq!((got.lane.as_str(), got.gpus, got.release.as_str()), ("fp8", 2, "1.0.0"));
        assert_eq!(got.manifest, format!("sha256:{}", "00".repeat(32)));
        let got = self::exact(&laddered, "touch.models.source", ("NVIDIA A40", 1)).unwrap();
        assert_eq!(got.lane, "bf16");
        let named = domain::ModelChoice { repository: "proof/probe".into(), release: "1.0.0".into(), ..Default::default() };
        assert_eq!(self::exact(&named, "touch.models.source", ("", 0)).err().unwrap().0, "model_choice_inexact");
        let unnamed = domain::ModelChoice { manifest: pinned.manifest.clone(), ..Default::default() };
        assert_eq!(self::exact(&unnamed, "touch.models.source", ("", 0)).err().unwrap().0, "model_choice_inexact");
    }

    /// A source choice becomes the slot's grant through TensorFS `source_model`, as a local
    /// repository named by the source; a second choice of it is held. Real huggingface.co.
    #[test]
    #[ignore = "real network: huggingface.co"]
    fn a_source_choice_is_made_into_a_local_grant() {
        let root = std::env::temp_dir().join(format!("cm-source-{}", uuid::Uuid::new_v4()));
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), Default::default(), store).unwrap();
        let installation = Installation {
            actor: "alice".into(),
            alias: "x".into(),
            generation: String::new(),
            package: "org/sdxl".into(),
            release: "1".into(),
            interface: vec![],
        };
        let choice = domain::ModelChoice {
            parameter: "unet".into(),
            source: "hf://hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors".into(),
            ..Default::default()
        };
        let job = Job::default();
        let grant = publisher
            .source_grant(&installation, "generate.models.unet", &choice, &Providers::default(), &job)
            .unwrap();
        assert!(grant.repository.starts_with("local/source-"), "{grant:?}");
        assert!(grant.manifest.starts_with("sha256:"));
        assert!(!grant.components.is_empty());
        let again = publisher
            .source_grant(&installation, "generate.models.unet", &choice, &Providers::default(), &job)
            .unwrap();
        assert_eq!(again.manifest, grant.manifest);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A Civitai SDXL LoRA as an adapter source is normalized at ingest (sdxl.lora/1) into
    /// the PEFT factors the adapter view composes. Real civitai.com, anonymous.
    #[test]
    #[ignore = "real network: civitai.com"]
    fn a_civitai_lora_adapter_source_becomes_peft_factors() {
        let root = std::env::temp_dir().join(format!("cm-lora-{}", uuid::Uuid::new_v4()));
        let store = Store::ensure(&root.join("tensorfs")).unwrap();
        let made = make_source(&store, "civitai://145907", &[], &Providers::default(), &|_, _, _| {}).unwrap();
        assert_eq!(made.profiles, ["sdxl/lora-kohya/1"]);
        let manifest = store.read_manifest(&made.manifest).unwrap();
        let header = tensorfs_core::checkpoint::load_header(&store, manifest.header().unwrap()).unwrap();
        let (component, tensors) = &header.components[0];
        assert_eq!(component, "unet");
        assert!(tensors.iter().any(|(key, _)| key.ends_with("attn1.to_q.lora_A.weight")));
        let _ = std::fs::remove_dir_all(root);
    }
}
