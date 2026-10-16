//! Published packages and their models, prepared from a Hub by name (th-245) and reused: a
//! run whose installation and model resolution are held reads nothing from the Hub. Port of the
//! Python worker's release-root preparation (package set from the release's locked
//! requirements with uv; each model's lane by the lane rule (`precision`), then a TensorFS
//! download) with its `release_root_preparing` progress answers.
use crate::{
    access,
    api::domain,
    gpu_service::{Plan, Pool, ModelGrant},
    hub::{self, Catalog},
    journal::{Installation, Preparation},
    objects::Refused,
    precision,
    process::Supervised,
    service::Service,
};
use fs2::FileExt;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    os::unix::process::CommandExt,
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
    transport::{self, Anonymous, Deadline, Ledger, PullCancellation, Ranged, SourcePolicy},
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
    /// The machine's description client (`tensord_client`), installed into each generation.
    pub client_wheel: Option<PathBuf>,
    /// An image's baked uv cache (expanded wheels, index records), linked rather than copied.
    pub seed_cache: Option<PathBuf>,
    /// The development Runtime an update named by its wheel file (`0.24.6+e9baf7a4`): the
    /// operator's choice, which runs every package whatever its own bounds on the pair.
    pub runtime_build: String,
}

impl PackageSdk {
    /// Executors use the selected Runtime/TensorFS pair (an update's, else the image's), so they
    /// speak this machine's protocol; without a complete pair, each release's locked SDK.
    pub fn selected(wheels: &Path, root: &Path, uv: PathBuf) -> Self {
        // An activation replaces sdk/current, not the immutable candidate directory. Freeze
        // that selection once: both cache identity and the installer's actual wheel inputs
        // must name the same build even when filenames/version labels repeat.
        let wheels = wheels.canonicalize().unwrap_or_else(|_| wheels.to_path_buf());
        let mut pair: Vec<String> = fs::read_dir(&wheels)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "whl"))
            .map(|path| path.display().to_string())
            .collect();
        pair.sort();
        let complete = pair.iter().any(|w| w.contains("/cozy_runtime-")) && pair.iter().any(|w| w.contains("/tensorfs-"));
        let seed = root.join("opt/cozy/dependency-seed/uv-cache");
        Self {
            uv,
            python: "3.12".into(),
            runtime_build: match complete {
                true => fs::read_to_string(wheels.join(crate::machine::update::RUNTIME_BUILD)).unwrap_or_default(),
                false => String::new(),
            },
            find_links: complete.then_some(wheels),
            requirements: if complete { pair } else { vec![] },
            client_wheel: None,
            seed_cache: seed.is_dir().then_some(seed),
        }
    }
}

pub struct Prepared {
    pub installation: Installation,
    pub plan: Option<Plan>,
    /// The lane rule's warnings for the run (`precision`), held with its resolution.
    pub warnings: Vec<String>,
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
    /// Empty: the Hub's newest, as it named it under `catalog_revision`.
    pub release: String,
    /// Unpublished code this machine already holds, in place of the release.
    pub installed: Option<Installation>,
    /// The account an unpublished package's org-relative model names belong to.
    pub owner: String,
    /// The caller's catalog revision: a held resolution is reused only under the same one.
    pub catalog_revision: String,
    /// Provider tokens for source models (memory only).
    pub providers: Providers,
    pub entrypoint: String,
    pub choices: Vec<domain::ModelChoice>,
    /// 0 selects the widest supported machine group; otherwise exactly this many GPUs.
    pub gpus: u32,
    /// The request's weight precision (16, 8 or 4); 0 lets the lane rule choose per slot.
    pub bits: u32,
    /// A call's: its imports start as soon as it installs. A warm run imports only when its
    /// level asks (`downloaded` never does).
    pub imports: bool,
}

/// A Hub checkpoint as a run names it (th-245): `repository` at `release` (empty: the newest) in
/// `lane` (empty: the lane rule's), or the owner's checkpoint by hash (`manifest` and no
/// release). Resolved, it carries the release, lane and manifest the Hub named.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Named {
    pub repository: String,
    pub release: String,
    pub lane: String,
    /// `sha256:<hex>`; empty until resolved.
    pub manifest: String,
    /// The manifest's length, once resolved (0 for an adapter named by hash).
    pub length: u64,
}

impl Named {
    /// The checkpoint the Hub names it: the first of `asked` its release publishes (an empty
    /// lane: its only one). TensorFS's closure client names one lane, so it is one anonymous
    /// closure call per lane until one answers.
    fn at_hub(mut self, source: &hub::Source, asked: &[String]) -> Result<Self, Failure> {
        let refspec = match self.release.is_empty() {
            true => self.repository.clone(),
            false => format!("{}@{}", self.repository, self.release),
        };
        let catalog = Catalog::new(source).map_err(|e| ("catalog_read_failed", e.0))?;
        let mut answer = catalog.closure(&refspec, &asked[0]);
        for lane in &asked[1..] {
            if !matches!(&answer, Err(e) if e.code == Code::REF_NOT_FOUND) {
                break;
            }
            answer = catalog.closure(&refspec, lane);
        }
        let closure = answer.map_err(|e| refused("model_resolution_failed", e))?;
        (self.release, self.lane, self.manifest, self.length) =
            (closure.release, closure.lane, closure.manifest.id(), closure.manifest.length);
        Ok(self)
    }

    pub(crate) fn object(&self) -> ObjectRef {
        ObjectRef { sha256: self.manifest.trim_start_matches("sha256:").into(), length: self.length }
    }
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
    /// Its run's stop: a cancel ends its downloads and installs.
    stop: Option<PullCancellation>,
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

/// A preparation its run's cancel stopped.
pub const CANCELED: &str = "preparation_canceled";
fn canceled() -> Failure {
    (CANCELED, "the run was canceled while it prepared".into())
}

/// What one environment generation is built from: the release's locked requirements, its
/// Python, its package interface and its version.
#[derive(Clone, Copy)]
struct Environment<'a> {
    split: &'a Lock,
    org: &'a str,
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
    /// One maker per source: a second asker waits, then finds it held (runs 5393/5394).
    making: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl Publisher {
    pub fn new(root: &Path, sdk: PackageSdk, store: Arc<Store>) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            sdk,
            store,
            jobs: Mutex::new(HashMap::new()),
            making: Mutex::new(HashMap::new()),
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
                        // An update does not activate while this installs or downloads for a run.
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
        observe: Option<Observer>,
        stop: Option<&PullCancellation>,
    ) -> Result<Prepared, Failure> {
        if let Ok(Some(prepared)) = self.held(service, actor, request) {
            return Ok(prepared);
        }
        let job = Job {
            observer: observe,
            stop: stop.cloned(),
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
        let made = self.make(&choice.source, &choice.profiles, providers, job.stop.as_ref(), &|stage, moved, total| {
            job.set(Progress::Preparing {
                stage: format!("{stage} for {path}"),
                moved,
                total,
            })
        })?;
        Ok(ModelGrant {
            package: installation.package.clone(),
            slot: path.to_string(),
            repository: made.repository,
            release: String::new(),
            lane: String::new(),
            manifest: made.manifest.id(),
            components: self.components(&made.manifest.id())?,
        })
    }

    /// The component names a held checkpoint's header declares.
    fn components(&self, manifest: &str) -> Result<Vec<String>, Failure> {
        let header = access::held(&self.store, manifest).and_then(|held| access::header(&self.store, &held));
        let header = header.map_err(|e| ("checkpoint_absent", format!("{manifest}: {e}")))?;
        Ok(header.components.into_iter().map(|(name, _)| name).collect())
    }

    /// A slot's model this machine's store holds, by its exact manifest.
    fn held_grant(&self, installation: &Installation, path: &str, reference: &domain::Ref) -> Result<ModelGrant, Failure> {
        let manifest = format!("sha256:{}", sha256::hex(&reference.digest));
        let absent = |_| ("checkpoint_absent", format!("{path} names checkpoint {manifest}, which this machine does not hold"));
        Ok(ModelGrant {
            package: installation.package.clone(),
            slot: path.to_string(),
            repository: String::new(),
            release: String::new(),
            lane: String::new(),
            components: self.components(&manifest).map_err(absent)?,
            manifest,
        })
    }

    /// The store models are made and held in.
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    /// A provider source made into a local model (`make_source`), as a run's refusal.
    pub fn make_source(
        &self,
        source: &str,
        profiles: &[String],
        providers: &Providers,
        stop: Option<&PullCancellation>,
        progress: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<tensorfs_core::source_model::Made, Refused> {
        self.make(source, profiles, providers, stop, progress)
            .map_err(|(code, message)| Refused { code, message })
    }

    /// A provider source made into a local model (`make_source`), one maker per source.
    fn make(&self, source: &str, profiles: &[String], providers: &Providers,
        stop: Option<&PullCancellation>, progress: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<tensorfs_core::source_model::Made, Failure> {
        let mut sorted = profiles.to_vec();
        sorted.sort();
        let key = json!([source, sorted]).to_string();
        let gate = self.making.lock().unwrap().entry(key.clone()).or_default().clone();
        let made = {
            let _one = gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            make_source(&self.store, source, profiles, providers, stop.cloned(), progress)
        };
        let mut making = self.making.lock().unwrap();
        if Arc::strong_count(&gate) == 2 {
            making.remove(&key);
        }
        made
    }

    /// `named` downloaded whole on the Store's fetcher, as a request's weights are, and
    /// recorded so the GC keeps it: what a caller asked for whole (`cozy model download`, an
    /// adapter).
    fn ensure(&self, source: &hub::Source, named: &Named, stop: Option<&PullCancellation>,
        bytes: &(dyn Fn(u64, u64) + Sync)) -> Result<(u64, u64), Failure> {
        let origin = crate::access::origin(source, &named.repository, &named.release, &named.lane, &named.object());
        let catalog = access::of(source, &origin)?;
        let fetcher = tensorfs_core::fetch::Fetcher::of(&self.store).map_err(|e| refused("model_download_failed", e))?;
        let stop = stop.cloned().unwrap_or_default();
        let at = tensorfs_core::fetch::Priority::Now;
        tensorfs_core::fetch::whole(&fetcher, &origin, catalog.clone(), at, &stop, bytes).map_err(|e| match stop.is_cancelled() {
            true => canceled(),
            false => {
                let (code, message) = refused("model_download_failed", e);
                catalog.reason(code, message)
            }
        })
    }

    /// Each Hub checkpoint's structure (manifest, header, assets), landed at once: which of its
    /// objects come, and when, its calls direct (`access`). A held structure asks the Hub nothing.
    fn structure(&self, source: &hub::Source, checkpoints: &[Named], job: &Job) -> Result<(), Failure> {
        if checkpoints.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = checkpoints.iter().map(|n| format!("{}@{} {}", n.repository, n.release, n.lane)).collect();
        job.stage(format!("resolving {}", names.join(" and ")));
        let fetcher = tensorfs_core::fetch::Fetcher::of(&self.store).map_err(|e| refused("model_download_failed", e))?;
        let stop = job.stop.clone().unwrap_or_default();
        let mut wanted = vec![];
        for named in checkpoints {
            let origin = crate::access::origin(source, &named.repository, &named.release, &named.lane, &named.object());
            let catalog = access::of(source, &origin)?;
            let ticket = fetcher.want(&origin, catalog.clone(), &[], tensorfs_core::fetch::Priority::Now);
            wanted.push((ticket, catalog));
        }
        for (ticket, catalog) in wanted {
            ticket.wait(&[], &stop).map_err(|e| match stop.is_cancelled() {
                true => canceled(),
                false => {
                    let (code, message) = refused("model_download_failed", e);
                    catalog.reason(code, message)
                }
            })?;
        }
        Ok(())
    }

    /// The checkpoint `named` is: a hash as given, a name as the Hub answers it in its lane,
    /// else in the first of `lanes` its release publishes, else in its only lane. Journaled per
    /// catalog revision so a warm run, across restarts too, asks the Hub nothing (th-245).
    pub(crate) fn resolve(&self, service: &Service, source: &hub::Source, named: Named, lanes: &[&str], revision: &str) -> Result<Named, Failure> {
        if !named.manifest.is_empty() {
            return Ok(named);
        }
        let asked: Vec<String> = match named.lane.is_empty() {
            true => lanes.iter().chain(&[""]).map(|lane| lane.to_string()).collect(),
            false => vec![named.lane.clone()],
        };
        let origin = hub::origin_key(&source.origin).unwrap_or_default();
        let name = json!([origin, named.repository, named.release, asked]).to_string();
        let held = service.engine.with_journal(|j| j.named(&name, revision)).map_err(io_failure)?;
        if let Some((release, lane, manifest, length)) = held.and_then(|held| serde_json::from_str(&held).ok()) {
            return Ok(Named { release, lane, manifest, length, ..named });
        }
        let resolved = named.at_hub(source, &asked)?;
        let answer = json!([resolved.release, resolved.lane, resolved.manifest, resolved.length]).to_string();
        service.engine.with_journal(|j| j.bind_named(&name, revision, &answer)).map_err(io_failure)?;
        Ok(resolved)
    }

    /// A Hub checkpoint resolved (`resolve`) and downloaded whole into the store (a warm run's
    /// model choice: `cozy model download`); answers what the Hub named and the bytes that came
    /// from the Hub and from the datacenter volume.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn download(
        &self,
        service: &Service,
        source: &hub::Source,
        named: Named,
        lanes: &[&str],
        revision: &str,
        stop: Option<&PullCancellation>,
        bytes: &(dyn Fn(u64, u64) + Sync),
    ) -> Result<(Named, (u64, u64)), Refused> {
        let failed = |(code, message)| Refused { code, message };
        let named = self.resolve(service, source, named, lanes, revision).map_err(failed)?;
        // A checkpoint this store already holds whole, in that repository, moves nothing.
        let mut moved = (0, 0);
        if tensorfs_core::checkpoint_root::check_source(&self.store, &named.repository, &named.object()).is_err() {
            moved = self.ensure(source, &named, stop, bytes).map_err(failed)?;
        }
        Ok((named, moved))
    }

    /// The store's share of the machine's self-managing caches (`reclaim`), with what it
    /// must keep (`protected`).
    pub fn caches(&self, service: &Service) -> io::Result<crate::reclaim::StoreCaches<'_>> {
        Ok(crate::reclaim::StoreCaches {
            store: &self.store,
            keep: self.protected(service)?,
        })
    }

    /// What no store GC may evict: what live executors read and the checkpoints of every
    /// unfinished run, unknown states included (what is being fetched, its tickets hold). A
    /// journal or plan that cannot be read is an error, never a shorter list: the caller then
    /// runs no store GC this pass.
    fn protected(&self, service: &Service) -> io::Result<Vec<String>> {
        let mut keep = service.pool().map(|pool| pool.serving()).unwrap_or_default();
        for record in service.engine.with_journal(|journal| journal.unfinished())? {
            let Some(submission) = record.submission.filter(|s| !s.preparation_id.is_empty()) else {
                continue;
            };
            let preparation = service
                .engine
                .preparation(&submission.actor, &submission.preparation_id)?;
            if let Some(preparation) = preparation {
                keep.extend(crate::gpu_service::Plan::of(&preparation)?.manifests());
            }
        }
        keep.sort();
        keep.dedup();
        Ok(keep)
    }

    /// The installation a request names. An unnamed release is the one the Hub named under
    /// the caller's catalog revision (`newest`), so a warm run asks the Hub nothing.
    fn alias(&self, service: &Service, request: &Request) -> io::Result<String> {
        if let Some(installed) = &request.installed {
            return Ok(installed.alias.clone());
        }
        let release = match request.release.is_empty() {
            true => service.engine.with_journal(|j| j.named(&newest_key(request), &request.catalog_revision))?.unwrap_or_default(),
            false => request.release.clone(),
        };
        Ok(self.release_alias(request, &release))
    }

    /// The Hub's newest release of the request's package, journaled per catalog revision as a
    /// model's name is (`resolve`): a moved revision asks once more.
    fn newest(&self, service: &Service, request: &Request, catalog: &Catalog, org: &str, name: &str) -> Result<String, Failure> {
        let key = newest_key(request);
        if let Some(held) = service.engine.with_journal(|j| j.named(&key, &request.catalog_revision)).map_err(io_failure)? {
            return Ok(held);
        }
        let release = newest_release(catalog, org, name)?;
        service.engine.with_journal(|j| j.bind_named(&key, &request.catalog_revision, &release)).map_err(io_failure)?;
        Ok(release)
    }

    /// A release's card and locked requirements as an earlier installation of it kept them
    /// beside its environment (`CARD`): an exact release never changes, so its environment for
    /// another SDK reads nothing from the Hub.
    fn held_card(&self, service: &Service, actor: &str, request: &Request, release: &str) -> Option<Card> {
        let origin = hub::origin_key(&request.source.origin).unwrap_or_default();
        service.engine.installations(actor).ok()?.into_iter()
            .filter(|i| i.package == request.package && i.release == release && i.hub == origin)
            .find_map(|i| fs::read(service.catalog.root().join(&i.generation).join(CARD)).ok())
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    /// The installation of `release` of the request's package at its Hub, by this machine's SDK.
    fn release_alias(&self, request: &Request, release: &str) -> String {
        let origin = hub::origin_key(&request.source.origin).unwrap_or_default();
        let sdk = format!(
            "{:?}{:?}{:?}",
            self.sdk.requirements, self.sdk.find_links, self.sdk.client_wheel
        );
        let key = format!("{origin}\0{}\0{release}\0{sdk}", request.package);
        format!("hub-{}", &sha256::hex_digest(key.as_bytes())[..32])
    }

    /// A held resolution's key: the request, under the machine's GPUs as the lane rule reads
    /// them (`gpus`), never a card's name.
    fn resolution_key(alias: &str, request: &Request, gpus: &str) -> String {
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
        let key = json!({"installation":alias,"package":package,"release":release,"hub":origin,"owner":request.owner,"catalog":request.catalog_revision,"entrypoint":request.entrypoint,"choices":choices,"row":gpus,"gpus":request.gpus,"bits":request.bits});
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
            None => service.engine.installation(actor, &self.alias(service, request)?)?,
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
                warnings: vec![],
            }));
        }
        let Some(gpu) = service.pool() else {
            return Ok(None);
        };
        let key = Self::resolution_key(&installation.alias, request, &machine_row(&gpu));
        let Some(id) = service.engine.with_journal(|j| j.resolution(actor, &key))? else {
            return Ok(None);
        };
        let Some(preparation) = service.engine.preparation(actor, &id)? else {
            return Ok(None);
        };
        let plan = gpu.plan(&preparation)?;
        // Its structure may have been reclaimed since; then the model resolves again. An
        // object reclaimed is fetched again by whatever reads it.
        if !gpu.holds(&plan) {
            return Ok(None);
        }
        let warnings = service.engine.with_journal(|j| j.named(&warnings_key(&key), &request.catalog_revision))?;
        Ok(Some(Prepared {
            installation,
            plan: Some(plan),
            warnings: warnings.and_then(|held| serde_json::from_str(&held).ok()).unwrap_or_default(),
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
                .installation(actor, &self.alias(service, request).map_err(io_failure)?)
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
                warnings: vec![],
            });
        }
        let gpu = service.pool().ok_or((
            "capability_unavailable",
            "this machine runs no functions".to_string(),
        ))?;
        let (plan, warnings) = self.model(service, &gpu, actor, &installation, request, job)?;
        Ok(Prepared {
            installation,
            plan: Some(plan),
            warnings,
        })
    }

    /// The request's release installed for this signer: its card and locked requirements as an
    /// earlier installation of it kept them, else read at its Hub by name (th-245); the Hub's
    /// newest release when it names none (`newest`). An installation of that release this
    /// signer holds already is the answer.
    fn install(
        &self,
        service: &Service,
        actor: &str,
        request: &Request,
        job: &Job,
    ) -> Result<Installation, Failure> {
        let (org, name) = request.package.split_once('/').ok_or((
            "release_root_invalid",
            "package must be org/name".to_string(),
        ))?;
        let catalog = || Catalog::new(&request.source).map_err(|e| ("catalog_read_failed", e.0));
        let release = match request.release.is_empty() {
            true => self.newest(service, request, &catalog()?, org, name)?,
            false => request.release.clone(),
        };
        let alias = self.release_alias(request, &release);
        if let Some(held) = service.engine.installation(actor, &alias).map_err(io_failure)? {
            if service.catalog.resolve(&held.generation).is_ok() {
                return Ok(held);
            }
        }
        job.stage(format!("installing {}@{release}", request.package));
        let card = match self.held_card(service, actor, request, &release) {
            Some(card) => card,
            None => hub_card(&catalog()?, &request.package, &release, &self.sdk.python)?,
        };
        let Card { interface, python, lock } = &card;
        let split = split_lock(lock, name, &release, !self.sdk.requirements.is_empty())?;
        let identity = sha256::hex_digest(json!({"lock":lock,"python":python,"hub":hub::origin_key(&request.source.origin),"sdk":self.sdk.requirements,"links":self.sdk.find_links,"client":self.sdk.client_wheel}).to_string().as_bytes())[..32].to_string();
        let wanted = Environment {
            split: &split,
            org,
            python,
            interface,
            release: &release,
            source: &request.source,
        };
        self.generation(service.catalog.root(), &identity, &wanted, job)?;
        keep_card(&service.catalog.root().join(&identity), &card);
        let held = service.catalog.resolve(&identity).map_err(io_failure)?;
        if let Some(gpu) = service.gpu() {
            // Its kernels compile, and a call's imports run, while its weights come.
            gpu.kernel_boot(held.clone());
            if request.imports {
                gpu.prespawn(held.clone());
            }
        }
        service
            .engine
            .bind_installation(Installation {
                actor: actor.into(),
                alias,
                generation: identity,
                package: request.package.clone(),
                release,
                interface: serde_json::to_vec(&held.record.interface)
                    .map_err(|e| io_failure(io::Error::other(e)))?,
                hub: hub::origin_key(&request.source.origin).unwrap_or_default(),
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
            org,
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
        ], job.stop.as_ref())?;
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
            job.stop.as_ref(),
            &|read| job.bytes(read, 0),
        )?;
        let (mut sdk_choice, mut sdk_fallback) = ("locked", String::new());
        if sdk {
            job.stage("installing the machine's Runtime and TensorFS".into());
            (sdk_choice, sdk_fallback) = self.install_sdk(&py, &dir, &split.sdk, compile(!client), job.stop.as_ref())?;
            if !sdk_fallback.is_empty() {
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
            ], job.stop.as_ref())?;
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
            Some(_) => describe_environment(&py, &split.distribution, &hub_packages(&split.exact, org, source))?,
            None => (String::new(), vec![]),
        };
        let runtime_build = if sdk_choice == "machine" { self.sdk.runtime_build.as_str() } else { "" };
        let record = json!({"identity":identity,"package":split.distribution,"version":release,"application":application,"python":interpreter,"dependencies":installed_sdk(&env),"interface":interface,"sdk":sdk_choice,"sdk_fallback":sdk_fallback,"runtime_build":runtime_build,"source_digest":source_digest,"callees":callees});
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
                .map(|file| scope.spawn(|| self.hub_wheel(file, &policy, &wheels, job.stop.as_ref())))
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

    fn hub_wheel(&self, file: &HubFile, policy: &SourcePolicy, wheels: &Path, stop: Option<&PullCancellation>) -> Result<PathBuf, Failure> {
        let failed = |e: Refusal| match stop.is_some_and(PullCancellation::is_cancelled) {
            true => canceled(),
            false => ("package_file_download_failed", format!("{}: {e}", file.url)),
        };
        if !self.store.contains(&file.sha256) {
            let ledger = match stop {
                Some(stop) => Ledger::new().with_cancellation(stop.clone()),
                None => Ledger::new(),
            };
            let object = ObjectRef {
                sha256: file.sha256.clone(),
                length: probe(&file.url, policy, &ledger).map_err(failed)?,
            };
            fetch_file(&self.store, &file.url, &object, policy, &ledger).map_err(failed)?;
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
    /// requirement); else the release's locked SDK, and why. A development Runtime an update
    /// named (`--runtime-wheel`) is the operator's choice: it runs whatever the package's bounds
    /// on the pair, which the warning names, and one that cannot install refuses the release.
    /// Returns which SDK the environment holds and its run warning, if any: never silent.
    fn install_sdk(&self, py: &str, dir: &Path, locked: &str, compile: &str, stop: Option<&PullCancellation>) -> Result<(&'static str, String), Failure> {
        let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
        let mut args = vec!["pip", "install", "--no-config", "--python", py, compile, "--constraints", &constraints];
        let links = self.sdk.find_links.as_ref().map(|p| p.to_string_lossy().to_string());
        if let Some(links) = &links {
            args.extend(["--find-links", links]);
        }
        args.extend(self.sdk.requirements.iter().map(String::as_str));
        // uv's failure names its step (`uv pip install` or `uv pip check`) and its output.
        let own = self.uv(&args, stop).and_then(|()| self.uv(&["pip", "check", "--no-config", "--python", py], stop));
        let Err((code, detail)) = own else {
            return Ok(("machine", String::new()));
        };
        let build = &self.sdk.runtime_build;
        if code == CANCELED {
            return Err((code, detail));
        } else if !build.is_empty() {
            // uv's check names each requirement the installed pair misses; any other failure
            // (the pair's own dependencies against the lock's) leaves nothing it could run.
            let missed: Vec<&str> = detail.lines().map(str::trim).filter(|l| l.starts_with("The package `")).collect();
            if missed.is_empty() || !missed.iter().all(|l| pair_bound(l)) {
                return Err((code, format!("this machine's development Runtime {build} cannot run this release: {detail}")));
            }
            return Ok(("machine", format!("{}; this machine's development Runtime {build} (`--runtime-wheel`) runs it regardless", missed.join("; "))));
        }
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
        self.uv(&["pip", "install", "--no-config", "--python", py, compile, "--require-hashes", "--no-deps", "--requirements", &sdk], stop)?;
        let detail: Vec<&str> = detail.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let detail = detail[detail.len().saturating_sub(12)..].join(" | ");
        Ok(("locked", format!(
            "this machine's own Runtime and TensorFS did not install ({code}: {}); it runs the release's locked {}",
            crate::triage::ends(&detail, 2000),
            rows.join(", ")
        )))
    }

    fn uv(&self, args: &[&str], stop: Option<&PullCancellation>) -> Result<(), Failure> {
        self.uv_observed(args, stop, &|_| ())
    }

    /// `uv` with `read` told, about once a second, how many bytes the process has read so far
    /// (its downloads and the cache it copies from), so a long install shows it moves. Its
    /// process group is killed when `stop` is canceled, or when its CPU and bytes stop moving
    /// past its measured patience (`process::supervise`).
    fn uv_observed(&self, args: &[&str], stop: Option<&PullCancellation>, read: &(dyn Fn(u64) + Sync)) -> Result<(), Failure> {
        let mut command = Command::new(&self.sdk.uv);
        command.args(args).env_clear().stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
        command.process_group(0);
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
        let mut child = command.spawn().map_err(|e| {
            (
                "package_installation_uv_absent",
                format!("cannot run uv: {e}"),
            )
        })?;
        let mut stderr = child.stderr.take().expect("piped uv stderr");
        let (ended, stderr) = std::thread::scope(|scope| {
            let drained = scope.spawn(move || {
                let mut kept = Vec::new();
                let _ = stderr.read_to_end(&mut kept);
                kept
            });
            let ended = crate::process::supervise(&mut child, stop, "uv", read);
            (ended, drained.join().unwrap_or_default())
        });
        let step = args[..2.min(args.len())].join(" ");
        match ended.map_err(|e| ("package_installation_uv_failed", format!("uv: {e}")))? {
            Supervised::Exited(status) if status.success() => Ok(()),
            Supervised::Exited(_) => Err((
                "package_installation_uv_failed",
                format!("uv {step}:\n{}", crate::api::install::stderr_tail(&stderr)),
            )),
            Supervised::Canceled => Err(canceled()),
            Supervised::Wedged(verdict) => Err(("package_installation_stalled", format!("uv {step}: {verdict}"))),
        }
    }

    /// The run's model grants and its precision warnings. Each slot takes the run's choice, else
    /// its declared default (also under a choice of adapters alone); a Hub one is exact as its
    /// choice names it (th-241), else in the first lane of the rule's order its release
    /// publishes (`precision`). The request's GPU
    /// count chooses the group independently of the weight lane; 0 uses the widest group every
    /// slot supports on the machine. A slot with neither is refused.
    fn model(
        &self,
        service: &Service,
        gpu: &Pool,
        actor: &str,
        installation: &Installation,
        request: &Request,
        job: &Job,
    ) -> Result<(Plan, Vec<String>), Failure> {
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
        let degree = crate::gpu_service::group_degree(&slots, request.gpus, gpu.width());
        let gpus = gpu.gpus(degree);
        // One slot's grant, its Hub checkpoint and its warnings, `short` of memory or not.
        let pick = |slot: &Value, short: bool| -> Result<(ModelGrant, Option<Named>, Vec<String>), Failure> {
            let path = slot.get("path").and_then(Value::as_str).unwrap_or_default().to_string();
            let parameter = path.rsplit('.').next().unwrap_or_default().to_string();
            job.stage(format!("choosing the model for {path}"));
            let asked = request.choices.iter().find(|c| c.parameter == path || c.parameter == parameter)
                .filter(|c| !c.repository.is_empty() || c.manifest.is_some() || !c.source.is_empty());
            let declared = default_choice(slot, &path, &installation.package, &request.owner);
            let choice = asked.or(declared.as_ref()).ok_or_else(|| choice_absent(&path, &installation.package))?;
            if !choice.source.is_empty() {
                return Ok((self.source_grant(installation, &path, choice, &request.providers, job)?, None, vec![]));
            }
            if let (Some(reference), true) = (
                choice.manifest.as_ref().filter(|m| m.digest.len() == 32),
                choice.repository.is_empty(),
            ) {
                // A model passed by value (a ModelArtifact a run here produced): held here.
                return Ok((self.held_grant(installation, &path, reference)?, None, vec![]));
            }
            let named = choose(choice, &path)?;
            let tolerances = precision::Slot::of(slot);
            let bits = tolerances.bits(request.bits, short);
            let ruled = named.lane.is_empty() && named.manifest.is_empty();
            let lanes = if ruled { tolerances.lanes(bits, &gpus) } else { vec![] };
            let named = self.resolve(service, &request.source, named, &lanes, &request.catalog_revision)?;
            let model = format!("{}@{}", named.repository, named.release);
            let warnings = if ruled { tolerances.warnings(bits, &gpus, &model, &named.lane) } else { vec![] };
            let grant = ModelGrant {
                package: installation.package.clone(),
                slot: path,
                repository: named.repository.clone(),
                release: named.release.clone(),
                lane: named.lane.clone(),
                manifest: named.manifest.clone(),
                components: vec![],
            };
            Ok((grant, (!named.repository.starts_with("local/")).then_some(named), warnings))
        };
        // Each checkpoint lands as its structure: what of it comes, and when, its calls direct.
        let land = |picked: &[(ModelGrant, Option<Named>, Vec<String>)]| {
            self.structure(&request.source, &distinct(picked.iter().filter_map(|(_, named, _)| named.as_ref())), job)
        };
        let mut picked: Vec<_> = slots.iter().map(|slot| pick(slot, false)).collect::<Result<_, _>>()?;
        land(&picked)?;
        // Memory short (design §2.2): a `min_bits=4` compute slot takes 4 bits when the weights it
        // constructs exceed the budget, its last measured activations on these GPUs taken out.
        let shrinks = |slot: &Value| {
            let tolerances = precision::Slot::of(slot);
            request.bits == 0 && tolerances.min_bits == 4 && !tolerances.decode && !tolerances.raw
        };
        if slots.iter().any(&shrinks) {
            let constructed = slots.iter().zip(&picked).filter(|(slot, _)| crate::gpu_service::constructed(slot));
            let checkpoints = distinct(constructed.filter_map(|(_, (_, named, _))| named.as_ref()));
            let bytes: u64 = checkpoints.iter().map(|named| self.bytes(named)).sum();
            let measured: Vec<String> = service
                .engine
                .with_journal(|j| j.preparations_of(actor, &installation.alias))
                .map_err(io_failure)?
                .iter()
                .filter_map(|preparation| Plan::of(preparation).ok())
                .filter(|plan| plan.entrypoint == request.entrypoint)
                .map(|plan| plan.construction)
                .collect();
            if gpu.budget(degree, &measured).is_some_and(|budget| bytes > budget) {
                for (slot, chosen) in slots.iter().zip(&mut picked).filter(|(slot, _)| shrinks(slot)) {
                    *chosen = pick(slot, true)?;
                }
                land(&picked)?;
            }
        }
        let (mut grants, mut warnings) = (vec![], vec![]);
        for (grant, _, said) in picked {
            grants.push(grant);
            warnings.extend(said);
        }
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
                let resolve = |named| self.resolve(service, &request.source, named, &[], &request.catalog_revision);
                apply_adapters(self, &request.source, &resolve, choice, grant, &request.providers, job)?;
            }
        }
        job.stage(format!("preparing {}", request.package));
        let plan = gpu
            .prepare_root(
                installation,
                &request.entrypoint,
                &request.choices,
                &grants,
                request.gpus,
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
        let key = Self::resolution_key(&installation.alias, request, &machine_row(gpu));
        let said = Value::from(warnings.clone()).to_string();
        service
            .engine
            .with_journal(|j| {
                j.bind_resolution(actor, &key, &installation.package, &plan.id)?;
                j.bind_named(&warnings_key(&key), &request.catalog_revision, &said)
            })
            .map_err(io_failure)?;
        Ok((plan, warnings))
    }

    /// A landed checkpoint's bytes: its manifest and every object its header names.
    fn bytes(&self, named: &Named) -> u64 {
        let header = access::header(&self.store, &named.object());
        named.length + header.map_or(0, |header| access::objects(&header, &[]).iter().map(|o| o.length).sum())
    }
}

/// Each checkpoint once.
fn distinct<'a>(named: impl Iterator<Item = &'a Named>) -> Vec<Named> {
    let mut checkpoints = Vec::<Named>::new();
    for named in named {
        if !checkpoints.iter().any(|c| c.manifest == named.manifest) {
            checkpoints.push(named.clone());
        }
    }
    checkpoints
}

/// The machine's GPUs as a held resolution is keyed by: how many, and what they run.
fn machine_row(gpu: &Pool) -> String {
    format!("{}x{:?}", gpu.width(), gpu.gpus(gpu.width() as u32))
}

/// Where a held resolution's precision warnings are journaled.
fn warnings_key(resolution: &str) -> String {
    json!(["warnings", resolution]).to_string()
}

/// A Hub model slot's checkpoint as its choice names it (th-245): the owner's by hash, else
/// `repository` at `release` in the choice's lane; no lane is the lane rule's (`resolve`).
pub(crate) fn choose(choice: &domain::ModelChoice, path: &str) -> Result<Named, Failure> {
    if choice.repository.is_empty() {
        return Err(("model_choice_absent", format!("{path}'s choice names no model repository")));
    }
    let manifest = choice.manifest.as_ref().map(|m| format!("sha256:{}", sha256::hex(&m.digest))).unwrap_or_default();
    let length = choice.manifest.as_ref().map_or(0, |m| m.length);
    Ok(Named { repository: choice.repository.clone(), release: choice.release.clone(), lane: choice.lane.clone(), manifest, length })
}

fn make_source(
    store: &Store,
    source: &str,
    profiles: &[String],
    providers: &Providers,
    cancellation: Option<PullCancellation>,
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
            cancellation,
        },
    )
    .map_err(|e| refused("model_source_failed", e))
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
    publisher: &Publisher,
    source: &hub::Source,
    resolve: &dyn Fn(Named) -> Result<Named, Failure>,
    choice: &domain::ModelChoice,
    grant: &mut ModelGrant,
    providers: &Providers,
    job: &Job,
) -> Result<(), Failure> {
    if choice.adapters.is_empty() {
        return Ok(());
    }
    let store = &publisher.store;
    job.stage(format!("preparing model adapters for {}", grant.slot));
    let object = |manifest: &str| {
        access::held(store, manifest).map_err(|e| ("model_download_failed", format!("manifest {manifest} is not held: {e}")))
    };
    let mut selections = vec![];
    for adapter in &choice.adapters {
        let manifest = if adapter.source.is_empty() {
            // By name, resolved at the Hub, or the owner's by hash (th-245).
            if adapter.model.is_empty() {
                return Err(("adapter_absent", format!("an adapter of {} names no model", grant.slot)));
            }
            let named = resolve(Named {
                repository: adapter.model.clone(),
                release: adapter.release.clone(),
                lane: adapter.lane.clone(),
                manifest: adapter.manifest.clone(),
                length: 0,
            })?;
            // An adapter is small and its view composes from all of it: it comes whole.
            publisher.ensure(source, &named, job.stop.as_ref(), &|d, t| job.bytes(d, t))?;
            named.manifest
        } else {
            // A provider-source LoRA (civitai://, hf://) is made here, normalized at ingest.
            let made = publisher.make(&adapter.source, &adapter.profiles, providers, job.stop.as_ref(), &|stage, moved, total| {
                job.set(Progress::Preparing {
                    stage: format!("{stage} for an adapter of {}", grant.slot),
                    moved,
                    total,
                })
            })?;
            format!("sha256:{}", made.manifest.sha256)
        };
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

/// The newest release of `org/name` at the Hub (a run naming none): the newest final release
/// by version, else the newest pre-release; yanked ones never.
/// A catalog read the Hub answered 404 to is the thing named absent there; any other failure
/// is the read's.
fn absent(refusal: hub::Refusal, code: &'static str, named: impl FnOnce() -> String) -> Failure {
    match refusal.0.contains("HTTP 404") {
        true => (code, named()),
        false => ("catalog_read_failed", refusal.0),
    }
}

/// The journal name of an unnamed release of the request's package at its Hub.
fn newest_key(request: &Request) -> String {
    json!(["newest", hub::origin_key(&request.source.origin), request.package]).to_string()
}

/// What installing a release reads of it: its card's interface and Python, and its locked
/// requirements. Kept beside each environment built from it (`CARD`).
#[derive(serde::Deserialize, serde::Serialize)]
struct Card {
    interface: Value,
    python: String,
    lock: String,
}
const CARD: &str = "release.json";

/// `release`'s card and locked requirements, read at its Hub.
fn hub_card(catalog: &Catalog, package: &str, release: &str, python: &str) -> Result<Card, Failure> {
    let (org, name) = package.split_once('/').unwrap_or_default();
    let base = format!("/v1/packages/{}/{}/releases/{}", hub::escape(org), hub::escape(name), hub::escape(release));
    let card = catalog.json(&base).map_err(|e| absent(e, "release_absent", || {
        format!("{package} has no release {release} at {}", catalog.origin())
    }))?;
    if card.pointer("/release/release").and_then(Value::as_str) != Some(release) {
        return Err(("catalog_read_failed", format!("{base} named another release")));
    }
    let interface = card.get("package_interface").filter(|v| v.is_object()).cloned().ok_or((
        "package_prepare_interface_missing",
        "the release carries no package interface".to_string(),
    ))?;
    let python = card["python_version"].as_str().filter(|v| !v.is_empty()).unwrap_or(python).to_string();
    let lock = catalog
        .get(&format!("{base}/locked-requirements"), 16 << 20)
        .map_err(|e| ("catalog_read_failed", e.0))?;
    let lock = String::from_utf8(lock).map_err(|_| {
        ("package_prepare_locked_requirements_invalid", "locked requirements are not UTF-8".to_string())
    })?;
    Ok(Card { interface, python, lock })
}

/// `card` kept beside the environment at `generation` for the release's later environments.
/// A cache, written by rename and never synced: one lost is read from the Hub again.
fn keep_card(generation: &Path, card: &Card) {
    let staged = generation.join(format!(".{CARD}.{}", uuid::Uuid::new_v4().simple()));
    let kept = serde_json::to_vec(card).map_err(io::Error::other)
        .and_then(|bytes| fs::write(&staged, bytes))
        .and_then(|()| fs::rename(&staged, generation.join(CARD)));
    if let Err(error) = kept {
        eprintln!("release card {}: {error}", generation.display());
    }
}

fn newest_release(catalog: &Catalog, org: &str, name: &str) -> Result<String, Failure> {
    let card = catalog
        .json(&format!("/v1/packages/{}/{}", hub::escape(org), hub::escape(name)))
        .map_err(|e| absent(e, "package_absent", || format!("{org}/{name} is not published at {}", catalog.origin())))?;
    card["releases"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row["yanked"].as_bool() != Some(true) && row["yanked_at"].as_str().is_none_or(str::is_empty))
        .filter_map(|row| row["release"].as_str())
        .max_by(|a, b| release_order(a).cmp(&release_order(b)))
        .map(str::to_string)
        .ok_or(("release_absent", format!("{org}/{name} has no release that is not yanked")))
}

/// Finals after pre-releases, then by the numbers of the version and of its tag.
fn release_order(release: &str) -> (bool, Vec<u64>, Vec<u64>) {
    let numbers = |text: &str| -> Vec<u64> {
        text.split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().unwrap_or(u64::MAX))
            .collect()
    };
    let split = release.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(release.len());
    let (version, tag) = release.split_at(split);
    (tag.is_empty(), numbers(version), numbers(tag))
}

fn stage_of(progress: &Progress) -> String {
    match progress {
        Progress::Preparing { stage, .. } => stage.clone(),
        _ => String::new(),
    }
}

/// The Hub package each lock row its Hub publishes installs, by distribution: the row's own
/// name in the release's org, the one org a Hub lock pins. Never read from the row's URL: the
/// file door's shape is the Hub's to change (th-245 did, and every callee became
/// `local/<name>`, so choices addressed to `<org>/<name>/...` reached none: run 5234). Those
/// rows install from local files (`hub_files`), so the installed wheel names no Hub either.
fn hub_packages(exact: &str, org: &str, source: &hub::Source) -> BTreeMap<String, String> {
    exact
        .lines()
        .filter(|line| hub_file(line, source).is_some())
        .filter_map(|line| line.split_whitespace().next())
        .map(|name| (normalized(name), format!("{org}/{}", normalized(name))))
        .collect()
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

/// One file named by its digest, into this machine's store: a download that stops moving
/// fails on its measured stillness, `ledger`'s stop ends it, and what landed stays for the
/// next ask. Answers the blob's path.
pub(crate) fn fetch_file(store: &Store, url: &str, object: &ObjectRef, policy: &SourcePolicy, ledger: &Ledger) -> Result<PathBuf, Refusal> {
    let (plan, _) = FetchPlan::of_objects(store, "package-files", std::slice::from_ref(object))?;
    if !plan.wanted.is_empty() {
        let grant = DeliveryGrant::mint(&plan, object)?;
        let ranged = Ranged { streams: transport::STREAMS, part: 1 << 20 };
        transport::fetch_ranged(store, &grant, url, policy, &Anonymous, Deadline::none(), ledger, ranged, None)?;
    }
    Ok(store.blob_path(&object.sha256))
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

/// A slot's declared default as a choice (its `default`: `[org/]model@release`): what a run
/// that names no model for the slot gets, in the lane rule's lane. An org-relative model is the
/// package's account's: the package's org, or for unpublished code the run's `owner`.
pub(crate) fn default_choice(slot: &Value, path: &str, package: &str, owner: &str) -> Option<domain::ModelChoice> {
    let account = match package.split_once('/') {
        Some((org, _)) if org != "local" => org,
        _ => owner,
    };
    let (model, release) = slot.get("default")?.as_str()?.split_once('@')?;
    let repository = match model.contains('/') {
        true => model.into(),
        false if account.is_empty() => return None,
        false => format!("{account}/{model}"),
    };
    Some(domain::ModelChoice { parameter: path.into(), repository, release: release.into(), ..Default::default() })
}

/// A slot with neither a choice nor a declared default.
pub(crate) fn choice_absent(path: &str, package: &str) -> Failure {
    ("model_choice_absent", format!("the run names no model for {path}, and {package} declares no default for it"))
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
pub fn describe_environment(
    python: &str,
    root: &str,
    packages: &BTreeMap<String, String>,
) -> Result<(String, Vec<crate::catalog::Callee>), Failure> {
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
        runpy.run_module('tensord_client.runtime_describe',run_name='__main__')";
    let described = Command::new(python)
        .args(["-I", "-S", "-c", BOOTSTRAP])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .and_then(|mut child| {
            let request = json!({"kind": "describe_environment", "root": root, "packages": packages}).to_string();
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

/// uv's check line for a bound on the machine's own pair that the installed pair misses:
/// "The package `anima` requires `cozy-runtime[media]>=0.24.7`, but `0.24.6` is installed".
fn pair_bound(line: &str) -> bool {
    let required = line.split_once("` requires `").and_then(|(_, rest)| rest.split_once("`, but `")).map(|(r, _)| r);
    required.is_some_and(|r| {
        let end = r.find(|c: char| !(c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))).unwrap_or(r.len());
        matches!(normalized(&r[..end]).as_str(), "cozy-runtime" | "tensorfs")
    })
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
#[path = "../tests/common/names_hub.rs"]
pub(crate) mod names_hub;

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
                        // The Hub's file door (th-245): a wheel by its claiming package and release.
                        ["", "v1", "index", "acme", _, _, wheel] => format!(
                            "HTTP/1.1 302 Found\r\nLocation: {base}/objects/{wheel}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .into_bytes(),
                        ["", "objects", wheel] => {
                            let body = &objects[*wheel];
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
        let origin = file_door(HashMap::from([("acme_pkg-1.0-py3-none-any.whl".to_string(), bytes)]), asks.clone(), None);
        let source = hub::Source::new(&origin, None, vec![], None).unwrap();
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), PackageSdk::default(), store).unwrap();
        let pypi = "six @ https://files.pythonhosted.org/packages/six-1.17.0-py2.py3-none-any.whl --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274";
        let exact = format!(
            "--index-url https://pypi.org/simple\n{pypi}\nacme-pkg @ {origin}/v1/index/acme/acme-pkg/1.0/acme_pkg-1.0-py3-none-any.whl --hash=sha256:{good}\n"
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
        // Installed from disk, the wheel names no Hub; the lock row still does. The machine's
        // describer reads it inside the environment (msgspec is its one dependency).
        assert!(Command::new("uv").args(["pip", "install", "-q", "--no-config", "--python", python.to_str().unwrap(), "msgspec"])
            .env("UV_CACHE_DIR", root.join("uv-cache")).status().unwrap().success());
        let packages = hub_packages(&exact, "acme", &source);
        assert_eq!(packages, BTreeMap::from([("acme-pkg".to_string(), "acme/acme-pkg".to_string())]));
        let named = |packages: &BTreeMap<String, String>| {
            let script = format!(
                "import importlib.metadata,json;from tensord_client.distribution import package_name;\
                 print(package_name(importlib.metadata.distribution('acme-pkg'),json.loads('{}')))",
                serde_json::to_string(packages).unwrap()
            );
            let out = Command::new(&python).args(["-c", &script])
                .env("PYTHONPATH", Path::new(env!("CARGO_MANIFEST_DIR")).join("python")).output().unwrap();
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        assert_eq!(named(&packages), "acme/acme-pkg");
        assert_eq!(named(&BTreeMap::new()), "local/acme-pkg", "without the lock's identity");

        let tampered = format!("acme-pkg @ {origin}/v1/index/acme/acme-pkg/1.0/acme_pkg-1.0-py3-none-any.whl --hash=sha256:{bad}\n");
        let other = root.join("tampered");
        fs::create_dir_all(&other).unwrap();
        let (code, why) = publisher.hub_files(&tampered, &source, &other, &Job::default()).unwrap_err();
        assert_eq!(code, "package_file_download_failed", "{why}");
        assert!(!publisher.store.contains(&bad));
        let _ = fs::remove_dir_all(root);
    }

    /// A part whose first ask never answers is asked again once its peers are home, so the
    /// file lands in seconds where one GET (uv's) would wait out its read timeout.
    /// A package or release the Hub does not have is named absent there, not a failed read.
    #[test]
    fn an_absent_package_is_named() {
        let origin = file_door(HashMap::new(), Arc::new(Mutex::new(vec![])), None);
        let catalog = Catalog::new(&hub::Source::new(&origin, None, vec![], None).unwrap()).unwrap();
        let (code, message) = newest_release(&catalog, "acme", "nope").unwrap_err();
        assert_eq!(code, "package_absent");
        assert!(message.starts_with("acme/nope is not published at http://127.0.0.1:"), "{message}");
    }

    #[test]
    fn a_stalled_part_of_a_hub_file_is_asked_again() {
        let root = std::env::temp_dir().join(format!("cm-hub-stall-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("generation")).unwrap();
        let bytes: Vec<u8> = (0..(3u32 << 20)).map(|i| (i.wrapping_mul(2246822519) >> 11) as u8).collect();
        let sha = sha256::hex_digest(&bytes);
        let asks = Arc::new(Mutex::new(vec![]));
        let origin = file_door(HashMap::from([("acme_data-1.0-py3-none-any.whl".to_string(), bytes)]), asks.clone(), Some(1 << 20));
        let source = hub::Source::new(&origin, None, vec![], None).unwrap();
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), PackageSdk::default(), store).unwrap();
        let exact = format!("acme-data @ {origin}/v1/index/acme/acme-data/1.0/acme_data-1.0-py3-none-any.whl --hash=sha256:{sha}\n");
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
        assert_eq!(publisher.install_sdk(&py, &dir, &lock, "--no-compile-bytecode", None).unwrap(), ("machine", String::new()));
        assert_eq!(installed_sdk(&env), vec![json!({"name": "cozy-runtime", "version": "9.0"})]);

        let (dir, env) = environment("refuses", &bounded);
        let py = env.join("bin/python").to_string_lossy().to_string();
        let (sdk, why) = publisher.install_sdk(&py, &dir, &lock, "--no-compile-bytecode", None).unwrap();
        assert_eq!(sdk, "locked");
        assert!(why.contains("uv pip check") && why.contains("needs-old"), "{why}");
        assert!(why.ends_with("it runs the release's locked cozy-runtime==1.0"), "{why}");
        assert_eq!(installed_sdk(&env), vec![json!({"name": "cozy-runtime", "version": "1.0"})]);
        let _ = fs::remove_dir_all(root);
    }

    /// A Runtime wheel an update named (`cozy rental update --runtime-wheel`) runs every
    /// package: one whose floor it meets without a word, one whose floor it misses regardless,
    /// its runs warned with that floor and the build. Real update and uv, no network.
    #[test]
    fn a_named_runtime_wheel_runs_a_package_whose_floor_it_misses_and_says_so() {
        use crate::machine::update::{Choice, Paths, Request, Updates};
        let root = std::env::temp_dir().join(format!("cm-sdk-override-{}", uuid::Uuid::new_v4()));
        let (image, dev, wheels) = (root.join("opt/cozy/wheels"), root.join("dev"), root.join("wheels"));
        for dir in [&image, &dev, &wheels] {
            fs::create_dir_all(dir).unwrap();
        }
        tiny_wheel(&image, "cozy-runtime", "2.0", &[]);
        tiny_wheel(&image, "tensorfs", "1.0", &[]);
        let named = tiny_wheel(&dev, "cozy-runtime", "2.1", &[]);
        let paths = Paths::new(&root.join("engine"), &root);
        let updates = Updates::open(paths.clone(), Box::new(|| true), None, None).unwrap();
        let file = named.file_name().unwrap().to_string_lossy().to_string();
        let (digest, _) = updates.stage(&file, &mut File::open(&named).unwrap()).unwrap();
        let runtime = Some(Choice { file, sha256: digest.clone(), ..Default::default() });
        let request = Request { operation: "dev".into(), agent: "explicit".into(), pin: None, runtime, tensorfs: None };
        updates.request(request, |_| ()).unwrap();
        loop {
            let status = updates.update("dev").unwrap();
            if status.state == "starting" {
                break;
            }
            assert!(!status.terminal(), "{}: {}", status.state, status.error);
            std::thread::sleep(Duration::from_millis(10));
        }
        updates.commit().unwrap();
        let build = format!("2.1+{}", &digest[..8]);
        let sdk = PackageSdk::selected(&paths.sdk(), &root, "uv".into());
        assert_eq!(sdk.runtime_build, build);

        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), sdk, store).unwrap();
        let source = hub::Source::new("http://127.0.0.1:9", None, vec![], None).unwrap();
        let generations = root.join("generations");
        fs::create_dir_all(&generations).unwrap();
        let locked = tiny_wheel(&wheels, "cozy-runtime", "3.0", &[]);
        let hash = |wheel: &Path| sha256::hex_digest(&fs::read(wheel).unwrap());
        for (package, floor, warning) in [
            ("meets", "cozy-runtime>=2.1", String::new()),
            ("misses", "cozy-runtime[media]>=3", format!(
                "The package `misses` requires `cozy-runtime[media]>=3`, but `2.1` is installed; \
                 this machine's development Runtime {build} (`--runtime-wheel`) runs it regardless"
            )),
        ] {
            let wheel = tiny_wheel(&wheels, package, "1.0", &[floor]);
            let lock = format!(
                "--no-index\n--find-links {}\n{package}==1.0 --hash=sha256:{}\ncozy-runtime==3.0 --hash=sha256:{}\n",
                wheels.display(), hash(&wheel), hash(&locked),
            );
            let split = split_lock(&lock, package, "1.0", true).unwrap();
            let interface = json!({"application": format!("{package}:app")});
            let wanted = Environment { split: &split, org: "acme", python: "3.12", interface: &interface, release: "1.0", source: &source };
            publisher.generation(&generations, package, &wanted, &Job::default()).unwrap();
            let record: crate::catalog::Generation =
                serde_json::from_slice(&fs::read(generations.join(package).join("generation.json")).unwrap()).unwrap();
            let held: Vec<_> = record.dependencies.iter().map(|d| (d.name.as_str(), d.version.as_str())).collect();
            assert_eq!(held, [("cozy-runtime", "2.1"), ("tensorfs", "1.0")], "{package}");
            assert_eq!((record.runtime_build.as_str(), record.sdk_fallback), (build.as_str(), warning), "{package}");
        }
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

    /// A Hub slot's checkpoint is a name (in its lane, else the lane rule's) or the owner's
    /// hash; a choice naming no repository is absent.
    #[test]
    fn a_hub_choice_is_a_name_or_the_owners_hash() {
        let choose = |choice: &domain::ModelChoice| self::choose(choice, "touch.models.source");
        let pinned = domain::ModelChoice {
            repository: "proof/probe".into(),
            lane: "bf16".into(),
            manifest: Some(domain::Ref { digest: vec![7; 32], length: 9 }),
            ..Default::default()
        };
        let got = choose(&pinned).unwrap();
        assert_eq!((got.manifest, got.release), (format!("sha256:{}", "07".repeat(32)), String::new()));
        let named = domain::ModelChoice { repository: "proof/probe".into(), release: "1.0.0".into(), ..Default::default() };
        let got = choose(&named).unwrap();
        assert_eq!((got.release.as_str(), got.lane.as_str(), got.manifest.as_str()), ("1.0.0", "", ""));
        let unnamed = domain::ModelChoice { manifest: pinned.manifest.clone(), ..Default::default() };
        assert_eq!(choose(&unnamed).err().unwrap().0, "model_choice_absent");
    }

    #[test]
    fn model_identity_does_not_choose_the_execution_width() {
        let slot = json!({"path": "render.models.base", "sequence_parallel": {"degrees": [2, 4, 7, 8]}});
        for (requested, expected) in [(0, 4), (1, 1), (2, 2), (4, 4), (3, 2)] {
            assert_eq!(crate::gpu_service::group_degree(std::slice::from_ref(&slot), requested, 4), expected);
        }
    }

    #[test]
    fn an_explicit_gpu_count_and_bits_have_their_own_preparation_cache_key() {
        let mut request = Request {
            source: hub::Source::new("https://hub.example", None, vec![], None).unwrap(),
            package: "org/package".into(), release: "1".into(), installed: None,
            owner: "org".into(), catalog_revision: String::new(), providers: Providers::default(),
            entrypoint: "render".into(), choices: vec![], gpus: 0, bits: 0, imports: true,
        };
        let mut keys = vec![Publisher::resolution_key("installed", &request, "4xrow")];
        for (gpus, bits) in [(2, 0), (1, 0), (1, 4)] {
            (request.gpus, request.bits) = (gpus, bits);
            keys.push(Publisher::resolution_key("installed", &request, "4xrow"));
        }
        keys.push(Publisher::resolution_key("installed", &request, "4xother"));
        let distinct: std::collections::BTreeSet<_> = keys.iter().collect();
        assert_eq!(distinct.len(), keys.len());
    }

    /// A slot the run names no model for takes its declared default release (run 5327's
    /// `generate_image.models.model` failed `model_choice_absent`). An org-relative model is
    /// the package's account's: its org, or unpublished code's owner.
    #[test]
    fn an_unchosen_slot_takes_its_declared_default() {
        let path = "generate_image.models.model";
        let slot = json!({"path": path, "default": "qwen-image-2@1.0.0"});
        let published = default_choice(&slot, path, "paul/qwen-image-2", "").unwrap();
        assert_eq!((published.parameter.as_str(), published.repository.as_str(), published.release.as_str(), published.lane.as_str()),
            (path, "paul/qwen-image-2", "1.0.0", ""));
        assert_eq!(default_choice(&slot, path, "local/qwen-copy", "fidika").unwrap().repository, "fidika/qwen-image-2");
        assert!(default_choice(&slot, path, "local/qwen-copy", "").is_none(), "an org-relative default needs its account");
        let explicit = json!({"default": "proof/probe@2.1"});
        assert_eq!(default_choice(&explicit, path, "local/qwen-copy", "").unwrap().repository, "proof/probe");
        assert!(default_choice(&json!({"path": path}), path, "paul/qwen-image-2", "").is_none());
        assert_eq!(choice_absent(path, "paul/qwen-image-2").0, "model_choice_absent");
    }

    #[test]
    fn releases_order_by_version() {
        assert!(release_order("2.0.0") > release_order("10.0.0rc1"));
        assert!(release_order("10.0.0") > release_order("9.1.0"));
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
            hub: String::new(),
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

    /// Two runs making one source at once both get it, downloaded once: the second waits for
    /// the first instead of refusing STORE_BUSY (run 5394). Real huggingface.co.
    #[test]
    #[ignore = "real network: huggingface.co"]
    fn concurrent_makers_of_one_source_share_it() {
        let root = std::env::temp_dir().join(format!("cm-source-{}", uuid::Uuid::new_v4()));
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        let publisher = Publisher::new(&root.join("published"), Default::default(), store).unwrap();
        let source = "hf://hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors";
        let make = || publisher.make_source(source, &[], &Providers::default(), None, &|_, _, _| {}).unwrap();
        let (a, b) = std::thread::scope(|scope| {
            let a = scope.spawn(make);
            let b = scope.spawn(make);
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_eq!(a.manifest, b.manifest);
        assert!(publisher.making.lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    /// A Civitai SDXL LoRA as an adapter source is normalized at ingest (sdxl.lora/1) into
    /// the PEFT factors the adapter view composes. Real civitai.com, anonymous.
    #[test]
    #[ignore = "real network: civitai.com"]
    fn a_civitai_lora_adapter_source_becomes_peft_factors() {
        let root = std::env::temp_dir().join(format!("cm-lora-{}", uuid::Uuid::new_v4()));
        let store = Store::ensure(&root.join("tensorfs")).unwrap();
        let made = make_source(&store, "civitai://145907", &[], &Providers::default(), None, &|_, _, _| {}).unwrap();
        assert_eq!(made.profiles, ["sdxl/lora-kohya/1"]);
        let manifest = store.read_manifest(&made.manifest).unwrap();
        let header = tensorfs_core::checkpoint::load_header(&store, manifest.header().unwrap()).unwrap();
        let (component, tensors) = &header.components[0];
        assert_eq!(component, "unet");
        assert!(tensors.iter().any(|(key, _)| key.ends_with("attn1.to_q.lora_A.weight")));
        let _ = std::fs::remove_dir_all(root);
    }

    /// A real checkpoint of `objects` distinct one-object tensors, in a source store.
    fn checkpoint(store: &Store, objects: usize, tag: u8) -> ObjectRef {
        use tensorfs_core::{
            dtype::Dtype,
            header::{Body, Closure, Header, Part, Tensor},
            manifest::{Draft, Entry},
            registry,
            store::Fault,
        };
        let put = |bytes: &[u8]| store.put_stream(&mut &bytes[..], None, &Fault::default()).unwrap().obj;
        let spec = registry::seeds().into_iter().find(|seed| seed.alias == "plain/1").unwrap().spec;
        let tensors = (0..objects)
            .map(|at| {
                let value: Vec<u8> = (0..4096u32).flat_map(|i| [i.to_le_bytes(), [at as u8, tag, 0, 0]].concat()).collect();
                let shape = vec![2048u64, 4];
                let body = Body::Segments(vec![put(&value)]);
                let part = Part { dtype: Dtype::F32, shape: shape.clone(), body };
                let tensor = Tensor { dtype: Dtype::F32, shape, encoding: spec.object_id(), parts: vec![("value".into(), part)] };
                (format!("w{at:03}"), tensor)
            })
            .collect();
        let header = Header { configs: vec![], assets: vec![], encodings: vec![spec], components: vec![("model".into(), tensors)] };
        header.validate(&Closure::default()).unwrap();
        let header = put(&header.canonical_bytes().unwrap());
        let manifest = Draft { entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header))] }.seal().unwrap();
        store.put_manifest(&manifest).unwrap().obj
    }

    /// A machine's publisher and service, and a loopback Hub serving `model` from `source`
    /// whose every object answer takes `pace`.
    struct Bench {
        root: PathBuf,
        source: Arc<Store>,
        publisher: Arc<Publisher>,
    }

    impl Bench {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("cm-lanes-{}", uuid::Uuid::new_v4()));
            let source = Arc::new(Store::ensure(&root.join("source")).unwrap());
            let store = Arc::new(Store::ensure_owned(&root.join("store")).unwrap());
            let publisher = Publisher::new(&root.join("published"), PackageSdk::default(), store).unwrap();
            Self { root, source, publisher }
        }

        fn hub(&self, repository: &str, manifest: &ObjectRef, pace: u64) -> (hub::Source, Named, names_hub::Heard) {
            let model = names_hub::Model {
                repository: repository.into(),
                release: "1.0.0".into(),
                lane: "bf16".into(),
                manifest: manifest.clone(),
            };
            let (origin, heard) = names_hub::paced(&self.source, model, None, Duration::from_millis(pace));
            let named = Named { repository: repository.into(), release: "1.0.0".into(), lane: "bf16".into(), manifest: manifest.id(), length: manifest.length };
            (hub::Source::new(&origin, None, vec!["localhost".into()], None).unwrap(), named, heard)
        }
    }

    fn count(heard: &names_hub::Heard, asked: &str) -> usize {
        heard.lock().unwrap().iter().filter(|h| h.starts_with(asked)).count()
    }

    /// One channel request answered (frames: a kind byte, a u32 length, JSON); Null once the
    /// channel is closed.
    fn ask(channel: &mut std::os::unix::net::UnixStream, request: Value) -> Value {
        use std::io::{Read, Write};
        let body = serde_json::to_vec(&request).unwrap();
        let frame = [&[0u8][..], &(body.len() as u32).to_be_bytes(), &body].concat();
        let mut head = [0; 5];
        if channel.write_all(&frame).is_err() || channel.read_exact(&mut head).is_err() {
            return Value::Null;
        }
        let mut reply = vec![0; u32::from_be_bytes(head[1..].try_into().unwrap()) as usize];
        channel.read_exact(&mut reply).unwrap();
        serde_json::from_slice(&reply).unwrap()
    }

    /// A checkpoint a run chose lands as its structure alone: one closure, its manifest and
    /// header, and an origin attested for what it lacks. Resolved again, it asks the Hub nothing.
    #[test]
    fn a_chosen_checkpoint_lands_as_its_structure_alone() {
        let bench = Bench::new();
        let manifest = checkpoint(&bench.source, 20, 4);
        let (hub, named, heard) = bench.hub("acme/source", &manifest, 0);
        let job = Job::default();
        bench.publisher.structure(&hub, std::slice::from_ref(&named), &job).unwrap();
        let store = bench.publisher.store();
        assert_eq!(count(&heard, "GET /o/"), 2, "the manifest and the header: {heard:?}");
        assert_eq!(count(&heard, "POST /v1/tensorfs/closure"), 1);
        assert!(!crate::access::whole(store, &manifest.id(), &[]));
        let origin = tensorfs_core::origin::of(store.root(), &manifest).unwrap().unwrap();
        assert_eq!((origin.refspec.as_str(), origin.lane.as_str()), ("acme/source@1.0.0", "bf16"));
        let asked = heard.lock().unwrap().len();
        bench.publisher.structure(&hub, &[named], &job).unwrap();
        assert_eq!(heard.lock().unwrap().len(), asked, "a held structure asks nothing");
        let _ = fs::remove_dir_all(&bench.root);
    }

    /// Two calls reading one checkpoint held in part (two sessions, one model) fetch each object
    /// once; once all of it is here, a call's waits and channel ask the Hub nothing (a warm run).
    #[test]
    fn calls_sharing_a_checkpoint_fetch_each_object_once_and_a_warm_one_asks_nothing() {
        let bench = Bench::new();
        let manifest = checkpoint(&bench.source, 24, 5);
        let (hub, named, heard) = bench.hub("acme/shared", &manifest, 20);
        bench.publisher.structure(&hub, std::slice::from_ref(&named), &Job::default()).unwrap();
        let store = bench.publisher.store().clone();
        let objects = crate::access::objects(&crate::access::header(&store, &manifest).unwrap(), &[]);
        let call = |order| crate::access::Fetch::new(store.clone(), Some(hub.clone()), vec![manifest.id()], order);
        let (first, second) = (call(1), call(2));
        std::thread::scope(|scope| {
            scope.spawn(|| first.wait(&manifest, &objects).unwrap());
            scope.spawn(|| second.wait(&manifest, &objects[..12]).unwrap());
        });
        assert!(crate::access::whole(&store, &manifest.id(), &[]));
        assert_eq!(count(&heard, "GET /o/"), 2 + 24, "each object once: {heard:?}");
        let asked = heard.lock().unwrap().len();
        let warm = call(3);
        warm.wait(&manifest, &objects).unwrap();
        let mut channel = std::os::unix::net::UnixStream::from(std::os::fd::OwnedFd::from(warm.channel().unwrap()));
        let rows: Vec<Value> = objects.iter().map(|o| json!({"sha256": o.sha256, "length": o.length})).collect();
        let waited = ask(&mut channel, json!({"id": 1, "op": "wait", "manifest": manifest.id(), "objects": rows}));
        assert_eq!(waited, json!({"id": 1, "result": null}));
        assert_eq!(heard.lock().unwrap().len(), asked, "a warm call asks the Hub nothing");
        let _ = fs::remove_dir_all(&bench.root);
    }

    /// What a call's executor schedules downloads only while the call lives: its end (or its
    /// run's cancel) closes the channel, fails its waits and drops its tickets, and the fetcher
    /// stops. What landed stays.
    #[test]
    fn an_ended_call_drops_its_tickets() {
        let bench = Bench::new();
        let manifest = checkpoint(&bench.source, 40, 6);
        let (hub, named, heard) = bench.hub("acme/canceled", &manifest, 100);
        bench.publisher.structure(&hub, std::slice::from_ref(&named), &Job::default()).unwrap();
        let store = bench.publisher.store().clone();
        let objects = crate::access::objects(&crate::access::header(&store, &manifest).unwrap(), &[]);
        let fetch = crate::access::Fetch::new(store.clone(), Some(hub.clone()), vec![manifest.id()], 7);
        let mut channel = std::os::unix::net::UnixStream::from(std::os::fd::OwnedFd::from(fetch.channel().unwrap()));
        let rows: Vec<Value> = objects.iter().map(|o| json!({"sha256": o.sha256, "length": o.length})).collect();
        let scheduled = ask(&mut channel, json!({"id": 1, "op": "schedule", "manifest": manifest.id(), "objects": rows, "rank": 0}));
        assert_eq!(scheduled, json!({"id": 1, "result": null}));
        while count(&heard, "GET /o/") < 2 + 4 {
            std::thread::sleep(Duration::from_millis(5));
        }
        fetch.end();
        let waited = ask(&mut channel, json!({"id": 2, "op": "wait", "manifest": manifest.id(), "objects": rows}));
        assert_ne!(waited, json!({"id": 2, "result": null}), "a wait after the end never answers present");
        // Every request went out at once and each answer takes 100 ms: the ones on the wire stop.
        let landed = || objects.iter().filter(|o| store.contains(&o.sha256)).count();
        std::thread::sleep(Duration::from_millis(300));
        let at_end = landed();
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(landed(), at_end, "nothing lands once its tickets dropped");
        assert!(at_end < objects.len(), "{at_end} of {} landed: it stopped short", objects.len());
        assert!(store.manifest_path(&manifest.sha256).is_file(), "the structure stays");
        let _ = fs::remove_dir_all(&bench.root);
    }

    /// A warm set member's `downloaded`: its call wants what its constructions read, behind every
    /// run, and the level is reached as that ticket completes.
    #[test]
    fn warm_downloaded_comes_from_its_tickets_completion() {
        let bench = Bench::new();
        let manifest = checkpoint(&bench.source, 16, 7);
        let (hub, named, heard) = bench.hub("acme/warm", &manifest, 10);
        bench.publisher.structure(&hub, std::slice::from_ref(&named), &Job::default()).unwrap();
        let store = bench.publisher.store().clone();
        let selected = [(manifest.id(), vec!["model".to_string()])];
        assert!(!crate::access::whole(&store, &manifest.id(), &selected[0].1));
        let member = crate::access::Fetch::new(store.clone(), Some(hub.clone()), vec![manifest.id()], u64::MAX);
        member.keep(&selected).unwrap();
        let started = Instant::now();
        while !crate::access::whole(&store, &manifest.id(), &selected[0].1) {
            assert!(started.elapsed() < Duration::from_secs(60), "never downloaded: {heard:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(count(&heard, "GET /o/"), 2 + 16);
        let _ = fs::remove_dir_all(&bench.root);
    }

    /// `--warm=downloaded` never starts an executor or imports, even where calls prespawn:
    /// the member wants its constructions' objects behind every run and reports `installed`
    /// (downloading) until every one is here, then exactly `downloaded`. An on-demand model
    /// waits for its first use: none of its objects is fetched.
    #[test]
    fn a_member_warmed_to_downloaded_only_downloads() {
        use crate::gpu_service::{Level, Member, Plan, PlanSlot, Pool};
        let bench = Bench::new();
        let manifest = checkpoint(&bench.source, 16, 8);
        let (hub, named, heard) = bench.hub("acme/warmed", &manifest, 200);
        bench.publisher.structure(&hub, std::slice::from_ref(&named), &Job::default()).unwrap();
        let later = checkpoint(&bench.source, 4, 18);
        let (later_hub, later_named, later_heard) = bench.hub("acme/later", &later, 200);
        bench.publisher.structure(&later_hub, std::slice::from_ref(&later_named), &Job::default()).unwrap();
        let structure = count(&later_heard, "GET /o/");
        let store = bench.publisher.store().clone();
        let generations = crate::catalog::Catalog::new(&bench.root.join("generations")).unwrap();
        let identity = "ab".repeat(16);
        let dir = generations.root().join(&identity);
        fs::create_dir_all(dir.join("env/bin")).unwrap();
        fs::write(dir.join("env/bin/python"), "").unwrap();
        File::create(dir.join(".hold")).unwrap();
        let interface = json!({"application": "pkg:app", "entrypoints": [{"name": "render",
            "models": [{"path": "render.models.model", "class": "M"}]}]});
        fs::write(dir.join("generation.json"), json!({"identity": identity, "package": "acme/pkg", "version": "1.0.0",
            "application": "pkg:app", "python": dir.join("env/bin/python"), "interface": interface}).to_string()).unwrap();
        let held = generations.resolve(&identity).unwrap();
        let slot = |manifest: &ObjectRef, on_demand| PlanSlot { selected_encoded_bytes: 0, on_demand,
            binding: crate::device_executor::Binding { snapshot: manifest.id(), components: vec!["model".into()], ..Default::default() } };
        let plan = Plan { id: "warm".into(), construction: "warm".into(), installation: "i".into(), generation: identity.clone(),
            entrypoint: "render".into(), slots: vec![slot(&manifest, false), slot(&later, true)], sources: vec![], degree: 1 };
        assert_eq!(plan.upfront(), [(manifest.id(), vec!["model".to_string()])]);
        let config = serde_json::from_value(json!({"devices": "GPU-test-0", "authorized_device_limit_bytes": null, "prespawn": true})).unwrap();
        let pool = Pool::new(&bench.root.join("gpu"), config, store.clone()).unwrap();
        let engine = crate::execution::Engine::open(&bench.root.join("execution")).unwrap();
        let fetch = crate::access::Fetch::new(store.clone(), Some(hub), plan.manifests(), u64::MAX);
        pool.set_members("alice", vec![Member::new(held, Some(plan), Level::Downloaded, Some(fetch))]);
        pool.keep(&engine, "alice");
        let [(level, held_back)] = pool.members("alice")[..] else { panic!("one member") };
        assert_eq!((level, held_back), (Level::Installed, "its weights are downloading"));
        let started = Instant::now();
        while pool.members("alice")[0].0 < Level::Downloaded {
            assert!(pool.levels().is_empty(), "nothing imports or starts");
            assert!(started.elapsed() < Duration::from_secs(60), "never downloaded: {heard:?}");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(pool.members("alice"), [(Level::Downloaded, "")]);
        assert!(pool.levels().is_empty(), "nothing imports or starts");
        assert_eq!(count(&heard, "GET /o/"), 2 + 16, "each object once");
        assert_eq!(count(&later_heard, "GET /o/"), structure, "the on-demand model: none");
        pool.stop().unwrap();
        let _ = fs::remove_dir_all(&bench.root);
    }

    /// A call's weights come from its preparation on (Next, smallest component first) before its
    /// executor can ask, and keep coming while its first channel is open: nothing is dropped as
    /// the executor takes over. Once that channel closes, the preparation's fetches end.
    #[test]
    fn a_calls_preparation_fetches_ahead_until_its_first_channel_closes() {
        let bench = Bench::new();
        let manifest = checkpoint(&bench.source, 30, 10);
        let (hub, named, _) = bench.hub("acme/ahead", &manifest, 100);
        bench.publisher.structure(&hub, std::slice::from_ref(&named), &Job::default()).unwrap();
        let store = bench.publisher.store().clone();
        let ahead = crate::access::Fetch::new(store.clone(), Some(hub.clone()), vec![manifest.id()], 3);
        ahead.ahead(&[(manifest.id(), vec!["model".to_string()])]).unwrap();
        let call = crate::access::Fetch::new(store.clone(), Some(hub), vec![manifest.id()], 3);
        call.replaces(Some(ahead.clone()));
        let channel = call.channel().unwrap();
        let started = Instant::now();
        while !crate::access::whole(&store, &manifest.id(), &[]) {
            assert!(started.elapsed() < Duration::from_secs(60), "the preparation's fetches stopped");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(ahead.channel().is_ok(), "the preparation's fetches last while the first channel is open");
        drop(channel);
        let started = Instant::now();
        while ahead.channel().is_ok() {
            assert!(started.elapsed() < Duration::from_secs(10), "the first channel closed and they went on");
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = fs::remove_dir_all(&bench.root);
    }

    /// A machine on the GPUs its inventory names (compute capability, memory) holding `package`,
    /// whose `render` declares `slots`, and a Hub publishing `acme/model@1.0.0` in `lanes` and the
    /// LoRA `acme/lora@1.0.0`.
    struct Machine {
        root: PathBuf,
        service: Arc<Service>,
        publisher: Arc<Publisher>,
        installation: Installation,
        hub: hub::Source,
        heard: names_hub::Heard,
        /// Each published manifest's lane.
        lanes: HashMap<String, String>,
    }

    impl Machine {
        fn new(package: &str, gpus: &[(&str, u64)], slots: Value, lanes: &[&str]) -> Self {
            use crate::adapter_views::tests::{checkpoint as tensors, f32s};
            let root = std::env::temp_dir().join(format!("cm-precision-{}", uuid::Uuid::new_v4()));
            let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
            let store = Arc::new(Store::ensure_owned(&root.join("store")).unwrap());
            let publisher = Publisher::new(&root.join("published"), PackageSdk::default(), store.clone()).unwrap();
            let inventory: Vec<Value> = gpus.iter().enumerate().map(|(at, (cc, memory))| json!({"device_index": at,
                "device_name": "GPU", "device_uuid": format!("GPU-{at}"), "driver_version": "1", "memory_bytes": memory,
                "pci_bus_id": format!("{at}"), "compute_cap": cc})).collect();
            let devices: Vec<String> = (0..gpus.len()).map(|at| at.to_string()).collect();
            let config = json!({"devices": devices.join(","), "inventory": inventory, "authorized_device_limit_bytes": null, "prespawn": false});
            service.configure_pool(Pool::new(&root.join("gpu"), serde_json::from_value(config).unwrap(), store).unwrap()).unwrap();
            let identity = "cd".repeat(16);
            let dir = service.catalog.root().join(&identity);
            fs::create_dir_all(dir.join("env/bin")).unwrap();
            fs::write(dir.join("env/bin/python"), "").unwrap();
            File::create(dir.join(".hold")).unwrap();
            let interface = json!({"application": "pkg:app", "entrypoints": [{"name": "render", "models": slots}]});
            fs::write(dir.join("generation.json"), json!({"identity": identity, "package": package, "version": "1.0.0",
                "application": "pkg:app", "python": dir.join("env/bin/python"), "interface": interface}).to_string()).unwrap();
            let installation = service.engine.bind_installation(Installation { actor: "alice".into(), alias: "held".into(),
                generation: identity, package: package.into(), release: "1.0.0".into(),
                interface: serde_json::to_vec(&interface).unwrap(), hub: String::new() }).unwrap();
            let source = Store::ensure(&root.join("source")).unwrap();
            let meta = tensorfs_core::meta::Meta::open(&source).unwrap();
            let model = |repository: &str, lane: &str, manifest| names_hub::Model { repository: repository.into(),
                release: "1.0.0".into(), lane: lane.into(), manifest };
            let mut models: Vec<_> = lanes.iter().zip(30u8..).map(|(lane, tag)| {
                model("acme/model", lane, tensors(&source, &meta, "unet", &[("proj.weight", vec![4, 3], f32s(&[tag.into(); 12]))], tag))
            }).collect();
            let lanes = models.iter().map(|model| (model.manifest.id(), model.lane.clone())).collect();
            let lora = [("proj.lora_A.weight", vec![2, 3], f32s(&[0.25; 6])), ("proj.lora_B.weight", vec![4, 2], f32s(&[0.75; 8]))];
            models.push(model("acme/lora", "bf16", tensors(&source, &meta, "adapter", &lora, 0xa0)));
            let (origin, heard) = names_hub::models(&source, models);
            let hub = hub::Source::new(&origin, None, vec!["localhost".into()], None).unwrap();
            Self { root, service, publisher, installation, hub, heard, lanes }
        }

        /// `render` prepared for `acme`'s run at `bits` with `choices`.
        fn prepare(&self, bits: u32, choices: Vec<domain::ModelChoice>) -> Prepared {
            let request = Request { source: self.hub.clone(), package: self.installation.package.clone(), release: "1.0.0".into(),
                installed: Some(self.installation.clone()), owner: "acme".into(), catalog_revision: "r1".into(),
                providers: Providers::default(), entrypoint: "render".into(), choices, gpus: 0, bits, imports: false };
            self.publisher.prepare_now(&self.service, "alice", &request, None, None).unwrap()
        }

        /// A run of `render` at `bits`: each slot's lane (constructed, then raw), the run's
        /// warnings and what it asked the Hub.
        fn run(&self, bits: u32) -> (Vec<String>, Vec<String>, Vec<String>) {
            let prepared = self.prepare(bits, vec![]);
            let plan = prepared.plan.unwrap();
            let constructed = plan.slots.iter().map(|slot| self.lanes[&slot.binding.snapshot].clone());
            let lanes = constructed.chain(plan.sources.iter().map(|source| source.lane.clone())).collect();
            (lanes, prepared.warnings, std::mem::take(&mut *self.heard.lock().unwrap()))
        }
    }

    /// The lanes closure reads named (resolving, then landing what resolved).
    fn closed(asked: &[String]) -> std::collections::BTreeSet<&str> {
        asked.iter().filter_map(|heard| heard.strip_prefix("POST /v1/tensorfs/closure acme/model@1.0.0 ")).collect()
    }

    impl Drop for Machine {
        fn drop(&mut self) {
            let _ = self.service.pool().unwrap().stop();
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    const GIB: u64 = 1 << 30;

    /// A compute slot takes its row's 8-bit lane and a `min_bits=4` decode slot `nvfp4`, each
    /// one closure read; a warm run asks the Hub nothing; `--bits=16` is another resolution.
    #[test]
    fn each_slot_takes_its_rows_lane_and_a_warm_run_asks_the_hub_nothing() {
        let slots = json!([
            {"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "accept"},
            {"path": "render.models.text", "class": "T", "default": "model@1.0.0", "encoded_leaves": "accept",
                "min_bits": 4, "phase": "decode"}]);
        let hopper = Machine::new("acme/pkg", &[("9.0", 80 * GIB)], slots.clone(), &["bf16", "fp8", "int8", "nvfp4"]);
        let (lanes, warnings, asked) = hopper.run(0);
        assert_eq!((lanes, warnings), (vec!["int8".to_string(), "nvfp4".into()], vec![]));
        assert_eq!(closed(&asked), std::collections::BTreeSet::from(["int8", "nvfp4"]), "no lane past the first published one");
        let (lanes, _, asked) = hopper.run(0);
        assert_eq!((lanes, asked), (vec!["int8".to_string(), "nvfp4".into()], vec![]), "a warm run asks the Hub nothing");
        let (lanes, warnings, _) = hopper.run(16);
        assert_eq!((lanes, warnings), (vec!["bf16".to_string(), "bf16".into()], vec![]));
        let blackwell_ultra = Machine::new("acme/pkg", &[("10.3", 288 * GIB)], slots, &["bf16", "fp8", "int8", "nvfp4"]);
        assert_eq!(blackwell_ultra.run(0).0, ["fp8", "nvfp4"]);
    }

    /// Only a `min_bits=4` compute slot drops to 4 bits, and only when the function's 8-bit
    /// weights exceed the GPU's memory less its activations (a quarter before any measured).
    #[test]
    fn a_memory_short_function_drops_to_4_bits_only_where_min_bits_allows() {
        let slots = json!([
            {"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "accept", "min_bits": 4},
            {"path": "render.models.other", "class": "O", "default": "model@1.0.0", "encoded_leaves": "accept"}]);
        let lanes = ["int8", "nvfp4", "bf16"];
        let short = Machine::new("acme/pkg", &[("8.9", 64)], slots.clone(), &lanes);
        assert_eq!(short.run(0).0, ["nvfp4", "int8"]);
        assert_eq!(short.run(8).0, ["int8", "int8"], "the request's bits win");
        let ample = Machine::new("acme/pkg", &[("8.9", 24 * GIB)], slots, &lanes);
        assert_eq!(ample.run(0).0, ["int8", "int8"]);
    }

    /// A release without the wanted lane serves the next with one warning per step; a slot
    /// declining encoded leaves skips them with one warning; a warm run says the same again.
    #[test]
    fn a_fallback_and_a_declined_encoding_each_warn() {
        let slots = json!([{"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "accept"}]);
        let ada = Machine::new("acme/pkg", &[("8.9", 24 * GIB)], slots, &["int8", "bf16"]);
        let warned = ["precision 4: acme/model@1.0.0 has no nvfp4 lane for sm89; served int8".to_string()];
        let (lanes, warnings, asked) = ada.run(4);
        assert_eq!((lanes, warnings), (vec!["int8".to_string()], warned.to_vec()));
        assert_eq!(closed(&asked), std::collections::BTreeSet::from(["int8", "nvfp4"]));
        let (_, warnings, asked) = ada.run(4);
        assert_eq!((warnings, asked), (warned.to_vec(), vec![]), "a warm run warns again and asks nothing");
        let declines = json!([{"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "refuse"}]);
        let ada = Machine::new("acme/pkg", &[("8.9", 24 * GIB)], declines, &["int8", "bf16"]);
        let (lanes, warnings, _) = ada.run(0);
        assert_eq!((lanes, warnings), (vec!["bf16".to_string()], vec!["precision 8: acme/model@1.0.0 declines encoded leaves; served bf16".to_string()]));
    }

    /// A mixed local set runs what every GPU in it runs: an Ampere and an Ada card share no fp8.
    #[test]
    fn a_mixed_local_set_takes_the_lane_every_gpu_runs() {
        let slots = json!([{"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "accept",
            "sequence_parallel": {"degrees": [2]}}]);
        let mixed = Machine::new("acme/pkg", &[("8.6", 24 * GIB), ("8.9", 24 * GIB)], slots, &["fp8", "bf16"]);
        let (lanes, warnings, _) = mixed.run(0);
        assert_eq!((lanes, warnings), (vec!["bf16".to_string()],
            vec!["precision 8: acme/model@1.0.0 has no int8 lane for sm86+sm89; served bf16".to_string()]));
    }

    /// `--lora` on a slot naming no model: the slot's default, in the rule's lane, under the
    /// adapters.
    #[test]
    fn an_adapter_only_choice_takes_the_slots_default() {
        let slots = json!([{"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "accept"}]);
        let hopper = Machine::new("acme/pkg", &[("9.0", 80 * GIB)], slots, &["bf16", "int8"]);
        let adapter = domain::DownloadAdapterRef { model: "acme/lora".into(), release: "1.0.0".into(), ..Default::default() };
        let lora = domain::ModelChoice { parameter: "model".into(), adapters: vec![adapter], ..Default::default() };
        let plan = hopper.prepare(0, vec![lora]).plan.unwrap();
        let asked = std::mem::take(&mut *hopper.heard.lock().unwrap());
        assert_eq!(closed(&asked), std::collections::BTreeSet::from(["int8"]));
        assert!(asked.iter().any(|heard| heard == "POST /v1/tensorfs/closure acme/lora@1.0.0 "), "{asked:?}");
        assert!(plan.slots[0].binding.model.starts_with("local/adapters-"), "{:?}", plan.slots[0].binding);
    }

    /// Unpublished code's org-less default is the run owner's model.
    #[test]
    fn a_local_packages_default_is_its_owners() {
        let slots = json!([{"path": "render.models.model", "class": "M", "default": "model@1.0.0", "encoded_leaves": "accept"}]);
        assert_eq!(Machine::new("local/script", &[("8.9", 24 * GIB)], slots, &["int8"]).run(0).0, ["int8"]);
    }

    /// Raw weights (`weights: true`) are data: 16-bit lanes whatever their class declares and
    /// the request asks, else the release's only lane, with no precision warning.
    #[test]
    fn a_raw_weights_slot_asks_16_bits_only() {
        let source = json!({"path": "render.models.source", "class": "M", "weights": true, "default": "model@1.0.0",
            "encoded_leaves": "accept", "min_bits": 4, "phase": "decode"});
        let blackwell = Machine::new("acme/pkg", &[("10.0", 180 * GIB)], json!([source]), &["nvfp4", "int8", "fp16"]);
        assert_eq!(blackwell.run(4).0, ["fp16"]);
        let (lanes, warnings, _) = Machine::new("acme/pkg", &[("10.0", 180 * GIB)], json!([source]), &["nvfp4"]).run(0);
        assert_eq!((lanes, warnings), (vec!["nvfp4".to_string()], vec![]));
    }
}
