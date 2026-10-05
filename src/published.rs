//! Published packages and their models, prepared from a Hub under the owner's access and
//! reused: a run whose installation and model resolution are held reads nothing from the
//! Hub. Port of the Python worker's release-root preparation (package set from the
//! release's locked requirements with uv; model by owner binding ladder, then a TensorFS
//! download) with its `release_root_preparing` progress answers.
use crate::{
    api::pb,
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
use tensorfs_core::{sha256, store::Store};

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
    /// Unpublished code this machine already holds, in place of the release.
    pub installed: Option<Installation>,
    /// The account an unpublished package's org-relative model names belong to.
    pub owner: String,
    /// The owner's binding revision the caller knows: a held resolution made under another
    /// revision resolves again.
    pub binding_revision: String,
    /// Provider tokens for source models (memory only).
    pub providers: Providers,
    pub entrypoint: String,
    pub choices: Vec<pb::ModelChoice>,
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
}

pub struct Publisher {
    root: PathBuf,
    sdk: PackageSdk,
    store: Arc<Store>,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    /// Manifests preparations are fetching, counted per preparation (`Fetching`).
    fetching: Mutex<HashMap<String, usize>>,
}

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
        choice: &pb::ModelChoice,
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
        let catalog = Catalog::new(source).map_err(|e| Refused {
            code: "catalog_read_failed",
            message: e.0,
        })?;
        let _fetching = self.fetch(vec![manifest.to_string()]);
        let keep = self.protected(service).map_err(io_failure).map_err(|(code, message)| Refused { code, message })?;
        ensure(&self.store, &catalog, repository, manifest, &keep, bytes)
            .map_err(|(code, message)| Refused { code, message })
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
        let key = json!({"installation":alias,"package":package,"release":release,"hub":origin,"owner":request.owner,"bindings":request.binding_revision,"entrypoint":request.entrypoint,"choices":choices,"gpu":gpu});
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
        let catalog = Catalog::new(&request.source).map_err(|e| ("catalog_read_failed", e.0))?;
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
            _ => self.install(service, actor, request, &catalog, job)?,
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
        let plan = self.model(service, &gpu, actor, &installation, request, &catalog, job)?;
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
        catalog: &Catalog,
        job: &Job,
    ) -> Result<Installation, Failure> {
        job.stage(format!(
            "installing {}@{}",
            request.package, request.release
        ));
        let (org, name) = request.package.split_once('/').ok_or((
            "release_root_invalid",
            "package must be org/name".to_string(),
        ))?;
        let base = format!(
            "/v1/packages/{}/{}/releases/{}",
            hub::escape(org),
            hub::escape(name),
            hub::escape(&request.release)
        );
        let release = catalog
            .json(&base)
            .map_err(|e| ("catalog_read_failed", e.0))?;
        if release.pointer("/release/release").and_then(Value::as_str)
            != Some(request.release.as_str())
        {
            return Err((
                "catalog_read_failed",
                format!("{base} named another release"),
            ));
        }
        let interface = release
            .get("package_interface")
            .filter(|v| v.is_object())
            .cloned()
            .ok_or((
                "package_prepare_interface_missing",
                "the release carries no package interface".to_string(),
            ))?;
        let python = release
            .get("python_version")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .unwrap_or(&self.sdk.python)
            .to_string();
        let lock = catalog
            .bytes(&format!("{base}/locked-requirements"), 16 << 20)
            .map_err(|e| ("catalog_read_failed", e.0))?;
        let lock = String::from_utf8(lock).map_err(|_| {
            (
                "package_prepare_locked_requirements_invalid",
                "locked requirements are not UTF-8".to_string(),
            )
        })?;
        let split = split_lock(
            &lock,
            name,
            &request.release,
            !self.sdk.requirements.is_empty(),
        )?;
        let identity = sha256::hex_digest(json!({"lock":lock,"python":python,"sdk":self.sdk.requirements,"links":self.sdk.find_links,"client":self.sdk.client_wheel}).to_string().as_bytes())[..32].to_string();
        let wanted = Environment {
            split: &split,
            python: &python,
            interface: &interface,
            release: &request.release,
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
        write("requirements.txt", &split.exact)?;
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
        if sdk {
            job.stage("installing the machine's Runtime and TensorFS".into());
            let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
            let mut args = vec![
                "pip",
                "install",
                "--no-config",
                "--python",
                &py,
                compile(!client),
                "--constraints",
                &constraints,
            ];
            let links = self
                .sdk
                .find_links
                .as_ref()
                .map(|p| p.to_string_lossy().to_string());
            if let Some(links) = &links {
                args.extend(["--find-links", links]);
            }
            args.extend(self.sdk.requirements.iter().map(String::as_str));
            // As the Go stack chooses: this machine's own pair where the package's bounds admit
            // it (uv's check of every installed requirement), else the release's locked SDK.
            // A package's bounds are never overridden.
            let own = self
                .uv(&args)
                .and_then(|()| self.uv(&["pip", "check", "--no-config", "--python", &py]));
            if let Err((code, detail)) = own {
                if !split.sdk.lines().any(|l| !l.starts_with("--")) {
                    return Err((code, detail));
                }
                write("sdk-requirements.txt", &split.sdk)?;
                let rows = dir
                    .join("sdk-requirements.txt")
                    .to_string_lossy()
                    .to_string();
                self.uv(&[
                    "pip",
                    "install",
                    "--no-config",
                    "--python",
                    &py,
                    compile(!client),
                    "--require-hashes",
                    "--no-deps",
                    "--requirements",
                    &rows,
                ])?;
                sdk_choice = "locked";
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
            Some(_) => describe_environment(&py, &split.distribution)?,
            None => (String::new(), vec![]),
        };
        let record = json!({"identity":identity,"package":split.distribution,"version":release,"application":application,"python":interpreter,"dependencies":[],"interface":interface,"sdk":sdk_choice,"source_digest":source_digest,"callees":callees});
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

    #[allow(clippy::too_many_arguments)]
    fn model(
        &self,
        service: &Service,
        gpu: &GpuPool,
        actor: &str,
        installation: &Installation,
        request: &Request,
        catalog: &Catalog,
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
        let (org, _) = installation.package.split_once('/').unwrap_or_default();
        // Unpublished code (local/) has no owner bindings; its org-relative names are its owner's.
        let local = org == "local";
        let account = if local { request.owner.as_str() } else { org };
        let mut bindings: Option<Value> = None;
        let mut grants = vec![];
        // The widest fitting rung's GPU count is the group's width (Runtime
        // `machine_model_defaults.select`); 0 lets every slot's declared degrees decide.
        let mut degree = 0u32;
        for slot in &slots {
            let path = slot
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let parameter = path.rsplit('.').next().unwrap_or_default().to_string();
            job.stage(format!("resolving the model for {path}"));
            let choice = request
                .choices
                .iter()
                .find(|c| c.parameter == path || c.parameter == parameter)
                .cloned()
                .unwrap_or_default();
            if !choice.source.is_empty() {
                grants.push(self.source_grant(installation, &path, &choice, &request.providers, job)?);
                continue;
            }
            let (model, release, lane, manifest) = if let Some(reference) =
                choice.manifest.as_ref().filter(|m| m.digest.len() == 32)
            {
                (
                    choice.repository.clone(),
                    format!("sha256:{}", sha256::hex(&reference.digest)),
                    choice.lane.clone(),
                    true,
                )
            } else if !choice.repository.is_empty() {
                (
                    choice.repository.clone(),
                    choice.release.clone(),
                    choice.lane.clone(),
                    false,
                )
            } else {
                let row = model_binding(catalog, installation, slot, &mut bindings)?;
                let mut model = row
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !model.contains('/') && !model.is_empty() {
                    if account.is_empty() {
                        return Err((
                            "model_binding_absent",
                            format!("{path} names its owner's model and this run names no owner"),
                        ));
                    }
                    model = format!("{account}/{model}");
                }
                let (lane, gpus) = rung(row.get("ladder"), &gpu_model, width).ok_or((
                    "model_binding_absent",
                    format!("no rung of {path}'s ladder fits {width}x {gpu_model:?}"),
                ))?;
                degree = degree.max(gpus);
                (
                    model,
                    row.get("release")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    lane,
                    false,
                )
            };
            if model.is_empty() {
                return Err((
                    "model_binding_absent",
                    format!("{path} names no model repository"),
                ));
            }
            let reference = if release.is_empty() {
                model.clone()
            } else {
                format!("{model}@{release}")
            };
            let mut query = format!("/v1/models/resolve?ref={}", hub::escape(&reference));
            if !lane.is_empty() && !manifest {
                query.push_str(&format!("&lane={}", hub::escape(&lane)));
            }
            let resolved = catalog
                .json(&query)
                .map_err(|e| ("catalog_read_failed", e.0))?;
            let manifest_id = resolved
                .get("manifest_id")
                .and_then(Value::as_str)
                .ok_or((
                    "catalog_read_failed",
                    "model resolution named no manifest".to_string(),
                ))?
                .to_string();
            let manifest_id = if manifest_id.starts_with("sha256:") {
                manifest_id
            } else {
                format!("sha256:{manifest_id}")
            };
            let field = |name: &str, fallback: &str| {
                resolved
                    .get(name)
                    .and_then(Value::as_str)
                    .unwrap_or(fallback)
                    .to_string()
            };
            grants.push(ModelGrant {
                package: installation.package.clone(),
                slot: path.clone(),
                repository: field("model", &model),
                release: field("release", &release),
                lane: field("lane", &lane),
                manifest: manifest_id,
                components: resolved
                    .get("components")
                    .and_then(Value::as_array)
                    .map(|c| {
                        c.iter()
                            .filter_map(Value::as_str)
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default(),
            });
        }
        let _fetching = self.fetch(grants.iter().map(|g| g.manifest.clone()).collect());
        let keep = self.protected(service).map_err(io_failure)?;
        let mut fetched = std::collections::BTreeSet::new();
        for grant in &grants {
            if !fetched.insert(grant.manifest.clone()) || grant.repository.starts_with("local/") {
                continue; // one download per checkpoint; a source model is already here
            }
            job.stage(format!(
                "downloading {}@{} {}",
                grant.repository, grant.release, grant.lane
            ));
            ensure(&self.store, catalog, &grant.repository, &grant.manifest, &keep, &|d, t| job.bytes(d, t))?;
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
                apply_adapters(&self.store, catalog, choice, grant, &keep, &request.providers, job)?;
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

/// One package's own slot default: its Hub bindings, then its authored ladder. A dependency
/// installed in another package's environment still asks under its own package identity.
pub(crate) fn model_binding(
    catalog: &Catalog,
    installation: &Installation,
    slot: &Value,
    cached: &mut Option<Value>,
) -> Result<Value, Failure> {
    let (org, name) = installation.package.split_once('/').ok_or((
        "model_binding_absent", "the model's package identity must be org/name".into(),
    ))?;
    let path = slot["path"].as_str().unwrap_or_default();
    if cached.is_none() && org != "local" {
        *cached = Some(catalog.json(&format!(
            "/v1/packages/{}/{}/bindings", hub::escape(org), hub::escape(name),
        )).map_err(|e| ("catalog_read_failed", e.0))?);
    }
    cached.as_ref()
        .and_then(|b| b["bindings"].as_array())
        .and_then(|rows| rows.iter().find(|row| row["slot"].as_str() == Some(path)))
        .cloned()
        .or_else(|| authored(slot))
        .ok_or(("model_binding_absent", format!(
            "{} binds no model to {path}; bind one with `cozy package bind`", installation.package,
        )))
}

/// A provider source made into a local model by TensorFS `source_model`, kept as the local
/// repository named by the source and its profiles (a later choice of it is held).
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

/// Download one exact checkpoint; `keep` names what its GC must not evict (`protected`).
fn ensure(
    store: &Store,
    catalog: &Catalog,
    repository: &str,
    manifest: &str,
    keep: &[String],
    bytes: &(dyn Fn(u64, u64) + Sync),
) -> Result<(), Failure> {
    let credential = catalog.credential();
    let refspec = format!("{repository}@{manifest}");
    let mut keep = keep.to_vec();
    keep.push(manifest.to_string());
    let on_event = |event: &tensorfs_core::ensure::Event| bytes(event.bytes_done, event.bytes_total);
    let mut request = tensorfs_core::ensure::Request::new(
        store,
        catalog.origin(),
        &refspec,
        &credential,
        catalog.policy(),
    );
    request.keep = &keep;
    request.on_event = Some(&on_event);
    tensorfs_core::ensure::ensure(&request)
        .map(drop)
        .map_err(|e| refused("model_download_failed", e))
}

/// A TensorFS refusal as a run's reason; a download the disk cannot fit is its own.
fn refused(code: &'static str, refusal: tensorfs_core::err::Refusal) -> Failure {
    match refusal.code {
        tensorfs_core::err::Code::CAPACITY_EXHAUSTED => ("machine_disk_full", refusal.to_string()),
        _ => (code, refusal.to_string()),
    }
}

/// One caller adapter's exact checkpoint at the Hub (the worker's `checkpoint()` for an
/// adapter: no ladder; a named lane, else the release's bf16/fp16/fp32 lane, else its
/// smallest), returned as (repository, "sha256:<hex>").
fn resolve_adapter(
    catalog: &Catalog,
    adapter: &pb::DownloadAdapterRef,
) -> Result<(String, String), Failure> {
    let model = adapter.model.clone();
    let Some((org, name)) = model.split_once('/') else {
        return Err((
            "model_override_invalid",
            format!("adapter {model:?} names no org/model repository"),
        ));
    };
    let mut lane = adapter.lane.clone();
    let reference = if !adapter.manifest.is_empty() {
        format!("{model}@{}", adapter.manifest)
    } else {
        if lane.is_empty() {
            let card = catalog
                .json(&format!(
                    "/v1/models/{}/{}",
                    hub::escape(org),
                    hub::escape(name)
                ))
                .map_err(|e| ("catalog_read_failed", e.0))?;
            let releases = card
                .get("releases")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let release = releases
                .iter()
                .rev()
                .find(|r| {
                    r.get("yanked").and_then(Value::as_bool) != Some(true)
                        && (adapter.release.is_empty()
                            || r.get("release").and_then(Value::as_str)
                                == Some(adapter.release.as_str()))
                })
                .ok_or((
                    "model_override_invalid",
                    format!("{model} has no such release"),
                ))?;
            let lanes: Vec<(String, u64)> = release
                .get("lanes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|l| {
                    Some((
                        l.get("lane")?.as_str()?.to_string(),
                        l.get("bytes").and_then(Value::as_u64).unwrap_or(u64::MAX),
                    ))
                })
                .collect();
            lane = ["bf16", "fp16", "fp32"]
                .into_iter()
                .find(|full| lanes.iter().any(|(l, _)| l == full))
                .map(String::from)
                .or_else(|| {
                    lanes
                        .iter()
                        .min_by_key(|(l, bytes)| (*bytes, l.clone()))
                        .map(|(l, _)| l.clone())
                })
                .ok_or((
                    "model_override_invalid",
                    format!("{model} has no lane to serve as an adapter"),
                ))?;
        }
        if adapter.release.is_empty() {
            model.clone()
        } else {
            format!("{model}@{}", adapter.release)
        }
    };
    let mut query = format!("/v1/models/resolve?ref={}", hub::escape(&reference));
    if adapter.manifest.is_empty() && !lane.is_empty() {
        query.push_str(&format!("&lane={}", hub::escape(&lane)));
    }
    let resolved = catalog
        .json(&query)
        .map_err(|e| ("catalog_read_failed", e.0))?;
    let manifest = resolved.get("manifest_id").and_then(Value::as_str).ok_or((
        "catalog_read_failed",
        "adapter resolution named no manifest".to_string(),
    ))?;
    let manifest = if manifest.starts_with("sha256:") {
        manifest.to_string()
    } else {
        format!("sha256:{manifest}")
    };
    Ok((
        resolved
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&model)
            .to_string(),
        manifest,
    ))
}

/// A slot's caller adapters applied to its resolved base: each adapter downloaded, then one
/// adapter view composed (`adapter_views`); the grant then names the view.
fn apply_adapters(
    store: &Store,
    catalog: &Catalog,
    choice: &pb::ModelChoice,
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
            let (repository, manifest) = resolve_adapter(catalog, adapter)?;
            ensure(store, catalog, &repository, &manifest, &keep, &|d, t| job.bytes(d, t))?;
            manifest
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

fn io_failure(error: io::Error) -> Failure {
    ("release_root_preparation_failed", error.to_string())
}

/// A slot's authored default ladder (`[org/]model@release/lane` rungs) in an owner binding's
/// shape: one model and release, each rung naming its lane.
fn authored(slot: &Value) -> Option<Value> {
    let rungs = slot.get("default_ladder")?.as_array()?;
    let parse = |rung: &Value| {
        let (model, rest) = rung.get("lane")?.as_str()?.split_once('@')?;
        let (release, lane) = rest.split_once('/')?;
        Some((model.to_string(), release.to_string(), lane.to_string()))
    };
    let (model, release, _) = parse(rungs.first()?)?;
    let ladder: Vec<Value> = rungs
        .iter()
        .filter_map(|rung| {
            let mut rung = rung.clone();
            rung["lane"] = json!(parse(&rung)?.2);
            Some(rung)
        })
        .collect();
    Some(json!({"model": model, "release": release, "ladder": ladder}))
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
fn rung(ladder: Option<&Value>, gpu: &str, available: usize) -> Option<(String, u32)> {
    let fits = |pattern: &str| {
        if pattern == "*" {
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
    let gpus = |r: &Value| r.get("gpus").and_then(Value::as_u64).unwrap_or(0);
    ladder?
        .as_array()?
        .iter()
        .rev()
        .filter(|r| gpus(r) as usize <= available)
        .filter(|r| fits(r.get("gpu").and_then(Value::as_str).unwrap_or("*")))
        .max_by_key(|r| gpus(r))
        .and_then(|r| Some((r.get("lane")?.as_str()?.to_string(), gpus(r) as u32)))
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

fn normalized(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars().map(|c| c.to_ascii_lowercase()) {
        let c = if matches!(c, '_' | '.') { '-' } else { c };
        if !(c == '-' && out.ends_with('-')) {
            out.push(c);
        }
    }
    out
}

/// The environment's root digest and the other Apps it holds, described inside it by the
/// machine's client without importing them (`runtime_describe`). A failed description fails
/// this preparation; it never publishes an environment with silently missing callees.
pub fn describe_environment(python: &str, root: &str) -> Result<(String, Vec<crate::catalog::Callee>), Failure> {
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
            let request = json!({"kind": "describe_environment", "root": root}).to_string();
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

/// The release's lock as uv input: index lines and every row, minus the SDK rows when the
/// machine supplies its own SDK, whose resolution the other pins then constrain.
fn split_lock(lock: &str, name: &str, release: &str, own_sdk: bool) -> Result<Lock, Failure> {
    let (mut exact, mut constraints, mut distribution, mut sdk) =
        (String::new(), String::new(), None, String::new());
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

    #[test]
    fn rung_is_the_widest_group_the_machine_holds() {
        let ladder = json!([{"gpu":"*","lane":"bf16"},{"gpu":"rtx 4090","lane":"fp8"},{"gpu":"h100","gpus":4,"lane":"bf16"}]);
        assert_eq!(
            rung(Some(&ladder), "NVIDIA GeForce RTX 4090", 1),
            Some(("bf16".into(), 0))
        );
        let ladder = json!([{"gpu":"rtx 4090","lane":"fp8"}]);
        assert_eq!(
            rung(Some(&ladder), "NVIDIA GeForce RTX 4090", 1),
            Some(("fp8".into(), 0))
        );
        assert_eq!(rung(Some(&ladder), "NVIDIA A40", 2), None);
        let h3 = json!([{"gpu":"H100","gpus":2,"lane":"fp8-pruned"},{"gpu":"H100","gpus":4,"lane":"fp8-pruned"},{"gpu":"H200","gpus":1,"lane":"fp8-pruned"}]);
        assert_eq!(rung(Some(&h3), "NVIDIA H100 80GB HBM3", 1), None);
        assert_eq!(
            rung(Some(&h3), "NVIDIA H100 80GB HBM3", 2),
            Some(("fp8-pruned".into(), 2))
        );
        assert_eq!(
            rung(Some(&h3), "NVIDIA H100 80GB HBM3", 8),
            Some(("fp8-pruned".into(), 4))
        );
    }

    #[test]
    fn an_authored_default_reads_as_a_binding_of_its_one_model() {
        let slot = json!({"default_ladder":[{"gpu":"H100","gpus":2,"lane":"h3@1.2.0/fp8"},{"gpu":"*","lane":"h3@1.2.0/bf16"}]});
        let row = authored(&slot).unwrap();
        assert_eq!(row["model"], "h3");
        assert_eq!(row["release"], "1.2.0");
        assert_eq!(
            rung(row.get("ladder"), "NVIDIA H100 80GB HBM3", 2),
            Some(("fp8".into(), 2))
        );
        assert_eq!(
            rung(row.get("ladder"), "NVIDIA A40", 1),
            Some(("bf16".into(), 0))
        );
        let slot = json!({"default_ladder":[{"gpu":"*","lane":"cozy/sdxl@1/plain"}]});
        assert_eq!(authored(&slot).unwrap()["model"], "cozy/sdxl");
        assert_eq!(authored(&json!({})), None);
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
        let choice = pb::ModelChoice {
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
