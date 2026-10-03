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
    pub entrypoint: String,
    pub choices: Vec<pb::ModelChoice>,
}

struct Job {
    progress: Mutex<Progress>,
    changed: Condvar,
}
impl Job {
    fn set(&self, progress: Progress) {
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
            moved: m, total: t, ..
        } = &mut *self.progress.lock().unwrap()
        {
            (*m, *t) = (moved, total);
        }
    }
}

type Failure = (&'static str, String);

pub struct Publisher {
    root: PathBuf,
    sdk: PackageSdk,
    store: Arc<Store>,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
}

impl Publisher {
    pub fn new(root: &Path, sdk: PackageSdk, store: Arc<Store>) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            sdk,
            store,
            jobs: Mutex::new(HashMap::new()),
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
                            stage: format!("preparing {}@{}", request.package, request.release),
                            moved: 0,
                            total: 0,
                        }),
                        changed: Condvar::new(),
                    });
                    let (this, service, actor, worker) = (
                        self.clone(),
                        service.clone(),
                        actor.to_string(),
                        job.clone(),
                    );
                    std::thread::spawn(move || {
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

    fn alias(&self, request: &Request) -> String {
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
                        .map(|m| (sha256::hex(&m.digest), m.length))
                ])
            })
            .collect();
        let key = json!({"installation":alias,"entrypoint":request.entrypoint,"choices":choices,"gpu":gpu});
        format!("hub-{}", sha256::hex_digest(key.to_string().as_bytes()))
    }

    /// The held installation and model plan, with no Hub read, or None.
    fn held(
        &self,
        service: &Service,
        actor: &str,
        request: &Request,
    ) -> io::Result<Option<Prepared>> {
        let Some(installation) = service.engine.installation(actor, &self.alias(request))? else {
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
            &gpu_name(&gpu.config().devices),
        );
        let Some(id) = service.engine.with_journal(|j| j.resolution(actor, &key))? else {
            return Ok(None);
        };
        let Some(preparation) = service.engine.preparation(actor, &id)? else {
            return Ok(None);
        };
        let plan = gpu.plan(&preparation)?;
        // The bytes may have been reclaimed since; then the model is fetched again.
        if gpu
            .source_facts(&[crate::model_sources::SelectedManifest {
                manifest: plan.binding.snapshot.clone(),
                components: plan.binding.components.clone(),
            }])
            .is_err()
        {
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
        let installation = match service
            .engine
            .installation(actor, &self.alias(request))
            .map_err(io_failure)?
        {
            Some(held) if service.catalog.resolve(&held.generation).is_ok() => held,
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
        self.generation(
            service.catalog.root(),
            &identity,
            &split,
            &python,
            &interface,
            &request.release,
        )?;
        let held = service.catalog.resolve(&identity).map_err(io_failure)?;
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
        split: &Lock,
        python: &str,
        interface: &Value,
        release: &str,
    ) -> Result<(), Failure> {
        let locks = generations.join(".locks");
        fs::create_dir_all(&locks).map_err(io_failure)?;
        let lock = File::create(locks.join(format!("{identity}.lock"))).map_err(io_failure)?;
        lock.lock_exclusive().map_err(io_failure)?;
        let dir = generations.join(identity);
        if dir.join("generation.json").is_file() {
            return Ok(());
        }
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
        self.uv(&[
            "venv",
            "--no-project",
            "--no-config",
            "--python",
            python,
            &env.to_string_lossy(),
        ])?;
        let requirements = dir.join("requirements.txt").to_string_lossy().to_string();
        self.uv(&[
            "pip",
            "install",
            "--no-config",
            "--python",
            &py,
            "--require-hashes",
            "--no-deps",
            "--requirements",
            &requirements,
        ])?;
        if !self.sdk.requirements.is_empty() {
            let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
            let mut args = vec![
                "pip",
                "install",
                "--no-config",
                "--python",
                &py,
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
        let record = json!({"identity":identity,"package":split.distribution,"version":release,"application":application,"python":interpreter,"dependencies":[],"interface":interface,"sdk":sdk_choice});
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

    fn uv(&self, args: &[&str]) -> Result<(), Failure> {
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
        let output = command.output().map_err(|e| {
            (
                "package_installation_uv_absent",
                format!("cannot run uv: {e}"),
            )
        })?;
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
        let slot = model_slots(&interface, &request.entrypoint)
            .and_then(|slots| slots.first().cloned())
            .ok_or((
                "package_prepare_interface_invalid",
                "declared model slot absent".to_string(),
            ))?;
        let path = slot
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let gpu_model = gpu_name(&gpu.config().devices);
        job.stage(format!("resolving the model for {path}"));
        let choice = request.choices.first().cloned().unwrap_or_default();
        let (org, name) = request.package.split_once('/').unwrap_or_default();
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
            let bindings = catalog
                .json(&format!(
                    "/v1/packages/{}/{}/bindings",
                    hub::escape(org),
                    hub::escape(name)
                ))
                .map_err(|e| ("catalog_read_failed", e.0))?;
            let row = bindings
                .get("bindings")
                .and_then(Value::as_array)
                .and_then(|rows| rows.iter().find(|row| row.get("slot").and_then(Value::as_str) == Some(path.as_str())))
                .cloned()
                .or_else(|| slot.get("default_ladder").map(|ladder| json!({"model":slot.get("default_model"),"release":slot.get("default_release"),"ladder":ladder})))
                .ok_or(("model_binding_absent", format!("{} binds no model to {path}; bind one with `cozy package bind`", request.package)))?;
            let mut model = row
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !model.contains('/') && !model.is_empty() {
                model = format!("{org}/{model}");
            }
            let lane = rung(row.get("ladder"), &gpu_model).ok_or((
                "model_binding_absent",
                format!("no rung of {path}'s ladder fits {gpu_model:?}"),
            ))?;
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
        let grant = ModelGrant {
            package: installation.package.clone(),
            slot: path.clone(),
            repository: field("model", &model),
            release: field("release", &release),
            lane: field("lane", &lane),
            manifest: manifest_id.clone(),
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
        };
        job.stage(format!(
            "downloading {}@{} {}",
            grant.repository, grant.release, grant.lane
        ));
        let credential = catalog.credential();
        let refspec = format!("{}@{manifest_id}", grant.repository);
        let keep = [manifest_id.clone()];
        let on_event =
            |event: &tensorfs_core::ensure::Event| job.bytes(event.bytes_done, event.bytes_total);
        let mut ensure = tensorfs_core::ensure::Request::new(
            &self.store,
            catalog.origin(),
            &refspec,
            &credential,
            catalog.policy(),
        );
        ensure.keep = &keep;
        ensure.on_event = Some(&on_event);
        tensorfs_core::ensure::ensure(&ensure)
            .map_err(|e| ("model_download_failed", e.to_string()))?;
        job.stage(format!("preparing {}", grant.repository));
        let plan = gpu
            .prepare_root(
                actor,
                installation,
                &request.entrypoint,
                &request.choices,
                Some(&grant),
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
        let key = Self::resolution_key(&installation.alias, request, &gpu_model);
        service
            .engine
            .with_journal(|j| j.bind_resolution(actor, &key, &installation.package, &plan.id))
            .map_err(io_failure)?;
        Ok(plan)
    }
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

fn declares_models(installation: &Installation, entrypoint: &str) -> bool {
    serde_json::from_slice(&installation.interface)
        .ok()
        .and_then(|i: Value| model_slots(&i, entrypoint))
        .is_some_and(|m| !m.is_empty())
}

/// The widest one-GPU rung whose GPU pattern fits this device (the first among equals); its lane.
fn rung(ladder: Option<&Value>, gpu: &str) -> Option<String> {
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
    ladder?
        .as_array()?
        .iter()
        .rev()
        .filter(|r| r.get("gpus").and_then(Value::as_u64).unwrap_or(0) <= 1)
        .filter(|r| fits(r.get("gpu").and_then(Value::as_str).unwrap_or("*")))
        .max_by_key(|r| r.get("gpus").and_then(Value::as_u64).unwrap_or(0))
        .and_then(|r| r.get("lane").and_then(Value::as_str))
        .map(String::from)
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
    fn rung_is_the_widest_one_gpu_fit() {
        let ladder = json!([{"gpu":"*","lane":"bf16"},{"gpu":"rtx 4090","lane":"fp8"},{"gpu":"h100","gpus":4,"lane":"bf16"}]);
        assert_eq!(
            rung(Some(&ladder), "NVIDIA GeForce RTX 4090").as_deref(),
            Some("bf16")
        );
        let ladder = json!([{"gpu":"rtx 4090","lane":"fp8"}]);
        assert_eq!(
            rung(Some(&ladder), "NVIDIA GeForce RTX 4090").as_deref(),
            Some("fp8")
        );
        assert_eq!(rung(Some(&ladder), "NVIDIA A40"), None);
    }
}
