//! A run's local source: unpublished code, named by the digest of its manifest object.
//!
//! A remote editable project and its libraries are kept under the exact manifest's source
//! home. Each environment points at that home, so a later install cannot rewrite code or data
//! an accepted run still reads. This computer's explicit live path is read in place, never
//! copied. Both use the shared dependency cache.
use crate::{
    api::{
        install::{prepare_editable, prepare_uploaded, InstallerConfig},
        workspaces::{RootSet, UploadedFile, UploadedPackage},
    },
    journal::Installation,
    objects::{Objects, Refused},
    service::Service,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};
use tensorfs_core::{ids::ObjectRef, sha256, store::Store, transport::PullCancellation};

/// The manifest object (JSON). Members are objects the same signer wrote.
#[derive(Deserialize)]
pub struct Manifest {
    pub package: String,
    #[serde(default)]
    pub release: String,
    #[serde(default)]
    pub python_requires: String,
    #[serde(default)]
    pub python_version: String,
    /// The project's source tree as a tar archive.
    #[serde(default)]
    pub source: Option<Member>,
    /// Wheels the project vendors, by file name.
    #[serde(default)]
    pub wheels: Vec<Member>,
    /// The locked dependency requirements (`uv export` text).
    #[serde(default)]
    pub requirements: Option<Member>,
    /// Installed dependency distribution -> the package whose invocables it owns.
    #[serde(default)]
    pub callees: std::collections::BTreeMap<String, String>,
    /// A live project and the local path dependencies its lock names, file by file under
    /// their common directory; `project` is the project's own directory there.
    #[serde(default)]
    pub files: Vec<Member>,
    #[serde(default)]
    pub project: String,
    /// This computer's machine: that common directory itself, read in place and never copied.
    #[serde(default)]
    pub path: String,
    /// uv indexes the project's sources name without declaring, by name: an authored tree
    /// names its account's index (`tensorhub`), whose URL depends on who runs it.
    #[serde(default)]
    pub indexes: BTreeMap<String, String>,
    /// The local path dependencies its lock installs, as directories or files under the tree:
    /// their code is the environment's, not the project's own.
    #[serde(default)]
    pub locals: Vec<String>,
}
#[derive(Deserialize)]
pub struct Member {
    #[serde(default)]
    pub name: String,
    pub digest: String,
    pub length: u64,
    #[serde(default)]
    pub executable: bool,
}
impl Manifest {
    /// The project's directory under the tree, as a prefix of its files' names.
    fn own(&self) -> String {
        match self.project.is_empty() {
            true => String::new(),
            false => format!("{}/", self.project),
        }
    }
    /// Whether `name` is a local path dependency's file, or outside the project.
    fn beside(&self, name: &str) -> bool {
        !name.starts_with(&self.own()) || self.locals.iter().any(|dir| name == dir || name.starts_with(&format!("{dir}/")))
    }
    /// The project's own code by content, the same wherever it is installed: what keys its own
    /// memoized calls (an editable install's files say nothing of the code they point at).
    fn source_digest(&self) -> String {
        let own = self.own();
        let code: Vec<_> = self.files.iter().filter(|f| !self.beside(&f.name)).map(|f| (&f.name[own.len()..], &f.digest)).collect();
        format!("sha256:{}", sha256::hex_digest(serde_json::json!(code).to_string().as_bytes()))
    }
}
impl Member {
    /// What a tree holds for this file: its bytes and whether it is executable.
    fn held(&self) -> String {
        format!("{}{}", self.digest, if self.executable { "+x" } else { "" })
    }
}

pub struct LocalSources {
    pub objects: Arc<Objects>,
    pub installer: InstallerConfig,
    pub store: Arc<Store>,
    /// Each signer's packages: remote `source-<manifest>/{tree,env-*}`, direct paths `env-*`.
    editable: PathBuf,
    installing: Mutex<()>,
}

/// `name` as a path inside a tree: relative, with no step out of it.
fn inside(name: &str) -> Result<&Path, Refused> {
    let path = Path::new(name);
    if path.components().all(|c| matches!(c, Component::Normal(_))) {
        return Ok(path);
    }
    Err(refused("local_source_invalid", format!("{name:?} is not a path inside the package's tree")))
}

/// The files the tree under `home` holds, by name; none while it is absent or half written.
fn held_tree(home: &Path) -> BTreeMap<String, String> {
    fs::read(home.join("tree.json")).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default()
}

fn keyed(name: &str, prefix: &str, length: usize) -> bool {
    name.strip_prefix(prefix).is_some_and(|key| key.len() == length && key.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Only this layout's environments leave. Unknown entries in a package home are preserved.
fn collect_environments(home: &Path, named: &HashSet<PathBuf>) -> usize {
    let root = fs::canonicalize(home).unwrap_or_else(|_| home.to_path_buf());
    // A named but missing environment still retains its source for recovery.
    let kept = named.iter().filter(|env| env.parent() == Some(root.as_path())).count();
    for entry in fs::read_dir(home).into_iter().flatten().flatten() {
        let path = entry.path();
        if !keyed(&entry.file_name().to_string_lossy(), "env-", 16) || !path.is_dir() {
            continue;
        }
        if !named.contains(&fs::canonicalize(&path).unwrap_or_else(|_| path.clone())) {
            let _ = fs::remove_dir_all(&path);
            let _ = fs::remove_file(path.with_extension("constraints.txt"));
            let _ = fs::remove_dir_all(path.with_extension("wheels"));
        }
    }
    kept
}

fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

/// One member as the object this signer wrote.
fn written_member(objects: &Objects, actor: &str, m: &Member) -> Result<ObjectRef, Refused> {
    match objects.path(actor, &m.digest)? {
        Some((_, length)) if length == m.length => Ok(ObjectRef {
            sha256: m.digest.trim_start_matches("sha256:").to_string(),
            length,
        }),
        _ => Err(refused(
            "local_source_incomplete",
            format!("{} {} was not written to this machine", m.name, m.digest),
        )),
    }
}

/// The manifest `digest` (`sha256:<hex>`) as this signer wrote it.
fn open_manifest(
    objects: &Objects,
    actor: &str,
    digest: &str,
) -> Result<(Manifest, ObjectRef), Refused> {
    let (path, length) = objects.path(actor, digest)?.ok_or_else(|| {
        refused(
            "local_source_incomplete",
            format!("the local package manifest {digest} was not written to this machine"),
        )
    })?;
    if length > 1 << 20 {
        return Err(refused(
            "local_source_invalid",
            "a local package manifest is at most 1 MiB",
        ));
    }
    let manifest = serde_json::from_slice(&fs::read(path)?).map_err(|e| {
        refused(
            "local_source_invalid",
            format!("the local package manifest is invalid: {e}"),
        )
    })?;
    let sha256 = digest.trim_start_matches("sha256:").to_string();
    Ok((manifest, ObjectRef { sha256, length }))
}

impl LocalSources {
    pub fn new(objects: Arc<Objects>, installer: InstallerConfig, store: Arc<Store>) -> Self {
        Self {
            editable: installer.generations.with_file_name("editable"),
            objects,
            installer,
            store,
            installing: Mutex::new(()),
        }
    }

    /// Where `actor`'s live `package` is kept.
    fn home(&self, actor: &str, package: &str) -> Result<PathBuf, Refused> {
        let name = crate::catalog::normalized(package.strip_prefix("local/").unwrap_or_default());
        if name.is_empty() || name.contains('/') {
            return Err(refused("local_source_invalid", "a local package manifest names local/<name>"));
        }
        Ok(self.editable.join(&sha256::hex_digest(actor.as_bytes())[..16]).join(name))
    }

    /// A local source's objects, its manifest and every member this machine needs written:
    /// what a run takes custody of when it is accepted. Every captured member is retained:
    /// a cached generation may be collected while this run waits to install. The caller's
    /// object cache still avoids reuploading unchanged bytes. A directory read in place needs
    /// none of its files. One the signer did not write is `local_source_incomplete`, naming it.
    pub fn written(&self, actor: &str, digest: &str) -> Result<Vec<ObjectRef>, Refused> {
        let (manifest, own) = open_manifest(&self.objects, actor, digest)?;
        let mut written = vec![own];
        if !manifest.files.is_empty() || !manifest.path.is_empty() {
            self.home(actor, &manifest.package)?;
        }
        let files = manifest.files.iter().filter(|_| manifest.path.is_empty());
        let members = manifest.source.iter().chain(&manifest.wheels).chain(&manifest.requirements).chain(files);
        for m in members {
            written.push(written_member(&self.objects, actor, m)?);
        }
        Ok(written)
    }

    /// The environment a live manifest runs in: one per lock, project metadata, local path
    /// dependencies, carried wheels and this machine's interpreter, SDK pair and client.
    /// Remote environments live beside their immutable source; direct paths remain live.
    fn environment(&self, home: &Path, manifest: &Manifest) -> PathBuf {
        let own = manifest.own();
        let held = |name: &str| {
            let path = format!("{own}{name}");
            manifest.files.iter().find(|f| f.name == path).map(|f| f.digest.as_str()).unwrap_or_default()
        };
        let beside: Vec<_> = manifest.files.iter().filter(|f| manifest.beside(&f.name)).map(|f| (&f.name, f.held())).collect();
        let carried: Vec<_> = manifest.wheels.iter().map(|w| (&w.name, &w.digest)).collect();
        let key = serde_json::json!({
            "lock": held("uv.lock"), "project": held("pyproject.toml"), "beside": beside, "wheels": carried,
            "indexes": manifest.indexes,
            "sdk": self.installer.sdk, "client": self.installer.client_wheel,
            "python": self.installer.python, "python_version": manifest.python_version,
        });
        home.join(format!("env-{}", &sha256::hex_digest(key.to_string().as_bytes())[..16]))
    }

    /// Make the tree under `home` the manifest's files: changed ones are replaced from the
    /// objects the signer wrote, and ones it no longer names leave.
    fn sync(&self, actor: &str, home: &Path, files: &[Member]) -> Result<PathBuf, Refused> {
        let tree = home.join("tree");
        let held = held_tree(home);
        let next: BTreeMap<String, String> = files.iter().map(|f| (f.name.clone(), f.held())).collect();
        if next != held {
            // Half synced, the tree holds what neither manifest names: forget it until done.
            let _ = fs::remove_file(home.join("tree.json"));
        }
        for file in files {
            let target = tree.join(inside(&file.name)?);
            if held.get(&file.name) == Some(&file.held()) && target.is_file() {
                continue;
            }
            let object = written_member(&self.objects, actor, file)?;
            fs::create_dir_all(target.parent().expect("a file inside the tree"))?;
            let incoming = home.join(".incoming");
            fs::copy(self.store.object_path(&object.sha256), &incoming)?;
            fs::set_permissions(&incoming, fs::Permissions::from_mode(if file.executable { 0o755 } else { 0o644 }))?;
            fs::rename(&incoming, &target)?;
        }
        for gone in held.keys().filter(|name| !next.contains_key(*name)) {
            let path = tree.join(gone);
            let _ = fs::remove_file(&path);
            // Directories it emptied leave with it.
            let mut parent = path.parent();
            while let Some(directory) = parent.filter(|d| *d != tree && fs::remove_dir(d).is_ok()) {
                parent = directory.parent();
            }
        }
        if next != held {
            fs::create_dir_all(&tree)?;
            fs::write(home.join(".tree.json"), serde_json::to_vec(&next).map_err(std::io::Error::other)?)?;
            fs::rename(home.join(".tree.json"), home.join("tree.json"))?;
        }
        Ok(tree)
    }

    /// Install editable against this manifest's remote tree, or the explicit direct path.
    fn install_live(
        &self,
        service: &Service,
        actor: &str,
        alias: String,
        manifest: Manifest,
        digest: &str,
        stop: Option<&PullCancellation>,
    ) -> Result<Installation, Refused> {
        let home = self.home(actor, &manifest.package)?;
        let home = if manifest.path.is_empty() { home.join(format!("source-{digest}")) } else { home };
        fs::create_dir_all(&home)?;
        let root = match manifest.path.is_empty() {
            true => self.sync(actor, &home, &manifest.files)?,
            false => PathBuf::from(&manifest.path),
        };
        let project = match manifest.project.is_empty() {
            true => root.clone(),
            false => root.join(inside(&manifest.project)?),
        };
        if !root.is_absolute() || !project.join("pyproject.toml").is_file() {
            return Err(refused(
                "local_source_absent",
                format!("{} is not a project directory on this machine", project.display()),
            ));
        }
        let environment = self.environment(&home, &manifest);
        let source_digest = manifest.source_digest();
        // Wheels the CLI carries for locked dependencies this machine cannot fetch itself (a
        // Hub's on the caller's computer, a private git source): beside the environment they
        // install into, and needed only while it is built.
        let mut wheels = vec![];
        if !environment.join(".ready").is_file() {
            let carried = environment.with_extension("wheels");
            let _ = fs::remove_dir_all(&carried);
            fs::create_dir_all(&carried)?;
            for wheel in &manifest.wheels {
                if !wheel.name.ends_with(".whl") || wheel.name.contains('/') {
                    return Err(refused("local_source_invalid", format!("{:?} is not a wheel file name", wheel.name)));
                }
                let object = written_member(&self.objects, actor, wheel)?;
                fs::copy(self.store.object_path(&object.sha256), carried.join(&wheel.name))?;
                wheels.push(carried.join(&wheel.name));
            }
        }
        let root = RootSet {
            operation_id: String::new(),
            package: manifest.package.clone(),
            release: manifest.release.clone(),
            installation_id: alias.clone(),
            python_requires: manifest.python_requires,
            python_version: manifest.python_version,
            source_archive: String::new(),
            dependency_requirements: Vec::new(),
            callees: manifest.callees,
            files: Vec::new(),
        };
        for (name, url) in &manifest.indexes {
            let named = !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b));
            if !named || !(url.starts_with("https://") || url.starts_with("http://")) || url.contains(char::is_whitespace) {
                return Err(refused("local_source_invalid", format!("index {name:?} = {url:?} is not a named HTTP(S) index")));
            }
        }
        let prepared = prepare_editable(&self.installer, &root, &project, &environment, &wheels, &manifest.indexes, &source_digest, stop)
            .map_err(installer_refusal)?;
        let _ = fs::remove_dir_all(environment.with_extension("wheels"));
        let installed = service.engine.bind_installation(Installation {
            actor: actor.to_string(),
            alias,
            generation: prepared.record.identity,
            package: manifest.package,
            release: manifest.release,
            interface: prepared.interface_bytes,
            hub: String::new(),
        })?;
        service.changed_environment()?;
        Ok(installed)
    }

    /// Keep remote source homes while a generation names their environment. Only completed,
    /// recognized source homes leave; partial or unknown homes are preserved for recovery.
    pub fn collect(&self, actor: &str, package: &str) {
        let Ok(home) = self.home(actor, package) else {
            return;
        };
        let _installing = self.installing.lock().unwrap();
        // An unavailable or partial inventory says nothing about which sources are unused.
        let Ok(generations) = fs::read_dir(&self.installer.generations)
            .and_then(|entries| entries.collect::<std::io::Result<Vec<_>>>()) else { return; };
        let mut named = HashSet::new();
        for generation in generations {
            let env = generation.path().join("env");
            match fs::read_link(&env) {
                Ok(target) => { named.insert(target); }
                // Published generations own an ordinary environment directory.
                Err(error) if error.kind() == std::io::ErrorKind::InvalidInput && env.is_dir() => {}
                Err(_) => return,
            }
        }
        let kept = collect_environments(&home, &named);
        for entry in fs::read_dir(&home).into_iter().flatten().flatten() {
            let path = entry.path();
            if keyed(&entry.file_name().to_string_lossy(), "source-", 64)
                && !held_tree(&path).is_empty() && collect_environments(&path, &named) == 0
            {
                let _ = fs::remove_dir_all(&path);
            }
        }
        if kept == 0 {
            // The pre-generation layout's tree, once no old environment points at it.
            if !held_tree(&home).is_empty() {
                let _ = fs::remove_dir_all(home.join("tree"));
                let _ = fs::remove_file(home.join("tree.json"));
            }
            let _ = fs::remove_dir(&home); // Only an empty home; never erase unknown entries.
        }
    }

    /// This signer's installation of the manifest `digest` (`sha256:<hex>`).
    pub fn install(
        &self,
        service: &Service,
        actor: &str,
        digest: &str,
        stop: Option<&PullCancellation>,
    ) -> Result<Installation, Refused> {
        // Like published installations, local aliases include the selected SDK and
        // client. The source digest alone cannot identify the environment after an update.
        let key = serde_json::json!({
            "manifest": digest,
            "sdk": self.installer.sdk,
            "client": self.installer.client_wheel,
            "python": self.installer.python,
        });
        let alias = format!("local-{}", &tensorfs_core::sha256::hex_digest(key.to_string().as_bytes())[..32]);
        let _installing = self.installing.lock().unwrap();
        if let Some(held) = service.engine.installation(actor, &alias)? {
            if service.catalog.resolve(&held.generation).is_ok() {
                return Ok(held);
            }
        }
        let member = |m: &Member| written_member(&self.objects, actor, m);
        let (manifest, object) = open_manifest(&self.objects, actor, digest)?;
        if !manifest.files.is_empty() || !manifest.path.is_empty() {
            return self.install_live(service, actor, alias, manifest, &object.sha256, stop);
        }
        if !manifest.package.starts_with("local/") || (manifest.source.is_none() && manifest.wheels.is_empty()) {
            return Err(refused(
                "local_source_invalid",
                "a local package manifest names local/<name> and its source archive or root wheel",
            ));
        }
        let requirements = match &manifest.requirements {
            Some(m) => {
                let object = member(m)?;
                fs::read(self.store.object_path(&object.sha256))?
            }
            None => Vec::new(),
        };
        let (source_name, mut files) = match &manifest.source {
            Some(source) => ("source.tar".to_string(), vec![UploadedFile::held(
                "source.tar".into(), &member(source)?, self.store.clone(),
            )]),
            None => (String::new(), vec![]),
        };
        for wheel in &manifest.wheels {
            if !wheel.name.ends_with(".whl") || wheel.name.contains('/') {
                return Err(refused(
                    "local_source_invalid",
                    format!("{:?} is not a wheel file name", wheel.name),
                ));
            }
            files.push(UploadedFile::held(
                wheel.name.clone(),
                &member(wheel)?,
                self.store.clone(),
            ));
        }
        let uploaded = UploadedPackage {
            root: RootSet {
                operation_id: String::new(),
                package: manifest.package.clone(),
                release: manifest.release.clone(),
                installation_id: alias.clone(),
                python_requires: manifest.python_requires,
                python_version: manifest.python_version,
                source_archive: source_name,
                dependency_requirements: requirements,
                callees: manifest.callees,
                files: Vec::new(),
            },
            files,
        };
        let prepared = prepare_uploaded(&self.installer, &uploaded, stop).map_err(installer_refusal)?;
        let installed = service.engine.bind_installation(Installation {
            actor: actor.to_string(),
            alias,
            generation: prepared.record.identity,
            package: manifest.package,
            release: manifest.release,
            interface: prepared.interface_bytes,
            hub: String::new(),
        })?;
        service.changed_environment()?;
        Ok(installed)
    }
}


/// The trusted installer's end as a run's reason: canceled with its run, wedged (no CPU or
/// bytes moved past its measured patience), or failed.
fn installer_refusal(status: tonic::Status) -> Refused {
    let code = match status.code() {
        tonic::Code::Cancelled => crate::published::CANCELED,
        tonic::Code::Aborted => "package_installation_stalled",
        _ => "package_installation_failed",
    };
    refused(code, status.message())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::{Path, PathBuf},
        process::Command,
    };
    use tensorfs_core::sha256;

    fn write(objects: &Objects, actor: &str, bytes: &[u8]) -> Member {
        let digest = format!("sha256:{}", sha256::hex_digest(bytes));
        let mut writer = objects
            .begin(actor, &digest, bytes.len() as u64, 0)
            .unwrap();
        writer.append(bytes).unwrap();
        writer.finish().unwrap();
        Member {
            name: String::new(),
            digest,
            length: bytes.len() as u64,
            executable: false,
        }
    }

    /// The files `names` of `dir` as a live tree's members, each written as its own object.
    fn live_files(objects: &Objects, actor: &str, dir: &Path, names: &[&str]) -> Vec<serde_json::Value> {
        names
            .iter()
            .map(|name| {
                let written = write(objects, actor, &fs::read(dir.join(name)).unwrap());
                serde_json::json!({"name": name, "digest": written.digest, "length": written.length})
            })
            .collect()
    }

    /// A machine with the real installer helper and the client wheel built from this tree.
    struct Machine {
        root: PathBuf,
        service: Arc<Service>,
        store: Arc<Store>,
        objects: Arc<Objects>,
        installer: InstallerConfig,
    }

    fn machine(name: &str) -> Machine {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = std::env::temp_dir().join(format!("{name}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        run(Command::new("uv").current_dir(repo).args(["build", "--wheel", "--out-dir"]).arg(root.join("client")));
        let helper = run(Command::new("uv").current_dir(repo).args([
            "run", "--locked", "--extra", "test", "python", "-c", "import sys; print(sys.executable)",
        ]));
        let client = fs::read_dir(root.join("client"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "whl"))
            .unwrap();
        let state = root.join("state");
        let service = Service::open(&state, &root.join("generations"), 1).unwrap();
        let store = Arc::new(Store::ensure(&state.join("tensorfs")).unwrap());
        let objects = Arc::new(
            Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap(),
        );
        let installer = InstallerConfig {
            helper_python: helper.into(),
            python: "3.12".into(),
            generations: root.join("generations"),
            client_wheel: client,
            staging_root: root.join("staging"),
            sdk: vec![],
            uv: "uv".into(),
            seed_cache: None,
        };
        Machine { root, service, store, objects, installer }
    }

    /// The fixtures' author surface needs the Runtime of this cohort: a dev wheel
    /// (`COZY_TEST_RUNTIME_WHEEL`) as the machine's own until a release carries it.
    fn current_runtime(installer: &InstallerConfig) -> InstallerConfig {
        let sdk = std::env::var_os("COZY_TEST_RUNTIME_WHEEL").map(PathBuf::from).into_iter().collect();
        InstallerConfig { sdk, ..installer.clone() }
    }

    /// Its stdout, trimmed; a failure names the command and its stderr.
    fn run(command: &mut Command) -> String {
        let output = command.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{command:?}: {stderr}");
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn sdk_pair(root: &Path) -> Vec<PathBuf> {
        let sdk = root.join("sdk");
        fs::create_dir_all(&sdk).unwrap();
        SDK_PAIR
            .iter()
            .map(|(name, url)| {
                run(Command::new("curl").args(["-sfL", "-o"]).arg(sdk.join(name)).arg(url));
                sdk.join(name)
            })
            .collect()
    }

    fn archive(dir: &Path, names: &[&str]) -> Vec<u8> {
        let mut archive = tar::Builder::new(Vec::new());
        for name in names {
            archive.append_path_with_name(dir.join(name), name).unwrap();
        }
        archive.into_inner().unwrap()
    }

    fn installed_sdk(python: &Path) -> String {
        run(Command::new(python).args([
            "-c",
            "import importlib.metadata as m; print(m.version('cozy-runtime'), m.version('tensorfs'))",
        ]))
    }

    /// The real installer from objects the signer wrote; the same manifest reopens it.
    #[test]
    fn a_written_local_package_reuses_only_the_same_sdk_environment() {
        let Machine { root, service, store, objects, installer } = machine("cm-local");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let sources = LocalSources::new(objects.clone(), installer.clone(), store.clone());
        let fixture = repo.join("tests/fixtures/cpu_input");
        let tree = archive(&fixture, &["pyproject.toml", "package.toml", "cpu_input/__init__.py"]);
        let source = write(&objects, "alice", &tree);
        let manifest = serde_json::json!({
            "package": "local/tensord-cpu-input",
            "release": "0.1.0",
            "python_version": "3.12",
            "source": {"digest": source.digest, "length": source.length},
        });
        let manifest = write(&objects, "alice", manifest.to_string().as_bytes());

        // Another signer cannot install what it did not write.
        let refused = sources
            .install(&service, "bob", &manifest.digest, None)
            .err()
            .unwrap();
        assert_eq!(refused.code, "local_source_incomplete");

        let installed = sources
            .install(&service, "alice", &manifest.digest, None)
            .unwrap();
        assert_eq!(installed.package, "local/tensord-cpu-input");
        let interface: serde_json::Value = serde_json::from_slice(&installed.interface).unwrap();
        assert!(interface["entrypoints"]
            .as_array()
            .is_some_and(|e| !e.is_empty()));
        let again = sources
            .install(&service, "alice", &manifest.digest, None)
            .unwrap();
        assert_eq!(again.generation, installed.generation);

        // With the machine's own SDK pair, a local package runs that pair, not PyPI's newest.
        let pair = sdk_pair(&root);
        let pinned = LocalSources::new(
            objects.clone(),
            InstallerConfig {
                sdk: pair,
                ..installer
            },
            store,
        );
        // The source and manifest stay byte-identical across this SDK change. Reusing
        // the first alias here would silently keep its original Runtime/TensorFS.
        let updated = pinned.install(&service, "alice", &manifest.digest, None).unwrap();
        assert_ne!(updated.alias, installed.alias);
        assert_ne!(updated.generation, installed.generation);
        let same_sdk = pinned.install(&service, "alice", &manifest.digest, None).unwrap();
        assert_eq!(same_sdk.alias, updated.alias);
        assert_eq!(same_sdk.generation, updated.generation);
        // A prior generation remains valid for already accepted work.
        assert!(service.catalog.resolve(&installed.generation).is_ok());
        let held = service.catalog.resolve(&updated.generation).unwrap();
        assert_eq!(installed_sdk(&held.record.python), "0.18.101 0.3.92");
        assert_eq!(held.record.sdk_fallback, "");
        let _ = fs::remove_dir_all(root);
    }

    /// Remote edits get independent editable trees/environments. Explicit direct paths keep
    /// reading the authored directory in place.
    #[test]
    fn remote_edits_keep_prior_sources_and_direct_paths_stay_live() {
        let m = machine("cm-live");
        let sources = LocalSources::new(m.objects.clone(), m.installer.clone(), m.store.clone());
        let project = m.root.join("project");
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_input");
        fs::create_dir_all(project.join("cpu_input")).unwrap();
        for name in ["pyproject.toml", "package.toml", "cpu_input/__init__.py"] {
            fs::copy(fixture.join(name), project.join(name)).unwrap();
        }
        fs::write(project.join("notes.txt"), "dropped by the edit").unwrap();
        let manifest = |files: Vec<serde_json::Value>, extra: serde_json::Value| {
            let mut manifest = serde_json::json!({"package": "local/tensord-cpu-input", "release": "0.1.0",
                "python_version": "3.12", "files": files});
            manifest.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            write(&m.objects, "alice", manifest.to_string().as_bytes()).digest
        };
        let none = serde_json::json!({});
        let first = manifest(live_files(&m.objects, "alice", &project, &["pyproject.toml", "package.toml", "cpu_input/__init__.py", "notes.txt"]), none.clone());
        let installed = sources.install(&m.service, "alice", &first, None).unwrap();
        let home = sources.home("alice", "local/tensord-cpu-input").unwrap();
        let first_home = home.join(format!("source-{}", first.trim_start_matches("sha256:")));
        assert_eq!(fs::read_to_string(first_home.join("tree/notes.txt")).unwrap(), "dropped by the edit");
        let held = m.service.catalog.resolve(&installed.generation).unwrap();
        let environment = fs::read_link(m.installer.generations.join(&installed.generation).join("env")).unwrap();
        assert!(environment.starts_with(fs::canonicalize(&first_home).unwrap()), "{environment:?}");
        // Published generations have a directory, not an editable env symlink. They remain
        // ordinary inventory entries; a failed directory read must retain every live source.
        fs::create_dir_all(m.installer.generations.join("published/env")).unwrap();
        let unavailable = m.root.join("unavailable-generations");
        fs::rename(&m.installer.generations, &unavailable).unwrap();
        sources.collect("alice", "local/tensord-cpu-input");
        assert!(environment.join("bin/python").is_file());
        assert!(first_home.join("tree/notes.txt").is_file());
        fs::rename(&unavailable, &m.installer.generations).unwrap();
        // The environment imports the tree itself: nothing of the project was copied into it.
        let imported = run(Command::new(&held.record.python).args(["-I", "-c", "import cpu_input; print(cpu_input.__file__)"]));
        assert_eq!(Path::new(&imported), fs::canonicalize(first_home.join("tree/cpu_input/__init__.py")).unwrap());

        // Unchanged objects are cached, but acceptance retains every captured member.
        let source = fs::read_to_string(project.join("cpu_input/__init__.py")).unwrap();
        fs::write(project.join("cpu_input/__init__.py"), source + "\nEDITED = True\n").unwrap();
        let unchanged = |name: &str| {
            let bytes = fs::read(project.join(name)).unwrap();
            serde_json::json!({"name": name, "digest": format!("sha256:{}", sha256::hex_digest(&bytes)), "length": bytes.len()})
        };
        let mut files = vec![unchanged("pyproject.toml"), unchanged("package.toml")];
        files.extend(live_files(&m.objects, "alice", &project, &["cpu_input/__init__.py"]));
        let second = manifest(files, none.clone());
        assert_eq!(sources.written("alice", &second).unwrap().len(), 4, "the manifest and every captured file");
        // Another signer holds no tree: every file is asked of it.
        let theirs = sources.written("bob", &second).err().unwrap();
        assert_eq!(theirs.code, "local_source_incomplete");
        let edited = sources.install(&m.service, "alice", &second, None).unwrap();
        assert_ne!(edited.generation, installed.generation);
        // Its own memoized calls are keyed by its code: the edit is another computation.
        let code = m.service.catalog.resolve(&edited.generation).unwrap().record.source_digest;
        assert!(held.record.source_digest.starts_with("sha256:") && code.starts_with("sha256:") && code != held.record.source_digest,
            "{} then {code}", held.record.source_digest);
        assert_ne!(fs::read_link(m.installer.generations.join(&edited.generation).join("env")).unwrap(), environment);
        let second_home = home.join(format!("source-{}", second.trim_start_matches("sha256:")));
        assert!(!second_home.join("tree/notes.txt").exists());
        assert!(first_home.join("tree/notes.txt").is_file());
        assert_eq!(run(Command::new(&held.record.python).args(["-I", "-c", "import cpu_input; print(hasattr(cpu_input, 'EDITED'))"])), "False");
        let python = m.service.catalog.resolve(&edited.generation).unwrap().record.python;
        assert_eq!(run(Command::new(&python).args(["-I", "-c", "import cpu_input; print(cpu_input.EDITED)"])), "True");
        assert_eq!(fs::read_dir(&home).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("source-") && e.path().is_dir()).count(), 2);

        // This computer's machine reads the live directory in place: nothing is written for it
        // but the manifest, and its environment imports the project where it is.
        let live = manifest(vec![unchanged("pyproject.toml"), unchanged("package.toml"), unchanged("cpu_input/__init__.py")],
            serde_json::json!({"path": project.to_string_lossy()}));
        assert_eq!(sources.written("carol", &live).err().unwrap().code, "local_source_incomplete", "the manifest is hers to write");
        let live = {
            let files = vec![unchanged("pyproject.toml"), unchanged("package.toml"), unchanged("cpu_input/__init__.py")];
            let document = serde_json::json!({"package": "local/tensord-cpu-input", "release": "0.1.0", "python_version": "3.12",
                "files": files, "path": project.to_string_lossy()});
            write(&m.objects, "carol", document.to_string().as_bytes()).digest
        };
        assert_eq!(sources.written("carol", &live).unwrap().len(), 1);
        let placed = sources.install(&m.service, "carol", &live, None).unwrap();
        let python = m.service.catalog.resolve(&placed.generation).unwrap().record.python;
        let imported = run(Command::new(&python).args(["-I", "-c", "import cpu_input; print(cpu_input.__file__)"]));
        assert_eq!(Path::new(&imported), fs::canonicalize(project.join("cpu_input/__init__.py")).unwrap());
        assert!(!sources.home("carol", "local/tensord-cpu-input").unwrap().join("tree").exists());

        // An environment no generation names leaves; a package with none left leaves whole.
        fs::remove_dir_all(m.installer.generations.join(&placed.generation)).unwrap();
        sources.collect("carol", "local/tensord-cpu-input");
        assert!(!sources.home("carol", "local/tensord-cpu-input").unwrap().exists());
        sources.collect("alice", "local/tensord-cpu-input");
        assert!(environment.is_dir());
        fs::remove_dir_all(m.installer.generations.join(&installed.generation)).unwrap();
        fs::create_dir_all(home.join("unknown-owner")).unwrap();
        fs::write(home.join("unknown-owner/work"), "preserve").unwrap();
        sources.collect("alice", "local/tensord-cpu-input");
        assert!(!first_home.exists());
        assert!(second_home.is_dir());
        assert_eq!(fs::read_to_string(home.join("unknown-owner/work")).unwrap(), "preserve");
        let second_env = fs::read_link(m.installer.generations.join(&edited.generation).join("env")).unwrap();
        fs::remove_dir_all(second_env).unwrap();
        sources.collect("alice", "local/tensord-cpu-input");
        assert!(second_home.join("tree/cpu_input/__init__.py").is_file(), "a named generation retains source even if its environment is missing");
        let _ = fs::remove_dir_all(m.root);
    }

    /// Two already-running processes import a library module and read its package data only
    /// after a second remote install. The main stays unchanged; late root/library files change.
    #[test]
    fn concurrent_remote_runs_keep_late_imports_and_data() {
        use std::io::{BufRead, BufReader, Write};
        use std::process::{Child, ChildStdout, Stdio};

        fn waiting(python: &Path) -> (Child, BufReader<ChildStdout>) {
            let code = "from pathlib import Path\nimport cpu_library, cpu_input\nprint('ready', flush=True)\ninput()\nfrom cpu_library.late import VALUE\nfrom cpu_input.late import VALUE as ROOT\nprint(ROOT + '|' + Path(cpu_input.__file__).with_name('root-data.txt').read_text() + '|' + VALUE + '|' + Path(cpu_library.__file__).with_name('data.txt').read_text(), flush=True)\n";
            let mut child = Command::new(python).args(["-I", "-u", "-c", code])
                .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
            let mut reader = BufReader::new(child.stdout.take().unwrap());
            let mut ready = String::new();
            reader.read_line(&mut ready).unwrap();
            assert_eq!(ready.trim(), "ready");
            (child, reader)
        }
        fn finish(mut child: Child, mut reader: BufReader<ChildStdout>) -> String {
            child.stdin.take().unwrap().write_all(b"\n").unwrap();
            let mut result = String::new();
            reader.read_line(&mut result).unwrap();
            assert!(child.wait().unwrap().success());
            result.trim().to_string()
        }

        let m = machine("cm-live-libraries");
        let sources = LocalSources::new(m.objects.clone(), m.installer.clone(), m.store.clone());
        let project = m.root.join("project");
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::create_dir_all(project.join("cpu_input")).unwrap();
        for name in ["pyproject.toml", "package.toml", "cpu_input/__init__.py"] {
            fs::copy(fixtures.join("cpu_input").join(name), project.join(name)).unwrap();
        }
        fs::create_dir_all(project.join("deps/library/cpu_library")).unwrap();
        fs::write(project.join("deps/library/pyproject.toml"), "[build-system]\nrequires = [\"uv_build>=0.12.7,<0.13\"]\nbuild-backend = \"uv_build\"\n[project]\nname = \"late-library\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n[tool.uv.build-backend]\nmodule-name = \"cpu_library\"\nmodule-root = \"\"\n").unwrap();
        fs::write(project.join("deps/library/cpu_library/__init__.py"), "# An ordinary editable library, with no callee capture.\n").unwrap();
        let pyproject = fs::read_to_string(project.join("pyproject.toml")).unwrap();
        fs::write(project.join("pyproject.toml"), pyproject.replace("\"msgspec>=0.19,<1\"", "\"msgspec>=0.19,<1\", \"late-library\"")
            + "\n[tool.uv.sources]\nlate-library = { path = \"deps/library\", editable = true }\n").unwrap();
        let library = project.join("deps/library/cpu_library");
        fs::write(project.join("cpu_input/late.py"), "VALUE = 'old-root-code'\n").unwrap();
        fs::write(project.join("cpu_input/root-data.txt"), "old-root-data").unwrap();
        fs::write(library.join("late.py"), "VALUE = 'old-code'\n").unwrap();
        fs::write(library.join("data.txt"), "old-data").unwrap();
        run(Command::new("uv").args(["lock", "--python", "3.12", "--project"]).arg(&project));
        let files = ["pyproject.toml", "uv.lock", "package.toml", "cpu_input/__init__.py", "cpu_input/late.py", "cpu_input/root-data.txt",
            "deps/library/pyproject.toml", "deps/library/cpu_library/__init__.py",
            "deps/library/cpu_library/late.py", "deps/library/cpu_library/data.txt"];
        let manifest = || {
            let document = serde_json::json!({"package":"local/tensord-cpu-input", "release":"0.1.0", "python_version":"3.12",
                "files":live_files(&m.objects, "alice", &project, &files), "locals":["deps/library"]});
            write(&m.objects, "alice", document.to_string().as_bytes()).digest
        };
        let first = manifest();
        let a = sources.install(&m.service, "alice", &first, None).unwrap();
        let held_a = m.service.catalog.resolve(&a.generation).unwrap();
        let (child_a, reader_a) = waiting(&held_a.record.python);
        fs::write(project.join("cpu_input/late.py"), "VALUE = 'new-root-code'\n").unwrap();
        fs::write(project.join("cpu_input/root-data.txt"), "new-root-data").unwrap();
        fs::write(library.join("late.py"), "VALUE = 'new-code'\n").unwrap();
        fs::write(library.join("data.txt"), "new-data").unwrap();
        let second = manifest();
        let b = sources.install(&m.service, "alice", &second, None).unwrap();
        let held_b = m.service.catalog.resolve(&b.generation).unwrap();
        assert_ne!(held_a.record.source_digest, held_b.record.source_digest, "late authored files are another computation");
        assert_ne!(held_a.record.python, held_b.record.python);
        let (child_b, reader_b) = waiting(&held_b.record.python);
        sources.collect("alice", "local/tensord-cpu-input");
        assert_eq!(finish(child_a, reader_a), "old-root-code|old-root-data|old-code|old-data");
        assert_eq!(finish(child_b, reader_b), "new-root-code|new-root-data|new-code|new-data");

        // Cached source is not a custody substitute after Store GC. A caller must write the
        // missing object before acceptance; doing so reopens the exact existing generation.
        let old = write(&m.objects, "alice", b"VALUE = 'old-code'\n");
        fs::remove_file(m.store.object_path(old.digest.trim_start_matches("sha256:"))).unwrap();
        assert_eq!(sources.written("alice", &first).unwrap_err().code, "local_source_incomplete");
        write(&m.objects, "alice", b"VALUE = 'old-code'\n");
        assert_eq!(sources.written("alice", &first).unwrap().len(), files.len() + 1);
        assert_eq!(sources.install(&m.service, "alice", &first, None).unwrap().generation, a.generation);
        let _ = fs::remove_dir_all(m.root);
    }

    /// A locked dependency this machine cannot fetch (a Hub's on the caller's computer) rides
    /// as its wheel: the lock installs everything else, and the carried one installs from its
    /// own file at the locked version, a callee App under the package the caller names for it.
    #[test]
    fn a_live_tree_carries_the_locked_wheels_its_machine_cannot_fetch() {
        let m = machine("cm-carried");
        let sources = LocalSources::new(m.objects.clone(), current_runtime(&m.installer), m.store.clone());
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let (project, index) = (m.root.join("project"), m.root.join("index"));
        fs::create_dir_all(project.join("cpu_caller")).unwrap();
        let files = ["pyproject.toml", "package.toml", "cpu_caller/__init__.py"];
        for name in files {
            fs::copy(fixtures.join("cpu_caller").join(name), project.join(name)).unwrap();
        }
        run(Command::new("uv").current_dir(fixtures.join("cpu_memo")).args(["build", "--wheel", "--out-dir"]).arg(&index));
        run(Command::new("uv").args(["lock", "--python", "3.12", "--find-links"]).arg(&index).arg("--project").arg(&project));
        let memo = fs::read_dir(&index).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap();
        let carried = write(&m.objects, "alice", &fs::read(&memo).unwrap());
        // The index the lock names is gone: only the carried wheel has the dependency.
        fs::remove_dir_all(&index).unwrap();
        let manifest = |wheels: serde_json::Value| {
            let document = serde_json::json!({"package": "local/tensord-cpu-caller", "release": "0.1.0", "python_version": "3.12",
                "files": live_files(&m.objects, "alice", &project, &[&files[..], &["uv.lock"]].concat()),
                "wheels": wheels, "callees": {"tensord-cpu-memo": "proof/tensord-cpu-memo"}});
            write(&m.objects, "alice", document.to_string().as_bytes()).digest
        };
        let name = memo.file_name().unwrap().to_string_lossy();
        let with = manifest(serde_json::json!([{"name": name, "digest": carried.digest, "length": carried.length}]));
        let installed = sources.install(&m.service, "alice", &with, None).unwrap();
        let held = m.service.catalog.resolve(&installed.generation).unwrap();
        let callee = held.record.callees.iter().find(|c| c.distribution == "tensord-cpu-memo").expect("the carried callee App");
        assert_eq!((callee.package.as_str(), callee.version.as_str()), ("proof/tensord-cpu-memo", "0.1.0"));
        // Acceptance roots the captured files and carried wheel even when its environment
        // exists: that environment could be reclaimed before a queued run uses it.
        assert_eq!(sources.written("alice", &with).unwrap().len(), files.len() + 3);
        // Without the wheel the lock's own source is all there is, and uv says so.
        let why = sources.install(&m.service, "alice", &manifest(serde_json::json!([])), None).err().unwrap().message;
        assert!(why.contains("package_dependency_operation_failed: uv sync --frozen exited") && why.contains("tensord-cpu-memo"), "{why}");
        let _ = fs::remove_dir_all(m.root);
    }

    /// An authored tree names its account's index without declaring it (the URL depends on who
    /// runs it). Locked, it installs read in place with no index at all: its lock says where each
    /// dependency comes from. Unlocked, it resolves against the index its manifest names, and
    /// without one uv says the index is undeclared.
    #[test]
    fn an_authored_tree_installs_with_the_index_its_manifest_names() {
        struct Served(std::process::Child);
        impl Drop for Served {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let m = machine("cm-indexes");
        let sources = LocalSources::new(m.objects.clone(), current_runtime(&m.installer), m.store.clone());
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let (project, served) = (m.root.join("project"), m.root.join("served"));
        fs::create_dir_all(project.join("cpu_caller")).unwrap();
        let files = ["pyproject.toml", "package.toml", "cpu_caller/__init__.py"];
        for name in files {
            fs::copy(fixtures.join("cpu_caller").join(name), project.join(name)).unwrap();
        }
        let pyproject = fs::read_to_string(project.join("pyproject.toml")).unwrap();
        fs::write(project.join("pyproject.toml"), pyproject + "\n[tool.uv.sources]\ntensord-cpu-memo = { index = \"tensorhub\" }\n").unwrap();
        let simple = served.join("simple/tensord-cpu-memo");
        run(Command::new("uv").current_dir(fixtures.join("cpu_memo")).args(["build", "--wheel", "--out-dir"]).arg(&simple));
        let memo = fs::read_dir(&simple).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap();
        let name = memo.file_name().unwrap().to_string_lossy().to_string();
        let digest = sha256::hex_digest(&fs::read(&memo).unwrap());
        fs::write(simple.join("index.html"), format!("<a href=\"{name}#sha256={digest}\">{name}</a>")).unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let _server = Served(Command::new("python3").args(["-m", "http.server", &port.to_string(), "--bind", "127.0.0.1", "--directory"])
            .arg(&served).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn().unwrap());
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let url = format!("http://127.0.0.1:{port}/simple/");
        run(Command::new("uv").args(["lock", "--python", "3.12", "--index", &format!("tensorhub={url}"), "--project"]).arg(&project));
        let unlocked = m.root.join("unlocked");
        fs::create_dir_all(unlocked.join("cpu_caller")).unwrap();
        for name in files {
            fs::copy(project.join(name), unlocked.join(name)).unwrap();
        }
        let manifest = |dir: &Path, names: &[&str], indexes: serde_json::Value| {
            let document = serde_json::json!({"package": "local/tensord-cpu-caller", "release": "0.1.0", "python_version": "3.12",
                "files": live_files(&m.objects, "alice", dir, names), "path": dir.to_string_lossy(),
                "indexes": indexes, "callees": {"tensord-cpu-memo": "proof/tensord-cpu-memo"}});
            write(&m.objects, "alice", document.to_string().as_bytes()).digest
        };
        let callee = |installed: Installation| {
            let held = m.service.catalog.resolve(&installed.generation).unwrap();
            assert!(held.record.callees.iter().any(|c| c.distribution == "tensord-cpu-memo"), "{:?}", held.record.callees);
        };
        let locked = [&files[..], &["uv.lock"]].concat();
        callee(sources.install(&m.service, "alice", &manifest(&project, &locked, serde_json::json!({})), None).unwrap());
        let why = sources.install(&m.service, "alice", &manifest(&unlocked, &files, serde_json::json!({})), None).err().unwrap().message;
        assert!(why.contains("undeclared index"), "{why}");
        callee(sources.install(&m.service, "alice", &manifest(&unlocked, &files, serde_json::json!({"tensorhub": url})), None).unwrap());
        let bad = sources.install(&m.service, "alice", &manifest(&unlocked, &files, serde_json::json!({"tensorhub": "file:///etc"})), None).err().unwrap();
        assert_eq!(bad.code, "local_source_invalid");
        let _ = fs::remove_dir_all(m.root);
    }

    /// A path dependency's code is the package's too: an edit of it builds the environment again
    /// with that dependency built from its new code, though its metadata did not change.
    #[test]
    fn an_edited_path_dependency_is_built_again() {
        let m = machine("cm-beside");
        let sources = LocalSources::new(m.objects.clone(), current_runtime(&m.installer), m.store.clone());
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let project = m.root.join("live");
        for (fixture, to) in [("cpu_caller", ""), ("cpu_memo", "deps/cpu_memo")] {
            let from = repo.join("tests/fixtures").join(fixture);
            fs::create_dir_all(project.join(to).join(fixture)).unwrap();
            for name in ["pyproject.toml", "package.toml", &format!("{fixture}/__init__.py")] {
                fs::copy(from.join(name), project.join(to).join(name)).unwrap();
            }
        }
        let pyproject = fs::read_to_string(project.join("pyproject.toml")).unwrap();
        // A settled Runtime: the newest may be mid-upload (its sdist visible before its wheels).
        fs::write(project.join("pyproject.toml"), pyproject + "\n[tool.uv]\nconstraint-dependencies = [\"cozy-runtime==0.24.1\"]\n[tool.uv.sources]\ntensord-cpu-memo = { path = \"deps/cpu_memo\" }\n").unwrap();
        run(Command::new("uv").args(["lock", "--python", "3.12", "--project"]).arg(&project));
        let names = ["pyproject.toml", "package.toml", "uv.lock", "cpu_caller/__init__.py",
            "deps/cpu_memo/pyproject.toml", "deps/cpu_memo/package.toml", "deps/cpu_memo/cpu_memo/__init__.py"];
        let install = || {
            let document = serde_json::json!({"package": "local/tensord-cpu-caller", "release": "0.1.0", "python_version": "3.12",
                "files": live_files(&m.objects, "alice", &project, &names), "path": project.to_string_lossy(), "locals": ["deps/cpu_memo"]});
            let installed = sources.install(&m.service, "alice", &write(&m.objects, "alice", document.to_string().as_bytes()).digest, None).unwrap();
            m.service.catalog.resolve(&installed.generation).unwrap().record.python
        };
        let python = install();
        assert_eq!(run(Command::new(&python).args(["-I", "-c", "import cpu_memo; print(hasattr(cpu_memo, 'EDITED'))"])), "False");
        let memo = project.join("deps/cpu_memo/cpu_memo/__init__.py");
        let source = fs::read_to_string(&memo).unwrap();
        fs::write(&memo, source + "\nEDITED = True\n").unwrap();
        let python = install();
        assert_eq!(run(Command::new(&python).args(["-I", "-c", "import cpu_memo; print(cpu_memo.EDITED)"])), "True");
        let _ = fs::remove_dir_all(m.root);
    }

    /// With an image's seeded cache, a local package's dependencies are links into it, as a
    /// published package's are: on a pod's overlayfs a hardlink copies each file up (the A4000
    /// rental reclaimed 196 MiB of copies from one weightless edit's environment).
    #[test]
    fn a_local_package_links_its_dependencies_from_the_seeded_cache() {
        let m = machine("cm-seeded");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let seed = m.root.join("seed");
        fs::create_dir_all(&seed).unwrap();
        let sources = LocalSources::new(
            m.objects.clone(),
            InstallerConfig { seed_cache: Some(seed.clone()), ..m.installer.clone() },
            m.store.clone(),
        );
        // As `cozy run ./dir` captures it: the project's wheel and its locked requirements.
        let fixture = repo.join("tests/fixtures/cpu_input");
        run(Command::new("uv").current_dir(&fixture).args(["build", "--wheel", "--out-dir"]).arg(m.root.join("wheel")));
        let built = fs::read_dir(m.root.join("wheel")).unwrap().map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap();
        let requirements = run(Command::new("uv").args(["pip", "compile", "--quiet", "--generate-hashes", "--python-version", "3.12", "--no-header"])
            .arg(fixture.join("pyproject.toml")));
        let wheel = write(&m.objects, "alice", &fs::read(&built).unwrap());
        let locked = write(&m.objects, "alice", requirements.as_bytes());
        let name = built.file_name().unwrap().to_string_lossy();
        let manifest = serde_json::json!({"package": "local/tensord-cpu-input", "release": "0.1.0", "python_version": "3.12",
            "wheels": [{"name": name, "digest": wheel.digest, "length": wheel.length}],
            "requirements": {"digest": locked.digest, "length": locked.length}});
        let manifest = write(&m.objects, "alice", manifest.to_string().as_bytes());
        let installed = sources.install(&m.service, "alice", &manifest.digest, None).unwrap();
        let python = m.service.catalog.resolve(&installed.generation).unwrap().record.python;
        let site = python.parent().unwrap().parent().unwrap().join("lib/python3.12/site-packages");
        // A locked dependency links into the seed; the App the Runtime describes is copied.
        let dependency = site.join("msgspec/__init__.py");
        assert!(fs::symlink_metadata(&dependency).unwrap().file_type().is_symlink(), "msgspec was copied into the environment");
        assert!(fs::canonicalize(&dependency).unwrap().starts_with(fs::canonicalize(&seed).unwrap()));
        assert!(!fs::symlink_metadata(site.join("cpu_input/__init__.py")).unwrap().file_type().is_symlink());

        // A callee App among the locked dependencies is copied too: the environment describes it.
        let sources = LocalSources::new(
            m.objects.clone(),
            InstallerConfig { seed_cache: Some(seed.clone()), ..current_runtime(&m.installer) },
            m.store.clone(),
        );
        let wheel_of = |fixture: &str| {
            let out = m.root.join(fixture);
            run(Command::new("uv").current_dir(repo.join("tests/fixtures").join(fixture)).args(["build", "--wheel", "--out-dir"]).arg(&out));
            fs::read_dir(&out).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap()
        };
        let (caller, memo) = (wheel_of("cpu_caller"), wheel_of("cpu_memo"));
        // A settled Runtime: the newest may be mid-upload (its sdist visible before its wheels).
        fs::write(m.root.join("settled.txt"), "cozy-runtime==0.24.1\n").unwrap();
        let compiled = run(Command::new("uv").args(["pip", "compile", "--quiet", "--generate-hashes", "--python-version", "3.12", "--no-header", "--constraint"])
            .arg(m.root.join("settled.txt")).arg("--find-links")
            .arg(memo.parent().unwrap()).arg(repo.join("tests/fixtures/cpu_caller/pyproject.toml")));
        let requirements = compiled.replace("tensord-cpu-memo==0.1.0", &format!("tensord-cpu-memo @ file://{}", memo.display()));
        let wheel = write(&m.objects, "alice", &fs::read(&caller).unwrap());
        let locked = write(&m.objects, "alice", requirements.as_bytes());
        let name = caller.file_name().unwrap().to_string_lossy();
        let manifest = serde_json::json!({"package": "local/tensord-cpu-caller", "release": "0.1.0", "python_version": "3.12",
            "wheels": [{"name": name, "digest": wheel.digest, "length": wheel.length}],
            "requirements": {"digest": locked.digest, "length": locked.length}});
        let manifest = write(&m.objects, "alice", manifest.to_string().as_bytes());
        let installed = sources.install(&m.service, "alice", &manifest.digest, None).unwrap();
        let held = m.service.catalog.resolve(&installed.generation).unwrap();
        assert!(held.record.callees.iter().any(|c| c.distribution == "tensord-cpu-memo"), "{:?}", held.record.callees);
        let site = held.record.python.parent().unwrap().parent().unwrap().join("lib/python3.12/site-packages");
        assert!(!fs::symlink_metadata(site.join("cpu_memo/__init__.py")).unwrap().file_type().is_symlink());
        assert!(fs::symlink_metadata(site.join("msgspec/__init__.py")).unwrap().file_type().is_symlink());

        // A locked source capture (a directory with its uv.lock, as an editable install sends):
        // the same, with the callee its lock holds by path.
        let project = m.root.join("locked");
        for (fixture, to) in [("cpu_caller", ""), ("cpu_memo", "deps/cpu_memo")] {
            let (from, module) = (repo.join("tests/fixtures").join(fixture), fixture);
            fs::create_dir_all(project.join(to).join(module)).unwrap();
            for name in ["pyproject.toml", "package.toml", &format!("{module}/__init__.py")] {
                fs::copy(from.join(name), project.join(to).join(name)).unwrap();
            }
        }
        let pyproject = fs::read_to_string(project.join("pyproject.toml")).unwrap();
        fs::write(project.join("pyproject.toml"), pyproject + "\n[tool.uv]\nconstraint-dependencies = [\"cozy-runtime==0.24.1\"]\n[tool.uv.sources]\ntensord-cpu-memo = { path = \"deps/cpu_memo\" }\n").unwrap();
        run(Command::new("uv").args(["lock", "--python", "3.12", "--project"]).arg(&project));
        let tree = archive(&project, &["pyproject.toml", "uv.lock", "package.toml", "cpu_caller/__init__.py",
            "deps/cpu_memo/pyproject.toml", "deps/cpu_memo/package.toml", "deps/cpu_memo/cpu_memo/__init__.py"]);
        let source = write(&m.objects, "alice", &tree);
        let manifest = serde_json::json!({"package": "local/tensord-cpu-caller", "release": "0.1.0", "python_version": "3.12",
            "source": {"digest": source.digest, "length": source.length}});
        let manifest = write(&m.objects, "alice", manifest.to_string().as_bytes());
        let installed = sources.install(&m.service, "alice", &manifest.digest, None).unwrap();
        let held = m.service.catalog.resolve(&installed.generation).unwrap();
        assert!(held.record.callees.iter().any(|c| c.distribution == "tensord-cpu-memo"), "{:?}", held.record.callees);
        let site = held.record.python.parent().unwrap().parent().unwrap().join("lib/python3.12/site-packages");
        for copied in ["cpu_caller/__init__.py", "cpu_memo/__init__.py"] {
            assert!(!fs::symlink_metadata(site.join(copied)).unwrap().file_type().is_symlink(), "{copied} is linked");
        }
        assert!(fs::symlink_metadata(site.join("msgspec/__init__.py")).unwrap().file_type().is_symlink());
        let _ = fs::remove_dir_all(m.root);
    }

    /// A package pinning a vendored dev Runtime (how unreleased Runtime code is tested; it is
    /// never published) runs that Runtime where the machine's own pair is refused by its pin, and
    /// every run says so. Both capture shapes the CLI sends; runs 5252/5274/5276 failed here.
    #[test]
    fn a_vendored_dev_runtime_runs_where_its_pin_refuses_the_machine_pair() {
        let m = machine("cm-vendored");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let pair = sdk_pair(&m.root);
        let sources = LocalSources::new(
            m.objects.clone(),
            InstallerConfig { sdk: pair.clone(), ..m.installer.clone() },
            m.store.clone(),
        );
        let project = m.root.join("project");
        let fixture = repo.join("tests/fixtures/cpu_input");
        fs::create_dir_all(project.join("vendor")).unwrap();
        fs::create_dir_all(project.join("cpu_input")).unwrap();
        for name in ["package.toml", "cpu_input/__init__.py"] {
            fs::copy(fixture.join(name), project.join(name)).unwrap();
        }
        // The machine's own Runtime wheel, rebuilt as a developer's local version.
        let dev = "0.18.101+dev.vendored";
        run(Command::new(&m.installer.helper_python).args(["-c", REVERSION]).arg(&pair[0]).arg(project.join("vendor")).arg(dev));
        let vendored = format!("vendor/cozy_runtime-{dev}-cp312-abi3-manylinux_2_28_x86_64.whl");
        let pyproject = fs::read_to_string(fixture.join("pyproject.toml")).unwrap();
        fs::write(
            project.join("pyproject.toml"),
            pyproject.replace("cozy-runtime>=0.18.99,<0.19", &format!("cozy-runtime=={dev}"))
                + &format!("\n[tool.uv.sources]\ncozy-runtime = {{ path = \"{vendored}\" }}\n"),
        )
        .unwrap();
        run(Command::new("uv").args(["lock", "--python", "3.12", "--project"]).arg(&project));
        let manifest = |extra: serde_json::Value| {
            let mut manifest = serde_json::json!({"package": "local/tensord-cpu-input", "release": "0.1.0", "python_version": "3.12"});
            manifest.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            write(&m.objects, "alice", manifest.to_string().as_bytes()).digest
        };
        let member = |name: &str, bytes: &[u8]| {
            let written = write(&m.objects, "alice", bytes);
            serde_json::json!({"name": name, "digest": written.digest, "length": written.length})
        };
        let files = ["pyproject.toml", "uv.lock", "package.toml", "cpu_input/__init__.py"];
        // The project tree with its lock and vendor/ wheel.
        let tree = archive(&project, &[&files[..], &[vendored.as_str()]].concat());
        let source = manifest(serde_json::json!({"source": member("source.tar", &tree)}));
        // What `cozy run ./project` sends: the root wheel, the vendored wheel, the hashed rest.
        let wheels = m.root.join("wheels");
        run(Command::new("uv").args(["build", "--wheel", "--out-dir"]).arg(&wheels).arg(&project));
        let requirements = run(Command::new("uv").args([
            "export", "--frozen", "--no-dev", "--no-emit-project", "--no-emit-package", "cozy-runtime", "--project",
        ]).arg(&project));
        let root_wheel = wheels.join("tensord_cpu_input-0.1.0-py3-none-any.whl");
        let carried = manifest(serde_json::json!({
            "wheels": [
                member("tensord_cpu_input-0.1.0-py3-none-any.whl", &fs::read(root_wheel).unwrap()),
                member(vendored.trim_start_matches("vendor/"), &fs::read(project.join(&vendored)).unwrap()),
            ],
            "requirements": member("requirements.txt", requirements.as_bytes()),
        }));
        // The live tree, file by file: the project as it is, its lock and vendor/ wheel.
        let live = manifest(serde_json::json!({"files": live_files(&m.objects, "alice", &project, &[&files[..], &[vendored.as_str()]].concat())}));
        for digest in [source, carried, live] {
            let installed = sources.install(&m.service, "alice", &digest, None).unwrap();
            let held = m.service.catalog.resolve(&installed.generation).unwrap();
            let versions = installed_sdk(&held.record.python);
            assert!(versions.starts_with(&format!("{dev} ")), "{versions}");
            let why = &held.record.sdk_fallback;
            assert!(why.contains(&format!("requires `cozy-runtime=={dev}`, but `0.18.101` is installed")), "{why}");
            assert!(why.ends_with(&format!("it runs the package's own cozy-runtime=={dev}, tensorfs=={}", &versions[dev.len() + 1..])), "{why}");
        }

        // A refusal carries its reason: a typed one as itself, a failed operation in uv's words.
        let refused = |digest: &str| sources.install(&m.service, "alice", digest, None).err().unwrap().message;
        let absent = manifest(serde_json::json!({"source": member("source.tar", &archive(&project, &files))}));
        let why = refused(&absent);
        assert!(why.contains("package_dependency_operation_failed: uv sync --frozen exited") && why.contains("vendor/cozy_runtime-0.18.101"), "{why}");
        let python = manifest(serde_json::json!({"source": member("source.tar", &tree), "python_requires": ">=4"}));
        let why = refused(&python);
        assert!(why.starts_with("package_python_unavailable: configured interpreter does not satisfy"), "{why}");
        let _ = fs::remove_dir_all(&m.root);
    }

    /// Rewrites a wheel at another version: what a developer's local Runtime build is.
    const REVERSION: &str = r#"
import base64, hashlib, pathlib, sys, zipfile
source, out, version = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
name, old = source.name.split("-")[:2]
rows, target = [], out / source.name.replace(f"-{old}-", f"-{version}-", 1)
with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
    for item in src.infolist():
        path, data = item.filename.replace(f"{name}-{old}.", f"{name}-{version}.", 1), src.read(item)
        if path.endswith(".dist-info/RECORD"):
            record = path
            continue
        if path.endswith(".dist-info/METADATA"):
            data = data.replace(f"\nVersion: {old}\n".encode(), f"\nVersion: {version}\n".encode(), 1)
        info = zipfile.ZipInfo(path, item.date_time)
        info.external_attr, info.compress_type = item.external_attr, zipfile.ZIP_DEFLATED
        dst.writestr(info, data)
        rows.append(f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}\n")
    dst.writestr(record, "".join(rows) + f"{record},,\n")
"#;

    /// A Runtime/TensorFS pair other than PyPI's newest, standing in for the machine's own.
    const SDK_PAIR: [(&str, &str); 2] = [
        ("cozy_runtime-0.18.101-cp312-abi3-manylinux_2_28_x86_64.whl", "https://files.pythonhosted.org/packages/23/be/107c41ae8c978de51c315ba0c18152625454cd3767fb584683420b799d6b/cozy_runtime-0.18.101-cp312-abi3-manylinux_2_28_x86_64.whl"),
        ("tensorfs-0.3.92-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "https://files.pythonhosted.org/packages/f4/fc/8bc1e8fd0927b08257663e0b599da5e503a09adc8ee1db9cc183e78a73eb/tensorfs-0.3.92-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"),
    ];
}
