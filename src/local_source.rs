//! A run's local source: unpublished code the CLI uploaded with Write, named by the digest of
//! its manifest object. It installs once per manifest, signer and selected SDK; a later run
//! reopens that installation only while its environment inputs remain the same.
use crate::{
    api::{
        install::{prepare_uploaded, InstallerConfig},
        workspaces::{RootSet, UploadedFile, UploadedPackage},
    },
    journal::Installation,
    objects::{Objects, Refused},
    service::Service,
};
use serde::Deserialize;
use std::{fs, sync::Arc, sync::Mutex};
use tensorfs_core::{ids::ObjectRef, store::Store};

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
}
#[derive(Deserialize)]
pub struct Member {
    #[serde(default)]
    pub name: String,
    pub digest: String,
    pub length: u64,
}

pub struct LocalSources {
    pub objects: Arc<Objects>,
    pub installer: InstallerConfig,
    pub store: Arc<Store>,
    installing: Mutex<()>,
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

/// A local source's objects, its manifest and every member: what a run takes custody of when
/// it is accepted. One the signer did not write is `local_source_incomplete`, naming it.
pub fn written(objects: &Objects, actor: &str, digest: &str) -> Result<Vec<ObjectRef>, Refused> {
    let (manifest, own) = open_manifest(objects, actor, digest)?;
    let members = manifest
        .source
        .iter()
        .chain(&manifest.wheels)
        .chain(&manifest.requirements);
    let mut written = vec![own];
    for m in members {
        written.push(written_member(objects, actor, m)?);
    }
    Ok(written)
}

impl LocalSources {
    pub fn new(objects: Arc<Objects>, installer: InstallerConfig, store: Arc<Store>) -> Self {
        Self {
            objects,
            installer,
            store,
            installing: Mutex::new(()),
        }
    }

    /// This signer's installation of the manifest `digest` (`sha256:<hex>`).
    pub fn install(
        &self,
        service: &Service,
        actor: &str,
        digest: &str,
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
        let (manifest, _) = open_manifest(&self.objects, actor, digest)?;
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
        let prepared = prepare_uploaded(&self.installer, &uploaded)
            .map_err(|status| refused("package_installation_failed", status.message()))?;
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
        }
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
            "package": "local/cozy-machine-cpu-input",
            "release": "0.1.0",
            "python_version": "3.12",
            "source": {"digest": source.digest, "length": source.length},
        });
        let manifest = write(&objects, "alice", manifest.to_string().as_bytes());

        // Another signer cannot install what it did not write.
        let refused = sources
            .install(&service, "bob", &manifest.digest)
            .err()
            .unwrap();
        assert_eq!(refused.code, "local_source_incomplete");

        let installed = sources
            .install(&service, "alice", &manifest.digest)
            .unwrap();
        assert_eq!(installed.package, "local/cozy-machine-cpu-input");
        let interface: serde_json::Value = serde_json::from_slice(&installed.interface).unwrap();
        assert!(interface["entrypoints"]
            .as_array()
            .is_some_and(|e| !e.is_empty()));
        let again = sources
            .install(&service, "alice", &manifest.digest)
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
        let updated = pinned.install(&service, "alice", &manifest.digest).unwrap();
        assert_ne!(updated.alias, installed.alias);
        assert_ne!(updated.generation, installed.generation);
        let same_sdk = pinned.install(&service, "alice", &manifest.digest).unwrap();
        assert_eq!(same_sdk.alias, updated.alias);
        assert_eq!(same_sdk.generation, updated.generation);
        // A prior generation remains valid for already accepted work.
        assert!(service.catalog.resolve(&installed.generation).is_ok());
        let held = service.catalog.resolve(&updated.generation).unwrap();
        assert_eq!(installed_sdk(&held.record.python), "0.18.101 0.3.92");
        assert_eq!(held.record.sdk_fallback, "");
        let _ = fs::remove_dir_all(root);
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
        let manifest = serde_json::json!({"package": "local/cozy-machine-cpu-input", "release": "0.1.0", "python_version": "3.12",
            "wheels": [{"name": name, "digest": wheel.digest, "length": wheel.length}],
            "requirements": {"digest": locked.digest, "length": locked.length}});
        let manifest = write(&m.objects, "alice", manifest.to_string().as_bytes());
        let installed = sources.install(&m.service, "alice", &manifest.digest).unwrap();
        let python = m.service.catalog.resolve(&installed.generation).unwrap().record.python;
        let site = python.parent().unwrap().parent().unwrap().join("lib/python3.12/site-packages");
        // A locked dependency links into the seed; the App the Runtime describes is copied.
        let dependency = site.join("msgspec/__init__.py");
        assert!(fs::symlink_metadata(&dependency).unwrap().file_type().is_symlink(), "msgspec was copied into the environment");
        assert!(fs::canonicalize(&dependency).unwrap().starts_with(fs::canonicalize(&seed).unwrap()));
        assert!(!fs::symlink_metadata(site.join("cpu_input/__init__.py")).unwrap().file_type().is_symlink());

        // A callee App among the locked dependencies is copied too: the environment describes it.
        let wheel_of = |fixture: &str| {
            let out = m.root.join(fixture);
            run(Command::new("uv").current_dir(repo.join("tests/fixtures").join(fixture)).args(["build", "--wheel", "--out-dir"]).arg(&out));
            fs::read_dir(&out).unwrap().map(|e| e.unwrap().path()).find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap()
        };
        let (caller, memo) = (wheel_of("cpu_caller"), wheel_of("cpu_memo"));
        // A settled Runtime: the newest may be mid-upload (its sdist visible before its wheels).
        fs::write(m.root.join("settled.txt"), "cozy-runtime<0.19\n").unwrap();
        let compiled = run(Command::new("uv").args(["pip", "compile", "--quiet", "--generate-hashes", "--python-version", "3.12", "--no-header", "--constraint"])
            .arg(m.root.join("settled.txt")).arg("--find-links")
            .arg(memo.parent().unwrap()).arg(repo.join("tests/fixtures/cpu_caller/pyproject.toml")));
        let requirements = compiled.replace("cozy-machine-cpu-memo==0.1.0", &format!("cozy-machine-cpu-memo @ file://{}", memo.display()));
        let wheel = write(&m.objects, "alice", &fs::read(&caller).unwrap());
        let locked = write(&m.objects, "alice", requirements.as_bytes());
        let name = caller.file_name().unwrap().to_string_lossy();
        let manifest = serde_json::json!({"package": "local/cozy-machine-cpu-caller", "release": "0.1.0", "python_version": "3.12",
            "wheels": [{"name": name, "digest": wheel.digest, "length": wheel.length}],
            "requirements": {"digest": locked.digest, "length": locked.length}});
        let manifest = write(&m.objects, "alice", manifest.to_string().as_bytes());
        let installed = sources.install(&m.service, "alice", &manifest.digest).unwrap();
        let held = m.service.catalog.resolve(&installed.generation).unwrap();
        assert!(held.record.callees.iter().any(|c| c.distribution == "cozy-machine-cpu-memo"), "{:?}", held.record.callees);
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
        fs::write(project.join("pyproject.toml"), pyproject + "\n[tool.uv]\nconstraint-dependencies = [\"cozy-runtime<0.19\"]\n[tool.uv.sources]\ncozy-machine-cpu-memo = { path = \"deps/cpu_memo\" }\n").unwrap();
        run(Command::new("uv").args(["lock", "--python", "3.12", "--project"]).arg(&project));
        let tree = archive(&project, &["pyproject.toml", "uv.lock", "package.toml", "cpu_caller/__init__.py",
            "deps/cpu_memo/pyproject.toml", "deps/cpu_memo/package.toml", "deps/cpu_memo/cpu_memo/__init__.py"]);
        let source = write(&m.objects, "alice", &tree);
        let manifest = serde_json::json!({"package": "local/cozy-machine-cpu-caller", "release": "0.1.0", "python_version": "3.12",
            "source": {"digest": source.digest, "length": source.length}});
        let manifest = write(&m.objects, "alice", manifest.to_string().as_bytes());
        let installed = sources.install(&m.service, "alice", &manifest.digest).unwrap();
        let held = m.service.catalog.resolve(&installed.generation).unwrap();
        assert!(held.record.callees.iter().any(|c| c.distribution == "cozy-machine-cpu-memo"), "{:?}", held.record.callees);
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
            let mut manifest = serde_json::json!({"package": "local/cozy-machine-cpu-input", "release": "0.1.0", "python_version": "3.12"});
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
        let root_wheel = wheels.join("cozy_machine_cpu_input-0.1.0-py3-none-any.whl");
        let carried = manifest(serde_json::json!({
            "wheels": [
                member("cozy_machine_cpu_input-0.1.0-py3-none-any.whl", &fs::read(root_wheel).unwrap()),
                member(vendored.trim_start_matches("vendor/"), &fs::read(project.join(&vendored)).unwrap()),
            ],
            "requirements": member("requirements.txt", requirements.as_bytes()),
        }));
        for digest in [source, carried] {
            let installed = sources.install(&m.service, "alice", &digest).unwrap();
            let held = m.service.catalog.resolve(&installed.generation).unwrap();
            let versions = installed_sdk(&held.record.python);
            assert!(versions.starts_with(&format!("{dev} ")), "{versions}");
            let why = &held.record.sdk_fallback;
            assert!(why.contains(&format!("requires `cozy-runtime=={dev}`, but `0.18.101` is installed")), "{why}");
            assert!(why.ends_with(&format!("it runs the package's own cozy-runtime=={dev}, tensorfs=={}", &versions[dev.len() + 1..])), "{why}");
        }

        // A refusal carries its reason: a typed one as itself, a failed operation in uv's words.
        let refused = |digest: &str| sources.install(&m.service, "alice", digest).err().unwrap().message;
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
