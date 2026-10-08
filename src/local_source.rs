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
        })?;
        service.changed_environment()?;
        Ok(installed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::Path, process::Command};
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

    /// The real installer from objects the signer wrote; the same manifest reopens it.
    #[test]
    fn a_written_local_package_reuses_only_the_same_sdk_environment() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = std::env::temp_dir().join(format!("cm-local-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let built = Command::new("uv")
            .current_dir(repo)
            .args(["build", "--wheel", "--out-dir"])
            .arg(root.join("client"))
            .output()
            .unwrap();
        assert!(built.status.success());
        let helper = Command::new("uv")
            .current_dir(repo)
            .args([
                "run",
                "--locked",
                "--extra",
                "test",
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ])
            .output()
            .unwrap();
        let helper = String::from_utf8(helper.stdout).unwrap().trim().to_string();
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
        };
        let sources = LocalSources::new(objects.clone(), installer.clone(), store.clone());
        let mut archive = tar::Builder::new(Vec::new());
        let fixture = repo.join("tests/fixtures/cpu_input");
        for name in ["pyproject.toml", "package.toml", "cpu_input/__init__.py"] {
            archive
                .append_path_with_name(fixture.join(name), name)
                .unwrap();
        }
        let source = write(&objects, "alice", &archive.into_inner().unwrap());
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
        let sdk = root.join("sdk");
        fs::create_dir_all(&sdk).unwrap();
        let mut pair = vec![];
        for (name, url) in SDK_PAIR {
            let path = sdk.join(name);
            assert!(Command::new("curl")
                .args(["-sfL", "-o"])
                .arg(&path)
                .arg(url)
                .status()
                .unwrap()
                .success());
            pair.push(path);
        }
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
        let python = service
            .catalog
            .resolve(&updated.generation)
            .unwrap()
            .record
            .python
            .clone();
        let version = Command::new(python)
            .args(["-c", "import importlib.metadata as m; print(m.version('cozy-runtime'), m.version('tensorfs'))"])
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8(version.stdout).unwrap().trim(),
            "0.18.101 0.3.92"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// A Runtime/TensorFS pair other than PyPI's newest, standing in for the machine's own.
    const SDK_PAIR: [(&str, &str); 2] = [
        ("cozy_runtime-0.18.101-cp312-abi3-manylinux_2_28_x86_64.whl", "https://files.pythonhosted.org/packages/23/be/107c41ae8c978de51c315ba0c18152625454cd3767fb584683420b799d6b/cozy_runtime-0.18.101-cp312-abi3-manylinux_2_28_x86_64.whl"),
        ("tensorfs-0.3.92-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "https://files.pythonhosted.org/packages/f4/fc/8bc1e8fd0927b08257663e0b599da5e503a09adc8ee1db9cc183e78a73eb/tensorfs-0.3.92-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"),
    ];
}
