//! Materialize verified package carriers and call the trusted Python/uv installer.
//! The service commits actor/installation aliases in its one execution journal.
use super::workspaces::{RootSet, UploadedPackage};
use fs2::FileExt;
use serde::Deserialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
};
use tonic::Status;

#[derive(Clone)]
pub struct InstallerConfig {
    pub helper_python: PathBuf,
    pub python: String,
    pub generations: PathBuf,
    pub client_wheel: PathBuf,
    pub staging_root: PathBuf,
    /// The machine's own Runtime/TensorFS wheels: a local package runs this pair too.
    pub sdk: Vec<PathBuf>,
    /// The machine's uv (`machine::client::uv`): the helper runs it, whatever PATH the machine had.
    pub uv: PathBuf,
}
#[derive(Clone, Debug, Deserialize, serde::Serialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
}
#[derive(Debug, Deserialize, serde::Serialize)]
pub struct GenerationRecord {
    pub identity: String,
    pub package: String,
    pub version: String,
    pub application: String,
    pub python: PathBuf,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    pub interface: Box<serde_json::value::RawValue>,
}
pub struct PreparedGeneration {
    pub record: GenerationRecord,
    pub interface_bytes: Vec<u8>,
    pub hold: Arc<File>,
}
pub struct MaterializedPackage {
    pub root_set: RootSet,
    pub project: Option<PathBuf>,
    pub wheels: Vec<PathBuf>,
    pub requirements: Option<PathBuf>,
    root: PathBuf,
    _hold: File,
}
impl Drop for MaterializedPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

pub fn materialize_uploaded(
    uploaded: &UploadedPackage,
    staging: &Path,
) -> Result<MaterializedPackage, Status> {
    fs::create_dir_all(staging).map_err(storage)?;
    let uuid = fs::read_to_string("/proc/sys/kernel/random/uuid").map_err(storage)?;
    let root = staging.join(uuid.trim());
    fs::create_dir(&root).map_err(storage)?;
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).map_err(storage)?;
    let hold = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(root.join("active.lock"))
        .map_err(storage)?;
    hold.lock_exclusive().map_err(storage)?;
    let mut materialized = MaterializedPackage {
        root_set: uploaded.root.clone(),
        project: None,
        wheels: vec![],
        requirements: None,
        root,
        _hold: hold,
    };
    for carrier in &uploaded.files {
        let file = carrier.open()?;
        if carrier.filename == uploaded.root.source_archive {
            let project = materialized.root.join("source");
            fs::create_dir(&project).map_err(storage)?;
            let mut archive = tar::Archive::new(file);
            for entry in archive.entries().map_err(storage)? {
                let mut entry = entry.map_err(storage)?;
                let path = entry.path().map_err(storage)?.into_owned();
                if path.is_absolute()
                    || path
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                    || !(entry.header().entry_type().is_file()
                        || entry.header().entry_type().is_dir())
                {
                    return Err(Status::failed_precondition(
                        "unsafe source archive member during materialization",
                    ));
                }
                let target = project.join(&path);
                if entry.header().entry_type().is_dir() {
                    fs::create_dir_all(target).map_err(storage)?;
                    continue;
                }
                fs::create_dir_all(target.parent().expect("source member parent"))
                    .map_err(storage)?;
                let mut output = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .mode(0o600)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&target)
                    .map_err(storage)?;
                std::io::copy(&mut entry, &mut output).map_err(storage)?;
                output.sync_all().map_err(storage)?;
                fs::set_permissions(
                    &target,
                    fs::Permissions::from_mode(entry.header().mode().map_err(storage)? & 0o777),
                )
                .map_err(storage)?;
            }
            materialized.project = Some(project);
        } else {
            let path = materialized.root.join(&carrier.filename);
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&path)
                .map_err(storage)?;
            std::io::copy(&mut { file }, &mut output).map_err(storage)?;
            output.sync_all().map_err(storage)?;
            materialized.wheels.push(path);
        }
    }
    if !uploaded.root.dependency_requirements.is_empty() {
        let path = materialized.root.join("requirements.txt");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .map_err(storage)?;
        file.write_all(&uploaded.root.dependency_requirements)
            .map_err(storage)?;
        file.sync_all().map_err(storage)?;
        materialized.requirements = Some(path);
    }
    Ok(materialized)
}

pub fn prepare_uploaded(
    config: &InstallerConfig,
    uploaded: &UploadedPackage,
) -> Result<PreparedGeneration, Status> {
    let materialized = materialize_uploaded(uploaded, &config.staging_root)?;
    let captured_python = &materialized.root_set.python_version;
    let python = if !captured_python.is_empty() {
        let parts: Vec<_> = captured_python.split('.').collect();
        if !(2..=3).contains(&parts.len())
            || parts
                .iter()
                .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(Status::invalid_argument(
                "captured Python selection must be a numeric major.minor[.patch]",
            ));
        }
        captured_python
    } else {
        &config.python
    };
    let mut command = Command::new(&config.helper_python);
    crate::process::inherit_nothing(&mut command);
    // The uploaded build backend runs here: it gets the machine's environment without
    // credentials or Runtime-owned names, never the whole service environment.
    command
        .env_clear()
        .envs(crate::launch_identity::inherited())
        .env("PATH", helper_path(&config.uv))
        .arg("-m")
        .arg("cozy_machine_client.packages")
        .arg("install-captured")
        .arg("--generations")
        .arg(&config.generations)
        .arg("--client-wheel")
        .arg(&config.client_wheel)
        .args(
            config
                .sdk
                .iter()
                .flat_map(|wheel| [std::ffi::OsStr::new("--sdk-wheel"), wheel.as_os_str()]),
        )
        .arg("--python")
        .arg(python)
        .arg("--distribution")
        .arg(
            materialized
                .root_set
                .package
                .rsplit('/')
                .next()
                .expect("declared package"),
        )
        .arg("--release")
        .arg(&materialized.root_set.release)
        .arg("--callees")
        .arg(serde_json::to_string(&materialized.root_set.callees)
            .map_err(|error| Status::internal(error.to_string()))?)
        .arg(format!(
            "--python-requires={}",
            materialized.root_set.python_requires
        ))
        .arg(format!(
            "--python-version={}",
            materialized.root_set.python_version
        ));
    if let Some(project) = &materialized.project {
        command.arg("--project").arg(project);
    }
    for wheel in &materialized.wheels {
        command.arg("--wheel").arg(wheel);
    }
    if let Some(requirements) = &materialized.requirements {
        command.arg("--requirements").arg(requirements);
    }
    // Bounds on output memory do not kill a progressing installer. Always drain
    // its stdout and wait for its actual exit; no wall-clock kill is inserted.
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(storage)?;
    let mut stdout = child.stdout.take().expect("piped installer stdout");
    let mut output = Vec::new();
    stdout
        .by_ref()
        .take((8 << 20) + 1)
        .read_to_end(&mut output)
        .map_err(storage)?;
    std::io::copy(&mut stdout, &mut std::io::sink()).map_err(storage)?;
    let exited = child.wait().map_err(storage)?;
    if !exited.success() {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename = "install_failed")]
        struct InstallationFailure {
            code: String,
            detail: String,
        }
        if let Ok(failure) = serde_json::from_slice::<InstallationFailure>(&output) {
            let message = format!("{}: {}", failure.code, failure.detail);
            return Err(if failure.code.ends_with("_unsupported") {
                Status::unimplemented(message)
            } else {
                Status::failed_precondition(message)
            });
        }
        return Err(Status::failed_precondition("package_installation_failed: trusted Python/uv installer did not complete the captured operation"));
    }
    if output.len() > 8 << 20 {
        return Err(Status::resource_exhausted(
            "prepared package interface exceeds its control-record bound",
        ));
    }
    let returned: GenerationRecord = serde_json::from_slice(&output)
        .map_err(|_| Status::data_loss("installer returned an invalid typed generation record"))?;
    let generations = config.generations.canonicalize().map_err(storage)?;
    if returned.identity.len() != 32
        || !returned
            .identity
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Status::data_loss(
            "installer returned an invalid native generation identity",
        ));
    }
    let directory = generations.join(&returned.identity);
    if directory
        .symlink_metadata()
        .map_err(storage)?
        .file_type()
        .is_symlink()
        || directory.canonicalize().map_err(storage)? != directory
    {
        return Err(Status::data_loss(
            "published generation path is not an owned native directory",
        ));
    }
    let hold = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(directory.join(".hold"))
        .map_err(storage)?;
    hold.lock_shared().map_err(storage)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(directory.join("generation.json"))
        .map_err(storage)?;
    let record: GenerationRecord = serde_json::from_reader(file.take(8 << 20))
        .map_err(|_| Status::data_loss("published generation manifest is invalid"))?;
    if record.identity != returned.identity
        || record.python != directory.join("env/bin/python")
        || !record.python.is_file()
    {
        return Err(Status::data_loss(
            "published generation does not name its native interpreter",
        ));
    }
    let interface_bytes = record.interface.get().as_bytes().to_vec();
    Ok(PreparedGeneration {
        record,
        interface_bytes,
        hold: Arc::new(hold),
    })
}
fn storage(error: std::io::Error) -> Status {
    Status::unavailable(format!(
        "package installation storage/process unavailable: {error}"
    ))
}

/// The helper's PATH: the machine's uv's directory first, then the machine's own PATH. The helper
/// calls `uv` by name, and a machine started with no PATH still installs local packages.
fn helper_path(uv: &Path) -> std::ffi::OsString {
    let mut dirs: Vec<PathBuf> = uv
        .parent()
        .filter(|d| d.is_absolute())
        .map(Path::to_path_buf)
        .into_iter()
        .collect();
    dirs.extend(
        std::env::var_os("PATH")
            .iter()
            .flat_map(std::env::split_paths),
    );
    std::env::join_paths(dirs).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_helper_finds_the_machines_uv_first() {
        let path = super::helper_path(std::path::Path::new("/machine/usr/local/bin/uv"));
        let first = std::env::split_paths(&path).next();
        assert_eq!(
            first.as_deref(),
            Some(std::path::Path::new("/machine/usr/local/bin"))
        );
        // A bare `uv` (no root copy) leaves the machine's own PATH as it is.
        assert_eq!(
            super::helper_path(std::path::Path::new("uv")),
            std::env::var_os("PATH").unwrap_or_default()
        );
    }
}
