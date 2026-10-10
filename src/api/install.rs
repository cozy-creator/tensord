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
    /// An image's baked uv cache: local environments symlink into it as published ones do, so
    /// overlayfs never copies its payload (Torch) into each environment of each edit.
    pub seed_cache: Option<PathBuf>,
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
        .envs(config.seed_cache.iter().flat_map(|seed| {
            [("UV_CACHE_DIR", seed.as_os_str()), ("UV_LINK_MODE", std::ffi::OsStr::new("symlink"))]
        }))
        .arg("-m")
        .arg("cozy_machine_client.installer")
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
        .stderr(Stdio::piped())
        .spawn()
        .map_err(storage)?;
    let stderr = child.stderr.take().expect("piped installer stderr");
    let tail = std::thread::spawn(move || forward_tail(stderr));
    let mut stdout = child.stdout.take().expect("piped installer stdout");
    let mut output = Vec::new();
    stdout
        .by_ref()
        .take((8 << 20) + 1)
        .read_to_end(&mut output)
        .map_err(storage)?;
    std::io::copy(&mut stdout, &mut std::io::sink()).map_err(storage)?;
    let exited = child.wait().map_err(storage)?;
    let tail = tail.join().unwrap_or_default();
    if !exited.success() {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename = "install_failed")]
        struct InstallationFailure {
            code: String,
            detail: String,
        }
        // A typed reason stands alone; an operation that failed or a helper that crashed is
        // explained by the helper's own last output lines (uv's words, or a traceback).
        let failure = serde_json::from_slice::<InstallationFailure>(&output).ok();
        let message = match &failure {
            Some(f) if f.code != "package_dependency_operation_failed" => format!("{}: {}", f.code, f.detail),
            Some(f) => format!("{}: {}: {tail}", f.code, f.detail),
            None => format!("package_installation_failed: the trusted Python/uv installer stopped ({exited}): {tail}"),
        };
        return Err(match failure {
            Some(f) if f.code.ends_with("_unsupported") => Status::unimplemented(message),
            _ => Status::failed_precondition(message),
        });
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

/// The helper's stderr, passed on to the machine log as it comes. Returns its last lines,
/// bounded and with URL credentials masked: what a refusal quotes.
fn forward_tail(mut stderr: impl Read) -> String {
    let (mut chunk, mut kept) = ([0u8; 8192], Vec::new());
    loop {
        match stderr.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                let _ = std::io::stderr().write_all(&chunk[..n]);
                kept.extend_from_slice(&chunk[..n]);
                kept.drain(..kept.len().saturating_sub(4096));
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    let text = String::from_utf8_lossy(&kept);
    let lines: Vec<_> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    masked(&lines[lines.len().saturating_sub(12)..].join(" | "))
}

/// `scheme://user:secret@host` becomes `scheme://***@host`.
fn masked(text: &str) -> String {
    let (mut out, mut rest) = (String::new(), text);
    while let Some(at) = rest.find("://") {
        let (head, after) = rest.split_at(at + 3);
        out.push_str(head);
        let authority = after.find(|c: char| c.is_whitespace() || c == '/').unwrap_or(after.len());
        rest = match after[..authority].rfind('@') {
            Some(user) => {
                out.push_str("***");
                &after[user..]
            }
            None => after,
        };
    }
    out + rest
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
    fn a_quoted_installer_line_never_carries_url_credentials() {
        assert_eq!(
            super::masked("GET https://user:tok@hub.example/simple/ and file:///x/y.whl"),
            "GET https://***@hub.example/simple/ and file:///x/y.whl"
        );
        let tail = super::forward_tail(std::io::Cursor::new(
            (0..40).map(|i| format!("line {i}\n")).collect::<String>(),
        ));
        assert!(tail.starts_with("line 28 | ") && tail.ends_with("line 39"), "{tail}");
    }

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
