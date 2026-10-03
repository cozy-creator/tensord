//! `runtime-update/1` through the CLI's maintenance contract (`cozy rental update`): stage and
//! verify a Runtime/TensorFS pair, wait for measured idleness, then restart the service on it
//! in place. Executors' package environments take the new pair; a candidate Runtime wheel that
//! bundles a Rust machine also replaces the service binary (the stable parent runs it). The
//! previous pair and binary stay installed; a candidate that never proves readiness is rolled
//! back by the parent. Boot id, leaf, journal and outputs are kept.
use super::receipt::Readiness;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    os::unix::fs::{symlink, PermissionsExt},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub const CAPABILITY: &str = "runtime-update/1";
/// The service's exit status asking its parent to exec the selected application.
pub const REPLACE_EXIT: i32 = 75;
const MAX_WHEEL_BYTES: u64 = 256 << 20;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pair {
    pub runtime: String,
    pub tensorfs: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Status {
    pub operation: String,
    /// waiting | preparing | waiting_activation | installing | starting | succeeded | rolled_back | failed
    pub state: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    pub from: Pair,
    pub to: Pair,
    #[serde(default)]
    pub pinned: bool,
}
impl Status {
    fn terminal(&self) -> bool {
        matches!(self.state.as_str(), "succeeded" | "rolled_back" | "failed")
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub version: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    pub operation: String,
    #[serde(default)]
    pub agent: String,
    #[serde(default)]
    pub pin: Option<bool>,
    pub runtime: Option<Choice>,
    pub tensorfs: Option<Choice>,
}

/// The activation in flight: what a rollback restores.
#[derive(Serialize, Deserialize)]
struct Pending {
    status: Status,
    sdk_before: Option<PathBuf>,
    agent_before: Option<PathBuf>,
}

/// Where the machine keeps its software: everything under its own engine directory.
#[derive(Clone)]
pub struct Paths {
    engine: PathBuf,
    image_wheels: PathBuf,
}
impl Paths {
    pub fn new(engine: &Path, root: &Path) -> Self {
        // A development image may vendor this machine's own executor SDK beside the Go agent's.
        let own = root.join("opt/cozy/machine/wheels");
        let image_wheels = if own.is_dir() {
            own
        } else {
            root.join("opt/cozy/wheels")
        };
        Self {
            engine: engine.into(),
            image_wheels,
        }
    }
    fn update(&self, name: &str) -> PathBuf {
        self.engine.join("update").join(name)
    }
    fn current_sdk_link(&self) -> PathBuf {
        self.engine.join("sdk/current")
    }
    fn current_agent_link(&self) -> PathBuf {
        self.engine.join("agent/current")
    }
    /// The executors' Runtime/TensorFS wheels: the last activated pair, else the image's.
    pub fn sdk(&self) -> PathBuf {
        let link = self.current_sdk_link();
        if link.exists() {
            link
        } else {
            self.image_wheels.clone()
        }
    }
}

/// The machine binary an activated update installed, which the parent runs as its service.
pub fn activated_binary(paths: &Paths) -> Option<PathBuf> {
    let link = paths.current_agent_link();
    link.exists().then_some(link)
}

/// Restores what a failed activation replaced. True when there was one to undo.
pub fn rollback_pending(paths: &Paths, cause: &str) -> io::Result<bool> {
    let Ok(raw) = fs::read(paths.update("pending.json")) else {
        return Ok(false);
    };
    let pending: Pending = serde_json::from_slice(&raw)?;
    relink(&paths.current_sdk_link(), pending.sdk_before.as_deref())?;
    relink(&paths.current_agent_link(), pending.agent_before.as_deref())?;
    let mut status = pending.status;
    status.state = "rolled_back".into();
    status.error = cause.into();
    write_json(&paths.update("status.json"), &status)?;
    fs::remove_file(paths.update("pending.json"))?;
    Ok(true)
}

pub struct Updates {
    paths: Paths,
    readiness: Arc<Readiness>,
    status: Mutex<Option<Status>>,
    idle: Box<dyn Fn() -> bool + Send + Sync>,
}

impl Updates {
    pub fn open(
        paths: Paths,
        readiness: Arc<Readiness>,
        idle: Box<dyn Fn() -> bool + Send + Sync>,
    ) -> io::Result<Arc<Self>> {
        fs::create_dir_all(paths.update("staged"))?;
        let status = fs::read(paths.update("status.json"))
            .ok()
            .and_then(|raw| serde_json::from_slice::<Status>(&raw).ok());
        Ok(Arc::new(Self {
            paths,
            readiness,
            status: Mutex::new(status),
            idle,
        }))
    }

    /// After this process proved readiness: an activation in flight is committed.
    pub fn commit(&self) -> io::Result<()> {
        let Ok(raw) = fs::read(self.paths.update("pending.json")) else {
            return Ok(());
        };
        let pending: Pending = serde_json::from_slice(&raw)?;
        let mut status = pending.status;
        status.state = "succeeded".into();
        status.to = pair_in(&self.paths.sdk());
        write_json(&self.paths.update("status.json"), &status)?;
        fs::remove_file(self.paths.update("pending.json"))?;
        *self.status.lock().unwrap() = Some(status);
        Ok(())
    }

    /// GET /v1/machine/runtime.
    pub fn state(&self, agent_capabilities: &[&str]) -> serde_json::Value {
        let pair = pair_in(&self.paths.sdk());
        let (sha256, selection) = match fs::read("/proc/self/exe") {
            Ok(bytes) => (
                hex(&Sha256::digest(bytes)),
                if self.paths.current_agent_link().exists() {
                    "bundled"
                } else {
                    "explicit"
                },
            ),
            Err(_) => (String::new(), "explicit"),
        };
        serde_json::json!({
            "phase": if self.readiness.proved() { "ready" } else { "booting" },
            "capabilities": [CAPABILITY],
            "runtime": pair.runtime,
            "tensorfs": pair.tensorfs,
            "agent": {"version": env!("CARGO_PKG_VERSION"), "sha256": sha256, "selection": selection, "capabilities": agent_capabilities},
            "bootstrap": {"abi": "machine-bootstrap/1", "version": env!("CARGO_PKG_VERSION"), "update_boundary": "measured-idle"},
            "update": *self.status.lock().unwrap(),
        })
    }

    /// PUT /v1/machine/runtime/wheels/{file}: kept as staged/<sha256>/<file>.
    pub fn stage(&self, file: &str, body: &mut dyn Read) -> Result<(String, u64), (u16, String)> {
        if self.busy() {
            return Err((
                409,
                "the machine is running a Runtime update; stage wheels after it finishes".into(),
            ));
        }
        if wheel_name(file).is_none() {
            return Err((400, "stage a cozy_runtime or tensorfs wheel".into()));
        }
        let temp = self.paths.update(&format!(
            "staged/.upload-{}",
            hex(&super::identity::random::<8>().map_err(server)?)
        ));
        let mut out = fs::File::create(&temp).map_err(server)?;
        let mut hash = Sha256::new();
        let (mut buffer, mut length) = (vec![0u8; 1 << 20], 0u64);
        loop {
            let n = body
                .read(&mut buffer)
                .map_err(|e| (400, format!("the wheel upload broke: {e}")))?;
            if n == 0 {
                break;
            }
            length += n as u64;
            if length > MAX_WHEEL_BYTES {
                let _ = fs::remove_file(&temp);
                return Err((400, "the wheel exceeds 256 MiB".into()));
            }
            hash.update(&buffer[..n]);
            out.write_all(&buffer[..n]).map_err(server)?;
        }
        out.sync_all().map_err(server)?;
        let sha = hex(&hash.finalize());
        let dir = self.paths.update(&format!("staged/{sha}"));
        fs::create_dir_all(&dir).map_err(server)?;
        fs::rename(&temp, dir.join(file)).map_err(server)?;
        Ok((sha, length))
    }

    fn busy(&self) -> bool {
        self.status
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|s| !s.terminal())
    }

    /// POST /v1/machine/runtime/update: accepts one operation and runs it in the background.
    pub fn request(
        self: &Arc<Self>,
        request: Request,
        exit: fn(i32),
    ) -> Result<Status, (u16, String)> {
        let valid = !request.operation.is_empty()
            && request.operation.len() <= 128
            && request
                .operation
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
            && matches!(request.agent.as_str(), "" | "bundled" | "explicit")
            && (request.runtime.is_some() || request.tensorfs.is_some());
        if !valid {
            return Err((
                400,
                "an update requires operation, valid agent selection, and a Runtime or TensorFS"
                    .into(),
            ));
        }
        let mut current = self.status.lock().unwrap();
        if let Some(status) = current.as_ref() {
            if status.operation == request.operation {
                return Ok(status.clone());
            }
            if !status.terminal() {
                return Err((
                    409,
                    format!(
                        "the machine is running another Runtime update: {}",
                        status.operation
                    ),
                ));
            }
        }
        let status = Status {
            operation: request.operation.clone(),
            state: "waiting".into(),
            from: pair_in(&self.paths.sdk()),
            pinned: request.pin.unwrap_or(false),
            ..Status::default()
        };
        write_json(&self.paths.update("status.json"), &status).map_err(server)?;
        *current = Some(status.clone());
        drop(current);
        let updates = self.clone();
        std::thread::Builder::new()
            .name("runtime-update".into())
            .spawn(move || {
                if let Err(error) = updates.run(&request, exit) {
                    updates.set(|s| {
                        s.state = "failed".into();
                        s.error = error.to_string();
                    });
                }
            })
            .map_err(server)?;
        Ok(status)
    }

    fn set(&self, change: impl FnOnce(&mut Status)) {
        let mut current = self.status.lock().unwrap();
        if let Some(status) = current.as_mut() {
            change(status);
            if let Err(error) = write_json(&self.paths.update("status.json"), status) {
                eprintln!("cozy-machine: Runtime update status: {error}");
            }
        }
    }

    fn run(&self, request: &Request, exit: fn(i32)) -> io::Result<()> {
        self.set(|s| s.state = "preparing".into());
        let candidate = self.paths.engine.join("sdk").join(&request.operation);
        let _ = fs::remove_dir_all(&candidate);
        fs::create_dir_all(&candidate)?;
        let mut agent = None;
        for (distribution, choice) in [
            ("cozy_runtime", &request.runtime),
            ("tensorfs", &request.tensorfs),
        ] {
            let wheel = match choice {
                Some(choice) => self.resolve(distribution, choice)?,
                None => existing_wheel(&self.paths.sdk(), distribution)?,
            };
            let name = wheel
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| invalid("wheel name"))?
                .to_owned();
            let bundled = verify_wheel(&wheel, distribution)?;
            fs::copy(&wheel, candidate.join(&name))?;
            if let Some(binary) = bundled.filter(|_| request.agent != "explicit") {
                agent = rust_machine(&binary, &candidate)?;
            }
        }
        let to = pair_in(&candidate);
        self.set(|s| s.to = to.clone());
        if !(self.idle)() {
            self.set(|s| s.state = "waiting_activation".into());
            while !(self.idle)() {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        self.set(|s| s.state = "installing".into());
        let pending = Pending {
            status: self.status.lock().unwrap().clone().unwrap_or_default(),
            sdk_before: fs::read_link(self.paths.current_sdk_link()).ok(),
            agent_before: fs::read_link(self.paths.current_agent_link()).ok(),
        };
        write_json(&self.paths.update("pending.json"), &pending)?;
        relink(&self.paths.current_sdk_link(), Some(&candidate))?;
        if let Some(binary) = agent {
            relink(&self.paths.current_agent_link(), Some(&binary))?;
        }
        self.set(|s| s.state = "starting".into());
        eprintln!(
            "cozy-machine: Runtime update {}: restarting on {} / {}",
            request.operation, to.runtime, to.tensorfs
        );
        exit(REPLACE_EXIT);
        Ok(())
    }

    /// A staged {file, sha256} or a published {version} fetched from PyPI with its digest checked.
    fn resolve(&self, distribution: &str, choice: &Choice) -> io::Result<PathBuf> {
        if !choice.file.is_empty() {
            let path = self
                .paths
                .update(&format!("staged/{}/{}", choice.sha256, choice.file));
            if choice.file.contains('/')
                || !path.is_file()
                || hex(&Sha256::digest(fs::read(&path)?)) != choice.sha256
            {
                return Err(invalid(&format!(
                    "{} was not staged with digest {}",
                    choice.file, choice.sha256
                )));
            }
            return Ok(path);
        }
        if choice.version.is_empty() {
            return Err(invalid("a pair member names a staged file or a version"));
        }
        super::pypi::wheel(
            distribution,
            &choice.version,
            &self.paths.update("downloads"),
        )
    }
}

fn existing_wheel(dir: &Path, distribution: &str) -> io::Result<PathBuf> {
    wheels(dir)
        .into_iter()
        .find(|(d, _, _)| d == distribution)
        .map(|(_, _, path)| path)
        .ok_or_else(|| invalid(&format!("no installed {distribution} wheel")))
}

/// Checks the wheel's METADATA names the distribution and version its file name does; returns
/// a bundled `cozy-machine` executable when a Runtime wheel carries one.
fn verify_wheel(path: &Path, distribution: &str) -> io::Result<Option<Vec<u8>>> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let (wheel_distribution, version) =
        wheel_name(name).ok_or_else(|| invalid("not a Runtime or TensorFS wheel"))?;
    if wheel_distribution != distribution {
        return Err(invalid(&format!("{name} is not a {distribution} wheel")));
    }
    let mut archive = zip::ZipArchive::new(fs::File::open(path)?).map_err(io::Error::other)?;
    let metadata = format!("{distribution}-{version}.dist-info/METADATA");
    let mut text = String::new();
    archive
        .by_name(&metadata)
        .map_err(|_| invalid(&format!("{name} has no {metadata}")))?
        .read_to_string(&mut text)?;
    let field = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key))
            .map(str::trim)
            .unwrap_or_default()
            .to_owned()
    };
    if field("Name:").to_lowercase().replace(['-', '.'], "_") != distribution
        || field("Version:") != version
    {
        return Err(invalid(&format!(
            "{name} METADATA names another distribution or version"
        )));
    }
    let script = format!("{distribution}-{version}.data/scripts/cozy-machine");
    let Ok(mut entry) = archive.by_name(&script) else {
        return Ok(None);
    };
    let mut binary = Vec::new();
    entry.read_to_end(&mut binary)?;
    Ok(Some(binary))
}

/// The bundled executable when it is a Rust machine (its `version --json` says so). A wheel
/// bundling another machine agent leaves this one running.
fn rust_machine(binary: &[u8], dir: &Path) -> io::Result<Option<PathBuf>> {
    let path = dir.join("cozy-machine");
    fs::write(&path, binary)?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
    let output = std::process::Command::new(&path)
        .args(["version", "--json"])
        .output();
    #[derive(Deserialize)]
    struct Identity {
        name: String,
        #[serde(default)]
        implementation: String,
    }
    let rust = output
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| serde_json::from_slice::<Identity>(&o.stdout).ok())
        .is_some_and(|i| i.name == "cozy-machine" && i.implementation == "rust");
    if !rust {
        fs::remove_file(&path)?;
    }
    Ok(rust.then_some(path))
}

/// `cozy_runtime-0.18.102-cp312-…whl` → ("cozy_runtime", "0.18.102").
fn wheel_name(file: &str) -> Option<(String, String)> {
    let stem = file.strip_suffix(".whl")?;
    let mut parts = stem.split('-');
    let (distribution, version) = (parts.next()?.to_lowercase(), parts.next()?.to_owned());
    matches!(distribution.as_str(), "cozy_runtime" | "tensorfs").then_some((distribution, version))
}

fn wheels(dir: &Path) -> Vec<(String, String, PathBuf)> {
    let mut found: Vec<_> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let (d, v) = wheel_name(&name)?;
            Some((d, v, e.path()))
        })
        .collect();
    found.sort();
    found
}

pub fn pair_in(dir: &Path) -> Pair {
    let mut pair = Pair::default();
    for (distribution, version, _) in wheels(dir) {
        match distribution.as_str() {
            "cozy_runtime" => pair.runtime = version,
            _ => pair.tensorfs = version,
        }
    }
    pair
}

/// Points `link` at `target` atomically, or removes it.
fn relink(link: &Path, target: Option<&Path>) -> io::Result<()> {
    let Some(target) = target else {
        return match fs::remove_file(link) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    };
    let dir = link
        .parent()
        .ok_or_else(|| invalid("a link needs a directory"))?;
    fs::create_dir_all(dir)?;
    let temp = dir.join(format!(".link-{}", hex(&super::identity::random::<8>()?)));
    symlink(target, &temp)?;
    fs::rename(&temp, link)?;
    fs::File::open(dir)?.sync_all()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    super::identity::write_atomic(path, &serde_json::to_vec(value)?, 0o600)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail.to_owned())
}

fn server(error: io::Error) -> (u16, String) {
    (500, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal wheel: METADATA, optionally a bundled script.
    fn wheel(dir: &Path, distribution: &str, version: &str, script: Option<&[u8]>) -> PathBuf {
        let path = dir.join(format!("{distribution}-{version}-py3-none-any.whl"));
        let mut zip = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file(
            format!("{distribution}-{version}.dist-info/METADATA"),
            options,
        )
        .unwrap();
        zip.write_all(
            format!(
                "Metadata-Version: 2.1\nName: {}\nVersion: {version}\n",
                distribution.replace('_', "-")
            )
            .as_bytes(),
        )
        .unwrap();
        if let Some(script) = script {
            zip.start_file(
                format!("{distribution}-{version}.data/scripts/cozy-machine"),
                options,
            )
            .unwrap();
            zip.write_all(script).unwrap();
        }
        zip.finish().unwrap();
        path
    }

    #[test]
    fn wheels_verify_by_metadata_and_a_rust_machine_is_recognized() {
        let dir = std::env::temp_dir().join(format!("cozy-update-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let script = b"#!/bin/sh\necho '{\"name\":\"cozy-machine\",\"implementation\":\"rust\"}'\n";
        let runtime = wheel(&dir, "cozy_runtime", "0.18.103", Some(script));
        let binary = verify_wheel(&runtime, "cozy_runtime").unwrap().unwrap();
        assert!(rust_machine(&binary, &dir).unwrap().is_some());
        let go = b"#!/bin/sh\necho '{\"name\":\"cozy-machine\",\"version\":\"0.18.103\"}'\n";
        assert!(
            rust_machine(go, &dir).unwrap().is_none(),
            "another agent leaves this machine running"
        );
        assert!(verify_wheel(&runtime, "tensorfs").is_err());
        let wrong = dir.join("tensorfs-0.3.94-py3-none-any.whl");
        fs::copy(wheel(&dir, "tensorfs", "0.3.93", None), &wrong).unwrap();
        assert!(
            verify_wheel(&wrong, "tensorfs").is_err(),
            "METADATA must agree with the file name"
        );
        assert_eq!(pair_in(&dir).runtime, "0.18.103");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_failed_activation_rolls_back_links_and_reports_it() {
        let dir = std::env::temp_dir().join(format!("cozy-rollback-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let paths = Paths::new(&dir.join("engine"), &dir);
        let (old, new) = (dir.join("engine/sdk/old"), dir.join("engine/sdk/new"));
        fs::create_dir_all(&old).unwrap();
        fs::create_dir_all(&new).unwrap();
        relink(&paths.current_sdk_link(), Some(&old)).unwrap();
        let status = Status {
            operation: "op".into(),
            state: "starting".into(),
            ..Status::default()
        };
        write_json(
            &paths.update("pending.json"),
            &Pending {
                status,
                sdk_before: Some(old.clone()),
                agent_before: None,
            },
        )
        .unwrap();
        relink(&paths.current_sdk_link(), Some(&new)).unwrap();
        relink(&paths.current_agent_link(), Some(&new.join("cozy-machine"))).unwrap();
        assert!(rollback_pending(&paths, "candidate never became ready").unwrap());
        assert_eq!(fs::read_link(paths.current_sdk_link()).unwrap(), old);
        assert!(fs::symlink_metadata(paths.current_agent_link()).is_err());
        let status: Status =
            serde_json::from_slice(&fs::read(paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(
            (status.state.as_str(), status.error.as_str()),
            ("rolled_back", "candidate never became ready")
        );
        assert!(!rollback_pending(&paths, "again").unwrap());
        fs::remove_dir_all(dir).unwrap();
    }
}
