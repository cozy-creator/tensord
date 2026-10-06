//! A software update, as `Run kind: update` on `cozy.machine.v1` and (until cutover)
//! `runtime-update/1` behind the CLI's maintenance routes: stage and verify a Runtime/TensorFS
//! pair, wait for measured idleness, then restart the service on it in place. Executors' package environments take the new pair; a candidate Runtime wheel that
//! bundles a Rust machine also replaces the service binary (the stable parent runs it). The
//! previous pair and binary stay installed; a candidate that never proves readiness is rolled
//! back by the parent. Boot id, leaf, journal and outputs are kept.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    os::unix::fs::{symlink, PermissionsExt},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

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
    /// Every state it passed through: the update's run log.
    #[serde(default)]
    pub history: Vec<Step>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub state: String,
    pub at_ms: i64,
}
impl Status {
    pub fn terminal(&self) -> bool {
        matches!(self.state.as_str(), "succeeded" | "rolled_back" | "failed")
    }
    fn enter(&mut self, state: &str) {
        if self.state != state || self.history.is_empty() {
            self.state = state.into();
            self.history.push(Step {
                state: state.into(),
                at_ms: super::lifecycle::now_ms(),
            });
        }
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
pub fn activated_binary(paths: &Paths) -> io::Result<Option<PathBuf>> {
    let link = paths.current_agent_link();
    // A dangling activation is still the selected candidate. Its exec must fail and roll back,
    // rather than silently starting the original binary and committing the candidate as ready.
    match fs::symlink_metadata(&link) {
        Ok(_) => Ok(Some(link)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Restores what a failed activation replaced. True when there was one to undo.
pub fn rollback_pending(paths: &Paths, cause: &str) -> io::Result<bool> {
    let Some(pending) = read_pending(paths)? else {
        return Ok(false);
    };
    let mut status = latest(paths, pending.status)?;
    status.error = cause.into();
    status.enter("rolled_back");
    // Persist the decision before changing either link. A crash during rollback then resumes
    // rollback, even when the candidate had already published its starting marker.
    write_json(&paths.update("status.json"), &status)?;
    relink(&paths.current_sdk_link(), pending.sdk_before.as_deref())?;
    relink(&paths.current_agent_link(), pending.agent_before.as_deref())?;
    remove_pending(paths)?;
    Ok(true)
}

/// Recovers a publication interrupted before its durable `starting` marker. The parent calls
/// this before choosing its service binary; a new service also calls it when an older parent
/// launched it. True means links were restored and that child must let its parent select again.
/// A succeeded marker means readiness already committed the update: only its cleanup remains.
pub fn recover_activation(paths: &Paths) -> io::Result<bool> {
    let Some(pending) = read_pending(paths)? else {
        return Ok(false);
    };
    let status = latest(paths, pending.status)?;
    match status.state.as_str() {
        "starting" => Ok(false),
        "succeeded" => {
            remove_pending(paths)?;
            Ok(false)
        }
        "rolled_back" => rollback_pending(paths, &status.error),
        "waiting" | "preparing" | "waiting_activation" | "installing" | "failed" => {
            rollback_pending(paths, "machine_restarted: software publication did not finish")
        }
        state => Err(invalid(&format!("cannot recover update state {state}"))),
    }
}

fn read_pending(paths: &Paths) -> io::Result<Option<Pending>> {
    match fs::read(paths.update("pending.json")) {
        Ok(raw) => Ok(Some(serde_json::from_slice(&raw)?)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn remove_pending(paths: &Paths) -> io::Result<()> {
    fs::remove_file(paths.update("pending.json"))?;
    fs::File::open(paths.update(""))?.sync_all()
}

/// The activation's status as last written (its later steps), else as it was when pending.
/// A missing status has its pending fallback; an unreadable status must not commit by accident.
fn latest(paths: &Paths, pending: Status) -> io::Result<Status> {
    match fs::read(paths.update("status.json")) {
        Ok(raw) => {
            let status: Status = serde_json::from_slice(&raw)?;
            Ok(if status.operation == pending.operation { status } else { pending })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(pending),
        Err(error) => Err(error),
    }
}

pub struct Updates {
    paths: Paths,
    status: Mutex<Option<Status>>,
    idle: Box<dyn Fn() -> bool + Send + Sync>,
    lifecycle: Option<Arc<super::lifecycle::Lifecycle>>,
}

impl Updates {
    pub fn open(
        paths: Paths,
        idle: Box<dyn Fn() -> bool + Send + Sync>,
        lifecycle: Option<Arc<super::lifecycle::Lifecycle>>,
    ) -> io::Result<Arc<Self>> {
        fs::create_dir_all(paths.update("staged"))?;
        let mut status = fs::read(paths.update("status.json"))
            .ok()
            .and_then(|raw| serde_json::from_slice::<Status>(&raw).ok());
        // An update the service stopped during before activation ran no further: it failed.
        let pending = paths.update("pending.json").is_file();
        if let Some(stopped) = status.as_mut().filter(|s| !s.terminal() && !pending) {
            stopped.error = "machine_restarted: the machine stopped before the update activated".into();
            stopped.enter("failed");
            write_json(&paths.update("status.json"), stopped)?;
        }
        Ok(Arc::new(Self {
            paths,
            status: Mutex::new(status),
            idle,
            lifecycle,
        }))
    }

    /// After this process proved readiness: an activation in flight is committed.
    pub fn commit(&self) -> io::Result<()> {
        let Some(pending) = read_pending(&self.paths)? else {
            return Ok(());
        };
        let mut status = latest(&self.paths, pending.status)?;
        if !matches!(status.state.as_str(), "starting" | "succeeded") {
            return Err(invalid("cannot commit an update before software publication finished"));
        }
        status.to = pair_in(&self.paths.sdk());
        status.enter("succeeded");
        self.commit_status(status)
    }

    /// The durable success is authoritative even if removing the rollback record fails. A
    /// later startup finishes that cleanup; observers of this live service see success now.
    fn commit_status(&self, status: Status) -> io::Result<()> {
        write_json(&self.paths.update("status.json"), &status)?;
        *self.status.lock().unwrap() = Some(status);
        remove_pending(&self.paths)
    }

    /// The update `operation` names, if it is this machine's latest.
    pub fn update(&self, operation: &str) -> Option<Status> {
        self.status
            .lock()
            .unwrap()
            .clone()
            .filter(|s| s.operation == operation)
    }

    /// The executors' Runtime/TensorFS pair.
    pub fn software(&self) -> Pair {
        pair_in(&self.paths.sdk())
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
        // A committed terminal outcome may still have cleanup to do. Finish that before a
        // different operation overwrites its status and loses the pending decision's meaning.
        recover_activation(&self.paths).map_err(server)?;
        // Preparing holds idle release; activation then closes admission.
        let admitted = self.lifecycle.as_ref().map(|l| l.admit()).transpose()
            .map_err(|refused| (503, refused.message().to_string()))?;
        let mut status = Status {
            operation: request.operation.clone(),
            from: pair_in(&self.paths.sdk()),
            pinned: request.pin.unwrap_or(false),
            ..Status::default()
        };
        status.enter("waiting");
        write_json(&self.paths.update("status.json"), &status).map_err(server)?;
        *current = Some(status.clone());
        drop(current);
        let updates = self.clone();
        std::thread::Builder::new()
            .name("runtime-update".into())
            .spawn(move || {
                if let Err(error) = updates.run(&request, exit, admitted) {
                    if let Err(persist) = updates.set(|s| {
                        if !s.terminal() {
                            s.error = error.to_string();
                            s.enter("failed");
                        }
                    }) {
                        eprintln!("cozy-machine: Runtime update failed: {error}; status: {persist}");
                    }
                }
            })
            .map_err(server)?;
        Ok(status)
    }

    fn set(&self, change: impl FnOnce(&mut Status)) -> io::Result<()> {
        let mut current = self.status.lock().unwrap();
        if let Some(mut status) = current.clone() {
            change(&mut status);
            write_json(&self.paths.update("status.json"), &status)?;
            *current = Some(status);
        }
        Ok(())
    }

    fn run(&self, request: &Request, exit: fn(i32), admitted: Option<super::lifecycle::Admission>) -> io::Result<()> {
        self.set(|s| s.enter("preparing"))?;
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
            let prepared = candidate.join(&name);
            fs::copy(&wheel, &prepared)?;
            fs::File::open(prepared)?.sync_all()?;
            if let Some(binary) = bundled.filter(|_| request.agent != "explicit") {
                agent = rust_machine(&binary, &candidate)?;
            }
        }
        fs::File::open(&candidate)?.sync_all()?;
        let to = pair_in(&candidate);
        self.set(|s| s.to = to.clone())?;
        // No new work is admitted from here; work admitted before drains first.
        let _activation = self.lifecycle.as_ref().map(|l| l.activate()).transpose()
            .map_err(|refused| io::Error::other(refused.message().to_string()))?;
        drop(admitted);
        if !(self.idle)() {
            self.set(|s| s.enter("waiting_activation"))?;
            while !(self.idle)() {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        self.set(|s| s.enter("installing"))?;
        let pending = Pending {
            status: self.status.lock().unwrap().clone().unwrap_or_default(),
            sdk_before: previous_link(&self.paths.current_sdk_link())?,
            agent_before: previous_link(&self.paths.current_agent_link())?,
        };
        let publication = (|| {
            write_json(&self.paths.update("pending.json"), &pending)?;
            relink(&self.paths.current_sdk_link(), Some(&candidate))?;
            if let Some(binary) = agent {
                relink(&self.paths.current_agent_link(), Some(&binary))?;
            }
            self.set(|s| s.enter("starting"))
        })();
        if let Err(error) = publication {
            // Admission is still closed. Do not leave a partial SDK/binary pair available to
            // newly accepted work, and do not overwrite a truthful rolled_back outcome.
            match rollback_pending(&self.paths, &format!("software publication failed: {error}")) {
                Ok(true) => {
                    *self.status.lock().unwrap() = Some(latest(&self.paths, pending.status)?);
                }
                Ok(false) => {}
                Err(rollback) => {
                    eprintln!("cozy-machine: update publication: {error}; rollback: {rollback}");
                    // The links cannot be trusted. Restart while admission remains closed;
                    // startup recovers before selecting software, or refuses this launch.
                    exit(REPLACE_EXIT);
                }
            }
            return Err(error);
        }
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
    if rust {
        fs::File::open(&path)?.sync_all()?;
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

fn previous_link(link: &Path) -> io::Result<Option<PathBuf>> {
    match fs::read_link(link) {
        Ok(target) => Ok(Some(target)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Points `link` at `target` atomically, or removes it.
fn relink(link: &Path, target: Option<&Path>) -> io::Result<()> {
    let Some(target) = target else {
        return match fs::remove_file(link) {
            Ok(()) => fs::File::open(link.parent().expect("link has a parent"))?.sync_all(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
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

    fn state(updates: &Updates, id: &str, until: &str) {
        loop {
            let status = updates.update(id).unwrap();
            if status.state == until {
                return;
            }
            assert!(!status.terminal(), "{}: {}", status.state, status.error);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Activation admits nothing new while work admitted before it drains, so no run is
    /// accepted while the service exits; it reopens once the update is done.
    #[test]
    fn activation_closes_admission_while_admitted_work_drains() {
        let root = std::env::temp_dir().join(format!("cm-activation-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        fs::create_dir_all(&paths.image_wheels).unwrap();
        wheel(&paths.image_wheels, "cozy_runtime", "0.1.0", None);
        wheel(&paths.image_wheels, "tensorfs", "0.1.0", None);
        let lifecycle = super::super::lifecycle::Lifecycle::open(root.join("idle.json"), false, true).unwrap();
        let earlier = lifecycle.admit().unwrap();
        let drained = lifecycle.clone();
        let idle = Box::new(move || drained.admitted() == 0);
        let updates = Updates::open(paths.clone(), idle, Some(lifecycle.clone())).unwrap();
        let candidate = wheel(&root, "cozy_runtime", "0.2.0", None);
        let file = candidate.file_name().unwrap().to_str().unwrap().to_owned();
        let (sha256, _) = updates.stage(&file, &mut fs::File::open(&candidate).unwrap()).unwrap();
        let runtime = Some(Choice { file, sha256, ..Default::default() });
        let request = Request { operation: "u1".into(), agent: "explicit".into(), pin: None, runtime, tensorfs: None };
        updates.request(request, |code| assert_eq!(code, REPLACE_EXIT)).unwrap();
        state(&updates, "u1", "waiting_activation");
        assert!(lifecycle.admit().is_err(), "a new run was admitted during activation");
        assert!(!paths.update("pending.json").is_file());
        drop(earlier);
        state(&updates, "u1", "starting");
        while lifecycle.admit().is_err() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        fs::remove_dir_all(root).unwrap();
    }

    /// An update the service stopped during before activation is failed, not left running.
    #[test]
    fn an_update_stopped_before_activation_fails() {
        let root = std::env::temp_dir().join(format!("cm-stopped-update-{}", uuid::Uuid::new_v4()));
        let paths = Paths::new(&root.join("engine"), &root);
        let mut status = Status { operation: "u1".into(), ..Default::default() };
        status.enter("preparing");
        fs::create_dir_all(paths.update("")).unwrap();
        write_json(&paths.update("status.json"), &status).unwrap();
        let updates = Updates::open(paths, Box::new(|| true), None).unwrap();
        let stopped = updates.update("u1").unwrap();
        assert_eq!(stopped.state, "failed");
        assert!(stopped.error.starts_with("machine_restarted"));
        fs::remove_dir_all(root).unwrap();
    }

    struct Activation {
        root: PathBuf,
        paths: Paths,
        old: PathBuf,
        new: PathBuf,
    }
    impl Activation {
        fn prepare() -> Self {
            let root = std::env::temp_dir().join(format!("cm-publication-{}", uuid::Uuid::new_v4()));
            let paths = Paths::new(&root.join("engine"), &root);
            let old = root.join("engine/sdk/old");
            let new = root.join("engine/sdk/new");
            for (dir, version) in [(&old, "0.1.0"), (&new, "0.2.0")] {
                fs::create_dir_all(dir).unwrap();
                wheel(dir, "cozy_runtime", version, None);
                wheel(dir, "tensorfs", version, None);
                fs::write(dir.join("cozy-machine"), b"executable").unwrap();
            }
            relink(&paths.current_sdk_link(), Some(&old)).unwrap();
            relink(&paths.current_agent_link(), Some(&old.join("cozy-machine"))).unwrap();
            let mut status = Status { operation: "transaction".into(), ..Default::default() };
            status.enter("installing");
            write_json(&paths.update("status.json"), &status).unwrap();
            write_json(&paths.update("pending.json"), &Pending {
                status,
                sdk_before: Some(old.clone()),
                agent_before: Some(old.join("cozy-machine")),
            }).unwrap();
            Self { root, paths, old, new }
        }
        fn state(&self, state: &str) {
            let mut status: Status = serde_json::from_slice(&fs::read(self.paths.update("status.json")).unwrap()).unwrap();
            status.enter(state);
            write_json(&self.paths.update("status.json"), &status).unwrap();
        }
        fn links(&self, dir: &Path) {
            assert_eq!(fs::read_link(self.paths.current_sdk_link()).unwrap(), dir);
            assert_eq!(fs::read_link(self.paths.current_agent_link()).unwrap(), dir.join("cozy-machine"));
        }
    }
    impl Drop for Activation {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.root); }
    }

    #[test]
    fn every_unpublished_activation_boundary_rolls_back_before_service_selection() {
        // Process death after pending, after the SDK link, or after both links but before the
        // durable starting marker: none may be reported as successfully activated.
        for changed_links in 0..=2 {
            let transaction = Activation::prepare();
            if changed_links >= 1 {
                relink(&transaction.paths.current_sdk_link(), Some(&transaction.new)).unwrap();
            }
            if changed_links >= 2 {
                relink(&transaction.paths.current_agent_link(), Some(&transaction.new.join("cozy-machine"))).unwrap();
            }
            let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None).unwrap();
            assert!(updates.commit().is_err(), "unpublished state committed");
            assert!(recover_activation(&transaction.paths).unwrap());
            transaction.links(&transaction.old);
            assert!(!transaction.paths.update("pending.json").exists());
            let recovered = Updates::open(transaction.paths.clone(), Box::new(|| true), None).unwrap();
            assert_eq!(recovered.update("transaction").unwrap().state, "rolled_back");
            assert!(!recover_activation(&transaction.paths).unwrap());
        }
    }

    #[test]
    fn published_activation_waits_for_readiness_and_committed_cleanup_keeps_it() {
        let transaction = Activation::prepare();
        relink(&transaction.paths.current_sdk_link(), Some(&transaction.new)).unwrap();
        relink(&transaction.paths.current_agent_link(), Some(&transaction.new.join("cozy-machine"))).unwrap();
        transaction.state("starting");
        assert!(!recover_activation(&transaction.paths).unwrap());
        assert!(transaction.paths.update("pending.json").exists());
        transaction.links(&transaction.new);
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None).unwrap();
        // Keep the old pending bytes to simulate death after writing succeeded, before removal.
        let pending = fs::read(transaction.paths.update("pending.json")).unwrap();
        updates.commit().unwrap();
        assert_eq!(updates.update("transaction").unwrap().state, "succeeded");
        fs::write(transaction.paths.update("pending.json"), pending).unwrap();
        assert!(!recover_activation(&transaction.paths).unwrap());
        transaction.links(&transaction.new);
        assert!(!transaction.paths.update("pending.json").exists());
    }

    #[test]
    fn rolled_back_cleanup_can_resume_without_changing_the_outcome() {
        let transaction = Activation::prepare();
        let pending = fs::read(transaction.paths.update("pending.json")).unwrap();
        rollback_pending(&transaction.paths, "candidate could not start").unwrap();
        let settled = fs::read(transaction.paths.update("status.json")).unwrap();
        fs::write(transaction.paths.update("pending.json"), pending).unwrap();
        assert!(recover_activation(&transaction.paths).unwrap());
        transaction.links(&transaction.old);
        let expected: Status = serde_json::from_slice(&settled).unwrap();
        let actual: Status = serde_json::from_slice(&fs::read(transaction.paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(actual.state, expected.state);
        assert_eq!(actual.history, expected.history);
        assert_eq!(actual.error, expected.error);
    }

    #[test]
    fn interruption_during_rollback_resumes_the_same_decision() {
        for restored_links in 0..=1 {
            let transaction = Activation::prepare();
            relink(&transaction.paths.current_sdk_link(), Some(&transaction.new)).unwrap();
            relink(&transaction.paths.current_agent_link(), Some(&transaction.new.join("cozy-machine"))).unwrap();
            transaction.state("starting");
            transaction.state("rolled_back");
            if restored_links == 1 {
                relink(&transaction.paths.current_sdk_link(), Some(&transaction.old)).unwrap();
            }
            assert!(recover_activation(&transaction.paths).unwrap());
            transaction.links(&transaction.old);
            assert!(!transaction.paths.update("pending.json").exists());
        }
    }

    #[test]
    fn unreadable_transaction_files_do_not_mean_no_pending_update() {
        let transaction = Activation::prepare();
        fs::write(transaction.paths.update("status.json"), b"truncated").unwrap();
        assert!(recover_activation(&transaction.paths).is_err());
        transaction.links(&transaction.old);
        fs::remove_file(transaction.paths.update("pending.json")).unwrap();
        fs::create_dir(transaction.paths.update("pending.json")).unwrap();
        assert!(recover_activation(&transaction.paths).is_err());
        assert!(rollback_pending(&transaction.paths, "test").is_err());
    }

    #[test]
    fn committed_success_is_visible_even_when_pending_cleanup_fails() {
        let transaction = Activation::prepare();
        transaction.state("starting");
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None).unwrap();
        let mut succeeded = updates.update("transaction").unwrap();
        succeeded.enter("succeeded");
        // Fail the actual cleanup operation after the commit decision has already been read.
        fs::remove_file(transaction.paths.update("pending.json")).unwrap();
        fs::create_dir(transaction.paths.update("pending.json")).unwrap();
        assert!(updates.commit_status(succeeded).is_err());
        assert_eq!(updates.update("transaction").unwrap().state, "succeeded");
        let persisted: Status = serde_json::from_slice(&fs::read(transaction.paths.update("status.json")).unwrap()).unwrap();
        assert_eq!(persisted.state, "succeeded");
        let next = Request {
            operation: "next".into(), agent: "explicit".into(), pin: None,
            runtime: Some(Choice { version: "0.2.0".into(), ..Default::default() }),
            tensorfs: None,
        };
        let rejected = updates.request(next, |_| panic!("unsettled cleanup started an update")).unwrap_err();
        assert_eq!(rejected.0, 500);
        assert_eq!(updates.update("transaction").unwrap().state, "succeeded");
    }

    #[test]
    fn a_failed_status_write_does_not_advance_the_in_memory_state() {
        let transaction = Activation::prepare();
        let updates = Updates::open(transaction.paths.clone(), Box::new(|| true), None).unwrap();
        fs::remove_file(transaction.paths.update("status.json")).unwrap();
        fs::create_dir(transaction.paths.update("status.json")).unwrap();
        assert!(updates.set(|s| s.enter("starting")).is_err());
        assert_eq!(updates.update("transaction").unwrap().state, "installing");
    }

}
