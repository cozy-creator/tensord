//! Trusted immutable environment selection, independent of request-supplied paths.
use crate::execution::RunnerConfig;
use crate::journal::Invocation;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Generation {
    pub identity: String,
    pub package: String,
    pub version: String,
    pub application: String,
    pub python: PathBuf,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    pub interface: serde_json::Value,
    /// Empty when the CPU runner's SDK adapter imported at install; else the reason.
    #[serde(default)]
    pub cpu_bridge: String,
    /// The package's own installed files, hashed; empty when its installer recorded none.
    #[serde(default)]
    pub source_digest: String,
    /// Other packages' Apps this environment holds as dependencies: a job calls them here.
    #[serde(default)]
    pub callees: Vec<Callee>,
}
/// Another package's App installed in an environment: its calls run as child runs of its own
/// package, from the caller's environment.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Callee {
    pub distribution: String,
    pub version: String,
    pub application: String,
    pub interface: serde_json::Value,
    #[serde(default)]
    pub source_digest: String,
    #[serde(default)]
    pub package: String,
}
impl Generation {
    /// The package an application of this environment belongs to (the root's or a callee's),
    /// its release and its interface.
    pub fn app(&self, application: &str) -> Option<(&str, &str, &serde_json::Value)> {
        if application.is_empty() || application == self.application {
            return Some((&self.package, &self.version, &self.interface));
        }
        self.callees
            .iter()
            .find(|c| c.application == application)
            .map(|c| (if c.package.is_empty() { c.distribution.as_str() } else { c.package.as_str() }, c.version.as_str(), &c.interface))
    }
}
#[derive(Clone)]
pub struct HeldGeneration {
    pub record: Generation,
    hold: Arc<File>,
}
#[derive(Clone)]
pub struct Catalog {
    root: PathBuf,
}
fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail)
}
impl Catalog {
    pub fn new(root: &Path) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        Ok(Self {
            root: root.canonicalize()?,
        })
    }
    pub fn resolve(&self, identity: &str) -> io::Result<HeldGeneration> {
        if identity.len() != 32
            || !identity
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(invalid(
                "generation must name a published 32-hex installation",
            ));
        }
        let directory = self.root.join(identity);
        if directory.symlink_metadata()?.file_type().is_symlink()
            || directory.canonicalize()? != directory
        {
            return Err(invalid(
                "generation directory must not be an alias or symlink",
            ));
        }
        let hold = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory.join(".hold"))?;
        FileExt::lock_shared(&hold)?;
        // Last use, for least-recently-used eviction of unheld generations.
        // SAFETY: a live descriptor; a null time sets both times to now.
        unsafe { libc::futimens(std::os::fd::AsRawFd::as_raw_fd(&hold), std::ptr::null()) };
        let manifest = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory.join("generation.json"))?;
        let record: Generation = serde_json::from_reader(manifest).map_err(io::Error::other)?;
        if record.identity != identity {
            return Err(invalid("generation identity differs from its installation"));
        }
        let expected = directory.join("env/bin/python");
        // The venv's interpreter may be a symlink into uv's immutable interpreter store.
        if record.python != expected || !record.python.is_file() {
            return Err(invalid(
                "generation interpreter does not name its held environment",
            ));
        }
        Ok(HeldGeneration {
            record,
            hold: Arc::new(hold),
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Every published installation, the most recently installed first.
    pub fn installed(&self) -> Vec<HeldGeneration> {
        let mut found: Vec<_> = fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let installed = entry
                    .path()
                    .join("generation.json")
                    .metadata()
                    .ok()?
                    .modified()
                    .ok()?;
                Some((installed, self.resolve(&name).ok()?))
            })
            .collect();
        found.sort_by(|a, b| b.0.cmp(&a.0));
        found.into_iter().map(|(_, held)| held).collect()
    }
}
impl HeldGeneration {
    /// One App's executor view, with the environment and lifetime hold unchanged.
    pub fn application(&self, application: &str) -> io::Result<Self> {
        if application.is_empty() || application == self.record.application {
            return Ok(self.clone());
        }
        let callee = self.record.callees.iter().find(|c| c.application == application)
            .ok_or_else(|| invalid("this environment holds no such application"))?;
        let mut held = self.clone();
        held.record.package = match callee.package.as_str() {
            "" => callee.distribution.clone(),
            _ => callee.package.clone(),
        };
        held.record.version = callee.version.clone();
        held.record.application = callee.application.clone();
        held.record.interface = callee.interface.clone();
        held.record.source_digest = callee.source_digest.clone();
        Ok(held)
    }
    pub fn retention(&self) -> Arc<File> {
        self.hold.clone()
    }
    pub fn invocation(&self, entrypoint: &str, input: serde_json::Value) -> io::Result<Invocation> {
        self.invocation_of("", entrypoint, input)
    }
    /// A call of `application` (empty: the root's; else a callee's) in this environment.
    pub fn invocation_of(
        &self,
        application: &str,
        entrypoint: &str,
        input: serde_json::Value,
    ) -> io::Result<Invocation> {
        if entrypoint.is_empty() || entrypoint.starts_with('_') || entrypoint.contains('/') {
            return Err(invalid(
                "entrypoint must name a public package registration",
            ));
        }
        let (package, _, _) = self
            .record
            .app(application)
            .ok_or_else(|| invalid("this environment holds no such application"))?;
        Ok(Invocation {
            package: package.into(),
            generation: self.record.identity.clone(),
            module: match application {
                "" => self.record.application.clone(),
                callee => callee.into(),
            },
            entrypoint: entrypoint.into(),
            input,
            attention_kernel: String::new(),
            inputs: vec![],
            ..Default::default()
        })
    }
    pub fn runner(&self) -> RunnerConfig {
        RunnerConfig {
            python: self.record.python.clone(),
            module: "cozy_machine_client.runner".into(),
            import_paths: vec![],
            generation_hold: Some(self.hold.clone()),
            unavailable: (!self.record.cpu_bridge.is_empty()).then(|| {
                format!(
                    "CPU runner adapter unavailable in this environment: {}",
                    self.record.cpu_bridge
                )
            }),
        }
    }
}
