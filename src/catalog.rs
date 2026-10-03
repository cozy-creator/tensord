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
}
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
}
impl HeldGeneration {
    pub fn retention(&self) -> Arc<File> {
        self.hold.clone()
    }
    pub fn invocation(&self, entrypoint: &str, input: serde_json::Value) -> io::Result<Invocation> {
        if entrypoint.is_empty() || entrypoint.starts_with('_') || entrypoint.contains('/') {
            return Err(invalid(
                "entrypoint must name a public package registration",
            ));
        }
        Ok(Invocation {
            package: self.record.package.clone(),
            generation: self.record.identity.clone(),
            module: self.record.application.clone(),
            entrypoint: entrypoint.into(),
            input,
            attention_kernel: String::new(),
        })
    }
    pub fn runner(&self) -> RunnerConfig {
        RunnerConfig {
            python: self.record.python.clone(),
            module: "cozy_machine_client.runner".into(),
            import_paths: vec![],
            generation_hold: Some(self.hold.clone()),
        }
    }
}
