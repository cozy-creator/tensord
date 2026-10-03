//! The machine's Python client, embedded at build time (build.rs packs python/ as a wheel).
//! Package environments and the installer helper take it from the binary, so an image or a
//! local install ships only the machine, uv and the executor SDK wheels.
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const WHEEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/client.whl"));
const WHEEL_NAME: &str = env!("COZY_CLIENT_WHEEL_NAME");

/// The root's own uv (an image's, or the one a local install links), else uv on PATH.
pub fn uv(root: &Path) -> PathBuf {
    let own = root.join("usr/local/bin/uv");
    if own.exists() {
        own
    } else {
        "uv".into()
    }
}

/// Writes the embedded wheel under `engine/client` when this build's differs, and answers it.
pub fn wheel(engine: &Path) -> io::Result<PathBuf> {
    let path = engine.join("client").join(WHEEL_NAME);
    if fs::read(&path).ok().as_deref() != Some(WHEEL) {
        fs::create_dir_all(engine.join("client"))?;
        super::identity::write_atomic(&path, WHEEL, 0o644)?;
    }
    Ok(path)
}

/// The installer helper: an environment over Python 3.12 holding the client with its installer
/// extra, made with uv at `engine/helper` and made again when the embedded client changes.
pub fn helper(engine: &Path, uv: &Path, wheel: &Path) -> io::Result<PathBuf> {
    let dir = engine.join("helper");
    let (python, marker) = (dir.join("bin/python"), dir.join(".client-sha256"));
    let digest = format!("{:x}", Sha256::digest(WHEEL));
    if python.exists() && fs::read_to_string(&marker).ok().as_deref() == Some(digest.as_str()) {
        return Ok(python);
    }
    if dir.exists() {
        fs::remove_dir_all(&dir)?;
    }
    let run = |command: &mut Command| -> io::Result<()> {
        let output = command.output()?;
        if output.status.success() {
            return Ok(());
        }
        Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    };
    run(Command::new(uv)
        .args(["venv", "--no-project", "--no-config", "--python", "3.12"])
        .arg(&dir))?;
    run(Command::new(uv)
        .args(["pip", "install", "--no-config", "--python"])
        .arg(&python)
        .arg(format!("{}[installer]", wheel.display())))?;
    fs::write(&marker, digest)?;
    Ok(python)
}
