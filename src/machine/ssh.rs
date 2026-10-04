//! A development rental's SSH maintenance: the granted key for root and the image's sshd
//! config. Started by the stable parent, so it outlives service restarts.
//!
//! The key is kept outside `/root/.ssh`: a provider may write its own key there after the
//! container starts (vast.ai replaces `authorized_keys` about a second in).
use std::{
    fs,
    io::{self, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    process::Command,
};

const KEY_DIR: &str = "/run/cozy-dev";
const KEYS: &str = "/run/cozy-dev/authorized_keys";

pub fn start(public_key: &str) -> io::Result<()> {
    if public_key.len() > 8192 || public_key.contains(['\0', '\r', '\n']) {
        return Err(io::Error::other(
            "the SSH key must be one public key of at most 8192 bytes",
        ));
    }
    fs::create_dir_all(KEY_DIR)?;
    fs::set_permissions(KEY_DIR, fs::Permissions::from_mode(0o700))?;
    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(KEYS)?
        .write_all(format!("{public_key}\n").as_bytes())?;
    fs::create_dir_all("/run/sshd")?;
    let keys = Command::new("/usr/bin/ssh-keygen").arg("-A").output()?;
    if !keys.status.success() {
        return Err(io::Error::other(format!(
            "ssh host keys: {}",
            String::from_utf8_lossy(&keys.stderr)
        )));
    }
    Command::new("/usr/sbin/sshd")
        .args(["-D", "-e", "-f", "/opt/cozy/dev/sshd_config", "-o"])
        .arg(format!("AuthorizedKeysFile={KEYS}"))
        .spawn()?;
    Ok(())
}
