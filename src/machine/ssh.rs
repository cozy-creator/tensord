//! A development rental's SSH maintenance, as the Go agent serves it: the granted key for root
//! and the image's sshd config. Started by the stable parent, so it outlives service restarts.
use std::{
    fs,
    io::{self, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    process::Command,
};

pub fn start(public_key: &str) -> io::Result<()> {
    if public_key.len() > 8192 || public_key.contains(['\0', '\r', '\n']) {
        return Err(io::Error::other(
            "PUBLIC_KEY must be one SSH public key of at most 8192 bytes",
        ));
    }
    fs::create_dir_all("/root/.ssh")?;
    fs::set_permissions("/root/.ssh", fs::Permissions::from_mode(0o700))?;
    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open("/root/.ssh/authorized_keys")?
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
        .args(["-D", "-e", "-f", "/opt/cozy/dev/sshd_config"])
        .spawn()?;
    Ok(())
}
