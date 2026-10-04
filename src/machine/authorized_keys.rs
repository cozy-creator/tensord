//! A computer's machine admits the keys in `<root>/authorized_keys`, read as sshd reads its
//! file: one `ssh-ed25519 <base64> [comment]` per line, `#` comments and blank lines skipped,
//! a line it cannot read reported and skipped. A key appended or deleted by hand applies at
//! the next look (each second); a deleted key's open streams end then. Every key is the
//! machine's owner, as every key in an account's authorized_keys is that account.
use crate::api::auth::Keys;
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use std::{
    fs, io,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    time::Duration,
};

pub const FILE: &str = "authorized_keys";
const KIND: &str = "ssh-ed25519";

/// The file's keys and a description of each line it skipped.
pub fn parse(text: &str) -> (Vec<VerifyingKey>, Vec<String>) {
    let (mut keys, mut skipped) = (Vec::new(), Vec::new());
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next().map(key)) {
            (Some(KIND), Some(Some(found))) => {
                if !keys.contains(&found) {
                    keys.push(found)
                }
            }
            _ => skipped.push(format!("line {}: not an {KIND} public key", number + 1)),
        }
    }
    (keys, skipped)
}

/// The OpenSSH wire blob: string "ssh-ed25519", then the 32-byte key as a string.
fn key(encoded: &str) -> Option<VerifyingKey> {
    let blob = STANDARD.decode(encoded).ok()?;
    let mut rest = blob.as_slice();
    let mut field = || {
        let (length, tail) = rest.split_first_chunk::<4>()?;
        let length = u32::from_be_bytes(*length) as usize;
        let (value, tail) = (tail.get(..length)?, tail.get(length..)?);
        rest = tail;
        Some(value)
    };
    if field()? != KIND.as_bytes() {
        return None;
    }
    let raw: [u8; 32] = field()?.try_into().ok()?;
    if !rest.is_empty() {
        return None;
    }
    VerifyingKey::from_bytes(&raw).ok()
}

/// One public key as an `authorized_keys` line.
pub fn line(key: &VerifyingKey, comment: &str) -> String {
    let mut blob = Vec::with_capacity(51);
    for field in [KIND.as_bytes(), key.as_bytes().as_slice()] {
        blob.extend_from_slice(&(field.len() as u32).to_be_bytes());
        blob.extend_from_slice(field);
    }
    format!("{KIND} {} {comment}", STANDARD.encode(blob))
        .trim_end()
        .to_owned()
}

/// Adds a launcher's keys to `<root>/authorized_keys` where it lacks them, so a launcher that
/// grants keys in its environment reaches the machine it starts. Every other line is kept.
pub fn grant(root: &Path, keys: &[VerifyingKey]) -> io::Result<()> {
    let path = root.join(FILE);
    let mut text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error),
    };
    let held = parse(&text).0;
    let missing: Vec<_> = keys.iter().filter(|k| !held.contains(k)).collect();
    if missing.is_empty() {
        return Ok(());
    }
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    for key in missing {
        text.push_str(&line(key, "launcher"));
        text.push('\n');
    }
    let staged = root.join(".authorized_keys.new");
    fs::write(&staged, text)?;
    fs::set_permissions(&staged, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    fs::rename(staged, path)
}

/// The keys `<root>/authorized_keys` admits now, kept current by a watcher thread. An absent
/// file admits nobody, as with sshd.
pub fn watch(root: &Path) -> io::Result<Keys> {
    let (path, keys) = (root.join(FILE), Keys::owner(vec![]));
    let seen = read(&path, None, &keys)?;
    std::thread::Builder::new()
        .name("authorized-keys".into())
        .spawn({
            let keys = keys.clone();
            move || look(path, seen, keys)
        })?;
    Ok(keys)
}

/// The file's modification time, inode and length; None while it is absent.
type Seen = Option<(i64, i64, u64, u64)>;

fn look(path: PathBuf, mut seen: Seen, keys: Keys) {
    loop {
        std::thread::sleep(Duration::from_secs(1));
        match read(&path, Some(seen), &keys) {
            Ok(now) => seen = now,
            Err(error) => eprintln!("cozy-machine: {}: {error}", path.display()),
        }
    }
}

/// Re-reads the file unless it is as `seen` last time; returns what it saw.
fn read(path: &Path, seen: Option<Seen>, keys: &Keys) -> io::Result<Seen> {
    let now = match fs::metadata(path) {
        Ok(meta) => Some((meta.mtime(), meta.mtime_nsec(), meta.ino(), meta.len())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    if seen == Some(now) {
        return Ok(now);
    }
    let text = match now {
        Some(_) => fs::read_to_string(path)?,
        None => String::new(),
    };
    let (found, skipped) = parse(&text);
    for problem in skipped {
        eprintln!("cozy-machine: {}: {problem}", path.display());
    }
    keys.set(found);
    Ok(now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn reads_openssh_lines_and_skips_the_rest() {
        let a = SigningKey::from_bytes(&[7; 32]).verifying_key();
        let b = SigningKey::from_bytes(&[9; 32]).verifying_key();
        let text = format!(
            "# devices\n{}\n\n{}\nssh-rsa AAAAB3Nza other\nssh-ed25519 !!!\n{}\n",
            line(&a, "paul@laptop"),
            line(&b, ""),
            line(&a, "again"),
        );
        let (keys, skipped) = parse(&text);
        assert_eq!(keys, vec![a, b]);
        assert_eq!(skipped.len(), 2);
    }

    #[test]
    fn a_launchers_keys_join_the_file_and_other_lines_stay() {
        let root = std::env::temp_dir().join(format!("authorized-keys-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let (a, b) = (
            SigningKey::from_bytes(&[1; 32]).verifying_key(),
            SigningKey::from_bytes(&[2; 32]).verifying_key(),
        );
        grant(&root, &[a]).unwrap();
        let first = fs::read_to_string(root.join(FILE)).unwrap();
        grant(&root, &[a]).unwrap();
        assert_eq!(fs::read_to_string(root.join(FILE)).unwrap(), first);
        fs::write(root.join(FILE), format!("# mine\n{}", line(&a, "by hand"))).unwrap();
        grant(&root, &[b, a]).unwrap();
        let text = fs::read_to_string(root.join(FILE)).unwrap();
        assert!(text.starts_with("# mine\n") && text.contains("by hand\n"));
        assert_eq!(parse(&text).0, vec![a, b]);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn matches_ssh_keygen_spelling() {
        // OpenSSH's spelling of the all-zero-seed key (cryptography's OpenSSH encoding agrees).
        let key = SigningKey::from_bytes(&[0; 32]).verifying_key();
        assert_eq!(
            line(&key, "c"),
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDtqJ7zOtqQtYqOo0CpvDXNlMhV3HeJDpjrASKGLWdop c"
        );
    }
}
