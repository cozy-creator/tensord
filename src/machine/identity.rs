//! The machine lifetime: its boot id and TLS leaf, persisted with the root so a restarted
//! service is the same machine. Files and formats are the Go agent's (`layout.go`).
use super::grant::Layout;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::Path,
};

/// The one DNS name of every machine leaf; clients pin the leaf itself.
pub const SERVER_NAME: &str = "cozy-worker";

pub struct Lifetime {
    pub boot_id: String,
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
}

/// Mints or reopens this root's lifetime. A root booted by the Go agent keeps its boot id
/// and leaf; so does a later Go agent on a root this service booted.
pub fn open(layout: &Layout) -> io::Result<Lifetime> {
    for dir in [&layout.bootstrap, &layout.state] {
        fs::create_dir_all(dir)?;
    }
    let path = layout.state.join("boot-id");
    let boot_id = match fs::read_to_string(&path) {
        Ok(text) if !text.trim().is_empty() => text.trim().to_owned(),
        Ok(_) => return Err(invalid("the persisted boot id is empty")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let inherited =
                fs::read_to_string(layout.bootstrap.join("pod-boot-id")).unwrap_or_default();
            let boot_id = match inherited.trim() {
                "" => URL_SAFE_NO_PAD.encode(random::<32>()?),
                kept => kept.to_owned(),
            };
            write_atomic(&path, boot_id.as_bytes(), 0o400)?;
            boot_id
        }
        Err(e) => return Err(e),
    };
    if URL_SAFE_NO_PAD.decode(&boot_id).map(|b| b.len()) != Ok(32) {
        return Err(invalid(
            "the persisted boot id is not unpadded base64url for 32 bytes",
        ));
    }
    write_atomic(
        &layout.bootstrap.join("pod-boot-id"),
        format!("{boot_id}\n").as_bytes(),
        0o444,
    )?;
    let (cert_path, key_path) = (
        layout.bootstrap.join("tls.crt"),
        layout.bootstrap.join("tls.key"),
    );
    if !cert_path.exists() {
        let (cert_pem, key_pem) = mint_leaf()?;
        write_atomic(&key_path, key_pem.as_bytes(), 0o400)?;
        write_atomic(&cert_path, cert_pem.as_bytes(), 0o444)?;
    }
    let cert_pem = fs::read_to_string(&cert_path)?;
    let key_pem = fs::read_to_string(&key_path)?;
    let certs: Vec<_> =
        rustls_pemfile::certs(&mut cert_pem.as_bytes()).collect::<Result<_, _>>()?;
    let [cert] = certs.as_slice() else {
        return Err(invalid("the machine leaf must be exactly one certificate"));
    };
    rustls_pemfile::private_key(&mut key_pem.as_bytes())?
        .ok_or_else(|| invalid("the machine leaf has no private key"))?;
    Ok(Lifetime {
        boot_id,
        cert_der: cert.as_ref().to_vec(),
        cert_pem,
        key_pem,
    })
}

/// The leaf the Hub admits: P-256, self-signed, exactly the DNS SAN `cozy-worker`, usable as
/// both TLS server and Tensorhub client, valid until year 9999 (clients pin the bytes).
fn mint_leaf() -> io::Result<(String, String)> {
    use rcgen::{
        CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, KeyPair,
        KeyUsagePurpose,
    };
    let key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(io::Error::other)?;
    let mut params = CertificateParams::new(vec![SERVER_NAME.into()]).map_err(io::Error::other)?;
    let mut name = DistinguishedName::new();
    name.push(DnType::CommonName, "cozy-machine");
    params.distinguished_name = name;
    params.not_before = time::OffsetDateTime::now_utc() - time::Duration::minutes(5);
    params.not_after = rcgen::date_time_ymd(9999, 12, 31);
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    let cert = params.self_signed(&key).map_err(io::Error::other)?;
    Ok((cert.pem(), key.serialize_pem()))
}

pub fn random<const N: usize>() -> io::Result<[u8; N]> {
    let mut out = [0; N];
    File::open("/dev/urandom")?.read_exact(&mut out)?;
    Ok(out)
}

pub fn write_atomic(target: &Path, data: &[u8], mode: u32) -> io::Result<()> {
    let dir = target
        .parent()
        .ok_or_else(|| invalid("a target needs a directory"))?;
    fs::create_dir_all(dir)?;
    let temp = dir.join(format!(".write-{}", URL_SAFE_NO_PAD.encode(random::<9>()?)));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        file.write_all(data)?;
        file.set_permissions(fs::Permissions::from_mode(mode))?;
        file.sync_all()?;
        fs::rename(&temp, target)?;
        File::open(dir)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifetime_survives_restart_and_adopts_a_go_booted_root() {
        let root = std::env::temp_dir().join(format!(
            "cozy-machine-identity-{}",
            URL_SAFE_NO_PAD.encode(random::<9>().unwrap())
        ));
        let layout = Layout::new(&root, None);
        let first = open(&layout).unwrap();
        let again = open(&layout).unwrap();
        assert_eq!(
            (first.boot_id.as_str(), &first.cert_der),
            (again.boot_id.as_str(), &again.cert_der)
        );
        let cert = x509_parser::parse_x509_certificate(&first.cert_der)
            .unwrap()
            .1;
        let names: Vec<_> = cert
            .subject_alternative_name()
            .unwrap()
            .unwrap()
            .value
            .general_names
            .iter()
            .map(|n| format!("{n}"))
            .collect();
        assert_eq!(names, ["DNSName(cozy-worker)"]);
        // A root the Go agent booted: its boot id is adopted, never replaced.
        fs::remove_file(layout.state.join("boot-id")).unwrap();
        let go = URL_SAFE_NO_PAD.encode([7u8; 32]);
        write_atomic(
            &layout.bootstrap.join("pod-boot-id"),
            format!("{go}\n").as_bytes(),
            0o444,
        )
        .unwrap();
        assert_eq!(open(&layout).unwrap().boot_id, go);
        fs::remove_dir_all(root).unwrap();
    }
}
