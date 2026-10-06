use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use cozy_machine::api::{
    capability::{self, Grant, MACHINE},
    identity::{AuthorizedKeys, MachineConfig, ReadinessSecret},
    MachineIdentity,
};
use ed25519_dalek::SigningKey;
use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf};

struct Area(PathBuf);
impl Area {
    fn new() -> Self {
        let id = fs::read_to_string("/proc/sys/kernel/random/uuid").unwrap();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/identity-tests")
            .join(id.trim());
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn config(&self, key: &SigningKey) -> MachineConfig {
        let keys = self.0.join("keys.json");
        private(
            &keys,
            &AuthorizedKeys {
                keys: vec![URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes())],
            },
        );
        let secret = self.0.join("readiness.json");
        private(
            &secret,
            &ReadinessSecret {
                key_b64url: URL_SAFE_NO_PAD.encode([42; 32]),
            },
        );
        MachineConfig {
            worker_id: "identity-test".into(),
            identity_directory: self.0.join("identity"),
            authorized_keys_file: keys,
            readiness_hmac_key_file: secret,
        }
    }
}
impl Drop for Area {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn private(path: &std::path::Path, value: &impl serde::Serialize) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn capability(identity: &MachineIdentity, key: &SigningKey) -> String {
    capability::mint(
        key,
        Grant {
            machine: identity.authority.worker_id.clone(),
            action: MACHINE.into(),
            expires: i64::MAX,
            ..Default::default()
        },
    )
}
fn allowed(identity: &MachineIdentity, token: &str) -> bool {
    capability::verify_signer(
        token,
        &identity.authority.worker_id,
        &identity.authority.keys.admitted(),
        1,
        MACHINE,
    )
    .is_ok()
}

#[test]
fn pinned_leaf_and_lifetime_survive_restart_and_keys_refresh() {
    let area = Area::new();
    let key = SigningKey::from_bytes(&[1; 32]);
    let config = area.config(&key);
    let journal = area.0.join("execution-journal-witness");
    fs::write(&journal, b"durable engine is owned elsewhere").unwrap();
    let first = MachineIdentity::retained(&config).unwrap();
    let before = capability(&first, &key);
    let next = MachineIdentity::retained(&config).unwrap();
    assert_eq!(first.cert_der, next.cert_der);
    assert_eq!(first.key_pem, next.key_pem);
    assert_eq!(first.authority.boot_id, next.authority.boot_id);
    assert_eq!(
        URL_SAFE_NO_PAD
            .decode(&next.authority.boot_id)
            .unwrap()
            .len(),
        32
    );
    assert!(allowed(&next, &before));
    assert!(allowed(&next, &capability(&next, &key)));
    let new_key = SigningKey::from_bytes(&[2; 32]);
    private(
        &config.authorized_keys_file,
        &AuthorizedKeys {
            keys: vec![URL_SAFE_NO_PAD.encode(new_key.verifying_key().as_bytes())],
        },
    );
    let rotated = MachineIdentity::retained(&config).unwrap();
    assert_eq!(first.cert_der, rotated.cert_der);
    assert!(!allowed(&rotated, &capability(&rotated, &key)));
    assert!(allowed(&rotated, &capability(&rotated, &new_key)));
    assert_eq!(
        fs::read(journal).unwrap(),
        b"durable engine is owned elsewhere"
    );
}

#[test]
fn unsafe_secret_modes_and_changed_worker_identity_preserve_existing_state() {
    let area = Area::new();
    let config = area.config(&SigningKey::from_bytes(&[1; 32]));
    let first = MachineIdentity::retained(&config).unwrap();
    let path = config.identity_directory.join("identity.json");
    let original = fs::read(&path).unwrap();
    let mut changed = config.clone();
    changed.worker_id = "different-worker".into();
    assert!(MachineIdentity::retained(&changed).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::set_permissions(
        &config.readiness_hmac_key_file,
        fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    assert!(MachineIdentity::retained(&config).is_err());
    fs::set_permissions(
        &config.readiness_hmac_key_file,
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert_eq!(
        MachineIdentity::retained(&config).unwrap().cert_der,
        first.cert_der
    );
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(MachineIdentity::retained(&config).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
}
