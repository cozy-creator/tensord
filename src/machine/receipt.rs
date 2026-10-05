//! Readiness: what this machine measured about itself, sealed once under the attempt's key and
//! carried by Status without a capability. The reader is Tensorhub `internal/podreadiness`.
use base64::{engine::general_purpose::STANDARD, Engine};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{
    collections::BTreeMap,
    fs, io,
    path::PathBuf,
    process::Command,
    sync::{Arc, Condvar, Mutex},
};

pub const DOMAIN: &[u8] = b"cozy.pod-readiness/1\0";
const MAX_ENVELOPE_BYTES: usize = 64 << 10;

#[derive(serde::Serialize, serde::Deserialize)]
struct Envelope {
    #[serde(with = "standard_base64")]
    payload: Vec<u8>,
    #[serde(default)]
    hmac_sha256: String,
}

/// This boot's one envelope. A retained envelope is verified with a replayed key and served
/// byte for byte: a provider restart may verify what was sealed, never sign again.
pub struct Readiness {
    path: Option<PathBuf>,
    state: Mutex<State>,
    proved: Condvar,
}
#[derive(Default)]
struct State {
    key: Option<Vec<u8>>,
    retained: Option<Vec<u8>>,
    sealed: Option<Vec<u8>>,
    proved: bool,
}

impl Readiness {
    /// `path` None keeps the envelope in memory (development front doors). `required` is a
    /// rental: with neither a key nor a retained envelope the boot cannot prove itself.
    pub fn open(
        path: Option<PathBuf>,
        key: Option<Vec<u8>>,
        required: bool,
    ) -> io::Result<Arc<Self>> {
        let mut state = State {
            key,
            ..State::default()
        };
        let retained = match &path {
            Some(path) => match fs::read(path) {
                Ok(raw) => Some(raw),
                Err(e) if e.kind() == io::ErrorKind::NotFound => None,
                Err(e) => return Err(e),
            },
            None => None,
        };
        match retained {
            Some(raw) => {
                let envelope = parse(&raw)?;
                if let Some(key) = &state.key {
                    if mac(key, &envelope.payload) != envelope.hmac_sha256 {
                        return Err(invalid("the retained readiness envelope does not verify under the granted key"));
                    }
                }
                state.key = None;
                state.retained = Some(envelope.payload);
                state.sealed = Some(raw);
            }
            None if required && state.key.is_none() => {
                return Err(invalid("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL is required when no readiness envelope is retained"))
            }
            None => (),
        }
        Ok(Arc::new(Self {
            path,
            state: Mutex::new(state),
            proved: Condvar::new(),
        }))
    }

    /// A persisted receipt belongs to a real machine, which measures its GPUs. Development
    /// front doors report none rather than query the driver of the computer they run on.
    pub fn measures_gpus(&self) -> bool {
        self.path.is_some()
    }

    /// The sealed envelope, or None before readiness.
    pub fn envelope(&self) -> Option<Vec<u8>> {
        self.state.lock().unwrap().sealed.clone()
    }

    /// The GPUs this boot's sealed receipt names.
    pub fn gpus(&self) -> Vec<Gpu> {
        #[derive(serde::Deserialize)]
        struct Gpus {
            #[serde(default)]
            runtime_gpus: Vec<Gpu>,
        }
        let sealed = self.state.lock().unwrap().sealed.clone();
        sealed
            .and_then(|raw| parse(&raw).ok())
            .and_then(|e| serde_json::from_slice::<Gpus>(&e.payload).ok())
            .map(|g| g.runtime_gpus)
            .unwrap_or_default()
    }

    /// The payload a retained envelope or this boot's seal attested.
    pub fn attested(&self) -> Option<Vec<u8>> {
        self.state.lock().unwrap().retained.clone()
    }

    /// Seals this boot's payload once; true when this call signed it. A later process of the
    /// same boot must measure the same boot, leaf and GPU set; the original bytes stay served.
    pub fn seal(&self, payload: Vec<u8>) -> io::Result<bool> {
        let mut state = self.state.lock().unwrap();
        if let Some(retained) = &state.retained {
            same_attestation(retained, &payload)?;
            state.proved = true;
            self.proved.notify_all();
            return Ok(false);
        }
        let hmac_sha256 = match &state.key {
            Some(key) => mac(key, &payload),
            None if self.path.is_none() => String::new(),
            None => {
                return Err(invalid(
                    "the readiness key was spent; this boot signs nothing again",
                ))
            }
        };
        let raw = serde_json::to_vec(&Envelope {
            payload: payload.clone(),
            hmac_sha256,
        })?;
        if raw.len() > MAX_ENVELOPE_BYTES {
            return Err(invalid("the readiness envelope exceeds 64 KiB"));
        }
        if let Some(path) = &self.path {
            super::identity::write_atomic(path, &raw, 0o444)?;
            if let Some(key) = state.key.as_mut() {
                key.fill(0);
            }
            state.key = None;
            state.retained = Some(payload);
        }
        state.sealed = Some(raw);
        state.proved = true;
        self.proved.notify_all();
        Ok(true)
    }

    pub fn proved(&self) -> bool {
        self.state.lock().unwrap().proved
    }

    /// Blocks until this process has proved readiness (sealed, or matched the retained seal).
    pub fn wait_proved(&self) {
        let state = self.state.lock().unwrap();
        drop(self.proved.wait_while(state, |s| !s.proved).unwrap());
    }
}

/// One accelerator as the Runtime and Hub name it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Gpu {
    pub device_index: u32,
    pub device_name: String,
    pub device_uuid: String,
    pub driver_version: String,
    pub memory_bytes: u64,
    pub pci_bus_id: String,
}

/// The driver's inventory through the Runtime's fixed query; this process never loads CUDA or
/// NVML. No `nvidia-smi` is a CPU machine; a driver that answers wrongly refuses the boot.
pub fn gpus() -> io::Result<Vec<Gpu>> {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    if visible.as_deref() == Some("") {
        return Ok(vec![]);
    }
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e),
        Ok(output) if !output.status.success() => {
            return Err(invalid(&format!(
                "nvidia-smi failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )))
        }
        Ok(output) => output,
    };
    let mut rows = parse_gpus(&String::from_utf8_lossy(&output.stdout))?;
    if let Some(visible) = visible {
        let named: Vec<&str> = visible.split(',').map(str::trim).collect();
        rows.retain(|g| {
            named.contains(&g.device_index.to_string().as_str())
                || named.contains(&g.device_uuid.as_str())
        });
    }
    Ok(rows)
}

fn parse_gpus(text: &str) -> io::Result<Vec<Gpu>> {
    let mut rows = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let row = match fields.as_slice() {
            [index, name, uuid, bus, memory, driver] if fields.iter().all(|f| !f.is_empty()) => {
                Gpu {
                    device_index: index
                        .parse()
                        .map_err(|_| invalid("nvidia-smi returned an invalid index"))?,
                    device_name: (*name).into(),
                    device_uuid: (*uuid).into(),
                    driver_version: (*driver).into(),
                    memory_bytes: memory
                        .parse::<u64>()
                        .map_err(|_| invalid("nvidia-smi returned invalid memory"))?
                        << 20,
                    pci_bus_id: (*bus).into(),
                }
            }
            _ => return Err(invalid("nvidia-smi returned an invalid row")),
        };
        rows.push(row);
    }
    rows.sort_by_key(|g| g.device_index);
    if rows.len() > 16
        || rows
            .windows(2)
            .any(|w| w[0].device_index == w[1].device_index)
    {
        return Err(invalid("GPU inventory is duplicate or oversized"));
    }
    Ok(rows)
}

/// The facts the Hub's readiness reader requires of a machine image, as measured here on
/// `cozy.machine.v1`, the one protocol this listener serves.
pub struct Measured<'a> {
    pub boot_id: &'a str,
    pub worker_port: u16,
    pub cert_der: &'a [u8],
    pub gpus: Vec<Gpu>,
    pub listener_bound: bool,
    pub foreign_credential_refused: bool,
    /// What it serves on `cozy.machine.v1`, as Status names it.
    pub capabilities: Vec<String>,
    /// The port `cozy/1` is served on; the Hub derives its pin from the leaf.
    pub webrtc_port: Option<u16>,
}
impl Measured<'_> {
    pub fn payload(&self) -> Vec<u8> {
        let mut value = serde_json::json!({
            "pod_boot_id": self.boot_id,
            "worker_internal_port": self.worker_port,
            "worker_protocol": "cozy.machine.v1",
            "worker_listener_bound": self.listener_bound,
            "worker_foreign_credential_refused": self.foreign_credential_refused,
            "tls_certificate_der_base64": STANDARD.encode(self.cert_der),
            "runtime_gpus": self.gpus,
            "machine_version": env!("CARGO_PKG_VERSION"),
            "machine_capabilities": self.capabilities,
        });
        if let Some(port) = self.webrtc_port {
            value["webrtc"] = serde_json::json!({ "port": port });
        }
        serde_json::to_vec(&value).expect("a JSON value serializes")
    }
}

/// The attested facts, as the Go agent and Runtime compare them across a same-boot restart.
fn same_attestation(before: &[u8], after: &[u8]) -> io::Result<()> {
    fn facts(raw: &[u8]) -> io::Result<BTreeMap<&'static str, String>> {
        #[derive(serde::Deserialize, Default)]
        #[serde(default)]
        struct Auth {
            control_public_key_ed25519_b64url: String,
            media_token_sha256: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Uuid {
            device_uuid: String,
        }
        #[derive(serde::Deserialize)]
        struct Attested {
            pod_boot_id: String,
            tls_certificate_der_base64: String,
            #[serde(default)]
            observed_record_owner_auth: Auth,
            #[serde(default)]
            runtime_gpus: Vec<Uuid>,
        }
        let a: Attested = serde_json::from_slice(raw)?;
        let mut tokens = a.observed_record_owner_auth.media_token_sha256;
        tokens.sort();
        let mut gpus: Vec<_> = a.runtime_gpus.into_iter().map(|g| g.device_uuid).collect();
        gpus.sort();
        Ok(BTreeMap::from([
            ("boot", a.pod_boot_id),
            ("leaf", a.tls_certificate_der_base64),
            (
                "owner",
                a.observed_record_owner_auth
                    .control_public_key_ed25519_b64url,
            ),
            ("tokens", tokens.join(",")),
            ("gpus", gpus.join(",")),
        ]))
    }
    if facts(before)? != facts(after)? {
        return Err(invalid(
            "this boot attested a different boot, leaf, owner or GPU set before restarting",
        ));
    }
    Ok(())
}

fn parse(raw: &[u8]) -> io::Result<Envelope> {
    match serde_json::from_slice::<Envelope>(raw) {
        Ok(envelope) if raw.len() <= MAX_ENVELOPE_BYTES && !envelope.payload.is_empty() => {
            Ok(envelope)
        }
        _ => Err(invalid("the retained readiness envelope is unreadable")),
    }
}

fn mac(key: &[u8], payload: &[u8]) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC admits any key size");
    mac.update(DOMAIN);
    mac.update(payload);
    tensorfs_core::sha256::hex(&mac.finalize().into_bytes())
}

/// The boot id the payload names.
pub fn boot_id(payload: &[u8]) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Boot {
        pod_boot_id: String,
    }
    serde_json::from_slice::<Boot>(payload)
        .ok()
        .map(|b| b.pod_boot_id)
}

mod standard_base64 {
    use base64::{engine::general_purpose::STANDARD, Engine};
    pub fn serialize<S: serde::Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(bytes))
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let text: String = serde::Deserialize::deserialize(d)?;
        STANDARD.decode(text).map_err(serde::de::Error::custom)
    }
}

fn invalid(detail: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(boot: &str, gpu: &str) -> Vec<u8> {
        Measured {
            boot_id: boot,
            worker_port: 8443,
            cert_der: b"leaf",
            gpus: vec![Gpu {
                device_index: 0,
                device_name: "NVIDIA A40".into(),
                device_uuid: gpu.into(),
                driver_version: "580.65.06".into(),
                memory_bytes: 1 << 30,
                pci_bus_id: "00000000:01:00.0".into(),
            }],
            listener_bound: true,
            foreign_credential_refused: true,
            capabilities: vec![],
            webrtc_port: None,
        }
        .payload()
    }

    #[test]
    fn one_seal_per_boot_and_a_restart_only_verifies() {
        let dir = std::env::temp_dir().join(format!("cozy-readiness-{}", std::process::id()));
        let path = dir.join("readiness-envelope.json");
        let _ = fs::remove_dir_all(&dir);
        assert!(
            Readiness::open(Some(path.clone()), None, true).is_err(),
            "a rental needs its key"
        );
        let key = vec![9u8; 32];
        let first = Readiness::open(Some(path.clone()), Some(key.clone()), true).unwrap();
        assert!(first.envelope().is_none());
        assert!(first.seal(payload("boot", "GPU-a")).unwrap());
        let sealed = first.envelope().unwrap();
        let envelope = parse(&sealed).unwrap();
        assert_eq!(envelope.hmac_sha256, mac(&key, &envelope.payload));
        // The provider replays the key on a container restart: verify, keep the bytes.
        let again = Readiness::open(Some(path.clone()), Some(key), true).unwrap();
        assert_eq!(again.envelope().unwrap(), sealed);
        assert!(!again.seal(payload("boot", "GPU-a")).unwrap());
        assert!(
            again.seal(payload("boot", "GPU-b")).is_err(),
            "a different GPU set is another boot"
        );
        assert_eq!(again.envelope().unwrap(), sealed);
        assert!(
            Readiness::open(Some(path), Some(vec![8; 32]), true).is_err(),
            "a foreign key refuses"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn nvidia_smi_rows_parse_as_the_runtime_reads_them() {
        let rows = parse_gpus("1, NVIDIA A40, GPU-b, 00000000:02:00.0, 46068, 580.65.06\n0, NVIDIA A40, GPU-a, 00000000:01:00.0, 46068, 580.65.06\n").unwrap();
        assert_eq!(
            rows.iter().map(|g| g.device_index).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(rows[0].memory_bytes, 46068 << 20);
        assert!(parse_gpus("0, NVIDIA A40, , 00:01.0, 1, 1").is_err());
    }
}
