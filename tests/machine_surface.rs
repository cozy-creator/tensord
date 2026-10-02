//! The real `cozy-machine serve` process behind its TLS/gRPC API, called as the CLI calls it.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use cozy_machine::api::{auth::Authority, pb};
use ed25519_dalek::{Signer, SigningKey};
use std::{
    fs,
    io::BufReader,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

struct Machine {
    child: Child,
    root: PathBuf,
    client: pb::pod_host_client::PodHostClient<Channel>,
    claim: pb::Claim,
}

impl Machine {
    async fn start() -> Self {
        let root = std::env::temp_dir().join(format!("cm-surface-{}", uuid::Uuid::new_v4()));
        let config = root.join("config");
        fs::create_dir_all(&config).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let signer = SigningKey::from_bytes(&[33; 32]);
        let write = |name: &str, value: serde_json::Value| {
            let path = config.join(name);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        };
        write(
            "keys.json",
            serde_json::json!({"keys":[URL_SAFE_NO_PAD.encode(signer.verifying_key().to_bytes())]}),
        );
        write(
            "readiness.json",
            serde_json::json!({"key_b64url":URL_SAFE_NO_PAD.encode([7u8; 32])}),
        );
        write(
            "machine.json",
            serde_json::json!({"worker_id":"surface-test","identity_directory":"identity",
                "authorized_keys_file":"keys.json","readiness_hmac_key_file":"readiness.json"}),
        );
        let state = root.join("state");
        let child = Command::new(env!("CARGO_BIN_EXE_cozy-machine"))
            .args(["serve", "--state"])
            .arg(&state)
            .arg("--machine-config")
            .arg(config.join("machine.json"))
            .args(["--listen", "127.0.0.1:0", "--host-bytes", "0"])
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        // Test harness bound on a broken start; the product has no such limit.
        let until = Instant::now() + Duration::from_secs(60);
        let ready = loop {
            if let Ok(bytes) = fs::read(state.join("api-ready.json")) {
                break serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
            }
            assert!(
                Instant::now() < until,
                "machine did not publish api-ready.json"
            );
            std::thread::sleep(Duration::from_millis(50));
        };
        let pem = ready["cert_pem"].as_str().unwrap().to_owned();
        let der = rustls_pemfile::certs(&mut BufReader::new(pem.as_bytes()))
            .next()
            .unwrap()
            .unwrap();
        let authority = Authority {
            worker_id: ready["worker_id"].as_str().unwrap().into(),
            boot_id: ready["boot_id"].as_str().unwrap().into(),
            leaf_digest: tensorfs_core::sha256::digest(der.as_ref()),
            keys: vec![],
        };
        let claim = pb::Claim {
            worker_id: authority.worker_id.clone(),
            worker_boot_id: authority.boot_id.clone(),
            record_owner_epoch: 1,
            proof: signer
                .sign(&authority.transcript(1).unwrap())
                .to_bytes()
                .to_vec(),
            ..Default::default()
        };
        let channel =
            Endpoint::from_shared(format!("https://{}", ready["address"].as_str().unwrap()))
                .unwrap()
                .tls_config(
                    ClientTlsConfig::new()
                        .ca_certificate(Certificate::from_pem(pem))
                        .domain_name("localhost"),
                )
                .unwrap()
                .connect()
                .await
                .unwrap();
        Self {
            child,
            root,
            client: pb::pod_host_client::PodHostClient::new(channel),
            claim,
        }
    }
    fn store(&self) -> PathBuf {
        self.root.join("state/tensorfs")
    }
    async fn machine_log(
        &mut self,
        log: pb::MachineLog,
        tail_bytes: u64,
    ) -> Result<Vec<u8>, tonic::Status> {
        let mut stream = self
            .client
            .read_machine_log(pb::MachineLogQuery {
                claim: Some(self.claim.clone()),
                log: log as i32,
                tail_bytes,
            })
            .await?
            .into_inner();
        let mut data = vec![];
        while let Some(chunk) = stream.message().await? {
            assert!(!chunk.data.is_empty() && chunk.data.len() <= 64 << 10);
            data.extend(chunk.data);
        }
        Ok(data)
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        // The exact process this test started.
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[tokio::test]
async fn machine_log_reads_the_stores_transport_log_with_rotation_and_tail() {
    let mut machine = Machine::start().await;
    let transport = pb::MachineLog::TensorfsTransport;
    assert!(machine.machine_log(transport, 0).await.unwrap().is_empty());

    let logs = machine.store().join("logs");
    fs::create_dir_all(&logs).unwrap();
    fs::write(logs.join("transport.log.1"), "1.000 pull a 1 old\n").unwrap();
    let current: String = (0..3000)
        .map(|n| format!("{n}.000 hedge object-{n} 4096 grant\n"))
        .collect();
    fs::write(logs.join("transport.log"), &current).unwrap();
    let whole = machine.machine_log(transport, 0).await.unwrap();
    assert_eq!(whole, format!("1.000 pull a 1 old\n{current}").into_bytes());
    assert!(whole.len() > 64 << 10, "spans several chunks");

    let tail = String::from_utf8(machine.machine_log(transport, 100).await.unwrap()).unwrap();
    assert!(tail.len() <= 100 && tail.ends_with("2999.000 hedge object-2999 4096 grant\n"));
    // From a line start: the byte before it in the log is a newline.
    assert!(
        current.ends_with(&tail) && current.as_bytes()[current.len() - tail.len() - 1] == b'\n'
    );

    let unknown = machine
        .machine_log(pb::MachineLog::Unspecified, 0)
        .await
        .unwrap_err();
    assert_eq!(unknown.code(), tonic::Code::NotFound);
    let unauthenticated = machine
        .client
        .read_machine_log(pb::MachineLogQuery {
            claim: None,
            log: transport as i32,
            tail_bytes: 0,
        })
        .await
        .unwrap_err();
    assert_eq!(unauthenticated.code(), tonic::Code::Unauthenticated);
}
