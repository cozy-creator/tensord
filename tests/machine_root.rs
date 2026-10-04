//! One root, one machine, one store: the real binary as this computer's launcher starts it.
//! A second machine on the root refuses, a holder of the Go agent's lock excludes it, and the
//! store the launcher names is the store it fills, with what another writer left there.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use cozy_machine::api::{
    capability::{self, Grant},
    v1::{self, machine_client::MachineClient},
};
use ed25519_dalek::SigningKey;
use fs2::FileExt;
use std::{
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tensorfs_core::{ids::ObjectRef, sha256, store::Store};
use tonic::{
    metadata::MetadataValue,
    transport::{Certificate, ClientTlsConfig, Endpoint},
    Request,
};

const WORKER: &str = "root-test";
const OWNER: [u8; 32] = [44; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn scratch(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(name)
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn spawn(root: &Path, port: u16, store: Option<&Path>, stderr: Stdio) -> Child {
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    let mut command = Command::new(env!("CARGO_BIN_EXE_cozy-machine"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("COZY_MACHINE_ROOT", root)
        .env("COZY_MACHINE_LIFETIME", "persistent")
        .env("COZY_WORKER_ID", WORKER)
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env(
            "COZY_AUTHORIZED_KEYS",
            URL_SAFE_NO_PAD.encode(owner.as_bytes()),
        )
        .env(
            "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
            URL_SAFE_NO_PAD.encode([7; 32]),
        )
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::null())
        .stderr(stderr);
    if let Some(store) = store {
        command.env("COZY_TENSORFS_ROOT", store);
    }
    command.spawn().unwrap()
}

fn boot(root: &Path, port: u16, store: Option<&Path>) -> Machine {
    Machine(spawn(root, port, store, Stdio::inherit()))
}

fn served(root: &Path, port: u16) -> bool {
    let output = Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "--cacert"])
        .arg(root.join("run/cozy/bootstrap/tls.crt"))
        .args(["--resolve", &format!("cozy-worker:{port}:127.0.0.1")])
        .arg(format!("https://cozy-worker:{port}/v1/bootstrap/receipt"))
        .output()
        .unwrap();
    output.stdout == b"200"
}

fn ready(machine: &mut Machine, root: &Path, port: u16) {
    let start = Instant::now();
    while !served(root, port) {
        assert!(
            machine.0.try_wait().unwrap().is_none(),
            "the machine exited before readiness"
        );
        assert!(start.elapsed() < Duration::from_secs(300), "no receipt");
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// A machine started on a taken root: it must exit, and this is what it said.
fn refused(root: &Path) -> String {
    let output = spawn(root, free_port(), None, Stdio::piped())
        .wait_with_output()
        .unwrap();
    assert!(!output.status.success(), "a second machine ran on the root");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn one_root_runs_one_machine() {
    let root = scratch("machine-root-guard");
    let port = free_port();
    let mut first = boot(&root, port, None);
    ready(&mut first, &root, port);

    // Another port changes nothing: the root is taken, and its machine keeps serving.
    let said = refused(&root);
    assert!(said.contains("another machine owns the root"), "{said}");
    assert!(served(&root, port));

    // A stopped machine frees the root at once.
    Command::new("kill")
        .args(["-TERM", &first.0.id().to_string()])
        .status()
        .unwrap();
    first.0.wait().unwrap();
    let lock = std::fs::File::open(root.join("var/lib/cozy/machine/agent.lock")).unwrap();
    lock.try_lock_exclusive()
        .expect("the stopped machine still holds its root");

    // The lock is the Go agent's: whoever holds it owns the root.
    let said = refused(&root);
    assert!(said.contains("another machine owns the root"), "{said}");
    FileExt::unlock(&lock).unwrap();
    let port = free_port();
    let mut again = boot(&root, port, None);
    ready(&mut again, &root, port);
    drop(again);
    std::fs::remove_dir_all(&root).unwrap();
}

fn cap() -> String {
    let expires = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 3600;
    capability::mint(
        &SigningKey::from_bytes(&OWNER),
        Grant {
            machine: WORKER.into(),
            action: capability::MACHINE.into(),
            expires,
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn the_named_store_is_the_machines_store() {
    let root = scratch("machine-root-store");
    // The box's store, as another TensorFS writer left it (the Go machine, the CLI's tfs).
    let named = root.join("box/tensorfs");
    let kept = b"weights the box already holds".to_vec();
    let kept_hex = sha256::hex(&sha256::digest(&kept));
    {
        let store = Store::ensure(&named).unwrap();
        let object = ObjectRef {
            sha256: kept_hex.clone(),
            length: kept.len() as u64,
        };
        store
            .put_stream(&mut &kept[..], Some(&object), &Default::default())
            .unwrap();
    }
    let blob = Store::open(&named).unwrap().object_path(&kept_hex);
    let inode = std::fs::metadata(&blob).unwrap().ino();

    let machine_root = root.join("machine");
    std::fs::create_dir_all(&machine_root).unwrap();
    let port = free_port();
    let mut machine = boot(&machine_root, port, Some(&named));
    ready(&mut machine, &machine_root, port);

    // What the owner writes lands in the named store, beside what it held.
    let pem = std::fs::read(machine_root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let channel = Endpoint::from_shared(format!("https://127.0.0.1:{port}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(pem))
                .domain_name("cozy-worker"),
        )
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = MachineClient::new(channel);
    let written = b"an input this run uploads".to_vec();
    let written_hex = sha256::hex(&sha256::digest(&written));
    let frames = vec![
        v1::WriteFrame {
            digest: format!("sha256:{written_hex}"),
            length: written.len() as u64,
            ..Default::default()
        },
        v1::WriteFrame {
            data: written.clone(),
            ..Default::default()
        },
    ];
    let mut request = Request::new(tokio_stream::iter(frames));
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Cozy-Cap {}", cap())).unwrap(),
    );
    let held = client.write(request).await.unwrap().into_inner().held;
    assert_eq!(held, written.len() as u64);

    let store = Store::open(&named).unwrap();
    assert!(store.contains(&written_hex), "the write went elsewhere");
    assert_eq!(std::fs::read(&blob).unwrap(), kept);
    assert_eq!(std::fs::metadata(&blob).unwrap().ino(), inode);
    assert!(
        !machine_root
            .join("var/lib/cozy/rust-machine/tensorfs")
            .exists(),
        "the machine made a second store"
    );
    drop(machine);
    std::fs::remove_dir_all(&root).unwrap();
}
