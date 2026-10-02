//! Actual CPU process/fd lifetime gates; these never emulate GPU completion.
use cozy_machine::execution::{process_birth, Engine};
use cozy_machine::journal::{Invocation, Journal, SubmissionContext};
use cozy_machine::resident_custody::{Phase, ResidentCustody, ResidentKey, Resources, Role};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixListener;
use std::process::{Command, Stdio};
use std::sync::Arc;

const CHILD: &str = r#"
import array,os,socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(sys.argv[1])
_,rights,_flags,_address=s.recvmsg(1,socket.CMSG_SPACE(4))
fds=array.array('i');fds.frombytes(rights[0][2]);fd=fds[0]
assert os.pread(fd,8,0)==b'weights!'
s.sendall(b'R')
while s.recv(1): pass
os.close(fd)
"#;

#[test]
fn explicit_cancel_keeps_the_file_and_live_process_obligations() {
    let root = std::env::temp_dir().join(format!("resident-custody-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let socket = root.join("peer.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let mut child = Command::new("python3")
        .args(["-I", "-c", CHILD, socket.to_str().unwrap()])
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    let (mut peer, _) = listener.accept().unwrap();
    let object = root.join("weights");
    fs::write(&object, b"weights!").unwrap();
    let backing = Arc::new(File::open(&object).unwrap());
    let native_hold = Arc::downgrade(&backing);
    cozy_machine::protocol::send_fd(&peer, &backing).unwrap();
    let mut ready = [0];
    peer.read_exact(&mut ready).unwrap();
    assert_eq!(ready, b"R"[..]);
    let birth = process_birth(child.id()).unwrap();
    let record = {
        let mut journal = Journal::open(&root).unwrap();
        let context = SubmissionContext {
            actor: "verified-ed25519-fixture".into(),
            request_id: "request".into(),
            submission_id: "submission".into(),
            expected_workspace_id: journal.workspace_id().into(),
            capture_digest: "capture".into(),
            invocation_digest: "invocation".into(),
            payload_digest: "payload".into(),
            publication_authorization_id: "publication".into(),
            preparation_id: String::new(),
        };
        let record = journal
            .accept_public(
                context,
                Invocation {
                    package: "cpu-lifetime".into(),
                    generation: "fixture".into(),
                    module: "cpu_lifetime:app".into(),
                    entrypoint: "hold".into(),
                    input: serde_json::json!({}),
                },
            )
            .unwrap();
        assert!(journal.claim(&record.id).unwrap());
        journal.register_process(&record.id, birth.clone()).unwrap();
        journal.running(&record.id).unwrap();
        record
    };
    let engine = Engine::open(&root).unwrap();
    let mut custody = ResidentCustody::new(engine.clone(), 4, 4);
    let allocation = custody
        .register(
            ResidentKey {
                actor: "verified-ed25519-fixture".into(),
                device_uuid: "CPU-lifetime-test-only".into(),
                content_sha256: "1".repeat(64),
                layout_sha256: "2".repeat(64),
                representation: "immutable-test-bytes".into(),
            },
            8,
            Resources {
                backing: backing.clone(),
                sources: backing.clone(),
            },
        )
        .unwrap();
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, child.id(), 0) } as i32;
    assert!(raw >= 0);
    let death = unsafe { File::from_raw_fd(raw) };
    let ticket = custody
        .attach(&allocation, &record.id, Role::Writer, death)
        .unwrap();
    drop(backing);
    fs::remove_file(object).unwrap();
    assert_eq!(custody.records()[0].recipients, vec![ticket]);
    assert_eq!(custody.charged_bytes(), 8);
    engine.cancel(&record.id, "explicit-actor").unwrap();
    assert!(custody.reap_ended().unwrap().is_empty());
    assert!(custody.take_for_release(&allocation).is_err());
    assert!(native_hold.upgrade().is_some());
    // Observation/liveness accounting never sends a cancel or closes the actual peer stream.
    peer.write_all(b"still-alive").unwrap();
    drop(peer);
    assert!(child.wait().unwrap().success());
    let ended = custody.reap_ended().unwrap();
    assert_eq!(ended.len(), 1);
    assert_eq!(
        custody.records()[0].phase,
        Phase::Quarantined,
        "an ended unknown fill never becomes ready"
    );
    assert_eq!(custody.charged_bytes(), 8);
    let resources = custody.take_for_release(&allocation).unwrap();
    assert!(native_hold.upgrade().is_some());
    drop(resources);
    assert!(native_hold.upgrade().is_none());
    assert_eq!(
        custody.charged_bytes(),
        8,
        "dropping owner references is not asserted GPU physical-release evidence"
    );
    assert_eq!(custody.records()[0].phase, Phase::Releasing);
    let replacement = ResidentCustody::new(engine, 4, 4);
    assert_ne!(replacement.epoch(), allocation.owner_epoch);
    // No GPU or model inference result is claimed; the retained resource was an actual file.
    let maps = fs::read_to_string("/proc/self/maps").unwrap();
    assert!(!maps.contains("libcuda.so"));
    assert!(!maps.contains("libnvidia-ml.so"));
}

#[test]
fn wrong_fd_cannot_substitute_for_a_process_lifetime() {
    let root = std::env::temp_dir().join(format!("resident-no-peer-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&root).unwrap();
    let record = {
        let mut journal = Journal::open(&root).unwrap();
        let context = SubmissionContext {
            actor: "actor".into(),
            request_id: "request".into(),
            submission_id: "submission".into(),
            expected_workspace_id: journal.workspace_id().into(),
            capture_digest: "capture".into(),
            invocation_digest: "invoke".into(),
            payload_digest: "payload".into(),
            publication_authorization_id: "publication".into(),
            preparation_id: String::new(),
        };
        let record = journal
            .accept_public(
                context,
                Invocation {
                    package: "cpu-lifetime".into(),
                    generation: "fixture".into(),
                    module: "cpu_lifetime:app".into(),
                    entrypoint: "hold".into(),
                    input: serde_json::json!({}),
                },
            )
            .unwrap();
        assert!(journal.claim(&record.id).unwrap());
        journal
            .register_process(&record.id, process_birth(std::process::id()).unwrap())
            .unwrap();
        record
    };
    let engine = Engine::open(&root).unwrap();
    let mut custody = ResidentCustody::new(engine, 1, 1);
    let backing = Arc::new(File::open("/dev/null").unwrap());
    let allocation = custody
        .register(
            ResidentKey {
                actor: "actor".into(),
                device_uuid: "CPU-test-only".into(),
                content_sha256: "1".repeat(64),
                layout_sha256: "2".repeat(64),
                representation: "test".into(),
            },
            1,
            Resources {
                backing: backing.clone(),
                sources: backing,
            },
        )
        .unwrap();
    let mut old = allocation.clone();
    old.owner_epoch = "previous-owner".into();
    assert!(custody.begin_revoke(&old).is_err());
    let error = custody
        .attach(
            &allocation,
            &record.id,
            Role::Reader,
            File::open("/dev/null").unwrap(),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        std::io::ErrorKind::InvalidInput,
        "the accepted live attempt cannot use an arbitrary readable fd as a death witness"
    );
    assert_eq!(custody.charged_bytes(), 1);
    assert!(custody.records()[0].recipients.is_empty());
}
