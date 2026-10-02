//! Real native TensorFS stores/allocations, SCM_RIGHTS and exact child process lifetimes.
use cozy_machine::{
    device_executor::{Frame, Kind},
    execution::process_birth,
    protocol::send_fd,
    shared_host_plane::{
        HostConfig, HostKey, HostPeer, HostPreparation, HostScope, SharedHostPlane,
    },
};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            fs::{FileExt, MetadataExt},
            net::UnixStream,
            process::CommandExt,
        },
    },
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Arc,
};
use tensorfs_core::{
    dtype::Dtype,
    header::{Header, Part, Tensor},
    ids::ObjectRef,
    manifest::{Draft, Entry},
    read::{self, ReadPlan},
    registry,
    store::{Fault, Store},
};

struct Fixture {
    root: PathBuf,
    manifest: String,
    plan: ReadPlan,
    regions: Vec<Vec<String>>,
    data: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "machine-host-{}-{}",
            std::process::id(),
            tensorfs_core::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let data = vec![7; 1 << 20];
        let object = ObjectRef::of(&data);
        store
            .put_stream(&mut data.as_slice(), Some(&object), &Fault::default())
            .unwrap();
        let plain = registry::seeds()
            .into_iter()
            .find(|e| e.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: vec![],
            assets: vec![],
            encodings: vec![plain.clone()],
            components: vec![(
                "linear".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![data.len() as u64],
                        encoding: plain.object_id(),
                        parts: vec![(
                            "value".into(),
                            Part::plan(Dtype::U8, vec![data.len() as u64], &data),
                        )],
                    },
                )],
            )],
        };
        let bytes = header.canonical_bytes().unwrap();
        let object = ObjectRef::of(&bytes);
        store
            .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
            .unwrap();
        let snapshot = Draft {
            entries: vec![("model".into(), Entry::CozyTensors(object))],
        }
        .seal()
        .unwrap();
        store.put_manifest(&snapshot).unwrap();
        let manifest = snapshot.manifest_id();
        let plan = read::plan_for_traversal(
            &header,
            &[("linear".into(), "weight".into())],
            &["linear".into()],
            4 << 20,
        )
        .unwrap();
        Self {
            root,
            manifest,
            plan,
            regions: vec![vec!["linear/weight".into()]],
            data,
        }
    }
    fn plane(&self, budget: u64, entries: usize) -> Arc<SharedHostPlane> {
        SharedHostPlane::new(
            &self.root,
            HostConfig {
                budget_bytes: budget,
                readers: 2,
                max_entries: entries,
            },
        )
        .unwrap()
    }
    fn scope(&self) -> HostScope {
        HostScope {
            actor: "actor-a".into(),
            plan: "authorized-plan-a".into(),
        }
    }
    fn key(&self, name: &str) -> HostKey {
        HostKey::for_regions(name.into(), &self.manifest, &self.regions).unwrap()
    }
    fn prepare(&self, plane: &SharedHostPlane, key: &HostKey) -> bool {
        plane
            .prepare(HostPreparation {
                scope: &self.scope(),
                key,
                manifest: &self.manifest,
                plan: &self.plan,
                regions: &self.regions,
            })
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

// A real CPU weighted inference fixture, with no Torch/CUDA imports or mock function route.
// It waits until commanded, so parent assertions do not use elapsed time as lifecycle policy.
const CHILD: &str = r#"
import array, json, os, socket, sys
s = socket.socket(fileno=197)
print('ready', flush=True)
fd = None
for line in sys.stdin:
    action = line.strip()
    if action == 'exit': break
    if action == 'grant':
        _, anc, _, _ = s.recvmsg(1, socket.CMSG_SPACE(array.array('i').itemsize))
        fds = array.array('i'); fds.frombytes(anc[0][2]); fd = fds[0]
        print('held', flush=True)
        continue
    values = os.pread(fd, 3, 0)
    score = sum(x * w for x, w in zip((2, 3, 5), values))
    print(json.dumps({'weighted_score': score, 'bytes': list(values)}), flush=True)
if fd is not None: os.close(fd)
"#;
struct Recipient {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    socket: UnixStream,
    peer: HostPeer,
}
impl Recipient {
    fn new(scope: &HostScope, plane: &SharedHostPlane) -> Self {
        let (socket, child_socket) = UnixStream::pair().unwrap();
        let fd = child_socket.as_raw_fd();
        let mut command = Command::new("python3");
        command
            .arg("-c")
            .arg(CHILD)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(fd, 197) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().unwrap();
        drop(child_socket);
        let input = child.stdin.take().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut ready = String::new();
        output.read_line(&mut ready).unwrap();
        assert_eq!(ready.trim(), "ready");
        let peer = HostPeer {
            actor: scope.actor.clone(),
            plan: scope.plan.clone(),
            birth: process_birth(child.id()).unwrap(),
        };
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, child.id(), 0) } as i32;
        assert!(raw >= 0);
        plane
            .register_peer(peer.clone(), unsafe { File::from_raw_fd(raw) })
            .unwrap();
        Self {
            child,
            input,
            output,
            socket,
            peer,
        }
    }
    fn grant(&mut self, plane: &SharedHostPlane, key: &HostKey) -> File {
        let frame = Frame {
            kind: Kind::HostTier,
            name: key.name.clone(),
            layout: key.layout.clone(),
            ..Frame::default()
        };
        let (answer, file) = plane.request(&self.peer, &frame, None).unwrap().unwrap();
        assert!(answer.ok && answer.held);
        let file = file.unwrap();
        send_fd(&self.socket, &file).unwrap();
        writeln!(self.input, "grant").unwrap();
        let mut response = String::new();
        self.output.read_line(&mut response).unwrap();
        assert_eq!(response.trim(), "held");
        file
    }
    fn infer(&mut self) {
        writeln!(self.input, "infer").unwrap();
        let mut response = String::new();
        self.output.read_line(&mut response).unwrap();
        let result: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(result["weighted_score"], 70);
        assert_eq!(result["bytes"], serde_json::json!([7, 7, 7]));
    }
    fn end(mut self) {
        writeln!(self.input, "exit").unwrap();
        assert!(self.child.wait().unwrap().success());
    }
}

#[test]
fn native_allocation_fills_once_and_stays_cuda_free() {
    let fx = Fixture::new();
    let plane = fx.plane(3 << 20, 4);
    let key = fx.key("generation/linear");
    assert!(fx.prepare(&plane, &key));
    let first = plane.stats().unwrap();
    assert_eq!(first.filled_bytes, fx.data.len() as u64);
    assert!(first.backing_bytes >= fx.data.len() as u64);
    assert!(first.backing_bytes <= first.budget_bytes);
    assert!(fx.prepare(&plane, &key));
    assert_eq!(plane.stats().unwrap().filled_bytes, first.filled_bytes);
    let maps = fs::read_to_string("/proc/self/maps").unwrap();
    assert!(!maps.contains("libcuda.so"));
    assert!(!maps.contains("libnvidia-ml"));
    for fd in fs::read_dir("/proc/self/fd").unwrap().flatten() {
        if let Ok(path) = fs::read_link(fd.path()) {
            assert!(!path.to_string_lossy().starts_with("/dev/nvidia"));
        }
    }
}

#[test]
fn recipient_crash_survival_and_shared_backing_have_one_charge() {
    let fx = Fixture::new();
    let plane = fx.plane(3 << 20, 4);
    let key = fx.key("generation/linear");
    assert!(fx.prepare(&plane, &key));
    let first_charge = plane.stats().unwrap().backing_bytes;
    let mut a = Recipient::new(&fx.scope(), &plane);
    let file_a = a.grant(&plane, &key);
    a.infer();
    let mut b = Recipient::new(&fx.scope(), &plane);
    let file_b = b.grant(&plane, &key);
    b.infer();
    assert_eq!(
        file_a.metadata().unwrap().ino(),
        file_b.metadata().unwrap().ino()
    );
    let charged = plane.stats().unwrap();
    assert_eq!(charged.backing_bytes, first_charge);
    assert_eq!(charged.active_backing_bytes, first_charge);
    assert_eq!(charged.recipients, 2);
    // Explicit test-controlled abrupt process death, not a timeout or a production kill rule.
    a.child.kill().unwrap();
    a.child.wait().unwrap();
    drop(a);
    b.infer();
    assert_eq!(plane.stats().unwrap().recipients, 1);
    assert_eq!(plane.stats().unwrap().backing_bytes, first_charge);
    let mut c = Recipient::new(&fx.scope(), &plane);
    let file_c = c.grant(&plane, &key);
    c.infer();
    assert_eq!(
        file_c.metadata().unwrap().ino(),
        file_b.metadata().unwrap().ino()
    );
    assert_eq!(plane.stats().unwrap().filled_bytes, fx.data.len() as u64);
    b.end();
    c.end();
    assert_eq!(plane.stats().unwrap().recipients, 0);
}

#[test]
fn budget_change_never_punches_live_receiver_and_reclaims_after_exit() {
    let fx = Fixture::new();
    let plane = fx.plane(3 << 20, 1);
    let key = fx.key("generation/linear");
    assert!(fx.prepare(&plane, &key));
    let mut child = Recipient::new(&fx.scope(), &plane);
    let file = child.grant(&plane, &key);
    let before = file.metadata().unwrap().blocks() * 512;
    let lowered = plane.set_budget(0).unwrap();
    assert_eq!(lowered.backing_bytes, before);
    assert_eq!(lowered.over_budget_bytes, before);
    child.infer();
    assert!(!fx.prepare(&plane, &fx.key("other/linear")));
    child.end();
    let freed = plane.set_budget(0).unwrap();
    assert_eq!(freed.backing_bytes, 0);
    assert_eq!(freed.entries, 0);
    assert!(file.metadata().unwrap().blocks() * 512 < before);
    let mut bytes = [1; 3];
    file.read_exact_at(&mut bytes, 0).unwrap();
    assert_eq!(bytes, [0; 3]);
}

#[test]
fn owner_loss_keeps_inflight_grant_bytes_and_zero_budget_is_disk_fallback() {
    let fx = Fixture::new();
    let off = fx.plane(0, 1);
    assert!(!fx.prepare(&off, &fx.key("generation/linear")));
    assert_eq!(off.stats().unwrap().backing_bytes, 0);
    let plane = fx.plane(3 << 20, 1);
    let key = fx.key("generation/linear");
    assert!(fx.prepare(&plane, &key));
    let mut child = Recipient::new(&fx.scope(), &plane);
    let file = child.grant(&plane, &key);
    // The transferred descriptor itself retains its independent native hold while adoption
    // may still be pending; owner drop cannot punch that receiver's pages.
    drop(plane);
    drop(file);
    child.infer();
    child.infer();
    child.end();
}

#[test]
fn actor_plan_and_recorded_birth_gate_host_grants() {
    let fx = Fixture::new();
    let plane = fx.plane(3 << 20, 2);
    let key = fx.key("generation/linear");
    assert!(fx.prepare(&plane, &key));
    let mut stranger = fx.scope();
    stranger.actor = "actor-b".into();
    let other = Recipient::new(&stranger, &plane);
    let frame = Frame {
        kind: Kind::HostTier,
        name: key.name.clone(),
        layout: key.layout.clone(),
        ..Frame::default()
    };
    let (answer, fd) = plane.request(&other.peer, &frame, None).unwrap().unwrap();
    assert!(answer.ok && !answer.held);
    assert!(fd.is_none());
    let mut forged = other.peer.clone();
    forged.birth.start_ticks += 1;
    assert_eq!(
        plane
            .request(&forged, &frame, None)
            .unwrap()
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
    let mut mismatch = key.clone();
    mismatch.layout = "arbitrary".into();
    assert!(plane
        .prepare(HostPreparation {
            scope: &fx.scope(),
            key: &mismatch,
            manifest: &fx.manifest,
            plan: &fx.plan,
            regions: &fx.regions
        })
        .is_err());
    other.end();
}
