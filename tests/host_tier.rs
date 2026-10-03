//! The machine's sealed host tier on the real path: a real TensorFS store, TensorFS's verified
//! fill, real executor stand-in processes (their pidfds hold layouts), and TensorFS's own
//! read-only adoption (`Plane::register_sealed`) checking every byte.
use cozy_machine::{
    host_memory::HostMemory,
    host_tier::{HostGrant, HostTier, HostTierConfig, SealedRequest, TierLimit},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::Write,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::FileExt,
    },
    path::PathBuf,
    process::{Child, Command},
    sync::Arc,
    time::Duration,
};
use tensorfs_core::{
    dtype::Dtype,
    header::{Header, Part, Tensor},
    ids::ObjectRef,
    manifest::{Draft, Entry},
    read, registry,
    store::{Fault, Store},
};
use tensorfs_plane::{layout::Layout, Plane, PlaneConfig, Tier};

const MIB: usize = 1 << 20;

fn bytes_of(key: &str, n: usize) -> Vec<u8> {
    let seed = key
        .bytes()
        .fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..n)
        .map(|i| (seed.wrapping_add(i as u32).wrapping_mul(2654435761) >> 13) as u8)
        .collect()
}

/// One model, component "unet", tensors of the given sizes, in a fresh store.
struct Fixture {
    root: PathBuf,
    store: Arc<Store>,
    manifest: String,
    header: Header,
    tensors: Vec<(String, usize)>,
}
impl Fixture {
    fn new(tag: &str, sizes: &[usize]) -> Self {
        let root = std::env::temp_dir().join(format!(
            "machine-tier-{tag}-{}-{}",
            std::process::id(),
            tensorfs_core::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let plain = registry::seeds()
            .into_iter()
            .find(|e| e.alias == "plain/1")
            .unwrap()
            .spec;
        let tensors: Vec<(String, usize)> = sizes
            .iter()
            .enumerate()
            .map(|(i, n)| (format!("{tag}{i}.weight"), *n))
            .collect();
        let mut rows = Vec::new();
        for (key, n) in &tensors {
            let data = bytes_of(key, *n);
            let part = Part::plan(Dtype::U8, vec![*n as u64], &data);
            if let tensorfs_core::header::Body::Segments(segments) = &part.body {
                let mut at = 0;
                for segment in segments {
                    let chunk = &data[at..at + segment.length as usize];
                    store
                        .put_stream(&mut &chunk[..], Some(segment), &Fault::default())
                        .unwrap();
                    at += segment.length as usize;
                }
            }
            rows.push((
                key.clone(),
                Tensor {
                    dtype: Dtype::U8,
                    shape: vec![*n as u64],
                    encoding: plain.object_id(),
                    parts: vec![("value".into(), part)],
                },
            ));
        }
        let header = Header {
            configs: vec![],
            assets: vec![],
            encodings: vec![plain],
            components: vec![("unet".into(), rows)],
        };
        let header = Header::parse(&header.canonical_bytes().unwrap()).unwrap();
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
        Self {
            root,
            store: Arc::new(store),
            manifest: snapshot.manifest_id(),
            header,
            tensors,
        }
    }
    fn grant(&self) -> HostGrant {
        HostGrant {
            manifest: self.manifest.clone(),
            header: self.header.clone(),
            components: BTreeSet::from(["unet".to_string()]),
        }
    }
    fn traversal(&self) -> Vec<(String, String)> {
        self.tensors
            .iter()
            .map(|(k, _)| ("unet".to_string(), k.clone()))
            .collect()
    }
    /// One region per tensor: the executor's grouping.
    fn regions(&self) -> Vec<Vec<String>> {
        self.tensors
            .iter()
            .map(|(k, _)| vec![format!("unet/{k}")])
            .collect()
    }
    /// The plan as an executor sends it: canonical JSON in a sealed memfd.
    fn plan(&self, components: &[&str]) -> (File, String, u64) {
        let doc = serde_json::json!({
            "manifest": self.manifest, "name": "sdxl/unet", "layout": "sha256:00", "window": 4 << 20,
            "traversal": self.traversal(), "components": components, "regions": self.regions(), "parts": [],
        });
        let body = serde_json::to_vec(&doc).unwrap();
        (
            sealed(&body),
            format!("{:x}", Sha256::digest(&body)),
            body.len() as u64,
        )
    }
    fn layout(&self) -> Layout {
        let plan =
            read::plan_for_traversal(&self.header, &self.traversal(), &["unet".into()], 4 << 20)
                .unwrap();
        Layout::build(&plan, &self.regions()).unwrap()
    }
    /// The executor's half: adopt read-only through TensorFS, wait per region, check every
    /// byte.
    fn adopt(&self, granted: &File) {
        assert!(self.adopt_filling(granted), "the layout's fill failed");
    }
    /// An executor that waits per region: adopt a layout still filling, pin it (each region
    /// once its filler marks it Ready), then check every byte. Never reads the store. False
    /// when the fill failed.
    fn adopt_filling(&self, granted: &File) -> bool {
        let plane = Plane::open(PlaneConfig {
            readers: 2,
            ..Default::default()
        })
        .unwrap();
        let plan =
            read::plan_for_traversal(&self.header, &self.traversal(), &["unet".into()], 4 << 20)
                .unwrap();
        let pinned = plane
            .register_sealed("unet", &plan, &self.regions(), granted.as_raw_fd())
            .and_then(|ws| {
                plane.set_pinned_budget(1 << 30)?;
                plane.want(ws, Tier::Pinned, None, 0, false)?.wait()?;
                Ok(ws)
            });
        if let Ok(ws) = pinned {
            self.check(granted, &plane.layout(ws).unwrap());
            assert_eq!(
                plane.stats().host.counters.fill_bytes,
                0,
                "waited, never read"
            );
        }
        let _ = plane.close();
        pinned.is_ok()
    }
    fn check(&self, granted: &File, layout: &Layout) {
        for part in &layout.parts {
            let key = part
                .what
                .trim_start_matches("unet/")
                .trim_end_matches("#value");
            let n = self.tensors.iter().find(|(k, _)| k == key).unwrap().1;
            let mut got = vec![0; n];
            granted.read_exact_at(&mut got, part.offset).unwrap();
            assert!(
                got == bytes_of(key, n),
                "{} differs in the sealed layout",
                part.what
            );
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn sealed(body: &[u8]) -> File {
    // SAFETY: plain syscalls on a descriptor this test creates and owns.
    unsafe {
        let fd = libc::memfd_create(
            c"plan".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0);
        let mut file = File::from_raw_fd(fd);
        file.write_all(body).unwrap();
        let seals =
            libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
        assert_eq!(libc::fcntl(fd, libc::F_ADD_SEALS, seals), 0);
        file
    }
}

/// An executor stand-in: a live process the tier holds layouts for until it exits.
struct Executor(Child);
impl Executor {
    fn spawn(tier: &HostTier) -> (Self, u64) {
        let child = Command::new("sleep").arg("1000").spawn().unwrap();
        // SAFETY: pidfd_open of our own live child.
        let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, child.id() as libc::c_int, 0) };
        assert!(pidfd >= 0);
        let peer = tier.register_peer(unsafe { File::from_raw_fd(pidfd as i32) });
        (Self(child), peer)
    }
    fn exit(mut self) {
        self.0.kill().unwrap();
        self.0.wait().unwrap();
    }
}
impl Drop for Executor {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Fixed(u64);
impl TierLimit for Fixed {
    fn limit(&self, _: &HostMemory, _: u64) -> u64 {
        self.0
    }
}

fn tier(fx: &Fixture, limit: u64) -> Arc<HostTier> {
    HostTier::new(
        fx.store.clone(),
        HostTierConfig {
            fill_threads: 3,
            ttl: Duration::from_secs(3600),
            plans: None,
        },
        Box::new(Fixed(limit)),
    )
    .unwrap()
}

/// The tier's facts once no fill is running (adopters see each region before the filler
/// records the fill).
fn settled(tier: &HostTier) -> cozy_machine::host_tier::HostTierFacts {
    for _ in 0..500 {
        let facts = tier.facts();
        if facts.filling_bytes == 0 {
            return facts;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("a fill never finished: {:?}", tier.facts());
}

fn ask(tier: &HostTier, peer: u64, fx: &Fixture) -> Option<File> {
    let (plan, sha256, length) = fx.plan(&["unet"]);
    tier.seal(
        peer,
        &[fx.grant()],
        SealedRequest {
            sha256: &sha256,
            length,
        },
        plan,
    )
    .unwrap()
}

#[test]
fn one_fill_serves_every_executor_read_only_and_outlives_them() {
    let fx = Fixture::new("share", &[3 * MIB + 100, 70 * MIB + 12, 200]);
    let tier = tier(&fx, 1 << 30);
    let (first, a) = Executor::spawn(&tier);
    let granted = ask(&tier, a, &fx).expect("room");
    // Read-only, sealed: the executor can neither write nor resize nor punch it.
    // SAFETY: scalar fcntl queries on a descriptor we hold.
    unsafe {
        assert_eq!(
            libc::fcntl(granted.as_raw_fd(), libc::F_GETFL) & libc::O_ACCMODE,
            libc::O_RDONLY
        );
        assert_eq!(
            libc::fcntl(granted.as_raw_fd(), libc::F_GET_SEALS) & 0x10, // F_SEAL_FUTURE_WRITE
            0x10
        );
    }
    fx.adopt(&granted);
    let facts = settled(&tier);
    assert_eq!(
        (facts.ledger.fills.len(), facts.ledger.hits, facts.entries),
        (1, 0, 1)
    );
    assert!(
        facts.charged_bytes >= 73 * MIB as u64 && facts.held_bytes == facts.charged_bytes,
        "{facts:?}"
    );
    let fill = &facts.ledger.fills[0];
    let read = fill.cached_bytes + fill.direct_bytes + fill.buffered_bytes;
    assert!(
        read >= (73 * MIB + 112) as u64 && read <= (73 * MIB + 312) as u64,
        "{fill:?}"
    );

    // The executor exits (a model switch): the layout stays, unheld, and serves the next one
    // with no read at all.
    first.exit();
    drop(granted);
    assert_eq!(tier.facts().held_bytes, 0);
    let (_second, b) = Executor::spawn(&tier);
    fx.adopt(&ask(&tier, b, &fx).expect("held"));
    let facts = tier.facts();
    assert_eq!((facts.ledger.fills.len(), facts.ledger.hits), (1, 1));
}

#[test]
fn unheld_layouts_go_oldest_first_and_held_ones_never() {
    let small = Fixture::new("small", &[8 * MIB]);
    let other = Fixture::new("other", &[8 * MIB]);
    // Room for one layout only.
    let size = small.layout().nbytes;
    let tier = HostTier::new(
        small.store.clone(),
        HostTierConfig {
            fill_threads: 2,
            ttl: Duration::from_secs(3600),
            plans: None,
        },
        Box::new(Fixed(size + size / 2)),
    )
    .unwrap();
    let (first, a) = Executor::spawn(&tier);
    small.adopt(&ask(&tier, a, &small).unwrap());
    // Held by a live executor: no room for the other model, so it streams (never refused).
    let other_tier_ask = |tier: &HostTier, peer| {
        let (plan, sha256, length) = other.plan(&["unet"]);
        tier.seal(
            peer,
            &[other.grant()],
            SealedRequest {
                sha256: &sha256,
                length,
            },
            plan,
        )
    };
    // `other` lives in another store: the tier reads its own, so bring the bytes over.
    copy_store(&other, &small);
    drop(other_tier_ask(&tier, a).unwrap().expect("streamed"));
    assert_eq!(tier.facts().windows, 1);
    // Unheld once its executor exits: the window goes with it, the small layout on demand,
    // and the other model then gets a whole layout.
    first.exit();
    let (_second, b) = Executor::spawn(&tier);
    let granted = other_tier_ask(&tier, b).unwrap().expect("room");
    other.adopt(&granted);
    let facts = settled(&tier);
    assert_eq!((facts.windows, facts.ledger.released), (0, 2), "{facts:?}");
    assert!(facts.ledger.released_bytes >= 8 * MIB as u64, "{facts:?}");
    assert_eq!(facts.entries, 1);
}

/// Copy `from`'s objects and manifest into `into`'s store (as a download would put them).
fn copy_store(from: &Fixture, into: &Fixture) {
    let snapshot = from
        .store
        .read_manifest(&ObjectRef {
            sha256: from.manifest.trim_start_matches("sha256:").into(),
            length: fs::metadata(
                from.store
                    .manifest_path(from.manifest.trim_start_matches("sha256:")),
            )
            .unwrap()
            .len(),
        })
        .unwrap();
    into.store.put_manifest(&snapshot).unwrap();
    for (key, n) in &from.tensors {
        let data = bytes_of(key, *n);
        let part = Part::plan(Dtype::U8, vec![*n as u64], &data);
        if let tensorfs_core::header::Body::Segments(segments) = &part.body {
            let mut at = 0;
            for segment in segments {
                let chunk = &data[at..at + segment.length as usize];
                into.store
                    .put_stream(&mut &chunk[..], Some(segment), &Fault::default())
                    .unwrap();
                at += segment.length as usize;
            }
        }
    }
    let bytes = from.header.canonical_bytes().unwrap();
    into.store
        .put_stream(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Fault::default(),
        )
        .unwrap();
}

#[test]
fn a_plan_outside_the_executors_selection_or_unsealed_is_refused() {
    let fx = Fixture::new("refuse", &[MIB]);
    let tier = tier(&fx, 1 << 30);
    let (_executor, a) = Executor::spawn(&tier);
    let (plan, sha256, length) = fx.plan(&["text_encoder"]);
    let err = tier
        .seal(
            a,
            &[fx.grant()],
            SealedRequest {
                sha256: &sha256,
                length,
            },
            plan,
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    let (plan, _, length) = fx.plan(&["unet"]);
    assert!(tier
        .seal(
            a,
            &[fx.grant()],
            SealedRequest {
                sha256: "00",
                length
            },
            plan
        )
        .is_err());
    // An unsealed plan could change after it was checked.
    let body = b"{}";
    let open = sealed_not(body);
    assert!(tier
        .seal(
            a,
            &[fx.grant()],
            SealedRequest {
                sha256: &format!("{:x}", Sha256::digest(body)),
                length: 2
            },
            open
        )
        .is_err());
    // An unregistered peer gets nothing.
    let (plan, sha256, length) = fx.plan(&["unet"]);
    assert!(tier
        .seal(
            a + 99,
            &[fx.grant()],
            SealedRequest {
                sha256: &sha256,
                length
            },
            plan
        )
        .is_err());
    assert_eq!(tier.facts().ledger.fills.len(), 0);
}

fn sealed_not(body: &[u8]) -> File {
    // SAFETY: plain syscall; the descriptor is ours.
    let mut file =
        unsafe { File::from_raw_fd(libc::memfd_create(c"plan".as_ptr(), libc::MFD_CLOEXEC)) };
    file.write_all(body).unwrap();
    file
}

/// The real limit (`HalfOfHeadroom` over live `memory.high`), in a memory-limited cgroup: a
/// layout past half the headroom is refused (its executor reads the store), a small one fits.
#[test]
fn the_tier_follows_live_headroom_in_a_memory_limited_cgroup() {
    in_scope(
        &["MemoryHigh=256M", "MemoryMax=1G"],
        "inside_a_256_mib_scope",
    );
}

/// Run the ignored test `name` in a transient user scope with these memory properties (no
/// swap); skipped where the session cannot create one.
fn in_scope(memory: &[&str], name: &str) {
    let mut scope = vec![
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        "MemorySwapMax=0",
    ];
    for p in memory {
        scope.extend(["-p", p]);
    }
    if !Command::new("systemd-run")
        .args(&scope)
        .arg("true")
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("this session cannot create a memory-limited scope");
        return;
    }
    let inner = Command::new("systemd-run")
        .args(&scope)
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            name,
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .status()
        .unwrap();
    assert!(inner.success());
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_a_256_mib_scope() {
    let host = cozy_machine::host_memory::read();
    assert!(
        host.available > 0 && host.available <= 256 << 20,
        "{host:?}"
    );
    let big = Fixture::new("big", &[8 * MIB; 20]);
    let tier = HostTier::new(
        big.store.clone(),
        HostTierConfig::default(),
        Box::new(cozy_machine::host_tier::HalfOfHeadroom),
    )
    .unwrap();
    let (_executor, a) = Executor::spawn(&tier);
    let streamed = ask(&tier, a, &big).expect("streamed, never refused");
    assert_eq!(stream_through(&big, &streamed, 1), 160 * MIB as u64);
    drop(streamed);
    let small = Fixture::new("fits", &[8 * MIB; 2]);
    copy_store(&small, &big);
    let (plan, sha256, length) = small.plan(&["unet"]);
    let granted = tier
        .seal(
            a,
            &[small.grant()],
            SealedRequest {
                sha256: &sha256,
                length,
            },
            plan,
        )
        .unwrap();
    let granted = granted.expect("never refused");
    let streamed =
        tensorfs_plane::host::HostMem::adopt_sealed(granted.as_raw_fd(), &small.layout())
            .unwrap()
            .window()
            .is_some();
    if streamed {
        stream_through(&small, &granted, 1);
    } else {
        small.adopt(&granted);
    }
    let facts = settled(&tier);
    assert_eq!(
        (facts.ledger.windows_opened, facts.entries),
        (1 + u64::from(streamed), 2),
        "{facts:?}"
    );
}

/// A machine restart (a new tier, empty) refills a model's layouts from the plans it
/// remembered while that model's executor starts; the executor's ask then finds them filled.
#[test]
fn remembered_plans_refill_layouts_before_the_executor_asks() {
    let fx = Fixture::new("prefill", &[3 * MIB, 40 * MIB]);
    let plans = fx.root.join("plans");
    let config = HostTierConfig {
        fill_threads: 2,
        ttl: Duration::from_secs(3600),
        plans: Some(plans.clone()),
    };
    let first = HostTier::new(fx.store.clone(), config.clone(), Box::new(Fixed(1 << 30))).unwrap();
    let (executor, a) = Executor::spawn(&first);
    fx.adopt(&ask(&first, a, &fx).unwrap());
    settled(&first); // remembered once filled
    executor.exit();
    drop(first);
    assert_eq!(fs::read_dir(&plans).unwrap().count(), 1);

    let second = HostTier::new(fx.store.clone(), config, Box::new(Fixed(1 << 30))).unwrap();
    second.prepare(vec![fx.grant()]);
    // The executor would be importing now; its ask waits for the fill under way, or hits.
    let (_executor, b) = Executor::spawn(&second);
    fx.adopt(&ask(&second, b, &fx).expect("prefilled"));
    let facts = settled(&second);
    assert_eq!(
        (
            facts.ledger.fills.len(),
            facts.ledger.prefills,
            facts.entries
        ),
        (1, 1, 1),
        "{facts:?}"
    );
}

/// A first-ever model: no plan yet, so at admission the tier allocates each component's
/// memory from the manifest; the executor's plan then fills into it (and frees the excess).
#[test]
fn a_first_load_fills_into_memory_reserved_at_admission() {
    let fx = Fixture::new("reserve", &[3 * MIB, 40 * MIB]);
    let tier = tier(&fx, 1 << 30);
    tier.prepare(vec![fx.grant()]);
    let (_executor, a) = Executor::spawn(&tier);
    // The ask waits for an allocation under way rather than allocating again.
    let granted = ask(&tier, a, &fx).expect("room");
    fx.adopt(&granted);
    let facts = tier.facts();
    assert_eq!(
        (
            facts.ledger.reserved,
            facts.ledger.reserved_used,
            facts.reserved_bytes
        ),
        (1, 1, 0),
        "{facts:?}"
    );
    assert!(
        facts.charged_bytes < (43 * MIB + (4 << 20)) as u64,
        "the excess was freed: {facts:?}"
    );
}

/// An executor sends all its components' plans at once (`SealedPrefetch`): the tier fills them
/// in the background and the ask that follows finds its layout filled or filling.
#[test]
fn a_prefetch_fills_while_the_executor_registers_and_the_ask_finds_it() {
    let fx = Fixture::new("prefetch", &[3 * MIB, 40 * MIB]);
    let tier = tier(&fx, 1 << 30);
    let (_executor, a) = Executor::spawn(&tier);
    let doc = serde_json::json!([{
        "manifest": fx.manifest, "name": "sdxl/unet", "layout": "sha256:00", "window": 4 << 20,
        "traversal": fx.traversal(), "components": ["unet"], "regions": fx.regions(), "parts": [],
    }]);
    let body = serde_json::to_vec(&doc).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&body));
    let request = SealedRequest {
        sha256: &sha256,
        length: body.len() as u64,
    };
    tier.prefetch(a, &[fx.grant()], request, sealed(&body))
        .unwrap();
    fx.adopt(&ask(&tier, a, &fx).expect("prefetched"));
    let facts = tier.facts();
    assert_eq!(
        (facts.ledger.fills.len(), facts.entries),
        (1, 1),
        "{facts:?}"
    );
}

/// A first load with nothing filled: the executor gets its layout the moment it is created and
/// each region the moment it is in, so its load never waits for the whole fill; an older
/// executor asking for the same layout gets it complete.
#[test]
fn an_executor_adopts_its_layout_while_it_fills_and_waits_per_region() {
    let fx = Fixture::new("filling", &[64 * MIB, 64 * MIB + 3, 64 * MIB, 5]);
    let tier = tier(&fx, 1 << 30);
    let (_executor, a) = Executor::spawn(&tier);
    let granted = ask(&tier, a, &fx).expect("room");
    let at_grant = tier.facts();
    assert!(
        at_grant.filling_bytes > 0 && at_grant.ledger.fills.is_empty(),
        "granted before its bytes: {at_grant:?}"
    );
    assert!(fx.adopt_filling(&granted));
    let (_other, b) = Executor::spawn(&tier);
    fx.adopt(&ask(&tier, b, &fx).expect("the same layout"));
    let facts = settled(&tier);
    assert_eq!(
        (
            facts.ledger.fills.len(),
            facts.ledger.hits,
            facts.filling_bytes,
            facts.entries
        ),
        (1, 1, 0, 1),
        "{facts:?}"
    );
    assert!(facts.held_bytes >= 192 * MIB as u64, "{facts:?}");
}

/// A fill that fails (a corrupt object the verified read path refuses): the executor that
/// adopted the layout early gets an error, never the bytes; the tier drops the layout, and the
/// next ask fails the same way.
#[test]
fn a_failed_fill_is_an_error_for_its_adopters_and_leaves_the_tier() {
    let fx = Fixture::new("corrupt", &[8 * MIB, 8 * MIB]);
    let layout = fx.layout();
    let last = layout.regions.last().unwrap().items.last().unwrap();
    let tensorfs_plane::layout::ItemSource::Object(range) = &last.source else {
        panic!("an object-backed tensor")
    };
    let blob = fx.store.blob_path(&range.obj.sha256);
    let mut perms = fs::metadata(&blob).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    fs::set_permissions(&blob, perms).unwrap();
    let file = fs::OpenOptions::new().write(true).open(&blob).unwrap();
    file.write_all_at(b"corrupt", 1024).unwrap();
    drop(file);

    let tier = tier(&fx, 1 << 30);
    let (_executor, a) = Executor::spawn(&tier);
    let granted = ask(&tier, a, &fx).expect("opened at once");
    assert!(!fx.adopt_filling(&granted), "a failed region is never read");
    let (_other, b) = Executor::spawn(&tier);
    let (plan, sha256, length) = fx.plan(&["unet"]);
    let again = tier.seal(
        b,
        &[fx.grant()],
        SealedRequest {
            sha256: &sha256,
            length,
        },
        plan,
    );
    assert!(match again {
        Ok(Some(granted)) => !fx.adopt_filling(&granted),
        _ => true,
    });
    let facts = settled(&tier);
    assert!(
        facts.ledger.failed >= 1 && facts.ledger.fills.is_empty() && facts.entries == 0,
        "{facts:?}"
    );
}

/// Read every region of a streamed layout as an executor's plane does (`with_region`), in
/// order, `passes` times, checking every byte; returns the bytes checked.
fn stream_through(fx: &Fixture, granted: &File, passes: usize) -> u64 {
    let layout = fx.layout();
    let host = tensorfs_plane::host::HostMem::adopt_sealed(granted.as_raw_fd(), &layout).unwrap();
    assert!(host.window().is_some(), "a streamed layout");
    let mut checked = 0;
    for _ in 0..passes {
        for (r, region) in layout.regions.iter().enumerate() {
            host.with_region(r as u32, |p| {
                for part in layout.parts.iter().filter(|p| p.region as usize == r) {
                    let key = part
                        .what
                        .trim_start_matches("unet/")
                        .trim_end_matches("#value");
                    let n = fx.tensors.iter().find(|(k, _)| k == key).unwrap().1;
                    // SAFETY: the region's bytes, held by our claim while this runs.
                    let got = unsafe {
                        std::slice::from_raw_parts(p.add((part.offset - region.offset) as usize), n)
                    };
                    assert!(
                        got == &bytes_of(key, n)[..],
                        "{} differs in the window",
                        part.what
                    );
                    checked += n as u64;
                }
                Ok(())
            })
            .unwrap();
        }
    }
    checked
}

/// The disk rung: a model eight times the tier's limit is streamed, not refused. The executor
/// reads every byte, twice over, through two slots; the window goes when the executor does. An
/// executor that cannot wait per region reads the store instead.
#[test]
fn a_model_far_larger_than_the_tier_streams_through_a_window() {
    let fx = Fixture::new("window", &[8 * MIB; 16]);
    let tier = tier(&fx, 16 * MIB as u64);
    let (executor, a) = Executor::spawn(&tier);
    let granted = ask(&tier, a, &fx).expect("streamed, never refused");
    assert_eq!(stream_through(&fx, &granted, 2), 2 * 128 * MIB as u64);
    let facts = tier.facts();
    assert_eq!(
        (facts.windows, facts.ledger.windows_opened),
        (1, 1),
        "{facts:?}"
    );
    assert!(
        facts.window_bytes <= 17 * MIB as u64,
        "two 8 MiB slots: {facts:?}"
    );
    assert!(facts.window_read_bytes >= 2 * 128 * MIB as u64, "{facts:?}");
    executor.exit();
    drop(granted);
    let facts = tier.facts();
    assert_eq!(
        (facts.windows, facts.entries),
        (0, 0),
        "released with its executor: {facts:?}"
    );
    assert!(facts.ledger.streamed_bytes >= 2 * 128 * MIB as u64);
}

/// In a cgroup whose hard limit is below the model's size, with the real limit
/// (`HalfOfHeadroom` over live headroom): the model streams, every byte checks, nothing is
/// refused and nothing is killed.
#[test]
fn a_model_larger_than_its_cgroup_streams_and_checks() {
    in_scope(&["MemoryMax=192M"], "inside_a_192_mib_scope");
}

#[test]
#[ignore = "run inside a memory-limited scope by the test above"]
fn inside_a_192_mib_scope() {
    let host = cozy_machine::host_memory::read();
    assert!(
        host.available > 0 && host.available <= 192 << 20,
        "{host:?}"
    );
    let fx = Fixture::new("cgroup-window", &[8 * MIB; 32]);
    let tier = HostTier::new(
        fx.store.clone(),
        HostTierConfig::default(),
        Box::new(cozy_machine::host_tier::HalfOfHeadroom),
    )
    .unwrap();
    let (_executor, a) = Executor::spawn(&tier);
    let granted = ask(&tier, a, &fx).expect("streamed, never refused");
    assert_eq!(stream_through(&fx, &granted, 1), 256 * MIB as u64);
    let facts = tier.facts();
    assert_eq!(facts.windows, 1, "{facts:?}");
    assert!(facts.window_bytes < 192 << 20, "{facts:?}");
    eprintln!(
        "window: {} bytes, {} read; host {:?}",
        facts.window_bytes, facts.window_read_bytes, facts.host
    );
}
