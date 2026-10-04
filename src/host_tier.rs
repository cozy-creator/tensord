//! Degree 1: the machine's host weights. Each weight set's layout is a memfd sealed at creation
//! (nobody else can write, resize or punch it), filled once through TensorFS's verified read
//! path by one background filler, in the order asked, and kept across executors and model
//! switches. Executors adopt it read-only (`host_tiers.sealed/1`, `Plane.register_sealed`) the
//! moment it exists and use each region once its Ready word is set (`host_tiers.filling/1`). The machine holds one descriptor per layout. The tier's
//! size follows live host headroom (`host_memory`), read at every admission; unheld, complete
//! layouts are released least recently used first, or after `ttl` unused. How much of the
//! headroom the tier may take is `TierLimit`'s decision (the memory policy module's). No lock
//! is held across a fill. A layout that does not fit even after releases is streamed instead
//! (the disk rung): a sealed window of a few slots (`TierLimit::staging` sizes it; one region
//! at least) that the machine refills from disk in order as the executor claims regions;
//! nothing is refused for size, and the executor reads no store either way.
use crate::{host_memory::HostMemory, os};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io,
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::fs::{FileExt, MetadataExt},
    },
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex, Weak,
    },
    time::{Duration, Instant},
};
use tensorfs_core::{
    header::Header,
    meta::Meta,
    read::{self, Source as Item},
    store::Store,
};
use tensorfs_plane::{
    host::{self, OpenFill},
    io::{IoTally, Source},
    layout::Layout,
    window::{self, OpenWindow},
};

#[derive(Clone, Debug)]
pub struct HostTierConfig {
    /// Threads of one fill (read + copy). Default: the CPUs this process may use, at most 8:
    /// more contended on RunPod (an A5000 pod's UNet fill: 8 threads 0.44 s, 16 threads 1.08 s).
    pub fill_threads: usize,
    /// An unheld layout unused this long is released.
    pub ttl: Duration,
    /// Where the plans of filled layouts are remembered (across machine restarts), so a
    /// model's layouts refill while its executor starts (`prefill`). None: not remembered.
    pub plans: Option<PathBuf>,
}
impl Default for HostTierConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self {
            fill_threads: cpus.clamp(1, 8),
            ttl: Duration::from_secs(30 * 60),
            plans: None,
        }
    }
}

/// The most the tier may charge, given live host memory and what it charges now.
pub trait TierLimit: Send + Sync {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64;
    /// The most a streamed layout's slots may take now (`charged` counts every layout): by
    /// default the room left under `limit`. A window gets one slot (its largest region)
    /// whatever this says.
    fn staging(&self, host: &HostMemory, charged: u64) -> u64 {
        self.limit(host, charged).saturating_sub(charged)
    }
}
/// Until the memory policy module decides: half of what the host has for the tier (free
/// before its tightest limit, plus what the tier holds), Runtime's `pinned_total` rule.
pub struct HalfOfHeadroom;
impl TierLimit for HalfOfHeadroom {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64 {
        if host.available < 0 {
            return 0; // unreadable: nothing is admitted, executors read the store
        }
        (host.available as u64 + charged) / 2
    }
}

/// What one executor may adopt: these components of this manifest (its selected model).
#[derive(Clone)]
pub struct HostGrant {
    pub manifest: String,
    pub header: Header,
    pub components: BTreeSet<String>,
}

/// Runtime `SealedPlan`: the read plan and region grouping the model code chose.
#[derive(Clone, Debug, Deserialize)]
struct SealedPlan {
    manifest: String,
    name: String,
    window: u64,
    traversal: Vec<(String, String)>,
    components: Vec<String>,
    regions: Vec<Vec<String>>,
    #[serde(default)]
    parts: Vec<String>,
}

/// (manifest, components): the latest plan per model part.
type PlanKey = (String, Vec<String>);

impl SealedPlan {
    fn identity(&self) -> PlanKey {
        (hex(&self.manifest).to_string(), self.components.clone())
    }
}

/// The `SealedTier` request's envelope; the plan itself arrives as a sealed memfd.
pub struct SealedRequest<'a> {
    pub sha256: &'a str,
    pub length: u64,
    /// The regions to stage (the blocks that stream every step): an executor that reads holes
    /// from object files (`host_tiers.staged/1`) gets the rest as holes. None: every region.
    pub stage: Option<&'a [u32]>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Fill {
    pub name: String,
    pub bytes: u64,
    pub ms: f64,
    pub cached_bytes: u64,
    pub direct_bytes: u64,
    pub buffered_bytes: u64,
    /// Bytes this process read from storage meanwhile (`/proc/self/io`): the fill's disk
    /// reads, plus any other reader's in the machine at the same time.
    pub disk_read_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Ledger {
    /// The latest fills, newest last (at most 32).
    pub fills: Vec<Fill>,
    pub hits: u64,
    /// Layouts filled from a remembered plan while their executor started.
    pub prefills: u64,
    /// Components whose memory was allocated at admission, before any plan, and the fills
    /// that used such an allocation.
    pub reserved: u64,
    pub reserved_used: u64,
    /// Layouts streamed through a window because they did not fit, and the bytes ended windows
    /// read from disk.
    pub windows_opened: u64,
    pub streamed_bytes: u64,
    /// Fills that failed: their adopters got an error, the layout was dropped.
    pub failed: u64,
    pub released: u64,
    /// What the released layouts charged, and what the kernel's shared memory fell by.
    pub released_bytes: u64,
    pub freed_bytes: u64,
    /// Released layouts whose memory did not come back (something still maps them): they
    /// stay charged.
    pub stranded_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostTierFacts {
    pub host: HostMemory,
    pub limit: u64,
    pub charged_bytes: u64,
    pub held_bytes: u64,
    pub filling_bytes: u64,
    /// Memory allocated at admission for components not yet planned.
    pub reserved_bytes: u64,
    /// Live windows (streamed layouts), what they charge, and what they have read so far.
    pub windows: usize,
    pub window_bytes: u64,
    pub window_read_bytes: u64,
    pub entries: usize,
    pub peers: usize,
    #[serde(flatten)]
    pub ledger: Ledger,
}

struct Entry {
    fd: File,
    /// A layout staged in part: the owner's mapping, kept to stage more.
    staging: Option<Staging>,
    charged: u64,
    used: Instant,
    holders: BTreeSet<u64>,
    /// Every region is in; until then nothing releases it.
    complete: bool,
    /// A streamed layout: its machine's staging loop. Released once its last holder exits.
    window: Option<Streamer>,
    granted: bool,
}
struct Streamer {
    stop: Arc<AtomicBool>,
    tally: Arc<IoTally>,
}
impl Streamer {
    fn read_bytes(&self) -> u64 {
        let t = &self.tally;
        [&t.cached_bytes, &t.direct_bytes, &t.buffered_bytes]
            .iter()
            .map(|n: &&AtomicU64| n.load(Ordering::Relaxed))
            .sum()
    }
}
/// What a layout staged in part needs to stage more of it.
struct Staging {
    open: Arc<OpenFill>,
    layout: Arc<Layout>,
    manifest: String,
    staged: BTreeSet<u32>,
}
enum Slot {
    /// Admitted, its memfd being created.
    Opening(u64),
    Open(Entry),
}
/// Work for the tier's filler, in order: a layout's first fill (of what it stages), or more
/// regions staged into a layout staged in part.
enum Job {
    Fill {
        key: String,
        plan: SealedPlan,
        body: Vec<u8>,
        layout: Arc<Layout>,
        open: Arc<OpenFill>,
        prefill: bool,
    },
    Stage {
        key: String,
        regions: Vec<u32>,
    },
}
struct Peer {
    pidfd: File,
    /// It reads a layout's holes from the object files (`host_tiers.staged/1`).
    staged: bool,
}
/// A component's memory allocated before its plan exists (`prepare`).
enum Reservation {
    Allocating(u64),
    Ready {
        fd: OwnedFd,
        bytes: u64,
        made: Instant,
    },
}
impl Reservation {
    fn bytes(&self) -> u64 {
        match self {
            Self::Allocating(n) | Self::Ready { bytes: n, .. } => *n,
        }
    }
}
#[derive(Default)]
struct State {
    /// By TensorFS layout digest: the same bytes at the same offsets, whoever asks.
    slots: BTreeMap<String, Slot>,
    peers: BTreeMap<u64, Peer>,
    next_peer: u64,
    ledger: Ledger,
    /// By (manifest, component).
    reserved: BTreeMap<(String, String), Reservation>,
}
impl State {
    fn charged(&self) -> u64 {
        self.slots
            .values()
            .map(|s| match s {
                Slot::Opening(n) => *n,
                Slot::Open(e) => e.charged,
            })
            .sum::<u64>()
            + self.reserved.values().map(Reservation::bytes).sum::<u64>()
            + self.ledger.stranded_bytes
    }
    fn filling(&self) -> bool {
        self.slots.values().any(|s| match s {
            Slot::Opening(_) => true,
            Slot::Open(e) => !e.complete && e.window.is_none(),
        })
    }
}

pub struct HostTier {
    store: Arc<Store>,
    meta: Arc<Meta>,
    config: HostTierConfig,
    limit: Box<dyn TierLimit>,
    state: Mutex<State>,
    filled: Condvar,
    /// The filler's queue: one fill at a time, in the order layouts were opened.
    fills: Mutex<mpsc::Sender<Job>>,
    /// The latest verified plan body per (manifest, components).
    plans: Mutex<BTreeMap<PlanKey, Vec<u8>>>,
}

fn failure(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
fn denied(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, what.to_string())
}
fn hex(value: &str) -> &str {
    value.strip_prefix("sha256:").unwrap_or(value)
}
fn disk_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("read_bytes: ")?.trim().parse().ok())
        })
        .unwrap_or(0)
}

impl HostTier {
    pub fn new(
        store: Arc<Store>,
        config: HostTierConfig,
        limit: Box<dyn TierLimit>,
    ) -> io::Result<Arc<Self>> {
        let meta = Arc::new(Meta::open(&store).map_err(failure)?);
        let mut plans = BTreeMap::new();
        if let Some(dir) = &config.plans {
            fs::create_dir_all(dir)?;
            for entry in fs::read_dir(dir)? {
                let body = fs::read(entry?.path())?;
                if let Ok(plan) = serde_json::from_slice::<SealedPlan>(&body) {
                    plans.insert(plan.identity(), body); // an unreadable one is ignored
                }
            }
        }
        let (fills, jobs) = mpsc::channel::<Job>();
        let tier = Arc::new(Self {
            store,
            meta,
            config,
            limit,
            state: Mutex::default(),
            filled: Condvar::new(),
            fills: Mutex::new(fills),
            plans: Mutex::new(plans),
        });
        // Ends with the tier; a job it never runs poisons its layout (`OpenFill` on drop).
        let weak: Weak<Self> = Arc::downgrade(&tier);
        std::thread::spawn(move || {
            for job in jobs {
                match weak.upgrade() {
                    Some(tier) => tier.run(job),
                    None => return,
                }
            }
        });
        Ok(tier)
    }

    /// Start on what an executor of these grants will ask for, while it starts: refill every
    /// layout a remembered plan describes, and for each component without one allocate its
    /// memory now (sized from the manifest) and read its objects ahead. The layout itself
    /// needs the model code's plan; when it arrives the fill mostly copies.
    pub fn prepare(self: &Arc<Self>, grants: Vec<HostGrant>, staged: bool) {
        if staged {
            // Its executor stages only what streams and reads the rest from the files: warm
            // the page cache (reclaimable), allocate nothing.
            let tier = self.clone();
            std::thread::spawn(move || {
                for grant in &grants {
                    for component in &grant.components {
                        if let Ok(plan) = component_plan(&grant.header, component) {
                            tier.read_ahead(&plan);
                        }
                    }
                }
            });
            return;
        }
        let plans = self.plans.lock().unwrap().clone();
        let mut bodies = Vec::new();
        let mut unplanned = Vec::new();
        for grant in &grants {
            for component in &grant.components {
                let known = plans.iter().find(|((manifest, components), _)| {
                    manifest == hex(&grant.manifest) && components.contains(component)
                });
                match known {
                    Some((_, body)) if !bodies.contains(body) => bodies.push(body.clone()),
                    Some(_) => {}
                    None => unplanned.push((
                        grant.manifest.clone(),
                        grant.header.clone(),
                        component.clone(),
                    )),
                }
            }
        }
        if bodies.is_empty() && unplanned.is_empty() {
            return;
        }
        let tier = self.clone();
        std::thread::spawn(move || {
            for (manifest, header, component) in unplanned {
                if let Err(error) = tier.reserve(&manifest, &header, &component) {
                    eprintln!("host tier reservation skipped: {error}");
                }
            }
            for body in bodies {
                let filled = serde_json::from_slice::<SealedPlan>(&body)
                    .map_err(failure)
                    .and_then(|plan| tier.ensure(&plan, &grants, &body, true, None).map(|_| ()));
                if let Err(error) = filled {
                    eprintln!("host tier prefill skipped: {error}");
                }
            }
        });
    }

    /// Allocate one component's memory before its plan: every byte its tensors declare, plus
    /// room for region padding; a fill frees what its layout does not use.
    fn reserve(&self, manifest: &str, header: &Header, component: &str) -> io::Result<()> {
        let plan = component_plan(header, component)?;
        let bytes = plan.bytes + plan.bytes / 50 + (32 << 20);
        let key = (hex(manifest).to_string(), component.to_string());
        {
            let mut state = self.state.lock().unwrap();
            self.reap(&mut state);
            if state.reserved.contains_key(&key) {
                return Ok(());
            }
            loop {
                let host = crate::host_memory::read();
                let charged = state.charged();
                if charged + bytes <= self.limit.limit(&host, charged) {
                    break;
                }
                if !self.release_lru(&mut state) {
                    return Ok(()); // no room: the fill allocates as it copies
                }
            }
            state
                .reserved
                .insert(key.clone(), Reservation::Allocating(bytes));
        }
        // Read ahead what the page cache lacks; the fill then copies instead of reading disk.
        self.read_ahead(&plan);
        let made = host::reserve(component, bytes, self.config.fill_threads);
        let mut state = self.state.lock().unwrap();
        let result = match made {
            Ok(fd) => {
                let made = Instant::now();
                state
                    .reserved
                    .insert(key, Reservation::Ready { fd, bytes, made });
                state.ledger.reserved += 1;
                Ok(())
            }
            Err(error) => {
                state.reserved.remove(&key);
                Err(failure(error))
            }
        };
        self.filled.notify_all();
        result
    }

    /// Ask the kernel to read `plan`'s objects into the page cache (asynchronous, reclaimable).
    fn read_ahead(&self, plan: &read::ReadPlan) {
        for item in &plan.items {
            if let Item::Object(range) = &item.source {
                if let Ok(file) = File::open(self.store.blob_path(&range.obj.sha256)) {
                    // SAFETY: advice on a descriptor we hold.
                    unsafe {
                        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED)
                    };
                }
            }
        }
    }

    /// An executor, by its pidfd: what it adopted stays held until that exact process exits.
    /// `staged`: it reads a layout's holes from object files (`host_tiers.staged/1`), so the
    /// tier stages only the regions it streams.
    pub fn register_peer(&self, pidfd: File, staged: bool) -> u64 {
        let mut state = self.state.lock().unwrap();
        state.next_peer += 1;
        let id = state.next_peer;
        state.peers.insert(id, Peer { pidfd, staged });
        id
    }

    fn staged_peer(&self, peer: u64) -> io::Result<bool> {
        match self.state.lock().unwrap().peers.get(&peer) {
            Some(p) => Ok(p.staged),
            None => Err(denied("host tier peer is not a registered live executor")),
        }
    }

    /// One weight set's layout for `peer`, read-only: from the tier, or opened now (whole and
    /// filling, or streamed when it does not fit). The executor waits per region either way.
    pub fn seal(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
    ) -> io::Result<Option<File>> {
        // Only an executor that reads holes from object files gets a layout staged in part.
        let stage = match self.staged_peer(peer)? {
            true => request.stage.map(<[u32]>::to_vec),
            false => None,
        };
        let (plan, body) = Self::plan(request, plan)?;
        let mut key = self.ensure(&plan, grants, &body, false, stage.clone())?;
        match stage.as_deref() {
            // A layout another executor had staged in part: this one stages what it asked for.
            Some(regions) => {
                self.stage_regions(&key, regions);
            }
            // It reads no holes (a CPU executor, an older Runtime): the rest is staged when the
            // tier has room for all of it; else it reads through a window of its own.
            None if !self.stage_rest(&key) => {
                let (read_plan, layout) = Self::layout(&plan, grants)?;
                let staging = {
                    let state = self.state.lock().unwrap();
                    self.limit
                        .staging(&crate::host_memory::read(), state.charged())
                };
                key = self.stream(format!("{key}.whole"), &plan, read_plan, layout, staging)?;
            }
            None => {}
        }
        let mut state = self.state.lock().unwrap();
        match state.slots.get(&key) {
            Some(Slot::Open(_)) => self.grant(&mut state, &key, peer).map(Some),
            _ => Err(failure("the layout's fill failed before it was granted")),
        }
    }

    /// The read plan and layout `plan` names, inside the executor's selection.
    fn layout(plan: &SealedPlan, grants: &[HostGrant]) -> io::Result<(read::ReadPlan, Layout)> {
        let grant = grants
            .iter()
            .find(|g| hex(&g.manifest) == hex(&plan.manifest))
            .ok_or_else(|| denied("manifest outside this executor's selection"))?;
        if plan.components.is_empty()
            || plan
                .components
                .iter()
                .chain(plan.traversal.iter().map(|(c, _)| c))
                .any(|c| !grant.components.contains(c))
        {
            return Err(denied("plan leaves the selected components"));
        }
        let mut read_plan = read::plan_for_traversal(
            &grant.header,
            &plan.traversal,
            &plan.components,
            plan.window,
        )
        .map_err(failure)?;
        if !plan.parts.is_empty() {
            read_plan = read_plan.select(&plan.parts).map_err(failure)?;
        }
        let layout = Layout::build(&read_plan, &plan.regions).map_err(failure)?;
        Ok((read_plan, layout))
    }

    /// The layout `plan` names, held or opened now: whole, or with only `stage` staged (the
    /// rest holes, read from object files), its fill queued, when the tier can make room. A
    /// staged layout that does not fit opens with nothing staged; a whole one streams through
    /// a window. Its key.
    fn ensure(
        &self,
        plan: &SealedPlan,
        grants: &[HostGrant],
        body: &[u8],
        prefill: bool,
        mut stage: Option<Vec<u32>>,
    ) -> io::Result<String> {
        let (read_plan, layout) = Self::layout(plan, grants)?;
        let key = layout.digest.clone();
        // A one-component plan uses the memory `prepare` allocated for that component.
        let reservation = match plan.components.as_slice() {
            [component] => Some((hex(&plan.manifest).to_string(), component.clone())),
            _ => None,
        };
        let mut state = self.state.lock().unwrap();
        loop {
            self.reap(&mut state);
            match state.slots.get(&key) {
                Some(Slot::Open(_)) => {
                    if !prefill {
                        state.ledger.hits += 1;
                    }
                    return Ok(key);
                }
                Some(Slot::Opening(_)) => state = self.filled.wait(state).unwrap(),
                None if reservation.as_ref().is_some_and(|r| {
                    matches!(state.reserved.get(r), Some(Reservation::Allocating(_)))
                }) =>
                {
                    state = self.filled.wait(state).unwrap()
                }
                None => break,
            }
        }
        let reserved = reservation.and_then(|r| match state.reserved.remove(&r) {
            Some(Reservation::Ready { fd, .. }) => Some(fd),
            _ => None,
        });
        // Admission over live headroom, read now: release unheld layouts, oldest first.
        let spans = |regions: &[u32]| -> u64 {
            regions
                .iter()
                .filter_map(|&r| layout.regions.get(r as usize))
                .map(|r| r.span)
                .sum()
        };
        let mut need = stage.as_deref().map_or(layout.nbytes, spans);
        loop {
            let host = crate::host_memory::read();
            let charged = state.charged();
            if charged + need <= self.limit.limit(&host, charged) {
                break;
            }
            if !self.release_lru(&mut state) {
                if stage.is_some() {
                    // Nothing staged: every region reads the object files (the disk rung).
                    stage = Some(Vec::new());
                    need = 0;
                    break;
                }
                let staging = self.limit.staging(&host, charged);
                drop(state);
                if let Some(fd) = reserved {
                    drop(fd); // the component's reservation: a window is smaller
                }
                return self.stream(key, plan, read_plan, layout, staging);
            }
        }
        state.ledger.reserved_used += u64::from(reserved.is_some());
        state.slots.insert(key.clone(), Slot::Opening(need));
        drop(state);
        // Sealed now, filled by the filler: adopters wait per region, never for the whole.
        let opened =
            host::open_sealed(&plan.name, &layout, reserved, stage.as_deref()).map_err(failure);
        let mut state = self.state.lock().unwrap();
        let result = match opened {
            Ok((fd, open)) => {
                let (open, layout) = (Arc::new(open), Arc::new(layout));
                let staging = stage.map(|regions| Staging {
                    open: open.clone(),
                    layout: layout.clone(),
                    manifest: hex(&plan.manifest).to_string(),
                    staged: regions.into_iter().collect(),
                });
                state.slots.insert(
                    key.clone(),
                    Slot::Open(Entry {
                        fd: File::from(fd),
                        staging,
                        charged: need,
                        used: Instant::now(),
                        holders: BTreeSet::new(),
                        complete: false,
                        window: None,
                        granted: false,
                    }),
                );
                let job = Job::Fill {
                    key: key.clone(),
                    plan: plan.clone(),
                    body: body.to_vec(),
                    layout,
                    open,
                    prefill,
                };
                // A send fails only once the filler is gone with the tier: the job's drop
                // poisons the layout, so no adopter waits for it.
                let _ = self.fills.lock().unwrap().send(job);
                Ok(key)
            }
            Err(error) => {
                state.slots.remove(&key);
                Err(error)
            }
        };
        self.filled.notify_all();
        result
    }

    /// The disk rung: `layout` streamed through a window of as many of its largest regions as
    /// `staging` bytes hold (one at least: the indivisible working set; two let one refill
    /// while another is read), served from disk on its own thread until its last holder exits.
    fn stream(
        &self,
        key: String,
        plan: &SealedPlan,
        read_plan: read::ReadPlan,
        layout: Layout,
        staging: u64,
    ) -> io::Result<String> {
        let span = layout
            .regions
            .iter()
            .map(|r| r.span)
            .max()
            .unwrap_or(1)
            .max(1);
        let slots = ((staging / span) as usize).clamp(1, layout.regions.len().max(1));
        let (fd, open) = window::open_window(&plan.name, &layout, slots).map_err(failure)?;
        let streamer = Streamer {
            stop: Arc::new(AtomicBool::new(false)),
            tally: Arc::new(IoTally::default()),
        };
        let (stop, tally) = (streamer.stop.clone(), streamer.tally.clone());
        let charged = open.bytes();
        let mut state = self.state.lock().unwrap();
        if matches!(
            state.slots.get(&key),
            Some(Slot::Open(_)) | Some(Slot::Opening(_))
        ) {
            return Ok(key); // another ask opened it meanwhile; ours goes unused
        }
        state.slots.insert(
            key.clone(),
            Slot::Open(Entry {
                fd: File::from(fd),
                staging: None,
                charged,
                used: Instant::now(),
                holders: BTreeSet::new(),
                complete: false,
                window: Some(streamer),
                granted: false,
            }),
        );
        state.ledger.windows_opened += 1;
        drop(state);
        self.filled.notify_all();
        let (store, meta, threads) = (
            self.store.clone(),
            self.meta.clone(),
            self.config.fill_threads,
        );
        let manifest = hex(&plan.manifest).to_string();
        std::thread::spawn(move || {
            if let Err(error) = serve(
                &store, &meta, &manifest, &read_plan, &layout, open, threads, &tally, &stop,
            ) {
                // Dropping the window failed every region: its readers get the error.
                eprintln!("host tier window stopped: {error}");
            }
        });
        Ok(key)
    }

    /// The filler: one layout's bytes, then it is complete (charged what it holds) or, failed,
    /// dropped (its adopters see the failure in its Ready words).
    fn run(&self, job: Job) {
        let (key, plan, body, layout, open, prefill) = match job {
            Job::Fill {
                key,
                plan,
                body,
                layout,
                open,
                prefill,
            } => (key, plan, body, layout, open, prefill),
            Job::Stage { key, regions } => return self.run_stage(&key, &regions),
        };
        let filled = self.fill(&plan, &layout, &open);
        drop(open); // a layout staged in part keeps the mapping in its entry; a whole one lets go
        let mut state = self.state.lock().unwrap();
        match filled {
            Ok(fill) => {
                if let Some(Slot::Open(entry)) = state.slots.get_mut(&key) {
                    entry.complete = true;
                    if let Ok(meta) = entry.fd.metadata() {
                        entry.charged = meta.blocks() * 512;
                    }
                }
                if state.ledger.fills.len() == 32 {
                    state.ledger.fills.remove(0);
                }
                state.ledger.fills.push(fill);
                state.ledger.prefills += u64::from(prefill);
                drop(state);
                if let Err(error) = self.remember(&plan, &body) {
                    eprintln!("host tier plan not remembered: {error}");
                }
            }
            Err(error) => {
                eprintln!("host tier fill of {} failed: {error}", plan.name);
                state.slots.remove(&key);
                state.ledger.failed += 1;
            }
        }
        self.filled.notify_all();
    }

    /// Stage `regions` of a layout staged in part, in order, as far as the tier has room;
    /// returns the regions now staged or staging (all of them for a whole layout). The filler
    /// copies them; adopters wait per region, and a region left a hole reads the files.
    fn stage_regions(&self, key: &str, regions: &[u32]) -> Vec<u32> {
        let mut state = self.state.lock().unwrap();
        let limit = |state: &State| {
            let host = crate::host_memory::read();
            let charged = state.charged();
            (charged, self.limit.limit(&host, charged))
        };
        let (mut charged, ceiling) = limit(&state);
        let Some(Slot::Open(entry)) = state.slots.get_mut(key) else {
            return Vec::new();
        };
        let Some(staging) = entry.staging.as_mut() else {
            return regions.to_vec();
        };
        let mut queued = Vec::new();
        for &r in regions {
            let Some(region) = staging.layout.regions.get(r as usize) else {
                continue;
            };
            if staging.staged.contains(&r) {
                continue;
            }
            if charged + region.span > ceiling {
                break; // the rest stream from the files
            }
            charged += region.span;
            entry.charged += region.span;
            staging.staged.insert(r);
            queued.push(r);
        }
        let staged = regions
            .iter()
            .copied()
            .filter(|r| staging.staged.contains(r))
            .collect();
        // Taken before the answer: the executor waits for these from now on, never reads them
        // as holes. The filler copies them.
        let taken = staging.open.take(&queued);
        if !taken.is_empty() {
            let job = Job::Stage {
                key: key.to_string(),
                regions: taken,
            };
            let _ = self.fills.lock().unwrap().send(job);
        }
        staged
    }

    /// Every remaining region of a layout staged in part, for an adopter that reads no holes:
    /// all of them when the tier has room, else none (the sparse copy does not grow). False:
    /// the layout keeps holes.
    fn stage_rest(&self, key: &str) -> bool {
        let rest: Vec<u32> = {
            let state = self.state.lock().unwrap();
            let Some(Slot::Open(Entry {
                staging: Some(staging),
                ..
            })) = state.slots.get(key)
            else {
                return true; // whole, or a window
            };
            let rest: Vec<u32> = (0..staging.layout.regions.len() as u32)
                .filter(|r| !staging.staged.contains(r))
                .collect();
            let need: u64 = rest
                .iter()
                .map(|&r| staging.layout.regions[r as usize].span)
                .sum();
            let charged = state.charged();
            if charged + need > self.limit.limit(&crate::host_memory::read(), charged) {
                return false;
            }
            rest
        };
        self.stage_regions(key, &rest).len() == rest.len()
    }

    /// The filler's half of `stage_regions`: a lease over the regions' objects, then copy.
    fn run_stage(&self, key: &str, regions: &[u32]) {
        let taken = {
            let state = self.state.lock().unwrap();
            match state.slots.get(key) {
                Some(Slot::Open(Entry {
                    staging: Some(s), ..
                })) => Some((s.open.clone(), s.layout.clone(), s.manifest.clone())),
                _ => None,
            }
        };
        let Some((open, layout, manifest)) = taken else {
            return;
        };
        let mut objects = BTreeMap::new();
        for &r in regions {
            for item in &layout.regions[r as usize].items {
                if let tensorfs_plane::layout::ItemSource::Object(range) = &item.source {
                    objects.insert(range.obj.sha256.clone(), range.obj.clone());
                }
            }
        }
        let staged = read::acquire(
            &self.store,
            &self.meta,
            &manifest,
            objects.into_values().collect(),
        )
        .map_err(failure)
        .and_then(|(lease, _)| {
            let source = Source::new(self.store.clone(), self.meta.clone(), lease, true, 0);
            let tally = IoTally::default();
            open.stage(regions, &layout, &source, self.config.fill_threads, &tally)
                .map_err(failure)
        });
        let mut state = self.state.lock().unwrap();
        if let Some(Slot::Open(entry)) = state.slots.get_mut(key) {
            if let Ok(meta) = entry.fd.metadata() {
                entry.charged = entry.charged.max(meta.blocks() * 512);
            }
        }
        if let Err(error) = staged {
            eprintln!("host tier staging of {regions:?} failed: {error}");
            state.ledger.failed += 1;
        }
        drop(state);
        self.filled.notify_all();
    }

    /// More of a layout staged in part, for `peer`: `regions` it now streams every step
    /// (`SealedStage`). Returns the regions staged or staging; the rest stay holes.
    pub fn stage(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
    ) -> io::Result<Vec<u32>> {
        if !self.staged_peer(peer)? {
            return Err(denied("this executor reads no holes"));
        }
        let regions = request.stage.unwrap_or_default().to_vec();
        let (plan, _) = Self::plan(request, plan)?;
        let (_, layout) = Self::layout(&plan, grants)?;
        Ok(self.stage_regions(&layout.digest, &regions))
    }

    /// The verified object files behind `plan`, read-only, for `peer`'s holes: the store's
    /// own descriptors (the file verified is the file read), opened per ask and handed over.
    pub fn object_files(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
    ) -> io::Result<Vec<(tensorfs_core::ids::ObjectRef, File)>> {
        if !self.staged_peer(peer)? {
            return Err(denied("this executor reads no holes"));
        }
        let (plan, _) = Self::plan(request, plan)?;
        let (read_plan, _) = Self::layout(&plan, grants)?;
        let mut objects = BTreeMap::new();
        for item in &read_plan.items {
            if let Item::Object(range) = &item.source {
                objects.insert(range.obj.sha256.clone(), range.obj.clone());
            }
        }
        objects
            .into_values()
            .map(|obj| {
                let verified = self.store.open_verified(&obj.sha256).map_err(failure)?;
                if verified.len() != obj.length {
                    return Err(failure(format!("{}: length differs", obj.sha256)));
                }
                Ok((obj, verified.into_file()))
            })
            .collect()
    }

    /// Release unheld layouts, least recently used first, until `want` bytes were charged
    /// to them or none is left; returns the bytes released. For the memory policy module.
    pub fn release(&self, want: u64) -> u64 {
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state);
        let before = state.ledger.released_bytes;
        while state.ledger.released_bytes - before < want && self.release_lru(&mut state) {}
        state.ledger.released_bytes - before
    }

    pub fn facts(&self) -> HostTierFacts {
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state);
        let host = crate::host_memory::read();
        let charged = state.charged();
        let (mut held, mut filling) = (0, 0);
        let (mut windows, mut window_bytes, mut window_read_bytes) = (0, 0, 0);
        for slot in state.slots.values() {
            match slot {
                Slot::Open(
                    e @ Entry {
                        window: Some(w), ..
                    },
                ) => {
                    windows += 1;
                    window_bytes += e.charged;
                    window_read_bytes += w.read_bytes();
                }
                Slot::Opening(n)
                | Slot::Open(Entry {
                    charged: n,
                    complete: false,
                    ..
                }) => filling += n,
                Slot::Open(e) if !e.holders.is_empty() => held += e.charged,
                Slot::Open(_) => (),
            }
        }
        HostTierFacts {
            host,
            limit: self.limit.limit(&host, charged),
            charged_bytes: charged,
            held_bytes: held,
            filling_bytes: filling,
            reserved_bytes: state.reserved.values().map(Reservation::bytes).sum(),
            windows,
            window_bytes,
            window_read_bytes,
            entries: state.slots.len(),
            peers: state.peers.len(),
            ledger: state.ledger.clone(),
        }
    }

    fn remember(&self, plan: &SealedPlan, body: &[u8]) -> io::Result<()> {
        let identity = plan.identity();
        if let Some(dir) = &self.config.plans {
            let name = format!("{:x}.json", Sha256::digest(serde_json::to_vec(&identity)?));
            let temporary = dir.join(format!(".{name}.tmp"));
            fs::write(&temporary, body)?;
            fs::rename(temporary, dir.join(name))?;
        }
        self.plans.lock().unwrap().insert(identity, body.to_vec());
        Ok(())
    }

    /// The plan the executor sent: a sealed memfd of exactly the declared bytes and digest.
    /// Open every plan of one construction (`SealedPrefetch`), in order, while its executor
    /// registers them: the filler fills them in that order and each `seal` ask finds its own.
    pub fn prefetch(
        self: &Arc<Self>,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plans: File,
    ) -> io::Result<()> {
        if !self.state.lock().unwrap().peers.contains_key(&peer) {
            return Err(denied("host tier peer is not a registered live executor"));
        }
        let body = Self::body(request, plans)?;
        let plans: Vec<serde_json::Value> = serde_json::from_slice(&body).map_err(failure)?;
        let mut queued = Vec::new();
        for value in plans {
            let body = serde_json::to_vec(&value)?;
            queued.push((
                serde_json::from_value::<SealedPlan>(value).map_err(failure)?,
                body,
            ));
        }
        // An executor that stages what streams gets layouts with nothing staged yet: it asks
        // for its streamed regions once its stages plan them.
        let stage = self.staged_peer(peer)?.then(Vec::new);
        let (tier, grants) = (self.clone(), grants.to_vec());
        std::thread::spawn(move || {
            for (plan, body) in queued {
                if let Err(error) = tier.ensure(&plan, &grants, &body, true, stage.clone()) {
                    eprintln!("host tier prefetch skipped: {error}");
                }
            }
        });
        Ok(())
    }

    fn plan(request: SealedRequest<'_>, plan: File) -> io::Result<(SealedPlan, Vec<u8>)> {
        let body = Self::body(request, plan)?;
        Ok((serde_json::from_slice(&body).map_err(failure)?, body))
    }

    /// A sealed memfd of exactly the declared bytes and digest.
    fn body(request: SealedRequest<'_>, plan: File) -> io::Result<Vec<u8>> {
        if os::seals(&plan)? & os::FULL_SEALS != os::FULL_SEALS
            || plan.metadata()?.len() != request.length
            || request.length > 64 << 20
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a sealed plan must be sealed at its declared length",
            ));
        }
        let mut body = vec![0; request.length as usize];
        plan.read_exact_at(&mut body, 0)?;
        if format!("{:x}", Sha256::digest(&body)) != request.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sealed plan digest differs",
            ));
        }
        Ok(body)
    }

    fn fill(&self, plan: &SealedPlan, layout: &Layout, open: &OpenFill) -> io::Result<Fill> {
        // A lease over the objects of what this layout stages, not its holes.
        let mut objects = BTreeMap::new();
        for (r, region) in layout.regions.iter().enumerate() {
            if open.hole(r as u32) {
                continue;
            }
            for item in &region.items {
                if let tensorfs_plane::layout::ItemSource::Object(range) = &item.source {
                    objects.insert(range.obj.sha256.clone(), range.obj.clone());
                }
            }
        }
        let started = (Instant::now(), disk_read_bytes());
        let tally = IoTally::default();
        if objects.is_empty() {
            // Nothing staged (or only inline bytes): no lease to take.
            let source = Source::files(Vec::new(), false, 0);
            open.fill(layout, &source, 1, &tally).map_err(failure)?;
            return Ok(Fill {
                name: plan.name.clone(),
                ..Fill::default()
            });
        }
        let (lease, _) = read::acquire(
            &self.store,
            &self.meta,
            hex(&plan.manifest),
            objects.into_values().collect(),
        )
        .map_err(failure)?;
        let source = Source::new(self.store.clone(), self.meta.clone(), lease, true, 0);
        open.fill(layout, &source, self.config.fill_threads, &tally)
            .map_err(failure)?;
        drop(source); // ends the lease and its descriptors: one descriptor per layout stays
        let read = |n: &std::sync::atomic::AtomicU64| n.load(std::sync::atomic::Ordering::Relaxed);
        let bytes = (0..layout.regions.len() as u32)
            .filter(|&r| !open.hole(r))
            .map(|r| layout.regions[r as usize].span)
            .sum();
        Ok(Fill {
            name: plan.name.clone(),
            bytes,
            ms: started.0.elapsed().as_secs_f64() * 1e3,
            cached_bytes: read(&tally.cached_bytes),
            direct_bytes: read(&tally.direct_bytes),
            buffered_bytes: read(&tally.buffered_bytes),
            disk_read_bytes: disk_read_bytes().saturating_sub(started.1),
        })
    }

    fn grant(&self, state: &mut State, key: &str, peer: u64) -> io::Result<File> {
        if !state.peers.contains_key(&peer) {
            return Err(denied("host tier peer is not a registered live executor"));
        }
        let Some(Slot::Open(entry)) = state.slots.get_mut(key) else {
            unreachable!()
        };
        entry.used = Instant::now();
        entry.holders.insert(peer);
        entry.granted = true;
        // A new read-only description: the recipient can neither write nor resize, and the
        // seals forbid it any other way.
        File::open(format!("/proc/self/fd/{}", entry.fd.as_raw_fd()))
    }

    /// Exited executors hold nothing; unheld layouts past their TTL go.
    fn reap(&self, state: &mut State) {
        let ended: Vec<u64> = state
            .peers
            .iter()
            .filter(|(_, peer)| os::ended(&peer.pidfd))
            .map(|(id, _)| *id)
            .collect();
        for id in &ended {
            state.peers.remove(id);
        }
        let ttl = self.config.ttl;
        let mut expired = Vec::new();
        for (key, slot) in &mut state.slots {
            if let Slot::Open(entry) = slot {
                entry.holders.retain(|h| !ended.contains(h));
                let unused = entry.used.elapsed() >= ttl;
                let done = match entry.window {
                    // A window holds nothing worth keeping: it goes with its last holder.
                    Some(_) => entry.granted || unused,
                    None => entry.complete && unused,
                };
                if entry.holders.is_empty() && done {
                    expired.push(key.clone());
                }
            }
        }
        for key in expired {
            self.release_entry(state, &key);
        }
        state
            .reserved
            .retain(|_, r| !matches!(r, Reservation::Ready { made, .. } if made.elapsed() >= ttl));
    }

    fn release_lru(&self, state: &mut State) -> bool {
        // Speculative memory goes first: a reservation no plan has claimed.
        let reservation = state
            .reserved
            .iter()
            .filter_map(|(key, r)| match r {
                Reservation::Ready { made, .. } => Some((*made, key.clone())),
                Reservation::Allocating(_) => None,
            })
            .min();
        if let Some((_, key)) = reservation {
            state.reserved.remove(&key);
            return true;
        }
        let oldest = state
            .slots
            .iter()
            .filter_map(|(key, slot)| match slot {
                Slot::Open(e) if e.holders.is_empty() && e.complete => Some((e.used, key.clone())),
                _ => None,
            })
            .min();
        match oldest {
            Some((_, key)) => {
                self.release_entry(state, &key);
                true
            }
            None => false,
        }
    }

    /// Close the tier's descriptor of an unheld layout. Its memory returns once the last
    /// mapping goes; the kernel's shared-memory count says whether it did. If not, the bytes
    /// stay charged (stranded): something still maps them.
    fn release_entry(&self, state: &mut State, key: &str) {
        let Some(Slot::Open(entry)) = state.slots.remove(key) else {
            return;
        };
        if let Some(window) = &entry.window {
            // Its staging loop unmaps the slots within a poll; nothing else maps them.
            window.stop.store(true, Ordering::Release);
            state.ledger.streamed_bytes += window.read_bytes();
            state.ledger.released += 1;
            state.ledger.released_bytes += entry.charged;
            return;
        }
        let quiet = !state.filling();
        let before = crate::host_memory::read().shmem;
        drop(entry.fd);
        let freed = (before - crate::host_memory::read().shmem).max(0) as u64;
        state.ledger.released += 1;
        state.ledger.released_bytes += entry.charged;
        state.ledger.freed_bytes += freed;
        if quiet && freed < entry.charged / 2 {
            state.ledger.stranded_bytes += entry.charged;
        }
    }
}

/// A window's staging loop: a read lease over its objects for as long as it serves.
#[allow(clippy::too_many_arguments)]
fn serve(
    store: &Arc<Store>,
    meta: &Arc<Meta>,
    manifest: &str,
    read_plan: &read::ReadPlan,
    layout: &Layout,
    open: OpenWindow,
    threads: usize,
    tally: &IoTally,
    stop: &AtomicBool,
) -> io::Result<()> {
    let mut objects = BTreeMap::new();
    for item in &read_plan.items {
        if let Item::Object(range) = &item.source {
            objects.insert(range.obj.sha256.clone(), range.obj.clone());
        }
    }
    let (lease, _) =
        read::acquire(store, meta, manifest, objects.into_values().collect()).map_err(failure)?;
    let source = Source::new(store.clone(), meta.clone(), lease, true, 0);
    open.serve(layout, &source, threads, tally, stop)
        .map_err(failure)
}

/// The SHA-256 (hex) of `files`' objects in order, one `<sha256>:<length>\n` line each: what an
/// `object_files` answer names, checked against the list the executor derives from its plan.
pub fn objects_digest(files: &[(tensorfs_core::ids::ObjectRef, File)]) -> String {
    let lines: String = files
        .iter()
        .map(|(obj, _)| format!("{}:{}\n", obj.sha256, obj.length))
        .collect();
    format!("{:x}", Sha256::digest(lines.as_bytes()))
}

/// One component's whole read plan, from its header.
fn component_plan(header: &Header, component: &str) -> io::Result<read::ReadPlan> {
    let traversal: Vec<(String, String)> = header
        .components
        .iter()
        .filter(|(name, _)| name == component)
        .flat_map(|(name, tensors)| {
            tensors
                .iter()
                .map(move |(key, _)| (name.clone(), key.clone()))
        })
        .collect();
    read::plan_for_traversal(header, &traversal, &[component.to_string()], 4 << 20)
        .map_err(failure)
}
