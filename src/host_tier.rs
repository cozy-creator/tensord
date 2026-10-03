//! Degree 1: the machine's host weights. Each weight set's layout is filled once through
//! TensorFS's verified read path into a memfd, sealed (nobody can write, resize or punch it)
//! and kept across executors and model switches; executors adopt it read-only
//! (`host_tiers.sealed/1`, `Plane.register_sealed`). The machine holds one descriptor per
//! layout. The tier's size follows live host headroom (`host_memory`), read at every prepare;
//! unheld layouts are released least recently used first, or after `ttl` unused. How much of
//! the headroom the tier may take is `TierLimit`'s decision (the memory policy module's).
//! No lock is held across a fill: one model's fill never stalls another executor.
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
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};
use tensorfs_core::{
    header::Header,
    meta::Meta,
    read::{self, Source as Item},
    store::Store,
};
use tensorfs_plane::{
    host,
    io::{IoTally, Source},
    layout::Layout,
};

#[derive(Clone, Debug)]
pub struct HostTierConfig {
    /// Threads of one fill (read + copy). Default: the CPUs this process may use, at most 16.
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
            fill_threads: cpus.clamp(1, 16),
            ttl: Duration::from_secs(30 * 60),
            plans: None,
        }
    }
}

/// The most the tier may charge, given live host memory and what it charges now.
pub trait TierLimit: Send + Sync {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64;
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
#[derive(Debug, Deserialize)]
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
    /// Asks refused because the tier could not make room: that weight set read the store.
    pub no_room: u64,
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
    pub entries: usize,
    pub peers: usize,
    #[serde(flatten)]
    pub ledger: Ledger,
}

struct Entry {
    fd: File,
    charged: u64,
    used: Instant,
    holders: BTreeSet<u64>,
}
enum Slot {
    Filling(u64),
    Ready(Entry),
}
#[derive(Default)]
struct State {
    /// By TensorFS layout digest: the same bytes at the same offsets, whoever asks.
    slots: BTreeMap<String, Slot>,
    peers: BTreeMap<u64, File>,
    next_peer: u64,
    ledger: Ledger,
}
impl State {
    fn charged(&self) -> u64 {
        self.slots
            .values()
            .map(|s| match s {
                Slot::Filling(n) => *n,
                Slot::Ready(e) => e.charged,
            })
            .sum::<u64>()
            + self.ledger.stranded_bytes
    }
    fn filling(&self) -> bool {
        self.slots.values().any(|s| matches!(s, Slot::Filling(_)))
    }
}

pub struct HostTier {
    store: Arc<Store>,
    meta: Arc<Meta>,
    config: HostTierConfig,
    limit: Box<dyn TierLimit>,
    state: Mutex<State>,
    filled: Condvar,
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
        Ok(Arc::new(Self {
            store,
            meta,
            config,
            limit,
            state: Mutex::default(),
            filled: Condvar::new(),
            plans: Mutex::new(plans),
        }))
    }

    /// Refill, in the background, every layout of these grants a remembered plan describes:
    /// the executor's asks then find them filled, or wait for the fill already under way.
    pub fn prefill(self: &Arc<Self>, grants: Vec<HostGrant>) {
        let bodies: Vec<Vec<u8>> = self
            .plans
            .lock()
            .unwrap()
            .iter()
            .filter(|((manifest, components), _)| {
                grants.iter().any(|g| {
                    hex(&g.manifest) == manifest
                        && components.iter().all(|c| g.components.contains(c))
                })
            })
            .map(|(_, body)| body.clone())
            .collect();
        if bodies.is_empty() {
            return;
        }
        let tier = self.clone();
        std::thread::spawn(move || {
            for body in bodies {
                let filled = serde_json::from_slice::<SealedPlan>(&body)
                    .map_err(failure)
                    .and_then(|plan| tier.ensure(&plan, &grants, &body, true).map(|_| ()));
                if let Err(error) = filled {
                    eprintln!("host tier prefill skipped: {error}");
                }
            }
        });
    }

    /// An executor, by its pidfd: what it adopted stays held until that exact process exits.
    pub fn register_peer(&self, pidfd: File) -> u64 {
        let mut state = self.state.lock().unwrap();
        state.next_peer += 1;
        let id = state.next_peer;
        state.peers.insert(id, pidfd);
        id
    }

    /// One weight set's sealed layout for `peer`, read-only: from the tier, or filled now.
    /// None when the tier cannot make room: the executor reads the store itself.
    pub fn seal(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
    ) -> io::Result<Option<File>> {
        if !self.state.lock().unwrap().peers.contains_key(&peer) {
            return Err(denied("host tier peer is not a registered live executor"));
        }
        let (plan, body) = Self::plan(request, plan)?;
        match self.ensure(&plan, grants, &body, false)? {
            Some(key) => self
                .grant(&mut self.state.lock().unwrap(), &key, peer)
                .map(Some),
            None => Ok(None),
        }
    }

    /// The layout `plan` names, filled now unless the tier holds it or a fill of it is under
    /// way (then waited for). Its key, or None when the tier cannot make room.
    fn ensure(
        &self,
        plan: &SealedPlan,
        grants: &[HostGrant],
        body: &[u8],
        prefill: bool,
    ) -> io::Result<Option<String>> {
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
        let key = layout.digest.clone();
        let mut state = self.state.lock().unwrap();
        loop {
            self.reap(&mut state);
            match state.slots.get(&key) {
                Some(Slot::Ready(_)) => {
                    if !prefill {
                        state.ledger.hits += 1;
                    }
                    return Ok(Some(key));
                }
                Some(Slot::Filling(_)) => state = self.filled.wait(state).unwrap(),
                None => break,
            }
        }
        // Admission over live headroom, read now: release unheld layouts, oldest first.
        let need = layout.nbytes;
        loop {
            let host = crate::host_memory::read();
            let charged = state.charged();
            if charged + need <= self.limit.limit(&host, charged) {
                break;
            }
            if !self.release_lru(&mut state) {
                state.ledger.no_room += 1;
                return Ok(None);
            }
        }
        state.slots.insert(key.clone(), Slot::Filling(need));
        drop(state);
        let filled = self.fill(plan, &read_plan, &layout);
        if filled.is_ok() {
            if let Err(error) = self.remember(plan, body) {
                eprintln!("host tier plan not remembered: {error}");
            }
        }
        let mut state = self.state.lock().unwrap();
        let result = match filled {
            Ok((fd, fill)) => {
                let fd = File::from(fd);
                let charged = fd.metadata()?.blocks() * 512;
                state.slots.insert(
                    key.clone(),
                    Slot::Ready(Entry {
                        fd,
                        charged,
                        used: Instant::now(),
                        holders: BTreeSet::new(),
                    }),
                );
                if state.ledger.fills.len() == 32 {
                    state.ledger.fills.remove(0);
                }
                state.ledger.fills.push(fill);
                state.ledger.prefills += u64::from(prefill);
                Ok(Some(key))
            }
            Err(error) => {
                state.slots.remove(&key);
                Err(error)
            }
        };
        self.filled.notify_all();
        result
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
        for slot in state.slots.values() {
            match slot {
                Slot::Filling(n) => filling += n,
                Slot::Ready(e) if !e.holders.is_empty() => held += e.charged,
                Slot::Ready(_) => (),
            }
        }
        HostTierFacts {
            host,
            limit: self.limit.limit(&host, charged),
            charged_bytes: charged,
            held_bytes: held,
            filling_bytes: filling,
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
    fn plan(request: SealedRequest<'_>, plan: File) -> io::Result<(SealedPlan, Vec<u8>)> {
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
        Ok((serde_json::from_slice(&body).map_err(failure)?, body))
    }

    fn fill(
        &self,
        plan: &SealedPlan,
        read_plan: &read::ReadPlan,
        layout: &Layout,
    ) -> io::Result<(OwnedFd, Fill)> {
        let mut objects = BTreeMap::new();
        for item in &read_plan.items {
            if let Item::Object(range) = &item.source {
                objects.insert(range.obj.sha256.clone(), range.obj.clone());
            }
        }
        let started = (Instant::now(), disk_read_bytes());
        let (lease, _) = read::acquire(
            &self.store,
            &self.meta,
            hex(&plan.manifest),
            objects.into_values().collect(),
        )
        .map_err(failure)?;
        let source = Source::new(self.store.clone(), self.meta.clone(), lease, true, 0);
        let tally = IoTally::default();
        let fd = host::fill_sealed(
            &plan.name,
            layout,
            &source,
            self.config.fill_threads,
            &tally,
        )
        .map_err(failure)?;
        drop(source); // ends the lease and its descriptors: one descriptor per layout stays
        let read = |n: &std::sync::atomic::AtomicU64| n.load(std::sync::atomic::Ordering::Relaxed);
        let fill = Fill {
            name: plan.name.clone(),
            bytes: layout.nbytes,
            ms: started.0.elapsed().as_secs_f64() * 1e3,
            cached_bytes: read(&tally.cached_bytes),
            direct_bytes: read(&tally.direct_bytes),
            buffered_bytes: read(&tally.buffered_bytes),
            disk_read_bytes: disk_read_bytes().saturating_sub(started.1),
        };
        Ok((fd, fill))
    }

    fn grant(&self, state: &mut State, key: &str, peer: u64) -> io::Result<File> {
        if !state.peers.contains_key(&peer) {
            return Err(denied("host tier peer is not a registered live executor"));
        }
        let Some(Slot::Ready(entry)) = state.slots.get_mut(key) else {
            unreachable!()
        };
        entry.used = Instant::now();
        entry.holders.insert(peer);
        // A new read-only description: the recipient can neither write nor resize, and the
        // seals forbid it any other way.
        File::open(format!("/proc/self/fd/{}", entry.fd.as_raw_fd()))
    }

    /// Exited executors hold nothing; unheld layouts past their TTL go.
    fn reap(&self, state: &mut State) {
        let ended: Vec<u64> = state
            .peers
            .iter()
            .filter(|(_, pidfd)| os::ended(pidfd))
            .map(|(id, _)| *id)
            .collect();
        for id in &ended {
            state.peers.remove(id);
        }
        let ttl = self.config.ttl;
        let mut expired = Vec::new();
        for (key, slot) in &mut state.slots {
            if let Slot::Ready(entry) = slot {
                entry.holders.retain(|h| !ended.contains(h));
                if entry.holders.is_empty() && entry.used.elapsed() >= ttl {
                    expired.push(key.clone());
                }
            }
        }
        for key in expired {
            self.release_entry(state, &key);
        }
    }

    fn release_lru(&self, state: &mut State) -> bool {
        let oldest = state
            .slots
            .iter()
            .filter_map(|(key, slot)| match slot {
                Slot::Ready(e) if e.holders.is_empty() => Some((e.used, key.clone())),
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
        let Some(Slot::Ready(entry)) = state.slots.remove(key) else {
            return;
        };
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
