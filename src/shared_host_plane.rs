//! Machine-created TensorFS host tiers. Trusted preparation supplies the exact native
//! plan/layout; a peer's cache name never grants model or actor authority. No CUDA context
//! exists here. Recipient holds end only on observed pidfd exit, not socket loss or age.
use crate::{
    device_executor::{Answer, Frame, Kind},
    execution::process_birth,
    journal::ProcessBirth,
    os,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{self, Read},
    os::unix::net::UnixStream,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::fs::{FileExt, MetadataExt},
    },
    path::Path,
    sync::{Arc, Mutex},
};
use tensorfs_core::{
    header::Header,
    meta::Meta,
    read::{self, ReadPlan, Source as ReadSource},
    store::Store,
};
use tensorfs_plane::{host, layout::Layout, Plane, PlaneConfig, Tier, WsId};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct HostScope {
    pub actor: String,
    pub plan: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct HostKey {
    pub name: String,
    pub layout: String,
}
impl HostKey {
    pub fn for_regions(name: String, manifest: &str, regions: &[Vec<String>]) -> io::Result<Self> {
        #[derive(Serialize)]
        struct Identity<'a> {
            manifest: &'a str,
            regions: &'a [Vec<String>],
        }
        let bytes =
            serde_json_canonicalizer::to_vec(&Identity { manifest, regions }).map_err(failure)?;
        Ok(Self {
            name,
            layout: format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes)),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPeer {
    pub actor: String,
    pub plan: String,
    pub birth: ProcessBirth,
}
impl HostPeer {
    fn scope(&self) -> HostScope {
        HostScope {
            actor: self.actor.clone(),
            plan: self.plan.clone(),
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct HostConfig {
    pub budget_bytes: u64,
    pub readers: usize,
    pub max_entries: usize,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct HostCharge {
    pub budget_bytes: u64,
    /// Kernel st_blocks, including the native state page; never logical per-executor sums.
    pub backing_bytes: u64,
    pub active_backing_bytes: u64,
    pub over_budget_bytes: u64,
    pub entries: usize,
    pub native_allocations: usize,
    pub recipients: usize,
    pub filled_bytes: u64,
}

// This is an in-process trusted API, not an executor-provided plan. The policy/metadata
// owner must authorize manifest, traversal and region membership before calling prepare.
pub struct HostPreparation<'a> {
    pub scope: &'a HostScope,
    pub key: &'a HostKey,
    pub manifest: &'a str,
    pub plan: &'a ReadPlan,
    pub regions: &'a [Vec<String>],
}
/// Sealed FD payload emitted by the existing SDK model library, not authored code.
#[derive(Debug, Deserialize, Serialize)]
pub struct HostTierPlan {
    pub manifest: String,
    pub name: String,
    pub layout: String,
    pub traversal: Vec<(String, String)>,
    pub components: Vec<String>,
    pub regions: Vec<Vec<String>>,
    #[serde(default)]
    pub parts: Vec<String>,
}
#[derive(Debug)]
pub struct HostTierRegistration {
    pub manifest: String,
    pub name: String,
    pub layout: String,
    pub sha256: String,
    pub length: u64,
}
struct AuthorizedModel {
    header: Header,
    components: BTreeSet<String>,
}
// pread makes verification/decoding independent of a sender's shared OFD seek position.
struct DescriptorReader<'a> {
    file: &'a File,
    offset: u64,
}
impl Read for DescriptorReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read_at(bytes, self.offset)?;
        self.offset += n as u64;
        Ok(n)
    }
}
type CacheKey = (HostScope, HostKey);
struct Entry {
    ws: WsId,
    file: File,
    layout: String,
    manifest: String,
    used: u64,
}
struct Recipient {
    peer: HostPeer,
    exit: File,
    holds: BTreeMap<CacheKey, OwnedFd>,
}
struct State {
    budget: u64,
    entries: BTreeMap<CacheKey, Entry>,
    recipients: Vec<Recipient>,
    clock: u64,
    authorized: BTreeMap<(HostScope, String), AuthorizedModel>,
}
pub struct SharedHostPlane {
    plane: Plane,
    store: Arc<Store>,
    meta: Arc<Meta>,
    config: HostConfig,
    state: Mutex<State>,
}
fn failure(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
fn duplicate(fd: i32) -> io::Result<File> {
    // SAFETY: duplicate a borrowed live descriptor; successful call creates our ownership.
    let raw = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(raw) })
}
fn charge(file: &File) -> io::Result<u64> {
    Ok(file.metadata()?.blocks() * 512)
}
fn active(state: &State, key: &CacheKey) -> bool {
    state.recipients.iter().any(|r| r.holds.contains_key(key))
}

impl SharedHostPlane {
    pub fn new(root: &Path, config: HostConfig) -> io::Result<Arc<Self>> {
        if config.readers == 0 || config.readers > 4 || config.max_entries == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host readers must be 1..4 and entry bound positive",
            ));
        }
        let store = Arc::new(Store::open(root).map_err(failure)?);
        let meta = Arc::new(Meta::open(&store).map_err(failure)?);
        let plane = Plane::open(PlaneConfig {
            devices: Vec::new(),
            readers: config.readers,
            ..PlaneConfig::default()
        })
        .map_err(failure)?;
        // The machine owns the admission/physical ledger; native books never evict an active
        // recipient's allocation behind that ledger. Whole-entry reclaim is done below.
        plane.set_pinned_budget(u64::MAX).map_err(failure)?;
        Ok(Arc::new(Self {
            plane,
            store,
            meta,
            config,
            state: Mutex::new(State {
                budget: config.budget_bytes,
                entries: BTreeMap::new(),
                recipients: Vec::new(),
                clock: 0,
                authorized: BTreeMap::new(),
            }),
        }))
    }

    /// CPU/service fixture adapter: policy scope is supplied by the authenticated owner;
    /// Linux SO_PEERPIDFD pins the actual connecting peer rather than a reused numeric PID.
    pub fn register_socket(&self, scope: HostScope, stream: &UnixStream) -> io::Result<HostPeer> {
        let exit = os::peer_pidfd(stream)?;
        let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{}", exit.as_raw_fd()))?;
        let pid = info
            .lines()
            .find_map(|line| line.strip_prefix("Pid:\t"))
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or_else(|| failure("peer pidfd did not identify a live process"))?;
        let peer = HostPeer {
            actor: scope.actor,
            plan: scope.plan,
            birth: process_birth(pid)?,
        };
        self.register_peer(peer.clone(), exit)?;
        Ok(peer)
    }

    /// Bind to the actual spawned process before sending any host capability. The supplied
    /// pidfd is checked against both its kernel fdinfo PID and the previously measured birth.
    pub fn register_peer(&self, peer: HostPeer, exit: File) -> io::Result<()> {
        let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{}", exit.as_raw_fd()))?;
        let pid = info
            .lines()
            .find_map(|line| line.strip_prefix("Pid:\t"))
            .and_then(|v| v.parse::<u32>().ok());
        if pid != Some(peer.birth.pid)
            || os::ended(&exit)
            || process_birth(peer.birth.pid)? != peer.birth
            || peer.actor.is_empty()
            || peer.plan.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "host peer is not the recorded live process birth",
            ));
        }
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state)?;
        if state.recipients.iter().any(|r| r.peer == peer) {
            return Ok(());
        }
        state.recipients.push(Recipient {
            peer,
            exit,
            holds: BTreeMap::new(),
        });
        Ok(())
    }

    /// Called only by the trusted metadata owner with its selected, verified header and
    /// allowed components. A socket peer cannot expand this authority by naming a manifest.
    pub fn authorize(
        &self,
        scope: HostScope,
        manifest: String,
        header: Header,
        components: Vec<String>,
    ) -> io::Result<()> {
        let components: BTreeSet<String> = components.into_iter().collect();
        if scope.actor.is_empty()
            || scope.plan.is_empty()
            || components.is_empty()
            || components
                .iter()
                .any(|c| !header.components.iter().any(|(name, _)| name == c))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host authorization requires selected header components and actor/plan",
            ));
        }
        let mut state = self.state.lock().unwrap();
        let key = (scope, manifest);
        if let Some(previous) = state.authorized.get(&key) {
            if previous.components != components
                || previous.header.canonical_bytes().map_err(failure)?
                    != header.canonical_bytes().map_err(failure)?
            {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "immutable host plan authorization changed",
                ));
            }
            return Ok(());
        }
        state
            .authorized
            .insert(key, AuthorizedModel { header, components });
        Ok(())
    }

    /// Fully verified typed FD registration, deliberately outside the 64KiB control cap.
    /// Success with cached=false is disk fallback, including zero budget or active tenants.
    pub fn prepare_peer(
        &self,
        peer: &HostPeer,
        request: HostTierRegistration,
        descriptor: File,
    ) -> io::Result<bool> {
        use sha2::{Digest, Sha256};
        if descriptor.metadata()?.len() != request.length
            || os::seals(&descriptor)? & os::FULL_SEALS != os::FULL_SEALS
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host partition descriptor must be sealed at its declared length",
            ));
        }
        let mut hash = Sha256::new();
        let mut reader = DescriptorReader {
            file: &descriptor,
            offset: 0,
        };
        let mut buffer = [0u8; 64 << 10];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        if format!("{:x}", hash.finalize()) != request.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "host partition descriptor checksum differs",
            ));
        }
        let payload: HostTierPlan = serde_json::from_reader(DescriptorReader {
            file: &descriptor,
            offset: 0,
        })
        .map_err(failure)?;
        if (
            payload.manifest.as_str(),
            payload.name.as_str(),
            payload.layout.as_str(),
        ) != (
            request.manifest.as_str(),
            request.name.as_str(),
            request.layout.as_str(),
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host partition envelope differs from its typed payload",
            ));
        }
        let scope = peer.scope();
        let plan = {
            let mut state = self.state.lock().unwrap();
            self.reap(&mut state)?;
            if !state.recipients.iter().any(|r| r.peer == *peer) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "host partition sender birth is not registered",
                ));
            }
            let authorized = state
                .authorized
                .get(&(scope.clone(), payload.manifest.clone()))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "host manifest was not selected for this actor/plan",
                    )
                })?;
            if payload.components.is_empty()
                || payload
                    .components
                    .iter()
                    .any(|c| !authorized.components.contains(c))
                || payload
                    .traversal
                    .iter()
                    .any(|(c, _)| !authorized.components.contains(c))
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "host partition leaves authorized model components",
                ));
            }
            let plan = read::plan_for_traversal(
                &authorized.header,
                &payload.traversal,
                &payload.components,
                4 << 20,
            )
            .map_err(failure)?;
            if payload.parts.is_empty() {
                plan
            } else {
                plan.select(&payload.parts).map_err(failure)?
            }
        };
        let key = HostKey {
            name: payload.name,
            layout: payload.layout,
        };
        self.prepare(HostPreparation {
            scope: &scope,
            key: &key,
            manifest: &payload.manifest,
            plan: &plan,
            regions: &payload.regions,
        })
    }

    /// Populate one exact owner-authorized plan. A false result is an optional cache miss:
    /// the existing source/disk path remains usable regardless of model/request size.
    pub fn prepare(&self, preparation: HostPreparation<'_>) -> io::Result<bool> {
        let HostPreparation {
            scope,
            key,
            manifest,
            plan,
            regions,
        } = preparation;
        if scope.actor.is_empty()
            || scope.plan.is_empty()
            || key.name.is_empty()
            || key.layout.is_empty()
            || plan.items.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host preparation requires explicit trusted scope/key/plan",
            ));
        }
        if HostKey::for_regions(key.name.clone(), manifest, regions)?.layout != key.layout {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "host key does not identify the prepared manifest/region roster",
            ));
        }
        let layout = Layout::build(plan, regions).map_err(failure)?;
        let cache_key = (scope.clone(), key.clone());
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state)?;
        if let Some(entry) = state.entries.get(&cache_key) {
            if entry.layout != layout.digest || entry.manifest != manifest {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "host key already binds a different native layout",
                ));
            }
            return Ok(true);
        }
        if state.budget == 0 || layout.nbytes >= state.budget {
            return Ok(false);
        }
        self.make_room(&mut state, layout.nbytes, true)?;
        if self.backing(&state)?.saturating_add(layout.nbytes) > state.budget
            || state.entries.len() >= self.config.max_entries
        {
            return Ok(false);
        }
        let mut objects = BTreeMap::new();
        for item in &plan.items {
            if let ReadSource::Object(range) = &item.source {
                objects.insert(range.obj.sha256.clone(), range.obj.clone());
            }
        }
        let (lease, _) = read::acquire(
            &self.store,
            &self.meta,
            manifest,
            objects.into_values().collect(),
        )
        .map_err(failure)?;
        let source = self
            .plane
            .source(self.store.clone(), self.meta.clone(), lease);
        let ws = self
            .plane
            .register(&key.name, source, plan, regions, None)
            .map_err(failure)?;
        let result = (|| {
            let file = duplicate(self.plane.host_fd(ws).map_err(failure)?)?;
            // Native allocation knows its own metadata size: don't copy its state-page format.
            let required = file.metadata()?.len();
            self.make_room(&mut state, required, true)?;
            if self.backing(&state)?.saturating_add(required) > state.budget
                || state.entries.len() >= self.config.max_entries
            {
                return Ok(None);
            }
            // Synchronous bounded region batches reuse TensorFS's reader pool. No unbounded
            // per-region future queue or private allocator/read implementation.
            let ids: Vec<u32> = (0..layout.regions.len() as u32).collect();
            for batch in ids.chunks(self.config.readers) {
                let mut tickets = Vec::new();
                let mut failed = None;
                for region in batch {
                    match self.plane.want(ws, Tier::Pinned, Some(&[*region]), 0, true) {
                        Ok(ticket) => tickets.push(ticket),
                        Err(error) => {
                            failed = Some(failure(error));
                            break;
                        }
                    }
                }
                // A failed member does not prove other native readers stopped writing. Drain
                // every started region before close_ws may unmap/punch partial backing.
                for ticket in tickets {
                    if let Err(error) = ticket.wait() {
                        failed.get_or_insert_with(|| failure(error));
                    }
                }
                if let Some(error) = failed {
                    return Err(error);
                }
            }
            if self.backing(&state)?.saturating_add(charge(&file)?) > state.budget {
                return Ok(None);
            }
            Ok(Some(file))
        })();
        match result {
            Ok(Some(file)) => {
                state.clock += 1;
                let used = state.clock;
                state.entries.insert(
                    cache_key,
                    Entry {
                        ws,
                        file,
                        layout: layout.digest,
                        manifest: manifest.to_owned(),
                        used,
                    },
                );
                Ok(true)
            }
            Ok(None) => {
                self.plane.close_ws(ws).map_err(failure)?;
                Ok(false)
            }
            Err(error) => {
                self.plane.close_ws(ws).map_err(failure)?;
                Err(error)
            }
        }
    }

    /// Existing SDK HostTier wire adapter. No untrusted offer becomes a cache allocation.
    /// The supplied peer was authenticated by the pool; names/layout only select among that
    /// actor+immutable plan's pre-authorized entries.
    pub fn request(
        &self,
        peer: &HostPeer,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> Option<io::Result<(Answer, Option<File>)>> {
        match frame.kind {
            Kind::HostTier => Some(self.tier(peer, frame, descriptor)),
            Kind::HostTierPrepare => Some((|| {
                let descriptor = descriptor.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "host partition registration omitted its sealed fd",
                    )
                })?;
                self.prepare_peer(
                    peer,
                    HostTierRegistration {
                        manifest: frame.manifest.clone(),
                        name: frame.name.clone(),
                        layout: frame.layout.clone(),
                        sha256: frame.sha256.clone(),
                        length: frame.length,
                    },
                    descriptor,
                )?;
                let mut answer = Answer::unavailable(frame.seq);
                answer.ok = true;
                answer.code.clear();
                answer.detail.clear();
                Ok((answer, None))
            })()),
            _ => None,
        }
    }

    fn tier(
        &self,
        peer: &HostPeer,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        drop(descriptor);
        let mut answer = Answer::unavailable(frame.seq);
        answer.ok = true;
        answer.code.clear();
        answer.detail.clear();
        if frame.offer {
            answer.detail = "machine allocations are prepared by the trusted plan owner".into();
            return Ok((answer, None));
        }
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state)?;
        let recipient = state
            .recipients
            .iter()
            .position(|r| r.peer == *peer)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "host recipient birth was not registered or has exited",
                )
            })?;
        let key = (
            peer.scope(),
            HostKey {
                name: frame.name.clone(),
                layout: frame.layout.clone(),
            },
        );
        if !state.entries.contains_key(&key) {
            return Ok((answer, None));
        }
        if !state.recipients[recipient].holds.contains_key(&key) {
            let fd = state.entries[&key].file.as_raw_fd();
            let held = host::hold(fd).map_err(failure)?;
            state.recipients[recipient].holds.insert(key.clone(), held);
        }
        state.clock += 1;
        let used = state.clock;
        let entry = state.entries.get_mut(&key).unwrap();
        entry.used = used;
        answer.held = true;
        Ok((
            answer,
            Some(duplicate(
                state.recipients[recipient].holds[&key].as_raw_fd(),
            )?),
        ))
    }

    /// A lower budget does not authorize reclaim of memory a live GPU recipient might read.
    /// Report that physical charge explicitly; future preparation falls back to disk.
    pub fn set_budget(&self, bytes: u64) -> io::Result<HostCharge> {
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state)?;
        state.budget = bytes;
        self.make_room(&mut state, 0, false)?;
        self.facts(&state)
    }
    pub fn stats(&self) -> io::Result<HostCharge> {
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state)?;
        self.facts(&state)
    }
    fn facts(&self, state: &State) -> io::Result<HostCharge> {
        let backing_bytes = self.backing(state)?;
        let mut active_backing_bytes = 0;
        for (key, entry) in &state.entries {
            if active(state, key) {
                active_backing_bytes += charge(&entry.file)?;
            }
        }
        let native = self.plane.stats();
        Ok(HostCharge {
            budget_bytes: state.budget,
            backing_bytes,
            active_backing_bytes,
            over_budget_bytes: backing_bytes.saturating_sub(state.budget),
            entries: state.entries.len(),
            native_allocations: native.sets.len(),
            recipients: state.recipients.len(),
            filled_bytes: native.host.counters.fill_bytes,
        })
    }
    fn backing(&self, state: &State) -> io::Result<u64> {
        state
            .entries
            .values()
            .try_fold(0u64, |sum, e| Ok(sum + charge(&e.file)?))
    }
    fn reap(&self, state: &mut State) -> io::Result<()> {
        // No timeout or cancellation inference. Native claims held by a dead process have
        // already gone; our pre-delivery holds end only after pidfd proves that exact exit.
        let mut index = 0;
        let mut reaped = false;
        while index < state.recipients.len() {
            if os::ended(&state.recipients[index].exit) {
                // SCM_RIGHTS/dup retain the same open file description. Merely dropping
                // our OwnedFd does not end that description's whole-file claim while an
                // outbound copy remains open. End this exact dead recipient's claim, not
                // another live recipient's independent description or native region lock.
                for hold in state.recipients[index].holds.values() {
                    let lock = libc::flock {
                        l_type: libc::F_UNLCK as _,
                        l_whence: libc::SEEK_SET as _,
                        l_start: 0,
                        l_len: 0,
                        l_pid: 0,
                    };
                    // SAFETY: our known native whole-file OFD hold; pidfd above proves the
                    // exact recipient has exited. Other recipients use different OFDs.
                    if unsafe { libc::fcntl(hold.as_raw_fd(), libc::F_OFD_SETLK, &lock) } < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                let recipient = state.recipients.remove(index);
                reaped = true;
                for (_, hold) in recipient.holds {
                    host::release_hold(hold).map_err(failure)?;
                }
            } else {
                index += 1;
            }
        }
        if reaped {
            self.make_room(state, 0, false)?;
        }
        Ok(())
    }
    fn make_room(&self, state: &mut State, need: u64, adding: bool) -> io::Result<()> {
        while self.backing(state)?.saturating_add(need) > state.budget
            || (adding && state.entries.len() >= self.config.max_entries)
        {
            let candidate = state
                .entries
                .iter()
                .filter(|(key, _)| !active(state, key))
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| key.clone());
            let Some(key) = candidate else {
                break;
            };
            // No live recipient may DMA/compute from this allocation. TensorFS drops its
            // own claims and punches the last-owned regions before our ledger forgets them.
            let entry = &state.entries[&key];
            self.plane.close_ws(entry.ws).map_err(failure)?;
            state.entries.remove(&key);
        }
        Ok(())
    }
}
impl Drop for SharedHostPlane {
    fn drop(&mut self) {
        // Native per-process claims protect living executor mappings even if this owner
        // disappears. GPU registration/unregister fencing belongs to that executor plane.
        let state = self.state.get_mut().unwrap();
        for recipient in state.recipients.drain(..) {
            for (_, held) in recipient.holds {
                drop(held);
            }
        }
        let _ = self.plane.close();
    }
}
