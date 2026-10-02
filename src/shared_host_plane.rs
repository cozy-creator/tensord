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
    collections::BTreeMap,
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::fs::MetadataExt,
    },
    path::Path,
    sync::{Arc, Mutex},
};
use tensorfs_core::{
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
#[derive(Clone, Copy, Debug)]
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
            }),
        }))
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
                self.plane
                    .want(ws, Tier::Pinned, Some(batch), 0, true)
                    .map_err(failure)?
                    .wait()
                    .map_err(failure)?;
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
        if frame.kind != Kind::HostTier {
            return None;
        }
        Some(self.tier(peer, frame, descriptor))
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
        Ok(HostCharge {
            budget_bytes: state.budget,
            backing_bytes,
            active_backing_bytes,
            over_budget_bytes: backing_bytes.saturating_sub(state.budget),
            entries: state.entries.len(),
            recipients: state.recipients.len(),
            filled_bytes: self.plane.stats().host.counters.fill_bytes,
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
