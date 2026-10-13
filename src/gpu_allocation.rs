//! GPU allocations, as Slurm's: a root run (a request with no parent) holds a contiguous lane
//! of GPUs from its first GPU step until `Context.release_gpus()` or its executor's exit, and
//! its calls (steps) run only there, one call per GPU at a time. Roots wait in acceptance
//! order: from its first request for GPU work (a child call or prefetch) until it binds, an
//! older root claims the lane it needs from younger ones (`Claims`). A released root's next
//! step queues at its original place and never preempts. An accelerator job keeps one GPU of
//! its root's lane for its life; only its family (it and its descendants) call there.
use std::{
    collections::BTreeMap,
    ops::Range,
    sync::{Arc, Condvar, Mutex},
};

/// Per GPU, the older waiting root that needs it in this dispatch pass.
pub(crate) type Claims = [Option<u64>];

pub(crate) struct Allocations {
    table: Mutex<Table>,
    freed: Condvar,
}
struct Table {
    gpus: Vec<Gpu>,
    roots: BTreeMap<u64, Root>,
}
#[derive(Default)]
struct Gpu {
    /// The root whose lane this GPU is.
    root: Option<u64>,
    /// The accelerator job keeping it, and how many jobs of its family do.
    family: Option<String>,
    keeps: usize,
    calling: bool,
    /// The accelerator family of the call running there, if any.
    caller: Option<String>,
}
struct Root {
    width: usize,
    lane: Option<Range<usize>>,
    released: bool,
    /// It asked for GPU work: from then on it claims its lane until bound.
    wants: bool,
    /// A wait was reported and no lane bound since.
    waiting: bool,
    holders: usize,
}
/// A root's lane, or the root it waits behind. `new`: the first answer since it changed.
pub(crate) enum Lane {
    Held { lane: Range<usize>, new: bool },
    Waiting { behind: Option<u64>, new: bool },
}
/// What a released root's executor holds: none of its lane, until its next step binds one.
pub(crate) struct Hold {
    table: Arc<Allocations>,
    root: u64,
}
pub(crate) struct Permit {
    table: Arc<Allocations>,
    devices: Range<usize>,
    keep: bool,
}

impl Table {
    fn free(&self, device: usize, root: u64, claims: &Claims) -> bool {
        let gpu = &self.gpus[device];
        gpu.root.is_none() && gpu.family.is_none() && !gpu.calling
            && claims.get(device).copied().flatten().is_none_or(|claimer| claimer == root)
    }
    /// Who besides `root` holds or needs a GPU of `lane`.
    fn other(&self, lane: Range<usize>, root: u64, claims: &Claims) -> impl Iterator<Item = u64> + '_ {
        let claims: Vec<_> = lane.clone().map(|device| claims.get(device).copied().flatten()).collect();
        lane.zip(claims).filter_map(move |(device, claim)| claim.or(self.gpus[device].root)).filter(move |other| *other != root)
    }
}

impl Allocations {
    pub(crate) fn new(width: usize) -> Arc<Self> {
        let gpus = (0..width).map(|_| Gpu::default()).collect();
        Arc::new(Self { table: Mutex::new(Table { gpus, roots: BTreeMap::new() }), freed: Condvar::new() })
    }
    /// `root` holds an allocation `width` GPUs wide, unbound until its first step, for as long
    /// as the hold lives.
    pub(crate) fn register(self: &Arc<Self>, root: u64, width: usize) -> Hold {
        let width = width.clamp(1, self.table.lock().unwrap().gpus.len().max(1));
        let mut table = self.table.lock().unwrap();
        table.roots.entry(root)
            .or_insert(Root { width, lane: None, released: false, wants: false, waiting: false, holders: 0 })
            .holders += 1;
        Hold { table: self.clone(), root }
    }
    pub(crate) fn registered(&self, root: u64) -> bool {
        self.table.lock().unwrap().roots.contains_key(&root)
    }
    /// Its lane while bound; `Some(None)` once released, `None` while not yet bound.
    pub(crate) fn bound(&self, root: u64) -> Option<Option<Range<usize>>> {
        let table = self.table.lock().unwrap();
        let root = table.roots.get(&root)?;
        match (&root.lane, root.released) {
            (Some(lane), _) => Some(Some(lane.clone())),
            (None, true) => Some(None),
            (None, false) => None,
        }
    }
    /// `root` asked for GPU work (a child call or prefetch): it keeps its place from now on.
    pub(crate) fn want(&self, root: u64) {
        if let Some(root) = self.table.lock().unwrap().roots.get_mut(&root) {
            root.wants = true;
        }
    }
    /// Roots that want GPUs, unbound and not released, oldest first, with their widths.
    pub(crate) fn pending(&self) -> Vec<(u64, usize)> {
        let table = self.table.lock().unwrap();
        table.roots.iter().filter(|(_, r)| r.wants && r.lane.is_none() && !r.released).map(|(seq, r)| (*seq, r.width)).collect()
    }
    /// A registered root's width (0: none).
    pub(crate) fn width(&self, root: u64) -> usize {
        self.table.lock().unwrap().roots.get(&root).map_or(0, |r| r.width)
    }
    /// Keep the lane `root` would take from later roots in this pass; the root it waits behind.
    pub(crate) fn claim(&self, root: u64, width: usize, claims: &mut Claims) -> Option<u64> {
        let table = self.table.lock().unwrap();
        let width = width.min(table.gpus.len());
        let start = (0..=table.gpus.len() - width)
            .max_by_key(|start| {
                let free = (*start..start + width).filter(|d| table.free(*d, root, claims)).count();
                (free, std::cmp::Reverse(*start))
            })
            .unwrap_or(0);
        let behind = table.other(start..start + width, root, claims).next();
        for claim in claims.iter_mut().skip(start).take(width) {
            claim.get_or_insert(root);
        }
        behind
    }
    /// `root`'s lane, binding the first lane of its width whose GPUs are free and claimed by
    /// no older root; else it claims one and waits.
    pub(crate) fn lane(&self, root: u64, claims: &mut Claims) -> Lane {
        let mut table = self.table.lock().unwrap();
        let Some(entry) = table.roots.get(&root) else {
            return Lane::Waiting { behind: None, new: false };
        };
        if let Some(lane) = &entry.lane {
            return Lane::Held { lane: lane.clone(), new: false };
        }
        let width = entry.width.min(table.gpus.len());
        let free = (0..=table.gpus.len() - width)
            .find(|start| (*start..start + width).all(|d| table.free(d, root, claims)));
        match free {
            Some(start) => {
                let lane = start..start + width;
                lane.clone().for_each(|d| table.gpus[d].root = Some(root));
                let entry = table.roots.get_mut(&root).expect("a registered root");
                (entry.lane, entry.released, entry.waiting) = (Some(lane.clone()), false, false);
                Lane::Held { lane, new: true }
            }
            None => {
                let new = !std::mem::replace(&mut table.roots.get_mut(&root).expect("a registered root").waiting, true);
                drop(table);
                Lane::Waiting { behind: self.claim(root, width, claims), new }
            }
        }
    }
    /// End `root`'s lane now; a call in flight keeps its GPUs until it returns. The lane ended.
    pub(crate) fn release(&self, root: u64) -> Option<Range<usize>> {
        let mut table = self.table.lock().unwrap();
        let entry = table.roots.get_mut(&root)?;
        entry.released = true;
        let lane = entry.lane.take()?;
        lane.clone().for_each(|d| table.gpus[d].root = None);
        drop(table);
        self.freed.notify_all();
        Some(lane)
    }
    /// A call on `devices`: inside `root`'s lane when it is registered, else on GPUs no root
    /// holds or older root claims (a standalone call: its own root; `None`: machine warming).
    /// No GPU may be calling, or kept by an accelerator job outside `family`.
    pub(crate) fn call(self: &Arc<Self>, root: Option<u64>, family: Option<&str>, devices: Range<usize>, claims: &Claims) -> Option<Permit> {
        let mut table = self.table.lock().unwrap();
        let registered = root.is_some_and(|root| table.roots.contains_key(&root));
        let usable = !devices.is_empty() && devices.end <= table.gpus.len() && devices.clone().all(|d| {
            let gpu = &table.gpus[d];
            let placed = match registered {
                true => gpu.root == root,
                false => gpu.root.is_none() && claims.get(d).copied().flatten().is_none_or(|c| Some(c) == root),
            };
            placed && !gpu.calling && gpu.family.as_deref().is_none_or(|kept| Some(kept) == family)
        });
        if !usable {
            return None;
        }
        for d in devices.clone() {
            (table.gpus[d].calling, table.gpus[d].caller) = (true, family.map(String::from));
        }
        Some(Permit { table: self.clone(), devices, keep: false })
    }
    /// Keep `device` of `root`'s lane for accelerator job family `family` while it lives.
    pub(crate) fn keep(self: &Arc<Self>, root: u64, family: &str, device: usize) -> Option<Permit> {
        let mut table = self.table.lock().unwrap();
        let gpu = table.gpus.get_mut(device)?;
        let ours = gpu.family.as_deref() == Some(family);
        let calls_here = gpu.calling && gpu.caller.as_deref() != Some(family);
        if gpu.root != Some(root) || !(ours || gpu.family.is_none() && !calls_here) {
            return None;
        }
        gpu.family = Some(family.into());
        gpu.keeps += 1;
        Some(Permit { table: self.clone(), devices: device..device + 1, keep: true })
    }
    /// Machine upkeep on `devices` (ending idle executors): no call or accelerator job there.
    pub(crate) fn upkeep(self: &Arc<Self>, devices: Range<usize>) -> Option<Permit> {
        let mut table = self.table.lock().unwrap();
        if devices.clone().any(|d| table.gpus.get(d).is_none_or(|gpu| gpu.calling || gpu.family.is_some())) {
            return None;
        }
        devices.clone().for_each(|d| table.gpus[d].calling = true);
        Some(Permit { table: self.clone(), devices, keep: false })
    }
    /// Wait until no call runs on `device` (an accelerator job's own keep may stay).
    pub(crate) fn quiet(&self, device: usize) {
        let mut table = self.table.lock().unwrap();
        while table.gpus[device].calling {
            table = self.freed.wait(table).unwrap();
        }
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        let mut table = self.table.table.lock().unwrap();
        for gpu in &mut table.gpus[self.devices.clone()] {
            match self.keep {
                true => {
                    gpu.keeps -= 1;
                    if gpu.keeps == 0 {
                        gpu.family = None;
                    }
                }
                false => (gpu.calling, gpu.caller) = (false, None),
            }
        }
        drop(table);
        self.table.freed.notify_all();
    }
}
impl Hold {
    pub(crate) fn root(&self) -> u64 {
        self.root
    }
}
impl Drop for Hold {
    fn drop(&mut self) {
        let mut table = self.table.table.lock().unwrap();
        let ended = table.roots.get_mut(&self.root).is_some_and(|root| {
            root.holders -= 1;
            root.holders == 0
        });
        if ended {
            table.roots.remove(&self.root);
            table.gpus.iter_mut().filter(|gpu| gpu.root == Some(self.root)).for_each(|gpu| gpu.root = None);
        }
        drop(table);
        self.table.freed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn held(lane: Lane) -> Range<usize> {
        match lane {
            Lane::Held { lane, .. } => lane,
            Lane::Waiting { .. } => panic!("expected a held lane"),
        }
    }

    #[test]
    fn a_younger_root_takes_only_gpus_no_older_waiting_root_needs() {
        let gpus = Allocations::new(4);
        let (_x, _a, _b) = (gpus.register(1, 2), gpus.register(2, 4), gpus.register(3, 2));
        assert_eq!(held(gpus.lane(1, &mut [None; 4])), 0..2);
        let mut claims = [None; 4];
        // A (older, in CPU start-up) asked for GPU work and needs every GPU: B may not bind
        // the free half. A root that never asks (a CPU-only job) keeps no place.
        assert!(gpus.pending().is_empty());
        (gpus.want(2), gpus.want(3));
        assert_eq!(gpus.pending(), [(2, 4), (3, 2)]);
        assert_eq!(gpus.claim(2, 4, &mut claims), Some(1));
        assert!(matches!(gpus.lane(3, &mut claims), Lane::Waiting { behind: Some(2), .. }));
        assert!(gpus.call(Some(4), None, 2..3, &claims).is_none(), "a standalone call waits too");
        // Without the older demand the younger root binds the free half.
        assert_eq!(held(gpus.lane(3, &mut [None; 4])), 2..4);
    }

    #[test]
    fn an_accelerator_job_keeps_its_gpu_for_its_family_only() {
        let gpus = Allocations::new(2);
        let _root = gpus.register(5, 2);
        held(gpus.lane(5, &mut [None; 2]));
        let job = gpus.keep(5, "job", 0).unwrap();
        assert!(gpus.keep(5, "sibling", 0).is_none());
        assert!(gpus.call(Some(5), None, 0..1, &[None; 2]).is_none(), "a sibling's call goes elsewhere");
        let borrowed = gpus.call(Some(5), Some("job"), 0..2, &[None; 2]).unwrap();
        let nested = gpus.keep(5, "job", 0).unwrap();
        assert!(gpus.upkeep(1..2).is_none());
        drop((borrowed, job));
        assert!(gpus.call(Some(5), None, 0..1, &[None; 2]).is_none(), "the nested job still keeps it");
        drop(nested);
        assert!(gpus.call(Some(5), None, 0..1, &[None; 2]).is_some());
    }
}
