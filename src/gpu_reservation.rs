//! One call slot per GPU. A call takes every slot of its lane; an accelerator job retains
//! its GPU's slot for its managed family, and serving descendants borrow one serialized call
//! there. A GPU an earlier waiting request needs is not taken afresh by a later one.
use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

pub(crate) struct Slots(Mutex<Vec<Option<State>>>);
struct State {
    family: Option<String>,
    holders: usize,
    calling: bool,
}
pub(crate) struct Permit {
    slots: Arc<Slots>,
    devices: Range<usize>,
    call: bool,
}
impl Slots {
    pub(crate) fn new(width: usize) -> Arc<Self> {
        Arc::new(Self(Mutex::new((0..width).map(|_| None).collect())))
    }
    /// Every slot of `devices`, or none. A slot `waiting` marks is only borrowed by its family.
    pub(crate) fn take(
        self: &Arc<Self>,
        family: Option<&str>,
        call: bool,
        devices: Range<usize>,
        waiting: &[bool],
    ) -> Option<Permit> {
        let mut slots = self.0.lock().unwrap();
        let usable = devices.clone().all(|device| match slots.get(device) {
            Some(None) => !waiting.get(device).copied().unwrap_or(false),
            Some(Some(state)) => {
                family.is_some() && state.family.as_deref() == family && (!call || !state.calling)
            }
            None => false,
        });
        if devices.is_empty() || !usable {
            return None;
        }
        for slot in &mut slots[devices.clone()] {
            match slot {
                Some(state) => {
                    state.holders += 1;
                    state.calling |= call;
                }
                None => {
                    *slot = Some(State {
                        family: family.map(str::to_owned),
                        holders: 1,
                        calling: call,
                    })
                }
            }
        }
        Some(Permit {
            slots: self.clone(),
            devices,
            call,
        })
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        let mut slots = self.slots.0.lock().unwrap();
        for slot in &mut slots[self.devices.clone()] {
            let state = slot.as_mut().expect("a live GPU reservation");
            state.holders -= 1;
            if self.call {
                state.calling = false;
            }
            if state.holders == 0 {
                *slot = None;
            }
        }
    }
}

/// Outermost accelerator ancestor, never an unconditional CPU root: sibling GPU
/// jobs of a CPU orchestrator must compete, while a CPU intermediary may borrow.
pub(crate) fn family(
    record: &crate::journal::Execution,
    mut get: impl FnMut(&str) -> std::io::Result<crate::journal::Execution>,
) -> std::io::Result<Option<String>> {
    let mut current = record.clone();
    let mut family = None;
    let mut seen = std::collections::HashSet::new();
    loop {
        if !seen.insert(current.id.clone()) {
            return Err(std::io::Error::other("cyclic managed parent chain"));
        }
        if current.invocation.accelerator {
            family = Some(current.id.clone());
        }
        if current.invocation.parent.is_empty() {
            return Ok(family);
        }
        current = get(&current.invocation.parent)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn family_keeps_the_slot_until_every_started_member_exits() {
        let slot = Slots::new(1);
        let parent = slot.take(Some("job-1"), false, 0..1, &[]).unwrap();
        let child = slot.take(Some("job-1"), true, 0..1, &[]).unwrap();
        assert!(slot.take(Some("job-1"), true, 0..1, &[]).is_none());
        assert!(slot.take(Some("job-2"), false, 0..1, &[]).is_none());
        assert!(slot.take(None, true, 0..1, &[]).is_none()); // unrelated call / warm / prefetch
        drop(parent); // pause/cancel ends root, started child keeps its reservation
        assert!(slot.take(None, true, 0..1, &[]).is_none());
        let resumed = slot.take(Some("job-1"), false, 0..1, &[]).unwrap();
        drop(child);
        assert!(slot.take(None, true, 0..1, &[]).is_none()); // live parent still owns the family
        let next = slot.take(Some("job-1"), true, 0..1, &[]).unwrap();
        drop(resumed);
        assert!(slot.take(None, true, 0..1, &[]).is_none());
        drop(next);
        assert!(slot.take(None, true, 0..1, &[]).is_some());
    }
    #[test]
    fn family_comes_from_durable_accelerator_ancestry_not_cpu_siblings() {
        use crate::{execution::Engine, journal::Invocation};
        let root = std::env::temp_dir().join(format!("job-family-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let submit = |key: &str, parent: &str, accelerator| {
            engine
                .submit(
                    key,
                    Invocation {
                        package: "family".into(),
                        generation: "0".repeat(32),
                        module: "family:app".into(),
                        entrypoint: "run".into(),
                        job: true,
                        accelerator,
                        parent: parent.into(),
                        ..Invocation::default()
                    },
                )
                .unwrap()
        };
        let cpu = submit("cpu", "", false);
        let first = submit("first", &cpu.id, true);
        let second = submit("second", &cpu.id, true);
        let middle = submit("middle", &first.id, false);
        let nested = submit("nested", &middle.id, true);
        assert_eq!(family(&cpu, |id| engine.get(id)).unwrap(), None);
        assert_eq!(
            family(&first, |id| engine.get(id)).unwrap(),
            Some(first.id.clone())
        );
        assert_eq!(
            family(&second, |id| engine.get(id)).unwrap(),
            Some(second.id.clone())
        );
        assert_eq!(
            family(&nested, |id| engine.get(id)).unwrap(),
            Some(first.id.clone())
        );
        drop(engine);
        let reopened = Engine::open(&root).unwrap();
        assert_eq!(
            family(&reopened.get(&nested.id).unwrap(), |id| reopened.get(id)).unwrap(),
            Some(first.id)
        );
        drop(reopened);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn managed_prefetch_borrows_its_parent_family_but_serializes_with_calls() {
        let slot = Slots::new(1);
        let parent = slot
            .take(Some("accelerator-parent"), false, 0..1, &[])
            .unwrap();
        let prefetch = slot
            .take(Some("accelerator-parent"), true, 0..1, &[])
            .unwrap();
        assert!(slot.take(None, true, 0..1, &[]).is_none());
        assert!(slot.take(Some("another-parent"), true, 0..1, &[]).is_none());
        assert!(slot
            .take(Some("accelerator-parent"), true, 0..1, &[])
            .is_none());
        drop(prefetch);
        assert!(slot
            .take(Some("accelerator-parent"), true, 0..1, &[])
            .is_some());
        assert!(slot.take(None, true, 0..1, &[]).is_none());
        drop(parent);
        assert!(slot.take(None, true, 0..1, &[]).is_some());
    }
    #[test]
    fn each_gpu_serves_its_own_call_and_a_waited_for_gpu_is_only_borrowed() {
        let slots = Slots::new(4);
        let first = slots.take(None, true, 0..1, &[]).unwrap();
        let second = slots.take(None, true, 1..2, &[]).unwrap();
        assert!(slots.take(None, true, 0..2, &[]).is_none()); // a group needs every GPU of it
        assert!(slots.take(None, true, 3..5, &[]).is_none()); // beyond the envelope
        let waiting = [false, false, true, false];
        assert!(slots.take(None, true, 2..3, &waiting).is_none());
        let job = slots.take(Some("job"), false, 2..3, &[]).unwrap();
        assert!(slots.take(Some("job"), true, 2..3, &waiting).is_some()); // its family borrows
        drop((first, second));
        assert!(slots.take(None, true, 0..2, &[]).is_some());
        drop(job);
        assert!(slots.take(None, true, 0..4, &[]).is_some());
    }
    #[test]
    fn ordinary_calls_and_warm_work_remain_exclusive() {
        let slot = Slots::new(1);
        let call = slot.take(None, true, 0..1, &[]).unwrap();
        assert!(slot.take(None, true, 0..1, &[]).is_none());
        assert!(slot.take(Some("job"), false, 0..1, &[]).is_none());
        drop(call);
        let root = slot.take(Some("job"), false, 0..1, &[]).unwrap();
        let nested = slot.take(Some("job"), false, 0..1, &[]).unwrap();
        drop(root);
        assert!(slot.take(None, true, 0..1, &[]).is_none());
        drop(nested);
        assert!(slot.take(None, true, 0..1, &[]).is_some());
    }
}
