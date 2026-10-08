//! One GPU call slot. An accelerator job retains it for its managed family;
//! serving descendants borrow one serialized call inside that family.
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(crate) struct Slot(Mutex<Option<State>>);
struct State {
    family: Option<String>,
    holders: usize,
    calling: bool,
}
pub(crate) struct Permit {
    slot: Arc<Slot>,
    call: bool,
}
impl Slot {
    pub(crate) fn take(self: &Arc<Self>, family: Option<&str>, call: bool) -> Option<Permit> {
        let mut held = self.0.lock().unwrap();
        match held.as_mut() {
            Some(state)
                if family.is_some()
                    && state.family.as_deref() == family
                    && (!call || !state.calling) =>
            {
                state.holders += 1;
                state.calling |= call;
            }
            Some(_) => return None,
            None => {
                *held = Some(State {
                    family: family.map(str::to_owned),
                    holders: 1,
                    calling: call,
                })
            }
        }
        Some(Permit {
            slot: self.clone(),
            call,
        })
    }
    pub(crate) fn family(&self, family: &str) -> bool {
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|s| s.family.as_deref() == Some(family))
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        let mut held = self.slot.0.lock().unwrap();
        let state = held.as_mut().expect("a live GPU reservation");
        state.holders -= 1;
        if self.call {
            state.calling = false;
        }
        if state.holders == 0 {
            *held = None;
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
        let slot = Arc::new(Slot::default());
        let parent = slot.take(Some("job-1"), false).unwrap();
        let child = slot.take(Some("job-1"), true).unwrap();
        assert!(slot.take(Some("job-1"), true).is_none());
        assert!(slot.take(Some("job-2"), false).is_none());
        assert!(slot.take(None, true).is_none()); // unrelated call / warm / prefetch
        drop(parent); // pause/cancel ends root, started child keeps its reservation
        assert!(slot.take(None, true).is_none());
        let resumed = slot.take(Some("job-1"), false).unwrap();
        drop(child);
        assert!(slot.take(None, true).is_none()); // live parent still owns the family
        let next = slot.take(Some("job-1"), true).unwrap();
        drop(resumed);
        assert!(slot.take(None, true).is_none());
        drop(next);
        assert!(slot.take(None, true).is_some());
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
    fn ordinary_calls_and_warm_work_remain_exclusive() {
        let slot = Arc::new(Slot::default());
        let call = slot.take(None, true).unwrap();
        assert!(slot.take(None, true).is_none());
        assert!(slot.take(Some("job"), false).is_none());
        drop(call);
        let root = slot.take(Some("job"), false).unwrap();
        let nested = slot.take(Some("job"), false).unwrap();
        drop(root);
        assert!(slot.family("job"));
        drop(nested);
        assert!(!slot.family("job"));
    }
}
