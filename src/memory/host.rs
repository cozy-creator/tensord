//! The host ledger: one rule for every host byte that holds weights. What the host has is read
//! live at every decision (`host_memory`: the tightest cgroup on the path and `MemAvailable`);
//! CPU weight buffers, TensorD's sealed tier and Runtime's own pinned tiers together, may hold
//! half of it plus what they hold now (Runtime `weight_policy.pinned_total`). The other half
//! stays for processes and the page cache, the tier's own fallback.
use crate::{host_memory::HostMemory, host_tier::TierLimit};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

/// Pinned CPU bytes Runtime holds outside TensorD's tier (its own staging buffers or an older
/// peer's store-reading path). Adopting a sealed layout page-locks the shared pages instead.
#[derive(Default)]
pub struct HostLedger {
    private: Mutex<BTreeMap<String, u64>>,
}

impl HostLedger {
    pub fn private(&self, plan: &str, pinned: Option<u64>) {
        let mut private = self.private.lock().unwrap();
        match pinned {
            Some(bytes) => private.insert(plan.into(), bytes),
            None => private.remove(plan),
        };
    }

    fn private_total(&self) -> u64 {
        self.private.lock().unwrap().values().sum()
    }
}

/// The tier's limit from the one ledger: the weights' half of the host, less what executors
/// pin outside the tier.
pub struct TierPolicy(pub Arc<HostLedger>);

impl TierLimit for TierPolicy {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64 {
        if host.available < 0 {
            return 0; // unreadable: no retained tier capacity; use a supported streaming path
        }
        let private = self.0.private_total();
        ((host.available as u64 + charged + private) / 2).saturating_sub(private)
    }
}

/// Budgets over `(plan, weights, pinned now)`, most recently used first. `available` is what
/// the host has before its tightest limit and `held` what is pinned or sealed now (shmem);
/// -1 is unreadable, and then nothing is decided. No tenant rises by more than half of what
/// is free now.
pub fn pinned_split(
    available: i64,
    held: i64,
    tenants: &[(String, u64, u64)],
) -> Option<BTreeMap<String, u64>> {
    if available < 0 {
        return None;
    }
    let free = available as u64;
    let pinned =
        u64::try_from(held).unwrap_or_else(|_| tenants.iter().map(|(_, _, pinned)| pinned).sum());
    let mut left = (free + pinned) / 2;
    let mut split = BTreeMap::new();
    for (plan, weights, pinned) in tenants {
        let share = (*weights).min(left).min(pinned + free / 2);
        split.insert(plan.clone(), share);
        left -= share;
    }
    Some(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;

    #[test]
    fn the_most_recent_tenant_takes_its_weights_and_the_rest_share_what_is_left() {
        let tenants = [
            ("anima".to_string(), 6 * GIB, 0),
            ("sdxl".to_string(), 7 * GIB, 7 * GIB),
        ];
        // 16 GiB free, 7 GiB pinned: a total of 11.5 GiB, Anima first.
        let split = pinned_split((16 * GIB) as i64, (7 * GIB) as i64, &tenants).unwrap();
        assert_eq!(split["anima"], 6 * GIB);
        // SDXL keeps what is left of the total, 5.5 GiB: lowered from the 7 it pins.
        assert_eq!(split["sdxl"], 11 * GIB / 2);
    }

    #[test]
    fn executors_pinning_outside_the_tier_shrink_its_limit_by_as_much() {
        let ledger = Arc::new(HostLedger::default());
        let policy = TierPolicy(ledger.clone());
        let host = HostMemory {
            available: (20 * GIB) as i64,
            shmem: 0,
            mem_available: -1,
        };
        assert_eq!(policy.limit(&host, 4 * GIB), 12 * GIB);
        ledger.private("anima", Some(2 * GIB));
        assert_eq!(policy.limit(&host, 4 * GIB), 11 * GIB);
        ledger.private("anima", None);
        assert_eq!(policy.limit(&host, 4 * GIB), 12 * GIB);
    }

    #[test]
    fn an_unreadable_host_decides_nothing() {
        assert!(pinned_split(-1, 0, &[("a".into(), GIB, 0)]).is_none());
    }
}
