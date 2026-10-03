//! Per-host pinned budgets: one machine-wide pinned total shared most recently used first
//! (Runtime `weight_policy.pinned_total` and `worker/memory.pinned_split`). Pinned memory is
//! unreclaimable: half of what the host has for it stays for processes and the page cache.
use std::collections::BTreeMap;

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
    fn an_unreadable_host_decides_nothing() {
        assert!(pinned_split(-1, 0, &[("a".into(), GIB, 0)]).is_none());
    }
}
