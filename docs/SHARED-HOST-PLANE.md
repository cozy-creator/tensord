# Shared host plane

`shared_host_plane.rs` (`SharedHostPlane`) is the machine-owned host-memory weight tier. It fills
TensorFS plane memfds once and lends them to executors. The native plane has `devices = []`, so no
CUDA or NVML is loaded.

## API

- `new(store_root, HostConfig { budget_bytes, readers: 1..4, max_entries })`.
- `register_peer(HostPeer { actor, plan, birth }, pidfd)`: the pidfd must match the live birth.
  `register_socket` derives the peer from `SO_PEERPIDFD`.
- `authorize(HostScope { actor, plan }, manifest, header, components)`: trusted and immutable.
- `prepare(HostPreparation)`: fills one native `ReadPlan` and region roster. `HostKey::for_regions`
  binds `layout` to the canonical {manifest, regions} digest. Returns `false` on a cache miss.
- `request(peer, frame, fd)`: SDK adapter for `host_tier` and `host_tier_prepare`.
- `set_budget`, `stats` return `HostCharge`.

## Rules

- Authority is the (actor, plan) scope plus `authorize`. Names only select among authorized
  entries, and an executor `offer` never allocates.
- `host_tier_prepare` sends a fully sealed memfd holding a `HostTierPlan`. The owner checks seals,
  length, SHA-256, envelope fields, the registered birth and component membership.
- A miss returns `held: false`. That covers zero or full budget, the entry bound, or a layout at
  least as large as the budget. Requests are never rejected for size.
- A grant takes a native whole-file OFD hold per recipient, so owner death cannot punch the data.
- The charge is kernel `st_blocks` per backing inode. LRU reclaim skips held entries.
- Lowering the budget reports `over_budget_bytes` and reclaims nothing in use.
- Holds end only when the recipient's pidfd exits. The owner then unlocks that OFD, because
  SCM_RIGHTS copies share it.
- A failed fill drains every started region ticket before closing the allocation.

## Known gaps

- Fine/refined layouts bypass the tier. A miss still allows private SDK pinned allocation.
- Live revoke, per-region refill and a cross-executor DMA completion contract are not implemented.
- The source lease keeps one fd per object.
- The state mutex is held during a fill.
