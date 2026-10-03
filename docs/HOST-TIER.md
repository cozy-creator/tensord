# Host tier (Degree 1)

`host_tier.rs` (`HostTier`) keeps the machine's host weights. Each weight set's layout is filled
once through TensorFS's verified read path (`tensorfs_plane::host::fill_sealed`: page-cache copies,
O_DIRECT for the rest, N threads) into a memfd and sealed: nobody can write, resize or punch it.
Executors adopt it read-only (`Plane.register_sealed`) and read no store, lease or object. The
layout survives executor death and model switches. No CUDA or NVML.

## Exchange

- The executor offers `host_tiers.sealed/1` (and `weight_plane/1`); `GpuPool` then sets
  `Load.sealed_tiers`.
- Per weight set the executor sends `sealed_tier` with a sealed memfd holding its `SealedPlan`
  (manifest, traversal, window, components, regions, parts). The machine checks seals, length and
  SHA-256, that the manifest and components are the session's (`HostGrant`), and rebuilds the read
  plan and layout from its own header. Cache key: the TensorFS layout digest.
- Answer `held: true` with a read-only reopen of the sealed memfd, or `held: false` (no room):
  that weight set reads the store. A refused plan is an answer too, never a session failure.

## Size and release

- At every ask the live headroom is read (`host_memory::read`: every cgroup on the path,
  `memory.high`/`memory.max` v2 or `limit_in_bytes` v1, minus non-reclaimable usage, and
  `MemAvailable`). `TierLimit` decides the most the tier may charge; until the memory policy
  module (B2) supplies one, `HalfOfHeadroom` = (available + charged) / 2.
- To make room, unheld layouts are released oldest first; unheld layouts past `ttl` go anyway.
  A layout is held while any executor it was granted to is alive (pidfd). Held layouts never go.
- Charge = kernel `st_blocks` per memfd. A release records the charged bytes and the fall in
  shared memory; if memory did not come back, the bytes stay charged (`stranded_bytes`).
- `release(want)` and `facts()` are the policy module's handles (see B1 `INTERFACE.md`).

## Bounds

- One descriptor per layout, plus one pidfd per executor. A fill's read lease (one descriptor per
  object of that component) ends with the fill; `GpuPool` lifts the soft NOFILE limit to the hard one.
- No lock across a fill: other asks, and other models' executors, proceed meanwhile; a second ask
  for a layout being filled waits for that fill.
- `GpuPool` appends one line per Load to `<state>/gpu/loads.jsonl`: executor load facts and the
  host tier's facts (fills with ms, bytes by read mode and disk reads; hits; releases).

## Fallbacks

Executor without `host_tiers.sealed/1`: descriptors (`model_sources.descriptors/1`), else legacy
store reads. With sealed tiers the header and configs come from the store, not descriptors.
