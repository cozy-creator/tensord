# Host tier (Degree 1)

`host_tier.rs` (`HostTier`) keeps shared **CPU weight buffers**, not Python/PyTorch models or
GPU tensors. TensorD chooses which buffers to keep under its host-memory policy and uses
embedded TensorFS to fill them from verified stored weight bytes. Each weight set's layout is a memfd
sealed at creation (`tensorfs_plane::host::open_sealed`: nobody else can write, resize or punch
it) and filled once through TensorFS's verified read path (page-cache copies, O_DIRECT for the
rest, N threads) by one background filler, layouts in the order opened, each region marked Ready
as its last byte lands. Runtime executors adopt it read-only (`Plane.register_sealed`); their
local TensorFS plane waits for each region before pinned staging or a GPU copy. Runtime
constructs the model and performs those device transfers in its own CUDA context. A fill that
fails or is abandoned marks the rest FAILED: adopters error, never read it. The layout survives
executor death and model switches. The host-tier implementation uses no CUDA or NVML.

## Exchange

- `GpuPool` enables `Load.sealed_tiers` when the executor offers both
  `host_tiers.sealed/1` and `weight_plane/1`. It reports the unsealed route for older peers
  without those capabilities; the capability negotiation does not grant arbitrary store access.
- Per weight set the executor sends `sealed_tier` with a sealed memfd holding its `SealedPlan`
  (manifest, traversal, window, components, regions, parts). The machine checks seals, length and
  SHA-256, that the manifest and components are the session's (`HostGrant`), and rebuilds the read
  plan and layout from its own header. Cache key: the TensorFS layout digest.
- Answer `held: true` with a read-only reopen of the sealed memfd at once: whole and filling, or
  streamed when it does not fit. A refused plan (outside the selection, unsealed) is an answer
  too, never a session failure by itself. The executor either uses another supported granted
  source or reports that it cannot load the weight set. Legacy peers may use their older
  store-reading route; current managed Runtime requires owner-provided sources.
- `sealed_prefetch` carries all of a construction's plans: each is opened and queued at once.

## Staged layouts (`host_tiers.staged/1`)

A GPU executor that offers `host_tiers.staged/1` (and is told `Load.staged_tiers`, so it never asks
an older machine) gets each layout with only the regions it streams every step staged; the rest are holes (`HOLE` in the state page). For the holes it asks
`object_files`: the machine opens each object of the plan through the store's verification and
hands the read-only descriptors over (no Store-opening authority or path is supplied). Runtime's
TensorFS plane copies a hole from those files to GPU memory (page cache, or O_DIRECT into its pinned staging
buffers), so resident blocks and idle models cost page cache, which the kernel reclaims, not tier
RAM. When its stages plan streaming, the executor asks `sealed_stage` for those regions; the
machine stages each one the tier has room for (`TierLimit::limit` per region; the rest keep
reading the files) and the executor pins what was staged. An executor that reads no holes (CPU,
or without the capability) gets every region staged when the tier has room for all of them;
adopting a layout staged in part that cannot grow, it reads through a window of its own.

## Size and release

- At every ask the live headroom is read (`host_memory::read`: every cgroup on the path,
  `memory.high`/`memory.max` v2 or `limit_in_bytes` v1, minus non-reclaimable usage, and
  `MemAvailable`). `TierLimit` decides the most the tier may charge. `GpuPool` supplies
  `memory::host::TierPolicy`, whose shared host ledger accounts for the tier and executors'
  private pinned buffers. `HalfOfHeadroom` is the standalone default policy.
- To make room, unheld, complete layouts are released oldest first; such layouts past `ttl` go
  anyway. A layout is held while any executor it was granted to is alive (pidfd). Held or filling
  layouts never go.
- Charge = the layout's size (a staged layout: its staged regions) while filling, then kernel
  `st_blocks` per memfd. A release records the charged bytes and the fall in
  shared memory; if memory did not come back, the bytes stay charged (`stranded_bytes`).
- `release(want)` and `facts()` are the policy module's handles (see B1 `INTERFACE.md`).

## Disk rung (streamed layouts)

For an executor that reads holes, the disk rung is a layout with nothing staged. For one that does
not, when admission cannot make room even after releases, it gets a streamed layout: a sealed window of as many of the layout's
largest regions as `TierLimit::staging` allows (one at least, the indivisible working set),
served on its own thread under a read lease. The executor claims a region while it reads it
(TensorFS `HostMem::with_region`); the machine stages claimed regions first, reads ahead in
order into unclaimed slots, and refills a slot only when nobody claims its region. The executor
reads no store. The window is released with its last holder; `facts()` reports live windows,
their bytes and reads, and the ledger `windows_opened` and `streamed_bytes`. A streamed layout
cannot be pinned (the plane answers `BelowFloor`).

## Remembered plans

A filled layout's verified plan is kept in `<gpu>/host-plans` (latest per manifest and
components). When `GpuPool` spawns an executor that adopts sealed tiers, layouts of its model the
tier no longer holds refill in the background while it imports (`prefill`); its asks then hit.
Components without a plan get their memory reserved from the manifest at spawn.

## Bounds

- One descriptor per layout, plus one pidfd per executor. A fill's read lease (one descriptor per
  object of that component) ends with the fill; `GpuPool` lifts the soft NOFILE limit to the hard one.
  `object_files` opens one descriptor per object for the answer and closes it once sent; the
  executor keeps them for its weight sets' life (it lifts its own NOFILE limit).
- No lock across a fill: other asks, and other models' executors, proceed meanwhile; a second ask
  for a layout being filled gets the same layout.
- Releasing a CPU layout frees its host backing when its holders finish. Evicting GPU mappings
  is a separate Runtime/device-plane operation requested by TensorD's memory policy; see
  [resident custody](RESIDENT-CUSTODY.md) for shared GPU handles.
- `GpuPool` appends one line per Load to `<state>/gpu/loads.jsonl`: executor load facts and the
  host tier's facts (fills with ms, bytes by read mode and disk reads; hits; prefills;
  releases), and the machine's and executor's RSS/PSS.
