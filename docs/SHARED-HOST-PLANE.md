# Shared host weight tier

`SharedHostPlane` links released TensorFS core and plane 0.3.91. Its native plane has
`devices=[]`: allocation, native ReadLease verification, native layout, O_DIRECT/buffered
reader pool and shared readiness/claims work without loading CUDA or NVML. It creates and
fills the memfd; it does not adopt arbitrary executor offers. Prepared actor/immutable-plan
scope and exact manifest/region roster authorize a cache key; names never confer authority.

Preparation is a trusted in-process call with `HostScope`, `HostKey`, manifest, native
`ReadPlan` and region roster. `HostKey::for_regions` uses the SDK's canonical semantic
manifest/regions digest, while TensorFS verifies its independent physical layout digest.
The pool registers the already measured `ProcessBirth` and actual pidfd before granting.
A grant uses a native independent whole-file OFD hold, retained in the transferred fd itself:
owner death before receiver adoption cannot punch the data. Receiver TensorFS adopts its
own mapping/region claims and closes the received fd after adoption.

The ledger measures each backing inode once with kernel `st_blocks`, including the state
page. The native reader queue is bounded by synchronous region batches (1–4 readers).
Whole idle allocations are reclaimed through native `close_ws`, with an entry-count bound
and LRU selection. A disabled/full cache returns `held:false`; it never rejects a model or
inference request for size. Existing descriptor/disk sources continue to work.

An allocation exported to a live recipient cannot be evicted. Lowering the budget reports
retained `over_budget_bytes`; it does not pretend progress, a socket close, or an idle result
is a CUDA completion/unregister fence. Only actual pidfd exit ends recipient custody in
this checkpoint. Ending that exact recipient's native whole-file OFD claim is explicit:
SCM_RIGHTS/dup copies otherwise keep the same lock alive after `release_hold` drops its
local descriptor. The CPU gate keeps extra grant descriptors open and proves data pages
are punched after exit, rather than trusting native logical release counters. Remaining
state pages may survive in externally retained unclaimed descriptors; source page cache,
private SDK scratch/activation/staging, process RSS and GPU registrations are not this
weight-tier backing ledger.

Five CPU gates use actual TensorFS objects/read plans, native host fills, SCM_RIGHTS,
separate Python CPU weighted-inference processes, abrupt owned-fixture death and pidfds.
They prove one fill/one physical charge across two recipients, survival after receiver
crash and machine object loss, actor/birth authorization, zero-budget fallback, and physical
reclaim after exact exit while file descriptors remain open. `/proc/self/maps` and open fds
prove no CUDA/NVML library or NVIDIA device opens in this host-only path. Five existing
model-source tests also pass after the released dependency update.

The optional SDK capability is `host_tiers.owner/1`, selected by omitted-default
`Load.host_tier_owner`. The SDK sends its existing native construction traversal, selected
physical parts and partition roster as `HostTierPlan` in a fully sealed memfd; only raw
SHA-256/length and name/manifest/layout cross the small `HostTierPrepare` frame. The owner
checks seals, exact bytes, envelope agreement, previously selected Header components and
actual process birth, then uses native plan/select/layout/fill. Unknown advisory fields are
harmless. Requests over 64KiB are not rejected for roster size. Native reader errors drain
all started region tickets before partial backing is unmapped; a failed cache allocation
must leave zero native allocations, tested with an actual invalid source range.

Eight Rust host gates and five existing source gates pass. The paired Runtime 46 CPU gates
include an actual Rust owner process, SDK `Executor._durable`, native readonly model sources
and TensorFS `register(host_fd)` adoption: all declared bytes match, recipient host fills
and source buffered-copy bytes are both zero. The CPU owner fixture binary is
`host-plane-cpu-owner`, with explicit typed config; it is a qualification server, not a new
production daemon. SDK request compatibility, sealed metadata larger than the control cap,
corruption/authorization, source integrity/FD failure cleanup and existing policy/skew gates
also pass. The GPU pool must still integrate these committed callbacks and qualify real
unchanged model loads.

This is not yet complete Degree 1. Existing fine/refined layouts retain the no-tier path;
optional cache misses still allow private SDK pinned/bounce allocation. Whole active
allocations remain charged until actual process exit: live cooperative unregister/revoke,
per-region refill and a cross-executor DMA completion contract are not yet implemented.
The native source ReadLease retains complete object FD closures and may hit the kernel
hard FD limit. Verified mapped source pages/page cache, process RSS, activation/scratch,
CUDA registrations and private staging are outside this backing ledger. Global machine
memory admission/ownership, SDXL/Anima/H3, GPU crash safety, NCCL/multi-GPU and normal CLI
qualification remain open. A host-only plane does not pin pages via a GPU context: each
recipient's native plane performs its CUDA registration while sharing the same physical
backing. The small CPU weighted fixture is source/lifetime evidence, not production model
or GPU qualification.
