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

This is not yet complete Degree 1. Existing HostTier asks carry only name/layout, so the
SDK needs an additive authorized partition/manifest registration before first ask. Actual
Runtime model consumption, GPU DMA/compute failures, live cooperative region revocation,
pageable/pinned resource coordination, full machine memory accounting, SDXL/Anima/H3,
NCCL/multi-GPU and normal public CLI qualification remain separate gates. The small CPU
weighted fixture is source/memory lifetime evidence, not a copied production model or SDK
GPU qualification.
