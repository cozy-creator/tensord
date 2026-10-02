# Plan and decision gates

The owner's requested boundary is (worker orchestration + machine-agent + TensorFS) in Rust,
with Python cozy-runtime retaining the executor, SDK and shared model integrations.
The standalone repository keeps this experiment away from currently owned implementation work.

## A/B review decision

Option A fixes the present stack first and has lower migration cost. Option B builds the new
machine service. We are doing a bounded, proof-first B experiment, with A's conservative defaults.
The existing owners keep #248/#297 and their safety/performance fixes.

Before expanding to a full replacement, compare equivalent old/new paths. Report measurable
benefits, migration cost and remaining consumer coverage. If most benefit comes from improvements
that already fit the old stack, prefer A and reuse the demonstrated mechanism/interface rather
than continuing a rewrite just because this repository exists.

## Slices

1. [CPU ownership proof (#2)](https://github.com/cozy-creator/cozy-machine/issues/2):
   linked TensorFS, typed current framing, verified/sealed weight descriptors, real CPU inference,
   exact process-death retention/reclaim and cache pressure. Working component proof; package
   supervision, recipient registration and broader reclamation remain incomplete.
2. [Durable package path (#3)](https://github.com/cozy-creator/cozy-machine/issues/3):
   authenticated intake, journal and immutable durable outputs; real Runtime package calls,
   existing SDK/command adapters and conservative core-death settlement.
3. [Degree 1 product path (#4)](https://github.com/cozy-creator/cozy-machine/issues/4):
   bounded host/disk/DMA staging, real model recovery, single-writer CLI integration, package
   generation holds, local/pod lifecycle and actual delivery/update consumers.
4. [Shared GPU backing (#5)](https://github.com/cozy-creator/cozy-machine/issues/5):
   only after measured sharing/replacement value; resident sharing and executor-issued copies
   first. A helper adds a process and offers stronger isolation than in-process driver actors.
   Decide explicitly with fault/API evidence; neither contains a whole-host GPU failure.
5. [Optional copier research (#6)](https://github.com/cozy-creator/cozy-machine/issues/6):
   not a completion prerequisite. It must outperform executor copies and verify the interrupted
   operation, safe old-DMA fencing and complete production feeder behavior.

The first-release core stays CUDA/NVML-free. Do not add an invisible helper or promise seamless
core-death adoption. Sealed slabs are evicted by releasing every holder; mutable cache regions
can be punched only after all readers/fills/DMA are quiescent. They are different backing kinds.

## Early feasibility gates

Prove the actual browser ICE-TCP/DTLS/scoped-capability wire and old install/update consumers
before promising a Rust front door or independent assets. The old Runtime-wheel delivery scheme
is an existing consumer contract; a standalone distribution must earn its adapter coverage.
Keep Hub's standalone tfs verifier operational.

Record ownership and fix coverage for each stack. No calendar deadline authorizes deleting
the old stack, retiring old peer commands, overriding package bounds or shipping unqualified code.

## Comparison

After equivalent real inference works, use the unchanged old stack as baseline under matched
CPU/RAM/VRAM/cache/thermal conditions and identical model/request semantics. Fresh prompts and
seeds prevent cache reuse. Compare cold submit→verified output, steady throughput, model switches,
physical memory and fault recovery. Include preparation count and executor reuse.

The current tiny classifier proves byte/lifetime contracts. Its attachment timings cannot decide
ComfyUI parity, old Runtime startup, SDXL/Anima/H3 throughput or the value of a full rewrite.
