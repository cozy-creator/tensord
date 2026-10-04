# Model sources for every rank

The machine owns the selected native TensorFS closure. Device executors carry the
plane-only wheel (`tensorfs.STORE is False`); neither rank 0 nor its followers open
the store. A model's header and inline construction configuration arrive through
`ModelSources::serve`, and declared assets use the same `model_source` exchange.
Weight layouts and verified ObjectFiles arrive through the existing host-tier
commands.

## Current source audit

Runtime `d7a8a556` already removed the old `rank > 0` exclusion from the sourced
checkpoint callback. The current `de40d6b1` candidate includes that change:

- `Executor._prepare_first` uses `host._model_source` whenever `Load.model_sources`
  is true, on every rank. Construction children retain their own rank's host.
- RankGroup's reader relays a follower request through rank 0 under its durable
  exchange lock. `_RELAYED` includes model_source, sealed_tier, sealed_prefetch,
  sealed_stage, and object_files; descriptors are passed to the follower unopened.
- Multi-model preparation propagates model_sources, sealed_tiers, and staged_tiers
  into each construction child.
- A sourced Checkpoint parses the native header, reads inline configs locally, and
  requests declared assets through its callback. It has no Store or read lease.
- H3's loader uses the ordinary Loader/Checkpoint metadata and asset path. Its
  model implementation does not open a Store.

The previous rank-fallback finding described the ancestor before `d7a8a556`, not
this candidate. The inherited real follower-process test uses a Python Store to
construct its fixture, so it does not qualify the deployed plane-only wheel.
Keep the existing source relay; do not add a second source API or Store fallback.

## CPU proof

Build the immutable checkpoint, source owner, and host-tier grants in Rust. Launch
an actual rank-0 process plus its RankGroup follower using the selected SDK Python
interpreter. Both must assert STORE=False, request the verified header and asset,
parse the inline configuration, and consume the same sealed layout. A later finer
layout and ObjectFiles request must also travel through the follower relay. Check
the bytes, readonly grants, and rejection of unselected manifests/components/assets.

Use native session roots associated with the receiver's exact birth and captured
scope. Remove the source repository and accepted-run root, then run an independent
native GC while a follower remains alive. Custody must survive the whole supported
scope, including a follower whose leader has exited. Known whole-scope exit releases
the roots; unknown scope evidence keeps them. No elapsed time establishes exit.

This is source/storage qualification, not an inference or NCCL benchmark. The
hardware gate is an ordinary authored H3 call through the combined machine/API/CLI
cohort with two rented GPUs, unchanged input, STORE=False on both ranks, positive
relayed model_source/ObjectFiles observations, successful output, and measured
per-rank caps. A failure must identify the unsupported operation rather than reopen
an embedded Store. Root owns rental/cohort authorization and execution.

Tracked in cozy-creator/tracker#321 and #320.
