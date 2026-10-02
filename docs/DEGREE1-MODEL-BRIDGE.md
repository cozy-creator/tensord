# Degree 1 model bridge: reuse the stock executor

The CPU author-session proof is useful for interpreter reuse and output custody. It
cannot yet run SDXL or Anima. Preserve the existing Python device executor behind a
typed Rust command adapter rather than growing another model runtime here.

Static census on 2026-10-02: Runtime `c04916f`, TensorFS `ec2002f`, packages `797f976`.
Those checkouts differ from the installed SDK gates (Runtime 0.18.89/0.18.99 with
TensorFS 0.3.89). The experimental Rust service still links TensorFS tag v0.3.87.
This is provenance, not a compatibility or exact-source gate. Qualify the specific
operations and negotiated capabilities across those independent versions.

## Existing reusable machinery

- Public `ModelRegistry.acquire` constructs and coalesces generations by binding identity,
  installs checked construction facts, and supplies `Invocation.models` through
  `models(callable_name)`. `unload_all` owns lifecycle teardown. Runtime
  `author/_model.py:949`, `:971`, `:1014`, `:1032`.
- Public `Loader` constructs the authored factory under the selected substrate and runs
  `Backend.fit → materialize → fill → commit`, with poison on failed construction.
  Models own `load`, optional `warm`, and `unload`; they do not own placement.
  Runtime `author/_loader.py:433`, `:656`, `:709`; `author/_model.py:560`.
- The stock executor combines that machinery with TensorFS fit/read plans, generic region
  partitioning, residency, encoded leaves, attention choice, adapters and measured OOM
  recovery. Runtime `internal/executor.py:1970`, `:2041`, `:2132`;
  `internal/weights.py:855`, `:1032`, `:1248`.
- SDXL and Anima already import maintained Diffusers components. Their serving functions
  require an admitted `model` parameter; the CPU session's empty `Invocation.models`
  cannot satisfy it. Packages `sdxl/sdxl/__init__.py:75`, `:323`, `:529`;
  `anima/anima/__init__.py:35`, `:244`, `:432`.

The real classifier test trains again on every invocation. Reusing its interpreter or
its SDK imports establishes no model construction or GPU-weight cache reuse.

## Stock executor control contract

Rank zero starts with the package generation's interpreter:

```text
python -m cozy_runtime.internal.executor --socket OWNER_SOCKET --root EXECUTOR_ROOT
```

It connects to an owner Unix stream. The existing seam uses four-byte network-order
length plus canonical JSON, a 64 KiB control cap, and typed commands/replies. Large result
bytes use the brokered spool (`result.canonical`) with digest/length descriptors. Runtime
`internal/executor.py:4361`; `internal/seam.py:1`, `:39`, `:52`.

The minimum device bridge is negotiated `Hello → Start → Load → PrepareRequest → Invoke`,
plus residency/probe, explicit attempt cancellation and shutdown. `Start.import_only`
requires the advertised capability; older peers must receive their supported baseline.
`Load` carries model `Binding`, selected artifacts, budgets and optional host-tier/stage
exchanges. Rust supplies the scheduling and journal authority, not another copy of the
Python construction path. Runtime `internal/executor_commands.py:122`, `:147`, `:205`,
`:265`, `:283`.

Importing authored code during preparation is still execution. Record and authorize that
preparation before `Start`; source description remains the AST operation. Keep legacy
environment/bootstrap handling an explicit adapter until typed startup records cover it.
Do not infer that this adapter already satisfies every greenfield rule.

## Blocking contracts for degree 1

| Contract | What must be supplied and qualified |
| --- | --- |
| Model selection | Typed binding path/class/parameter, immutable checkpoint identity, logical tensor schema, config, verified tokenizer/assets and supported encoding/adapter route. `Artifact` already defines this in `author/_loader.py:321`. |
| Construction | Existing fake-tensor substrate, `ModelRegistry` and backend fit/transaction. Never load weights by calling copied Diffusers code or bypassing the census. |
| Host layout | Weight-set identity, manifest hold, regions, part offsets/lengths/dtypes, layout span and live-holder accounting. Layout is derived after Python constructs and partitions the graph; first-install startup cannot assume it is already cached. |
| Read source | Current `Plane.source` requires a concrete TensorFS `Store` and `ReadLease`; registration requires a `ReadPlan`. A plain object fd is insufficient. TensorFS `python/tensorfs/plane.pyi:238`; Runtime `internal/weights.py:1266-1295`. |
| Host adoption | `Plane.register(..., host_fd=...)` can adopt a matching pinned tier. That does not remove the store/lease dependency or transfer sole writer authority. Executor and owner still need explicit region/hold lifetimes and physical reclamation. TensorFS `plane.pyi:241`, `:295`. |
| GPU admission | Reserve context/library/activation/staging/weight needs before context creation. Degree 1 leaves GPU allocations/copies in the executor; retain its event-fenced cleanup and actual OOM recovery. A ledger alone does not page arbitrary tensors. |
| Component stages | Reuse existing generic `WeightResidency`, block hooks and supported recovery paths. Declared model scopes must acquire before computation and drain before release; shared host buffers cannot be punched during DMA. Runtime `executor.py:2132`; `weights.py:1463`. |
| Encoded image custody | Both current packages call `save_image(..., format='webp')`, producing deferred `Attempt.frames`. The minimal author bridge rejects these today. Preserve the generic codec/post path: `author._codec.encode_frame` at418 handles PNG/WebP/FLAC/MP4. Hand off validated raw-frame facts and seal encoded output before success. SDXL `:631`, Anima `:468`. |
| Import removal | Current TensorFS Python `Store.open` exposes mutation APIs and has no read-only flag in its declared facade (`_ext.pyi:445`). A reader-only source capability or native descriptor-based plane source is needed before claiming executors stop importing TensorFS or only Rust can write the store. |

Two honest increments are available. First adapt the stock executor and its existing
host-tier exchange so the Rust owner retains host bytes across executor death, while
explicitly retaining the legacy TensorFS dependency. Then qualify a descriptor/read-source
API that removes store mutation authority and unnecessary package TensorFS imports.
Read-only mode alone does not establish object-level GC protection or safe shared-region
reclamation.

## Gates before a model claim

Use one coherent checkpoint/lane, the unchanged package and its normal request semantics.
First qualify the stock CPU command path; then root coordinates a non-display GPU and
the existing safety fixes before any laptop promotion. Prove saved WebP output, repeated
fresh-seed inference, preparation count, construction/executor reuse, model switching,
measured context admission, physical host reclamation and executor death. Compare the
unchanged old stack under matched conditions through ordinary Cozy CLI; this document and
the classifier component proof do not establish those gates. No GPU or rental was used
by this agent.
