# GPU service

`gpu_service.rs` (`GpuPool`) coordinates published, cached GPU callables through retained
Runtime executors. TensorD supplies model data and shared CPU buffers and uses
`memory::GpuMemory` to decide device budgets and eviction. Runtime constructs models and
performs GPU operations through its TensorFS plane in the executor's CUDA context.
Acceptance, journal, progress and output custody stay in `Engine`; scheduling stays in `service.rs`.

## Configuration

`cozy-machine serve --gpu-config <json>` loads a root-sealed `GpuConfig`:
- `devices`: the GPU envelope, `"0"` or `"0,1,..."`. A plan of degree K runs on the first K;
  its executor is sealed to those K (`CUDA_VISIBLE_DEVICES`).
- `models: [ModelGrant { package, slot, repository, release, lane, manifest, components }]`: the only
  authority over cached model bytes.
- `packages: [{ package, release, distribution, generation }]`: published-package mapping.
- `pinned_budget_bytes`; `authorized_device_limit_bytes` (Load's limit only where NVML cannot read
  the device total).
- `memory`: `{ floor_bytes, sample_log }`: raise the free floor; log each 1 s NVML reading.
- `environment`: keys containing `TOKEN`, `SECRET` or `PASSWORD` are refused.
- `host`: optional `{fill_threads, ttl_seconds}` for the always-on `HostTier`. `identity`: optional `{uid, gid}`.

## Preparation

`prepare_root` reads only the installed static interface.
- Every model the entrypoint declares at `<entrypoint>.models.<param>` gets a slot (H3 turbo:
  base and LoRA). Several slots load as one many-model construction (`Load.models`).
- Choices with `source`, `profiles` or `adapters` are unsupported.
- Exactly one `ModelGrant` must match each slot. Declared `component_use` must be within its components.
- Degree: the widest fitting ladder rung's `gpus` (published), else the widest group every slot
  declares (`sequence_parallel.degrees`, one always) within the envelope (Runtime `machine_lanes.widths`).
- The `GpuPlan` id hashes {actor, generation, entrypoint, slots, degree}. It is journaled as a
  `Preparation`, and `submit` records `preparation_id`.
- An empty `installation_id` binds a `published-<hash>` installation from `packages`.

## Dispatch

- One slot: GPU work dispatches only when no GPU record is active and no startup fence exists.
  CPU dispatch continues meanwhile.
- Startup: every journaled birth still alive from a previous machine run (GPU births including
  completed requests', and any nonterminal run's) is killed, since nothing can adopt it. GPU
  dispatch stays fenced until each exit is observed. Leftover executor scopes of this machine
  (cgroups, tokens) are killed and removed. A leader still tearing down (no pidfd yet) is watched by sampling.
- Groups (degree K > 1): one executor sealed to the K GPUs with the NCCL seal (`NCCL_NVLS_ENABLE=0`,
  `NCCL_P2P_LEVEL=NVL`); `Start`/`Load` carry `sequence_parallel_degree: K`. Rank 0 spawns, dials back
  and forms its followers inside `Start` (Runtime `RankGroup`); the machine never talks to a follower.
  Every command's watch meters rank 0 plus its process-group members, so formation ends only on
  measured lack of progress. Each GPU of the group is admitted by its own memory decision (device
  order). With `rank_cells/1` each rank gets its own GPU's cap (`rank_caps`) and its own budget cell
  (rank 0 hands the machine one per follower at formation), so every GPU's floor watchdog lowers the
  process on that GPU. Followers' pids from `Start` name their GPU's tenant for NVML.
  Teardown waits for every group member's exit. Startup fences GPU dispatch until a previous
  machine's leaders and their followers are gone. A refusal or poisoned group call fails the run
  with the executor's own code (a group's first fault names its GPU). Descriptor sources and
  Degree 2 custody stay world-one. A group forks from the generation's import-only parent like a
  single GPU does: the child takes the lane and NCCL seal with its environment; followers are spawned
  by rank 0.
- A cold session is admitted first: its context estimate (twice the largest measured here, else
  1 GiB) and known first working set are reserved, making room by the ladder below. It is spawned by
  the pool's `ChildLauncher`, which journals the birth, then sent Start, Load (with that cap),
  `Budget` (pinned, if `weight_plane/1`) and Activate once. Each request sends PrepareRequest and
  Invoke. Executors of other plans stay; the policy decides what they keep on the device.
- Each Invoke carries a real grant: `cap_bytes` (whole process: context + torch + plane) to
  `process_cap/1` executors, else a plane budget derived from it. `stages` is false: no per-stage
  exchanges.
- Making room, in order: idle executors unmap (`Budget{vram 0}`, LRU), idle executors end (only
  while the lowest rung is unmet), a running call shrinks through its budget cell. Never refused by
  size: with nothing left, the cap is what there is.
- Floor: 512 MiB or 1/16 of the card on a display GPU, 256 MiB otherwise. A 1 s sample below it
  during a call lowers the call's cap through its budget cell by the deficit.
- An executor's failed terminal (its own refusal before entry, or a failed invoke) first keeps a
  triage bundle (`triage.rs`: terminal, traceback, executor pid and stderr tail; canonical JSON
  under `execution/triage/`, fsynced) and binds it in the journal; the outcome body names it as
  `triage_bundle`, and `ReadMachineExecutionTriage` returns its bytes.
- Start happens only after `authorize_managed`.
- Invoke must be quiescent. Then `postprocess` and `managed_result` run. A custody failure is
  `failed`.
- On error the session is terminated (exit observed, wedge killed) before the run is settled:
  CANCELED if a cancel was journaled, otherwise FAILED with the reason, including a pre-start exit
  with its stderr tail. A never-authorized attempt hit by a transient OS shortage (EAGAIN,
  ENOMEM, EMFILE, ENFILE) is requeued. A retained session can lose its channel (EPIPE,
  ECONNRESET, EOF) at PrepareRequest, which enters no handler: a killed executor closes its
  socket before it is a zombie, so the pre-reuse check can miss it. That attempt stays with its
  dispatch (journal `redeliver`: back to awaiting its first executor) and starts once more on a
  fresh executor, whose own failure is FAILED. An unprovable exit leaves the run charged and
  nonterminal.

Executor requests:
- `device_room`: idle tenants give room (unmap, then end); the answer's `cap_bytes` raises the
  caller's cap into it. `stage_enter`/`stage_exit` (never asked for) keep the budget.
- `sealed_tier` goes to `HostTier` (`HOST-TIER.md`); `Load.pinned_bytes` carries the pinned budget.
- `model_source` goes to `ModelSources` for selected headers/assets in sealed descriptors;
  payload buffers and granted object files are supplied by `HostTier`.
- `publish` (`Outputs.publish`) goes to `products.rs`: the spool file is retained as a native
  one-file tree bound to the run's actor, then journaled as a `product` (list outputs APPEND,
  single outputs SET; re-publishing identical SET bytes is a no-op). Answers `Published`.
- Progress counts only `advance > 0`.

## Launch identity

With `identity`:
- A non-root owner may only name its own UID and GID.
- The trampoline gets `--uid/--gid`. Peer credentials must match.
- The pool root and its parent become 0710, keeping their owner. Executor dirs and per-request
  spools become identity-owned 0700.
- The interface file and cancel marker become 0440. Journal, results and generations are never
  changed.
- The socket path is `<root>/executor` and at most 107 bytes. Without an identity it is
  `/proc/<pid>/fd/<dirfd>/executor`.

`ChildLauncher` spawns from one pool-owned thread, because Linux ties `PDEATHSIG` to the creating
thread.

## Known gaps

- Groups always take the first K envelope GPUs; one GPU call runs at a time machine-wide. No adapters.
- Per-rank caps and cells need an executor with `rank_cells/1` and `process_cap/1`; otherwise every
  rank takes the smallest cap and only rank 0's GPU has a floor watchdog cell.
- Executors before `process_cap/1` get only a plane budget: their context and activations are
  estimated, not capped.
- One Python post helper per request.
- Each executor gets its own scope: a cgroup-v2 where the host delegates one (laptop systemd
  scopes, rootful hosts), else a token (containers with a read-only or v1 hierarchy, RunPod).
- Seal: `threads` config field (torch's allocator is set in Runtime code); `<root>/home`, `<root>/kernels` (per UID) and
  `<root>/jit/<run>/<generation>`. Earlier runs' JIT scopes, and executor roots (logs) older than a day, are removed at pool start.
- `PDEATHSIG` retention after a UID drop is unverified.
