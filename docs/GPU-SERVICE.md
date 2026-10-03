# GPU service

`gpu_service.rs` (`GpuPool`) runs published, cached GPU callables through one retained device
executor per plan; `memory::GpuMemory` decides every device byte. Acceptance, journal, progress and output custody stay in `Engine`. Scheduling stays in
`service.rs`.

## Configuration

`cozy-machine serve --gpu-config <json>` loads a root-sealed `GpuConfig`:
- `devices`: exactly one device, exported as `CUDA_VISIBLE_DEVICES`.
- `models: [ModelGrant { package, slot, repository, release, lane, manifest, components }]`: the only
  authority over cached model bytes.
- `packages: [{ package, release, distribution, generation }]`: published-package mapping.
- `source_mode`: `auto` (descriptors if offered), `legacy`, or `descriptors` (fails if not offered).
- `pinned_budget_bytes`; `authorized_device_limit_bytes` (Load's limit only where NVML cannot read
  the device total).
- `memory`: `{ floor_bytes, sample_log }`: raise the free floor; log each 1 s NVML reading.
- `environment`: keys containing `TOKEN`, `SECRET` or `PASSWORD` are refused.
- `host`: optional `{fill_threads, ttl_seconds}` for the always-on `HostTier`. `identity`: optional `{uid, gid}`.

## Preparation

`prepare_root` reads only the installed static interface.
- The entrypoint must declare exactly one model at `<entrypoint>.models.<param>`.
- Choices with `source`, `profiles` or `adapters` are unsupported.
- Exactly one `ModelGrant` must match. Declared `component_use` must be within its components.
- The `GpuPlan` id hashes {actor, generation, entrypoint, binding}. It is journaled as a
  `Preparation`, and `submit` records `preparation_id`.
- An empty `installation_id` binds a `published-<hash>` installation from `packages`.

## Dispatch

- One slot: GPU work dispatches only when no GPU record is active and no startup fence exists.
  CPU dispatch continues meanwhile.
- Startup: every journaled birth still alive from a previous machine run (GPU births including
  completed requests', and any nonterminal run's) is killed, since nothing can adopt it. GPU
  dispatch stays fenced until each exit is observed.
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
- Start happens only after `authorize_managed`.
- Invoke must be quiescent. Then `postprocess` and `managed_result` run. A custody failure is
  `failed`.
- On error the session is terminated (exit observed, wedge killed) before the run is settled:
  CANCELED if a cancel was journaled, otherwise FAILED with the reason, including a pre-start exit
  with its stderr tail. Only a never-authorized attempt hit by a transient OS shortage (EAGAIN,
  ENOMEM, EMFILE, ENFILE) is requeued. An unprovable exit leaves the run charged and nonterminal.

Executor requests:
- `device_room`: idle tenants give room (unmap, then end); the answer's `cap_bytes` raises the
  caller's cap into it. `stage_enter`/`stage_exit` (never asked for) keep the budget.
- `sealed_tier` goes to `HostTier` (`HOST-TIER.md`); `Load.pinned_bytes` carries the pinned budget.
- `model_source_read` goes to `ModelSources`.
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

- One device, world one, one model slot. No adapters. One call per GPU at a time.
- No host ledger (pinned tier, RSS/PSS, cgroup headroom) in the policy yet.
- Executors before `process_cap/1` get only a plane budget: their context and activations are
  estimated, not capped.
- One Python post helper per request.
- No separate cgroup scope; containment is the executor's process group.
- Seal: `alloc_conf`/`threads` config fields; `<root>/home`, `<root>/kernels` (per UID) and
  `<root>/jit/<run>/<generation>`. Earlier runs' JIT scopes are removed at pool start.
- `PDEATHSIG` retention after a UID drop is unverified.
