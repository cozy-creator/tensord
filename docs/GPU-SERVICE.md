# GPU service

`gpu_service.rs` (`GpuPool`) runs published, cached GPU callables through one retained device
executor. Acceptance, journal, progress and output custody stay in `Engine`. Scheduling stays in
`service.rs`.

## Configuration

`cozy-machine serve --gpu-config <json>` loads a root-sealed `GpuConfig`:
- `devices`: exactly one device, exported as `CUDA_VISIBLE_DEVICES`.
- `models: [ModelGrant { package, slot, repository, release, lane, manifest, components }]`: the only
  authority over cached model bytes.
- `packages: [{ package, release, distribution, generation }]`: published-package mapping.
- `source_mode`: `auto` (descriptors if offered), `legacy`, or `descriptors` (fails if not offered).
- `plane_budget_bytes` (default -1), `pinned_budget_bytes`, `authorized_device_limit_bytes`, `stages`.
- `environment`: keys containing `TOKEN`, `SECRET` or `PASSWORD` are refused.
- `host`: optional `SharedHostPlane`. `identity`: optional `{uid, gid}`.

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
- Startup fence: every journaled GPU birth that is live or unknown, including those of completed
  requests, blocks GPU dispatch until it exits.
- A cold session is spawned by the pool's `ChildLauncher`, which journals the birth. It then sends
  Start, Load, `Budget` (if `weight_plane/1`) and Activate once. Each request sends PrepareRequest
  and Invoke.
- A different plan shuts the old executor down, waiting for its exit, before spawning.
- Start happens only after `authorize_managed`.
- Invoke must be quiescent. Then `postprocess` and `managed_result` run. A custody failure is
  `failed`.
- On error: never started, back to queued. Birth ended, failed. Live or unknown, stays nonterminal.

Executor requests:
- `stage_enter`/`stage_exit` are granted the static `plane_budget_bytes`.
- `host_tier*` goes to `SharedHostPlane`, or is acknowledged if no host plane is configured.
- `model_source_read` goes to `ModelSources`.
- Progress counts only `advance > 0`.

## Launch identity

With `identity`:
- A non-root owner may only name its own UID and GID.
- The trampoline gets `--uid/--gid` and a new process group. Peer credentials must match.
- The pool root and its parent become 0710, keeping their owner. Executor dirs and per-request
  spools become identity-owned 0700.
- The interface file and cancel marker become 0440. Journal, results and generations are never
  changed.
- The socket path is `<root>/executor` and at most 107 bytes. Without an identity it is
  `/proc/<pid>/fd/<dirfd>/executor`.

`ChildLauncher` spawns from one pool-owned thread, because Linux ties `PDEATHSIG` to the creating
thread.

## Known gaps

- One device, world one, one model slot. No adapters.
- Static stage budget. It does not bound context or activation memory, so it is unsafe on display GPUs.
- One Python post helper per request.
- No separate cgroup scope.
- `PDEATHSIG` retention after a UID drop is unverified.
