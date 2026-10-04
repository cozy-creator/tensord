# Device executor

`device_executor.rs` is the Rust control seam around the published Runtime Python device
executor (`cozy_runtime.internal.executor`). Models, loaders, codecs and the author kernel stay in
the SDK. The Rust side owns spawn, identity checks, framing, capability gating, result verification
and output post-processing. It has no scheduler and no memory-admission policy.

## Spawn

`DeviceExecutor::spawn(ExecutorConfig)` (also `spawn_observed`, and `spawn_owned` through a
`ChildLauncher`) runs:

```
PYTHON -I -m cozy_runtime.internal.trampoline --expect-parent <pid> --oom-adj 1000 \
  --scope-backend inherit [--uid U --gid G] -- PYTHON -I -m cozy_runtime.internal.executor \
  --socket <path> --root <root>
```

- `ExecutorConfig`: `python`, `root` (0700), `socket` (0600), `environment` (configured
  locations), `seal`, `generation_hold`, optional `LaunchIdentity`.
- Environment (`Seal::environment`, the only place it is composed): the machine's environment
  without credential-like names or Runtime's erased prefixes (`COZY_`, `CUDA_`, `PYTORCH_`,
  `NCCL_`, `HF_`, `TMPDIR`, …), then configured locations, then the seal, which always wins:
  `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF` (default `expandable_segments:True`),
  `OMP_NUM_THREADS` (4), NCCL group seal for degree > 1, `COZY_HOME` (per-UID home, so the
  attention qualification is cached), JIT caches and `TMPDIR` (per machine run and generation),
  and the persistent per-UID kernel store.
- The SDK trampoline applies parent-death, no_new_privs and OOM ordering before imports. Every
  child leads its own process group; without a `LaunchIdentity` it keeps the service's UID and
  cgroup.
- The generation hold fd is inherited and survives exec and parent death.
- `on_birth(&ProcessBirth, &Cancellation)` runs right after spawn so the caller can journal it.
- Startup waits on a pidfd and the listener, with no timeout. A child that exits first returns
  `EndedBeforeStart` (status and stderr tail). Until Hello is verified a launch guard kills and
  reaps the child on every early return. `SO_PEERCRED` must match the child pid, UID and GID;
  Hello's pid, parent, group and every sealed value must match (the machine and its Runtime
  ship together).
- Start imports authored code, so journal authorization must come before `Start`.

## Wire

- 4-byte big-endian length + JSON, max 64 KiB (`MAX_DEVICE_FRAME`). Bulk data goes through the spool.
- Any outgoing object key named `token`, `credential`, `authorization`, `secret` or `jwt` is
  refused. Credentials cannot cross the seam.
- Commands (`cmd`): `hello`, `start`, `load`, `activate`, `prepare_request`, `invoke`, `budget`,
  `prefetch`, `share`, `revoke`, `unload`, `probe`, `shutdown`. The usual sequence is
  Hello -> Start -> Load -> Activate -> PrepareRequest -> Invoke.
- While waiting for a reply, `event=progress` frames go to `Services::progress`. `event=request`
  frames (optionally with an `SCM_RIGHTS` fd) go to `Services::request`. The reply must name the
  command, otherwise it is an error. Unknown events and fields are ignored.
- `Baseline` services answer every request `capability_unavailable`.
- A successful `model_source_read` answer must pass a read-only regular-file fd of the declared length.
- `device_tier` requests carry `descriptors: N` and are followed by N fds (an offer's exported GPU
  chunks); the answer names its own `descriptors` and the fds follow it (an ask that is `held`).
  They go to `Services::device_tier` ([resident custody](RESIDENT-CUSTODY.md)).

## Capabilities

`Hello.memory` lists offered capabilities. A missing one fails only that command with `Unsupported`.

| Command field | Requires |
|---|---|
| `Start.import_only` | `import_only` |
| `Budget` | `weight_plane/1` |
| `Load.device_weights`, `share` | `weights.attach/1` |
| `revoke` | `weights.revoke/1` |
| `Load.stages`, `Invoke.stages` | `stage/1` |
| `Load.descriptor_sources` | `model_sources.descriptors/1` and degree 1 |
| `Load.sealed_tiers` | `host_tiers.sealed/1` and `weight_plane/1` |
| `Load.pinned_bytes` | `load_pinned/1` and `weight_plane/1` |

Hello's version and revision fields are provenance only. There is no SDK or TensorFS version gate.

## Observations

`LoadFacts`, `PlaneFacts` and `Metrics` are typed optional records. An absent value is unknown,
and signed byte fields keep the SDK's `-1` sentinel. A `stage_exit` with `yielded` is the
executor's acknowledgement that it applied the returned budget, separate from the first exit.
Observations never authorize kills.

## Results

`read_result(spool, reply)` requires:
- `ok`, `outcome.terminal == "succeeded"`, `quiescent` and an empty `poisoned`;
- `result.canonical` in the spool, opened `O_PATH`/`NOFOLLOW`, matching `result_ref` SHA-256 and length.

`postprocess(codec, spool, reply)` runs `device_codec.py` with the generation's Python (`-I`, env
cleared, spool passed as an fd). It reuses the SDK's `encode_frame` and its own
`blob_path` resolver to encode deferred host frames and bind outputs. Each
`AssetBinding` carries `output_id`, `asset_ref`, spool `name`, `kind`, `media_type`, `length`,
`producer_digest` (SDK BLAKE2b) and `sha256`. Rust re-hashes every bound file before returning.
The root must then take durable custody before reporting success.

## Cancellation and lifetime

Nothing is killed because time passed. The wedge rule is Runtime's `liveness`: a meter still for
longer than eight times the longest pause it has shown, and at least six samples (30 s).

- Every `command` runs under a watch. Non-invoke commands use CPU plus bytes moved
  (`process::burn`); time the machine spends answering an executor request is excused, and so
  is a job root's while any of its child runs is unfinished (`DeviceExecutor.waits`: a child
  is watched by its own executor). A wedged executor and its process group are killed; the
  call returns the measurement.
- `cancel(request_id)` atomically writes the stock `executor.cancel` marker, keyed by request, so a
  stale marker cannot cancel a later request. The executor stops at its next safe point. From
  then on the invocation's frames are the meter; if they stop, it is killed. An invocation without
  a cancel is never judged. Observer teardown never cancels.
- `terminate()` closes the channel, lets the executor stop at its next exchange, kills only a
  measured wedge, then reaps it and every member left in its process group (a group's followers),
  and frees `retain_until_exit` resources. `shutdown()` asks first.
  `Drop` does the same before returning, so a slot or reservation is released only after the exit. If exit cannot be observed, resources are leaked, not released.
- Containment (`scope.rs`): with `cgroup_namespace`, each executor gets its own scope. Where the
  host delegates a writable unified hierarchy it is a cgroup-v2 below the machine's: the
  trampoline joins it before importing anything, Hello must show the executor inside it, and a
  kill writes `cgroup.kill`. Elsewhere it is a token (Docker, RunPod: cgroups read-only or v1, no
  `CAP_SYS_ADMIN`, measured in `E/pod-cgroup/`). The Runtime's own
  `COZY_EXECUTOR_SCOPE=<namespace>-<id>` goes in the leader's environment (a fork's, in the
  environment it is forked with) and every descendant inherits it. A kill and the end of the scope
  use a `/proc` census: processes carrying the token, plus anything still parented below one.
  Each is killed through its pidfd and its exit awaited, until a census finds none alive. Either
  backend reaches descendants that called `setsid` or double-forked. After the executor exits,
  whatever is still in the scope is killed and counted (`Ended.stragglers`). The machine's
  supervisor is the subreaper (`PR_SET_CHILD_SUBREAPER`): escaped processes are adopted and reaped
  there, never by the service, whose own child handles keep their statuses. A token on no live
  scope of this service (an earlier run's, or one dropped without an end) is killed at startup
  and at each token scope's end. Untokened processes, such as an operator's SSH jobs, are never
  touched.
- No copy drain before a kill: a kill only follows a measured wedge, so the executor is not
  cooperating and could not drain. Nothing it was copying is shared yet: Degree 2 regions are
  offered to custody only by `Share` after a synchronized, completed call; host-tier layouts are
  sealed read-only memfds the machine filled; outputs are taken into custody only from a
  quiescent reply. Sources it read stay retained until its exit is observed.

## Known gaps

- The legacy output-path resolver is imported from the SDK worker module. An executor
  `output_bindings` capability would remove it.
- Encoders read whole raw buffers. There is no streaming post-processing.
- Same-UID package code can leave containment on purpose: write a delegated ancestor's
  `cgroup.procs`, or (token) exec with a scrubbed environment and escape its ancestry. A fork's
  fork-only descendants carry its import-only parent's token, so they end with that parent. It
  can also reopen store paths.
- Descriptor sources still use native TensorFS plane, header and read-plan code inside the executor.
- Degree > 1 reads the store or the sealed tier (rank 0 shares it with followers); descriptor
  sources are world-one.
