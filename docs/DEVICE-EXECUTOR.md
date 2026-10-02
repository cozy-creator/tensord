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

- `ExecutorConfig`: `python`, `root` (0700), `socket` (0600), `environment` (env is cleared, then
  only these vars are set), `generation_hold`, optional `LaunchIdentity`.
- The SDK trampoline applies parent-death, no_new_privs and OOM ordering before imports. Without
  a `LaunchIdentity` the child inherits the service's cgroup, UID and process group.
- The generation hold fd is inherited and survives exec and parent death.
- `on_birth(&ProcessBirth, &Cancellation)` runs right after spawn so the caller can journal it.
- Startup waits on a pidfd and the listener, with no timeout. The peer's `SO_PEERCRED` must match
  the child pid, UID and GID. `Hello.pid` must match too.
- Start imports authored code, so journal authorization must come before `Start`.

## Wire

- 4-byte big-endian length + JSON, max 64 KiB (`MAX_DEVICE_FRAME`). Bulk data goes through the spool.
- Any outgoing object key named `token`, `credential`, `authorization`, `secret` or `jwt` is
  refused. Credentials cannot cross the seam.
- Commands (`cmd`): `hello`, `start`, `load`, `activate`, `prepare_request`, `invoke`, `budget`,
  `prefetch`, `unload`, `probe`, `shutdown`. The usual sequence is
  Hello -> Start -> Load -> Activate -> PrepareRequest -> Invoke.
- While waiting for a reply, `event=progress` frames go to `Services::progress`. `event=request`
  frames (optionally with an `SCM_RIGHTS` fd) go to `Services::request`. The reply must name the
  command, otherwise it is an error. Unknown events and fields are ignored.
- `Baseline` services answer every request `capability_unavailable`.
- A successful `model_source_read` answer must pass a read-only regular-file fd of the declared length.

## Capabilities

`Hello.memory` lists offered capabilities. A missing one fails only that command with `Unsupported`.

| Command field | Requires |
|---|---|
| `Start.import_only` | `import_only` |
| `Load.host_tier`, `Budget` | `weight_plane/1` |
| `Load.stages`, `Invoke.stages` | `stage/1` |
| `Load.descriptor_sources` | `model_sources.descriptors/1` and degree 1 |
| `Load.host_tier_owner` | `host_tiers.owner/1`, `host_tier`, and degree 1 |

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
cleared, spool passed as an fd). It reuses the SDK's `encode_frame` and the
`AttemptEngine._blob_path` resolver to encode deferred host frames and bind outputs. Each
`AssetBinding` carries `output_id`, `asset_ref`, spool `name`, `kind`, `media_type`, `length`,
`producer_digest` (SDK BLAKE2b) and `sha256`. Rust re-hashes every bound file before returning.
The root must then take durable custody before reporting success.

## Cancellation and lifetime

- `cancel(request_id)` atomically writes the stock `executor.cancel` marker, keyed by request, so a
  stale marker cannot cancel a later request. Observer teardown never calls it.
- `shutdown()` sends `Shutdown`, waits for exit and removes the socket.
- Dropping the handle closes the channel but writes no cancel marker. Resources passed to
  `retain_until_exit` are freed only after the pidfd reports exit. If exit cannot be observed, they
  are leaked rather than released. The supervisor, never a UI observer, owns the handle.

## Known gaps

- The legacy output-path resolver is imported from the SDK worker module. An executor
  `output_bindings` capability would remove it.
- Encoders read whole raw buffers. There is no streaming post-processing.
- No separate containment scope. Same-UID or privileged package code can reopen store paths.
- Descriptor sources still use native TensorFS plane, header and read-plan code inside the executor.
- Degree > 1 and multi-GPU/NCCL are not covered by descriptor or host-tier owner paths.
