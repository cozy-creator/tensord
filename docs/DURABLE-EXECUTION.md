# Durable CPU execution foundation

`journal.rs` and `execution.rs` provide one acceptance/transition authority. They do
not implement a second scheduler, package installer, public API or authentication system.
The machine owner admits work and resolves an immutable environment before dispatch.

## Owner API

```rust
let engine = Engine::open(state_root)?;
let record = engine.submit(stable_key, invocation)?; // FULL WAL commit before receipt
engine.dispatch(&record.id, trusted_runner_config)?; // only queued work is claimable
let current = engine.get(&record.id)?;               // observation has no side effects
engine.cancel(&record.id, authenticated_actor)?;    // explicit durable authority
let recovered = engine.reconcile()?;                // no adoption or automatic replay
let output = engine.open_result(&record.id, 0)?;    // revalidates durable content identity
```

`Invocation` contains package distribution, immutable generation, application module,
entrypoint and JSON input. `RunnerConfig` contains trusted interpreter/module/import
paths and an acquired shared environment-generation hold. Parent and runner independently
hold the generation. The API adapter must scope idempotency keys and authorize each resource;
neither journal identifiers nor application inputs grant execution authority.

`dispatch` spawns the configured interpreter directly, without shell evaluation. It
inherits one connected Unix-stream descriptor, null stdin and file-backed stdout/stderr.
No CUDA/NVML imports occur here. Imported package code belongs exclusively to the runner.
`PYTHONPATH`, if provided, configures import paths rather than changing execution logic.

States are `queued`, `starting`, `running`, `completed`, `failed`, `canceled`. `revision`
is a monotonic observation cursor; cancellation actor, actual completed units, failure
and waiting reason are typed fields. A repeated key with equivalent invocation returns
the same record; different semantics return conflict. Map ordering does not change equality.

Productive progress coalesces into one in-memory snapshot per supervised execution,
with detail capped at 2 KiB; duplicate/regressed counts produce no update. Queries and
duplicate receipts overlay this snapshot without SQLite writes. Cancellation and terminal
settlement merge the latest snapshot in the same FULL-WAL transaction as their authority.
Ordinary progress produces no fsync, event history or new dispatch loop. Losing observational
telemetry is not evidence of process death or permission to replay/kill an execution.

Each authoritative transition into `running`, including explicit cancellation, reserves
a window of 2^32 observation revisions in `revision_ceiling`. Volatile progress uses cursors
inside that window. Exhaustion reserves another window in a rare allocation commit; checked
arithmetic never wraps/reuses a cursor. Durable transitions jump beyond the reserved high-water.
A new owner's snapshot of a still-live former executor advances to the old ceiling, so lost
progress cannot move an observer's numeric revision backward. Older records without the field
default to no reservation. These are status cursors, separate from artifact/output event indexes;
the window is a sequence allocation mechanism, not a duration or process-liveness condition.

## Startup and recovery

1. Commit `starting`, increment attempt, then spawn a runner which waits for Invoke.
2. Record `(boot_id, pid, /proc start_ticks)` immediately after spawn. Read Ready before
   Runtime/package imports. Validate the actual PID and CPU-author capability.
3. Commit `running` **before** sending Invoke. This is conservative start authorization:
   even a failed write cannot establish that authored code never began.
4. Observe typed progress/result/failure on that private socket, await process termination,
   take output custody, and commit exactly one terminal outcome.

Observer disconnect, terminal loss and dropping query records do not invoke cancellation.
Explicit cancellation commits its actor before sending a cooperative cancel command. There
is no elapsed-time process kill. A runner canceled before start authorization executes no
authored code. A noncooperative progressing package retains its execution obligation.

After owner loss, a recorded live exact process birth remains nonterminal; a permission or
inspection error is not proof of death. Once that birth terminates, an authorized run becomes
failed unless completion was already durably committed. It never runs again automatically.
An attempt that never received start authorization can return to queued with a visible
waiting reason, after its recorded birth terminates. A starting record without a recorded
process was never authorized: a possible spawn-gap runner can only wait for Invoke/EOF.

The owner periodically calls `reconcile`; polling is observation, never a timer-based kill.
Launch failures likewise remain queued with `waiting_reason` and do not self-retry. The
scheduler must require an observed changed condition or an explicit retry before redispatching
these records. A duplicate submission alone is not that change.

## Runner wire

Frames are four-byte network-endian length plus JSON, at most 1 MiB. Unknown additive
fields and advisory event kinds are tolerated; no version number admits/refuses a runner.
Required operation capabilities are checked locally. All execution events identify the
accepted execution; another execution's event cannot complete it.

- Ready: `kind`, actual `pid`, `capabilities: ["runtime.author-cpu/1"]`.
- Invoke: `execution_id`, flattened Invocation and absolute `output_root`.
- Progress: `execution_id`, monotonically increasing `completed_units`, `detail`.
- Result: `execution_id`, `value`, `artifacts: [relative_path]`.
- Failed: `execution_id`, `code`, `detail`.
- Cancel/Canceled: `execution_id`; canceled needs the journal's explicit authority.

The production bridge executes through public Runtime author `prepare`/`invoke`, separate
from the fixture runner used to verify supervision boundaries.

## Durable output custody

Only after the executor terminates successfully does the machine copy declared regular
files into its private results tree, hash them with TensorFS SHA-256, fsync bytes/permissions,
rename, and fsync destination directories. Paths are opened component-by-component using
`openat` plus `O_NOFOLLOW`; absolute paths, parent escapes and symlinks are rejected.
Final inodes are inspected with `O_PATH` before reopening the retained regular file, so a
declared FIFO cannot block custody and a special-device artifact cannot make the core open
a driver. The journal's completed record then commits the result and content identities. Incomplete
copies and uncommitted files cannot produce a successful receipt; replay cannot acknowledge
new bytes for an old completion. Reads verify the stored digest and length before exposure.

The byte copies and read-only permissions are machine-managed immutability, not an adversarial
same-UID package sandbox. An executor's descendants, stronger process-tree isolation, sealed-fd
output transfer, cleanup of orphan pending files and bounded log/outbox policies remain product
integration work. In particular, same-UID malicious code can chmod an output; reads detect its
mutation, but this CPU foundation does not claim to contain untrusted packages.

## Evidence

`cargo test --test durable_execution` passed twelve actual-process/socket/filesystem cases
(0.36 s; one explicitly invoked child helper is ignored by the parent harness):
CPU matrix inference and persisted output; observer-safe duplicate acceptance; explicit
executor SIGKILL without repeated effects; actor-attributed cooperative cancel; visible
launch failure followed by changed-interpreter retry; symlink escape and mutation detection;
live exact-birth retention after journal reopen; and never-authorized restart retry after
exact termination. An actual Rust-owner SIGKILL case also keeps a live orphan's obligation,
then settles failure after that exact birth ends without repeating effects. Test deadlines
only bound observations; they never kill a package.

The coalescing gate publishes 257 completed units through actual socket frames: live queries,
list and duplicate receipts observe them while an independent SQLite reader and WAL size
remain unchanged. Terminal custody persists the latest count/revision; explicit cancellation
likewise commits the latest progress. A canceled queued run survives reopen without package
execution. Actual owner death demonstrates that missing volatile progress never settles a
still-live process.
The cursor boundary test renews an exhausted reservation, retains the latest count, and
reopens the monotonic terminal revision; an older record without the optional ceiling decodes.

These are supervision component checks. Ordinary Creator CLI, installed Runtime author
bridge, full public-service crash boundaries, systemd/container reaping, browser/Hub consumers,
child calls and diffusion/GPU/NCCL qualification remain separate required gates.
