# Durable execution

`journal.rs` is the machine's single SQLite acceptance and transition authority. `execution.rs`
(`Engine`) supervises runner processes over it. Neither schedules, installs or authenticates:
the owner (`service.rs`) admits work and resolves an immutable environment before dispatch.

## Journal

- `executions.sqlite3` in the state root, WAL with `synchronous=FULL`. A successful accept is a
  committed receipt. The same file holds installations, preparations, public terminal
  projections, native outputs, input intakes and submission closures.
- `workspace_id()` is a v4 UUID created once and persisted.
- States: `queued`, `starting`, `running`, `completed`, `failed`, `canceled`.
- `Invocation`: `package`, `generation`, `module`, `entrypoint`, `input` (JSON).

## Acceptance

- `accept(key, invocation)`: key 1..512 bytes. Equal retry returns the record. A different
  invocation under the same key is `AlreadyExists`.
- `accept_public(SubmissionContext, invocation)`: context is the verified `actor`, `request_id`,
  `submission_id`, `expected_workspace_id`, authored digests, `publication_authorization_id`
  and optional `preparation_id`. (actor, request) and (actor, submission) are unique. Changed
  bindings fail and never replace accepted work. A `preparation_id` must name a preparation
  whose installation generation equals `invocation.generation`.
- `close_submission` writes a tombstone in the same immediate transaction boundary as acceptance.
  If acceptance won, it returns the receipt and does not cancel. If closure won, later acceptance
  fails.
- `AdmissionError` (inside `io::Error`): `WorkspaceMismatch`, `BindingConflict`, `SubmissionClosed`.
- Credentials and grant secrets never enter the context, journal or input. A capture's
  `record_owner` label never selects authority.

## Engine

```rust
let engine = Engine::open(state_root)?;
let record = engine.submit(key, invocation)?;   // committed receipt
engine.dispatch(&record.id, runner_config)?;    // false unless queued
engine.cancel(&record.id, actor)?;              // durable actor first, then cooperative Cancel
let open = engine.reconcile()?;                 // no adoption, no replay
let file = engine.open_result(&record.id, 0)?;  // re-verifies digest and length
```

- `RunnerConfig`: trusted `python`, `module`, `import_paths` (`PYTHONPATH`), `generation_hold`.
  The runner holds its generation independently too.
- `dispatch` launches through the Runtime trampoline like an executor (parent-death SIGKILL,
  no_new_privs, OOM order, own process group) with the sealed environment and no GPU, one
  inherited socket (`--execution-fd`), null stdin, and stdout/stderr in `logs/<id>.*.log`. A machine
  that dies takes its runners with it.
- Scheduler hooks: `ready`, `active`, `nonterminal` (partial indexes). `activity_epoch()` +
  `wait_activity(epoch, wait)` is a process-local wake. The wait bounds observation only.
- `waiting_reason` excludes a queued record from `ready`. `wait_for_environment` sets or clears it.

## Lifecycle

1. `claim`: `queued` -> `starting`, `attempt += 1`.
2. Spawn, then record `(pid, boot_id, start_ticks)`.
3. `Ready` must carry the child's real pid and `runtime.author-cpu/1`.
4. Commit `running`, then send `Invoke`. A cancel recorded earlier ends the attempt canceled with
   no authored code run.
5. Read `Progress` until a terminal event, wait for exit, take output custody, commit one outcome.

- Only explicit `cancel` cancels. Disconnect, EOF and submission closure never do. No time-based
  kill: after a cancel the runner's frames are the meter, and a runner that stops moving for
  longer than eight times its longest gap (at least 30 s) is killed and the run is `canceled`.
- After the terminal event the channel closes; a runner that then neither exits nor makes
  measurable CPU/IO progress is killed before it is reaped.
- A runner `Canceled` without a durable `cancel_actor` becomes `failed`.
- A launch failure or an exit before `Ready` is `failed` with the reason (and stderr tail). Only a
  transient OS shortage (EAGAIN, ENOMEM, EMFILE, ENFILE) returns the record to `queued` with a
  `waiting_reason`.

`reconcile` (records not supervised by this process):
- birth alive, or liveness unknown: stays nonterminal;
- `starting` with ended or no recorded birth: back to `queued` with a reason;
- `running` with ended birth: `failed`, never replayed.

It returns at most 1,024 nonterminal records.

## Progress and revisions

- One in-memory snapshot per supervised execution. Detail is capped at 2 KiB.
  Non-increasing `completed_units` is dropped. Reads overlay it without writes. Cancel and settlement persist it.
- Durable transitions jump past `revision_ceiling`. `running` reserves 2^32 revisions for volatile
  progress and renews on exhaustion. A new owner reports the old ceiling, so cursors never regress.

## Output log (products)

- `run_products(execution, sequence, at_ms, product)` holds each published `RunProduct`.
  Appending is a durable transition: the product's event sequence is that revision, so products
  interleave with state and keep their sequence in the terminal page.
- Only a starting/running execution appends. The terminal page lists the log's products, then
  the result's products the log does not already show, then the outcome.
- The CPU runner does not publish mid-run yet: its publishes appear with the result.

## Runner wire

4-byte big-endian length + JSON, max 1 MiB. Unknown kinds are ignored. No version gate.

- Ready: `pid`, `capabilities`
- Invoke: `execution_id`, flattened `Invocation`, absolute `output_root`
- Cancel / Canceled: `execution_id`
- Progress: `execution_id`, `completed_units`, `detail`
- Result: `execution_id`, `value`, `artifacts` (relative paths), `asset_bindings`
- Failed: `execution_id`, `code`, `detail`

An event for another execution fails the attempt. The runner is
`python/cozy_machine_client/runner.py`. It imports nothing authored before `Invoke`.

## Output custody

After `Result` and a zero exit, each declared file is opened with `openat(O_NOFOLLOW)` per
component (no absolute, `..` or symlink), checked as a regular file via `O_PATH`, copied to
`results/<id>/`, SHA-256 hashed, fsynced, made 0400 and renamed. Every `asset_binding` must name
one held artifact of equal length.

Reclamation: the run's spool (`staging/<id>`, the executor's own output directory) is removed
before the run's terminal state is written. `results/<id>` is removed when the client releases
retention and the store holds the public products. The caches manage themselves (`reclaim.rs`, at
start and every 10 minutes; no purge verb): crash-leftover spools of settled runs go at once;
logs, collected results and generations go after a 7-day TTL (a generation's last use is its
`.hold` mtime) or, least recently used first, under the host's storage pressure (below 1/10 of
the filesystem free, until above 1/5: TensorFS's thresholds). Never evicted: uncollected results,
journal rows, a generation any run or executor holds (its `.hold` lock) or an installation or
configured package names. Under pressure the persistent kernel store goes too, least recently
used first (one compiled kernel, or one generation's torch kernels), before generations and never
while an executor of that identity is alive. Below the reserve, max(1 GiB, 1/50), an executor's
compiled kernels go to its run-scoped JIT directory instead of the persistent store.

Triage: every failed CPU run, and every GPU run whose executor ended, failed to start or was killed
without writing its own terminal, keeps a bundle (`triage.rs`) before it settles: the reason, which
carries any kill measurement, the process id and the end of its stderr. `Read{triage}` serves it.

Journal: `machine_metadata.journal_format` records the layout (1). A newer journal still opens;
a row this machine cannot decode is skipped (logged) instead of failing every list, and a state
it does not know is `unknown`: listed, never dispatched, settled or overwritten.

Runners get the same scope as executors (a cgroup or a token; see device executor).

Not implemented: same-UID isolation (reads detect mutation), cleanup of orphan `*.pending` files.
