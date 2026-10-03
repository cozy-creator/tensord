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
- `dispatch` spawns directly (no shell) with one inherited socket (`--execution-fd`), null stdin,
  and stdout/stderr files in staging.
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

- Only explicit `cancel` cancels. Disconnect, EOF and submission closure never do. No time-based kill.
- A runner `Canceled` without a durable `cancel_actor` becomes `failed`.
- Launch failure returns the record to `queued` with a `waiting_reason`. It stays there until the
  owner clears the reason after a changed condition. A duplicate submission is not one.

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

Not implemented: same-UID isolation (reads detect mutation), descendant containment,
cleanup of orphan `*.pending` files, bounded log policy.
