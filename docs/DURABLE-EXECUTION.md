# Durable execution

`journal.rs` is the machine's single SQLite acceptance and transition authority. `execution.rs`
(`Engine`) supervises the pool's attempts over it. Neither schedules, installs or authenticates:
the owner (`service.rs`) admits work and resolves an immutable environment before dispatch.

[Process replacement and recovery](RECOVERY.md) distinguishes retained executor weights from
durable daemon state, and describes the software activation transaction.

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
- `AdmissionError` (inside `io::Error`): `WorkspaceMismatch`, `BindingConflict`.
- Credentials and grant secrets never enter the context, journal or input. A capture's
  `record_owner` label never selects authority.

## Engine

```rust
let engine = Engine::open(state_root)?;
let record = engine.submit(key, invocation)?;   // committed receipt
engine.cancel(&record.id, actor)?;              // durable actor first, then cooperative Cancel
let open = engine.reconcile()?;                 // no adoption, no replay
let file = engine.open_result(&record.id, 0)?;  // re-verifies digest and length
```

- Every run executes on a pool executor (`gpu_service.rs`, [GPU service](GPU-SERVICE.md)),
  deviceless ones included. `dispatch_managed` claims the record and runs the pool's attempt
  on its own thread; `register_managed` records the executor birth and its stop.
- Scheduler hooks: `ready`, `active`, `nonterminal` (partial indexes). `activity_epoch()` +
  `wait_activity(epoch, wait)` is a process-local wake. The wait bounds observation only.
- `waiting_reason` excludes a queued record from `ready`. `wait_for_environment` sets or clears it.

## Lifecycle

1. `claim`: `queued` -> `starting`, `attempt += 1`.
2. The pool picks or starts an executor and records its birth `(pid, boot_id, start_ticks)`.
3. Commit `running`, then send `Invoke`. A cancel recorded earlier ends the attempt canceled with
   no authored code run.
4. Progress frames until the executor's terminal; output custody; one outcome.

- Only explicit `cancel` cancels. Disconnect, EOF and submission closure never do. No time-based
  kill: the executor's cooperative cancel is the meter (see device executor).
- An attempt that stops without a durable `cancel_actor` is `failed`.
- A launch failure or an executor that ends mid-call is `failed` with the reason (and stderr tail).
- Every run ends completed or failed: a queued run this machine cannot start (its environment not
  held, or GPU execution not configured) fails with that reason at once. Nothing is parked to
  wait, and nothing returns to the queue to fail again.

`reconcile` (records not supervised by this process):
- birth alive, or liveness unknown: stays nonterminal;
- `starting` with ended or no recorded birth: back to `queued` with a reason;
- `running` with ended birth: `canceled` if a cancel was journaled, else `failed`, never replayed.

It returns at most 1,024 nonterminal records.

## Progress and revisions

- One in-memory snapshot per supervised execution. Detail is capped at 2 KiB.
  `completed_units` is the attempt's monotone position (Runtime `Telemetry.advance`);
  non-increasing values are dropped. Reads overlay it without writes. Cancel and settlement
  persist it.
- Durable transitions jump past `revision_ceiling`. `running` reserves 2^32 revisions for volatile
  progress and renews on exhaustion. A new owner reports the old ceiling, so cursors never regress.

## Output log (products)

- `run_products(execution, sequence, at_ms, product)` holds each published `RunProduct`.
  Appending is a durable transition: the product's event sequence is that revision, so products
  interleave with state and keep their sequence in the terminal page.
- Only a starting/running execution appends. The terminal page lists the log's products, then
  the result's products the log does not already show, then the outcome.

## Output custody

After the executor's result, each declared file is opened with `openat(O_NOFOLLOW)` per
component (no absolute, `..` or symlink), checked as a regular file via `O_PATH`, copied to
`results/<id>/`, SHA-256 hashed, fsynced, made 0400 and renamed. Every `asset_binding` must name
one held artifact of equal length.

Reclamation: the run's spool (`staging/<id>`, the executor's own output directory) is removed
before the run's terminal state is written. `results/<id>` is removed when the client releases
retention and the store holds the public products. The caches manage themselves (`reclaim.rs`, at
start and every 10 minutes; no purge verb): crash-leftover spools of settled runs go at once;
logs, collected results, generations (last use: `.hold` mtime), compiled kernels (last read or
write), cached models and the uv cache's unpacked wheels no environment links (last use: the
last link or unlink, ctime) go 7 days after their last use. A disk is low when no more than
TensorFS's reserve is free (2% of the filesystem within 1 to 10 GiB; `ensure::Disk`). A low disk
drops a plan that lifts it back above the reserve, or only the store's garbage when nothing
covers that: collected results, settled logs, memoized stages, unpacked wheels no environment
links, generations idle for a sweep period, then the fewest least recently used cached models,
compiled kernels last and only if they alone cover what is still missing. Only bytes an unlink frees count: single-link files this
process does not hold open, on the measured filesystem. Never evicted: uncollected results,
journal rows, a generation any run or executor holds (its `.hold` lock) or an installation,
unfinished run or configured package names, a model a live executor, unfinished run or
preparation names, a kernel namespace a live executor or kernel boot holds (`kernels/.u<uid>.hold`,
shared; the sweep takes it exclusively), anything in the uv cache while an install holds its
`.lock` (uv per command, the publisher per environment build). An image's seeded uv cache is never
touched: environments symlink into it. On a low disk an executor's compiled kernels go to its
run-scoped JIT directory instead of the persistent store, and a download the disk cannot fit
fails as `machine_disk_full`.

Write objects: each verified object gets a TensorFS object root (`roots/objects/<sha>.json`)
before its Write is acknowledged, and a run's references (`run_objects`: its inputs, tree members
and local source; a parent's adopted child results) commit with its acceptance. `Objects::sweep`
releases a root once no unfinished run (any state but completed, failed or canceled) names it and
nothing used it for the 7-day TTL, or for 10 minutes while the store's disk is low; the store's GC
then takes the bytes. A root written just before a crash ages from its own mtime.

Triage: every run whose executor ended, failed to start or was killed
without writing its own terminal, keeps a bundle (`triage.rs`) before it settles: the reason, which
carries any kill measurement, the process id and the end of its stderr. `Read{triage}` serves it.

Journal: `machine_metadata.journal_format` records the layout (1). A newer journal still opens;
a row this machine cannot decode is skipped (logged) instead of failing every list, and a state
it does not know is `unknown`: listed, never dispatched, settled or overwritten.

Not implemented: same-UID isolation (reads detect mutation), cleanup of orphan `*.pending` files.

## Cancel and submit again

There is no pause. A canceled or failed run is submitted again as a new run. A memoized call's
success is held when that call completes, whatever its root then does, so the new run's finished
calls are answered from the memo and only the rest run. A derivation (weights output) is never
memoized; it replays the committed transaction of the same code and declaration instead.
