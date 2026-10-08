# Architecture

TensorD (`cozy-machine`) is the persistent Rust daemon that owns the API, execution journal,
node scheduling, memory budgets and executor supervision. TensorFS is an embedded library;
ordinary processes can share its store for reads and writes. Package code runs in Python
executors (cozy-runtime) inside each package environment. The machine never loads CUDA;
executors own their device contexts. NVML is read only on each GPU's sampler thread (`memory`).

TensorD makes allocation and eviction decisions across workloads. Runtime reports each rank's
needs and measured use and acts on its grants; the TensorFS weight plane moves and accounts for
bytes within those grants. TensorFS `Store::ensure_owned` retains the store claim through live
handles and read leases, coordinates GC pins and excludes outside collectors. Its
[owned-store contract](https://github.com/cozy-creator/tensorfs/blob/master/docs/owned-store.md)
describes collector compatibility. TensorD retains the node-specific sealed memfd cache, host
tiers, GPU-region custody and peer/process leases. [Recovery](RECOVERY.md) distinguishes
executor replacement, daemon restart and software activation guarantees.

Plan of record and workstream IDs (A–F):
`~/cozy_v2/outputs/cozy-machine-takeover-20261002/PLAN.md`.

## Process picture

```
cozy CLI ──TLS/gRPC (Cozy-Cap)──> api::server ─> machine_api::NativeBackend
                                                        │
machine.sock (weight peers) / admin.sock ─> main.rs ▼
                                          service::Service (dispatch policy)
                                           │ journal + supervision: execution::Engine / journal::Journal
                         CPU ──────────────┤
                          python -m cozy_machine_client.runner (one process per request)
                         GPU ──────────────┘
                          gpu_service::GpuPool ─> device_executor::DeviceExecutor
                            (one retained executor per plan, via the Runtime trampoline)
                            memory::GpuMemory: admission, process caps, eviction, floor
                            answers budget-cell, device-room, model-source and sealed-tier requests
```

A request is accepted durably (`Engine::submit_public`), then `Service::dispatch_ready` resolves
its immutable generation (`catalog`) and starts it: CPU work through the runner, GPU work through
the single retained executor slot in `GpuPool`. Progress, outputs and terminal state are written
by `Engine`; outputs are kept in native custody until the client acknowledges collection.

Tenancy: a pod has one owner, a package one publisher, and many users submit. A GPU plan (and so
its executor and its memory-policy tenant) is the package's generation, entrypoint, bound models
and group, never the submitter: every submitter of a construction is served by one executor, one
request at a time. Degree 2 holdings are per GPU and layout, shared by every executor on the pod.
What belongs to a submitter is in the journal and the API: installations, preparations, runs,
inputs, outputs, triage and controls are read by actor; a run's spool lives only for its call,
and a failure quotes only the stderr its own request wrote.

Local sockets, both 0600 and announced on start as `READY <machine.sock>` then
`ADMIN <admin.sock>`: `machine.sock` serves weight peers (hello, import, attach, release, stats);
`admin.sock` serves the owner (submit, executions, cancel, results, shutdown) and refuses any peer
that descends from the machine. Same-UID package code is still not sandboxed.

## Module map

| Module | Owns | Main types | Owner |
|---|---|---|---|
| `main.rs` | `serve`/`version` commands, weight-peer socket and owner-only admin socket, wiring | — | E (admin socket split), D2 (rental config) |
| `api/server.rs` | TLS listener: `cozy.machine.v1`, health, retired worker.v1 answers | `MachineIdentity`, `serve` | D1 |
| `api/auth.rs` | Admitted keys (a rental's Hub lease) and each open stream's authority | `Authority`, `Keys`, `StreamAuthority` | D1 |
| `api/identity.rs` | Persistent P-256 TLS identity, typed machine config, readiness secret | `MachineConfig`, `AuthorizedKeys`, `ReadinessSecret` | D2 |
| `api/domain.rs` | Current typed native execution/backend values; no RPC or version identity | Native queries, state, model and custody values | APIM |
| `archive.rs` | Private stored model/product/intake/terminal codecs, preserving existing byte tags | Stored records and codec functions | APIM |
| `api/backend.rs` | The one backend trait the server calls | `MachineBackend`, `InputTreeReceiver`, `Observation` | D1 |
| `api/workspaces.rs` | Resumable package uploads, scoped by owner key | `WorkspaceUploads`, `UploadSession`, `UploadedPackage` | D1 |
| `api/machine_v1.rs` | `cozy.machine.v1` (`proto/cozy/machine/v1/machine.proto`, `MACHINE-API.md`): Run, Control, Read under one `Cozy-Cap`; Write (D1) to come | `MachineV1` | G |
| `api/machine_status.rs` | Status: identity and receipt to anyone, the whole machine to a machine cap as it changes, `keepalive` | `status` | D2 |
| `api/machine_update.rs` | `Run kind: update`: a software cohort from versions or held wheel objects; its log is the update's state history | `run`, `owns` | D2 |
| `api/capability.rs` | `Cozy-Cap` grants (the Go agent's token) for run outputs and maintenance | `Grant`, `verify`, `mint` | G |
| `api/install.rs` | Materialize uploaded packages, run the uv installer helper | `InstallerConfig`, `PreparedGeneration` | D1 |
| `machine_api.rs` | `MachineBackend` implementation: submit, events, collect, list, inventory | `NativeBackend` | D1 |
| `products.rs` | Run output log: `Outputs.publish` custody and `product` events (SET/APPEND, composite parts) | `publish`, `retain`, `document` | G |
| `triage.rs` | One bounded triage bundle per failed attempt, named by its outcome | `TriageRef`, `Facts` | G |
| `hub.rs` | Delegated Hub access, the catalog reads it authorizes, and Hub writes under a machine-publication authorization (bearer renewed with the execution access) | `Grant`, `Catalog`, `Publishing` | D1 |
| `published.rs` | Package and model preparation: releases and models from the Hub, provider-source models via TensorFS `source_model`, held per release and resolution; downloads keep the serving set out of GC | `Publisher`, `Request`, `Prepared` | D1 |
| `adapter_views.rs` | Caller LoRA adapters as a zero-copy TensorFS derivation | — | D1 |
| `runs.rs` | Run sources and preparation inside a run: accepted at once, install/resolve/download as its progress, Hub token in memory only. A warm run with no entrypoint installs and fetches its choices; one with no code makes provider sources and uploads to its weights destination | `Runs`, `Spec`, `Source` | D1 |
| `weights.rs` | A job's weights: `weights_writer` sources and outputs over TensorFS's native channels; an adopted output kept in a local repository, recorded as a manifest product, published to the run's weights destination | `Weights`, `Grant` | D1 |
| `objects.rs` | Write: resumable content-addressed objects into the store, recorded per signer | `Objects`, `Writer` | D1 |
| `local_source.rs` | A run's local source: install written unpublished code once per manifest | `LocalSources`, `Manifest` | D1 |
| `machine/` | Launch grant, lifetime identity, readiness receipt, rental lifecycle, supervision, SSH, runtime update, the direct player endpoint (`player.rs`), the embedded Python client and installer helper (`client.rs`) | `Grant`, `Readiness`, `Lifecycle` | D2 |
| `native_inputs.rs` | Native input custody into TensorFS + journal | `SourceIntake`, `IntakeJournal` | D1 |
| `catalog.rs` | Immutable package environment generations and their holds | `Catalog`, `Generation`, `HeldGeneration` | D1 |
| `service.rs` | Sole dispatch policy (CPU parallelism, one GPU slot, startup GPU fences) | `Service` | B2 |
| `jobs.rs` | Jobs in deviceless executors, their child runs, pause/resume (root replay over finished children), scratch, checkpoint declarations | `Jobs` | G, L (pause/resume) |
| `execution.rs` | Acceptance, runner supervision, cancellation, progress coalescing, output custody, reconcile | `Engine`, `RunnerConfig` | E |
| `journal.rs` | SQLite journal: executions, installations, preparations, receipts, process births | `Journal`, `Execution`, `State`, `ProcessBirth` | E |
| `gpu_service.rs` | GPU pool: published package/model mapping, executor retention, request callbacks; GPU groups (a degree-K plan on the first K GPUs, one memory decision per GPU) and multi-slot plans | `GpuPool`, `GpuConfig`, `GpuPlan`, `PlanSlot`, `ModelGrant` | B2 (admission, grants), E (spawn/fencing), D1 (published mapping), I (groups, slots) |
| `memory/` | Per-GPU ledger and decisions (`policy`), NVML sampler thread (`nvml`), floor watchdog, per-host pinned budgets (`host`) | `GpuMemory`, `policy::Gpu`, `Step`, `Decision` | B2 |
| `device_executor.rs` | Typed control seam to the Runtime device executor, per-request output encoding | `DeviceExecutor`, `ExecutorConfig`, `DeviceCommand`, `Frame`, `Answer` | E |
| `launch_identity.rs` | Runtime trampoline command, the executor environment seal, optional UID/GID | `LaunchIdentity`, `Seal`, `trampoline` | E |
| `process.rs` | Exact process births, kill (scope and group), the progress meter and watch (leader plus process-group members), reaping (`reap_group`) | `Exact`, `Pace`, `Watching`, `Reaped` | E, I |
| `scope.rs` | Each executor's own scope (cgroup-v2, else an inherited token): kill, count, end; sweep of unclaimed and earlier runs' scopes | `Scope` | E |
| `reclaim.rs` | Self-managing caches: TTL and storage pressure for spools, logs, collected results, generations | `Disk`, `Swept`, `sweep` | E |
| `child_launcher.rs` | Pool-owned spawn thread (PDEATHSIG follows the creating thread) | `ChildLauncher` | E |
| `os.rs` | memfd, seals, peer credentials (`SO_PEERPIDFD`, `pidfd_open` fallback), pidfd exit | — | E |
| `owner.rs` | Weight peers, import, sealed memfd cache (LRU/TTL), leases per pidfd; store claim through TensorFS `ensure_owned` | `Owner` | B1 |
| `protocol.rs` | Private control-socket protocol (length-prefixed JSON + `SCM_RIGHTS`) | `Request`, `Command`, `Reply` | E |
| `host_tier.rs` | Degree 1 host tier: machine-filled sealed layouts, adopted read-only, sized by live headroom | `HostTier`, `HostGrant`, `TierLimit`, `HostTierFacts` | B1 |
| `host_memory.rs` | Live host headroom (cgroup v1/v2 path, `MemAvailable`) | `HostMemory` | B1 |
| `model_sources.rs` | Selected model byte grants (read-only descriptors) | `ModelSources`, `SelectedManifest`, `SourceGrant` | B1 |
| `model_source_driver.rs` | Answer one executor model-source request | `answer` | B1 |
| `resident_custody.rs` | Degree 2: executor-exported GPU regions kept as driver fds (no CUDA), leases, revocation | `ResidentCustody`, `HoldingKey`, `SharedRegion` | C |
| `boundary_json.rs` | Strict JSON parse (no duplicate keys) for boundary records | — | D1 |


## Python package `cozy_machine_client`

| Module | Role |
|---|---|
| `runner.py`, `runtime_bridge.py`, `execution_protocol.py`, `progress.py` | CPU runner: `Ready` before importing Runtime, then Runtime author `prepare`/`invoke` |
| `packages.py`, `captured_packages.py`, `package_records.py`, `runtime_describe.py` | Static description and uv generation install (`python -m cozy_machine_client.packages`) |
| `device_codec.py` | Output encoder, embedded into `device_executor.rs` with `include_str!` |
| `client.py`, `protocol.py`, `linux.py`, `execution_client.py`, `cpu_gate.py` | Private control-socket client and the CPU component gate |

## Development tools (not proof; see PLAN decision 5)

| Path | Use | Owner |
|---|---|---|
| `src/bin/device-pilot.rs` | Drive one stock executor from a JSON config on a rental | F |
| `src/bin/degree2-pilot.rs` | Degree 2 on a rental: stock executors share GPU weights through `ResidentCustody` (replacement, count-once, revoke, output equality) | C |
| `src/bin/model-source-check.rs`, `scripts/check-model-source-descriptor.py` | Check a selected snapshot's sources | B1 |
| `src/bin/install-capture.rs` | Run the installer path on a captured archive (used by Python tests) | D1 |
| `scripts/gate/` | Matched old-stack vs Rust-machine gate through ordinary `cozy run` | F |
| `scripts/benchmarks/permission_probe.py` | CPU executor permission probe | E |
| `scripts/service_cpu_gate.py` | CPU service end-to-end gate over the admin socket, including machine kill and restart | E |

## Contracts

Native storage grants belong to one durable run attempt. Pausing and resuming a run
creates a fresh grant; an earlier attempt cannot regain authority when the run becomes
active again. Closing a source/output channel fences that writer attempt but does not
abandon its transaction or release committed custody. TensorFS persists resumable parts
and committed receipts independently of channel acknowledgments.

Replaying or adopting a derived result requires retained native custody and a verified
model closure. Receipt metadata alone cannot recreate a disposed result. Adoption compares
the transaction, declaration and immutable manifest identity, then records the owner's
native facts; additional peer observations do not change authority. Independent consumers
retain their own roots, so releasing one consumer cannot make another's data collectible.
See [tracker #341](https://github.com/cozy-creator/tracker/issues/341).

Before native begin, the existing execution journal records only the run, output slot and
transaction association, atomically with current-attempt admission. Explicit cancellation
uses that association to fence and abandon unfinished work, or dispose a committed result
that still has only pending custody. Acknowledged native adoption and independent consumer
roots survive. Cleanup is idempotent and retried at startup/reclamation; pause, observer
disconnect and ordinary failure do not authorize abandonment. Older unbound transactions
are never guessed to belong to a canceled run.

TensorFS allocates writer epochs from its durable transaction fence through
`channel::Writer::begin_next`. Reopening after a restart or an old clock-derived high epoch
creates a new writer attempt of the same transaction; stale channel closure cannot fence
its replacement. Payload writes and publication remain concurrent; only local custody
metadata transitions are coordinated with cancellation.

- [Durable execution](DURABLE-EXECUTION.md): journal states, acceptance, cancellation, custody.
- [Recovery](RECOVERY.md): executor and daemon lifetimes, update publication and rollback.
- [Front door](FRONT-DOOR.md): the listener, TLS identity, install.
- [Package bridge](PACKAGE-BRIDGE.md): CPU runner and generation install.
- [Device executor](DEVICE-EXECUTOR.md): executor launch and control seam.
- [GPU service](GPU-SERVICE.md): GPU pool, config, launch identity.
- [Host tier](HOST-TIER.md), [model sources](MODEL-SOURCES.md): Degree 1.
- [Resident custody](RESIDENT-CUSTODY.md): Degree 2 foundation.
- [Matched gate](../scripts/gate/README.md): old stack vs Rust machine, ordinary CLI.

## Build and test

```sh
export CARGO_TARGET_DIR=~/cozy/.cargo-target/cozy-machine
L=~/cozy_v2/outputs/cozy-machine-takeover-20261002/locks
flock $L/rust-build.lock nice -n 19 cargo clippy --all-targets -j 2 -- -D warnings
flock $L/rust-build.lock nice -n 19 cargo test -j 2 -- --test-threads=2
nice -n 19 uv run --locked --extra test pytest -q
```

The test extra installs cozy-runtime; tests fail without it. Ignored Rust tests need installed SDK
generations (paths or `COZY_MACHINE_CPU_TEST_PYTHON`); run them with `--ignored` on purpose.
