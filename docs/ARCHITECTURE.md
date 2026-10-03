# Architecture

`cozy-machine` is the machine: one Rust process that owns the API, the execution journal,
scheduling, the TensorFS store (sole writer) and executor supervision. Package code runs in
Python executors (cozy-runtime) inside each package environment. The machine never loads CUDA;
executors own their device contexts. NVML is read only on each GPU's sampler thread (`memory`).

Plan of record and workstream IDs (A–F):
`~/cozy_v2/outputs/cozy-machine-takeover-20261002/PLAN.md`.

## Process picture

```
cozy CLI ──TLS/gRPC (ClaimProof)──> api::server ─> machine_api::NativeBackend
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

Local sockets, both 0600 and announced on start as `READY <machine.sock>` then
`ADMIN <admin.sock>`: `machine.sock` serves weight peers (hello, import, attach, release, stats);
`admin.sock` serves the owner (submit, executions, cancel, results, shutdown) and refuses any peer
that descends from the machine. Same-UID package code is still not sandboxed.

## Module map

| Module | Owns | Main types | Owner |
|---|---|---|---|
| `main.rs` | `serve`/`version` commands, weight-peer socket and owner-only admin socket, wiring | — | E (admin socket split), D2 (rental config) |
| `api/server.rs` | TLS/gRPC front door (worker.proto from `vendor/worker-protocol`) | `MachineIdentity`, `serve` | D1 |
| `api/auth.rs` | ClaimProof/1 verification per request | `Authority`, `VerifiedActor` | D1 |
| `api/identity.rs` | Persistent P-256 TLS identity, typed machine config, readiness secret | `MachineConfig`, `AuthorizedKeys`, `ReadinessSecret` | D2 |
| `api/backend.rs` | The one backend trait the server calls | `MachineBackend`, `InputTreeReceiver`, `Observation` | D1 |
| `api/workspaces.rs` | Resumable package uploads, scoped by owner key | `WorkspaceUploads`, `UploadSession`, `UploadedPackage` | D1 |
| `api/machine_v1.rs` | `cozy.machine.v1` (`proto/cozy/machine/v1/machine.proto`, `MACHINE-API.md`): Run, Control, Read under one `Cozy-Cap`; Status (D2) and Write (D1) to come | `MachineV1` | G |
| `api/capability.rs` | `Cozy-Cap` grants (the Go agent's token) for run outputs and maintenance | `Grant`, `verify`, `mint` | G |
| `api/install.rs` | Materialize uploaded packages, run the uv installer helper | `InstallerConfig`, `PreparedGeneration` | D1 |
| `machine_api.rs` | `MachineBackend` implementation: submit, events, collect, list, inventory | `NativeBackend` | D1 |
| `products.rs` | Run output log: `Outputs.publish` custody and `product` events (SET/APPEND, composite parts) | `publish`, `retain`, `document` | G |
| `triage.rs` | One bounded triage bundle per failed attempt, named by its outcome | `TriageRef`, `Facts` | G |
| `hub.rs` | Delegated Hub access and the catalog reads it authorizes | `Grant`, `Catalog` | D1 |
| `published.rs` | Published package/model preparation from the Hub, held per release and resolution | `Publisher`, `Request`, `Prepared` | D1 |
| `adapter_views.rs` | Caller LoRA adapters as a zero-copy TensorFS derivation | — | D1 |
| `runs.rs` | Run sources and preparation inside a run: accepted at once, install/resolve/download as its progress, Hub token in memory only | `Runs`, `Spec`, `Source` | D1 |
| `objects.rs` | Write: resumable content-addressed objects into the store, recorded per signer | `Objects`, `Writer` | D1 |
| `local_source.rs` | A run's local source: install written unpublished code once per manifest | `LocalSources`, `Manifest` | D1 |
| `machine/` | Launch grant, lifetime identity, readiness receipt, rental lifecycle, supervision, SSH, runtime update | `Grant`, `Readiness`, `Lifecycle` | D2 |
| `native_inputs.rs` | Native input custody into TensorFS + journal | `SourceIntake`, `IntakeJournal` | D1 |
| `catalog.rs` | Immutable package environment generations and their holds | `Catalog`, `Generation`, `HeldGeneration` | D1 |
| `service.rs` | Sole dispatch policy (CPU parallelism, one GPU slot, startup GPU fences) | `Service` | B2 |
| `execution.rs` | Acceptance, runner supervision, cancellation, progress coalescing, output custody, reconcile | `Engine`, `RunnerConfig` | E |
| `journal.rs` | SQLite journal: executions, installations, preparations, receipts, process births | `Journal`, `Execution`, `State`, `ProcessBirth` | E |
| `gpu_service.rs` | GPU pool: published package/model mapping, executor retention, request callbacks; GPU groups (a degree-K plan on the first K GPUs, one memory decision per GPU) and multi-slot plans | `GpuPool`, `GpuConfig`, `GpuPlan`, `PlanSlot`, `ModelGrant` | B2 (admission, grants), E (spawn/fencing), D1 (published mapping), I (groups, slots) |
| `memory/` | Per-GPU ledger and decisions (`policy`), NVML sampler thread (`nvml`), floor watchdog, per-host pinned budgets (`host`) | `GpuMemory`, `policy::Gpu`, `Step`, `Decision` | B2 |
| `device_executor.rs` | Typed control seam to the Runtime device executor, per-request output encoding | `DeviceExecutor`, `ExecutorConfig`, `DeviceCommand`, `Frame`, `Answer` | E |
| `launch_identity.rs` | Runtime trampoline command, the executor environment seal, optional UID/GID | `LaunchIdentity`, `Seal`, `trampoline` | E |
| `process.rs` | Exact process births, kill (cgroup or group), the progress meter and watch (leader plus process-group members), reaping (`reap_group`) | `Exact`, `Pace`, `Watching`, `Reaped` | E, I |
| `cgroup.rs` | Each executor's own cgroup-v2 scope: join, kill, count, remove; restart sweep | `CgroupScope` | E |
| `reclaim.rs` | Self-managing caches: TTL and storage pressure for spools, logs, collected results, generations | `Disk`, `Swept`, `sweep` | E |
| `child_launcher.rs` | Pool-owned spawn thread (PDEATHSIG follows the creating thread) | `ChildLauncher` | E |
| `os.rs` | memfd, seals, peer credentials (`SO_PEERPIDFD`, `pidfd_open` fallback), pidfd exit | — | E |
| `owner.rs` | TensorFS store owner: import, sealed memfd cache (LRU/TTL), leases per pidfd | `Owner` | B1 |
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
| `src/bin/front-door.rs`, `tests/*_client`, `tests/creator_*` | Isolated front-door fixture and Go consumer gates | D1 |
| `scripts/gate/` | Matched old-stack vs Rust-machine gate through ordinary `cozy run` | F |
| `scripts/benchmarks/permission_probe.py` | CPU executor permission probe | E |
| `scripts/service_cpu_gate.py` | CPU service end-to-end gate over the admin socket, including machine kill and restart | E |

## Contracts

- [Durable execution](DURABLE-EXECUTION.md): journal states, acceptance, cancellation, custody.
- [Front door](FRONT-DOOR.md): TLS identity, ClaimProof, backend hooks.
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
