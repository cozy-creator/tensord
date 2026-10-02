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
private control socket (owner protocol) ─> main.rs      ▼
                                          service::Service (dispatch policy)
                                           │ journal + supervision: execution::Engine / journal::Journal
                         CPU ──────────────┤
                          python -m cozy_machine_client.runner (one process per request)
                         GPU ──────────────┘
                          gpu_service::GpuPool ─> device_executor::DeviceExecutor
                            (one retained executor per plan, via the Runtime trampoline)
                            memory::GpuMemory: admission, process caps, eviction, floor
                            answers budget-cell, device-room, model-source and host-tier requests
```

A request is accepted durably (`Engine::submit_public`), then `Service::dispatch_ready` resolves
its immutable generation (`catalog`) and starts it: CPU work through the runner, GPU work through
the single retained executor slot in `GpuPool`. Progress, outputs and terminal state are written
by `Engine`; outputs are kept in native custody until the client acknowledges collection.

## Module map

| Module | Owns | Main types | Owner |
|---|---|---|---|
| `main.rs` | `serve`/`version` commands, private control socket, wiring | — | E (admin socket split), D2 (rental config) |
| `api/server.rs` | TLS/gRPC front door (worker.proto from `vendor/worker-protocol`) | `MachineIdentity`, `serve` | D1 |
| `api/auth.rs` | ClaimProof/1 verification per request | `Authority`, `VerifiedActor` | D1 |
| `api/identity.rs` | Persistent P-256 TLS identity, typed machine config, readiness secret | `MachineConfig`, `AuthorizedKeys`, `ReadinessSecret` | D2 |
| `api/backend.rs` | The one backend trait the server calls | `MachineBackend`, `InputTreeReceiver`, `Observation` | D1 |
| `api/workspaces.rs` | Resumable package uploads, scoped by owner key | `WorkspaceUploads`, `UploadSession`, `UploadedPackage` | D1 |
| `api/install.rs` | Materialize uploaded packages, run the uv installer helper | `InstallerConfig`, `PreparedGeneration` | D1 |
| `machine_api.rs` | `MachineBackend` implementation: submit, events, collect, list, inventory | `NativeBackend` | D1 |
| `native_inputs.rs` | Native input custody into TensorFS + journal | `SourceIntake`, `IntakeJournal` | D1 |
| `catalog.rs` | Immutable package environment generations and their holds | `Catalog`, `Generation`, `HeldGeneration` | D1 |
| `service.rs` | Sole dispatch policy (CPU parallelism, one GPU slot, startup GPU fences) | `Service` | B2 |
| `execution.rs` | Acceptance, runner supervision, cancellation, progress coalescing, output custody, reconcile | `Engine`, `RunnerConfig` | E |
| `journal.rs` | SQLite journal: executions, installations, preparations, receipts, process births | `Journal`, `Execution`, `State`, `ProcessBirth` | E |
| `gpu_service.rs` | GPU pool: published package/model mapping, executor retention, request callbacks | `GpuPool`, `GpuConfig`, `GpuPlan`, `ModelGrant` | B2 (admission, grants), E (spawn/fencing), D1 (published mapping) |
| `memory/` | Per-GPU ledger and decisions (`policy`), NVML sampler thread (`nvml`), floor watchdog | `GpuMemory`, `policy::Gpu`, `Step`, `Decision` | B2 |
| `device_executor.rs` | Typed control seam to the Runtime device executor, per-request output encoding | `DeviceExecutor`, `ExecutorConfig`, `DeviceCommand`, `Frame`, `Answer` | E |
| `launch_identity.rs` | Runtime trampoline command, optional sealed UID/GID | `LaunchIdentity`, `trampoline` | E |
| `child_launcher.rs` | Pool-owned spawn thread (PDEATHSIG follows the creating thread) | `ChildLauncher` | E |
| `os.rs` | memfd, seals, `SO_PEERPIDFD`, pidfd exit | — | E |
| `owner.rs` | TensorFS store owner: import, sealed memfd cache (LRU/TTL), leases per pidfd | `Owner` | B1 |
| `protocol.rs` | Private control-socket protocol (length-prefixed JSON + `SCM_RIGHTS`) | `Request`, `Command`, `Reply` | E |
| `shared_host_plane.rs` | Degree 1 host tier: machine-filled sealed host layouts shared across executors | `SharedHostPlane`, `HostScope`, `HostKey`, `HostTierPlan` | B1 |
| `model_sources.rs` | Selected model byte grants (read-only descriptors) | `ModelSources`, `SelectedManifest`, `SourceGrant` | B1 |
| `model_source_driver.rs` | Answer one executor model-source request | `answer` | B1 |
| `resident_custody.rs` | Degree 2: executor-exported GPU regions kept as driver fds (no CUDA), leases, revocation | `ResidentCustody`, `HoldingKey`, `SharedRegion` | C |
| `boundary_json.rs` | Strict JSON parse (no duplicate keys) for boundary records | — | D1 |

The host ledger (pinned tier, RSS/PSS, cgroup headroom) is not in `memory/` yet (B1, B2).

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
| `src/bin/host-plane-cpu-owner.rs` | One-shot CPU host-tier/model-source server for SDK adoption tests | B1 |
| `src/bin/model-source-check.rs`, `scripts/check-model-source-descriptor.py` | Check a selected snapshot's sources | B1 |
| `src/bin/install-capture.rs` | Run the installer path on a captured archive (used by Python tests) | D1 |
| `src/bin/front-door.rs`, `tests/*_client`, `tests/creator_*` | Isolated front-door fixture and Go consumer gates | D1 |
| `scripts/gate/` | Matched old-stack vs Rust-machine gate through ordinary `cozy run` | F |
| `scripts/benchmarks/permission_probe.py` | CPU executor permission probe | E |
| `scripts/service_cpu_gate.py` | CPU service end-to-end gate over the control socket | E |

## Contracts

- [Durable execution](DURABLE-EXECUTION.md): journal states, acceptance, cancellation, custody.
- [Front door](FRONT-DOOR.md): TLS identity, ClaimProof, backend hooks.
- [Package bridge](PACKAGE-BRIDGE.md): CPU runner and generation install.
- [Device executor](DEVICE-EXECUTOR.md): executor launch and control seam.
- [GPU service](GPU-SERVICE.md): GPU pool, config, launch identity.
- [Shared host plane](SHARED-HOST-PLANE.md), [model sources](MODEL-SOURCES.md): Degree 1.
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
