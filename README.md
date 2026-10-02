# Cozy machine

Experimental standalone Rust machine service: worker orchestration, machine-agent duties and
TensorFS belong here. Python cozy-runtime remains the executor/author SDK/model integration layer.
Development uses owned worktrees and fixtures; the existing released stack and default
running service remain in place. Consumer adapters are separate draft PRs.

This is an experimental implementation with a working CPU service and a separate GPU pilot.
The [program](https://github.com/cozy-creator/cozy-machine/issues/1) tracks the remaining slices.
After the prototype, compare the old stack before deciding whether a full rewrite earns its cost.

## Working CPU component

- Rust links released TensorFS core v0.3.87; it is the only store writer through this service.
- Typed length-prefixed JSON and SCM_RIGHTS use the deployed framing. Version strings do not gate
  peers; implemented capabilities are negotiated.
- Immutable weight memfds are shared by readers; idle LRU/TTL entries close as whole allocations.
  These sealed objects cannot be hole-punched. Disk-backed verified descriptors are the fallback
  when cache space is unavailable, including a zero-byte cache budget.
- Host cache charge comes from actual allocated backing, not logical tensor length.
- Process identity uses Linux SO_PEERPIDFD. Socket EOF alone does not release a live process's leases.
- Python clients map the verified fd and run real NumPy classifier inference without TensorFS.
  Live array views prevent premature unmap/release.

## Build and test

A Rust toolchain and Python 3.11+ are needed. Cargo requires read access to the private upstream
TensorFS repository. The experimental host capability currently needs Linux SO_PEERPIDFD support
(Linux 6.5+); older kernels do not get a weaker fabricated process identity.

```sh
cargo build --locked
uv sync --locked --extra test
cargo test --locked -- --test-threads=1
uv run --locked --extra test pytest -q
uv run --locked --extra test cozy-machine-cpu-gate \
  --machine target/debug/cozy-machine --output cpu-evidence --crashes 10
```

The last command launches its own private service and Python consumers, trains a small real
handwritten-digit classifier, imports its weights into TensorFS, and verifies reference predictions
across sharing, ten executor deaths, survivor/replacement reuse, LRU eviction and disk fallback.
It also checks version/additive-field handling and that the Rust process has no CUDA/NVML mappings
or GPU device descriptors. All generated evidence is under the requested output directory.

For a manually started private service:

```sh
target/debug/cozy-machine serve --state /path/to/private-state --host-bytes 16777216
target/debug/cozy-machine version --json
```

The gate uses a short ephemeral socket runtime directory because Unix socket paths are bounded.
Repositories, fixture sources and result artifacts are durable. No COZY_HOME override or owner
daemon is used.

## Integrated execution and GPU evidence

The integrated service now has authenticated pinned-leaf TLS/gRPC, static package description,
captured installation, a durable execution journal, explicit cancellation and native output
custody. A task-built ordinary Creator CLI has run the real sklearn application repeatedly
through this service using its normal default home and an explicit endpoint file. That adapter
is opt-in; it does not install the service as the user's current local machine or rental.

The separate Rust stock-executor pilot completed real SDXL on a headless RunPod A40. The normal
descriptor-backed model path uses an empty executor store while retaining the SDK's TensorFS
reader/GPU mechanisms. Across 48 clean matched adapter requests, 24 image pairs were byte-identical
and warm output medians were approximately 4.8 s in both arms. This is not a matched full-stack
speedup or proof of complete host/GPU ownership. See [GPU evidence](docs/GPU-BENCHMARK.md).

## Boundaries

The qualified public lane is a CPU application without asset inputs; the GPU pilot is separate
from that public API. Hub provisioning, browser media, child calls, jobs, full public-service
crash/OS-supervisor boundaries and old install/update consumers remain open. The service is not
a complete machine replacement. Degree 1 memory ownership, real low-memory diffusion recovery,
shared GPU backing, machine-issued copies and NCCL are unqualified. CPU code loads no CUDA/NVML.

Descriptors must not be forwarded or inherited by unregistered processes in the shared-weight
component. The connecting process is the tracked recipient. Recipient/descendant registration
still needs integration before this becomes an arbitrary-package shared-weight path; same-UID
code is not a hostile-code sandbox. Seals prove
immutability, not durable output custody. The cache budget does not account for all process RSS or
OS page cache. Unsealed region/DMA reclaim and concurrent TensorFS GC need their own integration.

## Design and implementation issues

The [selected contract](https://github.com/cozy-creator/tracker/blob/785c808efaaae8fee8243152e4db948ea05169a7/design/rust-machine.md)
and [A/B discussion](https://github.com/cozy-creator/tracker/blob/5ffd1431c/design/rust-machine-revision.md)
inform this repository. A has lower migration cost; the new repo is a proof-first B experiment.
We borrow A's conservative scope and leave its fixes with the existing owners.

| Issue | Slice |
|---|---|
| [#1](https://github.com/cozy-creator/cozy-machine/issues/1) | Program and old-stack comparison decision gate |
| [#2](https://github.com/cozy-creator/cozy-machine/issues/2) | CPU TensorFS owner and real Python shared-weight inference |
| [#3](https://github.com/cozy-creator/cozy-machine/issues/3) | Durable authenticated CPU package execution and legacy adapters |
| [#4](https://github.com/cozy-creator/cozy-machine/issues/4) | Degree 1, bounded host/disk staging and consumer/delivery gates |
| [#5](https://github.com/cozy-creator/cozy-machine/issues/5) | Shared GPU residency after measured need and failure-isolation proof |
| [#6](https://github.com/cozy-creator/cozy-machine/issues/6) | Deferred machine-copy research; never a first-release prerequisite |

The core stays CUDA/NVML-free for the first release. Future GPU ownership must explicitly choose
between qualified in-process actors and an isolated helper's extra-process cost; native driver
failure must not silently wedge API/journal/release. Resident-only sharing and executor-issued
copies precede any machine copier. Browser ICE-TCP/DTLS and old installer/update compatibility
need early real-consumer spikes before promising full control-plane replacement.

Comparison gates must use identical models/request semantics, fresh requests and matched
CPU/RAM/VRAM/cache/thermal conditions. Compare stopped-machine submit→saved output, hot inference,
model switches, physical memory and crash recovery. Tiny classifier attachment timings are not
ComfyUI or old-stack performance evidence. No automatic migration or release is enabled.
