# Cached GPU application service checkpoint

The authenticated native machine API can submit a published callable using a root-configured,
verified cached package/model mapping. It uses the same Engine acceptance, journal, scheduler,
actor authority, progress and output custody as CPU applications. There is one retained device
executor slot, world one and one declared model slot; class/slot/component use are derived from
the actual SDK static interface. No SDXL/Anima source or business planner is copied into Rust.
The root-sealed config explicitly maps qualified package to installed distribution and immutable
Catalog generation. Missing published mapping, uncached sources and unsupported operations return
operation errors; no SDK version floor is applied.

`serve --gpu-config FILE` adds typed GpuConfig to the existing retained TLS service. `source_mode`
is `auto` (default), `legacy` or `descriptors`. Auto negotiates `model_sources.descriptors/1` and
falls back to the existing SDK/TensorFS store path. Legacy comparison still allows SDK metadata
writers and is not sole-writer/complete Degree1 proof. Explicit descriptors mode fails only that
operation when unavailable; it never holds an older SDK in an endless queued wait.

Host-owner configuration is optional and experimental. When configured, the native machine
service authorizes the selected immutable header/components and exact actor/plan/process birth
before accepting the sealed SDK partition FD. Grants survive observer/socket loss until actual
recipient exit. Live revocation, disk-backed misses, initial pinned budget and refined-layout
reuse are not qualified. Without it, the existing SDK host tier owns its allocations.

Executors launch through the installed SDK trampoline with expected parent, parent-death,
no_new_privs and OOM score. Current scope is inherited from the launched service, without a new
executor cgroup/UID/PGID. Birth is committed before Start imports authored code. Changing models
requires the old context process to exit before another is spawned. On restart, bounded journal
pages include completed retained GPU request births: live/unknown prior-owner processes fence
all new GPU dispatch until exact exit, while CPU dispatch remains available. No timer kills,
telemetry estimate or terminal event releases that reservation. Running work is not adopted or
reconnected across upgrades; an authorized started attempt is never silently re-executed.

Results are encoded with the existing SDK codec and then copied/hashed into Engine custody,
projected with actual schemas into native trees/outcomes. Encoding currently starts a Python
helper for every request. Stage acknowledgements currently supply a static configured weight
budget, not adaptive full-memory grants. A weight ceiling does not bound context, allocator,
activation, workspace or display memory; this is not safe display-GPU admission.

CPU evidence: 35 focused process/store/auth/native tests passed, including a restart with 258
completed requests sharing one live retained birth (one fence until actual exit), plus all-target
clippy. An actual SDK100 trampoline CPU probe established parent-death SIGKILL, no_new_privs,
OOMscore1000 and equal inherited cgroup with no CUDA/NVML mappings. These are component gates,
not inference, GPU memory or ordinary CLI GPU qualification.

Typed comparison templates and concrete launch/cache/consumer prerequisites are in
`/home/fidika/cozy_v2/outputs/cozy-machine-continued-20261002/gpu-public-service/`.
The next decision gate is a matched ordinary Creator CLI comparison on the owned headless rental:
same client, SDK/TFS, dependency roster, weights, request, cgroups, allocator and cache states.
Source mode and host ownership remain separate axes. Full Degree1 still needs dynamic GPU/host
admission, context room and measured progress escalation, initial pinned limits, reclaim/cache
policy, low-memory recovery, boot refresh, callee/input/deadline and Hub/rental/media parity.
