# Existing device executor bridge

- Owner: `/root/durable_cpu_engine` (Codex sub-agent)
- Purpose: issue #4 thin Rust adapter to the retained published Python Runtime device Executor
- Branch: `feat/4-device-executor-bridge-20261002`
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/4-device-executor-bridge-20261002`
- Fetched origin base: `9ef3799b6a96e719f29f6a94ff9e9eff093153fe`
- Cumulative engine source checkpoint: `24082f9f2b4b59c2dc17de679babe4511579f30d`
- Locally stacked checkpoint: `c0ffa95` (six engine commits cherry-picked onto the fetched base)
- Owned new files: `src/device_executor.rs`, `tests/device_executor.rs`, `docs/DEVICE-EXECUTOR.md`
- Existing Runtime, TensorFS, Creator, Hub and authored packages remain read-only.
- Root owns GPU locking/rental budget. This agent makes no GPU calls or rentals independently.

Initial checkpoint: `2b6c66f`, draft PR https://github.com/cozy-creator/cozy-machine/pull/13.
Qualified unchanged stock Runtime 0.18.89/0.18.99: three invocations per executor, real sklearn
classification, exact spooled result metadata, stale-cancel isolation and local credential/result
errors. Process maps showed no CUDA/NVML before/after inference.

Post helper follow-up: exact SDK output resolver plus generic SDK encoder, typed producer BLAKE2b128
and independent CAS SHA-256 facts. A real installed classifier/image package passed deferred WebP
encoding and independent pixel/dimension inspection. Explicit SDK suite: 3 passed in 8.22 s; fmt
and focused clippy pass. Current image generation used Runtime 0.18.99 with TensorFS 0.3.90.
Legacy worker-module resolver dependency and whole-buffer encoder memory are documented limits.

Isolated `device-pilot validate|run CONFIG.json` now builds optimized. CPU-only validation passed
the actual published SDXL interface and coherent bf16 manifest. No GPU run occurred. Root supplies
the owned non-display rental, measured reservations and actual installed paths before run. The
pilot deliberately labels legacy store/single-writer/host ownership and normal API gates unqualified.
The descriptor-source control shape is additive and world-one capability gated; SourceBlob transfers
must be readonly regular fds. Latest explicit SDK suite 3 passed (6.47 s); pilot/SDK clippy and fmt pass.

Owned A40 pod preparation is CPU-only: recovered original published SDXL 2.4.0 wheel matches
the captured lock SHA-256; isolated sdk99 has Runtime 0.18.99/TFS 0.3.90 with 68 other distributions
restored to captured versions. Remote pilot binary and wheel checksums match local artifacts,
and remote `device-pilot validate` passes without launching a device. Native-derived 6.94GB
closure transfer resumes with 4MiB/s after uncapped SSH burst failures; destination was initialized
independently before copying files, no foreign SQLite. Root owns all model/GPU execution and
ordinary old-stack baseline. Transfer completion and target admission remain required gates.

Final transfer checkpoint: interrupted SSH partial stores were preserved. The existing local
Hub CheckpointReads route supplied ephemeral R2 grants; standard curl fetched four objects at
a time into a separately initialized store-http, checking every length and SHA-256. All 2,604
objects completed in 444.83 s. Native destination acquire/verification then passed the entire
6,939,571,699-byte closure, its 320,558-byte header and selected 2,641 tensors/3,746 plan items.
Final pod configuration selects store-http; CPU validation passes. Evidence is
`~/cozy_v2/outputs/cm-device-20261002/sdxl-pod-source-verified.json`. No GPU inference by this agent.

Typed load/plane/attempt observations now preserve absent/unreadable counters and StageExit's
yielded acknowledgement, passes and stalls. Pilot retains these in load-facts/results/event files.
Actual current/older SDK CPU suite 3 passed in 11.59 s; fmt, clippy and optimized build pass.
Separate pod binary `stage/device-pilot-observations` SHA-256 is
`f301a41a8cd0026f9b54524bbc0d644df5e20585d5a334ff8ddf50b963a5d28f`.
Pod soft FD limit is 1,024; legacy native verification passed, while descriptor-source caching
still needs its own measured FD-lifecycle gate before activation.

Root's first hardware attempt stopped before CUDA because the pilot incorrectly forced
stage-turn support on the published SDK. Source inspection proved actual 0.18.99 Hello
offers only vacate_ranks and its Load/Invoke schemas predate stage/plane fields and Budget.
Pilot now records negotiation and preserves legacy residency, selecting stage/plane controls
from offered capabilities without a version floor. Unsupported Budget is operation-local;
real inference still follows it successfully. Three actual CPU gates pass in 5.99 s.
First logs are preserved; the next configuration uses fresh legacy-sdk99-run-2 paths.

Root's second hardware attempt stopped before allocation because the prototype template
incorrectly put the catalog lane bf16 into Binding.variant. Published SDK executor.py
2127–2143 already measures its own device and derives an sm variant when this field is empty;
the SDK explicitly distinguishes repository lane from hardware claims. Fresh run3 metadata
clears only variant, keeps the exact manifest/payloads, and passes static validation. Native
header parsing (no package/model execution) proves all 2,641 logical tensors are f16 even
though the catalog label is bf16. Evidence: outputs/cm-device-20261002/sdxl-header-types.json.
Root owns the rerun; run1/run2 logs are preserved, no GPU calls by this agent.

Optional source pilot followup integrates committed Root2035bbd (merge5aaead0 preserves all
ancestry/current root Engine changes) and shared broker export d22f316 (localc1625a4).
Typed model_sources selector defaults legacy and prefers descriptor capability when requested;
older peers fall back. Shared model_source_driver::answer maps exact source grants for the GPU
pilot and CPU diagnostic consumer. Source custody is acquired before export and retained until
kernel-observed exact receiver exit; each transferred duplicate closes promptly. Four actual
SDK/process gates pass in10.94s, including resource custody through receiver-handle loss; five
native broker gates pass. Added spawn/source-selection/admission/read/export/FD/shutdown/total
timing evidence and per-run source counts; receiver cache hits remain unknown. Descriptor Load
receives empty Store path while owner broker uses original selected store. Trusted-root package
scope, normal SDK no-Store path versus literal TensorFS imports/security authority, retained
native reader/GPU mechanisms, owner whole-closure FD bound and host/GPU ownership are explicit.
Root's earlier three SDXL A40 images are documented as legacy inference proof only.

Root's paired candidate legacy/descriptor GPU runs each completed three full 1024×1024/20-step
SDXL images. Independent CPU revalidation on pod and after copying outputs checks WebP dimensions,
variance, exact producer BLAKE2b-128/length/SHA-256; all three images match byte-for-byte across
source modes. Descriptor Store path is empty, capability true; PID20015 reused; model-source
exports during all three requests are0. Native owner exported2,606 blobs; sampled owner FD peak
2,619. Load regressed21.933s vs6.403s (+15.530s/3.426x), setup71ms/read189ms. Receiver attribution
pending B CPU profile; Start10.643vs4.171s confounds full total comparison. Both modes still use
69 stage round trips/image:2encode+20denoise+1decode entries and46exit/yielded acknowledgements.
The no-per-step-RPC blanket claim is false despite no hot source RPC. Evidence in
outputs/cm-device-20261002/descriptor-comparison/independent-cpu-verification.json and event-counts.
