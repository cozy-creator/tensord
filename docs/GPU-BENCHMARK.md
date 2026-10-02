# GPU qualification and comparison

GPU work and RunPod rentals are authorized, with a $20 total cap for this task. The root
agent owns acquisition, accounting and teardown. The first owned headless A40 pilot completed
three real SDXL requests; its isolated legacy loader is distinct from Degree 1 qualification.
Two subsequent clean alternating adapter pairs completed 48 requests (22 warm samples per
arm). All 24 paired images had identical SHA-256 hashes. Warm prepare→encoded-output medians
were 4.831/4.856 s (Python/Rust) and 4.885/4.883 s. These two cycles support feasibility,
not a statistically established zero-overhead claim or a full-stack improvement.
The independent whole-device NVIDIA sample peaked at 11.68–11.94 GiB, whereas Runtime's
allocator-derived peak was about 8.96 GiB. Admission must include allocator reservations,
contexts and external allocations. The exact split is unmeasured, and sampled peaks may
miss brief higher allocations. These resident A40 runs do not qualify 8 GB operation or
the optional machine-copy/streamed-weight latency claim.

The negotiated descriptor source subsequently completed unchanged real SDXL inference with
an empty executor store path. Its first 21.933 s Load was observer-confounded: per-export FD
table scans and external sampling materially changed measurements. Removing those observers
gave 11.629 s; bounded parallel full verification then gave 7.507 s in a three-request trial.
All three parallel outputs match the same-version sequential outputs. Integrity checks remain;
three verification workers, six pending FDs and a 128 MiB queue were derived on this container.
Cold first import/cache state varied, so no aggregate startup speedup is established.

A single fixed-budget stage-policy pair produced 12 matching image pairs. Warm encoded-output
medians were 4.928/4.917 s with/without 69 stage round trips per image, an observed 0.22% difference.
This is a diagnostic comparison, with no memory pressure, fairness or revocation qualification.
Zero model-source exports during inference does not mean zero per-step stage exchanges.

The full old Go-agent/worker/Executor RPC reference completed three matching images with one
executor/load: sequential launch/preparation/first output 31.420 s; warm outputs 4.797/4.831 s.
Its allocator configuration uses expandable segments; matching that setting in the private
SDK99 pilot lowered sampled peak to the same 9,849 MiB. The private GPU pilot still lacks the
new public API/custody path, so this old full-stack reference cannot yield a full old/new ratio.

All 132 generated images were preserved and independently decoded before provider-confirmed
termination of the owned A40. Estimated compute/storage spend was $1.97 against the $20 cap.
Detailed records, exact inputs, excluded cohorts, counters and qualification limits are in
`~/cozy_v2/outputs/cozy-machine-continued-20261002/gpu-runpod/REPORT.md`.
Other live rentals and GPU processes are independently owned. The display-driving 8 GB
RTX 4070 is available for ordinary inference; fault and memory pressure tests use RunPod.

The old A40 Runtime descriptor candidate must not be used for local 8 GB inference:
the later source audit found missing compute-stream quiesce, measured retry-progress,
context-room and bind-OOM recovery fixes. Native failed-copy drain alone is insufficient
before Runtime unmaps a resident set used by compute. A separate latest-safe Runtime/
TensorFS descriptor candidate is being CPU-qualified before root's local run. Neither
fixed grants nor a single executor establish safety. The reviewer handoff and new local
evidence are in `outputs/cozy-machine-local-sdxl-20261002/`; A40 results retain their
original head/condition boundaries.

The first unchanged real SDXL loading/inference milestone has passed on a non-display GPU.
The next integration gate is that same request through the new authenticated public machine
API and ordinary Creator CLI, with normal preparation, events, output custody and collection.
Record every process, hardware/driver/CUDA/SDK/TensorFS version, model identity, request and
output. Keep this separate from complete single-writer custody, low-memory recovery and release
qualification. The current ordinary CLI CPU proof does not establish GPU API qualification.

After the same request works through both architectures on the same rented GPU:

1. Alternate old/new runs. Compare stopped-machine start, warm verified disk, fresh
   executor/model load, reused executor, and repeated inference separately. Record first
   submit to verified output and preparation/executor counts. Downloads get their own timing.
2. Submit several consecutive fresh prompts and seeds with identical dimensions, steps,
   model/precision/attention settings. Preserve each request and output. Measure distributions,
   steady throughput, gaps between device work and utilisation; a single smoke run is insufficient.
3. Switch A→B→A models. Measure shared-weight retention, physical host allocation, RSS/PSS,
   pinned memory and total per-device memory including every CUDA context and compute workspace.
4. Kill an owned executor during real work, then retry as a new attempt of that transaction.
   Verify surviving weights, other readers and final output. Kill/restart the Rust owner
   separately: observe exact executor death and conservative settlement; never manufacture
   completion or replay started effects. Distinguish safety from transparent recovery.
5. On non-display hardware, test natural memory pressure, failed copy and context admission.
   A constrained larger GPU is diagnostic evidence, not qualification of a real 8 GB GPU.
   Never downsize a request to turn a memory failure into a passing result.

SDXL and Anima on 8 GB, H3 streaming on 48–80 GB, and multi-GPU/NCCL are separate gates.
Host-only ownership (Degree 1) precedes shared GPU backing. Shared GPU allocation must then
prove both crash directions, revocation after all work drains and normal executor-issued copies.
Machine-issued streamed copies stay an optional research arm. Synthetic ring-copy measurements
do not establish real-model per-step overhead or end-to-end cold start.

Use the existing published baseline and the new isolated build. Do not patch the old checkout,
change default controller state, or reuse someone else's rental. Retain logs, exact requests,
outputs and spend records under `outputs/cozy-machine-continued-20261002/`. Stop an owned rental
as soon as its measurements finish and verify provider teardown. Budget exhaustion is an external
safety limit; test observation deadlines never authorize production cancellation.
