# Retained Runtime device executor

The Rust adapter uses the actual published Python device Executor. It does not implement
another ModelRegistry, duplicate Diffusers/model code or port the Python worker scheduler.
The CPU proof ran unchanged Runtime 0.18.89 and 0.18.99; source census used Runtime `c04916f`.
These are provenance, not admission floors. The root owns GPU locking and the rental budget.

## Lifecycle and control

`DeviceExecutor::spawn(ExecutorConfig)` starts the installed generation interpreter with
`-I -m cozy_runtime.internal.executor --socket PATH --root ROOT`. It waits for an actual
connection or kernel-observed child termination using pidfd/poll, without a clock-based kill.
SO_PEERCRED must identify the launched PID and UID. The generation shared hold survives exec
and parent death; the installed environment is never updated. Root journal authorization
must precede Start because Start imports authored code and may call package warmup.

One stream carries four-byte network-endian JSON frames, capped at 64 KiB for control.
The Rust records represent consumed command, reply, progress, output and durable-request
fields; unknown advisory fields are ignored. Hello version/revision are provenance.
`import_only`, host weight-plane and stage requests are selected by offered capabilities;
an absent capability fails only that operation. No SDK or TensorFS version equality gate is added.
Published Runtime 0.18.99 offers only `vacate_ranks` and predates the stage/plane command
fields and Budget command. Legacy model load/invoke stays available; the hardware pilot
automatically omits those optional controls instead of refusing the model request.

The qualified sequence is Hello → Start → Load → Activate → PrepareRequest → Invoke.
Preparation and invocation stay in the existing author kernel. One live executor can hold
multiple constructions and serve repeated prepared requests. Async progress and durable
`event=request` exchanges share this stream; Services handles policy and transferred descriptors.
The adapter contains no scheduler or memory-size admission decisions. Credentials cannot cross
the seam. Explicit cancellation writes the stock attempt-keyed marker; stale marks cannot cancel
a later request. Observer teardown must never call cancellation.

StageExit retains its explicit `yielded` acknowledgement separately from the first exit:
a policy owner cannot release a retained turn until the executor confirms application of the
returned budget. SDK load/attempt memory and streaming observations are typed optional records;
absent values remain unknown and signed byte fields preserve the SDK's unreadable `-1` sentinel.
These observations do not authorize kills or supply a scheduler.

Results remain in `result.canonical`: SHA-256 plus byte length verifies that fixed brokered file.
Success also requires typed `outcome.terminal=succeeded`, quiescence and an unpoisoned generation;
`ok=true` alone is not success. Files are opened through the existing O_PATH/NOFOLLOW helper.
Output rows name result field IDs and asset references, not arbitrary local paths. Legacy asset
path binding must come from the trusted SDK resolver, not a separately invented Rust grammar.
SDK BLAKE2b-128 identities and machine SHA-256 custody are separate checks.

`postprocess(codec_config, spool, reply)` runs a trusted ephemeral SDK helper, reusing
`author._codec.encode_frame` and the selected SDK's `AttemptEngine._blob_path` resolver.
It returns the verified canonical value and typed asset bindings (field ID, asset reference,
exact spool name, media/kind, length, producer digest and SHA-256). Files are opened by retained
regular-file descriptors; symlinks/special devices cannot become inputs. The helper fsyncs
encoded outputs; Rust rechecks SHA-256 before the root seals/copies them into durable custody.
The root must finish that custody before success. This post helper adds no persistent service.

The pure legacy path resolver currently lives in the SDK worker module, imported only by this
post helper. Its implementation is not removed from the wheel yet. An additive stock Executor
`output_bindings` capability would remove this dependency without an SDK version floor. Encoders
currently read whole raw/encoded buffers, matching the SDK; bounded/streaming post memory and
strong adversarial package containment remain broader qualification work.

## GPU/bootstrap contract and open gates

1. Root selects/holds an immutable package environment and statically described interface.
2. Reserve every selected device's contexts/library/activation/staging needs before Start;
   CUDA_VISIBLE_DEVICES is declared device affinity configuration and must match Start.devices.
3. Load supplies existing Binding class/path/parameter(s), exact component snapshot map, store,
   adapters, representation/objective and logical-byte facts. Supply authorized device limit.
   The SDK ModelRegistry/Loader/PlaneBackend owns construction, partitioning and supported recovery.
4. Negotiate stage/host-tier capabilities. StageEnter/StageExit ask the single root policy owner
   for component turns. HostTier exchanges retain matching layout memfds through executor death;
   reclaim only after readers/fills/DMA quiesce, charging bytes until physical reclamation succeeds.
5. Activate/PrepareRequest preserve the authored kernel, model and request semantics. Invoke uses
   an authored deadline only, and its plane allowance. No smaller substitute request is generated.
6. The root finalizes/retains output blobs and their exact SDK binding before successful custody.

GPU allocation/copy mechanisms still belong to the stock executor in this baseline. Published
GPU preparation opens TensorFS Store/ReadLease/ReadPlan. The candidate descriptor provider removes
normal executor Store/catalog/GC/writer calls; it deliberately retains native TensorFS plane, header,
fit and read-plan mechanisms. It does not satisfy the original literal no-TensorFS-import claim:
that would require packaging these retained mechanisms into Runtime. Neither source descriptors
nor adopted host memfds prove machine-owned host/GPU weights. Degree 2 and multi-GPU/NCCL remain
separate root-coordinated gates.
The hardware pilot runs a trusted published package as root on the isolated pod. A readonly
descriptor/provider is not a security boundary against same-UID or privileged package code
reopening store paths. UID/capability/seccomp/lifetime isolation must be qualified before any
security-enforced sole-writer or arbitrary-package containment claim.

SDXL and Anima save deferred WebP host frames. The first-party generic encoder is
`cozy_runtime.author._codec.encode_frame`; reuse it in the post path, then verify/seal encoded bytes.
The author-session kernel currently cannot supply model registry state or encode these frames.
GPU load/output encoding passed the scoped legacy SDXL pilot below. Model switching, fault
recovery, descriptor inference and ordinary CLI/Hub/browser consumer gates remain necessary
before declaring the machine replacement complete.

The driver can retain a broker/resource through kernel-observed exit of its exact receiver.
Losing the private owner handle closes its channel and moves retained custody to a pidfd observer;
it does not write the explicit cancellation marker or release sources from elapsed time. A monitor
creation failure observes death synchronously; an unobservable pidfd retains rather than frees
resources. The machine supervisor, not any UI observer, must own the driver handle.

## Root's scoped GPU evidence

Root ran the original published SDXL 2.4.0 package on its owned non-display A40: three consecutive
1024×1024/20-step requests produced independently verified WebP images through one executor PID
6912. Shutdown left no executor/device memory. Runtime 0.18.99/TensorFS 0.3.90 used the legacy
store/residency path. Its Start command took 4096 ms and Load 5129 ms; Start timing excluded
fork/Hello. Invoke times were 5186/4290/4310 ms. The first invocation began with 181 MB allocated
while later invocations began with 7.06 GB: Load completion did not establish GPU residency.
These are internal pilot timings, not full cold start or ordinary CLI comparisons. Evidence:
`~/cozy_v2/outputs/cozy-machine-continued-20261002/gpu-runpod/legacy-sdk99/`.

## CPU evidence

Direct stock Python probes on both released SDKs produced identical 437-byte canonical results
(`sha256:9b6c41dfdb20d1a566ae4ad4b800d4d0362be686f9746d0efe70bfcea9513f21`) and 273-byte JSON
classifier output (`blake2b:fcdfce60707837453da03fdcfb5eddf1`). Evidence lives in
`~/cozy_v2/outputs/cm-device-20261002/{current,older}`.

The Rust bridge then ran three consecutive real classifier invocations per SDK with one executor,
verified [0,2,1] predictions and increasing call_sequence, and retained matching result/output metadata.
Credential rejection leaves the executor usable; changed result bytes fail identity checking.
The three test-harness cases (two stock-executor cases plus the existing descriptor parser check)
passed in 12.25 s. Before and after inference, process maps contained no CUDA/NVML libraries.
No local GPU calls or independent rentals occurred.

The trusted post helper also passed both versions for exact file bindings, independently checked
producer checksums and SHA-256. An installed CPU classification/image package exercised stock deferred
WebP frames: [0,2,1] predictions produced a 32×32 WebP with the expected class colors, independently
decoded by Pillow. The frame encoder and model code were reused, not copied. Runtime 0.18.99 with
newer TensorFS 0.3.90 passed that image gate without an injected floor.

The explicit SDK suite passed three real-process cases in 8.22 s; focused clippy and formatting pass.
Run `cargo test --test device_executor -- --ignored --nocapture` for these gates after preparing the
generation manifests. Default cargo tests skip external SDK fixtures rather than treating missing
fixtures as inference qualification. Fresh-clone fixture installation and a portable gate CLI remain
follow-up. Evidence directories retain the canonical result, named blobs and codec diagnostics.
