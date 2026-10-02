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

The qualified sequence is Hello → Start → Load → Activate → PrepareRequest → Invoke.
Preparation and invocation stay in the existing author kernel. One live executor can hold
multiple constructions and serve repeated prepared requests. Async progress and durable
`event=request` exchanges share this stream; Services handles policy and transferred descriptors.
The adapter contains no scheduler or memory-size admission decisions. Credentials cannot cross
the seam. Explicit cancellation writes the stock attempt-keyed marker; stale marks cannot cancel
a later request. Observer teardown must never call cancellation.

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

GPU allocation/copy mechanisms still belong to the stock executor in this baseline. Current
published GPU preparation opens TensorFS Store/ReadLease/ReadPlan. Adopted host memfds do not
remove that dependency or prove one writer. A separately qualified descriptor/header/plan provider
is necessary before the no-TensorFS-import or sole-writer claims. Degree 2 and multi-GPU/NCCL
qualification remain root-coordinated gates; this component makes no GPU/model performance claim.

SDXL and Anima save deferred WebP host frames. The first-party generic encoder is
`cozy_runtime.author._codec.encode_frame`; reuse it in the post path, then verify/seal encoded bytes.
The author-session kernel currently cannot supply model registry state or encode these frames.
Actual GPU load, output encoding/custody, model switching, fault recovery and ordinary CLI/Hub/browser
consumer tests must pass before declaring the machine replacement complete.

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
