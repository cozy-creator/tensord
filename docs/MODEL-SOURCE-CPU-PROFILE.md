# Actual source CPU qualification

The isolated candidate uses Runtime0.18.100 and TensorFS0.3.90 descriptor candidate;
all 68 published SDXL dependency pins and unchanged SDXL2.4.0 wheel are retained.
The CPU owner uses the exact `model_source_driver::answer`, device framing and fd sender
used by the GPU pilot. The SDK consumer uses stock Executor._durable,
BrokeredModelSources and Checkpoint with `store=""`; no authored module, Start or Load
is executed. Every native constructor still performs its original integrity checks.

`probe-model-sources-cpu.py` reads the authoritative SDXL2884 header and complete
text_encoder traversal, imports 197 real object descriptors, and fills the actual native
host plane: 196 tensors / 246,120,960 logical bytes / 247,463,936 layout bytes. FD count
was 4 before and 7 after fill. Strace6.8 captured openat/clone for all consumer threads;
no TensorFS SQLite, lease-directory or NVIDIA device open occurred. No CUDA/NVML library
was mapped and SDK Torch initialization stayed false. This proves the observed source
path, not a hostile-process privilege boundary or full inference.

The first probe attempted a single tensor while requesting its whole component. Native
planning correctly rejected TRAVERSAL_INCOMPLETE. The successful probe uses the complete
196-tensor traversal; the failed diagnostic artifacts remain preserved separately.

The whole four-component profile imported 2,601 objects / 6,937,666,560 bytes through
the same source exchange. Actual installed native constructor calls were instrumented,
not replaced. OpenSSL reverified the same still-open fd immediately afterward, so that
comparison is explicitly warm-cache and uses 1 MiB reads versus native 64 KiB reads.

| Measured CPU phase | Wall | User CPU | System CPU |
| --- | ---: | ---: | ---: |
| 2,602 complete RPC/reply/SCM receives | 0.565 s | 0.178 s | 0.045 s |
| 2,601 native constructors | 6.558 s | 5.305 s | 1.250 s |
| OpenSSL same-fd verification | 6.910 s | 5.433 s | 1.474 s |

Native constructor throughput was 1.058 GB/s; OpenSSL was 1.004 GB/s. Whole profiling
took 14.219 s because it performed both complete verification passes. Actual pod flags
include SHA-NI/AVX2. TensorFS already dispatches hardware SHA, so replacing it with
OpenSSL or another SHA crate is not justified by this result. No fixed polling delay
appears in the shared Rust or SDK durable read path; the measured RPC total is small.

The separate same-native Rust copy diagnostic explicitly reported `x86-64 sha extensions`
over the same 2,601-object set. Constructor: 6.474 s. First mapped read into a reused
64 KiB destination: 0.780 s, 108,335 minor faults and zero major faults. Second read:
0.321 s, zero faults. Owner-side native object lookup was outside those timers. The
0.459 s first-touch difference does not account for the remaining actual GPU load tax;
this test excludes a full-size pinned destination, CUDA registration/copies and GPU
loader context. Actual-run phase attribution is still needed before choosing a
verification pipeline, map-first hashing or a typed machine-attested source operation.

Evidence: `outputs/cozy-machine-continued-20261002/descriptor-candidate/` contains
`evidence.json`, `profile.json`, `cpu-source-strace-2.trace.*`, and the native
`descriptor-read-profile.json`. The owned pod retains the same files below
`/workspace/cozy-machine-pilot/descriptor-candidate/`. GPU runs and their conclusions
belong to root; this agent performed no GPU or rental operations.
