# CPU component qualification

Passed with an actual trained sklearn digits classifier: 98.00% accuracy on 450 held-out samples, 5,200 weight bytes.

- 10 executor kills after successful attachment and inference; all survivor outputs and replacement outputs matched.
- Live NumPy views prevented unmapping/release until explicitly dropped.
- Dead leases reclaimed through pidfd-observed process death; no timeout or sleep kills.
- Cold/warm shared host attachment, active-borrower disk fallback, zero-host-budget disk inference and two idle LRU evictions passed.
- Older/newer runtime strings, additive fields and unknown-operation recovery passed.
- Runtime maps and descriptors contained no CUDA/NVML library or GPU device path.
- Single cold attach: 0.328 ms; five-attach warm median: 0.130 ms. These tiny component timings are not model cold-start benchmarks.

This is an experimental CPU component gate. It does not qualify ordinary `cozy run`, CUDA allocation sharing, real diffusion models, NCCL, arbitrary-low-memory recovery or interrupted GPU computation.

Provenance: Rust 1.91.1, Python 3.14.7, linked TensorFS 0.3.87. cargo build/test and clippy -D warnings passed; 12 Python transport tests passed. The native SCM_RIGHTS truncation regression passed. Exact result/fixture files are at ~/cozy_v2/outputs/cozy-machine-bootstrap-20261002/exact-head-gate/.
