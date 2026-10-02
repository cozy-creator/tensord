# CPU owner bootstrap

Owner: Codex /root. Purpose: standalone host-owner implementation.
Branch: feat/host-owner-bootstrap.
Base: f7c43c1e4801fd65a53715d47f060aca543c269b.
Worktree: ~/cozy/.worktrees/cozy-machine/host-owner-bootstrap-20261002.
Program: https://github.com/cozy-creator/cozy-machine/issues/1
Slice: https://github.com/cozy-creator/cozy-machine/issues/2

Existing repositories, the owner daemon and GPU machines remain untouched.

CPU component merged: https://github.com/cozy-creator/cozy-machine/pull/7
Code checkpoint: c557c1874df8b69ca82e9fdc56e2a9ad93ef4ad9.
Primary mirror: ~/cozy_v2/cozy-machine, fast-forwarded after a clean-tree check.

Validation: cargo build/test, fmt and clippy -D warnings passed; 12 Python tests
passed. Actual Rust/TensorFS + Python classifier gate passed 10 executor deaths,
11 survivor checks, replacement inference, zero-budget disk fallback, LRU pressure,
live-view lifetime and version/additive handling. Rust mapped no CUDA/NVML and
opened no GPU descriptors. Detailed evidence is in docs/CPU-EVIDENCE.md and
~/cozy_v2/outputs/cozy-machine-bootstrap-20261002/exact-head-gate/.

Issue #2 remains open: package spawning, recipient/descendant registration,
broader physical reclaim and TensorFS plane integration are not complete.
The comparison gate in docs/PLAN.md precedes expansion into a full rewrite;
ordinary Cozy CLI, diffusion/GPU/NCCL and old-stack benchmarks remain open.
All task processes ended; no GPU use, rentals or release publication occurred.
