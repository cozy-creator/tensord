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
