# Shared host plane ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: linked native TensorFS machine-owned host allocations, measured physical
  backing charge, trusted actor/plan grants and actual-recipient lifetime; CPU gates.
- Branch: `feat/4-shared-host-plane-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/4-shared-host-plane-20261002`.
- Fetched origin/master base: `9ef3799b6a96e719f29f6a94ff9e9eff093153fe`.
- Carried root integration: `3a5ec84488c4cd5e6e91929d6fc0c84f45fc3b4f`.
- Owned files: `src/shared_host_plane.rs`, related integration tests/docs and the
  released native dependency addition. No scheduler, journal or public GPU pool edits.
- Existing HostTier asks name/layout only: trusted native layout priming is required;
  raw cache names are never actor/model authority. Older peers retain optional-tier fallback.
- Root owns GPU/rentals/fault pressure; this task is CPU-only.
- Draft PR: https://github.com/cozy-creator/cozy-machine/pull/18
- First CPU checkpoint: 5 actual native host/process/fd gates and 5 existing model-source
  gates pass. Native host-only maps/fds contain no CUDA/NVML/NVIDIA opens.
- Exact exited-recipient whole-file OFD unlock fixes retained-grant duplication before
  native release; backing reclaim is tested while copied descriptors remain open.
- SDK typed partition registration and actual model consumer/GPU gates remain open.
- Imported frontdoor wire/header checkpoint e9c520a as4944938; Root should integrate that
  prerequisite separately if already present to avoid duplicate commits.
- Typed sealed HostTierPrepare registration now authorizes selected header components and
  native traversal/part/region planning. 8 native host +5 source gates pass; actual paired
  SDK/Rust-owner/native host adoption proves zero recipient fills/copied source bytes.
- Runtime dependency continuation: PR1112, fetched0d67384 +safe9eb29a1c; separate narrow SDK
  hook and46 CPU source/skew/policy/registration gates. No GPU used.
