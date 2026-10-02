# Model source broker ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: standalone linked-TensorFS model_source_read broker for device pilot;
  exact trusted manifest/component membership, readonly header/asset/object fd exports.
- Branch: `feat/4-model-source-broker-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/4-model-source-broker-20261002`.
- Fetched base: `origin/master` `9ef3799b6a96e719f29f6a94ff9e9eff093153fe`.
- Carried CPU/device checkpoint: `de37e89` and its prerequisite owned integration commits,
  recreated at `17ee23f`. Root integrates only this task's subsequent delta.
- Owned files: `src/model_sources.rs`, isolated CPU/source tests and documentation.
- No edits to native_backend, journal or primary repositories; root owns GPU/rentals.
- Authoritative CPU metadata gate: SDXL manifest
  `sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642`
  in `/home/fidika/.tensorfs`, verified anew rather than trusted from memory.
- Draft PR: https://github.com/cozy-creator/cozy-machine/pull/15
- CPU checkpoint: five linked-core tests pass; selected readonly/sealed descriptors,
  wrong-manifest/component rejection, copied closure native admission without foreign
  SQLite, and corrupted-transfer rejection. Clippy, rustfmt and script Ruff pass.
- Released helper `target/release/model-source-check --verify` admits 2,604 SDXL
  selected source objects / 6,939,571,699 bytes; header/read plan and actual
  selected object also pass candidate TensorFS native descriptor import without CUDA/NVML.
- Not qualified: SDXL inference, cold start, host/GPU memory ownership, whole-closure
  descriptor hard-limit behavior. Root owns hardware and rentals.
