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
