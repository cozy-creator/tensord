# Reusable executor session ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: repeated authorized CPU calls with immutable sealed output custody;
  research the existing Runtime model/loader bridge for degree 1.
- Branch: `feat/3-reusable-executor-session-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/3-reusable-executor-session-20261002`.
- Verified base: `origin/master` `9ef3799b6a96e719f29f6a94ff9e9eff093153fe`, fetched 2026-10-02.
- Carried checkpoint: owned Python bridge through `53d52a4`, recreated at `af680d1`.
- Owned files: new session protocol/runner/output helpers, CPU fixture session tests,
  and session/model-bridge documents. Existing single-use runner remains compatible.
- GPU authority: root owns the GPU lock and any rental budget; this agent does not
  independently use a GPU or rent hardware.
