# Python package bridge ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: issue #3, trusted process runner reusing Runtime's author invocation kernel,
  static package description and immutable package-environment experiments.
- Branch: `feat/3-python-package-bridge-20261002`.
- Base: `9ef3799b6a96e719f29f6a94ff9e9eff093153fe` (`origin/master`, fetched 2026-10-02).
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/3-python-package-bridge-20261002`.
- Owned files: new execution protocol/runner/packages modules, CPU package fixtures/tests,
  and `docs/PACKAGE-BRIDGE.md`. Existing Runtime/TensorFS/Creator/Hub remain read-only.
- Qualification boundary: CPU process/socket/SDK slice; no GPU, rentals or ordinary CLI proof.
