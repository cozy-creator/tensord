# Actual source CPU trace ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: CPU-only SDK model-source probe through the exact Rust pilot grant/framing
  helper, actual SDXL header/plan/bytes and native host fill; trace absence of consumer catalog.
- Branch: `feat/4-model-source-cpu-probe-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/4-model-source-cpu-probe-20261002`.
- Fetched base: origin/master `9ef3799b6a96e719f29f6a94ff9e9eff093153fe`.
- Carried root/DCE integration: `07a7adb` fast-forward; subsequent delta is only this probe.
- Owned files: new CPU diagnostic server, SDK probe and this progress note/documentation.
- No package/model imports, no Start/Load, no GPU operations, no primary checkout edits.
- Root owns all GPU operations/rentals; candidate environment remains separate from SDK99.
- Draft PR: https://github.com/cozy-creator/cozy-machine/pull/16
- Actual CPU gate passed complete196-tensor text_encoder fill from SDXL2884, empty store
  path, 197 descriptors; all-thread syscall trace shows no SQLite/leases/NVIDIA opens.
- Whole-plan profile separates RPC0.565s, nativeCtor6.558s and warmOpenSSL6.910s over
  2601objects/6.9377GB; hardwareSHA already exists. No integrity check was skipped.
- Ruff, Python compile and release diagnostic build pass; scripts use the coherent
  installed candidate and the exact shared typed source grant/framing helper.
