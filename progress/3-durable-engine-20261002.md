# Durable CPU execution

- Owner: `/root/durable_cpu_engine` (Codex sub-agent)
- Purpose: issue #3 SQLite execution journal, process supervision and durable result custody
- Branch: `feat/3-durable-engine-20261002`
- Base: `9ef3799b6a96e719f29f6a94ff9e9eff093153fe` (`origin/master`, fetched 2026-10-02)
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/3-durable-engine-20261002`
- Owned files: `src/journal.rs`, `src/execution.rs`, `tests/durable_execution.rs`, `docs/DURABLE-EXECUTION.md`
- Scope: CPU-only actual-process foundation. No GPU, rentals or existing-stack edits.
