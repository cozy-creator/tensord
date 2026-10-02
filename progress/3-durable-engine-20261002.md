# Durable CPU execution

- Owner: `/root/durable_cpu_engine` (Codex sub-agent)
- Purpose: issue #3 SQLite execution journal, process supervision and durable result custody
- Branch: `feat/3-durable-engine-20261002`
- Base: `9ef3799b6a96e719f29f6a94ff9e9eff093153fe` (`origin/master`, fetched 2026-10-02)
- Worktree: `/home/fidika/cozy/.worktrees/cozy-machine/3-durable-engine-20261002`
- Owned files: `src/journal.rs`, `src/execution.rs`, `tests/durable_execution.rs`, `docs/DURABLE-EXECUTION.md`
- Scope: CPU-only actual-process foundation. No GPU, rentals or existing-stack edits.

Draft PR: https://github.com/cozy-creator/cozy-machine/pull/10.
Implementation checkpoints: `940e346`, `6509248`.

Validation: nine actual-process/socket/filesystem tests passed (2.68 s), including
actual Rust-owner SIGKILL with live orphan retention and terminal settlement after
the exact birth ended. The one ignored helper was explicitly spawned and executed
inside that fault test. `cargo fmt --check` and focused clippy `-D warnings` pass.
Build/test commands used nice 19 and the program's heavy-job lock after load dropped.

Root integrates modules with the package bridge and public front door. This is not
ordinary CLI, installed SDK, Hub/browser, systemd/container, GPU or old-stack
benchmark qualification. No task GPU/rental processes were created.

Follow-up: bounded coalesced progress replaces per-event FULL-WAL updates. A 257-unit
actual socket gate advances get/list/duplicate receipts while independent persisted
state and WAL size stay unchanged; cancel/terminal commits retain the latest count.
Reserved status revision windows preserve monotonic cursors across actual owner death,
and renew without cursor reuse. The optional stored ceiling defaults for older records.
Twelve focused tests pass (0.36 s), including the allocator boundary and queued cancel.
