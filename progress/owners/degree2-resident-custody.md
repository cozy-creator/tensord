# Degree 2 resident custody

Owner: durable_cpu_engine (Paul Fidika).
Task: typed resident allocation/recipient custody using the existing process-birth engine/journal
and host owner; no parallel scheduler, copy engine, live opaque-handle adoption or advertised GPU capability.
Branch: feat/degree2-resident-custody-20261002.
Worktree: /home/fidika/cozy/.worktrees/cozy-machine/degree2-resident-custody-20261002.
Base: fetched origin/master 9ef3799b6a96e719f29f6a94ff9e9eff093153fe,
fast-forwarded to explicit Root integration checkpoint 86ec04a101673d3da6f03b6c0becf67addf620aa.
Native dependency experiment: TensorFS PR304 a6f448b (IPC) / 7f9b2f6 (separate OFD host cleanup).

CPU-only implementation/gates. Root owns every headless CUDA test, rental and remaining budget.
Existing API/engine/journal/catalog/host-module business decisions remain their owners' files.
This module tracks live allocation custody and emits exact blockers; it never schedules or kills.

Foundation: src/resident_custody.rs exposes typed identity/key/phase/recipient records; one fill
writer, immutable-ready readers, exact birth/pidfd retention, native-only completion and physical
release boundaries. Engine's existing accepted actor/process authority and activity wake are reused;
no scheduler/journal/host module changes. Actual CPU child+SCM fd cancellation/exit/custody gate and
wrong-pidfd gate pass (2 tests, 0.22 s). Focused library/test clippy -D warnings passes.
Native source probe checkpoint20a3aa0 is CPU-built only; Root owns headless run. Epoch restart rejects
old tickets without pretending that a new empty ledger recovers old device handles/physical bytes.
