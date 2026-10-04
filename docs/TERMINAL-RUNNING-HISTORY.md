# Replay the running fact when an observer misses the live transition

A Run observer can first read queued/preparing, then receive its next event page only
after the accepted work is terminal. `NativeBackend.terminal` currently preserves products
and recent genuine progress, but does not replay the journal's `running_revision` event.
Creator therefore still calls those later samples preparation and loses their positions.
The ordinary CPU burst proof passes when it observes running normally; that proof does not
cover this skipped-state case.

Replay the existing durable running revision and `started_at_ms` before later samples in
the terminal projection. The event describes authorized work that actually started; progress
does not supply lifecycle authority. Its original sequence/time survive journal reopen and
cursor filtering. Warm/preparation-only runs whose running revision is zero get no event.
No progress transaction, background flush, cancellation or invented RunState is added.

Proof must hold a real observer's first page until accepted work ends, while its first state
snapshot is genuinely queued. A controlled CPU producer on a real owned process supplies
frames; actual Engine authorization records running. Real TLS Run must then deliver that
fact before its denoise endpoint and outcome. Creator's queued-to-terminal consumer must
preserve progress positions once the actual running fact is replayed. Existing authored
SDK/ordinary CLI burst proof remains a separate regression gate.

CPU proof is green: `terminal_running_tests::skipped_running_replays_the_actual_start_before_terminal_progress`
forces this ordering through real TLS and a real owned CPU producer. It verifies the original
running sequence/time before genuine denoise 30 and outcome, plus the reopened projection.
The existing preparation-only control receives no invented running event. Creator's queued
consumer has two arms: without the fact it cannot infer state, while the replayed fact retains
the position before terminal and after reopening.

The final combined source (machine 3e7c01d = integration af3 plus #54; Creator e81f50e0 = integration df6
plus test-only #1028; TensorFS 78 tree-identical to 693) passes 161 top-level Rust cases, clippy and
build, six tested Creator component packages, and the actual authored SDK/ordinary CLI burst
and controller restart in 5.93 s. The seven ignored entries are not counted as passing;
they include parent-driven subprocess helpers, network cases and the explicit SDK codec case.
Runtime H370 SHA 0d50f6b1 / TensorFS f089 were explicit CPU inputs; this does not qualify H0C VAE
math, model inference, resource-constrained quality or GPU throughput. Tracker #322 records
the source/artifact maps and logs under outputs/codex-machine-audit-20261004/api/terminal-running-remote.
