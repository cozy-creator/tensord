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

Tracked under cozy-creator/tracker#322. Source work only until the host resource gate or
the coordinated rented CPU slot permits proof. No model/GPU qualification is implied.
