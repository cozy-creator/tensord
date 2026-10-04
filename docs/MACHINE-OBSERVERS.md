# Run observers, output grants and event cursors

A machine accepts work durably. A Run observer is a reader of that work, not its owner.
Disconnecting, losing an admitted signing key, a rental-authority lease expiring, or the
observer's Cozy-Cap expiring ends only that transport. Explicit Control or an authored
request deadline supplies cancellation authority.

Every authenticated Status, Run and Read stream checks authority before delivering data,
including while waiting on a quiet backend or when the caller stops consuming. The check
covers both the signer's current admission and the signed capability's Unix expiry. Write
checks the same authority for incoming frames and before finalizing an object. An interrupted
upload retains resumable staging bytes and does not make an object available to new runs.

A run grant with output restrictions may see only matching product events and terminal
output inventory. Inline results, logs, triage documents, measurements and arbitrary failure
text are private to unrestricted run and machine grants. Product list indexes stay one-based;
`name` grants every index and `name/i` grants that item. Machine grants preserve the full
owner's view.

An event cursor specifies which entries to transmit, not which product history to use.
An attaching observer rebuilds the output projection from durable history: each output's
revision ordinal, previous parts and latest product. It emits a current state/head snapshot
at sequence zero, then only log entries strictly after the caller's cursor. The outcome
carries the complete current allowed output inventory even if all products preceded the
cursor. A cursor at or beyond a terminal outcome ends after the state snapshot.

Read opens one snapshot, checks `if_rev` against it, and retains its file handles throughout
the read. A changed output cannot splice new bytes into that snapshot. A short file or an
offset whose bytes cannot be reached is DATA_LOSS, rather than a successful truncated read.

CPU checks in `tests/machine_v1_boundaries.rs` exercise the real TLS/gRPC server with
controlled events, backpressured output files and the real resumable object writer. They
prove API boundaries only. Ordinary CLI reconnect/collection, candidate-wheel deployment,
congested-disk result delivery and GPU inference remain separate product gates.
