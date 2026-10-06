# Cancellation before Run acceptance

An authenticated machine-scope `Control(CANCEL, id)` reserves its signer's run ID durably even
when Run has not accepted it. The journal creates a terminal canceled record in the same
actor/ID namespace as Run acceptance, with an internal marker that no spec was accepted.
SQLite immediate transactions serialize cancellation and acceptance.

If cancellation wins, every later Run spec for that actor/ID returns the canceled record
without preparing or executing it. `Run(id)` observes its ordinary canceled state and outcome,
including after restart. Repeated cancellation returns the same record. Other actors keep
their independent IDs. Pause and Resume still require an existing run. Completed outcomes
remain completed when cancellation arrives later.

If acceptance wins, existing durable engine cancellation applies. The caller must observe
the terminal outcome; a missing run or a transport/authentication failure never acknowledges
cancellation. Closing an observer has no cancellation authority. This adds no RPC, generation
selector, legacy submission closure or deployment compatibility gate.

Tracker: https://github.com/cozy-creator/tracker/issues/331
