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

Software updates retain their existing payload/status/rollback journal. They reserve actor/ID
ownership in the machine journal before update admission. Ordinary acceptance and cancellation
check that reservation in their SQLite transaction: update-first refuses Control as unsupported;
cancel-first refuses a later UPDATE. An update ID stays reserved across restart. A refused update
may be retried as an update under the same ID; it cannot become an ordinary run or cancellation.

Attachment prefers the actor's ordinary/canceled record over another actor's machine-wide update
status with the same textual ID. Update maintenance still requires a machine-scope capability.
This changes ID ownership only; it does not merge controllers, change update activation/rollback,
or introduce version/admission gates. Native tests and the real-binary update/rollback test pass;
ordinary CLI proof is still a separate consumer check.

Tracker: https://github.com/cozy-creator/tracker/issues/331
