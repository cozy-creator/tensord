# Legacy log snapshot lifetime authority

Tracker #322/#319, owner Codex api_audit. Base784d4e9 after the actual listener
census and registered Submit replay fix. Production/provider/personal daemon state is untouched.

Registered PodHost ReadMachineLog authenticates once, reads an owned retained file snapshot,
and streams64KiB chunks. Its raw iterator lacks the shared revocable wrapper used by other
owner streams. Source alone does not establish post-revocation disclosure: data already
copied into network/client buffers while authorized cannot be recalled.

Use actual NativeBackend with bounded real on-disk log content. First prove the exact stream
poll boundary: consume one response item, revoke the signed Claim key, then poll another
item with no network buffering involved. Complement this with the production TLS router and
client HTTP/2 receive windows constrained to64KiB so an8MiB snapshot meets backpressure;
report any already-buffered tail separately from a new unauthorized stream poll. Verify that
ending the observer never changes an accepted queued journal record. No inference is run.

Only if the CPU proof shows future unauthorized polls, wrap this existing stream with the
shared StreamAuthority through Api::revocable; do not invent another auth implementation or
cancel work. Retain normal complete snapshot reads and explicitly bounded test timeouts only
for fixture failure diagnostics. Root's Hub refusal review is a separate read-only task.
