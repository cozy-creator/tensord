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

CONFIRMED CPU red on the real NativeBackend: the next stream poll after key revocation
returned another64KiB chunk with no network buffers involved. On the production TLS router
with64KiB client receive windows, both the authorized control and revoked observer received
the entire8MiB snapshot and EOF. Logs: api/legacy-log-authority-red.log.

The sole production change wraps the existing snapshot iterator with Api::revocable, which
uses shared StreamAuthority. No new authority implementation or execution operation exists.
The same two CPU controls pass in0.54s: a direct future poll immediately returns
UNAUTHENTICATED and ends; a nonrevoked TLS reader still receives all8388608 bytes. The
revoked TLS reader received196608 bytes total (the initial65536 plus131072 in the delivery
tail), then UNAUTHENTICATED; a fresh read with the removed key also refuses. That tail can
be buffered before revocation and is not asserted to be a new unauthorized poll. The direct
poll control isolates the authority boundary. The test bounds total delivery below the
snapshot size rather than depending on this exact scheduling-sensitive tail count.

Both controls compare the entire accepted queued execution before/after, so transport
authority loss leaves its durable state unchanged. Snapshot bytes are real owned on-disk
logs; no mock backend, SDK/model execution, provider, production or personal daemon change
is part of this proof. Evidence lives under outputs/codex-machine-audit-20261004/api.

All library/test clippy targets pass with -D warnings; git diff --check is clean.
The targeted test filter is legacy_log_authority_tests (two cases). The only production
hunk changes the existing ReadMachineLog return to use Api::revocable.
