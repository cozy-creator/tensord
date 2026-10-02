# Reusable CPU executor sessions

`cozy_machine_client.session_runner` is a separate module with
`runtime.author-session/1` and `outputs.sealed-fd/1`. The current single-use
`cozy_machine_client.runner` and `runtime.author-cpu/1` are unchanged. The existing Rust
engine waits for process exit before mutable-file custody; it must not select the session
module until its worker-pool and sealed-descriptor adapter are implemented.

## Session and custody contract

1. The child sends `Ready` before importing SDK or package code.
2. `OpenSession` names the immutable package generation and session identity. The child
   verifies and holds that generation for its entire lifetime, then sends `SessionOpened`.
3. The owner durably authorizes an attempt before sending `SessionInvoke(seq, invocation)`.
   Sequences strictly increase; each attempt carries session, sequence and execution identity.
4. The existing Runtime `prepare`/`invoke` kernel executes the authored package. Progress
   counts completed positions and is lossy. Cancellation matches all three identities.
5. The child reads each declared regular output through bounded 64 KiB copies, checks that
   its metadata stayed stable and its bytes match the SDK checksum/length, and seals a memfd.
   It sends typed relative name, role, SHA-256 and length plus read-only CLOEXEC descriptors.
6. The owner validates descriptors and bytes, takes durable custody and commits the terminal
   outcome before `SessionAck`. Only then does the child release the outbox and send `ReadyNext`.

SDK asset checksums and machine CAS identities are distinct domains: the qualified SDK uses
`blake2b` with a 16-byte digest, while machine objects use SHA-256. The trusted adapter
normalizes the SDK checksum into a typed record once; the snapshot validates that declared
identity and independently computes the machine object hash. New unsupported algorithms
fail that output operation, not an entire peer version.

One terminal outbox is retained. An unacknowledged repeated sequence replays exactly the
same terminal and sealed bytes without executing the handler. An acknowledged or older
sequence cannot execute again; new work waits for the matching custody acknowledgement.
The machine journal still owns durable attempt identity and no-rerun guarantees across
sessions/core restarts. This child is not another record owner or recovery journal.

Owner EOF while idle releases the process and generation. During a call it does not create
user cancellation: entered code may finish, but no queued call starts and no output success
is claimed durable. An explicit shutdown closes the idle session. No elapsed-time kill or
automatic inference replay is implemented.

## Records

Control frames use the existing network-order four-byte length and typed JSON, capped at
1 MiB. Unknown advisory fields are accepted. There is no SDK-version floor. Output fd
markers follow the `SessionResult` frame in artifact order, one SCM_RIGHTS fd per NUL marker.

| Direction | Record |
| --- | --- |
| Owner → child | `open_session {session_id,package,generation,module}` |
| Owner → child | `session_invoke {session_id,seq,invocation: Invoke}` |
| Owner → child | `session_cancel {session_id,seq,execution_id}` |
| Owner → child | `session_ack {session_id,seq,execution_id}` |
| Owner → child | `session_shutdown {session_id}` |
| Child → owner | `session_opened {session_id}` |
| Child → owner | `session_progress {session_id,seq,execution_id,completed_units,detail}` |
| Child → owner | `session_result {session_id,seq,execution_id,value,artifacts}` + immutable fds |
| Child → owner | `session_failed` or `session_canceled`, with the same provenance |
| Child → owner | `ready_next {session_id,completed_seq,sdk_imports,package_imports,...}` |
| Child → owner | `session_error {session_id,seq,code,detail}` for an unsupported/control operation |

`ReadyNext` includes measured module-load counts and identities for this component's reuse
gate. It does not establish cached model generations. The classifier trains on every call.

## Real CPU gates

The installed classifier uses different seeds to shuffle its real training data, varied
input samples and actual sklearn inference. A → B → A uses one interpreter, imports the
SDK/package once, produces the expected different classifications, and reproduces A's
probabilities. Its report records a call counter, proving replay did not execute another
classifier call. The owner copies/fsyncs the sealed report before acknowledgement.

The tests retain A's descriptor while B and A run, overwrite A's original mutable spool
file, then stop the interpreter: the earlier descriptor's bytes remain correct and
unwritable. Other gates cover unacknowledged replay, replay after acknowledgement, mismatched
acks, next-call backpressure, stale/cross-session cancellation, idle EOF, active owner death
with queued work, SDK-identity corruption and descriptor cleanup. Single-use package and
transport tests remain included.

On 2026-10-02, all **41** combined tests passed in **61.17 s** on CPython 3.12.12.
The six real session lifecycle/inference gates ran against independently installed
Runtime **0.18.89** and **0.18.99** generations; TensorFS resolved to **0.3.89**. The
remaining gates include single-use real-package inference and transport/descriptor checks.

## Remaining limits

The sealed-output path and reused process are component proofs; the Rust engine/pool,
ordinary Cozy CLI, supervisor/core-death settlement and provider/browser consumers remain
separate gates. Descriptor count is one per declared artifact; general large/many-output
admission and disk-backed custody need the machine's resource policy. Deferred image/media
frames, models, tree outputs and child calls are still outside this author-session door.
See `DEGREE1-MODEL-BRIDGE.md` for the stock-device-executor path that retains the existing
model and codec implementations. No GPU or rental was used by this agent.
