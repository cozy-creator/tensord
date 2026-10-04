# Publicowner compatibility census before protocol removal

Tracker #322/#319. Owner: Codex api_audit. Frozen machine source946a03e;
Tensorhub source42b17fe9. No production or provider state changes are authorized by this proof.

The current Rust api::serve registers PodHost, WorkerControl and Machine services on the
same pinned TLS listener. main.rs uses that server for its real foreground and provisioned
machine. The retired-worker router has no production caller; its isolated fixture proves
only its own response. The earlier claim that production already served machine/v1 alone
was incorrect. No legacy RPC is counted as unreachable on that premise.

NativeBackend implements authenticated DescribeMachine and execution-workspace discovery,
plus legacy release-root submit/read/events/control/collect/ack operations over the sole
journal. PreparePackageSet is absent from the implementation and build.rs generates default
UNIMPLEMENTED stubs. Legacy captured publicowner submissions omit release_root, which
NativeBackend currently requires. Thus listener registration is not proof of that consumer's
preparation/submission compatibility.

Prove this boundary using the real cozy-machine serve process and retained identity,
actual NativeBackend and pinned TLS generated PodHost/WorkerControl clients. Check authorized
and foreign-key discovery, workspace persistence and replacement identity, unsupported
preparation, captured-submit refusal without accepted work, and the distinction between
Control's boot-only instance projection and publicowner's actual provisioned instance pin.
The test may stop only its own explicitly launched fixture processes. No model inference is
performed. Keep shared generated records/backend projections used by machine/v1 and HTTP.

Before proposing a publicowner port, preserve its durable stage-before-dispatch workspace
pin, exact accepted assignment, ambiguity after a lost reply, explicit cancellation,
terminal/media custody before ACK, and authenticated per-operation authority. Current v1
Status/Run do not expose the execution journal identity or caller-held workspace pin;
server-side query discovers whichever journal exists now. Migration cannot merely replace
RPC names and discard these fences. Any missing necessary contract should be justified by
actual consumer proof, rather than speculative wire extensions or deployment equality.

Production Hub configuration/use, real published serving preparation and media custody,
provider identity and cross-boot replacement recovery remain unqualified by this CPU census.

## Confirmed source and CPU boundary

On source946, server.rs842–1338 implements23 PodHost handlers and1340–1496
implements14 WorkerControl handlers (mostly forwarding). Both services are registered;
zero implemented handlers are proved unreachable by listener registration. Shared backend
workspace/get/events/control/read/output projections and worker-protocol products/outcomes
are called by machine/v1, HTTP outputs and executor/storage code. Removing the service
wrappers is not permission to remove those shared data types or the journal's workspace.

Two actual `cozy-machine serve` process/TLS tests pass in0.59s:
`actual_listener_legacy_discovery_preserves_workspace_but_publicowner_capture_is_unsupported`
and `actual_listener_legacy_control_and_collection_preserve_actor_and_exact_outcome`.
The first checks authorized/foreign/stale-boot discovery, both legacy services, real
unsupported preparation/captured submission with no accepted work, restart persistence,
and typed refusal after replacement of this owned fixture's journal. The second authors
one queued journal record, then uses actual API read/events/explicit cancel/collect/ACK;
another admitted actor cannot read it, changed outcome ACK is rejected, and collection
replays the identical terminal. These are transport/journal controls, not model inference
or a real PostgreSQL/public HTTP/media-provider qualification. Focused clippy passes.
Evidence: outputs/codex-machine-audit-20261004/api/publicowner-real-listener{,-clippy}.log.

Tensorhub42b publicrequests.OpenWorker checks Open plus Workspace and sets healthy; Ready
renews the public-owner lease. Rust passes those calls. PrepareServing later calls the
missing PreparePackageSet; its gRPC error is not a PreparationRefusal. Reconcile retains
worker_preparing, and succeeding Workspace probes retain healthy. An enabled operator
configuration can therefore advertise a queue whose selected Rust worker cannot prepare
that offering. Actual production use/configuration remains unverified.

A separate registered legacy-submit source defect needs its own red/green increment:
NativeBackend.submit returns accepted_public(actor, request OR submission) before comparing
both IDs or authored intent. Different IDs or changed release/payload can receive the old
receipt. The new machine/v1 semantic-intent proof does not cover this path.

## Minimal next sequence

Keep the working wrappers and shared records while proving/fixing legacy semantic replay.
Before enabling Rust public serving, prove an actual configured Hub owner and published
callable over the production listener, including definitive unsupported-operation admission,
ambiguous acceptance, journal replacement, cancellation and terminal/media custody.

A machine/v1 port should submit the retained authored release/model/input intent to the
existing machine engine, not create another execution owner in the Hub. Retain the Hub's
assignment before dispatch and preserve caller-observed workspace identity atomically with
new acceptance; v1 currently lacks that boundary. Reads and collection must use the accepted
identity, never refresh it to make an ambiguous request executable on a replacement journal.
Only propose a narrow contract change after review of the concrete consumer proof. Preserve
Hub lease fencing and adopt verified output bytes before collection acknowledgment. Delete
old wrapper RPCs only after real consumer/provisioned-identity census and a qualified port;
no new exact-source/version gate belongs to this decision.
