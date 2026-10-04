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
