# Authenticated CPU front door

Owner: `/root/machine_front_door`; issue: cozy-machine #3/#4.
Branch: `feat/3-front-door-20261002`; base:
`9ef3799b6a96e719f29f6a94ff9e9eff093153fe` (fetched origin/master).
Worktree: `~/cozy/.worktrees/cozy-machine/3-front-door-20261002`.

This slice implements the existing Creator-facing pinned-leaf TLS/gRPC boundary,
not a parallel unauthenticated run protocol. Generated Rust bindings consume the
worker-protocol schema. Every implemented operation verifies the current client's
Ed25519 ClaimProof bound to worker identity, boot identity and TLS leaf. ProtocolInfo
and the HMAC-authenticated bootstrap receipt are the existing bootstrap exceptions.
Version numbers report provenance; no accepted operation refuses a peer by version.
The service advertises baseline minimum 0 so an old controller does not refuse
the connection through a reported floor. Missing capabilities affect one operation.

First qualification: private fixture endpoints, real Go protobuf clients and TLS pins,
valid/invalid signatures, old/new/additive peers, identity/readiness, Control ClaimAck,
and reusable journal adapters. Full ordinary `cozy run` additionally needs package
preparation/upload, capture, invocation and output custody integration. Browser media,
GPU work, Hub provisioning and old installer/update are later gates. CPU code loads
no CUDA/NVML. Existing checkouts and production daemons remain untouched.

## Implemented boundary

`src/api` serves HTTP health/readiness and both deployed gRPC service names on one
TLS listener. Rust bindings are generated from the exact worker-protocol-v2 source
recorded in `vendor/worker-protocol/SOURCE`, rather than manually repeating schemas.
Source hashes are generation provenance, never runtime peer admission checks.

Implemented behavior is ProtocolInfo, authenticated DescribeMachine, and the
Creator bootstrap Control ClaimAck. Control does not implement the separate old
desired-state/snapshot owner protocol: a subsequent such frame refuses that stream
as unsupported. A completed/disconnected observer never cancels a durable run.

`MachineBackend` is a typed adapter into the single execution engine. Supported
run methods validate their direct or nested Claim before invoking that backend.
Every backend operation receives `VerifiedActor`, derived from the verified
Ed25519 public key. Claim's caller-selected `record_owner_id` is not signed by
ClaimProof and is never resource authority. Upload operations are owner-key-scoped.
It owns no scheduler/journal/store; unsupported backend operations return
UNIMPLEMENTED for that operation. Blocking storage operations run outside Tokio's
network threads. Mutation completion is independent of observer liveness. A future
long-poll backend needs cancellation-aware observation; an unbounded blocking wait
must not consume permanent executor threads after the observer disconnects.

Identity and owner keys are explicit startup configuration. Each fixture boot has
a fresh boot identity and self-signed leaf; the readiness envelope authenticates
that exact leaf with the existing `cozy.pod-readiness/1\0` HMAC domain. This slice
does not yet implement retained Hub receipts/grants or live authorization-key
rotation and cannot be substituted into a rental's provisioning/update lifecycle.

The standalone `front-door` binary is an inspection fixture: it advertises no run
capabilities and refuses execution. Production CPU runs use an engine adapter, not
the fixture. A service caller must finish durable engine readiness before calling
`api::serve`. Every missing package/publication/browser operation remains explicit.

## Reproduce the isolated consumer gate

```sh
cargo test --locked -j 2 --lib
cargo build --locked -j 2 --bin front-door
cd tests/front_door_client
GOPRIVATE=github.com/cozy-creator/* GONOSUMDB=github.com/cozy-creator/* go mod download
go run -p 2 . --machine ../../target/debug/front-door --output /path/to/owned/evidence
```

The Go gate uses the deployed generated schema and actual pinned TLS connection,
Control bootstrap, canonical signed claims, readiness HMAC and process inspections.
It checks old/current/future claim versions, unknown protobuf fields, signatures and
identity fencing, unsupported-operation isolation, and no CUDA/NVML/device access.
It creates and tears down its own service; it never selects the user's machine,
changes COZY_HOME, rents a pod or reads historical daemon results.

The expanded run passed all 12 checks, including wrong-pin rejection and GPU
descriptor inspection. The 2 Rust auth tests pass, including the deployed golden
ClaimProof document. The latest evidence belongs under
`outputs/cozy-machine-continued-20261002/front-door-gate/`.

This is a real transport/auth gate, not an ordinary `cozy run` or inference gate.
Full CLI execution still requires engine-backed PrepareLocalPackage/PackageSet,
workspace identity, captured offers, native input custody, long-poll execution
events, products/outcomes and output collection backed by the execution engine.
Hub/browser/installer/update and old binary consumers remain separate gates.

## Source ingress and real Creator capture

`WorkspaceUploads` implements the deployed LocalPackageUpload header/chunk stream,
with a durable prefix acknowledged after each bounded chunk. It retains prefixes
across disconnect/restart, refuses a changed known identity under the same owner
operation, and serializes concurrent writers with an actual operation file lock.
Schema transfer bounds apply to source carriers, wheels, chunk sizes and inventories.
These are control/source transport limits, not inference memory admission minima.

The standard tar crate validates source archives statically: safe unique regular
members, no links/devices/traversal, bounded expansion/extension records, and root
pyproject.toml/uv.lock. Package code is never imported or executed for this scan.
TensorFS's released source-artifact importer verifies and durably retains carriers
before VERIFIED. No store, tar format or hash implementation is copied. CAS hashes
establish custody; the source digest stays absent on the deployed wire and never
becomes a package/deployment fingerprint.

The installer receives `UploadedPackage` with typed `RootSet` metadata and verified
object descriptors. It opens bytes through TensorFS verified descriptors; no caller
path chooses machine files. RootSet compares only consumed typed metadata for an
operation replay, tolerating unknown optional fields. `prepare_local` remains
UNIMPLEMENTED until an actual installer/engine backend implements it. After durable
environment materialization, the installer can release transfer custody explicitly.
Global abandoned-upload cache pressure/reclamation still needs integration.

`tests/creator_capture_client` imports the current Creator helpers read-only:
WriteSourceArchive, LocalPackageSelection, canonical ClaimProof and UploadFile.
It transfers a 3.8 MB real source archive, interrupts after a durable 1 MiB prefix,
then uses Creator's actual pipelined uploader to resume and verify it. Replaying a
verified header works without the laptop carrier or its unsigned owner label.
Preparation correctly returns UNIMPLEMENTED from the inspection backend.

The fixture uses a temporary module configuration in the evidence directory with
a read-only Creator path replacement, preserving the source checkout. This is
consumer-source reuse for qualification, not a shipped local-path dependency:

```sh
cd tests/creator_capture_client
cp go.mod /path/to/evidence/gate.mod
go mod edit -modfile /path/to/evidence/gate.mod \
  -replace github.com/cozy-creator/cozy=/path/to/readonly/cozy-creator
GOPRIVATE=github.com/cozy-creator/* GONOSUMDB=github.com/cozy-creator/* \
  go mod tidy -modfile /path/to/evidence/gate.mod
go run -p 2 -modfile /path/to/evidence/gate.mod . \
  --machine ../../target/debug/front-door --output /path/to/owned/evidence
```

Evidence: `outputs/cozy-machine-continued-20261002/creator-capture-gate/results.json`;
read-only Creator source head `84830df01e132e9197346e1c62cba2536f54989f`.
The separate real filesystem/TensorFS tests cover restart/resume, owner isolation,
native verification, archive safety, checksum refusal and installer cleanup. These
qualify source intake; they do not establish installation or inference correctness.

## Retained identity and captured installation

`MachineConfig::load` and `MachineIdentity::retained` retain one P256 certificate,
private key and worker ID. A fresh UUID fences each process boot. Configured
authorized-key JSON and readiness-secret JSON are explicit files; identity/key/HMAC
secrets require owned regular mode0600 files and mode0700 identity directories.
Existing invalid or mismatched identities refuse startup without replacement.
Key-file changes take effect at an explicit restart, preserving the pinned leaf.
No ownership transfer between different keys is implicit.

The real Go `--retained-restart` gate reconnects the same connection through four
owned service processes under its original exact leaf pin. It proves fresh boots,
stale-claim rejection, native TensorFS custody surviving restart/key refresh, and
revoked/new key behavior. It does not prove automatic boot-claim refresh by an
existing Creator daemon, rental enrollment or execution-journal recovery.

`api::install::prepare_uploaded` materializes verified descriptors into one owned
stage and calls the trusted Python installer. Its `PreparedGeneration` returns the
native 32-hex generation, measured dependencies, exact native PackageInterface bytes
and a held generation fd. The service must commit actor/installation alias mappings
in its sole journal before acknowledgement; this helper owns no alias registry.

Source captures use standard `uv sync --frozen --no-editable` against their uv.lock.
Wheel captures use actual wheel bytes plus standard hashed exported requirements.
The private runner wheel is resolved under constraints for every already-selected
dependency; it cannot silently change the captured SDK or another dependency.
Source without a lock uses genuine declared bounds. Mixing a frozen source closure
with an independent wheel/requirements closure is explicitly unsupported. Configured
or captured numeric Python selectors are resolved through standard uv; metadata and
installed-wheel description reuse the selected package SDK's source-only readers.
The installer helper needs its explicit installer dependencies, not a Runtime SDK.
Description bootstraps the held environment with `-I -S`, adding owned site-packages
without executing authored `.pth` files. Installation/building remains a distinct
authorized packaging operation. Typed failure records preserve unsupported versus
dependency-resolution errors across the Python boundary.

Real tests preserve locked Runtime 0.18.89 and run the sklearn classifier for both
source and hashed-wheel installations. The separate Rust component gate covers
TensorFS upload/custody → verified fd materialization → trusted installer → native
generation/interface → real inference and saved matching outputs. It is component
qualification, not ordinary Creator CLI or full API replacement qualification.
Evidence is under `outputs/cozy-machine-continued-20261002/captured-installer-rust-gate/`.

The public byte stream uses an iterator and a two-chunk channel with backpressure;
production backends provide bounded fd reads instead of collecting output bytes.
`events_observed` supplies an observation-only cancellation flag so an abandoned
long poll can leave its wait without controlling the durable execution. A backend
must actually observe that flag (or bound observation pages) to reclaim its waiter.

## Integrated ordinary CLI CPU gate

The integrated service at `e7750e8` now passes an opt-in ordinary Creator CPU path.
The Creator endpoint adapter is draft PR #994, head `20c6f22d`; it uses the normal
default-home controller key, request records, watch/cancel and output export. The
explicit endpoint file is pinned to the worker, TLS leaf, current boot and workspace.
It does not replace the default daemon, enroll a rental or implicitly rebind identity.
An older daemon without the endpoint capability is refused before submission.

Three consecutive fresh SDK99 sklearn application runs completed with predictions
`[0,2,1]`, matching independently read 244-byte JSON outputs and correct MIME/digest
metadata. Fresh completion and explicit cancellation also pass on the final service:
collection is acknowledged, pending cancellation clears, and the final request-release
event appears exactly once with a stable replay cursor. Native output custody remains
under its separate explicit source-release authority; a collection acknowledgement
does not delete it. This accepted CPU path has no asset inputs, so the final event
does not claim general asset-input or shared-weight reclamation.

The consumer gate found and fixed two real skew bugs: equivalent JSON number/formatting
forms are now accepted without serialized-byte identity, and the older Creator may omit
optional acknowledgement boot metadata. Supplied boot metadata, signatures and actual
outcome/digest identities are still checked. Duplicate JSON keys are refused.

Observer process loss left accepted inference progressing; later explicit cancellation
recorded its actor and terminal outcome. A separate graceful SIGINT trial reached completion
before it could prove early detach, and is retained as diagnostic evidence only. Wrong
TLS pin, worker and workspace selections refused before creating request records.

Evidence: `outputs/cozy-machine-continued-20261002/ordinary-cli-cpu-fixture/ORDINARY-CLI.md`
and the sibling JSON/event/output artifacts. Issue #3 stays open for complete public-service
crash/OS-supervisor boundaries, package types and delivery consumers. Automatic receipt
refresh, Hub packages, jobs, browser media and GPU execution through this API remain
unqualified. The earlier sections describe component fixtures, not these integrated gates.
