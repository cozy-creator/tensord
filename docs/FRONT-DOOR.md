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
It owns no scheduler/journal/store; unsupported backend operations return
UNIMPLEMENTED for that operation. Blocking storage operations run outside Tokio's
network threads. Mutation completion is independent of observer liveness. A future
long-poll backend needs cancellation-aware observation; an unbounded blocking wait
must not consume permanent executor threads after the observer disconnects.

Identity and owner keys are explicit startup configuration. Each fixture boot has
a fresh boot identity and self-signed leaf; the readiness envelope authenticates
that exact leaf with the existing `cozy.pod-readiness/1\0` HMAC domain. This slice
does not yet implement retained receipts/identity, Hub grants or authorization-key
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
Full CLI intake still requires LocalPackageUpload, PrepareLocalPackage/PackageSet,
workspace identity, captured offers, native input custody, long-poll execution
events, products/outcomes and output collection backed by the execution engine.
Hub/browser/installer/update and old binary consumers remain separate gates.
