# Authenticated CPU front door

Owner: `/root/machine_front_door`; issue: cozy-machine #3/#4.
Branch: `feat/3-front-door-20261002`; base: `9ef3799` (fetched origin/master).
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
