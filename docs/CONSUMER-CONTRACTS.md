# Native consumer contracts

Owner: `/root/machine_front_door`; scope: issue #3 public consumer adapters.
Worktree: `~/cozy/.worktrees/cozy-machine/3-consumer-contracts-20261002`.
Branch: `feat/3-consumer-contracts-20261002`; fetched base:
`9ef3799b6a96e719f29f6a94ff9e9eff093153fe`, plus reviewed API/install checkpoints.

This slice owns typed backend hooks and authenticated transport dispatch for
description/inventory, native custody, input streaming and observation. The service
owner supplies actual journal/store callbacks. Missing implementation remains an
operation-local UNIMPLEMENTED response. Fixture transport success never claims
ordinary CLI or real inference qualification.

Implemented hooks: `describe_runtime`, `list_packages`, `list_models`, `retain_bytes`,
`release_bytes`, `begin_input_tree`, and `list_observed`. Runtime observations fill
the existing typed descriptor; service version is real and package SDK rows come
from actual held manifests. No Python version or legacy Runtime version is fabricated.
Native inputs authenticate their one header before resource access, forward bounded
1 MiB blobs sequentially to the store owner, and require an explicit final commit.
No whole input is assembled by the network layer. Interrupted sessions are dropped
without a run-control operation; store callbacks own native commit/abort durability.

Events/list requests normalize deployed page limits before callbacks. Observation
drop flags are separate from durable execution control. Byte output uses bounded
backpressure and refuses a backend's empty or oversized output chunk. Backends must
honor observation cancellation or bounded pages to reclaim abandoned waits.

`tests/consumer_transport.rs` passed three real TLS/socket tests for authentication,
typed dispatch, description/inventory, page bounds and input-frame semantics. Its
recording callback deliberately returns UNIMPLEMENTED for custody/commit; this is
transport proof, not native custody or consumer readiness. Actual Creator capture,
retention and readback against the service's ready native callbacks remain required.
