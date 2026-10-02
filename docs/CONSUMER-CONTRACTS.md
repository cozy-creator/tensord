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
