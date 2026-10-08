# Resident custody (Degree 2)

`resident_custody.rs` lets TensorD retain **exported GPU allocation handles** across executor
replacement. Runtime constructs the model and uses its local TensorFS plane to allocate and
fill GPU regions, then exports their VMM handles (`Plane.export_device`). TensorD holds the
offered fds as references to those allocations and coordinates retention/eviction; it neither
copies tensor payloads to the GPU nor creates a CUDA context.

Later Runtime executors get duplicated handles and map the allocations read-only using
their own TensorFS plane (`Plane.attach_device`). A GPU mapping is distinct from the shared
CPU buffer that originally supplied its bytes, and both are distinct from the stored artifact.
Rental proof that a CUDA-free holder keeps and frees the bytes:
`outputs/cozy-machine-takeover-20261002/C/FEASIBILITY.md`.

## Model

- A holding is one weight-set layout/variant on one GPU: `HoldingKey {device (GPU UUID), layout
  (plane digest), variant (empty or the baked LoRA delta identity)}`, a generation, regions
  (chunk sizes), one fd per chunk, readers, phase `Ready` or
  `Revoking`. Every executor on the pod whose layout matches shares it, whoever submitted the
  request and whichever package it runs.
- A reader is a **lease**: one end of a socket pair the executor keeps while it maps the holding
  (sent after the chunk fds, `lease: true`). Its close ends the lease with no message: the executor
  closes it once Runtime has unmapped and released its regions (revoke, close, a duplicate replaced), and
  the kernel closes it when the process dies. The reader's birth and pidfd remain the machine's
  own observation of death.
- Bytes (Σ chunk sizes) are charged once per GPU from the offer until the holding is let go.

## Rules

- `offer`: validated (64-hex layout, 2 MiB chunks, one character-device fd per chunk). Regions a
  Ready holding lacks extend it; an offer overlapping held regions is `Duplicate` (its fds close;
  the executor attaches the held one). A kept offer makes the offerer a reader (a lease).
- `attach`: Ready only; every attachment is its own lease. Answers name the holding's layout
  digest; the reader's plane maps only when it is the weight set's own.
- `begin_revoke(key, generation)`: TensorD permits no new attachments. Each reader gets `revoke` at its next idle
  boundary (`GpuPool::revoke`, or the end of the running call). A refusal keeps its lease charged
  until the process ends: no kill.
- `collect`: a lease whose connection closed, or whose pidfd ended with its birth gone, is dropped
  (the driver tore its references down); a Revoking holding with no leases is removed and its fds
  closed.
- Read-only is the reader's own mapping and cannot be imposed by the holder: CUDA exports take no
  access flags and any fd holder can map writable (rental probe, `C/SAFEGUARDS.md`). So a holding
  is as trustworthy as every package on the pod: one that writes into a mapping, or offers wrong
  bytes under a layout, reaches every package sharing it. A pod has one owner, who chooses its
  packages, and nothing else separates them either. A pod-level policy isolating publishers would
  add its domain to `HoldingKey`; the package is on the plan (`slots[].binding.package`).
- No adoption across machine restarts: custody is in memory; a restart drops the fds and the
  executors' own references keep whatever they map.

## Wiring

`GpuPool` holds custody unless its GPU drives a display (`nvidia-smi display_active`; unknown =
off). `Load.device_weights` when the executor offers `weights.attach/1`; `share` after every
call; `GpuPool::resident()` and `GpuPool::revoke()` for the memory policy (B2).
