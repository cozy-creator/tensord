# Resident custody (Degree 2)

`resident_custody.rs` keeps GPU weight regions alive across executors without CUDA in the
machine. An executor fills plane regions in its own context, exports their VMM handles
(TensorFS `Plane.export_device`) and offers the fds; the fds here are the allocation's
references. Later executors on the GPU get duplicates and map them read-only
(`Plane.attach_device`). Rental proof that a CUDA-free holder keeps and frees the bytes:
`outputs/cozy-machine-takeover-20261002/C/FEASIBILITY.md`.

## Model

- A holding is one weight-set layout on one GPU for one actor: `HoldingKey {actor, device
  (GPU UUID), layout (plane digest)}`, a generation, regions (chunk sizes), one fd per chunk,
  readers, phase `Ready` or `Revoking`.
- A reader is a **lease**: one end of a socket pair the executor keeps while it maps the holding
  (sent after the chunk fds, `lease: true`). Its close ends the lease with no message: the executor
  closes it once its regions are unmapped and released (revoke, close, a duplicate replaced), and
  the kernel closes it when the process dies. The reader's birth and pidfd remain the machine's
  own observation of death.
- Bytes (Σ chunk sizes) are charged once per GPU from the offer until the holding is let go.

## Rules

- `offer`: validated (64-hex layout, 2 MiB chunks, one character-device fd per chunk). Regions a
  Ready holding lacks extend it; an offer overlapping held regions is `Duplicate` (its fds close;
  the executor attaches the held one). A kept offer makes the offerer a reader (a lease).
- `attach`: Ready only; every attachment is its own lease. Answers name the holding's layout
  digest; the reader's plane maps only when it is the weight set's own.
- `begin_revoke(key, generation)`: no new attachments. Each reader gets `revoke` at its next idle
  boundary (`GpuPool::revoke`, or the end of the running call). A refusal keeps its lease charged
  until the process ends: no kill.
- `collect`: a lease whose connection closed, or whose pidfd ended with its birth gone, is dropped
  (the driver tore its references down); a Revoking holding with no leases is removed and its fds
  closed.
- Read-only is the reader's own mapping and cannot be imposed by the holder: CUDA exports take no
  access flags and any fd holder can map writable (rental probe, `C/SAFEGUARDS.md`). Holdings are
  shared only within one actor.
- No adoption across machine restarts: custody is in memory; a restart drops the fds and the
  executors' own references keep whatever they map.

## Wiring

`GpuPool` holds custody unless its GPU drives a display (`nvidia-smi display_active`; unknown =
off). `Load.device_weights` when the executor offers `weights.attach/1`; `share` after every
call; `GpuPool::resident()` and `GpuPool::revoke()` for the memory policy (B2).
