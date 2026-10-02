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
  readers (exact process birth + pidfd), phase `Ready` or `Revoking`.
- Bytes (Σ chunk sizes) are charged once per GPU from the offer until the holding is let go.

## Rules

- `offer`: validated (64-hex layout, 2 MiB chunks, one character-device fd per chunk). A second
  offer of a Ready layout is `Duplicate` (its fds close; the executor attaches the held one).
- `attach`: Ready only; records the reader.
- `begin_revoke(key, generation)`: no new attachments. Each reader gets `revoke` at its next idle
  boundary (`GpuPool::revoke`, or the end of the running call). A refusal keeps its lease charged
  until the process ends: no kill.
- `collect`: readers whose pidfd ended and whose birth is gone are dropped (the driver tore their
  references down); a Revoking holding with no readers is removed and its fds closed.
- No adoption across machine restarts: custody is in memory; a restart drops the fds and the
  executors' own references keep whatever they map.

## Wiring

`GpuPool` holds custody unless its GPU drives a display (`nvidia-smi display_active`; unknown =
off). `Load.device_weights` when the executor offers `weights.attach/1`; `share` after every
call; `GpuPool::resident()` and `GpuPool::revoke()` for the memory policy (B2).
