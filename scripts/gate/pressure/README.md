# Fresh physical pool qualification

This lane owns a new 3070 rental for each of 3, 1.5, 1.25 and 1 GiB available
VRAM. Start `holder.py` before the first model preparation on that rental. Keep
that exact PID and birth receipt across the grouped and alternating six-request
cells. Never adjust its allocation, reuse an executor from a larger pool, or
change an authored model request. All 48 original requests remain 1024 square,
SDXL 20 steps and Anima 30 steps.

Root supplies the prepared image cohort, ordinary default-home CLI, allocation
and provider receipt. Source/wheel identity checks belong to this experiment's
artifact custody; normal compatible peers do not require matching deployment
SHAs. `holder.py` rejects an existing CUDA process instead of trying to reclaim
it. `probe.py` proves the fixed holder is alive on the same actual device and
retains its physical allocation before/after every request. It performs no CUDA
work. Host memory and GPU usage are recorded independently.

For a newly allocated pool, using unique owned paths:

```bash
/opt/cozy/python/bin/python holder.py --gib 3 --rental OWNED_NAME \
  --rental-id OWNED_ID --operation OWNED_KEY \
  --receipt /root/OWNED/holder.json --release /root/OWNED/release.json
/opt/cozy/python/bin/python probe.py --receipt /root/OWNED/holder.json
```

The controller runs each literal original cell with ordinary `cozy run --rental
OWNED --await --json --out CONTROLLER_PATH`. Capture live native loads/invokes,
holder identity, the actual final denoise step, output geometry and image quality
against the preserved unconstrained reference. A single historical reference and
representative repeat controls establish diagnostic reliability limits. A strict
matched speed result additionally needs exact actual source tensors, measured
schedulers, conditioning/precision parity, three repeated controls per request
per engine, and three complete paired timings in every declared cell.

Observer disconnect, terminal loss and signals cannot end durable work. These
scripts never cancel a run, stop the machine, or end a rental. Root ends an owned
rental explicitly after terminal status and saved evidence, or under a separate
explicit abandonment decision. Release the holder only after both cells become
terminal by writing the exact JSON `{ "operation": "OWNED_KEY", "rental_id":
"OWNED_ID" }` to its new release path; then archive the release receipt and the
provider's absence/ended evidence. Keepalive runs only during that active purpose.
