# Matched gate: old stack vs Rust machine

`gate.py` drives ordinary `cozy run --rental=<pod> --await` from the controller against one rented
pod, alternating two machine arms on the same GPU, and writes one row per request. Nothing private:
the CLI, the daemon and the pod's published packages are the ones the owner uses.

```
python3 gate.py run manifest.json OUT     # alternate arms; append OUT/results.jsonl, OUT/runs/*
python3 gate.py report OUT                # OUT/summary.json with medians, spread and the verdict
```

## Threshold, declared before the run

The Rust machine passes when, with every output correct, at least one of these holds and warm
per-image is no more than 2% slower (median):

- stopped-machine first image (warm page cache) median at least 20% lower, or
- SDXL→Anima→SDXL switch (sum of the two switch requests) median at least 20% lower, or
- host peak at least 25% lower.

Reported whichever way it lands. Gains that come from shared Runtime/TensorFS fixes both arms run are
not Rust-machine gains: every row records the Runtime/TensorFS versions the run reports.

## One cycle (per arm, order from the manifest, e.g. old, rust, old, rust, ...)

Before the first cycle, each arm in `prime` runs one SDXL and one Anima request, unmeasured, to pay
its package install and model download.

1. Wait until the GPU is at or below `start_temp_c`, or has stopped cooling for 60 s (a card holding
   a CUDA context has an idle floor: an A40 sits at 53–56 °C). Start temperature is recorded per cycle.
2. `cold_first`: restart the arm's machine; once every executor of the previous machine is gone,
   evict `cache_paths` from the page cache (`POSIX_FADV_DONTNEED`); submit SDXL as soon as the new
   machine's root process exists. Wall = new root's birth (from `/proc/<pid>/stat`) to verified image.
3. `warm_first`: same, but read `cache_paths` into the page cache instead.
4. `warm` ×N: SDXL on the warm machine.
5. `to_anima`, then `to_sdxl`: the model switch and back.
6. `kill_next`: SIGKILL every executor (exact PIDs, listed in the row), then SDXL.

`kill_proof: {"arm": "rust", "n": 20}` adds, after the cycles, N times: SIGKILL every executor of the idle
arm and submit SDXL at once, one attempt each (a failure counts; no retry).

With `first_images: "alternate"` a cycle has one restart (cold first image in cycles 0–1, 4–5, …,
warm in 2–3, 6–7, …), so with alternating arms every restart is a switch to the other arm.

Every request has a never-used prompt and seed. Each image is decoded, checked for the declared size,
checked not flat (per-channel stddev > 2) and hashed.

## Measurements

- Wall (the gate's number): new root's birth, or submit, to the image saved and verified on the
  controller. The pod-clock birth is converted with an ssh round-trip offset (error ≤ rtt/2).
- `machine_s`: the same start to the machine's `machine.outcome` event, both on the pod clock.
- `result_lag_s`: pod outcome to the client noticing it. The controller's daemon is shared with other
  sessions, so it stalls at times (up to ~47 s seen); this separates that from either arm's machine.
- Stages: `cozy run show --json --full` per request (executor boot, load, denoise steps, decode).
- 1 Hz pod sampler: `nvidia-smi` (memory, temperature, SM clock, power, throttle reasons) and the
  container cgroup (`memory.stat`, `io.stat`).
- Host peak = max over the cycle of cgroup `anon + shmem`: every resident, unreclaimable page charged
  once, including memfd host tiers held only by descriptor. Per-process PSS is not readable on a rental
  (root lacks CAP_SYS_PTRACE and the machine processes are non-dumpable); cgroup charge is the
  whole-tree equivalent and is the same instrument for both arms.
- Disk reads: cgroup `io.stat` rbytes delta around each request (proves cold vs warm).

## Local mode (the owner's laptop)

Without `rental`, the machine runs on this computer: `work_dir` holds the helper, each arm may name
`cli`, `run_args` and `show_args` (e.g. a pinned `--machine-endpoint-file`), `cgroup` (the arm's unit,
followed by the sampler) and `limits` (recorded at each restart). `soak_s` adds equal GPU idle before
each cycle and `max_load` gates on the 1-minute load. An arm that is not relaunched by its platform
names `start` (run after the previous machine stopped and the cache step) and `after_start` (e.g. apply
and record limits). `first_images: "warm"` restarts once per cycle with a warm page cache; `kill: false`
leaves out the executor kill (fault injection stays on rentals). Disk reads fall back to the unit's
processes' `/proc/<pid>/io` when its cgroup has no io controller. `xid_watch: true` follows `journalctl -kf`; any `NVRM: Xid` runs `on_xid` (stop both
machines) and stops the harness at once.

An arm's optional `facts` command is recorded at every restart; a manifest `collect` command's stdout
(a tarball) is saved as `OUT/collect.tgz` at the end.

## Manifest

`rental`, `hub`, `salt` (new per run), `targets`, `requests` (fixed fields; prompt and seed added per
request), `shapes`, `warm`, `cache_paths`, `start_temp_c`, `order`, and per arm two shell commands run
as root on the pod: `root` prints the arm's live machine root PID, `restart` stops whatever machine runs
and makes this arm the next one (returns once the old root exited).
