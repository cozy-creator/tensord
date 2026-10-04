# Matched ComfyUI and Rust machine measurements

`gate.py` drives ordinary `cozy run --rental=<pod> --await` from the controller against one rented
pod, alternating two machine arms on the same GPU, and writes one row per request. Nothing private:
the CLI, the daemon and the pod's published packages are the ones the owner uses.

```
python3 gate.py run manifest.json OUT     # alternate arms; append OUT/results.jsonl, OUT/runs/*
python3 gate.py report OUT                # OUT/summary.json with medians, spread and the verdict
```

## Qualification declared before the run

`report` evaluates a complete paired ComfyUI matrix. A lower memory peak cannot substitute
for faster inference, and failed constrained-memory cells cannot be omitted. Without the
declaration or supporting evidence the verdict is inconclusive.

```json
{"comparison":{"reference":"comfy","candidate":"rust",
 "cells":["cold-sdxl","cold-anima","warm","switch","1gib-grouped"],"min_pairs":3}}
```

Each cell row needs a stable `pair` id, equal `request_digest` for normalized model bytes
and requests, and equal measured `hardware_key`. Both arms record
`timing_boundary: "submit_to_saved_output"` and the same `output_location` (`controller`
or `pod`). Three pairs is the minimum; declare more before running when variance warrants it.
The deterministic paired bootstrap reports the median candidate/reference ratio and its
95% interval. Every declared cell must have an upper interval below one, with no failed
candidate requests, for `timing_win`.

Qualification also needs `quality: {ok, method, reference_digest}` produced by actual
reference validation: method is `exact` or `declared_tolerance`. Shape/nonflat checks are
smoke checks and do not qualify model output. `timing_win` may be true while `pass` remains
false for missing quality evidence. Existing historical datasets remain reportable but
do not acquire missing proof retroactively. Keep measured RSS distinct from PSS.

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

A manifest `on_start` command runs once on the host before the first request (e.g. watchers). An arm's optional `facts` command is recorded at every restart; a manifest `collect` command's stdout
(a tarball) is saved as `OUT/collect.tgz` at the end.

`cells` with `cell_order` ([arm, cell] pairs) run the 2026-10-01 rebench's cells before the cycles. A cell is
a list of requests (model names, or `{"model", "input"}` payloads) or `{"requests", "budget", "cold"}`:
- each cell starts with equal soak and thermal start on a freshly started, ready machine;
- each request is its own `cozy run --await`, the next submitted once the previous one holds the GPU;
- total = first submit to last saved output; `machine_total_s` is the same on the pod's clock;
- `budget` (e.g. `6GiB`) holds a ballast (`ballast.start` / `ballast.stop`) so that much GPU memory stays free;
- `cold` clocks from the machine's birth with no ready wait;
- an arm with `command` runs another engine's own driver on the host (ComfyUI) and returns the same record;
- `target_args` adds per-model arguments (e.g. `model.model=...`), `cell_facts` records a host command per cell,
  `continue_on_failure` records a failed cell and goes on.
- a cell's `setup` command runs first (e.g. that cell's host limits, in place before the machine starts);
- an arm's `cache_list` command prints the files to read into the page cache, where `cache_paths` (whole directories)
  would be too much (a shared store);
- an arm with `cold_by_run` starts a cold cell from a stopped machine: the first `cozy run` starts it, and the total
  runs from that submit (the rebench's cold start on this computer).

Local mode counts only the executors in the arm's own cgroup, and the 1 Hz samples are fsynced, so a computer that
freezes keeps its last second.

## Manifest

`rental`, `hub`, `salt` (new per run), `targets`, `requests` (fixed fields; prompt and seed added per
request), `shapes`, `warm`, `cache_paths`, `start_temp_c`, `order`, and per arm two shell commands run
as root on the pod: `root` prints the arm's live machine root PID, `restart` stops whatever machine runs
and makes this arm the next one (returns once the old root exited).
