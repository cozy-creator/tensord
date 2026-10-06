# Matched gate: ComfyUI vs the Rust machine

`gate.py` drives ordinary `cozy run --rental=<pod> --await` from the controller (or this computer's machine) and writes
one row per request. Nothing private: the CLI, the daemon and the published packages are the ones the owner uses.

```
python3 gate.py run manifest.json OUT     # cells (and cycles) per the manifest; append OUT/results.jsonl, OUT/runs/*
python3 gate.py report OUT                # OUT/summary.json; with an `r1` key in the manifest, the R1 verdict
```

Both commands exit nonzero for failed requests, cells, setup, or preflight. `continue_on_failure`
collects further timed-cell evidence; it never makes a failed run successful. Setup and priming
failures are terminal and retain their command/request receipts before any timed requests.
`summary.pass` includes failures of either engine. The per-candidate R1 verdict is separate:
Rust completing a pool where ComfyUI runs out of memory is capacity evidence; that failed
reference has no completed timing and cannot establish a speed win.

## R1 verdict (CUTOVER.md section 4), per Rust cell

`"r1": {"candidate": "rust", "reference": "comfy", "baseline": {"cold-sdxl": 22.2, "cold-anima": 60.8}}` and
`"control": "rust"` in the manifest. A cell passes with:
- zero failed requests and no NVRM Xid (a failed cell counts; nothing is left out);
- disk reads at most 1.5× the reference engine's in the same cell on the same pod;
- a cold cell's submit → CLI exit with the image saved at most 1.10× the reference's on the same pod (server start
  included); the previous candidate's figure (`baseline`) and the pod's host load (median load1 and CPU pressure over
  the cell) are reported beside it, since a shared host slows CPU-bound startup for both engines;
- every planned candidate/reference cell is present, every candidate request has its successful
  file-hash/size and input receipt, and every low-memory request has one successful same-input
  full-card control, and the output's PSNR to it is at least 30 dB or a person has reviewed it as good.

Each low-memory (`budget`) request is run again, untimed, on the full card after every timed cell (`control`); its
output's PSNR to that control is reported, and anything under 30 dB goes to `review` and fails until a person records
it as good in `OUT/reviewed.json` (`[{"cell", "seed", "good", "by", "note"}]`). Lossy is fine if the image is good
(owner, 2026-10-04): whether the encoded bytes equal the control's is reported (`byte_identical`), not judged. Missing
controls or receipts and changed files or inputs fail qualification. This is a same-GPU comparison; it does not claim
cross-device determinism.

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

A manifest `setup`, `on_start`, then `preflight` command runs once on the host, in that order,
before sampling or requests. Use `setup` for model export/download verification rather than a
separate driver command that merely logs an error and continues. An arm's `preflight` command
runs after its restart/setup and before each cell's priming/timer; it must check the installed
package/Runtime pair after any overlays. These commands use the host shell and must return
nonzero on failure. Their outputs and failures are recorded. An arm's optional `facts` command is recorded at every restart; a manifest `collect` command's stdout
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
  `continue_on_failure` records a failed timed cell and goes on, then returns nonzero.
- a `prime` item with `warm` (e.g. `{"model": "sdxl", "warm": "imported"}`) is `cozy run <target> --warm=<level>`: the
  machine readies the function and runs nothing; a cell's `unwarm` models get `--warm=off` before anything else (a cold
  cell, or the other engine's cell, so the machine keeps nothing ready);
- a warm cell's `prime` requests (`{"model", "input"}`, e.g. one per model) run first, one at a time and outside the clock, so
  each engine's processes exist and its models are loaded when the timed requests start; an engine with its own driver gets
  the same requests through its `command`. A failed prime stops immediately, before submitting
  any timed request, even with `continue_on_failure`. Cold cells prime nothing: they measure start-up;
- an arm with `keeps_root` is never restarted (on a Rust-only image the machine is the container's main process): its
  `restart` only ends what the machine runs (its executors) and a cold cell is clocked from the submit; an arm's `before`
  command runs before each of its cells (e.g. the ComfyUI arm frees the card of the machine's executors);
- a cell's `setup` command runs first (e.g. that cell's host limits, in place before the machine starts);
- an arm's `cache_list` command prints the files to read into the page cache, where `cache_paths` (whole directories)
  would be too much (a shared store);
- an arm with `cold_by_run` starts a cold cell from a stopped machine: the first `cozy run` starts it, and the total
  runs from that submit (the rebench's cold start on this computer).

Local mode counts only the executors in the arm's own cgroup, and the 1 Hz samples are fsynced, so a computer that
freezes keeps its last second.

For Anima, the ordinary untimed prime exercises the actual package construction and Runtime plan
after overlays. An incompatible processor contract such as `anima_optimization_block_shape`
therefore stops the warm cell before timing. Run the cohort's untimed install/construction
checks before the cold-cell sequence too, then restart normally for the cold measurement.
Do not reset attention processors or alter the request to bypass a failed check. The separate
product decision is in [benchmark validation](../../docs/gate-validation.md).

## Manifest

`rental`, `hub`, `salt` (new per run), `targets`, `requests` (fixed fields; prompt and seed added per
request), `shapes`, `warm`, `cache_paths`, `start_temp_c`, `order`, and per arm two shell commands run
as root on the pod: `root` prints the arm's live machine root PID, `restart` stops whatever machine runs
and makes this arm the next one (returns once the old root exited).
