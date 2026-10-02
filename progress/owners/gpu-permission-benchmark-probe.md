# GPU permission and matched benchmark qualification

Owner: durable_cpu_engine. Branch: feat/gpu-permission-benchmark-probe-20261002.
Fetched origin/master: 9ef3799b6a96e719f29f6a94ff9e9eff093153fe.
Explicit integrated prerequisite: 6ccf6d3b200e996f00f1eb66b353216c66e09f8b.
Purpose: independent ordinary-CLI GPU qualification review, portable actual-process
CPU permission probe, and matched benchmark manifest/harness. Own only new
scripts and documentation. No service, lifecycle, API, scheduler or storage changes.

No GPU use, rentals, default controller changes, primary edits, or inferred serving
UID. GPU qualification remains a later root-owned headless gate. Existing private
source-mode measurements are evidence for that seam, not ordinary CLI acceptance.

## Stopped checkpoint

2026-10-02: user transferred implementation to Opus; stop new work. Preserve this
branch/worktree and all fixture bytes in ignored `target/permission-probes/`.
Draft PR: https://github.com/cozy-creator/cozy-machine/pull/23.

`permission_probe.py` compiled and ran with installed Runtime100/TensorFS91 in
owned no-network/no-GPU Docker fixtures (2 GiB RAM, two CPUs). Same UID0 reports
PDEATHSIG9 before a foreign fsuid transition, then0. Explicit foreign UID/GID12345
reports PDEATHSIG0 after the stock trampoline drop; inherited/fresh generation
hold, socket peer identity and spool succeed. Private journal/native Store deny.
No GPU device or CUDA/NVML library was present. UID12345 is a synthetic probe,
not measured old serving identity. Native store writes are restricted to these
new fixtures. Existing environments and generation holds were mounted readonly.
The initial container failed because the uv interpreter alias was not mounted;
that failure and a layout diagnosis are preserved. The corrected mount maps the
resolved interpreter tree at the venv's actual symlink name.

`ordinary_switch.py` is an unrun draft: no GPU/CLI benchmark was dispatched.
It captures ordinary default-home submit/watch/show, durable receipt, artifact
QA and process topology; model-ready/source-phase facts remain explicitly absent.
Before use, add exclusive per-arm execution and finish the matched manifest;
both retained architectures must not compete on the display GPU. It requires
measured serving UID, equal initial host budgets and external operator policy
evidence. Its reserve is a benchmark precondition, never a production size floor.
No accepted .101/.93 followup fixture was prepared before the stop instruction.

Initial host-budget reuse source: Runtime branch
`origin/feat/machine-host-tier-registration-20261002`, executor_commands.py:227
adds `initial_host_budget_bytes`; executor.py:1993-1996 calls existing
`weights.set_budget(-1, host_budget)` before PlaneBackend/registration; recursive
Load propagation is at2369-2375. Current accepted .93 already exposes
`Plane.set_pinned_budget`, so this narrow negotiation/apply-before-Load behavior
does not require restoring removed .92 CUDA/disk-host APIs. Root owns changes.
