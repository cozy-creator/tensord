# Matched GPU matrix

Tracker #319/#324. Root owns cohort, server, ballast and rental lifecycle. The
benchmark owner stages source and request artifacts before a purpose is handed
off. No current rental has been reassigned and no measured timing exists yet.

The planned GPU classes cover three architectures. Actual GPU UUID, edition,
driver, VRAM, provider, CPU/cgroup capacity and storage must be recorded before
every cohort. Availability and cost remain acquisition evidence; these names do
not authorize a substitution after a failed cell.

| GPU | Architecture | Memory class |
| --- | --- | --- |
| RTX 3070 | Ampere | 8 GiB |
| RTX 4090 | Ada Lovelace | 24 GiB |
| RTX PRO 6000 Blackwell | Blackwell, edition recorded | 96 GiB |

[NVIDIA RTX 4090 specifications](https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/)
and [NVIDIA RTX PRO 6000 specifications](https://www.nvidia.com/en-au/data-center/rtx-pro-6000-blackwell-server-edition/)
support the planned classes. Rental hardware measurements are authoritative for
the actual cohort. RunPod is available; provider-specific qualification on vast.ai
still requires the account access already requested by root. H3, two-GPU groups,
LoRA and lifecycle faults retain their separate runtime reliability gates.

Each architecture has the following fourteen independently reported cell types.
Every cell requires at least three complete paired measurements. Pairs alternate
engine order: Comfy/Cozy, Cozy/Comfy, Comfy/Cozy. Both engines run sequentially on
the same physical GPU and provider host, with the same submitted work. Changes
in drivers, models, software, resource ceilings or host pressure start a new cohort.

| Kind | Cell types | Requests per pair and engine |
| --- | --- | --- |
| Cold first output | SDXL; Anima | One per model, fresh executor/model residency |
| Consecutive warm | SDXL; Anima | Three fresh requests per model |
| Model switching | Grouped; alternating | Six, three SDXL and three Anima |
| Degraded VRAM | Grouped and alternating at 3, 1.5, 1.25 and 1 GiB available | All six at every budget |

Every request keeps 1024 square geometry, SDXL 20 steps/CFG 7/HiDiffusion false,
or Anima 30 steps/CFG 4.5/full CFG interval/no quality prefix/no first-block cache.
The benchmark input sets are declared before measurement with paired seeds and
reused across all three pairs. A fresh default-cache Comfy process per measured
cell prevents old sampler results from surviving across pairs. These are
new benchmark inputs; the original 48 reliability requests remain unchanged.
Neither the graph's save-file nonce nor a new prompt ID proves new sampling.

Cold means first submitted work with fresh executor/model residency and a recorded
weight-file cache condition. Server startup and environment preparation are
reported separately. Warm means unchanged software with prepared kernels and
resident or reusable model state; encoder/node cache hits remain visible.
Both arms receive the same declared OS cache condition. Each non-cold cell warms
with a separate seed outside timing, one full request per participating model.
Root records process PID/birth, readiness, preparation, temperature and file-cache
observations. Only startup before submission is excluded for both arms; every
preparation step after submission stays in the primary interval. The fixed owned
ballast allocation precedes either engine's residency in a pressure comparison.
The ordinary default-home CLI and stock Comfy HTTP endpoints both start their
clock at the first actual submission and end after every output file is closed
and fsynced on the controller. Quality and hashes run outside that window.

Before any scored request, require the complete logical tensor value inventory,
literal token/conditioning inputs, actual sigma grids, initial noise scaling,
and observed precision for the exact candidate packages. Stock Comfy is pinned
to 20ca544ee0436721d8eb5f544665e490609f72c8 with Torch 2.14. Candidate package
policy is staged in packages PR390, SDXL 2.7.0/Anima 0.3.1; it has CPU proof
only. Freeze the actual installed source, Rust/SDK/plane artifacts and execution
receipts before references. Source stamps identify evidence, never compatibility
or admission authority.

Each exact request needs three unconstrained same-engine reference outputs on
the same GPU/software. Reference Comfy work uses its stock `--cache-none` mode;
root restarts into its normal cache configuration before scored work and warms
with separate inputs. References do not contribute to timing. A tolerance of
at least 35 dB requires unconstrained repetition variance at least 6 dB better.
Actual decode strategies may differ under pressure only when this same quality
rule passes for both engines. A nonflat output is only a smoke check.

Keep every failed request and cell. Unknown submission state leaves later requests
explicitly unsubmitted until root resolves the durable run; the adapter cannot
cancel or resubmit it. A missing cell, failed request, cached sampler, failed
quality check or an upper 95% paired median ratio overlapping parity cannot
produce a speed win. Physical failures may establish a reliability difference,
but have no finite comparative completion time and do not qualify a speed claim.

This plan contains 42 GPU/cell combinations and at least 126 complete pairs.
The staged minimal request sets contain 1,224 scored image requests plus 1,224
independent reference requests, with 396 full-size setup warmups. Root should estimate actual provider spend from
the first complete cohort before acquiring the remaining classes. Source and
CPU evidence alone cannot close any GPU or speed gate.
