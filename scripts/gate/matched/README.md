# Controller saved-output adapters

These adapters produce new strict #35 rows. They do not start servers, rent, kill
processes, change requests or infer missing evidence from old smoke results.

`run.py MANIFEST OUT --engine cozy --cell CELL --pair PAIR` uses ordinary
default-home CLI with the manifest's existing rental selector. The `comfy` arm uses
stock `/prompt`, `/history` and `/view`. Both record the actual first submit and
close/fsync every output on the controller before stopping their clock. Image
decode, hashes and quality comparison occur after the timed window.

The manifest declares:

- `arms.cozy`: `name: cozy-machine`, `cli`, `selector_args` (existing rental and
  Hub), exact `targets` and `target_args`, and `commit_probe` command returning
  the rented Rust binary's JSON version.
- `arms.comfy`: `name: ComfyUI`, `base_url`, `client_id`, `commit_probe` returning
  that server's actual Git commit, `model_files` per model (denoiser, encoders,
  VAE), and `scheduler` per model proved below.
- `hardware_probe`: command returning actual remote `gpu_uuid`, `driver`,
  `total_bytes` and `remote: true` for this owned rental.
- `cells`: the six literal authored requests per cell. A reference cell's
  `budget` must be `unconstrained`; a pressure cell must retain the same requests.
- `parity`: measured proof JSON per model; `controls`: assembled reference JSON
  per engine/cell; `quality`: `method: exact`, or predeclared
  `method: declared_tolerance` and `minimum_psnr_db: 35` or higher.

Each parity proof contains actual logical tensor `name`, `shape`, `dtype`, `bytes`
and raw-value SHA256 inventories for both engines (`provenance.py` reads actual
safetensors), actual `sigmas.cozy`/`sigmas.comfy` arrays and a declared absolute
tolerance no larger than 1e-5, `prediction`, `comfy_scheduler`, and reviewed
`source_equations` plus `conditioning_evidence` naming actual source/config and
precision evidence. Hashing a proof is provenance, not proof that its assertions
are true: review the cited sources and inventories before qualification.

`schedules.py --comfy REPO --commit PIN --config ACTUAL_ANIMA_CONFIG --steps 30
--out schedule.json` extracts and executes the actual pinned scheduler symbols
with installed Diffusers. On the checked local source, normal differs by only
1.19e-7; inherited simple differs by 0.08658. This is schedule evidence only.
Published package source/config, exported model tensors, conditioning and decode
precision still require their own checks. `--model sdxl` compares its actual
EulerDiscrete config: the verified paul/sdxl1.0 header uses leading/offset1,
and Comfy ddim_uniform matches within4.77e-6 while normal/simple differ3.5863.
Unproved configs refuse a match; they are not filled from remembered defaults.

`native-source/` is an authored CPU package for supported model metadata custody:
ordinary `cozy package install ./native-source`, then
`cozy run local/codex-memory-metadata/metadata source.model=paul/sdxl@1.0.0
--rental=OWNED --await --json`. It calls `ctx.tensorfs_source(source).inspect()`
under the admitted model capability and returns exact source/config hashes and
native tensor/part geometry. It constructs no model, opens no Store, and is not
an inference/quality test. Source tensor export should use bounded
`capability.read_part_into`, retaining encoded/logical metadata; it must not assume
all quantized parts are dense logical values or read a foreign Store directly.

Collect same-engine unconstrained controls first with `--reference-only`. Repeat
the exact six requests three times, then run `controls.py ref1/result.json
ref2/result.json ref3/result.json --out controls.json`. Cozy must report current
denoise work through the final authored step (early observer gaps are retained);
Comfy must execute its sampler rather than return a cached
sampler. Cached encoders/loaders remain visible and are allowed in a warm arm.
For scored pairs, distinct predeclared prompts/seeds may avoid whole-sampler
cache, but each pair must match across engines and keep the authored geometry and
step count. The original 48 reliability requests remain unchanged.

The quality checker requires same-engine/model/request/hardware identity and
actual reference files. Exact equality is the default. A declared PSNR threshold
requires baseline repetition variance to be at least 6 dB better than that rule;
failure leaves the cell unqualified. This stronger benchmark-control rule is
separate from diagnostic reliability comparisons using one reference per request
and representative repeated controls. Cross-engine RNG differs and pixel equality
across engines is never claimed.

Feed all fresh `result.json` rows to the strict #35 reporter. Three complete paired
measurements in every declared cell, zero failures and upper95% median ratio below
1 in every cell are required. No smoke image or this harness's CPU checks establish
a Comfy win.
