# Isolated stock-executor SDXL hardware pilot

Root owns the non-display GPU/rental and the $20 total budget. The harness is prepared;
no GPU run has occurred in this agent. Do not run the local validation configuration on
the laptop/display GPU.

## Build and run

```sh
cargo build --release --bin device-pilot
target/release/device-pilot validate CONFIG.json
# Root only, on the selected non-display GPU:
target/release/device-pilot run CONFIG.json
```

`validate` reads configuration/interface paths and never launches Python or touches a
device. `run` uses the full stock Runtime device Executor and saves `events.jsonl`,
`load-facts.json` and `results.json`: preparation/invoke/post timings, reused executor PID,
canonical result, exact SDK-bound output checksums and typed memory/streaming observations.
Stage logs distinguish the first exit from its explicit yielded acknowledgement and retain
passes, growth, wall and stall counters. One Start/Load serves all three requests.
The trusted SDK encoder produces the actual WebP; root must independently inspect it.

This is a legacy hardware feasibility pilot. The unchanged SDK still imports TensorFS
and opens its store; host-tier retention is disabled. It does not establish sole writer,
machine-owned host/GPU weights, full API replacement or ordinary CLI compatibility.
Offered stage exchanges return the root's fixed allowance; the pilot has no scheduler.
`negotiation.json` records actual Hello capabilities. Published Runtime 0.18.99 offers only
`vacate_ranks`; it has no stage/plane flags or Budget command. The pilot keeps its legacy
residency path, using Load's authorized device limit, and sends stage/budget controls only
when that peer offers them. No SDK version comparison decides this path.

## Authoritative artifact/package inputs

Current read-only local-Hub catalog resolution: `cozy model info paul/sdxl --json` selects release
`1.0.0`, lane `bf16`, checkpoint
`sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642`.
Its complete closure is 6,939,571,699 bytes. The manifest is 164 bytes and names one
320,558-byte CozyTensors header (`sha256:f7d9f0392a00cdd4c382a610fac67d55f86b4b64d3171856026e5b1f2e8486ad`).
The native read plan covers 2,732 items and 6,937,675,734 logical weight bytes.

Use this one coherent manifest for every component:

| Component | Logical bytes |
| --- | ---: |
| text_encoder | 246,120,960 |
| text_encoder_2 | 1,389,319,680 |
| unet | 5,134,927,368 |
| vae | 167,307,726 |

Root can use the existing ordinary downloader on its owned rental:

```sh
cozy model download paul/sdxl@1.0.0 --lane bf16 --rental OWNED_RENTAL_ID --await
```

Use the unchanged published `paul/sdxl` 2.4.0 installed package environment and static
interface. Model class is `SdxlModel`, binding path `generate.models.model`, parameter
`model`, application `sdxl:app`. Source-only inspection and the cached published interface
confirm this contract. Configuration/tokenizer assets belong to the same model closure.
Do not combine arbitrary component lanes or override the package's dependency bounds.

The local CPU-only template is
`~/cozy_v2/outputs/cm-device-20261002/sdxl-pilot-local-validate.json`. Copy its shape, replacing
interpreter, static interface, store, output root/socket and device affinity with the rental's
actual paths. It intentionally names a local no-run output directory. Root supplies real
context/activation/weight/pinned allowances for its selected hardware; template numbers are
not hardware qualification or memory-size admission floors.

Requests preserve 1024×1024 (1 MP square), 20 Euler steps, CFG 7 and HiDiffusion off, with
three fresh prompts/seeds 67011–67013. The default published model scheduler determines
Euler; the harness does not copy or replace it. It never rewrites requests to fit memory.

## Comparison boundary

Run the existing ordinary Cozy CLI baseline with the same checkpoint, package, dimensions,
steps, guidance and fresh seeds. Account for old idle executors, contexts, cache state and
thermal effects before the new pilot. Report saved outputs, preparation count, executor
reuse, physical memory and utilization. A direct internal SDK pilot differs from ordinary
CLI submit-to-output, so its timings cannot by themselves claim full-stack improvement or
ComfyUI parity. Real failure/recovery and mixed-version API gates remain separate.

## Descriptor mode follow-up

`Load.descriptor_sources` defaults false and requires `model_sources.descriptors/1` in a
world-one executor. `model_source_read` exchanges manifest, role (`header`, `asset`, `object`),
name and length; replies contain verified SHA-256/length plus a readonly regular-file fd.
The SDK's separate file receiver is required: socket Handoff/Tier are different capabilities.
Headers/assets move through fds/spool, never an oversized control frame; SDK computes the
existing read plan locally. These are amortized load exchanges, not step/block RPC.
The dependency hook and native SourceDescriptor are undergoing independent qualification.

## Isolated transfer recipe

When the ordinary downloader cannot address an independently acquired pod, transfer the
native-derived closure from an already verified local store. Do not transfer its SQLite
catalog. On a new owned target, run `tfs store init TARGET` **before** copying CAS files;
the destination's catalog must be constructed from its own verified bytes.

The original published `sdxl-2.4.0-py3-none-any.whl` was recovered through the existing
package index, without rebuilding or extracting model code into this repository. Its
SHA-256 is `457f829c2170be7f5cd207d147e76578b2c41b493b4f001e3c2cc2c89967cfb3`, matching the
captured installation lock. Copy this wheel and its static `package-interface.json`.
Create a new environment, leaving the existing installation unchanged:

```sh
uv venv --python python3 /workspace/cozy-machine-pilot/sdk99
uv pip install --python /workspace/cozy-machine-pilot/sdk99/bin/python \
  --constraint /workspace/cozy-machine-pilot/stage/sdxl-sdk99-constraints.txt \
  /workspace/cozy-machine-pilot/stage/sdxl-2.4.0-py3-none-any.whl \
  'cozy-runtime[media]==0.18.99' 'tensorfs==0.3.90'
uv pip list --python /workspace/cozy-machine-pilot/sdk99/bin/python --format json
```

The constraint file pins the other 68 captured distributions, including Torch 2.14.0,
Diffusers 0.40.0 and Transformers 5.16.1. The captured local installation used Runtime
0.18.67 and TensorFS 0.3.74; those two changes must be reported and matched in a comparison
that isolates the Rust adapter. These are qualified fixture versions, not service-wide
version floors. Package-declared dependency bounds remain authoritative.

`model-source-check` derives `closure-files.txt` through the linked native core: 2,605
relative files (one manifest plus 2,604 selected blobs). Use `rsync --files-from=...` from
the source store into the initialized target; preserve partials on failure. The owned
pod's sustained SSH uploads interrupted even when capped; completed partial stores were
preserved. The existing local Hub's typed `CheckpointReads` route supplies presigned R2
HTTPS grants for the exact selected objects. Standard `curl`, in four parallel jobs, fetches
and checks each size/SHA-256 into a separately initialized `store-http`. Grants travel on
ephemeral stdin and are never put in durable files/logs or the package
interpreter. The production Hub's exact checkpoint lookup returned 404; this fixture's
publication provenance is the local Hub. Independently verify and admit the completed
destination closure in one CPU process:

```sh
model-source-check --verify /workspace/cozy-machine-pilot/store-http \
  sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642 \
  unet,text_encoder,text_encoder_2,vae \
  /workspace/cozy-machine-pilot/source-qualification
```

The prepared pod configuration is `/workspace/cozy-machine-pilot/stage/sdxl-pilot.json`.
It points at that store, the separate `sdk99` environment, its `.hold`, and a fresh output
root. Transfer completed in 444.83 seconds; native destination verification admitted all
2,604 objects and 6,939,571,699 bytes, with the expected header and four selected components.
The selected 4 MiB-window plan has 2,641 tensors/3,746 items; changing the window changes
item count, not weights. CPU `device-pilot validate` passed there with `gpu_started:false`.
Use the prepared `stage/device-pilot-observations` binary for the hardware run. These CPU
gates do not qualify CUDA/model execution. Pod soft FD limit is 1,024: the legacy native
gate passed, while descriptor caching needs a separate measured lifecycle gate.

For a whole old-stack baseline, launch its actual Go machine agent: `cozy-machine run`
owns the Runtime guardian and host handoff. The fixed `cozy-runtime-worker` takes no argv
and needs the agent-provided configuration plus lifetime/storage fds. Launching a Python
worker or Executor directly cannot stand in for ordinary CLI submit-to-output latency.
