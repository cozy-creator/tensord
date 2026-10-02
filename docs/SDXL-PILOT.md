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
device. `run` uses the full stock Runtime device Executor and saves `events.jsonl` plus
`results.json` containing preparation/invoke/post timings, reused executor PID, canonical
result and exact SDK-bound output checksums. One Start/Load serves all three requests.
The trusted SDK encoder produces the actual WebP; root must independently inspect it.

This is a legacy hardware feasibility pilot. The unchanged SDK still imports TensorFS
and opens its store; host-tier retention is disabled. It does not establish sole writer,
machine-owned host/GPU weights, full API replacement or ordinary CLI compatibility.
Stage exchanges return the root's fixed allowance; the pilot has no scheduler.

## Authoritative artifact/package inputs

Current read-only catalog resolution: `cozy model info paul/sdxl --json` selects release
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
