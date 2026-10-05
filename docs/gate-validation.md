# Benchmark validation

Owner: Codex `bench_takeover`. Branch: `fix/gate-fail-closed-20261005`.
Base: `fc876952224d704f2d23c79a23f8fd9ef4033a30` (`origin/master`, fetched 2026-10-05).

The 2026-10-05 `kosuda` switch benchmark continued after ComfyUI setup failed and
Anima priming failed. Its driver exited zero with failed cells. The R1 report also
passed missing controls and accepted changed low-memory images. The preserved
outputs are under `outputs/cozy-machine-takeover-20261002/MEM/switch12/` and
`F/small-cards/takehisa/` in the Cozy v2 workspace.

This increment changes the tracked gate harness only:

- Host setup and explicit preflight commands finish successfully before sampling,
  priming, or timed cells. Per-cell setup and preflight finish before its timer.
- Failed priming is terminal even when timed failures are collected for diagnosis.
- Collected failed cells and incomplete qualification return a nonzero exit status.
- Qualification requires every planned candidate cell and every low-memory
  request's successful control and verified file-hash receipts. Exact byte identity
  is the current same-GPU acceptance bar; PSNR remains separate diagnostic data.

CPU checks use real failing setup/CLI commands and real retained image files. This
is harness evidence, not GPU, inference, or performance qualification. No rental,
GPU, Runtime, package, allocator, or release change belongs to this increment.

## Anima product contract requiring separate review

Anima 0.3.3 installs its `_AttnProcessor` on the Cosmos attention modules during
construction. Runtime's Anima optimization requires exact
`CosmosAttnProcessor2_0` instances, then replaces them with its fused processors.
The package/Runtime pair therefore fails `anima_optimization_block_shape` before
inference. The benchmark must expose that incompatibility before timing rather
than reset processors, remove optimization, or modify the request.

Proposed product decision: agree one owner for the Anima attention processor.
Prefer retaining the package's normal Diffusers processor until Runtime applies
its documented optimization, or extend the Runtime contract to an explicitly
supported package processor after a semantic review. Do not accept arbitrary
processor types or version/source equality checks. Any chosen implementation must
preserve supported mixed package/Runtime versions, output semantics, and the
unchanged request. Compare the same prompt, seed, initial latent, intermediate
attention/denoise tensors, and final encoded image for sequential/batched CFG and
full-card/sub-block execution on one rented GPU before claiming exact equivalence.
Cross-device/numerical differences are reported separately.
