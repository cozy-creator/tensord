# Released old-stack benchmark preparation

These are explicit benchmark drivers, outside the machine service. `prepare`, `inspect`
and `validate` are CPU-only. Only the benchmark owner runs either GPU path.
No controller home, running daemon, Hub registration or system `/opt/cozy` is changed.

The held selection is Runtime **0.18.99**, TensorFS **0.3.90**, SDXL **2.4.0**, the
original `sdxl-pilot-3.json` model binding and store, and unchanged 1024×1024 requests:
20 steps, CFG 7, HiDiffusion off. Both drivers accept any number of payloads. Alternate
arms using the same request document, and record page-cache/qualification-cache state.

## Partial stock executor control

```sh
sdk99/bin/python -I old-baseline-prep/stock_control_v3.py validate stage/sdxl-pilot-3.json \
  --root /workspace/cozy-machine-pilot/sc99-cycle1 \
  --codec-helper /workspace/cozy-machine-pilot/old-baseline-prep/device_codec.py
sdk99/bin/python -I old-baseline-prep/stock_control_v3.py run stage/sdxl-pilot-3.json \
  --root /workspace/cozy-machine-pilot/sc99-cycle1 \
  --codec-helper /workspace/cozy-machine-pilot/old-baseline-prep/device_codec.py
```

This uses the released Runtime executor in its own process, Runtime's typed command decoder
and framed socket implementation, and **the identical Python post helper used by the Rust
pilot**. The helper is copied as a benchmark input, without an implementation fork; its
SHA is included in evidence. `prepare_ms`, `invoke_ms`, `post_ms`, and `wall_ms` use the
same windows as the pilot, including independent post-helper checksum verification.
Image decoding follows all requests and normal executor shutdown, avoiding extra device
gaps between the measured requests.
Spawn-to-Hello, import/start, load/qualification and activation are recorded separately.
The same PID must serve every request; all outputs are decoded and checked at 1024×1024.

This measures control overhead between Python and Rust controllers. It **does not** measure
the Go agent, Python worker, public machine API, their memory policies, or the old worker's
encoding path. Its shared ephemeral helper may add ~0.47 seconds that the old persistent
worker need not pay. This cannot prove effectively zero whole-system hot-path overhead.

## Full released services, gated

```sh
sdk99/bin/python -I old-baseline-prep/old_stack.py prepare stage/sdxl-pilot-3.json \
  --output /workspace/cozy-machine-pilot/old-stack-sdk99-2 --port 18443
sdk99/bin/python -I old-baseline-prep/old_stack.py run \
  /workspace/cozy-machine-pilot/old-stack-sdk99-2/benchmark.json
```

The first command creates an owned machine layout, with symlinks to released executables;
SDK99's `retain_environment` retains the exact preinstalled package environment. Its normal
Go launcher runs with a persistent, loopback-only, authorized-key grant and typed software
policy `startup_update: off`, `agent: bundled`. The second command actually starts that
Go agent and Python worker, connects through pinned TLS and signed ClaimProof, uploads the
real wheel, calls PrepareLocalPackage and PreparePrivatePlacement, then submits each
ReleaseRoot and reads durable events and verified native PNGs. Standard worker probes and
qualification are left enabled. This is a full **RPC** stack gate; default-home Creator CLI,
browser and Hub provisioning remain unqualified.

Agent-start→authenticated workspace ready is measured, along with reused-package prepare,
model placement and per-request submit→encoded/verified output. Cold timings exclude pod
provisioning/OS boot, wheel/dependency installation, model downloads and deliberate page-cache
eviction. The retained SDK environment is preinstalled, shared with the pilot, and not upgraded.
Do not run both arms concurrently against the same mutable TensorFS store. The Go launcher
may change that store directory's owner under its standard machine UID isolation.

Failures leave the benchmark's agent running and record its PID; observer loss never cancels
accepted work. Successful completion explicitly stops its own agent. An exhausted readiness
observer budget is recorded as a test limitation, not proof of a dead machine.

## Confirmed initial custody blocker

CPU proof on the owned A40, 2026-10-02:

```python
s = tensorfs.Store.open('/workspace/cozy-machine-pilot/store-http')
manifest = s.manifest('sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642')['manifest']
len(manifest)  # 164; snapshot bytes are present
s.verify_checkpoint_source('paul/sdxl', 'sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642', 164)
# tensorfs.errors.RepositoryAbsent: checkpoint source repository is absent
```

The old worker requires this repository check even for a held checkpoint before model
placement (`cozy-runtime/.../worker/machine_materialization.py:223`). The stock pilot loads
the snapshot directly and does not qualify this authority. Repair only through an actual
repository envelope/native transfer from the originating Hub/store, using released TensorFS
transport, then rerun `inspect`. No synthetic repository membership or SQL edits are admitted.
Until that gate passes and actual services execute successfully, there is no full old-stack
comparison. CPU evidence is `old-stack-sdk99-2/cpu-inspection.json` on the pod.

The real originating store's released `verify_checkpoint_source` succeeds for the same
repository/digest/length. Read-only `cozy hub list` and `cozy model info paul/sdxl` identify
the current LocalHub `http://127.0.0.1:8819`, release 1.0.0/revision 2, bf16→this checkpoint.
An anonymous actual `/v1/tensorfs/closure` read independently returns that exact identity,
runtime scope, length 164 and 2604 objects. Its bytes are retained at
`outputs/cozy-machine-continued-20261002/old-stack-prep/local-hub-closure.json`.

A private, loopback-only SSH reverse forward exposes that already-running Hub at the pod's
`127.0.0.1:18819`. After measured GPU arms finish, the benchmark owner can use the standard
released native transfer to reuse objects and record legitimate repository custody:

```sh
sdk99/bin/tfs ensure /workspace/cozy-machine-pilot/store-http \
  paul/sdxl@1.0.0@sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642 \
  --hub http://127.0.0.1:18819 --lane bf16 --allow-local
sdk99/bin/python -I old-baseline-prep/old_stack.py inspect stage/sdxl-pilot-3.json
```

Keep native import/hash I/O outside the paired warm measurement sequence. No SQL edits,
manufactured repository records, credentials, public listener or global Hub configuration
are used. A full old-stack RPC measurement still cannot be divided by a partial Rust pilot
and called a whole-architecture latency comparison; equivalent new public-front-door
execution is a separate gate.
