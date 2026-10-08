# Model sources

TensorD makes selected model weights and metadata available to Runtime. A source identifies an
exact manifest and selected components and supplies their metadata/bytes; it is not merely a
model name. The selection is also an access boundary: presence in the store alone does not
authorize a request.

`model_sources.rs` (`ModelSources`) reads verified headers and header-declared assets through
TensorFS. `GpuPool` answers executor `model_source` requests with those bytes in sealed
read-only descriptors. Tensor payloads reach Runtime through the shared CPU layouts,
streaming windows or granted object descriptors described in [Host tier](HOST-TIER.md).
TensorFS runs as a library in TensorD and in Runtime; these requests do not address a
separate TensorFS daemon.

## API

`open_shared(store, &[SelectedManifest { manifest, components }])` verifies the manifest/header
and the requested components. TensorFS's read plan calculates their encoded source bytes.
`selected_facts(manifest)` returns the component names, encoded byte count and manifest length;
the byte count is neither prepared CPU-buffer residency nor a prediction of GPU memory use.
`authorized_header(manifest)` supplies the header used to validate host-tier requests.

`source(manifest, name)` returns the verified header when `name` is empty, or the named
header-declared asset. Assets are read through a TensorFS lease over their own objects.
`serve(frame)` places those bytes in a sealed memfd and returns `sha256`, `length` and a
read-only descriptor; the control seam transfers it with `SCM_RIGHTS`.

Runtime parses the granted metadata using its local TensorFS library, constructs the model,
and arranges weight reads/transfers with its weight plane. Keeping that work in the executor
does not give it responsibility for TensorD's store lifetime, collection or machine-wide
budgets. Descriptor and buffer lifetimes are separate from the request that supplied them.

## Closure transfer

`model-source-check [--verify] STORE MANIFEST COMPONENTS OUTPUT` writes `closure-files.txt`. With
`--verify` it admits copied paths into the target's own catalog. No foreign SQLite is copied.

## Known gaps

- Cooperative same-user contract, not a sandbox.
- Host-tier object descriptors/read leases can reach the hard fd limit; see [Host tier](HOST-TIER.md).
- Assets are buffered whole in memory.
