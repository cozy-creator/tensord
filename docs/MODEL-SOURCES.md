# Model sources

`model_sources.rs` (`ModelSources`) exports read-only model bytes from the machine's TensorFS store
to a device executor. Only an explicit selection grants authority, never presence in the store.
`model_source_driver.rs` answers executor `model_source_read` frames with it.

## API

`open_shared(store, &[SelectedManifest { manifest, components }])` reads the native header.
The native read plan over the selected components defines the allowed objects.

`read(SourceRequest { manifest, role, name, length }) -> SourceGrant { sha256, length, file }`:
- `header`: the verified header file;
- `object`: must be in the selected plan, with an exact `length`;
- `asset`: must be declared by the selected header. It is returned as a sealed read-only memfd;
- other roles: `Unsupported`, failing only this request.

The answer carries `sha256`, `length` and one `SCM_RIGHTS` fd. It is used only after the executor
offers `model_sources.descriptors/1`. Descriptors outlive the broker.

## Closure transfer

`model-source-check [--verify] STORE MANIFEST COMPONENTS OUTPUT` writes `closure-files.txt`. With
`--verify` it admits copied paths into the target's own catalog. No foreign SQLite is copied.

## Known gaps

- Cooperative same-user contract, not a sandbox.
- The lease keeps one fd per selected object and can hit the hard fd limit.
- Assets are buffered whole in memory.
