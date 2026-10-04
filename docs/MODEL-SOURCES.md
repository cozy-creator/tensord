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


Live session custody is independent of accepted-request cache grace. `open_session_shared`
records the captured receiver birth, supported scope facts and exact native checkpoint-root
owners before installing roots under the same native writer/GC fence. The return value is
retained in DeviceExecutor's opt-in source exit/quarantine resources before exposing headers
or ObjectFiles. Losing a Session handle does not release while receiver/scope exit is live
or unknown. Native roots use the already-materialized runtime closure, no object descriptors
and no extra unselected downloads; ordinary metadata-only source facts retain their existing
short lifetime under accepted/preparing custody.

Startup and maintenance recover recorded partial constructor handoffs and release only after
exact receiver/group AND supported whole-scope exit. Original source repository metadata can
expire first; native materialized reader retention remains valid. A corrupt/unreadable record
or unknown scope removes destructive native-GC authority; fitting work remains possible.
No TTL is exit evidence. Boot change proves every old-boot receiver gone. Cgroups use their
exact inode and search a complete directory census after a rename/disappearance. Tree tokens
are supported same-UID/token-preserving lifecycle containment, not a sandbox: UID-changing or
token-scrubbing descendants are outside that boundary. Unrelated unreadable same-UID processes
can make strict Tree emptiness unavailable; keep custody/charge and visibly refuse reclamation.
Scoped pressure qualification therefore requires delegated cgroups or an isolated readable
provider PID/UID scope. Group exit alone does not prove setsid descendants gone.
