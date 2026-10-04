# Model sources

`model_sources.rs` (`ModelSources`) exports read-only model bytes from the machine's TensorFS store
to a device executor. Only an explicit selection grants authority, never presence in the store.
GPU service callbacks answer executor `model_source` frames with it. RankGroup forwards a
follower's request through rank 0 on the same seam; every rank remains a plane-only consumer.

## API

`open_shared(store, &[SelectedManifest { manifest, components }])` reads the native header.
The native read plan over the selected components defines the allowed objects.

`source(manifest, name) -> Vec<u8>` returns the verified header when name is empty,
or one exact header-declared asset. `serve(&Frame) -> (Answer, File)` wraps those bytes
in a sealed readonly memfd with their digest and length. A manifest outside the session's
selection or an undeclared asset is refused. HostTier's separate ObjectFiles exchange
checks the selected components and exact read plan before granting verified readonly files.

The answer carries `sha256`, `length` and one `SCM_RIGHTS` fd. The receiver verifies those bytes
and closes the temporary descriptor. No Store path or credentials are delegated.

## Known gaps

- Cooperative same-user contract, not a sandbox.
- Each ObjectFiles consumer holds descriptors for the objects behind its live layout;
  that consumer can hit its hard fd limit. ModelSources' durable roots hold no object fds.
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

See [RANK-MODEL-SOURCES.md](RANK-MODEL-SOURCES.md) for the inherited relay audit and the
plane-only two-rank CPU/hardware qualification gates.
