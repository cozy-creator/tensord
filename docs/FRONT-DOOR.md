# Front door

`src/api` serves the Creator-facing pinned-leaf TLS/gRPC API and authenticates every call with
ClaimProof. `machine_api.rs` (`NativeBackend`) implements it over the journal, TensorFS store,
package uploads, installer and native inputs. The API owns no scheduler, journal or store.

## Transport

- One TLS listener: gRPC `PodHost` and `WorkerControl` (same operations), 16 MiB message cap,
  plus HTTP `GET /v1/health` (204, or 401 if an `Authorization` header is sent) and
  `GET /v1/bootstrap/receipt`.
- `GET /v1/runs/{run}/outputs/{output}[/{index}]` (`index` 1-based, a list item) for a
  `Authorization: Cozy-Cap <token>` holder (`api/capability.rs`, the Go agent's token: an
  authorized key's Ed25519 grant naming this worker, the run, optional outputs, expiry). Answers
  the output's current bytes from the run's product log: `ETag: "r<rev>"`, `If-None-Match` 304,
  one byte range (206/416), `Repr-Digest` once the run is terminal. Capability failures are 403,
  an absent run or output 404.
- Bindings are generated from `vendor/worker-protocol` (`SOURCE` names the commit).
- `WIRE_MINOR = 72`, `WIRE_MINIMUM = 0`. Versions are reported, never used to refuse a peer.
  A missing operation returns `UNIMPLEMENTED` (`capability_unavailable: ...`) for that call only.
- Backend calls run in `spawn_blocking`. Observer disconnect never cancels a run.

## Authentication (`auth.rs`)

Every call except `ProtocolInfo`, health and the bootstrap receipt carries a `Claim`. Streams
carry it in their first frame (input header, upload header, Control `Claim`).

- `Claim.worker_id` and `worker_boot_id` must equal this process.
- `proof` is an Ed25519 `verify_strict` signature, by one configured key, over the canonical
  JSON `{format: "cozy.worker.v1.ClaimProof/1", worker_tls_certificate_digest: "sha256:<leaf>",
  record_owner_epoch, worker_boot_id, worker_id}`. Proto3 default values are omitted.
- Result: `VerifiedActor { public_key }`. The actor id is the key's SHA-256 hex.
  `record_owner_id` is unsigned and never authority. `Claim.wire_minor` is ignored.
- Control: the first frame must be a `Claim`. The reply is one `ClaimAck`. Any later frame gets
  `UNIMPLEMENTED`: legacy desired-state Control is not implemented. Control is not an execution lease.

## Identity and configuration (`identity.rs`)

`cozy-machine serve --machine-config <json> --listen <addr>` (both or neither). Optional
`--installer-python` + `--client-wheel` enable `PrepareLocalPackage`.

`MachineConfig` (relative paths resolve against the config file):
- `worker_id`: printable ASCII, at most 256 bytes.
- `identity_directory`: owned, mode 0700. Holds `identity.json` (P256 leaf, key, worker id,
  0600) under an exclusive `identity.lock`.
- `authorized_keys_file`: `{"keys": [base64url Ed25519]}`, 1..256 keys, not group/world writable.
- `readiness_hmac_key_file`: `{"key_b64url": ...}`, mode 0600, 32..4096 bytes.

Rules: an invalid or mismatched retained identity refuses startup and is never replaced. Each
process gets a fresh boot UUID and keeps the same leaf. Key file changes apply on restart.
The bootstrap receipt is `{payload, hmac_sha256}`, where the HMAC covers
`cozy.pod-readiness/1\0` + payload. After binding, the service writes `api-ready.json`
(`address`, `worker_id`, `boot_id`, `cert_pem`) to the state root.

## Backend hooks (`backend.rs`)

`MachineBackend` methods take a `VerifiedActor` and default to `UNIMPLEMENTED`. Only `workspace`
is required. Hooks: `describe_runtime`, `list_packages`, `list_models`, `retain_bytes`,
`release_bytes`, `begin_input_tree`, `workspace`, `submit`, `get`, `events[_observed]`,
`control`, `list[_observed]`, `close_submission`, `collect`, `ack_collection`,
`read_bytes`/`read_stream`, `uploads`, `prepare_local`, `read_machine_log`, `forget_package`,
`open_output`, `read_triage`.

- `Observation` is set when the reader goes away. It only ends a wait. It has no run-control authority.
- Event pages: default and max 256. Execution lists: default 64, max 256.
- Input tree: header first (manifest at most 1 MiB), blobs at most 1 MiB, explicit final commit.
  The network layer never assembles a whole input.
- Byte reads: two-chunk backpressure channel. An empty chunk or one over 1 MiB is `DATA_LOSS`.

## NativeBackend behavior

- `submit` accepts only a `release_root` callable. The machine computes the payload, capture and
  invocation digests itself. A callable that declares models needs the GPU pool, which prepares
  and binds a preparation. An empty `installation_id` resolves a published installation through
  the GPU pool.
- `forget_package` (`org/name`) drops the calling owner's held model resolutions of that
  package; installations are keyed by exact release and stay.
- `read_machine_log` serves `MACHINE_LOG_TENSORFS_TRANSPORT`: the store's `logs/transport.log.1`
  then `transport.log`, optionally the newest `tail_bytes` from a line start, in 64 KiB chunks.
  Any other log is `NOT_FOUND`.
- `events` with `wait` blocks up to 30 s on the engine activity epoch. Terminal pages come from the
  durable `public_terminals` projection.
- `control`: `Cancel` only, with an optional `expected_generation` check.
- `ack_collection` must match the retained outcome exactly. Its boot id may be omitted. Unless
  `retain_work`, it appends one `retention_released` event.
- `retain_bytes` requires the actor to own the producer root. `release_bytes` and `read_stream`
  require the exact recorded subject. Native outputs are actor-scoped.

## Package upload (`workspaces.rs`)

`LocalPackageUpload` sends a header, then chunks. Each (actor key, `operation_id`) has its own
directory and one live uploader (`ABORTED` otherwise).
- Carriers: `source.tar` (no digest, up to 1 GiB) or a `.whl` with a 32-byte SHA-256 (up to
  512 MiB). At most 129 files and 1 GiB per set. Chunks are at most 1 MiB, at the exact durable
  offset, and fsynced. A changed identity under the same operation is refused.
- `source.tar` is scanned statically and never executed. It allows regular files and dirs only,
  unique safe paths, bounded PAX/GNU records, and requires root `pyproject.toml` and `uv.lock`.
- Complete carriers enter TensorFS via `source_artifact::import_tree` under a per-actor retention.
- `package()` returns `UploadedPackage` (`RootSet` + verified files) once every carrier is
  verified. A changed `RootSet` for the same operation is refused.
  `release_after_install` drops the retentions.

## Install (`install.rs`)

`prepare_uploaded(InstallerConfig, UploadedPackage)` materializes verified descriptors into a
private stage and runs `cozy_machine_client.packages install-captured` (see `PACKAGE-BRIDGE.md`).
- No wall-clock kill. Stdout is capped at 8 MiB.
- A failure `{kind: "install_failed", code, detail}` maps `*_unsupported` to `UNIMPLEMENTED`.
  Anything else is `FAILED_PRECONDITION`.
- It returns `PreparedGeneration`: a 32-hex identity, the record, interface bytes, and a shared
  `.hold` lock.

`prepare_local` binds the actor's installation alias in the journal, calls `changed_environment`,
then releases the upload. A known alias with a different package or release is `ALREADY_EXISTS`.

## Native inputs (`native_inputs.rs`)

`SourceIntake` imports one input tree per (workspace, actor, request, input). The retention id
is the SHA-256 of a canonical `cozy.machine.input-intake/1` document.
- Spec is immutable in the journal (`begin_intake`). The manifest must verify and contain
  files only. `content_bytes` must match.
- A TensorFS `WriterGuard` excludes GC during the transfer. Objects already in the store are
  skipped. Blobs are contiguous per object.
- Commit imports the tree, then records the receipt and the actor's native source in one
  transaction. Retrying reconciles a commit whose reply was lost.
- Abort persists a released tombstone and releases intake custody. It never cancels a run.
  A late commit after abort returns released. A disconnect discards staging. There is no resume.

## Not implemented

- `list_models`, workspace `describe` and captured offers.
- Hub grant routing, asset inputs, jobs, deadlines, attention-kernel selection and publication.
- List `wait`, and control actions other than `Cancel`.
- Live key rotation, Hub receipts/grants and rental enrollment.
- Reclamation of abandoned uploads.
- The `front-door` binary is an inspection fixture with no run capabilities.
