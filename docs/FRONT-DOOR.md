# Front door

One TLS listener (`api/server.rs`) serves `cozy.machine.v1` (`MACHINE-API.md`), with a 16 MiB
message cap, plus HTTP `GET /v1/health` (204, or 401 if an `Authorization` header is sent). The
readiness receipt is Status's `receipt`, read without a capability. Every `cozy.worker.v1` call answers FAILED_PRECONDITION
(`api/retired.rs`): released CLIs print that message, which names the cozy CLI as the side to
upgrade. `machine_api.rs` (`NativeBackend`) implements the backend over the journal, TensorFS
store and installer; the API owns no scheduler, journal or store. Backend calls run in
`spawn_blocking`; an observer's disconnect never cancels a run.

## Identity and configuration (`identity.rs`)

`cozy-machine serve --machine-config <json> --listen <addr>` (both or neither).

`MachineConfig` (relative paths resolve against the config file):
- `worker_id`: printable ASCII, at most 256 bytes.
- `identity_directory`: owned, mode 0700. Holds `identity.json` (P256 leaf, key, worker id,
  0600) under an exclusive `identity.lock`.
- `authorized_keys_file`: `{"keys": [base64url Ed25519]}`, 1..256 keys, not group/world writable.
- `readiness_hmac_key_file`: `{"key_b64url": ...}`, mode 0600, 32..4096 bytes.

Rules: an invalid or mismatched retained identity refuses startup and is never replaced. Each
process gets a fresh boot UUID and keeps the same leaf. Key file changes apply on restart.
The readiness receipt is `{payload, hmac_sha256}`, where the HMAC covers
`cozy.pod-readiness/1\0` + payload. After binding, the service writes `api-ready.json`
(`address`, `worker_id`, `boot_id`, `cert_pem`) to the state root.

## Install (`install.rs`)

`prepare_uploaded(InstallerConfig, UploadedPackage)` materializes verified descriptors into a
private stage and runs `cozy_machine_client.installer install-captured` (see `PACKAGE-BRIDGE.md`).
- No wall-clock kill. Stdout is capped at 8 MiB.
- A failure `{kind: "install_failed", code, detail}` maps `*_unsupported` to `UNIMPLEMENTED`.
  Anything else is `FAILED_PRECONDITION`.
- It returns `PreparedGeneration`: a 32-hex identity, the record, interface bytes, and a shared
  `.hold` lock.
