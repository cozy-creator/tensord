# Resident custody

`resident_custody.rs` (`ResidentCustody`) is one device actor's bounded in-memory inventory of
device-resident weight allocations. It binds leases to Engine executions and process births. It
does not schedule, call CUDA, journal or kill.

## Model

- `ResidentKey`: `actor`, `device_uuid`, `content_sha256`, `layout_sha256`, `representation`.
- `AllocationId`: `owner_epoch` (fresh per instance), `id`, `generation`.
- Phases: `filling` -> `ready` -> `revoking`/`quarantined` -> `releasing` -> removed.

## Rules

- `register` starts in `filling`. When full it returns `WouldBlock`, a wait condition, not a refusal.
- `attach(allocation, execution_id, role, pidfd)` requires:
  - the execution is `starting`/`running` with no cancel actor;
  - its public actor equals `key.actor`;
  - the fd is a kernel pidfd for Engine's recorded birth.
  One writer may attach while `filling`. Readers attach only when `ready`.
- `complete_fill` and `release_recipient` need a `NativeCompletion` for the exact lease. It is minted
  only by `unsafe after_native_release`, after uses are fenced, DMA is done and handles are closed.
- `reap_ended` needs both pidfd exit and Engine's birth check. A dead writer makes the allocation
  `quarantined`.
- `begin_revoke` fences new leases. `take_for_release` needs no recipients. The charge stays until
  `unsafe confirm_physical_release`.
- No adoption across restarts: a new epoch rejects old ids.

## Known gaps

- Not wired into `GpuPool`. Nothing in production mints `NativeCompletion`.
- `quarantine` accepts any phase, including `releasing`.
