# Resident custody foundation

`resident_custody::ResidentCustody` is a bounded inventory owned by the existing device actor.
It does not schedule, issue CUDA work, add a journal, kill a process, or advertise Degree 2.
It uses `Engine::get` to bind an attachment to the existing durable accepted actor/attempt and
registered process birth. It uses `Engine::notify_activity` to wake the existing scheduler when
custody changes; observation does not change a lease.

The native actor supplies real backing and host-source holds through `Resources`. Those references
can retain TensorFS IPC allocations/export descriptors and the existing shared-host owner's
prepared claim. They remain retained across unknown fill failure and revocation. The key names
the verified actor, physical device, immutable content/layout and exact decoded representation;
these digests identify weight bytes, not a software-version/source fingerprint compatibility gate.

`register` returns a unique allocation ID and generation under this owner's fresh epoch. One
writer may attach while Filling. Readers attach only after `complete_fill` receives native release
evidence. `begin_revoke` fences further leases. `release_recipient` matches the exact generation,
execution and birth; old epochs or another recipient's completion cannot release it. An unknown
fill whose writer dies becomes Quarantined, never Ready. A replacement creates new physical
backing/identity rather than overwriting a generation that old work might still reference.

Every attachment retains the managed Executor's actual pidfd. The fd must be a kernel pidfd for
the Engine's registered live birth. `reap_ended` requires both kernel death and the existing birth
check. Durable cancellation, completed/failed status, socket EOF, observer loss, missing telemetry
and elapsed time are not death/completion evidence. The live warm executor therefore keeps its
weight obligations after individual requests become terminal.

`NativeCompletion` has no deserializer or safe constructor. Its unsafe native-bridge constructor
requires fencing all new uses/views, completing all successful DMA/kernel work across all streams,
unmapping, releasing tensor owners/imported handles and closing every received/export duplicate.
A CPU acknowledgement or progress/event counter cannot establish this contract. No public wire
adapter currently mints this token: the actual SDK/native tensor-view bridge remains a hardware
qualification gate. Native IPC mechanism tests do not qualify that bridge by themselves.

`take_for_release` hands resources to the native device actor only after no recipient remains.
The physical byte charge stays recorded during Releasing. `confirm_physical_release` is another
unsafe actor boundary: all owner/keeper/fd/handle references and physical release must be proved.
Dropping an Arc or subtracting a logical counter cannot credit capacity. An error keeps the charge.
There is no elapsed-time retry or forced revocation policy.

This live inventory is intentionally not a persistent opaque-handle adoption scheme. A new owner
gets a new epoch and rejects old attach/release identities. The existing journal preserves accepted
work and known process births; startup policy must quarantine still-live old device users and use
physical driver observations before new device admission. Persistent allocation inventory, warm
executor orphan inventory and live adoption need a separate explicit gate; this module does not
claim that its empty new map accounts for old mappings after restart.

The actual CPU process gate transfers a real file descriptor to an independent interpreter,
records its birth in the same durable Journal later opened by Engine, requests explicit cancellation,
and proves resources/charge remain while the process lives. After actual exit it quarantines the
unknown fill, releases owner references only through the release handoff and keeps the charge until
physical confirmation. A second gate refuses an arbitrary fd even for an accepted live attempt.
No CUDA/NVML or inference is simulated. The file is a CPU lifetime fixture, not GPU backing.

Integration: GPUService's managed birth callback registers the process in Engine before authored
Start/import. After capability and headless qualification, its device actor can register native
backing, attach the same exact pidfd and retain the host preparation. It must never perform driver
waits under Engine/journal/API locks. Degree 1 stays the available fallback. No multi-GPU/NCCL,
machine-issued copies, low-memory guarantee or fault/stress safety is advertised here.
