# Owner idle GPU memory reclaim

The pressure benchmark cannot allocate its unchanged physical ballast while idle retained
executors and D2 exported regions own nearly all VRAM. Provide one explicit authenticated
maintenance operation, `POST /v1/machine/memory/reclaim`, on the existing TLS HTTP front door.
Require a current owner-signed machine-scope Cozy-Cap; validate admitted key and expiry again
at mutation. No Hub read, unsigned SSH authority, private-socket bypass or new protobuf RPC.

Use NativeBackend → Service.gpu → GpuPool and the existing GPU reservation/Permit shared by
dispatch, prewarm and prefetch. Busy GPU work or unknown startup births refuse WouldBlock.
While held, every retained session is quiescent. Fence D2 future attachment first; ask idle
readers to release/detach, end only those retained executors through exact exit custody, then
collect revoked root exports whose readers released or provably exited. Keep unknown/live
leases charged. Never stop the Rust machine, cancel accepted work or force-close unknown leases.
The Permit drop wakes queued work through Engine activity.

Return typed counts and root-export byte receipts, keeping physical CUDA free memory a separate
before/after external observation. CPU transport/authority/reservation controls are necessary;
only actual rented CUDA before/after receipts qualify GPU reclaim. Older images refuse only
this new operation. The owner client reuses existing pinned machine-v1 target/key helpers.
