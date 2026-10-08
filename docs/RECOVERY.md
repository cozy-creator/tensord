# Process replacement and recovery

TensorD owns the persistent execution journal and the live node resource decisions. TensorFS
is a library in TensorD and in Runtime. TensorD can retain stored artifacts, shared CPU weight
buffers and exported GPU allocation handles; Runtime must still construct the model and create
its own CUDA mappings. Replacing an executor, restarting the service, and replacing a rental
have different recovery guarantees.

| Event | Retained state | State that must be reconstructed |
| --- | --- | --- |
| Executor replacement while the service lives | Verified TensorFS disk objects/metadata, TensorD-owned sealed CPU buffers, eligible exported GPU allocation handles, journal and collected outputs | Runtime's Python/PyTorch model objects, CUDA context/mappings, streams, activations and interrupted computation |
| Service or supervisor restart with the same machine root | Machine identity, journal, verified disk objects, package generations, collected outputs, job scratch and declared checkpoints | Host memory cache, GPU residency, leases, executor contexts and in-memory credentials |
| Replacement rental without the same durable storage | Only artifacts and records acknowledged by external custody | Local journal, disk cache, model residency and unfinished computation |

A cached artifact's presence does not mean a model is built or its execution completed. A
replacement Runtime executor can reuse retained buffers/allocations only through their ordinary
validated descriptor and lease protocol, constructing its own model and device mappings.
No interrupted copy, kernel, or inference result is declared complete to make recovery possible.
The control service remains CUDA-free. A service's stop flag is a shutdown request, not proof
that dispatch threads, their store references, or executors have ended. Full process-restart
recovery applies after the old process has actually exited.

After a service restart, reconciliation observes exact process births. A live or unobservably dead
executor leaves its obligation nonterminal; it is not adopted. A starting attempt whose process
has ended returns to the queue because no authored work was authorized. A running attempt whose
process has ended fails unless it was a job already pausing. Completed outcomes and collected
outputs remain durable. A paused job resumes as a new attempt of the same transaction, reusing
its declared checkpoints and already finished children; this is distinct from replaying an
interrupted inference. See [durable execution](DURABLE-EXECUTION.md) and
[resident custody](RESIDENT-CUSTODY.md).

## Software updates

An update first prepares its SDK and optional bundled machine executable in a fresh generation.
Caller operation IDs identify receipts; they never select filesystem paths or replace an older
generation's bytes. Activation closes
admission, drains accepted work, and records rollback targets before publishing the new links.
Only after both links are durable does the update enter `starting`. Readiness of that new service
commits `succeeded`. An interrupted publication rolls back before service startup; an ordinary
publication error rolls back while admission is still closed. A candidate that exits twice
before readiness is rolled back by the stable supervisor. Healthy work is not killed by an
update timer. A selected SDK or executable that has disappeared is rolled back while activation
is pending; it never silently falls back to the image and reports success.

If readiness succeeds but committing its outcome cannot write durable state, the service remains
available and the update stays nonterminal. Its observer receives `FailedPrecondition` with
`update_commit_blocked` instead of retrying forever or receiving a fabricated failure. Once the
filesystem is repaired, a service restart retries the pending commit. If success was already
durable and only pending-record cleanup failed, observers still see success; cleanup is retried
before the next update or startup.

The stable parent is deliberately not replaced during activation. A later full launch must use
an installed supervisor that supports handing the new boot's readiness key to an activated
service. Updating a service does not repair an already running older supervisor, or make its
original installed executable newer. Launcher maintenance belongs to the launcher owner.

CPU tests cover the filesystem transaction, actual service process replacement, readiness and
rollback. They do not qualify GPU residency recovery or a provider's container restart behavior.
