# Attributable CUDA context measurements

Work: [tracker #320](https://github.com/cozy-creator/tracker/issues/320), part of
the independent Rust-machine audit and qualification program, tracker #319.

At Runtime device startup, a device-wide used-memory difference includes allocations
made by unrelated processes. Such a difference cannot be persisted as one executor's
context. Group learning must also never copy rank 0's context to another GPU.

Runtime reads its own per-device driver row after context initialization and library
first launches. Context is the attributed bytes minus its own torch reservation;
later weight-plane measurements also subtract private and shared plane maps. When
the driver cannot attribute the process (including a PID namespace), the context
remains unknown. No arbitrary cap turns device-wide noise into a measurement.

`PlaneFacts.context_measurement = "process_driver"` identifies this provenance.
The field is additive. Peers without it still execute and contribute valid process,
weight and activation facts; their context values do not train future admissions.
Rank-indexed facts train only the corresponding GPU; a missing follower measurement
does not inherit rank 0's context.

Persisted `Learned.context_measurements` records provenance by device and driver.
Opening an old ledger drops only context rows without attributable provenance.
Valid shape peaks, per-method costs, holdings and plan data remain. New attributed
context measurements survive restart. Migration and per-rank/legacy-peer tests run
on CPU. Concurrent foreign allocation during startup and namespace behavior still
require actual CUDA rental qualification.
