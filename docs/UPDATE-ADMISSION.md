# Update admission and operation identity

An idle check alone cannot authorize a service replacement: a new root may be accepted
between that check and process exit. Update activation first closes external admission
under the same Lifecycle mutex used by root admission and idle release. Already admitted
calls remain counted until their acceptance finishes. Existing engine/preparation work then
drains before the cohort links change and the service exits. A failed activation drops the
barrier and reopens admission. Internal children of an already accepted parent continue;
blocking those children would prevent the parent from draining.

V1 root submission, uploads and explicit resume use that admission barrier. Accepted root
work renews the rental ledger before admission is released. A Run observer holds no admission;
its disconnect or authorization expiry cannot keep an idle rental alive or block activation.
Upload attempts renew activity so a valid partial upload can resume after its short cap ends.
Cancel/pause and read/observe operations remain available while activation waits for old work.

Each update operation retains its history independently of the latest status projection.
Its digest describes the requested cohort and normalizes omitted default agent/pin choices;
reusing the id for another cohort is a conflict. An older completed id stays attachable after
another update and a restart. Reattaching to a known update does not require its original
uploaded input blob to remain in TensorFS. Unknown advisory payload fields are ignored.

A preparation-thread spawn failure becomes a failed update rather than a durable waiting
operation with no executor. A restart before activation marks interrupted preparation failed
without replaying it. Pending activation follows the existing readiness/rollback path.

Targeted CPU checks cover the barrier's admission/release race, draining an existing admission,
durable historical ids, conflicting cohorts and interrupted preparation. Real service restart,
ordinary CLI update with concurrent inference/upload, and candidate rollback remain product
gates. These checks do not qualify GPU inference or ComfyUI performance.
