# Store custody and pressure audit

Plan: [tracker #321](https://github.com/cozy-creator/tracker/issues/321), under the
independent Rust runtime audit [#319](https://github.com/cozy-creator/tracker/issues/319).

The machine is the sole lifetime owner of its TensorFS store. State-root ownership is
also required, but it cannot stand in for a store lock: two different state roots can
name the same store. A lifetime `owner.lock` inside the store uses the same inode through
canonical-path aliases; an existing lock is checked before opening the catalog. Native
store initialization remains serialized by TensorFS before the first marker is created.

## Durable bare objects

Write inputs and adopted child files previously had journal authorization but no native
GC root. An ordinary store collection could delete a successfully acknowledged input
of an accepted run. Model caches and output trees having roots did not protect those
bare blobs.

TensorFS individual object roots keep verified blobs without keeping a descriptor open
or generating a synthetic manifest. Machine admission holds native `WriterGuard`
exclusion across put, native root creation and journal acknowledgment. The root is
installed first. A process death before the journal commit can leak an unacknowledged
root until its TTL; it cannot lose an acknowledged blob. These native roots survive
machine death independently of startup ordering or reconstructed read leases.

Acceptance takes the same machine custody mutex used by expiration and native writer
exclusion before validating references. It retains each input, each tree member, and
the local source manifest/archive/wheels/requirements before committing run references
in the acceptance transaction. A child output adopted for a parent gets its parent
reference before the output is returned to the authored job.

Uploaded roots expire after seven days without use. `object_uses` and `run_objects` are
durable policy metadata. Every state outside completed/failed/canceled keeps its
dependencies, including paused and unknown states. Terminal transition time extends
retention by the same TTL. Crash roots without journal metadata use the root timestamp.
The custody mutex excludes expiration while a new run takes references. Startup restores
old admitted objects before configuring publisher GC; older input references are
backfilled without rewriting execution records.

## Selected model custody

Selected checkpoints use TensorFS's existing independent `checkpoint_root` primitive.
One tracked native root serves every live run referencing a manifest; it is not one
descriptor or one native root per request. Native roots precede acknowledged model
dependencies. Journal `run_models` rows are attached in acceptance/binding transactions.
Paused and unknown states remain obligations. When no unfinished run references a root,
the short cache grace allows preparation handoff before the native root is released;
the ordinary managed cache repository then owns idle-cache retention and pressure policy.

Cold ensure transfers custody in an additive `on_ready` callback while the download
flight still holds its objects. This protects every native GC caller, including provider
source conversion, rather than only callers supplying Publisher's keep list. Provider
conversion already writes its authored local repository before disposing conversion and
source roots; the machine adds independent custody before that alias can evolve. Warm
held plans and configured grants are retained before inspecting/returning their bytes.
Normal inference, adapter downloads, job model inputs, and prefetch carry the accepted
execution ID into this handoff.

Startup restores older unfinished preparations and job-input contexts before publisher
GC. Unreadable obligations retain the native writer guard to disable destructive store
GC; ordinary fitting work and unrelated CPU acceptance remain available. Transfer
cancellation and liveness policy are unchanged.

The current ensure and provider conversion paths deliver self-contained runtime model
closures. Checkpoint custody verifies that existing closure; it downloads no extra
unselected data. Component selection limits execution access afterward. A partially
materialized manifest supplied outside those preparation paths remains an unqualified
boundary; no speculative scoped-custody API is introduced.

## Pressure plan still required

The inherited O checkpoint is not accepted as qualified implementation. Its proposed
covering plan counts held generations, multiply linked files and bytes on a different
filesystem. It can remove caches without being able to relieve the measured pressure.
Publisher keep sets also omit paused/unknown runs and lose roots on journal/plan errors.

Follow-up pressure policy must use per-filesystem free space at the reserve, retain
referenced resources conservatively, and count only bytes eligible for release on that
filesystem. A noncovering pressure plan must preserve useful cache roots and compiled
kernels; TTL expiration remains independent policy. Reserve remains the existing
max(1 GiB, capacity/50) pending an owner configuration decision.

## Qualification

CPU regressions cover restart/GC, native admission exclusion, death before journal
acknowledgment, paused/unknown/adopted dependencies, orphan expiration, and another
process attempting to own one store through a path alias and a different state root.
These establish storage behavior, not inference speed or constrained-GPU success.
Ordinary CLI inference, interrupted uploads, pressure while preparing/running/paused,
and candidate-wheel laptop idle sweeps remain integration gates.
