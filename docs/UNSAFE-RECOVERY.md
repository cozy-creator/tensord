# Uncertain operator state and retained GPU weights

[Tracker #320](https://github.com/cozy-creator/tracker/issues/320) covers this boundary.
Runtime reports a latched process poison when a failed mutable, random or opaque
operator has no rollback contract, including when authored code catches the failure.
The machine retires the affected executor and never repeats its started request.

TensorFS makes producer and attached consumer mappings read-only when a region is
exported. As a conservative boundary for unproved failures, the machine nevertheless
marks held allocations Revoking before another plan can attach them and retires idle
sessions with existing mappings. Invalidation closes no live reader descriptor.
Custody retains every allocation charge until its reader lease closes or exact exit
is observed; only then does collection drop the driver's handles.

CPU lease tests exercise this fence, live-reader retention, release and fresh holding
generation. They do not execute CUDA. Required rental proof is an unsafe failure,
retirement, replacement, and identical output from clean stored weights, with no
attachment to an invalidated generation.
