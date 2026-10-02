# Optional executor launch identity

This checkpoint adds only optional root-sealed GpuConfig `identity: {uid, gid}`. It reuses
Runtime's existing `--uid/--gid` trampoline before executor import; no new public API,
version floor, planner or scheduler is added. Omitted identity preserves the inherited UID/GID
and process group. Explicit identity creates a separate process group and verifies actual socket
peer PID/UID/GID against the launched birth/configuration. Nonprivileged owners can only select
their own UID/GID; privilege is checked before a child is launched.

Identity mode uses a deliberately short actual owned socket path instead of the owner /proc/fd
link, which another UID cannot traverse. Pool/state ancestors become group traverse-only (0710),
with owner UID unchanged. Each executor directory and its per-request output spool are handed
to the selected UID/GID (0700). Engine's journal/results/admin paths remain private; the shared
custody function receives this trusted adapter's spool rather than exposing execution ancestors.
Only the interface/cancel marker become owner-held group-readable metadata. Generations, model
store objects and arbitrary authored trees are never recursively chowned or loosened.

This is identity/owned-path preparation, not arbitrary-package containment. Current trampoline
scope is still inherit, not an independently qualified cgroup/process-tree scope. Root must
capture actual old executor UID/GID/cgroup/PGID/seal and CPU-test interpreter traversal,
generation.json/.hold read access, legacy Store metadata/lock permissions, cache/write roots and
ordinary output custody under that identity before creating a GPU context. Prefer root-owned
immutable generations with traversable directories and readable manifests/holds. This adapter
does not silently change their permissions. Root must retain the actual outer cgroup policy.
Long socket paths fail only that configured operation; preserve a short owned benchmark root.

CPU checks: actual installed SDK100 trampoline with configured same UID/GID passed; no CUDA/NVML
was imported or mapped. Parent-death9, no_new_privs1, OOMscore1000, same inherited cgroup and own
PGID were verified. An omitted-identity probe preserved the parent's PGID. Exact opened-directory
and socket ownership, readable peer metadata and symlink rejection passed; private metadata was
unchanged. Durable CPU/restart-fence regression passed 17 tests; all-target clippy passed. Rental
actual-foreign-UID permissions, GPU launch, scope equality and normal CLI output remain root gates.

The old SDK101 wheel's runtime_worker.py45–64 assigns root worker executor seat65533 and media/
supervisor65532; session.py1348–1366 assigns serving slots64000..64511. The Go agent UID65532 does
not identify its executor. Host-root `run/cozy` is the old COZY_HOME, imposed in child.py1004 and
config.runtime_config_from_host; merely adding COZY_HOME to old agent environment does not
change it. Qualification/cache/allocator/NCCL values must come from actual child seal, not guessed
incoming environment. SDXL2.4 generate's authored invocable.memoize is false in both current
committed pin and installed safe interface; no CLI memoization toggle is involved.
