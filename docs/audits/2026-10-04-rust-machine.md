# Rust machine independent audit

Status: initial independent audit complete; remediation and GPU qualification in progress.
Owner: Codex, 2026-10-04. Program: [tracker 319](https://github.com/cozy-creator/tracker/issues/319).

This audit evaluates the Rust machine against two requirements: correct inference under
constrained resources, and faster completion than ComfyUI on matched workloads across GPUs.
The inherited implementation and handoff are evidence, not a binding architecture.

## Baselines

| Repository | Fetched origin/master |
|---|---|
| cozy-machine | cf902bf86d9f980af2e8c912c3d76fef1f5c53b0 |
| cozy-runtime | b2a9373182db0536d177721b74a8fcacf705d80f |
| tensorfs | dd203e1a88329f30bbc507de28098a8441d082ab |
| cozy CLI | dc7c3ecfdbd5d49d00d0ad700b1292e201a98d4d |
| tensorhub | 42b17fe91a3c1db3d43c1b6d876fdb159491ab9a |

The machine actually links TensorFS d1aad922c2586574754b6c5cea764859aa6f0246,
which is older than TensorFS master. Runtime's machine pin is inspected separately.
Each repository has an owned audit worktree branched from the fetched remote master.
Primary checkouts and other worktrees are preserved.

## Evidence standard

- CONFIRMED SOURCE: a concrete failure path follows from the pinned code.
- CONFIRMED CPU: a reproduction or relevant check ran on this audit's snapshot.
- CONFIRMED GPU: actual inference or device execution ran on this audit's snapshot.
- REPORTED: inherited measurement; its inputs, revision and limits must remain attached.
- PLAUSIBLE: a hypothesis requiring additional evidence.

Historical tests, source review and component pilots are not current end-to-end qualification.

## Coverage

The review covers ownership and process topology; scheduling and concurrency; VRAM, host
memory and disk accounting; paging and OOM recovery; multi-GPU collectives; process fencing,
cancellation and restart; journal acceptance and output custody; GC roots and retention;
authentication and package boundaries; CLI and Hub consumers; API evolution; migration and
release gates; benchmark correctness; redundant implementations and simplification.

## Decision

Keep the main split: one Rust machine owns durable execution and TensorFS; Python owns
authored computation and its device contexts. Do not ship the current implementation or
describe it as faster than ComfyUI. The audit reproduced byte loss, nonexclusive store
ownership, broken reconnect projections and continuing observation after revocation.
Recovery also lacks the side-effect contracts needed for correct inference under pressure.

The prior small-card measurements are useful reported evidence. They establish neither a
complete current qualification nor a general ComfyUI speed win. The remaining work is
organized into memory/recovery ([320](https://github.com/cozy-creator/tracker/issues/320)),
storage/custody ([321](https://github.com/cozy-creator/tracker/issues/321)), and API/client
lifecycle ([322](https://github.com/cozy-creator/tracker/issues/322)). The revised architecture
and matrix are in tracker `design/rust-machine-audit-and-qualification.md` (PR 323).

## Confirmed blockers

### Acknowledged inputs can disappear during GC

**P0, CONFIRMED CPU and SOURCE.** `src/objects.rs:187` admits the blob, then
`src/objects.rs:201` records actor ownership in SQLite. Neither establishes a native GC
root. `src/runs.rs:179` checks path existence before `src/runs.rs:218` accepts the run,
without retaining the reference through that transaction. Parent-adopted child results
have the same bare-object lifetime (`src/objects.rs:74`).

The audit wrote a 33-byte input, committed an accepted run referencing it, and ran the
actual pinned TensorFS collector. It collected all 33 bytes while the run remained live.
The collector only knows native roots and live pins, not the machine's SQLite references.
`tests/audit_boundaries.rs::accepted_input_survives_gc` and `audit-reproductions.log`
record the red arm.

Establish durable native custody before acknowledgment, with native writer/GC exclusion
covering put/root/bind. Track accepted tree and local-code closure and parent adoption;
paused and unknown states must retain dependencies. An in-memory lease rebuilt at startup
alone does not protect every crash boundary. Model download-to-run handoff also needs
native custody because another legitimate native caller can run GC with a different keep
set. The latter path is source-confirmed and its independent CPU reproduction is pending.

### Store ownership is locked by state directory

**P1, CONFIRMED CPU and SOURCE.** `src/owner.rs:74` locks `<state>/owner.lock`, then
`src/owner.rs:82` opens the separately configured store. TensorFS `meta::own` is only an
in-process registry (`crates/tensorfs-core/src/meta.rs:69`). A second process with another
state directory accepted ownership of the same store in the audit reproduction.

Its in-memory GC pins are not shared with the first process. This violates the premise
that TensorFS is served by one owner. Use a canonical, inode-based per-store OS lifetime
lock in addition to the independent machine root/identity locks. Test path aliases and
process death, not only duplicate launch on one state directory.

### Learned input purity is not a replay contract

**P1, CONFIRMED SOURCE and CPU recovery-contract red arms.** Runtime
`internal/weights.py:3039` disables operator recovery after an earlier forward appeared
not to mutate tensor inputs. It can then repeat the complete forward after an OOM
(`3114` onward). `_versions` skips inference tensors (`3525`) and examines only one level
of input tensors. Buffers, Python attributes, container mutation and RNG are not restored.

The dispatcher recovery at `weights.py:1281` also repeats failed arbitrary operators,
including mutable, random and opaque custom kernels. Real CPU dispatcher operators that
change bytes/RNG/hidden state before reporting OOM reproduce the missing contract. These
are CPU logic checks, not CUDA inference proof.

Remove inferred whole-forward replay. Permit only defensible failed-operation replay,
distinguish missing recovery contracts from measured physical capacity, and preserve
successful operations exactly once. The unreviewed group-rerun branch repeats complete
mirrored calls without restoring state; rebuilding NCCL alone is not sufficient. Changes
to dispatch must be benchmarked because they add measurable Python overhead.

### GPU context estimates can be poisoned and copied across ranks

**P1, CONFIRMED SOURCE; the 6.8 GB incident is REPORTED.** Runtime
`internal/executor.py:735` samples device-wide memory before and after initialization and
attributes the difference to one process. Concurrent foreign allocation contaminates the
measurement. Machine `memory/learned.rs:137` persists its maximum, and `GpuPool::learn`
uses rank zero's context when training other GPUs even though per-rank facts exist.

Use attributable process observations with provenance, explicit unknown values and each
GPU's own rank. Invalidate old untrusted context entries without deleting valid shape or
cost history. CPU migration/rank tests exist on the fix branch; foreign allocation during
actual CUDA startup remains a required rental red arm.

### Module paging cannot page every leaf tensor

**P1 architectural limit, CONFIRMED SOURCE and CPU metadata reproduction.** Runtime
`internal/paging.py:131` keeps a childless module whole even when it exceeds the refinement
limit. A real meta Embedding with a 16 MiB target produced one 2 GiB region without
allocating that table. The actual H3 text embedding is 151936 by 5120 bf16, or
1,555,824,640 bytes (1.449 GiB), and its published lane retains bf16 embedding weights.

Thus the current mechanism cannot fit that leaf into a 1 GiB available pool before
context and activations. This is a concrete catalog problem, not the cause of Anima's
different repeated-call failure. Add a bounded selected-row embedding path using existing
verified file grants; prove stored values, dtype/device and lifecycle, then qualify H3
through the CLI. Double buffering must remain optional when a single block fits.

Weight paging also does not by itself prove that arbitrarily large activations or opaque
operators fit. Keep supported model/request qualification and recovery capabilities
explicit. Do not rename an unimplemented recovery path into physical impossibility.

### Reconnection changes output revisions and inventories

**P1, CONFIRMED CPU and SOURCE.** `api/machine_v1.rs:728` constructs an empty `Log`
for each attachment. It processes only entries after the requested cursor, while revisions
are local counters (`374`). A reconnect after the first SET labels the second SET revision
one; Read independently reports the actual revision two. A reconnect after the last product
returns a terminal outcome with no outputs (`429`). Both red arms ran over the real TLS
server using a controlled event backend; they are API proof, not inference qualification.

Rebuild projection from durable history, emit only entries past the cursor, and terminate
an already terminal attachment even when its cursor is at or beyond the final event.

### Stream authority and limited-output privacy are incomplete

**P1, CONFIRMED CPU and SOURCE.** `api/machine_v1.rs:489` starts Run observation
without the revocation watcher used by Status. After `keys.revoke`, the audit consumed 32
new events, exceeding the 16-frame buffer. Read and Write also authenticate only at entry.
Status watches key revocation but does not retain the capability expiry deadline.

Output-restricted run caps can observe unfiltered product/outcome/inline metadata and
triage. An author's progress/log text can disclose private inputs too. Protect every
stream for its lifetime and filter restricted observers by granted outputs. Expiry must
detach observation only; accepted work remains owned by the machine. Long Read/Write
streams require client resume with a newly minted capability, without retrying revoked
authority. HTTP and WebRTC bodies need the same lifetime protection and real player proof.

### Update activation and update intent are not durable fences

**P1, CONFIRMED SOURCE; focused CPU tests pending at the initial audit.**
`machine/update.rs:390` checks idle, then relinks and exits (`402` onward), without an
atomic fence against new admission. The v1 Run path does not take Lifecycle admission.
The update records only one latest operation; a reused id can silently attach despite a
changed cohort, and an older id can be executed again after another update replaces it.

Freeze external admission atomically while allowing children required by accepted work;
retain per-operation intent/outcome history; roll the fence back on failure. Run and
update preparation spawn failures must settle or expose recoverable unstarted work:
currently a failed thread spawn can leave a committed PREPARING run stuck forever.

### Resource eviction and file-backed staging are not ready

**P1, CONFIRMED SOURCE; new policy CPU proof pending.** `reclaim.rs:42` enters
pressure below 10% free and remains pressured until 20% (`45`). On this mostly full host,
`193` onward deletes fresh idle kernel caches at every sweep even when their removal
cannot relieve foreign disk usage. The machine still links TensorFS before its newer
reserve/TTL policy.

Do not transplant O's checkpoint unchanged: its proposed covering plan counts locked
generations, multiply linked bytes and cross-filesystem model bytes that cannot relieve
the measured filesystem. Count releasable allocated bytes, hold eligibility locks, check
mount identity, preserve caches when no useful plan exists, and expire only unneeded roots.

B1's machine staged-layout half is still absent. Its candidate also bypasses admission
when a non-staged peer requests every region of an existing sparse layout (comment:
"every region, room or not"). Fix and test that constrained transition before enabling
staging. Reported B1 disk-read improvement predates the staged flag and is not proof of
the new sparse mode on the integrated candidate.

### Benchmark verdict can pass failed and slower inference

**P1, CONFIRMED CPU and SOURCE.** `scripts/gate/gate.py:699` treats lower host peak
as a benefit and allows up to 2% slower warm inference. Its failure list is built from
top-level request rows, excluding failed cell rows. A constructed valid measurement set
returned `pass: true` with every timing 1% slower, host peak 50% smaller and a failed
1 GiB cell. `benchmark-red-arm/summary.json` preserves the actual result.

Require a complete predeclared ComfyUI matrix, explicit paired requests and hardware,
equal submission-to-saved-output boundaries, no failed cells, repeated timing evidence
and output quality validation. Shape plus nonflat pixel statistics only prove an image
was decoded. The Comfy wrapper's `engine_pss_peak` is actually summed RSS; it must not be
reported as PSS. Cold, warm, switching and degraded results remain separate measurements.

## Release and consumer blockers

Two additional identity findings arose while implementing the initial fixes. **P1,
CONFIRMED SOURCE:** the current JCS helper converts arbitrary JSON integers through f64.
Child-run intent hashing can therefore equate seeds 9007199254740992 and 9007199254740993;
arbitrary child returns and terminal inline results can also be rounded. Use precision
preserving structural JSON for these values and preserve ordered input semantics.
Refreshing access credentials or advisory binding/memo hints must not redefine accepted
request intent. Existing experimental rows that lack full authored source/model history
cannot be silently reinterpreted; no-spec observation remains available.

**P1, CONFIRMED SOURCE:** TensorFS plane `layout.rs:123` includes object source hashes
and ranges in its cache identity but omits inline source bytes. Equal layouts with
different inline values can incorrectly share cached host/GPU bytes. This requires a
native CPU cache-identity red arm and a correction based on artifact bytes, not any
client or server source revision equality check. Both fixes are tracked in the existing
component issues.

**P1, CONFIRMED SOURCE.** Runtime master is still version 0.18.102 and bundles the old
Go agent, while its `cozy-runtime-worker` entrypoint has been deleted. That agent launches
the missing worker. Its bundled Rust pin is 1b57a88, behind the audit baseline and fixes.
No 0.18.x release from this tree is safe; the candidate cohort must be built and qualified
before publication. Missing deploy-key secrets remain a release prerequisite.

The Hub public owner still calls DescribeMachine, WorkerControl, PodHost preparation and
SubmitMachineExecution (`tensorhub/internal/publicowner/session.go:128`,
`serving.go:91`, `execution.go:147`). The Rust server retires those operations. Its image
smoke script also exercises old RPCs. Public serving was excluded by the inherited plan;
that exclusion is not proof that global image activation cannot select Rust for this
consumer. Migrate the consumer or explicitly isolate incompatible image selection.

Base image Dockerfiles still name `cozy daemon`; the release override lives on an
unmerged image branch. The promotion hold is not on the audited Hub master. Preserve
existing production and local machines while preparing a tested candidate; image
publication, activation and local replacement are separate gates.

## Architecture and minimality

The good parts are durable accept-before-execute, observation without cancellation,
exact process births/pidfds, inherited environment holds, one byte owner, and optional
host/GPU weight sharing. Existing source/CPU checks support these mechanisms. Retain them
and prove their combination under device pressure rather than replacing the entire stack.

The largest avoidable complexity is treating five new RPCs as a facade over the old
Worker protocol and many default-unimplemented backend methods. After a real consumer
census, use native run/store types internally and keep generated wire adapters at the
edge. Retire duplicate CPU runner/control paths only after their replacement consumer
proof. Do not remove useful fault tests because the corresponding implementation changed.

Degree 2 is an optional speed path: exported allocation handles retain GPU weights after
an executor dies, but their readers can map them writable. The source explicitly assumes
trusted packages under one owner. Per-actor API privacy is not an untrusted-code sandbox;
same-UID package code can read service paths and escape token-based containment. Do not
advertise stronger isolation than the implementation provides.

Scheduling currently serializes GPU calls across the machine and chooses the first K
devices for a group. This is a throughput limitation, not a proven defect in single-call
latency. Measure disjoint-device scheduling demand before adding another scheduler.
Host retention and GPU residency should optimize completion time and reload cost, not
maximize cache occupancy. Keep one policy and explicit movement/lease adapters.

## Validation completed in this audit

- 14 targeted pure GPU-memory policy CPU tests passed.
- The broader selected Rust CPU/storage/process/API baseline passed 121 top-level cases
  with six ignored, plus two separately reported constrained-host subprocess cases.
  Coverage included acceptance/close races, process death/no replay, durable custody,
  host tiers, selected model descriptors, transport, root exclusivity and retired RPCs.
- Five negative boundary checks failed exactly as described, with one subprocess helper
  passing. They use real GC/process/TLS paths with a controlled API event source.
- Hub pod-readiness and worker-image CPU packages passed.
- The benchmark reporting red arm reproduced a false passing verdict.
- New fix-branch CPU evidence is recorded in the component issues; it is not integrated
  inference qualification.

Commands, source revisions and logs are under
`/home/fidika/cozy_v2/outputs/codex-machine-audit-20261004/`. Worktree ownership is recorded
there and in each git worktree's administrative metadata. The read-only workspace audit
found preexisting dirty/unique/metadata items; no cleanup was attempted. Primary checkouts
and unrelated staged/untracked work remain preserved.

## Qualification still required

There has been no new successful inference or ComfyUI comparison on a repaired candidate
at the time this initial audit was written. Historical 6/3 GiB success and 1.5 GiB overlay
success are REPORTED; 1.25/1 GiB, current H3, group recovery, staged host memory, per-step
LoRA, WebRTC, result delivery and release-cohort gates remain open.

The user now authorizes rental testing. Run the ordinary default-home CLI on the actual
integrated candidate, preserve the request, collect its output and trace each preparation,
executor lifetime, resource peak and byte movement. Repeat paired ComfyUI measurements on
the declared GPU matrix. A regression pass, CPU proof, private pilot or one successful
image does not close the program. Every unresolved gate remains in tracker 319–322.
