# Machine API (greenfield, hard cut): the minimal complete set

Every client of a machine uses it: the CLI, the Hub, the browser player, and a run's own child runs.
A laptop's machine and a rental's are the same. `worker.v1` (`PodHost`, `WorkerControl`,
`RuntimePreparation`) and their HTTPS execution routes are removed. The old URL namespace only
returns an upgrade refusal; no worker bindings or service implementations are built.

**Five gRPC calls** (`cozy.machine.v1.Machine`) on the machine's one TLS listener, whose leaf the
client pins by fingerprint. **One browser channel**: WebRTC `cozy/1`. **One credential**:
`Cozy-Cap`, sent as `authorization: Cozy-Cap …` metadata or in the `cozy/1` hello. A cap is an
authorized key's Ed25519 grant naming the machine, a scope (`machine` = every call; or
`run:<id>`, optionally limited to some outputs), and an expiry. ClaimProof, record-owner epochs
and wire ranges are deleted. Errors are a gRPC status plus one typed reason `{code, message}`,
the same reasons a run's outcome carries. Unknown fields are ignored with a warning.

## The calls

| Call | Shape | Real callers | Why it cannot fold into another |
|---|---|---|---|
| **Status** | server stream `{keepalive?}`. The first frame describes the machine: worker and boot id, versions, capabilities, phase, GPUs, Hubs, the caller's live runs, held environments, disk, idle deadline (held models come with the binding revision), and the player endpoint: the `cozy/1` listener's port, the addresses to dial and its DTLS certificate's SHA-256. Each later frame is the whole picture again, sent when it changes. Holding the stream is **not** activity. `keepalive: true` resets the idle deadline once, and the first frame carries the new deadline. Without a cap it answers one frame: identity and the sealed readiness receipt. | CLI `machine show`, `rental show`, endpoint check; `rental keepalive` (one reset); Hub readiness (no cap) | It describes the machine, not a run. A unary describe is just its first frame. |
| **Run** | server stream. `{id, after, spec?}`. With a spec and a new id it submits (idempotent on `id`). Every call then streams the run's log from `after`: `state`, `progress`, `product` (output, rev, length, label, media type; stable output names), `log`, `outcome` (status, typed reason, result, output list, triage). The spec holds `source` (published release, private placement, or an uploaded local package manifest), `kind` (`call`, `job`, `warm` or `update`), entrypoint, payload, input object digests, model choices, the binding revision the CLI knows, the known memo results, attention kernel, weights destination, and the Hub access token. | CLI `cozy run` (attached or `--detach`), `run watch/show`, `machine model download` and prewarm (`kind: warm`), `rental update` (`kind: update`); Hub public serving; seam child runs | Submit without watch is the first frame then close. Watch is Run without a spec. Prepare is a run that only prepares (`warm`), and an ordinary run prepares inside itself (its `progress` shows install and download). |
| **Control** | unary `{run, cancel \| pause \| resume}` | CLI `run cancel`, `job cancel`, `run pause/resume`; Hub public cancel; seam child cancel | Closing a Run stream must never cancel accepted work, so cancel needs its own call. |
| **Read** | server stream `{target, offset, if_rev}`; target is a run output (`output[/i]`), `triage`, or a machine log (with tail). Frames carry rev, total length, the sha256 once final, then the bytes. | CLI downloads (`cozy run --out`, `cp`), triage quote on failure, `machine logs`, `rental logs` | Bytes do not belong in Run's event stream: a 50 GB output, resume at an offset, and many parallel readers. |
| **Write** | client stream: one content-addressed object (sha256, length), resumable from the length already held | CLI `cozy run` file inputs and local package sources, Hub public inputs | Content addressing dedups across runs and resumes big uploads; inline bytes in Run would be re-sent on every attach. |

**WebRTC `cozy/1`** (ICE-lite, passive ICE-TCP, DTLS pinned to the leaf, the cap in its hello)
serves a run's output log and bytes, live or finished. Browsers have no other way to reach a pod
with a self-signed leaf; the CLI does not use it. Caller: the `cozy run play` link and the web player.

## What disappears

- **Describe, keepalive and the receipt route** fold into Status.
- **Prepare and warm-up** fold into Run (`kind: warm`).
- **Submit, Watch, Get, Collect and Close** fold into Run.
- **Acknowledge** is deleted. A Read that reaches the final length of an output's final revision
  marks it delivered, and delivered bytes are evicted first.
- **HubAccess** is deleted. The token travels in the run spec. It is held in memory for that run's
  preparation only, and never written to the journal or any durable record. All Hub reads happen
  inside a run, and child runs inherit the token. If the machine restarts before preparation
  completes, the run ends FAILED with a typed reason; started work is never re-run.
- **Binding freshness** is the spec's binding revision: the revision the CLI knows, since it ran
  `package bind`. The machine re-resolves a held model resolution only when the revision differs.
  There is no TTL guess and no forget verb.
- **List** is deleted. Status shows live runs; history is the CLI's own records plus Run by id.
- **Update** is `Run kind: update` (D2). It stages and verifies a software cohort, activates it at
  measured idle and keeps the known-good install for rollback. The rental keeps its downloaded
  weights; progress and outcome come with the run. The payload is `{runtime, tensorfs, agent}`:
  a published version, or a wheel file name whose bytes an input binds to an object sent with
  Write. A client that loses the stream across the restart attaches again with `Run{id, after}`.
- **Memo, derived retention, forget and prune** become one self-managing cache: TTL plus low-disk
  eviction, delivered entries first, and no verbs.
- **Memo** is the machine's own: a memoized call's result is held per signer and answers the same
  computation while the machine holds every file it names. The client carries no results; a memo
  answers where its results are (measured 2026-10-06: no cross-machine hit ever in the owner's runs).
- **Jobs** are a run kind. Weights outputs are outputs; with a weights destination the machine
  pushes them to the Hub and settles the publication itself.
- **Child runs** are Run and Control issued by a run through its executor seam, scoped to the
  parent: same records, no number, canceled with the parent, outputs granted to the parent's
  spool. Child outputs reach the parent by the seam, so Read is not used for them.

**Pause/resume** stay as Control actions (lead's decision). Only a job pauses (`pause_unsupported`
otherwise): its root stops, started children run to their end, unstarted ones wait, and the run
shows `paused`. Resume replays the root, whose calls find their finished children
(`DURABLE-EXECUTION.md`); `run_pausing` and `run_not_paused` refuse early or late resumes.

**Activity** (a rental's idle release) is non-terminal runs, preparations and explicit keepalives
only. Open streams and other calls are not activity: a daemon that holds a stream must not bill
forever.

## Today's 35 rows

| # | Today | New |
|---|---|---|
| 1 | ProtocolInfo | Status (capabilities) |
| 2 | WorkerControl.Control | deleted (Cozy-Cap per call) |
| 3 | DescribeMachine | Status |
| 4 | KeepRentalAlive | `Status{keepalive: true}` |
| 5 | receipt, health | Status without a cap |
| 6 | runtime state, wheel, update | Status (versions); Run `kind: update` |
| 7 | hubs/access set, forget | token in the Run spec |
| 8 | PreparePackageSet | Run (published, `warm` or inside a call) |
| 9 | PreparePrivatePlacement | Run (private source) |
| 10 | PrepareLocalPackage, LocalPackageUpload | Write + Run (local manifest) |
| 11 | Workspace, Submit, CloseSubmission | Run (idempotent on id; workspace in Status) |
| 12 | GetMachineExecution | Run (attach, first event) |
| 13 | ListMachineExecutionEvents | Run |
| 14 | Collect, AcknowledgeCollection | Run `outcome`; ack deleted (Read marks delivery) |
| 15 | ImportInputTree, Retain/ReleaseByteTree, ReadByteTreeObject | Write, Read |
| 16 | Control CANCEL | Control |
| 17 | Control PAUSE/RESUME | Control |
| 18 | seam publish → product events | Run `product` |
| 19 | HTTPS outputs | Read (CLI), `cozy/1` (browser) |
| 20 | WebRTC media | `cozy/1` |
| 21 | ReadMachineExecutionTriage | Read `triage` |
| 22 | ReadMachineLog | Read (log) |
| 23 | ListMachineExecutions | Status (live runs); history in the CLI |
| 24 | ListPackages, ListModels | Status |
| 25 | ForgetPackage | deleted (spec's binding revision) |
| 26 | memo answer, Record/Lookup/PruneOperation | the machine's held memos |
| 27 | seam child_call/poll/cancel/forget/events | seam Run, Control |
| 28 | jobs | Run `kind: job` |
| 29 | seam checkpoint, tree_member, writer | seam, internal |
| 30 | Retain/ReleaseDerivedResult | deleted (outputs in the cache) |
| 31 | NativeArtifactTransfer | deleted (machine pushes weights outputs) |
| 32 | RECONCILE_PUBLICATION | deleted (machine settles with the Hub) |
| 33 | seam stage, budget, device_room, gpu_release | seam, internal (B2) |
| 34 | seam tiers, model sources, prefetch | seam, internal (B1, C) |
| 35 | WatchProgress, ModelSource*, Checkpoint*, NumericalEnvironment, Weights.Upload | deleted |

## Who builds what (approved 2026-10-03)

- **G**: proto `cozy.machine.v1`, Cozy-Cap auth, Run (log, attach, idempotency), Control, Read,
  `cozy/1` from Read, child runs, jobs, cache and memo, pause/resume; the CLI's `cozy run`, `watch`,
  `cancel`, `play` and downloads; the browser player.
- **D1**: Run sources and preparation inside a run (published, private, local manifest, `warm`),
  Write, Hub token handling (memory only), binding revision; the CLI's upload and preparation paths.
- **D2**: Status (describe, keepalive, idle deadline, no-cap readiness), Run `kind: update`; the
  Hub's readiness read on Status. It is not deployed to production: it ships in the cutover cohort.

## Build order

1. Proto `cozy.machine.v1` and Cozy-Cap auth; Run, Control and Read in the machine; CLI `cozy run`
   on them (G with D1).
2. Write and Run sources/warm (D1); Status and readiness (D2).
3. Child runs and jobs (G, with I for H3 long-form); cache; pause/resume; `cozy/1` from Read.
4. Hub readiness and public serving on Status/Run/Control; the browser player on the same caps.
5. The machine builds only `cozy.machine.v1`. Current domain values and private persisted codecs
   are separate; worker RPC generation/vendor files and Claim verification are removed.
