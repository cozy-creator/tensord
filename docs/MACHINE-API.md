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
| **Status** | server stream `{keepalive?}`. The first frame describes the machine: worker and boot id, versions, capabilities, phase, GPUs, Hubs, the caller's live runs, held environments, disk, idle deadline, and the player endpoint: the `cozy/1` listener's port, the addresses to dial and its DTLS certificate's SHA-256. Each later frame is the whole picture again, sent when it changes. Holding the stream is **not** activity. `keepalive: true` resets the idle deadline once, and the first frame carries the new deadline. Without a cap it answers one frame: identity and the sealed readiness receipt. | CLI `machine show`, `rental show`, endpoint check; `rental keepalive` (one reset); Hub readiness (no cap) | It describes the machine, not a run. A unary describe is just its first frame. |
| **Run** | server stream. `{id, after, spec?}`. With a spec and a new id it submits (idempotent on `id`). Every call then streams the run's log from `after`: `state`, `progress`, `product` (output, rev, length, label, media type; stable output names), `log`, `outcome` (status, typed reason, result, output list, triage). The spec holds `source` (a published release by name, private placement, or an uploaded local package manifest), `kind` (`call`, `warm`, `remove` or `update`), entrypoint, payload, input object digests, model choices by name, the caller's catalog revision, attention kernel, weights destination, and the Hub to read with the run's capability there. | CLI `cozy run` (attached or `--detach`), `run watch/show`, `machine model download` and prewarm (`kind: warm`), `rental update` (`kind: update`); Hub public serving; seam child runs | Submit without watch is the first frame then close. Watch is Run without a spec. Prepare is a run that only prepares (`warm`), and an ordinary run prepares inside itself (its `progress` shows install and download). |
| **Control** | unary `{run, cancel}` | CLI `run cancel`; Hub public cancel; seam child cancel | Closing a Run stream must never cancel accepted work, so cancel needs its own call. A preparing run's cancel stops its downloads, installs and an upload's push (nothing is published; the Hub keeps what landed for an upload asked again); only an upload's finalization is past it (`cancel_unavailable`). |
| **Read** | server stream `{target, offset, length, if_rev}`; target is a run output (`output[/i]`), `triage`, or a machine log (with tail). Frames carry rev, total length, the sha256 once final and where this read ends, then the bytes. A `length` reads one range: the CLI reads a big output as ranges side by side on several connections. | CLI downloads (`cozy run --out`, `cp`), triage quote on failure, `machine logs`, `rental logs` | Bytes do not belong in Run's event stream: a 50 GB output, resume at an offset, and many parallel readers. |
| **Write** | client stream: one content-addressed object (sha256, length), resumable from the length already held | CLI `cozy run` file inputs and local package sources, Hub public inputs | Content addressing dedups across runs and resumes big uploads; inline bytes in Run would be re-sent on every attach. |

**WebRTC `cozy/1`** (ICE-lite, passive ICE-TCP, DTLS pinned to the leaf, the cap in its hello)
serves a run's output log and bytes, live or finished. Browsers have no other way to reach a pod
with a self-signed leaf; the CLI does not use it. Caller: the `cozy run play` link and the web player.


`RunSpec.gpus` is optional (`run-gpus/1`). Zero uses the widest GPU group supported by
all package-code model slots on the machine. A positive value requests exactly that many
GPUs; unsupported operations fail instead of reducing the count. This includes GPU child
calls of a deviceless root, whose explicit count reserves one subset while it runs. Checkpoint identity and weight
lanes never select parallelism. Busy GPUs cause queuing, not a smaller group.

`RunSpec.bits` and `WarmItem.bits` (16, 8 or 4) set the run's weight precision; children inherit
it and Runtime gets it with each request. Zero: the lane rule (`precision.rs`, tracker design
`precision-tiers.md` §2.2) picks each slot's lane from its GPUs' compute capability, its declared
`min_bits`, `phase` and `encoded_leaves`, and the memory budget. A fallback is one run warning.


## What disappears

- **Describe, keepalive and the receipt route** fold into Status.
- **Prepare and warm-up** fold into Run (`kind: warm`); uninstall is Run `kind: remove` (`remove/1`).
- **Submit, Watch, Get, Collect and Close** fold into Run.
- **Acknowledge** is deleted. A Read that reaches the final length of an output's final revision
  marks it delivered, and delivered bytes are evicted first.
- **Hub access** (th-241): released content is read anonymously. A run's private operations
  (reading the owner's checkpoint by hash, publishing to the owner's repositories) are named
  by `hub.capability`, which the owner's CLI signs offline with its device key for this machine's
  leaf. At the run's first private operation the machine trades it, once, at `hub.token_endpoint`
  (AuthKit's JWT-bearer grant: an assertion its leaf signs carries the capability, with a DPoP
  proof from the same key) and uses the token until the capability expires; child runs share it.
  A publication the capability does not name ends `capability_required` before the Hub hears of
  it; a refused trade or token `capability_refused`, an expired capability `capability_expired`,
  an operation the Hub narrowed away `capability_exceeded`. Tokens live in memory only. If the
  machine restarts before preparation completes, the run ends FAILED with a typed reason; started
  work is never re-run.
- **Names** (th-245): the machine resolves them at the run's Hub. A release (`org/name`, its
  release or none for the newest) installs from the card and locked requirements it reads there
  by name. A model slot is the run's choice, else its declared `default` release (a callee's
  too): `repository` at `release` (none: the newest) in its `lane`, else in the first lane of the
  rule's order its release publishes, else in its only lane. Anonymous closure calls name its
  manifest, then the pull is pinned to that lane's checkpoint. `manifest` names a
  checkpoint by hash only: the owner's (under the capability) or one the machine holds. Name
  resolutions and held preparations are kept per `catalog_revision`, so a warm run reads nothing
  at a Hub and a revision the caller moved resolves once more; there is no TTL guess or forget verb.
- **List** is deleted. Status shows live runs; history is the CLI's own records plus Run by id.
- **Update** is `Run kind: update` (D2). It stages and verifies a software cohort, activates it at
  measured idle and keeps the known-good install for rollback. The rental keeps its downloaded
  weights; progress and outcome come with the run. The payload is `{runtime, tensorfs, agent}`:
  a published version, or a wheel file name whose bytes an input binds to an object sent with
  Write. A client that loses the stream across the restart attaches again with `Run{id, after}`.
  A pair the machine already runs succeeds unchanged at once. Control CANCEL ends an update before
  it activates (`canceled`, candidate removed, admission reopened); once activating it is refused
  `cancel_unavailable`. Published wheels land through the store as package files do: a download
  that stops moving fails on its measured stillness.
- **Memo, derived retention, forget and prune** become one self-managing cache: TTL plus low-disk
  eviction, delivered entries first, and no verbs.
- **Memo** is the machine's own: a memoized function's success, run directly or as a call, is
  held per signer and answers the same computation (its code, request with files by content,
  attention choice and models) while the machine holds every file it names; the answered run starts
  nothing and its Outcome says `memoized`. An author's `memo_requires` names parameters a request
  must give to be memoizable; one that omits them runs, is not held, and says why (`unmemoized`).
  The client carries no results; a memo answers where its results are (measured 2026-10-06: no
  cross-machine hit ever in the owner's runs).
- **Jobs** are ordinary calls: any function may write weights outputs, and with a weights
  destination (the requester's repository) the machine pushes them to the Hub and settles the
  publication itself. A writer is never memoized.
- **Child runs** are Run and Control issued by a run through its executor seam, scoped to the
  parent: same records, no number, canceled with the parent, outputs granted to the parent's
  spool. Child outputs reach the parent by the seam, so Read is not used for them.

**Pause/resume** are deleted (Paul, 2026-10-14). A canceled or failed run is submitted again; its
finished memoized calls are answered from the memo (`DURABLE-EXECUTION.md`).

**Activity** (a rental's idle release, `rental-lifetime.md`) is a run queued or running (warm-ups
and uploads included), and an explicit keepalive restarts the clock once. Updates,
open streams and other calls are not activity: a Calcifer that holds a stream must not bill forever.

## Today's 35 rows

| # | Today | New |
|---|---|---|
| 1 | ProtocolInfo | Status (capabilities) |
| 2 | WorkerControl.Control | deleted (Cozy-Cap per call) |
| 3 | DescribeMachine | Status |
| 4 | KeepRentalAlive | `Status{keepalive: true}` |
| 5 | receipt, health | Status without a cap |
| 6 | runtime state, wheel, update | Status (versions); Run `kind: update` |
| 7 | hubs/access set, forget | the run's capability in the Run spec |
| 8 | PreparePackageSet | Run (published, `warm` or inside a call) |
| 9 | PreparePrivatePlacement | Run (private source) |
| 10 | PrepareLocalPackage, LocalPackageUpload | Write + Run (local manifest) |
| 11 | Workspace, Submit, CloseSubmission | Run (idempotent on id; workspace in Status) |
| 12 | GetMachineExecution | Run (attach, first event) |
| 13 | ListMachineExecutionEvents | Run |
| 14 | Collect, AcknowledgeCollection | Run `outcome`; ack deleted (Read marks delivery) |
| 15 | ImportInputTree, Retain/ReleaseByteTree, ReadByteTreeObject | Write, Read |
| 16 | Control CANCEL | Control |
| 17 | Control PAUSE/RESUME | deleted (cancel and submit again) |
| 18 | seam publish → product events | Run `product` |
| 19 | HTTPS outputs | Read (CLI), `cozy/1` (browser) |
| 20 | WebRTC media | `cozy/1` |
| 21 | ReadMachineExecutionTriage | Read `triage` |
| 22 | ReadMachineLog | Read (log) |
| 23 | ListMachineExecutions | Status (live runs); history in the CLI |
| 24 | ListPackages, ListModels | Status |
| 25 | ForgetPackage | deleted (exact choices) |
| 26 | memo answer, Record/Lookup/PruneOperation | the machine's held memos |
| 27 | seam child_call/poll/cancel/forget/events | seam Run, Control |
| 28 | jobs | Run `kind: call` |
| 29 | seam checkpoint, tree_member, writer | checkpoint deleted; tree_member, writer: seam, internal |
| 30 | Retain/ReleaseDerivedResult | deleted (outputs in the cache) |
| 31 | NativeArtifactTransfer | deleted (machine pushes weights outputs) |
| 32 | RECONCILE_PUBLICATION | deleted (machine settles with the Hub) |
| 33 | seam stage, budget, device_room, gpu_release | seam, internal (B2) |
| 34 | seam tiers, model sources, prefetch | seam, internal (B1, C) |
| 35 | WatchProgress, ModelSource*, Checkpoint*, NumericalEnvironment, Weights.Upload | deleted |

## Who builds what (approved 2026-10-03)

- **G**: proto `cozy.machine.v1`, Cozy-Cap auth, Run (log, attach, idempotency), Control, Read,
  `cozy/1` from Read, child runs, cache and memo; the CLI's `cozy run`, `watch`,
  `cancel`, `play` and downloads; the browser player.
- **D1**: Run sources and preparation inside a run (published, private, local manifest, `warm`),
  Write, Hub capability handling (memory only); the CLI's upload and preparation paths.
- **D2**: Status (keepalive, idle deadline, no-cap readiness), Run `kind: update`; the
  Hub's readiness read on Status. It is not deployed to production: it ships in the cutover cohort.

## Build order

1. Proto `cozy.machine.v1` and Cozy-Cap auth; Run, Control and Read in the machine; CLI `cozy run`
   on them (G with D1).
2. Write and Run sources/warm (D1); Status and readiness (D2).
3. Child runs (G, with I for H3 long-form); cache; `cozy/1` from Read.
4. Hub readiness and public serving on Status/Run/Control; the browser player on the same caps.
5. The machine builds only `cozy.machine.v1`. Current domain values and private persisted codecs
   are separate; worker RPC generation/vendor files and Claim verification are removed.
