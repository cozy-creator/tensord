# Registered legacy Submit replay identity

Tracker #322/#319, owner Codex api_audit. Base is actual listener census018d0b8 on
integrated946a03e. PodHost and WorkerControl remain registered and share NativeBackend.

NativeBackend.submit returns an accepted_public record found by request OR submission before
checking that both IDs and authored intent match. The receipt may therefore name another
submission or different code/payload/model choices. Fix this operation; do not delete working
handlers or compare raw serialized protobuf bytes/source SHAs.

Normalize a typed authored intent with precision-preserving application JSON. Include root
code selection, entrypoint, owner/Hub scope, explicit model/profile/ordered adapter choices,
input identities and their authored order, attention pin and authored output/deadline/kind.
Exclude transport Claim, expiring SourceCredentials/DeliveryGrant/input-access provenance,
publication bearer, and advisory memo hints. Model choices are a parameter-keyed set under
the existing contract; adapter stack/profile and explicitly numbered input order stay intact.

Require both accepted IDs and journal identity before reattachment. Persist the new semantic
intent atomically in SubmissionContext at new acceptance so its existing transactional equality
also rejects concurrent conflicting first accepts. Matching replay happens before mutable
installation/model/input preparation and does not require fresh source authority. Old records
lack the originally authored root; prove reconstructable retained fields conservatively and
never bind a new speculative intent to an unverifiable accepted record. Accepted work and its
Get/Collect paths remain independent of this validation.

Red/green proof uses actual NativeBackend behind pinned TLS and the production registered
listener: mismatched request/submission IDs, changed root/payload and equivalent JSON/key and
parameter ordering, credential refresh, plus exact replay after mutable preparation is absent.
No model inference or production serving qualification is claimed by these CPU controls.

CONFIRMED CPU red: the actual production serve process and NativeBackend accepted the
held root, then a changed request ID returned the original receipt. Evidence is
outputs/codex-machine-audit-20261004/api/legacy-submit-semantic-red.log. The fixture
performs acceptance/reattachment only: its generation deliberately lacks an executable SDK,
and any dispatched CPU executor fails before connecting. No inference succeeds or is claimed.

The implementation stores legacy_intent in the existing serialized SubmissionContext;
its default empty field preserves old records. New intent and application payload hashes use
boundary_json::intent_bytes, retaining u64 values and integer/float kind. The existing wire/JCS
writer remains unchanged. The optional package label on an installation alias is a checked
hint, not a second source choice. Model parameter order is normalized; input order values,
profile lists and ordered adapter stacks remain authored semantics. Expiring input access,
source credentials, transport Claim and memo hints are excluded.

The private replay_submit hook verifies current API Claim before validated existing replay,
and runs before Lifecycle admission. Other backends default to None; new work still takes the
existing admission guard and transactional acceptance. SubmissionContext equality includes
the intent, preventing concurrent conflicting accepts from sharing one durable row.

Old CPU installation-alias records can replay from retained installation/interface,
resolved capture/spec hashes and precise Invocation/input values after their generation is
removed. Precise values disambiguate even the old JCS payload-hash collision between adjacent
u64 seeds. If an older published/GPU record lacks enough originally authored choice data,
execution_replay_intent_unavailable refuses only this unprovable Submit replay; accepted work,
Get, explicit control and Collect remain accessible. This limitation neither refreshes source
authority nor makes another assignment. Old GPU/published replay needs its own retained-data
proof before expanding this fallback.

CONFIRMED CPU green: four actual legacy serve/TLS cases pass in1.11s, including both prior
census controls, both ID changes, changed root/entrypoint/release/payload/number kind,
semantic JSON and credential refresh, generation removal/restart, provable old metadata and
unprovable old-model metadata. Three semantic API cases pass in0.39s, including real
NativeBackend existing replay under activation, refusal of fresh work and preserved v1
reattachment/exact-result controls. Two unit controls prove model/input/adapter ordering and
concurrent journal acceptance. All library/test targets compile under clippy -D warnings.
Logs: legacy-submit-semantic-green.log, legacy-submit-freeze-and-v1.log,
legacy-submit-intent-unit.log and legacy-submit-all-tests-clippy.log under the same api output.
No production, public serving, provider or GPU qualification is implied.
