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
