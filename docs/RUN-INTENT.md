# Semantic run identity and application integers

An actor's run id names one authored operation. Reattachment must preserve that operation
while allowing JSON whitespace/object-key ordering and refreshed transport access. The intent
retains source, entrypoint, typed payload values, input order/metadata, ordered model choices,
attention selection, owner and weights destination. It excludes provider/publication credentials,
Hub token/expiry/CA/object-host hints, advisory binding revision and known memo results. The
Hub origin remains the source registry and is normalized only for its trailing slash.

Application JSON is normalized with exact integer precision and number kind. JCS casts i64/u64
to f64: seeds 9007199254740992 and 9007199254740993 therefore collide. They must remain distinct,
as must integer and float values when typed consumers distinguish them. Child intent uses the
same normalization. Arbitrary inline and child result values retain their exact integer values;
protocol envelope canonicalization keeps its existing profile.

Existing intent is checked before validating fresh Hub access or looking for old input blobs.
A matching accepted run attaches even when its prior credentials or blobs no longer exist.
New runs still validate fresh access and owned input custody. A changed source/payload/model
choice or input ordering remains a conflict and never replaces or replays accepted work.

Older experimental root records retain only their raw spec digest, without the complete
original authored source/model intent. Only an unchanged old digest can be proven compatible;
no-spec attach continues to work. Reformatted/refreshed specs against those old records cannot
always be proven equivalent, and the implementation preserves conflict checks rather than
silently accepting changed models. New rows use semantic intent.

Old finished-child digests are admitted only when module/export's prior digest and the stored
precise input, bound files/order, parent and entrypoint all match. This preserves unchanged old
children without allowing the known large-seed collision to replay the wrong result.

CPU checks exercise semantic formatting/access refresh, true conflict arms, input-less
reattachment to accepted work, finished-child collision rejection and exact uint64 result
transport. They do not qualify inference, ordinary CLI version skew or GPU performance.
