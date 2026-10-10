# Rental lifetime

Only an unused rental expires automatically after its fixed 900-second window. Once the machine accepts work or observes real preparation/execution, the durable used state prevents automatic release across service and SDK restarts. Used rentals end through explicit owner/provider action.

Status polling and software maintenance do not count as application use. Unknown activity holds release while it is unknown; it does not permanently convert an unused rental into a used one. A pending authenticated submission holds the release admission fence until acceptance has durably recorded use. The authoritative status reports no automatic deadline for a used rental.

Client-side source capture before any request reaches the machine is a separate preparation boundary; it cannot be inferred from machine activity. The controller must preserve that pending preparation and must not present a local settlement estimate as the worker's authoritative deadline.
