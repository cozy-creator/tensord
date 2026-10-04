# Bounded progress stage edges

The real3 GiB Anima4271 observer emitted positions1–29, then decoding and a successful
30-step terminal/artifact. Its native work completed30 steps. Source confirms two latest-only
windows: Rust has one ProgressSnapshot per execution, and Creator keeps latest by event kind
plus the first overall fraction. Back-to-back genuine final denoise/decode frames can replace
a stage endpoint before a reader polls. This is advisory observability, not request failure.

Retain at most8 genuine previous-stage endpoints plus the current latest sample per observed
execution/run, with existing per-sample byte limits. Stage changes capture the previously
emitted sample unchanged; no step, timestamp or completed count is invented. Machine event
pages expose those bounded endpoints by their actual observation sequences. Creator keeps the
received endpoints for live SSE and flushes them only with existing real state/product/outcome
transactions. No telemetry event creates a disk write or advances a durable cursor alone.

The same-stage flood remains latest-only, and count/byte caps bound arbitrary stage names.
A CPU burst must emit denoise29/30/decoding before the observer gets a turn, then prove both
actual machine and CLI/record consumers preserve emitted30 without per-event writes. Native
completion remains the work evidence; advisory delivery never becomes a success gate.
Host-load gate currently forbids new heavy builds; source/test preparation continues.
