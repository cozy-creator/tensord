# Bounded progress stage edges

The real 3 GiB Anima4271 observer emitted positions 1–29, then decoding and a successful
30-step terminal/artifact. Its native work completed 30 steps. Source confirms two latest-only
windows: Rust has one ProgressSnapshot per execution, and Creator keeps latest by event kind
plus the first overall fraction. Back-to-back genuine final denoise/decode frames can replace
a stage endpoint before a reader polls. This is advisory observability, not request failure.

Retain at most 8 genuine previous-stage endpoints plus the current latest sample per observed
execution/run, with existing per-sample byte limits. Stage changes capture the previously
emitted sample unchanged; no step, timestamp or completed count is invented. Machine event
pages expose those bounded endpoints by their actual observation sequences. Creator keeps the
received endpoints for live SSE and flushes them only with existing real state/product/outcome
transactions. No telemetry event creates a disk write or advances a durable cursor alone.

The same-stage flood remains latest-only, and count/byte caps bound arbitrary stage names.
A CPU burst must emit denoise29/30/decoding before the observer gets a turn, then prove both
actual machine and CLI/record consumers preserve emitted30 without per-event writes. Native
completion remains the work evidence; advisory delivery never becomes a success gate.
The additive `Execution.progress_samples` field defaults empty for older journals. Each
sample carries its actual allocated observation revision, units, bounded detail and the time
the machine received it. A source timestamp absent from the incoming frame is not invented.
The current stage replaces only its own last sample; changing stage preserves that sample.
Preparation binding retains endpoints but resets inference's work units to zero. Finish and
other existing observed transitions store the projection; the existing rare observation
cursor-window allocation remains, with no new progress transaction or periodic flush.

Prepared checks in `tests/progress_stage_edges.rs` cover a 2,000-frame same-stage burst,
100 distinct stages, default-empty older records, preparation unit reset, a real TLS Run
observer, preserved timestamps/cursors at terminal, and the reopened journal. The authored
CPU callable `tests/fixtures/cpu_progress_burst` emits actual SDK callbacks 0–29, immediately
emits decoding, and returns 30. Creator's ordinary CLI test submits it on the real machine,
then watches the stored run before and after restarting only its isolated controller.

Source checkpoint only: these new tests have not run while host load exceeds 10 and free
space is below 80 GiB. Rented GPU delivery, congested-disk timing and the ComfyUI matrix
remain separate gates; this advisory projection cannot qualify any of them.
