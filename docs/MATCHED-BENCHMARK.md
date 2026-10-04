# Matched controller-output benchmarks

Work: tracker #319, with memory/quality follow-up #320. The strict reporter from
machine #35 requires at least three complete paired measurements in every declared
cell, no failed requests, matched hardware/work, and upper95% median time ratio below1.
No historical smoke data acquires missing reference evidence retroactively.

Both adapters run on the controller. Cozy uses ordinary default-home `cozy run`
with the rental selector, while ComfyUI uses its stock HTTP API. Each submits one
request, saves its output on the controller, then submits the next. Cell timing starts
immediately before the first actual CLI invocation/HTTP submission and ends after
all requested output bytes are saved. Decode/hash/quality checks occur afterwards.
Engine server startup and environment preparation are separately recorded; they are
not substituted for the submit boundary. No timeout kills or smaller requests.

Every cell emits engine name/actual commit probe, stable pair ID, measured GPU UUID
and driver, normalized authored requests and model tensor provenance, actual controller
saved artifacts, and quality against that engine's own unconstrained controls.
Cross-engine noise generators differ and cross-engine pixel equality is not claimed.

The frozen F driver uses Comfy Anima Euler/simple, while the actual package uses
Diffusers FlowMatchEulerDiscreteScheduler with shift3. Before declaring a match,
compare actual sigma arrays and prediction/CFG/conditioning settings. Comfy normal
appears to construct the matching transformed endpoints, but that must be proved from
actual source/config; a common step count is insufficient. SDXL also needs its exact
EulerDiscrete configuration and epsilon conversion checked. Use every tensor's actual
logical name/shape/dtype/value digest, including encoders/conditioner/VAE, rather than
an assumed checkpoint filename or a manually supplied parity-confirmed flag.

Same-engine quality defaults to exact RGB pixel equality. A declared tolerance is
allowed only when recorded unconstrained repetitions establish normal variance first.
35dB is a proposed lower bound (8-bit RGB RMS error about4.53levels), not a measured
qualification. Shape/nonflat output checks remain smoke checks. Controls must have
the exact same authored request/model provenance and software/hardware; constrained
outputs never serve as their own references. Preserve artifact and pixel hashes.
