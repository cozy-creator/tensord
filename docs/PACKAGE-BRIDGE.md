# Package bridge

`python/cozy_machine_client` builds immutable uv environment generations and runs one CPU attempt
in one through the Runtime author kernel (`prepare` + `invoke`). The wire and journal side are in
`DURABLE-EXECUTION.md`.

## Generations (`packages.py`, `captured_packages.py`)

Layout: `<generations>/<32-hex>/` with `env/`, `.hold` and `generation.json` (`identity`, `package`,
`version`, `application`, `python`, `dependencies`, `interface`). `generation.json` is published
atomically last. A generation is never updated. `catalog.rs` resolves one only under a shared
`.hold` flock.

- `describe(project)`: the Runtime static reader runs in a subprocess, AST only. It never imports
  package code.
- `install`: `uv build --wheel`, then a new venv with the package and client wheels. The package's
  own bounds apply. No SDK version is injected.
- `install-captured` (`python -m cozy_machine_client.installer`, called by `api/install.rs`):
  - Python comes from `uv python find` and must match the captured `python_requires`/`python_version`.
  - Locked source: `uv sync --frozen --no-dev --no-editable`.
  - Hashed requirements: `--require-hashes`, then the wheels with `--no-deps`.
  - Anything else: `uv pip install` of the project and/or wheels.
  - A locked source mixed with wheels or requirements is refused.
  - The machine's own Runtime/TensorFS pair (`--sdk-wheel`) replaces the captured pair, every other
    version constrained, where `uv pip check` admits it. Else the capture's own pair (a vendored dev
    Runtime, say) is restored and `sdk_fallback` says why; every run of it logs that warning.
  - The client wheel is installed under constraints pinning every selected package. Any change is
    `package_lock_changed`. Then `uv pip check` runs.
  - The interface is described inside the env with `python -I -S` (no `.pth` execution).
  - Failure prints `{kind: "install_failed", code, detail}`. `install.rs` passes the helper's
    stderr to the machine log and quotes its last lines (credentials masked) in the refusal when
    the failure is an operation's (`package_dependency_operation_failed`) or untyped.
- `collect`: takes an exclusive non-blocking `.hold`, renames the generation, then deletes it.

## Calls between packages

A generation also records `callees`: the other installed distributions with a statically
described `cozy.application`, their application, interface, package identity and source digest.
Their child runs use the caller's immutable environment and the callee's own App. A callee job
can call its own internal exports; another package can call only its public exports. Memoized
calls use the callee's source digest.

A local-source manifest may carry an optional `callees` map from normalized distribution names
to package identities. It may carry a source archive, or an immutable root wheel together with
dependency wheels. A published release's map is its lock: every row its Hub publishes is
`<release org>/<row name>` (a Hub lock pins only its own org). A wheel's URL never names a
package; anything unmapped, and older records without a package identity, read as
`local/<normalized-distribution>`. Description
reads metadata and AST inside the environment without importing package code.

Model slots and resolution-cache entries belong to the callee package. The caller's model
choices do not override another package's slots; a choice addressed to a callee slot names it. GPU
execution uses the callee's interface; import-only parents are keyed by generation and App.
These are additive generation and manifest fields; `cozy.machine.v1` is unchanged.
An export registered by distinct Apps is refused only when that ambiguous call is made;
other calls and same-App aliases continue to work.

## Runner (`runner.py`, `runtime_bridge.py`)

`<generation python> -I -m cozy_runtime.internal.trampoline … -- <generation python> -m
cozy_machine_client.runner --execution-fd N` (see durable execution).

1. Send `Ready` before importing Runtime or package code.
2. Accept exactly one `Invoke`.
3. Hold its own generation, and require the held identity, package and application to equal `Invoke`.
4. Require the module to be the distribution's single `cozy.application` entry point and the
   entrypoint not internal or hidden.
5. Run `prepare` + `invoke` on CPU. Each output asset must be one regular file in the spool.

Rules:
- Progress counts positive `position` deltas per (stage, call request, call attempt), up to 256
  keys. SDK event sequence numbers are not work.
- Progress goes through a bounded queue of 64 that drops when full.
- Only a matching `Cancel` sets the cancel callback. A dead owner kills the runner (parent-death
  signal); a closed channel alone lets running code finish, and its result is not durable.
- One runner, one attempt.

## Not implemented

- Deferred media, output trees and committed files fail the attempt.
- Model loading, child calls, remote assets, secrets and executor reuse.
- No spooled-result path for results over the 1 MiB frame.
- Python `Invoke.input` must be a JSON object. Rust allows any JSON value.
