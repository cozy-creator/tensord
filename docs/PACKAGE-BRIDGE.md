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
- `install-captured` (called by `api/install.rs`):
  - Python comes from `uv python find` and must match the captured `python_requires`/`python_version`.
  - Locked source: `uv sync --frozen --no-dev --no-editable`.
  - Hashed requirements: `--require-hashes`, then the wheels with `--no-deps`.
  - Anything else: `uv pip install` of the project and/or wheels.
  - A locked source mixed with wheels or requirements is refused.
  - The client wheel is installed under constraints pinning every selected package. Any change is
    `package_lock_changed`. Then `uv pip check` runs.
  - The interface is described inside the env with `python -I -S` (no `.pth` execution).
  - Failure prints `{kind: "install_failed", code, detail}`.
- `collect`: takes an exclusive non-blocking `.hold`, renames the generation, then deletes it.

## Runner (`runner.py`, `runtime_bridge.py`)

`<generation python> -m cozy_machine_client.runner --execution-fd N`

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
- Only a matching `Cancel` sets the cancel callback. EOF is owner loss: running code may finish,
  and its result is not durable.
- One runner, one attempt.

## Not implemented

- Deferred media, output trees and committed files fail the attempt.
- Model loading, child calls, remote assets, secrets and executor reuse.
- No spooled-result path for results over the 1 MiB frame.
- Python `Invoke.input` must be a JSON object. Rust allows any JSON value.
