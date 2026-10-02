# CPU Runtime package bridge

The trusted runner sends `Ready` before importing Runtime or authored package code.
The owner records the child process birth and commits start authorization, then sends
`Invoke`. The runner calls the existing Runtime author `prepare` and `invoke` functions;
it does not invoke a registered Python function directly, copy the worker, or implement
another SDK. `Attempt` and pending-artifact inspection are narrow adapters to Runtime's
author internals and need capability qualification when the SDK evolves.

## Environment and source description

`cozy_machine_client.packages.describe(project, environment_python=None)` delegates to
the installed Runtime's static source reader in a trusted SDK subprocess. Its AST path
never imports package code. The reader's output remains opaque `msgspec.Raw` JSON; only
consumed pyproject name/version/application fields are normalized into typed records.
Unknown extras stay harmless and are never business-indexed through untyped dictionaries.
The gate adds a module-level exception to the real classifier package and still obtains
its interface. An explicit wheel build is a different operation: it may execute the
package's build backend and must run in the packaging containment tier.

`install(project, generations, client_wheel, python=...)` builds a package wheel, then
installs that wheel and the client wheel in a new uv environment. The dependency resolver
honors the package's lower and upper bounds; the service's SDK version is not injected.
It publishes typed `generation.json` after a successful install and static description.
The manifest lists immutable identity, package/version, application, interpreter,
resolved dependency versions and interface. Dispatch never updates that environment.

```sh
uv build --wheel --out-dir outputs/client
uv run --with 'cozy-runtime>=0.18.89,<0.19' python -m cozy_machine_client.packages install \
  tests/fixtures/cpu_classifier --generations outputs/generations \
  --client-wheel outputs/client/cozy_machine_client-0.1.0-py3-none-any.whl
```

The owner acquires a shared flock on `<generation>/.hold` before reading its manifest
and retains it through child exit. The runner independently holds the same file while
authored code runs, protecting the environment even when the core dies. `collect` takes
an exclusive nonblocking hold, verifies the published identity, removes its public name,
then reclaims it. The scheduler still needs policy for stale environments and cache
pressure; this module supplies the safe generation lifetime primitive.

## Engine interface

Use the manifest's interpreter with `-m cozy_machine_client.runner --execution-fd N`.
The optional SDK adapter has ordinary top-level imports in `runtime_bridge`; that trusted
module is loaded only after Invoke authorization. The base client and runner import
successfully without Runtime installed.
The inherited stream carries a network-order four-byte length and typed JSON, at most
1 MiB per envelope. Unknown advisory fields are accepted. No version equality is tested.

| Direction | Record |
| --- | --- |
| Child → owner | `ready {pid, capabilities:["runtime.author-cpu/1"]}` |
| Owner → child | `invoke {execution_id,package,generation,module,entrypoint,input,output_root}` |
| Owner → child | `cancel {execution_id}` |
| Child → owner | `progress {execution_id,completed_units,detail}` |
| Child → owner | `result {execution_id,value,artifacts:[relative spool filename]}` |
| Child → owner | `failed {execution_id,code,detail}` or `canceled {execution_id}` |

`generation` is the published identity; `module` is the declared `cozy.application`
value (`module:app`, or its bare module spelling). Package identity and application must
match the held installed generation. `output_root` is an absolute owner-created spool.
The owner validates and seals artifacts, commits durable output custody, then acknowledges
success. A runner result alone is tentative. A completed-unit telemetry queue is bounded
and lossy; model code never blocks on a telemetry socket write.

Runtime `ProgressFrame.advance` is an absolute **event** sequence in the qualified SDK,
including stage-open and fraction-only frames. It is not completed work. The adapter
accumulates positive `position` deltas with valid `total` per stage/child-attempt, ignores
repeated or regressed positions, and bounds remembered scopes to 256. Real inference
checks produce completed totals `[1, 2]` for two steps. The SDK lacks an identity for
restarted same-named scopes, so their resets are conservatively uncounted. Missing lossy
telemetry alone must never authorize a kill; richer progress identity remains a liveness
qualification requirement.

Only an attempt-matching explicit Cancel sets the Runtime cancellation callback. Parent
socket EOF does not manufacture user cancellation or dispatch another invocation: code
already entered may finish, and the owner must conservatively settle orphaned started
work after child exit unless output custody was already durable. No wall-clock kill is
implemented. This runner handles one attempt and exits; executor reuse is still open.

## Qualification and limits

Tests install an actual sklearn logistic classifier as a wheel in its own generation,
invoke it through the SDK/process/socket path, compare classifications and saved bytes
across three executors, cancel an active computation, exercise owner EOF, and check held
environment reclamation. Cancellation uses real repeated classifier inference, not a
mock function or sleep. `/proc/<pid>/maps` checks cover readiness and active inference
without CUDA/NVML library loads. The installed Runtime 0.18.99 still depends on TensorFS
0.3.88; this slice does **not** establish removal of that package dependency.

On 2026-10-02, all 14 package bridge and 14 transport tests passed in 19.63 s with CPython 3.12.12,
Runtime 0.18.99, TensorFS 0.3.88, sklearn 1.9.1 and NumPy 2.5.3. An independent
package pinned Runtime **0.18.89** and ran the same real inference successfully through
the unchanged runner, without an injected service-version floor. Bytecode compilation
and whitespace checks passed. The uv CPython 3.12 build lacks `os.memfd_create`; a small
Linux libc adapter now performs the same kernel operation and seals with preserved errno
and fd lifetime. All 14 transport tests also passed on CPython 3.14.7, which uses its native
os function. Direct libc tests on both interpreters verify seals, CLOEXEC and error paths.

Deferred media encoding, output trees, committed/child-call artifacts, model loading,
checkpoint publication, secrets, remote assets, subcalls, executor reuse and GPU/group
execution are not implemented by this door. Unsupported output modes fail the operation
instead of falsely reporting retained bytes. The current 1 MiB envelope and SDK output
ceiling need the existing spooled-result path before broad package qualification.

This proves a package bridge, not the existing full Runtime executor, ordinary Creator
CLI, browser, pod-supervisor or Hub compatibility gate. Do not activate this runner for
unqualified packages or replace the old stack from these component tests alone.
