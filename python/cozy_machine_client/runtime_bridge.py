"""Optional Runtime SDK adapter, loaded only after the owner's start authorization."""
import importlib
import importlib.metadata
import threading
from collections.abc import Callable
from pathlib import Path

import msgspec
from cozy_runtime.author import App, Asset, Device, Invocation, classify, invoke, prepare
from cozy_runtime.author._services import Attempt, ProgressFrame

from .execution_protocol import AssetBinding, Invoke, OutputChecksum, OutputFacts, Progress, Result
from .progress import CompletedWork


def error_code(exc: Exception) -> str:
    return classify(exc).code or type(exc).__name__


def asset_wire(value: object) -> object:
    if isinstance(value, Asset):
        row = value.row()
        return {"asset_ref": row.pop("ref"), **row}
    raise TypeError(f"{type(value).__name__} has no result wire representation")


def output_checksum(value: str) -> OutputChecksum:
    prefix, separator, digest = value.partition(":")
    if not separator or prefix not in ("sha256", "blake2b"):
        raise ValueError("SDK output checksum algorithm is not supported by this operation")
    return OutputChecksum("blake2b-128" if prefix == "blake2b" else "sha256", digest)


def execute(command: Invoke, canceled: threading.Event,
            progress_sink: Callable[[Progress], None], *,
            artifact_sink: Callable[[OutputFacts], None] | None = None) -> Result:
    distribution = importlib.metadata.distribution(command.package)
    applications = [entry.value for entry in distribution.entry_points
                    if entry.group == "cozy.application"]
    application = command.module if ":" in command.module else command.module + ":app"
    if applications != [application]:
        raise ValueError("invocation application is not the installed package declaration")
    module_name, export = application.split(":", 1)
    module = importlib.import_module(module_name)
    app = getattr(module, export)
    if not isinstance(app, App):
        raise ValueError("installed application must export a Runtime App")
    registration = app.get(command.entrypoint)
    if registration.internal or registration.hidden:
        raise ValueError("private or hidden registration is not an external entrypoint")
    spool = Path(command.output_root)
    if not spool.is_absolute() or spool.is_symlink():
        raise ValueError("output_root must be an absolute owned directory")
    spool.mkdir(parents=True, exist_ok=True)
    completed = CompletedWork()

    def progress(frame):
        if isinstance(frame, ProgressFrame):
            units = completed.observe(frame)
            if units is not None:
                progress_sink(Progress(command.execution_id, units, frame.stage))

    prepared = prepare(registration, command.input)
    record = Attempt(command.execution_id, spool, sink=progress)
    invocation = Invocation(command.execution_id, spool, float("inf"), device=Device("cpu"),
                            cancel=canceled.is_set, progress=progress)
    try:
        result = invoke(prepared, invocation, record)
    finally:
        record.closed = True
    if record.frames or record.pending_trees or record.committed_files:
        raise ValueError("this CPU executor does not support deferred media or tree outputs")
    artifacts = []
    bindings = []
    for asset in record.pending.values():
        local = asset._local
        if local is None or local.parent != spool or local.is_symlink() or not local.is_file():
            raise ValueError("output is not a single relative spool file")
        artifacts.append(local.name)
        row = asset.row()
        bindings.append(AssetBinding(local.name, row["ref"], row["media_type"],
                                     output_checksum(asset.digest), asset.size_bytes))
        if artifact_sink is not None:
            artifact_sink(OutputFacts(local.name, output_checksum(asset.digest), asset.size_bytes))
    return Result(command.execution_id,
                  msgspec.to_builtins(result.result, enc_hook=asset_wire), artifacts, bindings)
