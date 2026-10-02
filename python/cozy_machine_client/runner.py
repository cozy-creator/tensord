"""Trusted CPU process door into the existing Runtime author invocation kernel.

No Runtime or authored module is imported until Invoke follows Ready. The Rust owner
commits process identity and start authorization at that boundary. This runner owns
neither a store nor a journal. Outputs remain tentative until the owner takes custody.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import os
import queue
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from .execution_protocol import (
    COMMAND_DECODER, MAX_EXECUTION_FRAME, Cancel, Canceled, Failed, Invoke, Progress, Ready,
    Result,
)
from .packages import GenerationHold

if TYPE_CHECKING:
    from cozy_runtime.author._services import ProgressFrame


class CompletedWork:
    """Count completed step positions, never SDK event sequence or stage-open churn."""

    def __init__(self):
        self.positions: dict[tuple[str, str | None, int | None], int] = {}
        self.units = 0
        self.lock = threading.Lock()

    def observe(self, frame: ProgressFrame) -> int | None:
        position, total = frame.position, frame.total
        if (type(position) is not int or type(total) is not int
                or not 0 < position <= total):
            return None
        key = (frame.stage, frame.call_request, frame.call_attempt)
        with self.lock:
            previous = self.positions.get(key, 0)
            if position <= previous or (key not in self.positions and len(self.positions) >= 256):
                return None
            self.positions[key] = position
            self.units += position - previous
            return self.units


def receive(sock: socket.socket):
    def exact(size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            part = sock.recv(size - len(result))
            if not part:
                raise EOFError("machine execution socket closed")
            result.extend(part)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    if size == 0 or size > MAX_EXECUTION_FRAME:
        raise ValueError("invalid execution frame size")
    return COMMAND_DECODER.decode(exact(size))


def send(sock: socket.socket, event: object) -> None:
    encoded = msgspec.json.encode(event)
    if len(encoded) > MAX_EXECUTION_FRAME:
        raise ValueError("execution frame exceeds negotiated baseline size")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)


class EventWriter:
    """A bounded lossy progress lane; no telemetry socket write runs on model code."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.frames: queue.Queue = queue.Queue(64)
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while (frame := self.frames.get()) is not None:
                send(self.sock, frame)
        except (OSError, ValueError):
            self.failed.set()

    def progress(self, event: Progress):
        try:
            self.frames.put_nowait(event)
        except queue.Full:
            pass

    def finish(self, terminal: object):
        # The receiver must drain the channel. Peer death closes the socket instead
        # of authorizing a clock-based kill or inventing a successful result.
        while not self.failed.is_set():
            try:
                self.frames.put_nowait(terminal)
                break
            except queue.Full:
                self.failed.wait(0.01)
        while not self.failed.is_set():
            try:
                self.frames.put_nowait(None)
                break
            except queue.Full:
                self.failed.wait(0.01)
        self.thread.join()


def execute(command: Invoke, canceled: threading.Event, writer: EventWriter):
    # The child also owns a generation hold: losing the core's lock cannot let a
    # concurrent collector remove the environment while its package is running.
    root = Path(sys.prefix).parent
    with GenerationHold(root) as generation:
        application = command.module if ":" in command.module else command.module + ":app"
        if (generation.identity != command.generation or generation.package != command.package
                or generation.application != application):
            raise ValueError("invocation does not match this held environment generation")
        return _execute_author(command, canceled, writer)


def _execute_author(command: Invoke, canceled: threading.Event, writer: EventWriter):
    # Public Runtime imports are intentionally inside the authorized invocation.
    from cozy_runtime.author import App, Asset, Device, Invocation, invoke, prepare
    from cozy_runtime.author._services import Attempt, ProgressFrame

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
                writer.progress(Progress(command.execution_id, units, frame.stage))

    prepared = prepare(registration, command.input)
    record = Attempt(command.execution_id, spool, sink=progress)
    invocation = Invocation(command.execution_id, spool, float("inf"), device=Device("cpu"),
                            cancel=canceled.is_set, progress=progress)
    try:
        result = invoke(prepared, invocation, record)
    finally:
        record.closed = True
    if record.frames or record.pending_trees or record.committed_files:
        # The deferred media codecs/tree custody are not implemented by this CPU
        # door. Never acknowledge success with unencoded or unretained artifacts.
        raise ValueError("this CPU executor does not support deferred media or tree outputs")
    artifacts = []
    for asset in record.pending.values():
        local = asset._local
        if local is None or local.parent != spool or local.is_symlink() or not local.is_file():
            raise ValueError("output is not a single relative spool file")
        artifacts.append(local.name)

    def asset_wire(value):
        if isinstance(value, Asset):
            row = value.row()
            return {"asset_ref": row.pop("ref"), **row}
        raise TypeError(f"{type(value).__name__} has no result wire representation")

    return Result(command.execution_id,
                  msgspec.to_builtins(result.result, enc_hook=asset_wire), artifacts)


def run(sock: socket.socket) -> int:
    send(sock, Ready(os.getpid(), ["runtime.author-cpu/1"]))
    try:
        command = receive(sock)
    except EOFError:
        return 0
    if not isinstance(command, Invoke):
        raise ValueError("the first authorized command must be Invoke")
    canceled = threading.Event()
    disconnected = threading.Event()
    writer = EventWriter(sock)

    def controls():
        try:
            while True:
                update = receive(sock)
                if isinstance(update, Cancel) and update.execution_id == command.execution_id:
                    canceled.set()
                elif isinstance(update, Invoke):
                    raise ValueError("one runner executes one authorized attempt")
        except (EOFError, OSError, ValueError, msgspec.ValidationError):
            # EOF is loss of owner, not user cancellation. Already-started authored
            # code may finish; no new work is invoked, no result is claimed durable.
            disconnected.set()

    threading.Thread(target=controls, daemon=True).start()
    try:
        terminal = execute(command, canceled, writer)
    except Exception as exc:
        try:
            from cozy_runtime.author import classify

            code = classify(exc).code or type(exc).__name__
        except ImportError:
            code = "runtime_sdk_unavailable"
        terminal = Failed(command.execution_id, code, str(exc))
    if canceled.is_set():
        terminal = Canceled(command.execution_id)
    if disconnected.is_set():
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
    writer.finish(terminal)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-fd", type=int, required=True)
    args = parser.parse_args()
    with socket.socket(fileno=args.execution_fd) as sock:
        raise SystemExit(run(sock))


if __name__ == "__main__":
    main()
