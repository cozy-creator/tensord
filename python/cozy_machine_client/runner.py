"""Trusted CPU process door into the existing Runtime author invocation kernel.

No Runtime or authored module is imported until Invoke follows Ready. The Rust owner
commits process identity and start authorization at that boundary. This runner owns
neither a store nor a journal. Outputs remain tentative until the owner takes custody.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import queue
import socket
import struct
import sys
import threading
from pathlib import Path

import msgspec

from .execution_protocol import (
    COMMAND_DECODER, MAX_EXECUTION_FRAME, Cancel, Canceled, Failed, Invoke, Progress, Ready,
)
from .packages import GenerationHold


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
        bridge = importlib.import_module("cozy_machine_client.runtime_bridge")
        return bridge.execute(command, canceled, writer.progress)


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
        bridge = sys.modules.get("cozy_machine_client.runtime_bridge")
        code = (bridge.error_code(exc) if bridge is not None else
                "runtime_sdk_unavailable" if isinstance(exc, ImportError) else type(exc).__name__)
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
