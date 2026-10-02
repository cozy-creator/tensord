"""Reusable optional executor door with one bounded, acknowledged terminal outbox.

This is a separate module/capability from the single-use runner. No owner may treat
mutable spool filenames as output custody while this interpreter remains alive.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import importlib
import os
import queue
import socket
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .client import ProtocolError, read_exact
from .execution_protocol import MAX_EXECUTION_FRAME, Ready
from .output_custody import snapshot
from .packages import GenerationHold
from .runner import send
from .session_protocol import (
    COMMAND_DECODER, OpenSession, ReadyNext, SessionAck, SessionCancel, SessionCanceled,
    SessionError, SessionFailed, SessionInvoke, SessionOpened, SessionProgress, SessionResult,
    SessionShutdown,
)


def receive(sock: socket.socket):
    size = int.from_bytes(read_exact(sock, 4), "big")
    if not 0 < size <= MAX_EXECUTION_FRAME:
        raise ValueError("invalid session frame size")
    return COMMAND_DECODER.decode(read_exact(sock, size))


@dataclass
class Frame:
    event: object
    descriptors: tuple[int, ...] = ()
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


class Writer:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.queue: queue.Queue = queue.Queue(64)
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while (frame := self.queue.get()) is not None:
                try:
                    send(self.sock, frame.event)
                    for fd in frame.descriptors:
                        if self.sock.sendmsg([b"\0"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                                       array.array("i", [fd]))]) != 1:
                            raise ProtocolError("output descriptor was not sent")
                except Exception as exc:
                    frame.error = exc
                    raise
                finally:
                    frame.done.set()
        except Exception:
            self.failed.set()
            while True:
                try:
                    frame = self.queue.get_nowait()
                except queue.Empty:
                    break
                if frame is not None:
                    frame.error = BrokenPipeError("session writer ended")
                    frame.done.set()

    def progress(self, event):
        try:
            self.queue.put_nowait(Frame(event))
        except queue.Full:
            pass

    def send(self, event, descriptors=()):
        frame = Frame(event, tuple(descriptors))
        while not self.failed.is_set():
            try:
                self.queue.put_nowait(frame)
                break
            except queue.Full:
                self.failed.wait(0.01)
        if self.failed.is_set():
            raise BrokenPipeError("session writer ended")
        frame.done.wait()
        if frame.error is not None:
            raise frame.error

    def close(self):
        if not self.failed.is_set():
            self.queue.put(None)
        self.thread.join()


class Reader:
    def __init__(self, sock: socket.socket, session_id: str):
        self.sock, self.session_id = sock, session_id
        self.commands: queue.Queue = queue.Queue(16)
        self.lost = threading.Event()
        self.lock = threading.Lock()
        self.active = None
        self.floor = 0
        self.cancels: set[tuple[int, str]] = set()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while True:
                command = receive(self.sock)
                if isinstance(command, SessionCancel):
                    with self.lock:
                        if command.session_id != self.session_id or command.seq <= self.floor:
                            continue
                        key = (command.seq, command.execution_id)
                        if self.active is not None and key == self.active[:2]:
                            self.active[2].set()
                        elif len(self.cancels) < 64:
                            self.cancels.add(key)
                else:
                    self.commands.put_nowait(command)
        except Exception:
            self.lost.set()
            with contextlib.suppress(queue.Full):
                self.commands.put_nowait(None)

    def begin(self, seq: int, execution_id: str) -> threading.Event:
        canceled = threading.Event()
        with self.lock:
            self.active = (seq, execution_id, canceled)
            if (seq, execution_id) in self.cancels:
                self.cancels.remove((seq, execution_id))
                canceled.set()
        return canceled

    def finish(self, seq: int):
        with self.lock:
            self.active = None
            self.floor = seq
            self.cancels = {key for key in self.cancels if key[0] > seq}


def run(sock: socket.socket):
    send(sock, Ready(os.getpid(), ["runtime.author-session/1", "outputs.sealed-fd/1"]))
    opened = receive(sock)
    if not isinstance(opened, OpenSession):
        raise ValueError("reusable runner requires OpenSession")
    with GenerationHold(Path(sys.prefix).parent) as generation:
        application = opened.module if ":" in opened.module else opened.module + ":app"
        if (opened.generation, opened.package, application) != (
                generation.identity, generation.package, generation.application):
            raise ValueError("session does not match its held environment generation")
        send(sock, SessionOpened(opened.session_id))
        reader, writer = Reader(sock, opened.session_id), Writer(sock)
        last_seq = 0
        terminal = None
        descriptors: list[int] = []
        sdk_imports = package_imports = 0
        module_name = application.split(":", 1)[0]
        try:
            while not reader.lost.is_set():
                command = reader.commands.get()
                if command is None or reader.lost.is_set():
                    break
                if command.session_id != opened.session_id:
                    writer.send(SessionError(opened.session_id, "session_mismatch",
                                             "command belongs to another session"))
                    continue
                if isinstance(command, SessionShutdown):
                    break  # idle shutdown also abandons the in-memory unacked outbox
                if isinstance(command, SessionAck):
                    if (terminal is None or command.seq != terminal.seq
                            or command.execution_id != terminal.execution_id):
                        writer.send(SessionError(opened.session_id, "ack_mismatch",
                                                 "ack does not match the outstanding terminal",
                                                 command.seq))
                        continue
                    for fd in descriptors:
                        os.close(fd)
                    descriptors.clear()
                    terminal = None
                    writer.send(ReadyNext(opened.session_id, last_seq, sdk_imports, package_imports,
                                          id(sys.modules.get("cozy_runtime.author")),
                                          id(sys.modules.get(module_name))))
                    continue
                if not isinstance(command, SessionInvoke):
                    writer.send(SessionError(opened.session_id, "unsupported_operation",
                                             "operation does not belong to an opened session"))
                    continue
                invocation = command.invocation
                if command.seq <= last_seq:
                    if (terminal is not None and command.seq == terminal.seq
                            and invocation.execution_id == terminal.execution_id):
                        writer.send(terminal, descriptors)
                    else:
                        writer.send(SessionError(opened.session_id, "attempt_already_settled",
                                                 "this sequence cannot execute again", command.seq))
                    continue
                if terminal is not None:
                    writer.send(SessionError(opened.session_id, "custody_ack_required",
                                             "acknowledge the previous terminal before invoking",
                                             command.seq))
                    continue
                requested_application = (invocation.module if ":" in invocation.module else
                                         invocation.module + ":app")
                if (invocation.generation, invocation.package, requested_application) != (
                        generation.identity, generation.package, application):
                    writer.send(SessionError(opened.session_id, "generation_mismatch",
                                             "invocation changed the session generation",
                                             command.seq))
                    continue
                last_seq = command.seq
                canceled = reader.begin(command.seq, invocation.execution_id)

                def progress(event, seq=command.seq, execution_id=invocation.execution_id):
                    writer.progress(SessionProgress(opened.session_id, seq, execution_id,
                                                    event.completed_units, event.detail))

                had_sdk = "cozy_runtime.author" in sys.modules
                had_package = module_name in sys.modules
                try:
                    bridge = importlib.import_module("cozy_machine_client.runtime_bridge")
                    outputs = []
                    result = bridge.execute(invocation, canceled, progress,
                                            artifact_sink=outputs.append)
                    artifacts = []
                    for facts in outputs:
                        artifact, fd = snapshot(Path(invocation.output_root), facts)
                        artifacts.append(artifact)
                        descriptors.append(fd)
                    terminal = SessionResult(opened.session_id, command.seq, invocation.execution_id,
                                             result.value, artifacts)
                except Exception as exc:
                    bridge = sys.modules.get("cozy_machine_client.runtime_bridge")
                    code = bridge.error_code(exc) if bridge is not None else type(exc).__name__
                    terminal = SessionFailed(opened.session_id, command.seq, invocation.execution_id,
                                             code, str(exc))
                finally:
                    sdk_imports += int(not had_sdk and "cozy_runtime.author" in sys.modules)
                    package_imports += int(not had_package and module_name in sys.modules)
                if canceled.is_set():
                    terminal = SessionCanceled(opened.session_id, command.seq,
                                               invocation.execution_id)
                if not isinstance(terminal, SessionResult):
                    for fd in descriptors:
                        os.close(fd)
                    descriptors.clear()
                reader.finish(command.seq)
                if reader.lost.is_set():
                    break  # owner death never starts another queued invocation
                writer.send(terminal, descriptors)
        finally:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            writer.close()
            reader.thread.join()
            for fd in descriptors:
                os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-fd", type=int, required=True)
    args = parser.parse_args()
    with socket.socket(fileno=args.execution_fd) as sock:
        with contextlib.suppress(EOFError, ProtocolError, BrokenPipeError):
            run(sock)


if __name__ == "__main__":
    main()
