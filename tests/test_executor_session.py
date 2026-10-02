"""Real reusable interpreter/SDK inference and sealed output-custody experiments."""
import fcntl
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
from pathlib import Path

import msgspec
import pytest

from cozy_machine_client.client import FULL_SEALS, GET_SEALS, read_exact, receive_fd
from cozy_machine_client.execution_protocol import Invoke, OutputChecksum, OutputFacts, Ready
from cozy_machine_client.output_custody import snapshot
from cozy_machine_client.packages import GenerationHold, install
from cozy_machine_client.session_protocol import (
    Event, OpenSession, ReadyNext, SessionAck, SessionCancel, SessionCanceled, SessionError,
    SessionInvoke, SessionOpened, SessionProgress, SessionResult, SessionShutdown,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/cpu_classifier"
SAMPLES = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 5.1, 1.6], [6.7, 3.1, 4.7, 1.5]]
DECODER = msgspec.json.Decoder(Ready | Event)


@pytest.fixture(scope="module", params=["current", "older"])
def generation(request, tmp_path_factory):
    pytest.importorskip("cozy_runtime")
    output = tmp_path_factory.mktemp("executor-session-" + request.param)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output / "client"), str(ROOT)],
                   check=True)
    source = FIXTURE
    if request.param == "older":
        source = output / "older-package"
        shutil.copytree(FIXTURE, source)
        project = source / "pyproject.toml"
        project.write_text(project.read_text().replace(
            "cozy-runtime>=0.18.89,<0.18.101", "cozy-runtime==0.18.89"))
    generation = install(source, output / "generations",
                         next((output / "client").glob("*.whl")))
    with GenerationHold(Path(generation.python).parents[2]) as held:
        yield held


class Session:
    def __init__(self, generation):
        self.generation = generation
        self.socket, child = socket.socketpair()
        self.socket.settimeout(30)  # observation bound for a broken test, never a process kill
        self.process = subprocess.Popen(
            [generation.python, "-m", "cozy_machine_client.session_runner", "--execution-fd",
             str(child.fileno())], pass_fds=[child.fileno()], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        child.close()
        ready, fds = self.receive()
        assert isinstance(ready, Ready) and ready.pid == self.process.pid and not fds
        assert ready.capabilities == ["runtime.author-session/1", "outputs.sealed-fd/1"]
        maps = Path(f"/proc/{self.process.pid}/maps").read_text()
        assert "numpy" not in maps and "libcuda" not in maps
        self.id = "session-" + str(self.process.pid)
        self.send(OpenSession(self.id, generation.package, generation.identity,
                              generation.application))
        assert isinstance(self.receive()[0], SessionOpened)

    def send(self, command):
        data = msgspec.json.encode(command)
        self.socket.sendall(struct.pack("!I", len(data)) + data)

    def receive(self):
        size = int.from_bytes(read_exact(self.socket, 4), "big")
        event = DECODER.decode(read_exact(self.socket, size))
        fds = [receive_fd(self.socket) for _ in event.artifacts] if isinstance(
            event, SessionResult) else []
        return event, fds

    def invoke(self, seq, output, samples=SAMPLES, seed=19, iterations=2):
        invocation = Invoke("execution-" + str(seq), self.generation.package,
                            self.generation.identity, self.generation.application, "classify",
                            {"samples": samples, "seed": seed, "iterations": iterations},
                            str(output))
        command = SessionInvoke(self.id, seq, invocation)
        self.send(command)
        return command

    def terminal(self):
        while True:
            event, fds = self.receive()
            if not isinstance(event, SessionProgress):
                return event, fds
            assert event.session_id == self.id

    def custody(self, event, fds, directory):
        assert isinstance(event, SessionResult)
        directory.mkdir()
        result = []
        for artifact, fd in zip(event.artifacts, fds, strict=True):
            assert fcntl.fcntl(fd, GET_SEALS) & FULL_SEALS == FULL_SEALS
            assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
            assert os.fstat(fd).st_size == artifact.object.length
            data = os.pread(fd, artifact.object.length, 0)
            assert hashlib.sha256(data).hexdigest() == artifact.object.sha256
            with (directory / artifact.relative_path).open("wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            result.append(json.loads(data))
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return result

    def acknowledge(self, event):
        self.send(SessionAck(self.id, event.seq, event.execution_id))
        ready, fds = self.receive()
        assert isinstance(ready, ReadyNext) and not fds
        assert ready.sdk_imports == ready.package_imports == 1
        return ready

    def close(self):
        self.send(SessionShutdown(self.id))
        self.socket.close()
        stdout, stderr = self.process.communicate(timeout=30)
        assert self.process.returncode == 0, (stdout, stderr)


def test_a_b_a_real_calls_reuse_interpreter_sdk_and_preserve_output(generation, tmp_path):
    session = Session(generation)
    results, facts, held = [], [], []
    for seq, samples, seed in [(1, SAMPLES, 19), (2, list(reversed(SAMPLES)), 43),
                               (3, SAMPLES, 19)]:
        command = session.invoke(seq, tmp_path / f"spool-{seq}", samples, seed)
        event, fds = session.terminal()
        assert isinstance(event, SessionResult) and event.seq == seq
        assert event.execution_id == command.invocation.execution_id
        saved = session.custody(event, fds, tmp_path / f"durable-{seq}")
        assert saved[0]["seed"] == seed and saved[0]["call_sequence"] == seq
        assert saved[0]["probabilities"] == event.value["probabilities"]
        facts.append(session.acknowledge(event))
        results.append(event)
        held.extend(fds)
    assert results[0].value["predictions"] == results[2].value["predictions"] == [0, 2, 1]
    assert results[1].value["predictions"] == [1, 2, 0]
    assert results[0].value["probabilities"] == results[2].value["probabilities"]
    assert len({fact.sdk_module_identity for fact in facts}) == 1
    assert len({fact.package_module_identity for fact in facts}) == 1
    assert session.process.poll() is None  # outputs already have custody while process lives
    original = os.pread(held[0], results[0].artifacts[0].object.length, 0)
    (tmp_path / "spool-1" / results[0].artifacts[0].relative_path).write_bytes(b"changed later")
    assert os.pread(held[0], len(original), 0) == original
    with pytest.raises(OSError):
        os.pwrite(held[0], b"change", 0)
    session.close()
    assert os.pread(held[0], len(original), 0) == original
    for fd in held:
        os.close(fd)


def test_outbox_replay_and_acknowledged_sequence_never_execute_again(generation, tmp_path):
    session = Session(generation)
    first = session.invoke(1, tmp_path / "first")
    event, first_fds = session.terminal()
    session.send(first)
    replay, replay_fds = session.terminal()
    assert replay == event and replay.value["call_sequence"] == 1
    session.custody(replay, replay_fds, tmp_path / "durable")
    session.acknowledge(replay)
    session.send(first)
    error, _ = session.receive()
    assert isinstance(error, SessionError) and error.code == "attempt_already_settled"
    session.invoke(2, tmp_path / "second")
    second, second_fds = session.terminal()
    assert second.value["call_sequence"] == 2
    session.custody(second, second_fds, tmp_path / "durable-second")
    session.acknowledge(second)
    session.close()
    for fd in first_fds + replay_fds + second_fds:
        os.close(fd)


def test_cancel_provenance_cannot_cancel_a_later_call(generation, tmp_path):
    session = Session(generation)
    session.invoke(1, tmp_path / "first")
    first, first_fds = session.terminal()
    session.custody(first, first_fds, tmp_path / "durable-first")
    session.acknowledge(first)
    session.send(SessionCancel(session.id, 1, "execution-1"))
    session.invoke(2, tmp_path / "canceled", iterations=100000)
    while not isinstance(session.receive()[0], SessionProgress):
        pass
    session.send(SessionCancel("another-session", 2, "execution-2"))
    session.send(SessionCancel(session.id, 2, "another-attempt"))
    session.send(SessionCancel(session.id, 2, "execution-2"))
    canceled, fds = session.terminal()
    assert isinstance(canceled, SessionCanceled) and not fds
    session.acknowledge(canceled)
    session.invoke(3, tmp_path / "third")
    third, third_fds = session.terminal()
    assert third.value["call_sequence"] == 3
    session.custody(third, third_fds, tmp_path / "durable-third")
    session.acknowledge(third)
    session.close()
    for fd in first_fds + third_fds:
        os.close(fd)


def test_idle_owner_eof_releases_child_without_canceling_an_attempt(generation):
    session = Session(generation)
    session.socket.close()
    stdout, stderr = session.process.communicate(timeout=30)
    assert session.process.returncode == 0, (stdout, stderr)


def test_owner_eof_during_inference_does_not_start_queued_work(generation, tmp_path):
    session = Session(generation)
    first = session.invoke(1, tmp_path / "first", iterations=100)
    while not isinstance(session.receive()[0], SessionProgress):
        pass
    session.invoke(2, tmp_path / "never-entered")
    session.socket.close()
    stdout, stderr = session.process.communicate(timeout=30)
    assert session.process.returncode == 0, (stdout, stderr)
    reports = list(Path(first.invocation.output_root).iterdir())
    assert len(reports) == 1 and json.loads(reports[0].read_bytes())["iterations"] == 100
    assert not (tmp_path / "never-entered").exists()


def test_next_invocation_requires_matching_custody_ack(generation, tmp_path):
    session = Session(generation)
    session.invoke(1, tmp_path / "first")
    first, fds = session.terminal()
    second = session.invoke(2, tmp_path / "second")
    refusal, _ = session.receive()
    assert isinstance(refusal, SessionError) and refusal.code == "custody_ack_required"
    assert not (tmp_path / "second").exists()
    session.send(SessionAck(session.id, 1, "unrelated-execution"))
    mismatch, _ = session.receive()
    assert isinstance(mismatch, SessionError) and mismatch.code == "ack_mismatch"
    session.custody(first, fds, tmp_path / "durable-first")
    session.acknowledge(first)
    session.send(second)
    result, second_fds = session.terminal()
    assert result.value["call_sequence"] == 2
    session.custody(result, second_fds, tmp_path / "durable-second")
    session.acknowledge(result)
    session.close()
    for fd in fds + second_fds:
        os.close(fd)


def test_output_snapshot_checks_sdk_identity_and_does_not_leak_on_corruption(tmp_path):
    source = tmp_path / "report"
    source.write_bytes(b"declared bytes")
    facts = OutputFacts("report", OutputChecksum(
        "sha256", hashlib.sha256(b"declared bytes").hexdigest()), 14)
    before = len(list(Path("/proc/self/fd").iterdir()))
    source.write_bytes(b"different body")
    with pytest.raises(ValueError, match="SDK output identity"):
        snapshot(tmp_path, facts)
    assert len(list(Path("/proc/self/fd").iterdir())) == before
