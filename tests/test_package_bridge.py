"""Real installed-package/process/socket gates; no function doubles or CUDA calls."""
from __future__ import annotations

import json
import shutil
import socket
import struct
import subprocess
from pathlib import Path

import msgspec
import pytest
from packaging.version import Version

from cozy_machine_client.execution_protocol import (
    COMMAND_DECODER, EVENT_DECODER, Cancel, Canceled, Failed, Invoke, Progress, Ready, Result,
)
from cozy_machine_client.packages import GenerationHold, collect, describe, install, read_metadata
from cozy_machine_client.package_records import GENERATION_DECODER, Generation
from cozy_machine_client.progress import CompletedWork

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cpu_classifier"
SAMPLES = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 5.1, 1.6], [6.7, 3.1, 4.7, 1.5]]


def read_event(sock):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            part = sock.recv(size - len(data))
            if not part:
                raise EOFError
            data.extend(part)
        return bytes(data)

    return EVENT_DECODER.decode(exact(struct.unpack("!I", exact(4))[0]))


def send_command(sock, command):
    data = msgspec.json.encode(command)
    sock.sendall(struct.pack("!I", len(data)) + data)


@pytest.fixture(scope="module")
def generation(tmp_path_factory):
    pytest.importorskip("cozy_runtime")
    output = tmp_path_factory.mktemp("immutable-package")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output / "client"), str(ROOT)],
                   check=True)
    client = next((output / "client").glob("*.whl"))
    generation = install(FIXTURE, output / "generations", client)
    with GenerationHold(Path(generation.python).parents[2]) as held:
        yield held


def spawn_runner(generation):
    owner, child = socket.socketpair()
    # Timeout only bounds a broken test's wait; it never kills a production process.
    owner.settimeout(30)
    process = subprocess.Popen([generation.python, "-m", "cozy_machine_client.runner",
                                "--execution-fd", str(child.fileno())],
                               pass_fds=[child.fileno()], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    child.close()
    ready = read_event(owner)
    assert isinstance(ready, Ready) and ready.pid == process.pid
    return process, owner


def invoke(generation, output, *, execution_id="one", iterations=2):
    return Invoke(execution_id, generation.package, generation.identity,
                  generation.application, "classify",
                  {"samples": SAMPLES, "iterations": iterations}, str(output))


def finish(process, sock):
    sock.close()
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, (stdout, stderr)


def terminal(sock, completed=None):
    while isinstance(event := read_event(sock), Progress):
        assert event.completed_units >= 0
        if completed is not None:
            completed.append(event.completed_units)
    return event


def test_static_description_does_not_execute_package_top_level(tmp_path):
    pytest.importorskip("cozy_runtime")
    source = tmp_path / "package"
    shutil.copytree(FIXTURE, source)
    module = source / "cpu_classifier" / "__init__.py"
    module.write_text(module.read_text() + '\nraise RuntimeError("import must not occur")\n')
    interface = describe(source)
    assert isinstance(interface, msgspec.Raw)
    assert b"classify" in bytes(interface)
    assert b"cpu_classifier:app" in bytes(interface)


def test_additive_peer_fields_preserve_baseline_invocation():
    event = {"kind": "invoke", "execution_id": "one", "package": "p",
             "generation": "g", "module": "p:app", "entrypoint": "classify",
             "input": {"application_data": {"future": 2}}, "output_root": "/owned/spool",
             "future_advisory": {"version": "2000.0", "notes": True}}
    decoded = COMMAND_DECODER.decode(msgspec.json.encode(event))
    assert isinstance(decoded, Invoke)
    assert decoded.input == event["input"]


def test_only_new_completed_positions_count_as_productive_progress():
    ProgressFrame = pytest.importorskip("cozy_runtime.author._services").ProgressFrame

    work = CompletedWork()
    assert work.observe(ProgressFrame("stage", None, 1000)) is None
    assert work.observe(ProgressFrame("stage", .5, 1001, position=1, total=2)) == 1
    assert work.observe(ProgressFrame("stage", .5, 2000, position=1, total=2)) is None
    assert work.observe(ProgressFrame("stage", .5, 3000, position=0, total=2)) is None
    assert work.observe(ProgressFrame("stage", 1., 3001, position=2, total=2)) == 2
    assert work.observe(ProgressFrame("another", 1., 3002, position=3, total=3)) == 5


def test_consumed_metadata_is_typed_and_unknown_extras_are_harmless(tmp_path):
    source = tmp_path / "metadata"
    shutil.copytree(FIXTURE, source)
    project = source / "pyproject.toml"
    project.write_text(project.read_text() + '\n[future.optional]\nnested = {x = [1, 2]}\n')
    metadata = read_metadata(source)
    assert metadata.name == "cozy-machine-cpu-classifier"
    assert metadata.application == "cpu_classifier:app"
    project.write_text(project.read_text().replace('version = "0.1.0"', "version = 42"))
    with pytest.raises(msgspec.ValidationError):
        read_metadata(source)


def test_sdk_interface_remains_opaque_through_generation_boundary():
    document = msgspec.Raw(b'{"future":{"unknown":[1,true]},"entrypoints":[]}')
    original = Generation("g", "p", "1", "p:app", "/g/env/bin/python", [], document)
    replay = GENERATION_DECODER.decode(msgspec.json.encode(original))
    assert isinstance(replay.interface, msgspec.Raw)
    assert bytes(replay.interface) == bytes(document)


def test_ready_precedes_authored_import_and_sdk_import(generation):
    process, sock = spawn_runner(generation)
    maps = Path(f"/proc/{process.pid}/maps").read_text()
    assert "libcuda" not in maps and "libcudart" not in maps
    # NumPy is imported by both the SDK and this authored package. Its binary
    # extensions being absent proves Ready did not import either module graph.
    assert "numpy" not in maps and "tensorfs" not in maps
    finish(process, sock)  # EOF before authorization exits without entering code.


def test_real_sdk_inference_saved_output_and_repeated_executors(generation, tmp_path):
    reference = None
    for attempt in range(3):
        process, sock = spawn_runner(generation)
        output = tmp_path / str(attempt)
        send_command(sock, invoke(generation, output, execution_id=str(attempt)))
        completed = []
        event = terminal(sock, completed)
        assert completed == [1, 2]  # stage-open/fraction/event counters are not work
        assert isinstance(event, Result), event
        assert event.execution_id == str(attempt)
        assert len(event.artifacts) == 1
        saved = json.loads((output / event.artifacts[0]).read_bytes())
        assert saved["predictions"] == [0, 2, 1]
        assert saved["probabilities"] == event.value["probabilities"]
        if reference is None:
            reference = saved
        assert saved == reference
        finish(process, sock)


def test_cooperative_explicit_cancel_is_attempt_scoped(generation, tmp_path):
    process, sock = spawn_runner(generation)
    send_command(sock, invoke(generation, tmp_path / "canceled", iterations=100000))
    while not isinstance(read_event(sock), Progress):
        pass
    maps = Path(f"/proc/{process.pid}/maps").read_text()
    assert "libcuda" not in maps and "libcudart" not in maps and "libnvidia" not in maps
    held_files = [link.resolve() for link in Path(f"/proc/{process.pid}/fd").iterdir()]
    assert Path(generation.python).parents[2] / ".hold" in held_files
    send_command(sock, Cancel("different-attempt"))
    send_command(sock, Cancel("one"))
    assert isinstance(terminal(sock), Canceled)
    finish(process, sock)
    assert not list((tmp_path / "canceled").iterdir())


def test_owner_eof_does_not_author_user_cancellation(generation, tmp_path):
    process, sock = spawn_runner(generation)
    output = tmp_path / "detached"
    send_command(sock, invoke(generation, output, iterations=100))
    while not isinstance(read_event(sock), Progress):
        pass
    finish(process, sock)
    files = list(output.iterdir())
    assert len(files) == 1
    assert json.loads(files[0].read_bytes())["iterations"] == 100


def test_wrong_generation_fails_before_authored_import(generation, tmp_path):
    process, sock = spawn_runner(generation)
    command = invoke(generation, tmp_path / "invalid-generation")
    command.generation = "unrelated-env"
    send_command(sock, command)
    assert isinstance(terminal(sock), Failed)
    finish(process, sock)
    assert not (tmp_path / "invalid-generation").exists()


def test_invalid_application_fails_without_import(generation, tmp_path):
    process, sock = spawn_runner(generation)
    command = invoke(generation, tmp_path / "invalid")
    command.module = "os"
    send_command(sock, command)
    assert isinstance(terminal(sock), Failed)
    finish(process, sock)


def test_bound_is_preserved_and_live_generation_cannot_be_collected(generation):
    versions = {entry.name: entry.version for entry in generation.dependencies}
    assert Version("0.18.89") <= Version(versions["cozy-runtime"]) < Version("0.18.101")
    assert not collect(Path(generation.python).parents[2])


def test_real_older_sdk_package_runs_without_peer_version_floor(generation, tmp_path):
    source = tmp_path / "older-package"
    shutil.copytree(FIXTURE, source)
    project = source / "pyproject.toml"
    project.write_text(project.read_text().replace(
        "cozy-runtime>=0.18.89,<0.18.101", "cozy-runtime==0.18.89"))
    client_wheel = next((Path(generation.python).parents[4] / "client").glob("*.whl"))
    older = install(source, tmp_path / "older-generations", client_wheel)
    versions = {dependency.name: dependency.version for dependency in older.dependencies}
    assert versions["cozy-runtime"] == "0.18.89"
    with GenerationHold(Path(older.python).parents[2]):
        process, sock = spawn_runner(older)
        output = tmp_path / "older-result"
        send_command(sock, invoke(older, output))
        event = terminal(sock)
        assert isinstance(event, Result), event
        assert event.value["predictions"] == [0, 2, 1]
        saved = json.loads((output / event.artifacts[0]).read_bytes())
        assert saved["probabilities"] == event.value["probabilities"]
        finish(process, sock)


def test_generation_collected_only_after_last_holder(tmp_path):
    root = tmp_path / "generation"
    root.mkdir()
    (root / ".hold").touch()
    (root / "generation.json").write_bytes(msgspec.json.encode({
        "identity": "generation", "package": "p", "version": "1", "application": "p:app",
        "python": str(root / "env/bin/python"), "dependencies": [], "interface": {},
    }))
    with GenerationHold(root):
        assert not collect(root)
    assert collect(root)
    assert not root.exists()
