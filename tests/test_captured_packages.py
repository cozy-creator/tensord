"""Real captured uv installations and classifier inference; no implementation doubles."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import tomllib

from cozy_machine_client.captured_packages import install_captured
from cozy_machine_client.packages import GenerationHold, PackageError
from cozy_machine_client.execution_protocol import Result
from test_package_bridge import FIXTURE, ROOT, spawn_runner, send_command, invoke, terminal, finish


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    pytest.importorskip("cozy_runtime")
    root = tmp_path_factory.mktemp("real-captured-package")
    source = root / "source"
    shutil.copytree(FIXTURE, source)
    project = source / "pyproject.toml"
    project.write_text(project.read_text().replace("cozy-runtime>=0.18.89,<0.18.101", "cozy-runtime==0.18.89"))
    subprocess.run(["uv", "lock", "--project", str(source), "--python", "3.12"], check=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(root / "client"), str(ROOT)], check=True)
    client = next((root / "client").glob("*.whl"))
    frozen = install_captured(project=source, wheels=[], requirements=None,
        distribution="cozy-machine-cpu-classifier", release="0.1.0", python_requires=">=3.12", python_version="3.12",
        generations=root / "frozen", client_wheel=client, python="3.12")
    yield root, source, client, frozen


def assert_inference(generation, output):
    with GenerationHold(Path(generation.python).parents[2]):
        process, socket = spawn_runner(generation)
        send_command(socket, invoke(generation, output))
        result = terminal(socket)
        assert isinstance(result, Result), result
        assert result.value["predictions"] == [0, 2, 1]
        artifact = json.loads((output / result.artifacts[0]).read_bytes())
        assert artifact["probabilities"] == result.value["probabilities"]
        finish(process, socket)


def test_frozen_source_keeps_locked_sdk_and_runs_real_classifier(capture):
    root, source, client, generation = capture
    versions = {dependency.name: dependency.version for dependency in generation.dependencies}
    lock = tomllib.loads((source / "uv.lock").read_text())
    sdk = next(row["version"] for row in lock["package"] if row["name"] == "cozy-runtime")
    assert versions["cozy-runtime"] == sdk == "0.18.89"
    assert_inference(generation, root / "frozen-result")


def test_real_wheel_roster_and_hashed_requirements_preserve_dependency_versions(capture):
    root, source, client, frozen = capture
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(root / "wheels"), str(source)], check=True)
    wheel = next((root / "wheels").glob("*.whl"))
    requirements = root / "requirements.txt"
    subprocess.run(["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--project", str(source), "--output-file", str(requirements)], check=True)
    generation = install_captured(project=None, wheels=[wheel], requirements=requirements,
        distribution="cozy-machine-cpu-classifier", release="0.1.0", python_requires=">=3.12", python_version="3.12",
        generations=root / "wheel-generations", client_wheel=client, python="3.12")
    before = {dependency.name: dependency.version for dependency in frozen.dependencies}
    after = {dependency.name: dependency.version for dependency in generation.dependencies}
    assert after == before
    assert_inference(generation, root / "wheel-result")


def test_incoherent_source_lock_overlay_and_python_mismatch_refuse_only_installation(capture):
    root, source, client, generation = capture
    with pytest.raises(PackageError, match="cannot overlay"):
        install_captured(project=source, wheels=[client], requirements=None,
            distribution="cozy-machine-cpu-classifier", release="0.1.0", python_requires="", python_version="",
            generations=root / "unsupported", client_wheel=client, python="3.12")
    with pytest.raises(PackageError, match="does not match"):
        install_captured(project=source, wheels=[], requirements=None,
            distribution="cozy-machine-cpu-classifier", release="0.1.0", python_requires="", python_version="3.11",
            generations=root / "unsupported", client_wheel=client, python="3.12")
    assert_inference(generation, root / "surviving-result")
