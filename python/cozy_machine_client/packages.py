"""Source-only description and held immutable uv environment generations.

This delegates source interpretation to the existing Runtime static reader. Building a
wheel is an authorized packaging operation and may execute a build backend; describing
it never imports or runs authored modules. Generations are never updated in place.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from pathlib import Path

import msgspec

from .package_records import (
    DESCRIPTION_DECODER, GENERATION_DECODER, Dependency, Describe, DescribeFailed, Generation,
    InstallFailed, PackageMetadata, Pyproject,
)


class PackageError(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def describe(project: Path, environment_python: Path | None = None) -> msgspec.Raw:
    request = Describe(str(project), str(environment_python) if environment_python else None)
    result = subprocess.check_output([sys.executable, "-m", "cozy_machine_client.runtime_describe"],
                                     input=msgspec.json.encode(request))
    reply = DESCRIPTION_DECODER.decode(result)
    if isinstance(reply, DescribeFailed):
        raise PackageError(reply.code, reply.detail)
    return reply.interface


def read_metadata(project: Path) -> PackageMetadata:
    parsed = msgspec.convert(tomllib.loads((project / "pyproject.toml").read_text()),
                             type=Pyproject)
    metadata = parsed.project
    return PackageMetadata(metadata.name, metadata.version,
                           metadata.entry_points.application.default)


def probe_cpu_bridge(interpreter: Path) -> str:
    """Import the CPU runner's SDK adapter in the new environment, isolated from this one.
    Empty means it imports; otherwise the reason CPU runs of this generation cannot start.
    GPU executors do not use the adapter, so a failure never refuses the installation."""
    probe = subprocess.run([str(interpreter), "-I", "-c", "import cozy_machine_client.runtime_bridge"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if probe.returncode == 0:
        return ""
    lines = [line for line in probe.stderr.strip().splitlines() if line.strip()]
    return (lines[-1] if lines else f"bridge import exited {probe.returncode}")[:1024]


def publish_generation(root: Path, metadata: PackageMetadata, interface: msgspec.Raw) -> Generation:
    interpreter = root / "env" / "bin" / "python"
    inventory = subprocess.check_output(["uv", "pip", "list", "--python", str(interpreter), "--format", "json"])
    dependencies = msgspec.json.decode(inventory, type=list[Dependency])
    generation = Generation(root.name, metadata.name, metadata.version, metadata.application,
                            str(interpreter), dependencies, interface, probe_cpu_bridge(interpreter))
    with (root / ".generation.json.new").open("wb") as output:
        output.write(msgspec.json.encode(generation))
        output.flush()
        os.fsync(output.fileno())
    (root / ".generation.json.new").replace(root / "generation.json")
    for directory in [root, root.parent]:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return generation


class GenerationHold:
    """One shared generation lease; collectors may only take exclusive ownership."""

    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = (self.path / ".hold").open("rb")
        try:
            fcntl.flock(self.file, fcntl.LOCK_SH)
            return GENERATION_DECODER.decode((self.path / "generation.json").read_bytes())
        except BaseException:
            self.file.close()
            raise

    def __exit__(self, *exc):
        self.file.close()


def collect(path: Path) -> bool:
    """Delete only a completed unheld generation explicitly selected by the owner."""
    with (path / ".hold").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        generation = GENERATION_DECODER.decode((path / "generation.json").read_bytes())
        if generation.identity != path.name:
            raise ValueError("collector may only reclaim a published generation identity")
        # A rename removes the public generation name before reclamation begins;
        # a client must acquire its hold before resolving its immutable Python path.
        retired = path.with_name(".retired-" + uuid.uuid4().hex)
        path.rename(retired)
        shutil.rmtree(retired)
    return True


def install(project: Path, generations: Path, client_wheel: Path, *,
            python: str = sys.executable) -> Generation:
    project = project.resolve()
    client_wheel = client_wheel.resolve(strict=True)
    interface = describe(project)
    metadata = read_metadata(project)
    generations.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    # The interpreter embeds its absolute venv path. Create it at its final name;
    # only atomic generation.json publication makes it available for dispatch.
    root = generations.resolve() / identity
    root.mkdir()
    try:
        (root / ".hold").touch()
        with tempfile.TemporaryDirectory(prefix=".build-", dir=generations) as build:
            subprocess.run(["uv", "build", "--wheel", "--out-dir", build, str(project)],
                           check=True)
            wheels = list(Path(build).glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError("one source project must produce exactly one wheel")
            subprocess.run(["uv", "venv", "--python", python, str(root / "env")], check=True)
            interpreter = root / "env" / "bin" / "python"
            # uv resolves the package's declared lower AND upper dependency bounds;
            # no service-version floor or SDK upgrade is inserted into this solve.
            subprocess.run(["uv", "pip", "install", "--python", str(interpreter),
                            str(wheels[0]), str(client_wheel)], check=True)
            interface = describe(project, interpreter)
            return publish_generation(root, metadata, interface)
    except BaseException:
        # This uniquely owned unpublished generation has never been dispatched.
        shutil.rmtree(root)
        raise


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    static = commands.add_parser("describe")
    static.add_argument("project", type=Path)
    setup = commands.add_parser("install")
    setup.add_argument("project", type=Path)
    setup.add_argument("--generations", type=Path, required=True)
    setup.add_argument("--client-wheel", type=Path, required=True)
    setup.add_argument("--python", default=sys.executable)
    captured = commands.add_parser("install-captured")
    captured.add_argument("--project", type=Path)
    captured.add_argument("--wheel", action="append", type=Path, default=[])
    captured.add_argument("--requirements", type=Path)
    captured.add_argument("--distribution", required=True)
    captured.add_argument("--release", required=True)
    captured.add_argument("--python-requires", default="")
    captured.add_argument("--python-version", default="")
    captured.add_argument("--generations", type=Path, required=True)
    captured.add_argument("--client-wheel", type=Path, required=True)
    captured.add_argument("--sdk-wheel", action="append", type=Path, default=[])
    captured.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.command == "install-captured":
        from .captured_packages import install_captured
        try:
            result = install_captured(project=args.project, wheels=args.wheel, requirements=args.requirements,
                distribution=args.distribution, release=args.release, python_requires=args.python_requires,
                python_version=args.python_version, generations=args.generations, client_wheel=args.client_wheel, python=args.python,
                sdk=args.sdk_wheel)
        except PackageError as exc:
            print(msgspec.json.encode(InstallFailed(exc.code, str(exc))).decode())
            raise SystemExit(1)
        except subprocess.CalledProcessError:
            print(msgspec.json.encode(InstallFailed("package_dependency_operation_failed", "standard uv did not complete the captured dependency operation")).decode())
            raise SystemExit(1)
        print(msgspec.json.encode(result).decode())
        return
    result = (describe(args.project) if args.command == "describe" else
              install(args.project, args.generations, args.client_wheel, python=args.python))
    print(msgspec.json.encode(result).decode())


if __name__ == "__main__":
    main()
