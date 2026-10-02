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
from typing import Any

import msgspec


class Dependency(msgspec.Struct, frozen=True):
    name: str
    version: str


class Generation(msgspec.Struct, frozen=True):
    identity: str
    package: str
    version: str
    application: str
    python: str
    dependencies: list[Dependency]
    interface: dict[str, Any]


GENERATION_DECODER = msgspec.json.Decoder(Generation)


def describe(project: Path, environment_python: Path | None = None) -> dict[str, Any]:
    from cozy_runtime.internal.static_interface import build

    return build(project, environment_python=environment_python)


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
    metadata = tomllib.loads((project / "pyproject.toml").read_text())["project"]
    package, version = metadata["name"], metadata["version"]
    application = metadata["entry-points"]["cozy.application"]["default"]
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
            inventory = subprocess.check_output(
                ["uv", "pip", "list", "--python", str(interpreter), "--format", "json"])
            dependencies = msgspec.json.decode(inventory, type=list[Dependency])
            interface = describe(project, interpreter)
            generation = Generation(identity, package, version, application,
                                    str(interpreter), dependencies, interface)
            manifest = root / ".generation.json.new"
            with manifest.open("wb") as output:
                output.write(msgspec.json.encode(generation))
                output.flush()
                os.fsync(output.fileno())
            manifest.replace(root / "generation.json")
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return generation
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
    args = parser.parse_args()
    result = (describe(args.project) if args.command == "describe" else
              install(args.project, args.generations, args.client_wheel, python=args.python))
    print(msgspec.json.encode(result).decode())


if __name__ == "__main__":
    main()
