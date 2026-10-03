"""Install captured carriers with standard uv while preserving their lock/constraints."""
from __future__ import annotations

import configparser
import email.parser
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import msgspec
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from .package_records import Dependency, DescribeFailed, DescribeInstalled, DESCRIPTION_DECODER, PackageMetadata
from .packages import PackageError, publish_generation, read_metadata


def wheel_metadata(path: Path) -> PackageMetadata:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 1 << 20:
            raise PackageError("package_wheel_metadata_invalid", "one bounded wheel metadata record is required")
        headers = email.parser.Parser().parsestr(archive.read(names[0]).decode(), headersonly=True)
        entry_name = names[0].removesuffix("METADATA") + "entry_points.txt"
        if entry_name not in archive.namelist() or archive.getinfo(entry_name).file_size > 1 << 20:
            raise PackageError("package_application_absent", "the selected wheel has no bounded App declaration")
        entries = configparser.ConfigParser(interpolation=None)
        entries.read_string(archive.read(entry_name).decode())
        applications = list(entries.items("cozy.application"))
        if len(applications) != 1:
            raise PackageError("package_application_invalid", "one App declaration is required")
        return PackageMetadata(headers["Name"], headers["Version"], applications[0][1])


def installed_description(distribution: str, interpreter: Path) -> msgspec.Raw:
    request = DescribeInstalled(distribution, str(interpreter))
    # -S prevents authored .pth startup code; only the owned venv's regular
    # site-packages paths are inserted before loading the installed SDK reader.
    bootstrap = ("import pathlib,runpy,sys;"
                 "root=pathlib.Path(sys.executable).absolute().parent.parent;"
                 "sys.path[:0]=[str(p) for p in (root/'lib').glob('python*/site-packages')];"
                 "runpy.run_module('cozy_machine_client.runtime_describe',run_name='__main__')")
    raw = subprocess.check_output([str(interpreter), "-I", "-S", "-c", bootstrap], input=msgspec.json.encode(request))
    reply = DESCRIPTION_DECODER.decode(raw)
    if isinstance(reply, DescribeFailed):
        raise PackageError(reply.code, reply.detail)
    return reply.interface


def inventory(interpreter: Path) -> list[Dependency]:
    return msgspec.json.decode(subprocess.check_output(["uv", "pip", "list", "--python", str(interpreter), "--format", "json"]), type=list[Dependency])


def install_captured(*, project: Path | None, wheels: list[Path], requirements: Path | None,
                     distribution: str, release: str, python_requires: str, python_version: str,
                     generations: Path, client_wheel: Path, python: str, sdk: list[Path] = ()):
    python = subprocess.check_output(["uv", "python", "find", "--no-project", "--no-python-downloads", python]).decode().strip()
    version = Version(subprocess.check_output([python, "-I", "-S", "-c", "import platform;print(platform.python_version())"]).decode().strip())
    if python_requires and version not in SpecifierSet(python_requires):
        raise PackageError("package_python_unavailable", "configured interpreter does not satisfy captured Python requirements")
    if python_version and str(version) != python_version and ".".join(str(version).split(".")[:2]) != python_version:
        raise PackageError("package_python_unavailable", "configured interpreter does not match captured Python selection")
    project = project.resolve() if project else None
    wheels = [wheel.resolve(strict=True) for wheel in wheels]
    client_wheel = client_wheel.resolve(strict=True)
    if project and (project / "uv.lock").is_file() and (wheels or requirements):
        raise PackageError("package_capture_mixed_lock_unsupported", "a frozen source capture cannot overlay an independent wheel/requirements closure")
    if project:
        metadata = read_metadata(project)
    else:
        matches = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                if len(names) != 1 or archive.getinfo(names[0]).file_size > 1 << 20:
                    raise PackageError("package_wheel_metadata_invalid", "invalid bounded wheel metadata")
                name = email.parser.Parser().parsestr(archive.read(names[0]).decode(), headersonly=True)["Name"]
                if canonicalize_name(name) == canonicalize_name(distribution):
                    matches.append(wheel_metadata(wheel))
        if len(matches) != 1:
            raise PackageError("package_root_wheel_absent", "one captured wheel must name the requested distribution")
        metadata = matches[0]
    if canonicalize_name(metadata.name) != canonicalize_name(distribution) or Version(metadata.version) != Version(release):
        raise PackageError("package_metadata_changed", "uploaded source/wheel declares another package release")
    generations.mkdir(parents=True, exist_ok=True)
    root = generations.resolve() / uuid.uuid4().hex
    root.mkdir(); (root / ".hold").touch()
    try:
        interpreter = root / "env" / "bin" / "python"
        if project and (project / "uv.lock").is_file():
            source = root / "source"
            shutil.copytree(project, source)
            # UV_PROJECT_ENVIRONMENT is a standard destination configuration value.
            environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(root / "env")}
            subprocess.run(["uv", "sync", "--frozen", "--no-dev", "--no-editable", "--no-python-downloads",
                            "--project", str(source), "--python", python], env=environment, check=True)
        else:
            subprocess.run(["uv", "venv", "--python", python, str(root / "env")], check=True)
            if requirements and requirements.stat().st_size:
                # uv owns hashed requirements parsing and standard download integrity.
                command = ["uv", "pip", "install", "--python", str(interpreter), "--require-hashes", "-r", str(requirements)]
                if wheels:
                    command += ["--find-links", str(wheels[0].parent)]
                subprocess.run(command, check=True)
                if wheels:
                    subprocess.run(["uv", "pip", "install", "--python", str(interpreter), "--no-deps", *map(str,wheels)], check=True)
            elif project:
                subprocess.run(["uv", "pip", "install", "--python", str(interpreter), str(project), *map(str,wheels)], check=True)
            else:
                subprocess.run(["uv", "pip", "install", "--python", str(interpreter), *map(str,wheels)], check=True)
        if sdk:
            # The machine's own Runtime/TensorFS pair replaces what the capture resolved; every
            # other locked version stays, and uv reports a conflict with declared bounds.
            pair = {canonicalize_name(wheel.name.split("-")[0]) for wheel in sdk}
            kept = root / "sdk-constraints.txt"
            kept.write_text("".join(f"{row.name}=={row.version}\n" for row in inventory(interpreter)
                                    if canonicalize_name(row.name) not in pair))
            subprocess.run(["uv", "pip", "install", "--python", str(interpreter), "-c", str(kept),
                            *map(str, sdk)], check=True)
        locked = inventory(interpreter)
        constraints = root / "captured-constraints.txt"
        constraints.write_text("".join(f"{row.name}=={row.version}\n" for row in locked))
        # Runner dependencies may be added only without changing a single already
        # selected package/SDK version. uv reports any conflict with declared bounds.
        subprocess.run(["uv", "pip", "install", "--python", str(interpreter), "-c", str(constraints), str(client_wheel)], check=True)
        after = {canonicalize_name(row.name): row.version for row in inventory(interpreter)}
        if any(after.get(canonicalize_name(row.name)) != row.version for row in locked):
            raise PackageError("package_lock_changed", "runner installation changed the captured dependency closure")
        subprocess.run(["uv", "pip", "check", "--python", str(interpreter)], check=True)
        interface = installed_description(metadata.name, interpreter)
        return publish_generation(root, metadata, interface)
    except BaseException:
        shutil.rmtree(root)  # uniquely owned unpublished generation, never dispatched
        raise
