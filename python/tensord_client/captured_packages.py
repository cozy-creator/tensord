"""Install captured carriers with standard uv while preserving their lock/constraints."""
from __future__ import annotations

import configparser
import email.parser
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
import zipfile
from pathlib import Path

import msgspec
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from .package_records import Dependency, Describe, DescribeInstalled, DESCRIPTION_DECODER, PackageMetadata
from .packages import PackageError, in_environment, publish_generation, read_metadata


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
    return in_environment(interpreter, DescribeInstalled(distribution, str(interpreter)), DESCRIPTION_DECODER).interface


#: Under symlinks (an image's seeded cache) the code the Runtime describes (an App's wheel, the
#: SDK, the machine client) is copied into the environment: its static reader refuses metadata
#: and sources linked from elsewhere. Only the locked dependencies are linked.
COPY = ("--link-mode", "copy") if os.environ.get("UV_LINK_MODE") == "symlink" else ()


def local_builds(project: Path) -> list[str]:
    """The lock's packages built from a directory on this machine. uv keys a built directory by
    its metadata alone, so an environment built because their code changed builds them again."""
    lock = tomllib.loads((project / "uv.lock").read_text())
    return [row["name"] for row in lock.get("package", ())
            if {"directory", "editable", "path"} & set(row.get("source", {})) and row["source"].get("editable") != "."]


def linked_applications(interpreter: Path) -> list[str]:
    """Installed distributions that declare an App and were linked, not copied: the Runtime's
    static reader describes only copied ones (a callee package among the locked dependencies)."""
    names = []
    for site in interpreter.parents[1].glob("lib/python*/site-packages"):
        for metadata in site.glob("*.dist-info"):
            entries = metadata / "entry_points.txt"
            if entries.is_symlink() and "[cozy.application]" in entries.read_text():
                names.append(metadata.name.removesuffix(".dist-info").rpartition("-")[0])
    return names


def inventory(interpreter: Path) -> list[Dependency]:
    return msgspec.json.decode(subprocess.check_output(["uv", "pip", "list", "--python", str(interpreter), "--format", "json"]), type=list[Dependency])


def incompatibilities(interpreter: Path, step: subprocess.CompletedProcess | None = None) -> set[str]:
    """`uv pip check`'s findings, one line each."""
    step = step or subprocess.run(["uv", "pip", "check", "--python", str(interpreter)], capture_output=True, text=True)
    return {line.strip() for line in (step.stdout + step.stderr).splitlines() if line.strip().startswith("The package")}


def machine_sdk(interpreter: Path, sdk: list[Path], capture) -> str:
    """The machine's own Runtime/TensorFS pair over the captured closure, every other selected
    version constrained, where every installed requirement admits it (uv's check). Else the
    capture's own pair (a vendored dev Runtime, say) is restored: a package's bounds are never
    overridden. Returns "" for the machine's pair, else why and what runs instead; never silent."""
    pair = {canonicalize_name(wheel.name.split("-")[0]) for wheel in sdk}
    # What the capture already reports (NVIDIA's aarch64 cusparseLt wheel says sbsa, not aarch64)
    # is not the pair's doing: the pair is refused only for an incompatibility it adds.
    before = incompatibilities(interpreter)
    kept = interpreter.parents[2] / "sdk-constraints.txt"
    kept.write_text("".join(f"{row.name}=={row.version}\n" for row in inventory(interpreter)
                            if canonicalize_name(row.name) not in pair))
    step = subprocess.run(["uv", "pip", "install", *COPY, "--python", str(interpreter), "-c", str(kept), *map(str, sdk)],
                          stderr=subprocess.PIPE, text=True)
    sys.stderr.write(step.stderr)
    if step.returncode == 0:
        step = subprocess.run(["uv", "pip", "check", "--python", str(interpreter)], capture_output=True, text=True)
        if step.returncode == 0 or incompatibilities(interpreter, step) <= before:
            return ""
    reason = [line.strip() for line in ((step.stdout or "") + step.stderr).splitlines()
              if line.strip() and not line.startswith(("Using Python", "Checked "))][-12:]
    capture()
    own = ", ".join(f"{row.name}=={row.version}" for row in inventory(interpreter) if canonicalize_name(row.name) in pair)
    fallback = (f"this machine's own Runtime and TensorFS did not install ({' '.join(map(str, step.args[:3]))}: "
                f"{' | '.join(reason)[:2000]}); it runs the package's own {own}")
    print(fallback, file=sys.stderr)
    return fallback


def interpreter_for(python: str, python_requires: str, python_version: str) -> str:
    """The machine's interpreter for a package, refused when its declared Python excludes it."""
    python = subprocess.check_output(["uv", "python", "find", "--no-project", "--no-python-downloads", python]).decode().strip()
    version = Version(subprocess.check_output([python, "-I", "-S", "-c", "import platform;print(platform.python_version())"]).decode().strip())
    if python_requires and version not in SpecifierSet(python_requires):
        raise PackageError("package_python_unavailable", "configured interpreter does not satisfy captured Python requirements")
    if python_version and str(version) != python_version and ".".join(str(version).split(".")[:2]) != python_version:
        raise PackageError("package_python_unavailable", "configured interpreter does not match captured Python selection")
    return python


def add_client(interpreter: Path, constraints: Path, client_wheel: Path) -> None:
    """The machine client beside a selected closure: it may add its own dependencies only
    without changing one already selected package or SDK version (uv reports any conflict)."""
    locked, before = inventory(interpreter), incompatibilities(interpreter)
    constraints.write_text("".join(f"{row.name}=={row.version}\n" for row in locked))
    subprocess.run(["uv", "pip", "install", *COPY, "--python", str(interpreter), "-c", str(constraints), str(client_wheel)], check=True)
    after = {canonicalize_name(row.name): row.version for row in inventory(interpreter)}
    if any(after.get(canonicalize_name(row.name)) != row.version for row in locked):
        raise PackageError("package_lock_changed", "client installation changed the captured dependency closure")
    # Only what the client adds refuses; the selected closure's own findings stand as captured.
    if added := sorted(incompatibilities(interpreter) - before):
        raise PackageError("package_dependency_conflict", "; ".join(added)[:2000])


def live_paths(environment: Path) -> None:
    """Spell an editable install's source paths as `import` lines. Python adds them at startup
    all the same, and the Runtime's static reader, which never interprets such a line, reads the
    environment's installed packages without following the project out of it: the project is
    described from its own directory."""
    for path in environment.glob("lib/python*/site-packages/*.pth"):
        lines = path.read_text().splitlines()
        live = [line for line in lines if line and not line.startswith(("#", "import ", "import\t"))
                and not (path.parent / line.rstrip()).resolve().is_relative_to(environment.resolve())]
        if live:
            path.write_text("".join(f"import sys; sys.path.append({line.rstrip()!r})\n" if line in live else line + "\n" for line in lines))


def install_editable(*, project: Path, environment: Path, wheels: list[Path], distribution: str, release: str,
                     python_requires: str, python_version: str, generations: Path, client_wheel: Path,
                     python: str, sdk: list[Path] = (), callees: dict[str, str] | None = None,
                     indexes: list[str] = (), source_digest: str = ""):
    """Install editable in the controller's source home and environment. Remote captures keep
    separate source homes/environments; direct local paths reuse their live environment while
    its lock stands. A ready environment is reused, and every call publishes a generation
    describing that source. `wheels` are locked dependencies the caller carries because this machine cannot
    fetch them: each installs from its file, at the version the lock selected. `indexes` (name=url)
    are those the project's sources name without declaring: a lock already pins where each
    dependency comes from, so only an unlocked project resolves against them."""
    python = interpreter_for(python, python_requires, python_version)
    project = project.resolve(strict=True)
    metadata = read_metadata(project)
    if canonicalize_name(metadata.name) != canonicalize_name(distribution) or Version(metadata.version) != Version(release):
        raise PackageError("package_metadata_changed", "the live project declares another package release")
    interpreter, ready = environment / "bin" / "python", environment / ".ready"
    if not ready.is_file():
        shutil.rmtree(environment, ignore_errors=True)  # a build that never finished
        environment.parent.mkdir(parents=True, exist_ok=True)

        def capture(machine: set[str] = frozenset()):
            """The project's own selection, without the `machine` SDK distributions the machine's pair
            replaces next (the lock's may not install here: TensorFS has no aarch64 wheel on PyPI).
            Run again with none, it restores what the machine's pair replaced."""
            if (project / "uv.lock").is_file():
                # Each carried distribution at the version the lock selected, from its own
                # files (uv picks the one this platform takes); the lock installs the rest,
                # every dependency as an ordinary install: only the project itself is live.
                carried = sorted({tuple(wheel.name.split("-")[:2]) for wheel in wheels})
                sync = ["uv", "sync", "--frozen", "--no-dev", "--no-editable", "--no-install-project", "--no-python-downloads",
                        "--project", str(project), "--python", python,
                        *(f"--no-install-package={name}" for name in sorted({name for name, _ in carried} | machine)),
                        *(f"--refresh-package={name}" for name in local_builds(project))]
                placed = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
                subprocess.run(sync, env=placed, check=True)
                linked = linked_applications(interpreter) if COPY else []
                if linked:
                    subprocess.run([*sync, *COPY, *(f"--reinstall-package={name}" for name in linked)], env=placed, check=True)
                if carried:
                    subprocess.run(["uv", "pip", "install", *COPY, "--python", str(interpreter), "--no-deps", "--no-index",
                                    "--find-links", str(wheels[0].parent), *(f"{name}=={version}" for name, version in carried)], check=True)
                # Its dependencies are in place: the project alone, whatever indexes its sources name.
                subprocess.run(["uv", "pip", "install", "--python", str(interpreter), "--no-deps", "--no-sources",
                                "--editable", str(project)], check=True)
            else:
                subprocess.run(["uv", "venv", "--allow-existing", "--python", python, str(environment)], check=True)
                found = ["--find-links", str(wheels[0].parent)] if wheels else []
                named = [f"--index={index}" for index in indexes]
                subprocess.run(["uv", "pip", "install", "--python", str(interpreter), *found, *named, "--editable", str(project)], check=True)
            live_paths(environment)

        capture({canonicalize_name(wheel.name.split("-")[0]) for wheel in sdk})
        fallback = machine_sdk(interpreter, sdk, capture) if sdk else ""
        add_client(interpreter, environment.parent / (environment.name + ".constraints.txt"), client_wheel)
        ready.write_text(fallback)
    generations.mkdir(parents=True, exist_ok=True)
    root = generations.resolve() / uuid.uuid4().hex
    root.mkdir(); (root / ".hold").touch()
    try:
        (root / "env").symlink_to(environment.resolve(), target_is_directory=True)
        # Described by the environment's own Runtime, from the project's directory.
        live = root / "env" / "bin" / "python"
        interface = in_environment(live, Describe(str(project), str(live)), DESCRIPTION_DECODER).interface
        return publish_generation(root, metadata, interface, callees, ready.read_text(), source_digest)
    except BaseException:
        shutil.rmtree(root)  # uniquely owned unpublished generation, never dispatched
        raise


def install_captured(*, project: Path | None, wheels: list[Path], requirements: Path | None,
                     distribution: str, release: str, python_requires: str, python_version: str,
                     generations: Path, client_wheel: Path, python: str, sdk: list[Path] = (),
                     callees: dict[str, str] | None = None):
    python = interpreter_for(python, python_requires, python_version)
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
        frozen = project and (project / "uv.lock").is_file()
        if frozen:
            source = root / "source"
            shutil.copytree(project, source)
        else:
            subprocess.run(["uv", "venv", "--python", python, str(root / "env")], check=True)

        def capture():
            """The capture's own selection. Run again, it restores what the machine's pair replaced."""
            if frozen:
                # UV_PROJECT_ENVIRONMENT is a standard destination configuration value.
                environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(root / "env")}
                sync = ["uv", "sync", "--frozen", "--no-dev", "--no-editable", "--no-python-downloads",
                        "--project", str(source), "--python", python]
                if COPY:
                    # The locked dependencies link; then the project itself, and any App among
                    # them (a callee the lock holds by path), is copied.
                    subprocess.run([*sync, "--no-install-project"], env=environment, check=True)
                    subprocess.run([*sync, *COPY], env=environment, check=True)
                    linked = linked_applications(interpreter)
                    if linked:
                        subprocess.run([*sync, *COPY, *(f"--reinstall-package={name}" for name in linked)], env=environment, check=True)
                else:
                    subprocess.run(sync, env=environment, check=True)
            elif requirements and requirements.stat().st_size:
                # uv owns hashed requirements parsing and standard download integrity. The locked
                # dependencies link as the machine says (an image's seeded cache: symlinks).
                command = ["uv", "pip", "install", "--python", str(interpreter), "--require-hashes", "-r", str(requirements)]
                if wheels:
                    command += ["--find-links", str(wheels[0].parent)]
                subprocess.run(command, check=True)
                linked = linked_applications(interpreter) if COPY else []
                if linked:
                    subprocess.run([*command, *COPY, "--no-deps", *(f"--reinstall-package={name}" for name in linked)], check=True)
                if wheels:
                    subprocess.run(["uv", "pip", "install", *COPY, "--python", str(interpreter), "--no-deps", *map(str,wheels)], check=True)
            elif project:
                subprocess.run(["uv", "pip", "install", *COPY, "--python", str(interpreter), str(project), *map(str,wheels)], check=True)
            else:
                subprocess.run(["uv", "pip", "install", *COPY, "--python", str(interpreter), *map(str,wheels)], check=True)

        capture()
        fallback = machine_sdk(interpreter, sdk, capture) if sdk else ""
        add_client(interpreter, root / "captured-constraints.txt", client_wheel)
        return publish_generation(root, metadata, None, callees, fallback)
    except BaseException:
        shutil.rmtree(root)  # uniquely owned unpublished generation, never dispatched
        raise
