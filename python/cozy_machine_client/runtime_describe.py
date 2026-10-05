"""Trusted optional-SDK subprocess: source AST description, never package import."""
import importlib.metadata
import sys
from pathlib import Path

import msgspec
from cozy_runtime.internal import static_interface

from .distribution import normalized, package_name, record_digest
from .package_records import (Callee, Describe, DescribeEnvironment, DescribeInstalled, Described,
                              DescribedEnvironment, DescribeFailed)


def environment(root: str, packages: dict[str, str], hub_origin: str) -> DescribedEnvironment:
    """This environment's root digest and every other installed App, described statically."""
    found = {normalized(d.metadata["Name"]): d for d in importlib.metadata.distributions()}
    callees = []
    for key, distribution in sorted(found.items()):
        applications = [e for e in distribution.entry_points if e.group == "cozy.application"]
        if key == normalized(root) or len(applications) != 1:
            continue
        name = distribution.metadata["Name"]
        try:
            document = static_interface.build_installed(name, environment_python=Path(sys.executable))
        except Exception:  # an App the static reader refuses is not callable from here
            continue
        callees.append(Callee(name, distribution.version, applications[0].value,
                              msgspec.Raw(msgspec.json.encode(document)), record_digest(distribution),
                              package_name(distribution, packages, hub_origin)))
    return DescribedEnvironment(record_digest(found.get(normalized(root))), callees)


def main():
    request = msgspec.json.decode(sys.stdin.buffer.read(), type=Describe | DescribeInstalled | DescribeEnvironment)
    try:
        if isinstance(request, DescribeEnvironment):
            reply = environment(request.root, request.packages, request.hub_origin)
        else:
            document = (static_interface.build_installed(request.distribution, environment_python=Path(request.environment_python))
                        if isinstance(request, DescribeInstalled) else
                        static_interface.build(Path(request.project), environment_python=(
                            Path(request.environment_python) if request.environment_python is not None else None)))
            reply = Described(msgspec.Raw(msgspec.json.encode(document)))
    except Exception as exc:
        reply = DescribeFailed(getattr(exc, "code", type(exc).__name__), str(exc))
    sys.stdout.buffer.write(msgspec.json.encode(reply))


if __name__ == "__main__":
    main()
