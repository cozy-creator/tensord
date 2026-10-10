"""Trusted optional-SDK subprocess: source AST description, never package import."""
import importlib.metadata
import sys
from pathlib import Path

import msgspec
from cozy_runtime.internal import static_interface

from .distribution import normalized, package_name, record_digest
from .package_records import (Callee, Describe, DescribeEnvironment, DescribeInstalled, Described,
                              DescribedEnvironment, DescribeFailed)


#: The package a Runtime's built-in operations run as (the Runtime's rows name them `builtin`).
BUILTIN_PACKAGE = "runtime/operations"


class Undescribed(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def environment(root: str, packages: dict[str, str]) -> DescribedEnvironment:
    """This environment's root digest and every other installed App, described statically.
    An App the reader refuses fails the description: left out, its callers would call its
    raw function instead of the package."""
    found = {normalized(d.metadata["Name"]): d for d in importlib.metadata.distributions()}
    callees = []
    for key, distribution in sorted(found.items()):
        applications = [e for e in distribution.entry_points if e.group == "cozy.application"]
        if key == normalized(root) or not applications:
            continue
        name = distribution.metadata["Name"]
        try:
            document = static_interface.build_installed(name, environment_python=Path(sys.executable))
        except Exception as exc:
            raise Undescribed(getattr(exc, "code", type(exc).__name__),
                              f"installed package {name} {distribution.version} cannot be described: {exc}") from exc
        callees.append(Callee(name, distribution.version, applications[0].value,
                              msgspec.Raw(msgspec.json.encode(document)), record_digest(distribution),
                              package_name(distribution, packages)))
    runtime = found.get("cozy-runtime")
    if runtime is not None and hasattr(static_interface, "build_builtin"):
        # The Runtime's own operations (quantize, prepare_model) are an App every environment
        # holds: a job calls them as child runs, keyed by this Runtime's exact install.
        document = static_interface.build_builtin("operations")
        callees.append(Callee("cozy-runtime-operations", runtime.version, document["application"],
                              msgspec.Raw(msgspec.json.encode(document)), record_digest(runtime),
                              BUILTIN_PACKAGE))
    return DescribedEnvironment(record_digest(found.get(normalized(root))), callees)


def main():
    request = msgspec.json.decode(sys.stdin.buffer.read(), type=Describe | DescribeInstalled | DescribeEnvironment)
    try:
        if isinstance(request, DescribeEnvironment):
            reply = environment(request.root, request.packages)
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
