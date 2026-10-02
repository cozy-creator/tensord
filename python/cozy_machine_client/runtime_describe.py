"""Trusted optional-SDK subprocess: source AST description, never package import."""
import sys
from pathlib import Path

import msgspec
from cozy_runtime.internal.static_interface import build

from .package_records import Describe, Described, DescribeFailed


def main():
    request = msgspec.json.decode(sys.stdin.buffer.read(), type=Describe)
    try:
        document = build(Path(request.project), environment_python=(
            Path(request.environment_python) if request.environment_python is not None else None))
        reply = Described(msgspec.Raw(msgspec.json.encode(document)))
    except Exception as exc:
        reply = DescribeFailed(getattr(exc, "code", type(exc).__name__), str(exc))
    sys.stdout.buffer.write(msgspec.json.encode(reply))


if __name__ == "__main__":
    main()
