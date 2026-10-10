"""The trusted installer helper's command line: stdout carries exactly one typed record.

It is its own module so `PackageError` is one class: run as `-m ...packages`, that module would
be `__main__`, and the errors `captured_packages` raises would escape its handler untyped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import msgspec

from .captured_packages import install_captured
from .package_records import InstallFailed
from .packages import PackageError


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
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
    captured.add_argument("--callees", default="{}")
    args = parser.parse_args()
    try:
        result = install_captured(project=args.project, wheels=args.wheel, requirements=args.requirements,
            distribution=args.distribution, release=args.release, python_requires=args.python_requires,
            python_version=args.python_version, generations=args.generations, client_wheel=args.client_wheel,
            python=args.python, sdk=args.sdk_wheel, callees=msgspec.json.decode(args.callees, type=dict[str, str]))
    except PackageError as exc:
        print(msgspec.json.encode(InstallFailed(exc.code, str(exc))).decode())
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        # The machine attaches this helper's last stderr lines: the operation's own words.
        operation = " ".join(map(str, exc.cmd[:3]))
        print(msgspec.json.encode(InstallFailed("package_dependency_operation_failed", f"{operation} exited {exc.returncode}")).decode())
        raise SystemExit(1)
    print(msgspec.json.encode(result).decode())


if __name__ == "__main__":
    main()
