"""Installed distributions, read from their own metadata."""
import hashlib
import importlib.metadata
import re

import msgspec


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def record_digest(found: importlib.metadata.Distribution | None) -> str:
    """A distribution's own installed files by their RECORD hashes: the same wherever this code
    is installed, so a memoized call's result is reusable there. Empty when unrecorded or
    editable (a path file's hash says nothing of the code)."""
    rows = sorted((str(file), file.hash.value) for file in (found.files if found else None) or []
                  if file.hash and file.parts[0] != ".." and not file.parts[0].endswith(".dist-info"))
    if not rows or any(path.endswith(".pth") for path, _ in rows):
        return ""
    return "sha256:" + hashlib.sha256(msgspec.json.encode(rows)).hexdigest()


def package_name(found: importlib.metadata.Distribution, packages: dict[str, str]) -> str:
    """The Hub package the machine read from the release's lock, else local code. An installed
    wheel's own provenance is never read: Hub rows install from local files, and a file door's
    URL is the Hub's to reshape (th-245)."""
    name = normalized(found.metadata["Name"])
    package = packages.get(name)
    if package is None:
        return f"local/{name}"
    org, separator, distribution = package.partition("/")
    if not separator or not org or "/" in distribution or normalized(distribution) != name:
        raise ValueError(f"callee {name!r} names a different package: {package!r}")
    return package
