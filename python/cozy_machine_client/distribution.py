"""Installed distributions, read from their own metadata."""
import hashlib
import importlib.metadata
import re
from urllib.parse import unquote, urlsplit

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
    """The explicit capture identity, else the Hub index of its locked direct wheel URL.

    Reading installed PEP 610 metadata needs neither imports nor another catalog read.
    A wheel without Hub provenance is local code, never guessed to belong to the root's org.
    """
    name = normalized(found.metadata["Name"])
    if name in packages:
        package = packages[name]
        org, separator, distribution = package.partition("/")
        if not separator or not org or "/" in distribution or normalized(distribution) != name:
            raise ValueError(f"callee {name!r} names a different package: {package!r}")
        return package
    record = found.read_text("direct_url.json")
    if record:
        url = urlsplit(msgspec.json.decode(record).get("url", ""))
        path = [unquote(part) for part in url.path.split("/")]
        if (url.scheme in ("http", "https") and url.hostname and not url.username
                and len(path) == 7 and path[1:3] == ["v1", "index"]
                and path[4] == "files" and path[3] and "/" not in path[3]):
            return f"{path[3]}/{name}"
    return f"local/{name}"
