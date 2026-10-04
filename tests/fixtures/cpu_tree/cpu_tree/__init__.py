"""Reads one input tree the machine materialized and reports its files."""
from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime.author import App, Context, Tree

app = App()


class Request(msgspec.Struct):
    data: Tree


class Response(msgspec.Struct):
    files: list[str]
    sha256: str


@app.entrypoint
def count(payload: Request, ctx: Context) -> Response:
    files = payload.data.files()
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    root = payload.data.path
    return Response([str(path.relative_to(root)) for path in files], digest.hexdigest())
