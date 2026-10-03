"""Reads one file input the machine granted and reports what it received."""
from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime.author import App, Context, FileAsset

app = App()


class Request(msgspec.Struct):
    document: FileAsset


class Response(msgspec.Struct):
    length: int
    sha256: str
    media_type: str


@app.entrypoint
def measure(payload: Request, ctx: Context) -> Response:
    data = payload.document.read_bytes()
    return Response(len(data), hashlib.sha256(data).hexdigest(), payload.document.media_type)
