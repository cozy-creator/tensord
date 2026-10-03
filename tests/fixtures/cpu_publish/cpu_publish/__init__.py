"""Publishes products while it runs: list items grow, a single output is replaced."""
from __future__ import annotations

import time
from pathlib import Path

import msgspec

from cozy_runtime.author import App, Context, FileAsset, Outputs

app = App()


class Request(msgspec.Struct):
    gate: str = ""  # after its first publishes, the run waits for this file


class Response(msgspec.Struct):
    frames: list[FileAsset]
    preview: FileAsset


@app.entrypoint
def make(payload: Request, ctx: Context, out: Outputs) -> Response:
    first = out.save_bytes(b"frame-1", media_type="application/octet-stream")
    out.publish("frames", first, label="Frame 1")
    draft = out.save_bytes(b"preview-draft", media_type="text/plain")
    out.publish("preview", draft, label="Draft")
    while payload.gate and not Path(payload.gate).exists():
        ctx.raise_if_cancelled()
        time.sleep(0.01)
    second = out.save_bytes(b"frame-2", media_type="application/octet-stream")
    out.publish("frames", second, label="Frame 2")
    third = out.save_bytes(b"frame-3", media_type="application/octet-stream")
    final = out.save_bytes(b"preview-final", media_type="text/plain")
    return Response(frames=[first, second, third], preview=final)


class Nothing(msgspec.Struct):
    pass


@app.entrypoint
def explode(payload: Nothing, ctx: Context, out: Outputs) -> Response:
    """Fails inside authored code: its traceback is what a triage bundle keeps."""
    raise ValueError("boom from the fixture")
