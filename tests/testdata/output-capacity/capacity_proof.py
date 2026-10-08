"""CPU-only retained artifact fixture: no model, accelerator or external dependency."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated

import msgspec
from cozy_runtime.author import App, AssetBound, Context, Outputs, Tree

app = App()
CAPACITY = 64 << 30
LENGTH = (1 << 30) + (1 << 20)


class Request(msgspec.Struct):
    pass


class Result(msgspec.Struct):
    trace: Annotated[Tree, AssetBound(max_bytes=CAPACITY)]


class NestedResult(msgspec.Struct):
    artifact: Result


def retained(ctx: Context, out: Outputs) -> Result:
    if ctx.device.type != "cpu":
        raise RuntimeError("this fixture must run on CPU")
    if out._attempt.max_output_bytes != CAPACITY:
        raise RuntimeError(f"expected declared {CAPACITY}-byte grant, got {out._attempt.max_output_bytes}")
    directory = out.temporary_file()
    directory.mkdir()
    block = bytes(range(256)) * 4096
    digest = hashlib.sha256()
    with (directory / "payload.bin").open("wb") as stream:
        for _ in range(LENGTH // len(block)):
            stream.write(block)
            digest.update(block)
    (directory / "receipt.json").write_text(json.dumps({
        "length": LENGTH, "sha256": digest.hexdigest(),
        "capacity": out._attempt.max_output_bytes, "device": ctx.device.type,
    }))
    return Result(out.save_tree(directory))


@app.job
def job(payload: Request, ctx: Context, out: Outputs) -> Result:
    return retained(ctx, out)


@app.entrypoint
def serve(payload: Request, ctx: Context, out: Outputs) -> NestedResult:
    return NestedResult(retained(ctx, out))
