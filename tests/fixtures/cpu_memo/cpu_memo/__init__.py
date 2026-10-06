"""A job whose call is memoized: `measure` counts each time it truly runs in the file its
request names, so a call answered from its caller's known results shows as no run."""
from __future__ import annotations

import io
import struct
from pathlib import Path

import msgspec
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import (App, Context, ImageAsset, ImageFrame, ModelArtifact, Outputs, WeightsOutput,
                                 invocable)

app = App()


class Probe(msgspec.Struct):
    value: int
    counter: str


class Measured(msgspec.Struct):
    square: int


@invocable(memoize=True)
async def measure(ctx: Context, *, payload: Probe) -> Measured:
    counter = Path(payload.counter)
    counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
    return Measured(square=payload.value * payload.value)


class Survey(msgspec.Struct):
    values: list[int]
    counter: str


class Surveyed(msgspec.Struct):
    squares: list[int]


async def survey(ctx: Context, payload: Survey) -> Surveyed:
    squares = []
    for value in payload.values:
        squares.append((await measure(payload=Probe(value, payload.counter))).square)
    return Surveyed(squares=squares)


app.entrypoint(measure)
app.job(survey)


@invocable()
async def nested(ctx: Context, *, payload: Probe) -> Measured:
    return await restricted(payload=payload)


@invocable(memoize=True)
async def restricted(ctx: Context, *, payload: Probe) -> Measured:
    counter = Path(payload.counter)
    counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
    return Measured(square=payload.value * payload.value)


app.job(nested)
app.entrypoint(restricted, internal=True)


class Greeting(msgspec.Struct):
    text: str
    times: int = 1


class Greeted(msgspec.Struct):
    text: str


@app.entrypoint
def greet(ctx: Context, payload: Greeting) -> Greeted:
    """A serving entrypoint another package calls with its request's fields, as H3 calls
    qwen-image-2's generate_image."""
    return Greeted(" ".join([payload.text] * payload.times))


class Painting(msgspec.Struct):
    prompt: str
    shade: int = 120


class Painted(msgspec.Struct):
    image: ImageAsset


@app.entrypoint
def paint(ctx: Context, payload: Painting, out: Outputs) -> Painted:
    """A serving entrypoint that returns an image, as qwen-image-2's generate_image does."""
    return Painted(out.save_image(ImageFrame(8, 8, bytes([payload.shade, 90, 160]) * 64), format="png"))


@invocable(memoize=True)
async def produce(ctx: Context) -> ModelArtifact:
    """A small f16 model this package writes, for a caller to quantize."""
    plain = dict(tensorfs.seed_digests())["plain/1"]
    matrix = Tensor("f16", (16, 32), plain, {"value": Part("f16", (16, 32))})
    bias = Tensor("f16", (16,), plain, {"value": Part("f16", (16,))})
    with ctx.output("model").open(Derivation(sources={}, targets={"body": Target(
            add={"layer.weight": matrix, "layer.bias": bias})},
            configs={}, order=(("body", "layer.weight"), ("body", "layer.bias")))) as output:
        if output.receipt is not None:
            return ctx.adopt_model(output.receipt)
        output.add_part("body", "layer.weight", "value",
                        io.BytesIO(struct.pack("<512e", *(i / 257 - 1 for i in range(512)))))
        output.add_part("body", "layer.bias", "value", io.BytesIO(struct.pack("<16e", *range(16))))
        return ctx.adopt_model(output.commit())


app.job(produce, weights=(WeightsOutput("model", max_new_bytes=65536),))
