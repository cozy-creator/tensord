"""A job that calls another package's invocable (`cpu_memo.measure`), installed in its
environment as a dependency, as a published package's callee is."""
from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Context, ImageAsset, Outputs
from cpu_memo import Probe, greet, measure, nested, paint, restricted

app = App()


class Relay(msgspec.Struct):
    values: list[int]
    counter: str


class Relayed(msgspec.Struct):
    squares: list[int]
    greeting: str = ""


async def relay(ctx: Context, payload: Relay) -> Relayed:
    squares = []
    for value in payload.values:
        squares.append((await measure(payload=Probe(value, payload.counter))).square)
    # Another package's serving entrypoint, called with its request's fields (H3 -> qwen).
    greeted = await greet(text="squared", times=len(squares))
    return Relayed(squares=squares, greeting=greeted.text)


app.job(relay)


async def relay_nested(ctx: Context, payload: Relay) -> Relayed:
    squares = []
    for value in payload.values:
        squares.append((await nested(payload=Probe(value, payload.counter))).square)
    return Relayed(squares=squares)


app.job(relay_nested)


async def relay_restricted(ctx: Context, payload: Relay) -> Relayed:
    result = await restricted(payload=Probe(payload.values[0], payload.counter))
    return Relayed(squares=[result.square])


app.job(relay_restricted)


class Sitting(msgspec.Struct):
    names: list[str]


class Portraits(msgspec.Struct):
    references: list[ImageAsset]


async def portrait(ctx: Context, payload: Sitting, out: Outputs) -> Portraits:
    """H3's long_form references: each image another package's entrypoint returns is shown as
    this job's own output the moment it exists, and returned at the end."""
    references = []
    for index, name in enumerate(payload.names):
        result = await paint(prompt=name, shade=40 * (index + 1))
        out.publish("references", result.image, label=f"Reference: {name}")
        references.append(result.image)
    return Portraits(references)


app.job(portrait, emits_media=True)
