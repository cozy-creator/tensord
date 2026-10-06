"""A job whose call is memoized: `measure` counts each time it truly runs in the file its
request names, so a call answered from its caller's known results shows as no run."""
from __future__ import annotations

from pathlib import Path

import msgspec

from cozy_runtime.author import App, Context, invocable

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
