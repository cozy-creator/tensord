"""A job that calls another package's invocable (`cpu_memo.measure`), installed in its
environment as a dependency, as a published package's callee is."""
from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Context
from cpu_memo import Probe, measure

app = App()


class Relay(msgspec.Struct):
    values: list[int]
    counter: str


class Relayed(msgspec.Struct):
    squares: list[int]


async def relay(ctx: Context, payload: Relay) -> Relayed:
    squares = []
    for value in payload.values:
        squares.append((await measure(payload=Probe(value, payload.counter))).square)
    return Relayed(squares=squares)


app.job(relay)
