"""Memoizable only when its caller gives a seed: `paint` counts each time it truly runs in the
file its request names, and an unseeded call draws a seed of its own."""
from __future__ import annotations

import random
from pathlib import Path

import msgspec

from cozy_runtime.author import App, Context, Memoize, invocable

app = App()


class Painted(msgspec.Struct):
    prompt: str
    seed: int


@invocable(memoize=Memoize(requires=("seed",)))
async def paint(ctx: Context, *, prompt: str, counter: str, seed: int | None = None) -> Painted:
    path = Path(counter)
    path.write_text(str(int(path.read_text()) + 1 if path.exists() else 1))
    return Painted(prompt, random.randrange(1 << 62) if seed is None else seed)


class Gallery(msgspec.Struct):
    paintings: list[Painted]


@invocable()
async def gallery(ctx: Context, *, prompt: str, counter: str, seed: int | None = None) -> Gallery:
    return Gallery([await paint(prompt=prompt, counter=counter, seed=seed) for _ in range(2)])


app.entrypoint(paint)
app.job(gallery, accelerator=False)
