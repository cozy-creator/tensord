"""A job reads one input tree the machine materialized and reports its files; another makes a
tree in a child, hands it to a second child as its input and returns it as its own."""
from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime.author import App, Context, Outputs, Tree, invocable

app = App()


class Request(msgspec.Struct):
    data: Tree


class Response(msgspec.Struct):
    files: list[str]
    sha256: str


async def count(ctx: Context, payload: Request) -> Response:
    files = payload.data.files()
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    root = payload.data.path
    return Response([str(path.relative_to(root)) for path in files], digest.hexdigest())


class Bundled(msgspec.Struct):
    tree: Tree


@invocable()
async def bundle(ctx: Context, *, text: str, out: Outputs) -> Bundled:
    root = out.temporary_file()
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "a.txt").write_text(text)
    (root / "b.txt").write_text(text)
    (root / "empty.txt").write_bytes(b"")
    return Bundled(out.save_tree(root))


@invocable()
async def inspect(ctx: Context, *, data: Tree) -> Response:
    return await count(ctx, Request(data))


class Survey(msgspec.Struct):
    text: str


class Surveyed(msgspec.Struct):
    files: list[str]
    tree: Tree


async def survey(ctx: Context, payload: Survey) -> Surveyed:
    made = await bundle(text=payload.text)
    counted = await inspect(data=made.tree)
    return Surveyed(sorted(counted.files), made.tree)


app.job(count, accelerator=False)
app.job(bundle)
app.job(inspect)
app.job(survey)
