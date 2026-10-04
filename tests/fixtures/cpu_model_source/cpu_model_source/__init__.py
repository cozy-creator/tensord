"""A CPU native-source consumer; it constructs no inference model and opens no Store."""
from __future__ import annotations

import msgspec
from cozy_runtime.author import App, Context, Loader, Model

app = App()


class NativeSource(Model[object]):
    def load(self, loader: Loader) -> None:
        del loader


class Request(msgspec.Struct):
    pass


class Seen(msgspec.Struct):
    manifest: str
    components: list[str]


async def metadata(ctx: Context, payload: Request, *, source: NativeSource) -> Seen:
    with ctx.tensorfs_source(source) as capability:
        selected = capability.inspect()
        return Seen(selected.source.manifest, list(selected.components))


app.job(metadata, accelerator=False)
