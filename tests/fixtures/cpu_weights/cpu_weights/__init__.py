"""A quantizer's shape without torch: a CPU job writes one weights output through its owner's
native writer channel, adopts it and returns the model."""
from __future__ import annotations

import msgspec
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor, derive

from cozy_runtime.author import App, Context, ModelArtifact, WeightsOutput

app = App()


class TableInput(msgspec.Struct):
    size: int = 64


async def table(ctx: Context, payload: TableInput) -> ModelArtifact:
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    shape = (payload.size,)
    declaration = Derivation(
        {},
        {"model": Target(add={"table": Tensor("f32", shape, plain, {"value": Part("f32", shape)})})},
        {},
        [("model", "table")],
    )
    output = derive(ctx.output("model"), declaration)
    if output.receipt is None:
        output.add_part("model", "table", "value", bytes(range(256)) * (payload.size // 64))
    return ctx.adopt_model(output.commit())


app.job(table, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)
