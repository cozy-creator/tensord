"""A quantizer's shape without torch: a CPU job writes one weights output through its owner's
native writer channel, adopts it and returns the model."""
from __future__ import annotations

import asyncio

import msgspec
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor, derive

from cozy_runtime.author import App, Context, ModelArtifact, Telemetry, WeightsOutput

app = App()


class TableInput(msgspec.Struct):
    size: int = 64
    hold: str = ""


async def table(ctx: Context, payload: TableInput, tel: Telemetry) -> ModelArtifact:
    async def hold(phase: str) -> None:
        if payload.hold == phase:
            tel.step_callback(1, stage=f"native {phase}")(0)
            while True:
                ctx.raise_if_cancelled()
                await asyncio.sleep(0.01)

    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    shape = (payload.size,)
    declaration = Derivation(
        {},
        {"model": Target(add={"table": Tensor("f32", shape, plain, {"value": Part("f32", shape)})})},
        {},
        [("model", "table")],
    )
    with derive(ctx.output("model"), declaration) as output:
        if output.receipt is None:
            output.add_part("model", "table", "value", bytes(range(256)) * (payload.size // 64))
            if payload.hold == "checkpoint":
                output.checkpoint()
                await hold("checkpoint")
        receipt = output.commit()
        await hold("commit")
        result = ctx.adopt_model(receipt)
        await hold("adopt")
        return result


app.job(table, weights=(WeightsOutput("model", max_new_bytes=1 << 20),), accelerator=False)
