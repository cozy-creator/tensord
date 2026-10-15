"""H3 long_form's shape on GPUs: a CPU job holding its run's GPUs calls children whose model
slot declares one GPU. Each child answers the device it ran on."""
from __future__ import annotations

import asyncio
import os

import msgspec
import torch

from cozy_runtime.author import App, Config, Context, Loader, Model, invocable, uses_components

app = App()


class Graph:
    """The checkpoint's one tensor, `model.weight`."""

    def __init__(self) -> None:
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.empty(2048), requires_grad=False)
        self.components = {"model": module}


def _graph(config: Config) -> Graph:
    return Graph()


class Tiny(Model[Graph]):
    """A constructed slot that declares no group: one GPU, as birefnet's and the describer's."""

    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=_graph)

    @uses_components("model")
    def weight(self) -> object:
        return self.graph.components["model"].weight


class Ran(msgspec.Struct):
    device: str
    pid: int


class Request(msgspec.Struct):
    hold_s: float = 0.0


class Children(msgspec.Struct):
    children: list[Ran]


@invocable
async def modeled(ctx: Context, *, model: Tiny, hold_s: float = 0.0) -> Ran:
    if ctx.device.type != "cuda":
        raise ValueError("a modeled child got no GPU")
    for _ in range(max(0, int(hold_s * 20))):
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
    return Ran(str(ctx.device), os.getpid())


app.entrypoint(internal=True)(modeled)


async def cpu_job(ctx: Context, payload: Request) -> Children:
    async def child() -> Ran:
        return await modeled(hold_s=payload.hold_s)

    return Children(list(await asyncio.gather(child(), child())))


app.job(cpu_job, accelerator=False)
