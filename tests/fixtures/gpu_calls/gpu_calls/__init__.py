"""One path for every call on a GPU machine: a deviceless root calls a GPU child, a GPU root
calls a child on its own GPU, and a raw-GPU call runs beside a warm executor without ending
it. Each answers its process and device so a caller sees where it ran."""
from __future__ import annotations

import asyncio
import os

import msgspec

from cozy_runtime.author import App, Context, invocable

app = App()


class Ran(msgspec.Struct):
    device: str
    pid: int
    total: float


class Orchestrated(msgspec.Struct):
    root: Ran
    child: Ran


class Warm(msgspec.Struct):
    pid: int
    #: The calls this process has served, its retained CUDA tensor intact through them.
    served: int


def _torch():
    import torch

    return torch


@invocable
async def gpu_child(ctx: Context, *, hold_s: float = 0.0) -> Ran:
    """Undeclared: the machine's GPU."""
    if ctx.device.type != "cuda":
        raise ValueError("a GPU child got no GPU")
    tensor = _torch().full((4096,), 2.0, device=str(ctx.device))
    for _ in range(max(0, int(hold_s * 20))):
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
    return Ran(ctx.device.type, os.getpid(), float(tensor.sum().item()))


app.entrypoint(internal=True)(gpu_child)


class Request(msgspec.Struct):
    hold_s: float = 0.0


@app.entrypoint(accelerator=False)
async def cpu_root(ctx: Context, payload: Request) -> Orchestrated:
    """Deviceless: it never opens CUDA, and calls a GPU child."""
    if ctx.device.type != "cpu":
        raise ValueError("a deviceless root got a device")
    child = await gpu_child(hold_s=payload.hold_s)
    return Orchestrated(Ran("cpu", os.getpid(), 0.0), child)


@app.entrypoint
async def gpu_root(ctx: Context, payload: Request) -> Orchestrated:
    """On the GPU, holding a tensor while its child runs on the same GPU."""
    if ctx.device.type != "cuda":
        raise ValueError("a GPU root got no GPU")
    retained = _torch().ones(4096, device=str(ctx.device))
    child = await gpu_child(hold_s=payload.hold_s)
    if float(retained.sum().item()) != 4096 or child.total != 8192:
        raise ValueError("the child changed the root's tensor")
    if child.pid == os.getpid():
        raise ValueError("the child ran in its root's process")
    return Orchestrated(Ran(ctx.device.type, os.getpid(), 4096.0), child)


_held: list[object] = []


@app.entrypoint
async def warm(ctx: Context, payload: Request) -> Warm:
    """Keeps 256 MiB of CUDA memory across its calls, as a loaded model would."""
    if not _held:
        _held.append(_torch().ones(64 << 20, device=str(ctx.device)))
    _held.append(None)
    if float(_held[0].sum().item()) != 64 << 20:
        raise ValueError("the retained tensor changed")
    return Warm(os.getpid(), len(_held) - 1)


@app.entrypoint
async def raw(ctx: Context, payload: Request) -> Ran:
    """A raw-GPU call: its own torch memory, no model."""
    tensor = _torch().full((32 << 20,), 3.0, device=str(ctx.device))
    return Ran(ctx.device.type, os.getpid(), float(tensor.sum().item()))
