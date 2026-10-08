"""Ordinary CLI qualification of inline GPU allocation and managed descendants."""
from __future__ import annotations

import asyncio
import os

import msgspec

from cozy_runtime.author import App, Context, invocable

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    child_hold_s: float = 0.0


class DeviceResult(msgspec.Struct):
    device: str
    pid: int
    total: float


class BridgeResult(msgspec.Struct):
    device: str
    pid: int
    child: DeviceResult


class Result(msgspec.Struct):
    parent: DeviceResult
    bridge: BridgeResult
    retained_before: float
    retained_after: float


@invocable
async def gpu_child(ctx: Context, *, hold_s: float) -> DeviceResult:
    if ctx.device.type != "cuda":
        raise ValueError("GPU child received no admitted CUDA allocation")
    import torch

    tensor = torch.full((4096,), 2.0, device=str(ctx.device))
    for _ in range(max(0, int(hold_s * 20))):
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
    return DeviceResult(ctx.device.type, os.getpid(), float(tensor.sum().item()))


app.job(internal=True, accelerator=True)(gpu_child)


@invocable
async def cpu_bridge(ctx: Context, *, hold_s: float) -> BridgeResult:
    if ctx.device.type != "cpu":
        raise ValueError("CPU intermediary was incorrectly promoted to a GPU job")
    child = await gpu_child(hold_s=hold_s)
    return BridgeResult(ctx.device.type, os.getpid(), child)


app.job(internal=True, accelerator=False)(cpu_bridge)


@app.job(accelerator=True)
async def verify(payload: Request, ctx: Context) -> Result:
    if ctx.device.type != "cuda":
        raise ValueError("GPU parent received no admitted CUDA allocation")
    import torch

    retained = torch.ones(4096, device=str(ctx.device))
    before = float(retained.sum().item())
    bridge = await cpu_bridge(hold_s=payload.child_hold_s)
    after = float(retained.sum().item())
    if before != 4096 or after != before or bridge.child.total != 8192:
        raise ValueError("managed child changed the parent's retained CUDA tensor")
    if len({os.getpid(), bridge.pid, bridge.child.pid}) != 3:
        raise ValueError("parent and managed children did not execute independently")
    return Result(DeviceResult(ctx.device.type, os.getpid(), after), bridge, before, after)
