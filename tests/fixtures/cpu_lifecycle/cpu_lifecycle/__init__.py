"""Executor lifecycle fixture: cooperative steps, an uncooperative wedge, an abrupt exit."""
from __future__ import annotations

import os
import threading
import time

import msgspec

from cozy_runtime.author import App, Context, Telemetry

app = App()


class Steps(msgspec.Struct):
    steps: int = 3
    seconds: float = 0.05


class Done(msgspec.Struct):
    steps: int


@app.entrypoint
def steps(payload: Steps, ctx: Context, tel: Telemetry) -> Done:
    """Checks cancellation at every step and reports each one."""
    on_step = tel.step_callback(payload.steps, stage="steps")
    for index in range(payload.steps):
        ctx.raise_if_cancelled()
        time.sleep(payload.seconds)
        on_step(index)
    return Done(payload.steps)


@app.entrypoint
def wedge(payload: Steps, ctx: Context, tel: Telemetry) -> Done:
    """Reports one step, then blocks forever without checking cancellation."""
    tel.step_callback(1, stage="wedge")(0)
    threading.Event().wait()
    return Done(0)


@app.entrypoint
def exit_now(payload: Steps, ctx: Context) -> Done:
    """The process ends in the middle of a request."""
    os._exit(17)
