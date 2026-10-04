"""Real CPU authored telemetry: the final denoise sample immediately precedes decode."""
from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Telemetry

app = App()


class Request(msgspec.Struct):
    pass


class Done(msgspec.Struct):
    steps: int


@app.entrypoint
def burst(payload: Request, tel: Telemetry) -> Done:
    on_step = tel.step_callback(30, stage="denoise", overall_range=(0.1, 0.9))
    for index in range(30):
        on_step(index)
    tel.progress(0.0, stage="decoding", overall_fraction=0.95)
    return Done(30)
