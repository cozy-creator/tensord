"""Group fixture: one model slot declared for a 2-GPU group (Runtime cr-068). Starting it at
degree 2 makes rank 0 spawn a real follower executor; no torch is installed, so the start is
refused after the followers dialled back, and nothing ever touches a device."""
from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Loader, Model, sequence_parallel

app = App()


@sequence_parallel(degrees=(2,))
class Pair(Model[object]):
    def load(self, loader: Loader) -> None:
        return None


class Request(msgspec.Struct):
    n: int = 1


class Done(msgspec.Struct):
    n: int


@app.entrypoint
def ranks(payload: Request, model: Pair) -> Done:
    return Done(payload.n)
