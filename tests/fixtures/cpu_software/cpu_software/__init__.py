"""A CPU job names the SDK it actually imported beside its ordinary result."""
import msgspec
from cozy_runtime import __version__
from cozy_runtime.author import App

app = App()


class Input(msgspec.Struct):
    value: int


class Output(msgspec.Struct):
    doubled: int
    runtime: str


async def compute(payload: Input) -> Output:
    return Output(payload.value * 2, __version__)


app.job(compute, accelerator=False)
