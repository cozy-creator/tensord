"""CPU executor records. Unknown advisory fields are accepted across peer versions."""
from typing import Any

import msgspec

MAX_EXECUTION_FRAME = 1 << 20


class Ready(msgspec.Struct, tag="ready", tag_field="kind"):
    pid: int
    capabilities: list[str]


class Invoke(msgspec.Struct, tag="invoke", tag_field="kind"):
    execution_id: str
    package: str
    generation: str
    module: str
    entrypoint: str
    input: dict[str, Any]
    output_root: str


class Cancel(msgspec.Struct, tag="cancel", tag_field="kind"):
    execution_id: str


class Progress(msgspec.Struct, tag="progress", tag_field="kind"):
    execution_id: str
    completed_units: int
    detail: str


class Result(msgspec.Struct, tag="result", tag_field="kind"):
    execution_id: str
    value: Any
    artifacts: list[str]


class Failed(msgspec.Struct, tag="failed", tag_field="kind"):
    execution_id: str
    code: str
    detail: str


class Canceled(msgspec.Struct, tag="canceled", tag_field="kind"):
    execution_id: str


Command = Invoke | Cancel
Event = Ready | Progress | Result | Failed | Canceled
COMMAND_DECODER = msgspec.json.Decoder(Command)
EVENT_DECODER = msgspec.json.Decoder(Event)
