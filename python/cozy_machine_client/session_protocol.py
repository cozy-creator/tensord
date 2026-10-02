"""Separately negotiated session and immutable output-custody records."""
from typing import Any

import msgspec

from .execution_protocol import Invoke
from .protocol import ObjectRef


class OpenSession(msgspec.Struct, tag="open_session", tag_field="kind"):
    session_id: str
    package: str
    generation: str
    module: str


class SessionOpened(msgspec.Struct, tag="session_opened", tag_field="kind"):
    session_id: str


class SessionInvoke(msgspec.Struct, tag="session_invoke", tag_field="kind"):
    session_id: str
    seq: int
    invocation: Invoke


class SessionCancel(msgspec.Struct, tag="session_cancel", tag_field="kind"):
    session_id: str
    seq: int
    execution_id: str


class SessionAck(msgspec.Struct, tag="session_ack", tag_field="kind"):
    session_id: str
    seq: int
    execution_id: str


class SessionShutdown(msgspec.Struct, tag="session_shutdown", tag_field="kind"):
    session_id: str


class OutputArtifact(msgspec.Struct, frozen=True):
    relative_path: str
    object: ObjectRef
    role: str = "artifact"


class SessionProgress(msgspec.Struct, tag="session_progress", tag_field="kind"):
    session_id: str
    seq: int
    execution_id: str
    completed_units: int
    detail: str


class SessionResult(msgspec.Struct, tag="session_result", tag_field="kind"):
    session_id: str
    seq: int
    execution_id: str
    value: Any
    artifacts: list[OutputArtifact]


class SessionFailed(msgspec.Struct, tag="session_failed", tag_field="kind"):
    session_id: str
    seq: int
    execution_id: str
    code: str
    detail: str


class SessionCanceled(msgspec.Struct, tag="session_canceled", tag_field="kind"):
    session_id: str
    seq: int
    execution_id: str


class ReadyNext(msgspec.Struct, tag="ready_next", tag_field="kind"):
    session_id: str
    completed_seq: int
    sdk_imports: int
    package_imports: int
    sdk_module_identity: int
    package_module_identity: int


class SessionError(msgspec.Struct, tag="session_error", tag_field="kind"):
    session_id: str
    code: str
    detail: str
    seq: int = 0


Command = OpenSession | SessionInvoke | SessionCancel | SessionAck | SessionShutdown
Terminal = SessionResult | SessionFailed | SessionCanceled
Event = SessionOpened | SessionProgress | Terminal | ReadyNext | SessionError
COMMAND_DECODER = msgspec.json.Decoder(Command)
EVENT_DECODER = msgspec.json.Decoder(Event)
