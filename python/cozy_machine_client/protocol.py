"""Additive typed JSON records; version strings identify peers, never admit them."""
from typing import Literal

import msgspec

CAPABILITIES = ("weights.hosted/1", "objects.put-fd/1")
MAX_FRAME = 65536


class ObjectRef(msgspec.Struct, frozen=True):
    sha256: str
    length: int

    def __post_init__(self):
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("object sha256 must be 64 lowercase hexadecimal characters")
        if self.length < 0:
            raise ValueError("object length must be nonnegative")


class Hello(msgspec.Struct, tag="hello", tag_field="kind"):
    seq: int
    runtime: str
    capabilities: list[str]


class Import(msgspec.Struct, tag="import", tag_field="kind"):
    seq: int
    object: ObjectRef


class Attach(msgspec.Struct, tag="attach", tag_field="kind"):
    seq: int
    object: ObjectRef


class Release(msgspec.Struct, tag="release", tag_field="kind"):
    seq: int
    lease: int
    incarnation: str


class Stats(msgspec.Struct, tag="stats", tag_field="kind"):
    seq: int


class HelloReply(msgspec.Struct, tag="hello", tag_field="kind"):
    seq: int
    version: str
    tensorfs: str
    incarnation: str
    capabilities: list[str]


class Imported(msgspec.Struct, tag="imported", tag_field="kind"):
    seq: int
    object: ObjectRef
    admitted: bool


class Attached(msgspec.Struct, tag="attached", tag_field="kind"):
    seq: int
    object: ObjectRef
    lease: int
    incarnation: str
    tier: Literal["host", "disk"]


class Released(msgspec.Struct, tag="released", tag_field="kind"):
    seq: int


class StatsReply(msgspec.Struct, tag="stats", tag_field="kind"):
    seq: int
    host_bytes: int
    cached_objects: int
    active_leases: int
    host_budget: int


class ErrorReply(msgspec.Struct, tag="error", tag_field="kind"):
    seq: int
    code: str
    detail: str


class Shutdown(msgspec.Struct, tag="shutdown", tag_field="kind"):
    seq: int


class ShutdownReply(msgspec.Struct, tag="shutdown", tag_field="kind"):
    seq: int


Reply = ShutdownReply | HelloReply | Imported | Attached | Released | StatsReply | ErrorReply
REPLY_DECODER = msgspec.json.Decoder(Reply)
