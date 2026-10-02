"""Private same-user CPU service client; queries never cancel accepted work."""
import hashlib
import os
import struct

import msgspec
from .client import Client, MachineError, ProtocolError, read_exact, receive_fd, send_record
from .protocol import MAX_FRAME, ErrorReply

class Invocation(msgspec.Struct):
    package: str
    generation: str
    module: str
    entrypoint: str
    input: object

class Artifact(msgspec.Struct):
    name: str
    path: str
    sha256: str
    length: int

class ExecutionResult(msgspec.Struct):
    value: object
    artifacts: list[Artifact]

class ExecutionRecord(msgspec.Struct):
    id: str
    idempotency_key: str
    invocation: Invocation
    state: str
    revision: int
    attempt: int = 0
    waiting_reason: str | None = None
    completed_units: int = 0
    progress: str | None = None
    result: ExecutionResult | None = None
    failure: str | None = None
    cancel_actor: str | None = None

class Submit(msgspec.Struct, tag="submit", tag_field="kind"):
    seq: int
    key: str
    generation: str
    entrypoint: str
    input: object

class Query(msgspec.Struct, tag="execution", tag_field="kind"):
    seq: int
    id: str

class Cancel(msgspec.Struct, tag="cancel", tag_field="kind"):
    seq: int
    id: str

class ReadResult(msgspec.Struct, tag="read_result", tag_field="kind"):
    seq: int
    id: str
    index: int

class ExecutionReply(msgspec.Struct, tag="execution", tag_field="kind"):
    seq: int
    record: ExecutionRecord

class ArtifactReply(msgspec.Struct, tag="result_artifact", tag_field="kind"):
    seq: int
    id: str
    artifact: Artifact

DECODER=msgspec.json.Decoder(ExecutionReply | ArtifactReply | ErrorReply)

class ExecutionClient(Client):
    def __init__(self,path):
        super().__init__(path)
        self.hello(capabilities=["execution.cpu/1"])

    def call(self,record):
        fd=None
        try:
            send_record(self.socket,record)
            size=struct.unpack("!I",read_exact(self.socket,4))[0]
            if not 0<size<=MAX_FRAME: raise ProtocolError("invalid execution reply length")
            reply=DECODER.decode(read_exact(self.socket,size))
            if isinstance(reply,ArtifactReply): fd=receive_fd(self.socket)
            if reply.seq!=record.seq: raise ProtocolError("execution reply sequence differs")
            if isinstance(reply,ErrorReply): raise MachineError(reply)
            return reply,fd
        except MachineError:
            raise
        except BaseException:
            if fd is not None: os.close(fd)
            self.close()
            raise

    def submit(self,key,generation,entrypoint,inputs):
        reply,_=self.call(Submit(self.next_sequence(),key,generation,entrypoint,inputs))
        if not isinstance(reply,ExecutionReply): raise ProtocolError("expected execution receipt")
        return reply.record

    def get(self,identifier):
        reply,_=self.call(Query(self.next_sequence(),identifier))
        if not isinstance(reply,ExecutionReply): raise ProtocolError("expected execution state")
        return reply.record

    def cancel(self,identifier):
        reply,_=self.call(Cancel(self.next_sequence(),identifier))
        if not isinstance(reply,ExecutionReply): raise ProtocolError("expected execution state")
        return reply.record

    def read_result(self,identifier,index=0):
        reply,fd=self.call(ReadResult(self.next_sequence(),identifier,index))
        if not isinstance(reply,ArtifactReply) or fd is None: raise ProtocolError("expected result descriptor")
        try:
            with os.fdopen(fd,"rb") as source: data=source.read()
            if len(data)!=reply.artifact.length or hashlib.sha256(data).hexdigest()!=reply.artifact.sha256:
                raise ProtocolError("result bytes differ from durable custody record")
            return data
        except BaseException:
            self.close(); raise
