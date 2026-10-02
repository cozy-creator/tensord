"""Small blocking Linux client. FD ownership follows attach -> unmap -> close -> release."""
import array
import fcntl
import hashlib
import mmap
import os
import socket
import struct

import msgspec

from .protocol import (
    CAPABILITIES, MAX_FRAME, REPLY_DECODER, Attach, Attached, ErrorReply, Hello,
    HelloReply, Import, Imported, ObjectRef, Release, Released, Stats, StatsReply,
)


# Linux UAPI values from linux/fcntl.h. Some standalone Python builds omit
# the header-defined names despite supporting the kernel operations.
ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
FULL_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008


class ProtocolError(Exception):
    pass


class MachineError(Exception):
    def __init__(self, reply: ErrorReply):
        self.code = reply.code
        super().__init__(f"{reply.code}: {reply.detail}")


def read_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            raise ProtocolError("peer closed during a frame")
        chunks.extend(chunk)
    return bytes(chunks)


def send_record(sock: socket.socket, record: msgspec.Struct, fd: int | None = None):
    data = msgspec.json.encode(record)
    if not data or len(data) > MAX_FRAME:
        raise ProtocolError("frame length outside protocol limits")
    sock.sendall(struct.pack("!I", len(data)) + data)
    if fd is not None:
        sent = sock.sendmsg([b"\0"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])
        if sent != 1:
            raise ProtocolError("descriptor marker was not sent")


def receive_fd(sock: socket.socket) -> int:
    # Allocate enough ancillary space to detect/reject a small excess, and reject
    # MSG_CTRUNC for larger excess. Always close every descriptor on bad input.
    data, ancillary, flags, _ = sock.recvmsg(
        1, socket.CMSG_SPACE(16 * array.array("i").itemsize), socket.MSG_CMSG_CLOEXEC,
    )
    fds: list[int] = []
    unexpected = False
    for level, kind, value in ancillary:
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            unexpected = True
            continue
        ints = array.array("i")
        if len(value) % ints.itemsize:
            unexpected = True
        ints.frombytes(value[:len(value) - len(value) % ints.itemsize])
        fds.extend(ints)
    try:
        if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
            raise ProtocolError("descriptor control data was truncated")
        if data != b"\0" or unexpected or len(fds) != 1:
            raise ProtocolError("expected one descriptor with a NUL marker")
        if not fcntl.fcntl(fds[0], fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
            raise ProtocolError("received descriptor is not close-on-exec")
        return fds.pop()
    finally:
        for fd in fds:
            os.close(fd)


def sealed_memfd(data: bytes) -> int:
    fd = os.memfd_create("cozy-object", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(fd), "wb") as writer:
            writer.write(data)
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, ADD_SEALS, FULL_SEALS)
        return fd
    except BaseException:
        os.close(fd)
        raise


class BorrowedObject:
    def __init__(self, client: "Client", reply: Attached, fd: int):
        self.client, self.reply, self.fd = client, reply, fd
        self.mapping: mmap.mmap | None = None

    def map(self) -> mmap.mmap:
        if self.fd < 0 or self.reply.object.length == 0:
            raise ProtocolError("cannot map a closed or empty object")
        if self.mapping is None:
            self.mapping = mmap.mmap(self.fd, self.reply.object.length, access=mmap.ACCESS_READ)
        return self.mapping

    def close(self):
        if self.fd < 0:
            return
        # mmap.close raises BufferError if the caller still owns a NumPy view:
        # keep the fd and lease alive rather than claiming release prematurely.
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        os.close(self.fd)
        self.fd = -1
        self.client.release(self.reply.lease, self.reply.incarnation)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Client:
    def __init__(self, path: str):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.set_inheritable(False)
        try:
            self.socket.connect(path)
        except BaseException:
            self.socket.close()
            raise
        self.sequence = 0
        self.peer: HelloReply | None = None

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def exchange(self, request: msgspec.Struct, expected: type, fd: int | None = None):
        received_fd = None
        try:
            send_record(self.socket, request, fd)
            length = struct.unpack("!I", read_exact(self.socket, 4))[0]
            if not 0 < length <= MAX_FRAME:
                raise ProtocolError("reply length outside protocol limits")
            reply = REPLY_DECODER.decode(read_exact(self.socket, length))
            if isinstance(reply, Attached):
                received_fd = receive_fd(self.socket)
            if reply.seq != request.seq:
                raise ProtocolError("reply sequence does not match request")
            if isinstance(reply, ErrorReply):
                raise MachineError(reply)
            if not isinstance(reply, expected):
                raise ProtocolError("unexpected reply type")
            if received_fd is not None:
                if fcntl.fcntl(received_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
                    raise ProtocolError("attached descriptor is writable")
                if os.fstat(received_fd).st_size != reply.object.length:
                    raise ProtocolError("attached object length does not match descriptor")
                if reply.object != request.object:
                    raise ProtocolError("attached object identity does not match request")
                if self.peer is None or reply.incarnation != self.peer.incarnation:
                    raise ProtocolError("lease belongs to a different incarnation")
                return reply, received_fd
            return reply
        except MachineError:
            raise  # Operation errors do not invalidate a healthy transport.
        except BaseException:
            if received_fd is not None:
                os.close(received_fd)
            self.close()
            raise

    def hello(self, runtime: str = "experimental-python/0.1", capabilities=None) -> HelloReply:
        offered = list(CAPABILITIES if capabilities is None else capabilities)
        self.peer = self.exchange(Hello(self.next_sequence(), runtime, offered), HelloReply)
        if set(self.peer.capabilities) - set(offered):
            self.close()
            raise ProtocolError("peer advertised a capability the client did not offer")
        return self.peer

    def import_bytes(self, data: bytes) -> Imported:
        obj = ObjectRef(hashlib.sha256(data).hexdigest(), len(data))
        fd = sealed_memfd(data)
        try:
            reply = self.exchange(Import(self.next_sequence(), obj), Imported, fd)
            if reply.object != obj:
                self.close()
                raise ProtocolError("imported object identity changed")
            return reply
        finally:
            os.close(fd)

    def attach(self, obj: ObjectRef) -> BorrowedObject:
        reply, fd = self.exchange(Attach(self.next_sequence(), obj), Attached)
        return BorrowedObject(self, reply, fd)

    def release(self, lease: int, incarnation: str) -> Released:
        return self.exchange(Release(self.next_sequence(), lease, incarnation), Released)

    def stats(self) -> StatsReply:
        return self.exchange(Stats(self.next_sequence()), StatsReply)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
