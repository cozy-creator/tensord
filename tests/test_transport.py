import array
import fcntl
import os
import socket
import struct

import msgspec
import pytest

from cozy_machine_client.client import FULL_SEALS, GET_SEALS, ProtocolError, read_exact, receive_fd, sealed_memfd, send_record
from cozy_machine_client.protocol import MAX_FRAME, Hello, HelloReply, ObjectRef


def descriptor_count():
    return len(os.listdir("/proc/self/fd"))


def test_sealed_upload_is_immutable_and_readable_from_start():
    fd = sealed_memfd(b"model weights")
    try:
        seals = fcntl.fcntl(fd, GET_SEALS)
        expected = FULL_SEALS
        assert seals & expected == expected
        assert os.read(fd, 100) == b"model weights"
        with pytest.raises(OSError):
            os.pwrite(fd, b"corrupted", 0)
    finally:
        os.close(fd)


def test_json_framing_and_cloexec_fd():
    sender, receiver = socket.socketpair()
    fd = sealed_memfd(b"weights")
    try:
        hello = Hello(1, "old-peer/0", ["weights.hosted/1"])
        send_record(sender, hello, fd)
        length = struct.unpack("!I", read_exact(receiver, 4))[0]
        assert msgspec.json.decode(read_exact(receiver, length), type=Hello) == hello
        received = receive_fd(receiver)
        try:
            assert fcntl.fcntl(received, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
            assert os.pread(received, 7, 0) == b"weights"
        finally:
            os.close(received)
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


@pytest.mark.parametrize("marker,count", [(b"x", 1), (b"\0", 0), (b"\0", 2), (b"\0", 40)])
def test_bad_descriptor_packet_rejected_without_leaks(marker, count):
    sender, receiver = socket.socketpair()
    fd = sealed_memfd(b"weights")
    try:
        before = descriptor_count()
        control = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd] * count))] if count else []
        sender.sendmsg([marker], control)
        with pytest.raises(ProtocolError):
            receive_fd(receiver)
        assert descriptor_count() == before
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


def test_truncated_stream_returns_error():
    sender, receiver = socket.socketpair()
    sender.sendall(b"ab")
    sender.close()
    try:
        with pytest.raises(ProtocolError, match="peer closed"):
            read_exact(receiver, 3)
    finally:
        receiver.close()


def test_oversize_frame_rejected_before_send():
    sender, receiver = socket.socketpair()
    try:
        with pytest.raises(ProtocolError, match="frame length"):
            send_record(sender, Hello(1, "x" * MAX_FRAME, []))
    finally:
        sender.close()
        receiver.close()


def test_additive_reply_fields_and_arbitrary_version():
    payload = b'{"kind":"hello","seq":1,"version":"future/999","tensorfs":"old/0","incarnation":"i","capabilities":[],"future":{"x":7}}'
    reply = msgspec.json.decode(payload, type=HelloReply)
    assert reply.version == "future/999"


@pytest.mark.parametrize("payload", [
    b'{"sha256":"bad","length":1}',
    b'{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","length":-1}',
    b'{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","length":true}',
])
def test_invalid_typed_object_refs(payload):
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(payload, type=ObjectRef)
