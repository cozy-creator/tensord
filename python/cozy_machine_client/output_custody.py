"""Bounded-copy sealed output snapshots; live executors cannot alter handed-off bytes."""
import fcntl
import hashlib
import os
import stat
from pathlib import Path

from .client import ADD_SEALS, FULL_SEALS
from .execution_protocol import OutputFacts
from .linux import memfd_create
from .protocol import ObjectRef
from .session_protocol import OutputArtifact


def snapshot(spool: Path, facts: OutputFacts) -> tuple[OutputArtifact, int]:
    name = facts.relative_path
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("artifact must be a relative spool filename")
    source = os.open(spool / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    target = -1
    try:
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("artifact must be a regular file")
        target = memfd_create("cozy-output")
        digest = hashlib.sha256()
        sdk_digest = (hashlib.blake2b(digest_size=16) if facts.checksum.algorithm == "blake2b-128"
                      else hashlib.sha256())
        length = 0
        while chunk := os.read(source, 65536):
            digest.update(chunk)
            sdk_digest.update(chunk)
            length += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target, view)
                if written <= 0:
                    raise OSError("artifact snapshot write made no progress")
                view = view[written:]
        after = os.fstat(source)
        before_state = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_state = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_state != after_state or length != before.st_size:
            raise ValueError("artifact changed during its snapshot")
        if length != facts.length or sdk_digest.hexdigest() != facts.checksum.value:
            raise ValueError("artifact snapshot does not match the SDK output identity")
        fcntl.fcntl(target, ADD_SEALS, FULL_SEALS)
        readonly = os.open(f"/proc/self/fd/{target}", os.O_RDONLY | os.O_CLOEXEC)
        return OutputArtifact(name, ObjectRef(digest.hexdigest(), length)), readonly
    finally:
        os.close(source)
        if target >= 0:
            os.close(target)
