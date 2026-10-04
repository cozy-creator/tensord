"""The selected SDK's encoder for one executor's audio/video frames.

Embedded by Rust and started once per executor: one JSON request per stdin line, one JSON
answer per stdout line; it ends on stdin EOF. It imports the selected installed SDK; no
model or codec implementation is copied.
"""
import hashlib
import json
import os
import stat
import sys
import uuid
from pathlib import Path

from cozy_runtime.author._assets import digest_bytes
from cozy_runtime.author._codec import FRAME_MEDIA_TYPES, encode_frame


def blob_path(spool, ref):
    tail = ref.rsplit("/", 2)
    if len(tail) < 3 or not tail[-1] or tail[-1] != Path(tail[-1]).name:
        return None
    return spool / f"{tail[-2]}-{tail[-1]}"


def read_regular(directory, name):
    if not name or name != os.path.basename(name) or name in (".", ".."):
        raise ValueError("SDK output has no single owned spool name")
    handle = os.open(name, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise ValueError("SDK output is not a regular file")
        with open(f"/proc/self/fd/{handle}", "rb") as stream:
            return stream.read()
    finally:
        os.close(handle)


def encode(request, directory):
    spool = Path(f"/proc/self/fd/{directory}")
    for frame in request["frames"]:
        raw = read_regular(directory, frame["raw"])
        if len(raw) != frame["raw_bytes"]:
            raise ValueError("registered raw frame length changed")
        destination = blob_path(spool, frame["handle"])
        if destination is None:
            raise ValueError("SDK frame has no owned output binding")
        encoded = encode_frame(frame["codec"], frame["facts"], raw)
        temporary = ".codec-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600,
                     dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, destination.name, src_dir_fd=directory, dst_dir_fd=directory)
    os.fsync(directory)
    bindings = []
    frames = {frame["handle"]: frame for frame in request["frames"]}
    total = 0
    for output in request["outputs"]:
        destination = blob_path(spool, output["asset_ref"])
        if destination is None:
            raise ValueError("SDK output has no owned spool binding")
        encoded = read_regular(directory, destination.name)
        producer = digest_bytes(encoded)
        if output.get("digest") and output["digest"] != producer:
            raise ValueError("SDK producer checksum differs from retained output")
        if output.get("size_bytes") is not None and output["size_bytes"] != len(encoded):
            raise ValueError("SDK output byte length changed")
        frame = frames.get(output["asset_ref"])
        media = FRAME_MEDIA_TYPES[frame["codec"]] if frame else output["media_type"]
        total += len(encoded)
        if request.get("max_output_bytes") is not None and total > request["max_output_bytes"]:
            raise ValueError("encoded output exceeds the authored aggregate allowance")
        bindings.append({"output_id": output["output_id"], "asset_ref": output["asset_ref"],
                         "name": destination.name, "kind": output["kind"], "media_type": media,
                         "length": len(encoded), "producer_digest": producer,
                         "sha256": hashlib.sha256(encoded).hexdigest()})
    return {"bindings": bindings}


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            directory = os.open(request["spool"], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                answer = encode(request, directory)
            finally:
                os.close(directory)
        except Exception as exc:  # one refused output never ends the executor's encoder
            answer = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
