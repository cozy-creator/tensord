#!/usr/bin/env python3
"""Profile actual SDK model-source RPC and native integrity construction separately."""
from __future__ import annotations

import argparse
import hashlib
import os
import resource
import socket
import ssl
import time
from pathlib import Path
from typing import TypeVar

import msgspec
from cozy_runtime.author._executor_requests import (
    Answer,
    ModelSourceRead,
    Request,
    SourceBlob,
)
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.model_sources import BrokeredModelSources
from cozy_runtime.internal.seam import Channel
from tensorfs import plane

A = TypeVar("A", bound=Answer)


class Timing(msgspec.Struct):
    count: int = 0
    bytes: int = 0
    wall_ms: float = 0
    user_ms: float = 0
    system_ms: float = 0


class Report(msgspec.Struct):
    manifest: str
    components: list[str]
    source_objects: int
    rpc: Timing
    native_constructor: Timing
    openssl_warm_same_fd: Timing
    openssl_version: str
    cpu_flags: list[str]
    whole_wall_ms: float
    store_path: str
    cuda_mapped: bool
    catalog_open: bool


def clock():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return time.perf_counter(), usage.ru_utime, usage.ru_stime


def add(timing: Timing, before, length: int) -> None:
    after = clock()
    timing.count += 1
    timing.bytes += length
    timing.wall_ms += (after[0] - before[0]) * 1000
    timing.user_ms += (after[1] - before[1]) * 1000
    timing.system_ms += (after[2] - before[2]) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket", type=Path)
    parser.add_argument("manifest")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(args.socket))
    executor = Executor(Channel(connection), args.output / "executor")
    rpc = Timing()
    native = Timing()
    openssl = Timing()

    def exchange(request: Request, into: type[A]) -> A:
        assert isinstance(request, ModelSourceRead)
        before = clock()
        result = executor._durable(request, into)
        assert isinstance(result, SourceBlob)
        add(rpc, before, result.length)
        return result

    constructor = plane.SourceDescriptor

    def measured_constructor(sha256: str, length: int, fd: int):
        before = clock()
        result = constructor(sha256, length, fd)
        add(native, before, length)
        # The default constructor ran all integrity checks. Compare OpenSSL on the same
        # still-open sender fd immediately afterward, explicitly a warm-cache measurement.
        before = clock()
        digest = hashlib.sha256()
        offset = 0
        while offset < length:
            block = os.pread(fd, min(1 << 20, length - offset), offset)
            if not block:
                raise RuntimeError("source ended during OpenSSL comparison")
            digest.update(block)
            offset += len(block)
        assert digest.hexdigest() == sha256
        add(openssl, before, length)
        return result

    # Diagnostic instrumentation delegates to the actual installed native constructor;
    # it does not bypass verification or modify the installed SDK/environment files.
    plane.SourceDescriptor = measured_constructor
    started = time.perf_counter()
    provider = BrokeredModelSources(exchange)
    checkpoint = Checkpoint("", args.manifest, source_provider=provider)
    owner = plane.Plane(devices=[], readers=1, direct_io=True)
    components = ["text_encoder", "text_encoder_2", "unet", "vae"]
    sources = []
    for component in components:
        rows = checkpoint.rows(component)
        traversal = [(component, row.name) for row in rows]
        plan = checkpoint.read_plan(traversal, 4 << 20, [component])
        sources.append(provider.source(owner, args.manifest, plan))
    elapsed = (time.perf_counter() - started) * 1000
    maps = Path("/proc/self/maps").read_text()
    links = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            links.append(os.readlink(entry))
        except FileNotFoundError:
            pass
    flags = next(row.split(":", 1)[1].split() for row in Path("/proc/cpuinfo").read_text().splitlines()
                 if row.startswith("flags"))
    report = Report(args.manifest, components, len(provider.descriptors), rpc, native, openssl,
                    ssl.OPENSSL_VERSION, flags, elapsed, "",
                    "libcuda.so" in maps or "libnvidia-ml.so" in maps,
                    any("tensorfs.sqlite" in link for link in links))
    assert not report.cuda_mapped and not report.catalog_open and executor.torch is None
    (args.output / "profile.json").write_bytes(msgspec.json.encode(report))
    sources.clear()
    provider.close()
    owner.close()
    plane.SourceDescriptor = constructor
    connection.close()
    print(msgspec.json.encode(report).decode())


if __name__ == "__main__":
    main()
