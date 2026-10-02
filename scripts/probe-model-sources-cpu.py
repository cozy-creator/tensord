#!/usr/bin/env python3
"""Real SDK model metadata/byte-source probe; does not import or invoke authored code."""
from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import socket
from pathlib import Path

import msgspec
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.model_sources import BrokeredModelSources
from cozy_runtime.internal.seam import Channel
from tensorfs import plane


class Evidence(msgspec.Struct, frozen=True):
    manifest: str
    store_path: str
    component: str
    tensors: int
    logical_bytes: int
    encoded_bytes: int
    host_sha256: str
    descriptor_count: int
    initial_fds: int
    filled_fds: int
    catalog_open: bool
    nvidia_open: bool
    cuda_mapped: bool
    sdk_torch_initialized: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket", type=Path)
    parser.add_argument("manifest")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = len(list(Path("/proc/self/fd").iterdir()))
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(args.socket))
    executor = Executor(Channel(connection), args.output / "executor")
    provider = BrokeredModelSources(executor._durable)
    checkpoint = Checkpoint("", args.manifest, source_provider=provider)
    assert checkpoint.store is None
    # The native planner requires the complete traversal for a requested component.
    component = "text_encoder"
    rows = checkpoint.rows(component)
    plan = checkpoint.read_plan([(component, row.name) for row in rows], 4 << 20, [component])
    owner = plane.Plane(devices=[], readers=1, direct_io=True)
    source = provider.source(owner, args.manifest, plan)
    assert provider.descriptors
    weights = owner.register(component, source, plan, [[f"{component}/{row.name}" for row in rows]])
    owner.set_pinned_budget(weights.nbytes)
    owner.want(weights, "pinned").wait()
    with mmap.mmap(weights.host_fd, weights.nbytes, access=mmap.ACCESS_READ) as filled:
        checksum = hashlib.sha256(filled).hexdigest()
    links = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            links.append(os.readlink(entry))
        except FileNotFoundError:
            pass
    maps = Path("/proc/self/maps").read_text()
    result = Evidence(args.manifest, "", component, len(rows), sum(row.nbytes for row in rows), weights.nbytes,
                      checksum, len(provider.descriptors), initial,
                      len(list(Path("/proc/self/fd").iterdir())),
                      any("tensorfs.sqlite" in link for link in links),
                      any(link.startswith("/dev/nvidia") for link in links),
                      "libcuda.so" in maps or "libnvidia-ml.so" in maps,
                      executor.torch is not None)
    assert not result.catalog_open and not result.nvidia_open and not result.cuda_mapped
    assert not result.sdk_torch_initialized
    (args.output / "evidence.json").write_bytes(msgspec.json.encode(result))
    weights.close()
    del source
    provider.close()
    owner.close()
    connection.close()
    print(msgspec.json.encode(result).decode())


if __name__ == "__main__":
    main()
