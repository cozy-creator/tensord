#!/usr/bin/env python3
"""CPU gate for a broker-exported real model object with the candidate native source API.

This checks the descriptor receiver, not model inference or GPU/host-tier ownership.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import msgspec


class Evidence(msgspec.Struct):
    header_sha256: str
    header_length: int
    tensor_count: int
    plan_items: int
    object_sha256: str
    object_length: int
    copied: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensorfs_python", type=Path)
    parser.add_argument("extension", type=Path)
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    facts = msgspec.json.decode(
        (arguments.evidence / "metadata.json").read_bytes(), type=Evidence
    )
    if facts.copied != facts.object_length:
        raise ValueError("broker sample differs from its declared length")
    sys.path.insert(0, str(arguments.tensorfs_python.resolve()))
    spec = importlib.util.spec_from_file_location(
        "tensorfs._ext", arguments.extension.resolve()
    )
    if spec is None or spec.loader is None:
        raise ValueError("native extension cannot be loaded")
    extension = importlib.util.module_from_spec(spec)
    sys.modules["tensorfs._ext"] = extension
    spec.loader.exec_module(extension)
    # Optional native dependency is loaded only after selecting its explicit test artifact.
    plane = importlib.import_module("tensorfs.plane")
    fd = os.open(
        arguments.evidence / "selected-object", os.O_RDONLY | os.O_CLOEXEC
    )
    try:
        descriptor = plane.SourceDescriptor(facts.object_sha256, facts.object_length, fd)
    finally:
        os.close(fd)
    assert descriptor.sha256 == facts.object_sha256
    assert descriptor.length == facts.object_length
    owner = plane.Plane(devices=[], readers=1, direct_io=False)
    source = owner.source_from_descriptors([descriptor])
    del source, descriptor
    owner.close()
    maps = Path("/proc/self/maps").read_text()
    assert "libcuda.so" not in maps and "libnvidia-ml.so" not in maps
    print(msgspec.json.encode(facts).decode())
    print("PASS native readonly descriptor source for actual broker-selected model bytes; no CUDA/NVML")


if __name__ == "__main__":
    main()
