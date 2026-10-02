#!/usr/bin/env python3
"""Partial control baseline: released SDK Executor, Python controller, same SDK codec.

This excludes the Go agent and Python worker. validate never starts a GPU process.
Run only on the benchmark owner's explicitly selected hardware.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run"))
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--codec-helper", required=True, type=Path)
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_text())
    if not args.root.is_absolute() or args.root.exists():
        raise ValueError("--root must be a new absolute owned directory")
    if not args.codec_helper.is_file() or not Path(pilot["python"]).is_file():
        raise ValueError("selected released interpreter and shared codec helper required")
    interface = json.loads(Path(pilot["package_interface"]).read_text())
    if interface["application"] != pilot["binding"]["application"] or not pilot["payloads"]:
        raise ValueError("coherent interface and unchanged payloads required")
    if args.action == "validate":
        print(json.dumps({"validated": True, "gpu_started": False, "partial_baseline": True,
                          "requests": len(pilot["payloads"]),
                          "codec_helper_sha256": hashlib.sha256(args.codec_helper.read_bytes()).hexdigest()}))
        return
    from cozy_runtime.internal.seam import Channel, RESULT_DOCUMENT
    from cozy_runtime.internal import executor_commands
    from PIL import Image
    root = args.root
    root.mkdir(mode=0o700)
    address = root / "seam.sock"
    if len(os.fsencode(address)) >= 108:
        raise ValueError("owned socket path exceeds Linux sockaddr_un capacity")
    hold = os.open(pilot["generation_hold"], os.O_RDONLY | os.O_CLOEXEC)
    fcntl.flock(hold, fcntl.LOCK_SH)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(address)); os.chmod(address, 0o600); listener.listen(1)
    out, err = (root / "stdout.log").open("wb"), (root / "stderr.log").open("wb")
    began = time.perf_counter()
    child = subprocess.Popen([pilot["python"], "-I", "-m", "cozy_runtime.internal.executor",
                              "--socket", str(address), "--root", str(root)],
                             env=pilot["environment"], stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                             pass_fds=(hold,))
    death = os.pidfd_open(child.pid)
    poll = select.poll(); poll.register(listener.fileno(), select.POLLIN); poll.register(death, select.POLLIN)
    while True:
        ready = {fd for fd, _ in poll.poll()}
        if listener.fileno() in ready:
            break
        if death in ready:
            raise RuntimeError(f"executor exited before connection: {child.wait()}")
    stream, _ = listener.accept()
    pid, uid, _ = struct.unpack("3i", stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if pid != child.pid or uid != os.geteuid():
        raise RuntimeError("executor seam peer differs from launched process")
    channel = Channel(stream)
    events = (root / "events.jsonl").open("w")
    phase = "hello"
    def command(name: str, **fields):
        body = {"cmd": name, **fields}
        executor_commands.decode(body)  # The selected SDK owns the typed command contract.
        channel.send(body)
        while True:
            frame = channel.recv()
            if frame is None:
                raise RuntimeError("executor exited before command reply")
            if frame.get("event"):
                events.write(json.dumps({"phase": phase, **frame}) + "\n"); events.flush()
                if frame["event"] == "request":
                    # Residency-only baseline: no machine weight-plane/stage ownership offered.
                    descriptor = channel.recv_memfd() if frame.get("descriptor") or frame.get("held") else None
                    if descriptor is not None:
                        os.close(descriptor)
                    channel.send({"event": "answer", "seq": frame["seq"], "ok": False,
                                  "code": "capability_unavailable", "detail": "partial stock residency control"})
                continue
            if frame.get("reply") != name or not frame.get("ok"):
                raise RuntimeError(f"{name} failed: {frame}")
            return frame
    hello = command("hello")
    if hello["pid"] != child.pid:
        raise RuntimeError("hello did not identify launched executor")
    if "weight_plane/1" in hello.get("memory", ()) or "stage/1" in hello.get("memory", ()):
        raise RuntimeError("this control is the residency-only SDK99 baseline; plane comparison requires a qualified adapter")
    evidence = {"qualification": "partial stock SDK Executor control; full old architecture unqualified",
                "ordinary_cli_qualified": False, "full_old_stack_qualified": False,
                "pid": child.pid, "hello": hello, "runs": [], "timings": {
                    "spawn_to_hello_ms": (time.perf_counter() - began) * 1000},
                "codec_helper_sha256": hashlib.sha256(args.codec_helper.read_bytes()).hexdigest(),
                "cold_start_excludes": ["Go agent", "Python worker", "public API", "pod boot/provisioning",
                    "dependency installation", "model download", "filesystem page-cache eviction"]}
    devices = hello["sealed"].get("CUDA_VISIBLE_DEVICES", "")
    for phase, name, fields in (
        ("start", "start", {"devices": devices, "application": pilot["binding"]["application"],
                             "package_interface": pilot["package_interface"], "sequence_parallel_degree": 1}),
        ("load", "load", {"construction": "pilot-model", "devices": devices, "sequence_parallel_degree": 1,
                           "binding": pilot["binding"], "budgets": {"declared_weight_bytes": pilot["logical_weight_bytes"]},
                           "authorized_device_limit_bytes": pilot["authorized_device_limit_bytes"],
                           "host_tier": False, "stages": False}),
        ("activate", "activate", {"construction": "pilot-model"}),
    ):
        began = time.perf_counter(); reply = command(name, **fields)
        evidence["timings"][phase + "_ms"] = (time.perf_counter() - began) * 1000
        (root / (phase + "-reply.json")).write_text(json.dumps(reply, indent=2))
    for index, payload in enumerate(pilot["payloads"]):
        phase = f"pilot-{index}"
        spool = root / phase; spool.mkdir(mode=0o700)
        all_start = time.perf_counter(); began = all_start
        command("prepare_request", request_id=phase, construction="pilot-model", entrypoint="generate", payload=payload)
        prepare_ms = (time.perf_counter() - began) * 1000; began = time.perf_counter()
        reply = command("invoke", request_id=phase, construction="pilot-model", entrypoint="generate",
                        spool=str(spool), deadline_s=None, plane_budget_bytes=pilot["plane_budget_bytes"], stages=False)
        invoke_ms = (time.perf_counter() - began) * 1000; began = time.perf_counter()
        if not reply.get("quiescent") or reply.get("poisoned") or reply.get("outcome", {}).get("terminal") != "succeeded":
            raise RuntimeError("attempt did not succeed and settle quiescent")
        raw = (spool / RESULT_DOCUMENT).read_bytes(); ref = reply["result_ref"]
        if len(raw) != ref["length"] or "sha256:" + hashlib.sha256(raw).hexdigest() != ref["digest"]:
            raise RuntimeError("spooled result checksum differs")
        directory = os.open(spool, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            encoded = subprocess.run([pilot["python"], "-I", str(args.codec_helper), "--spool-fd", str(directory)],
                input=json.dumps(reply), text=True, capture_output=True, check=True,
                env=pilot["environment"], pass_fds=(directory,))
        finally:
            os.close(directory)
        bindings = json.loads(encoded.stdout)["bindings"]
        if not bindings:
            raise RuntimeError("successful run has no retained output bindings")
        for binding in bindings:
            path = spool / binding["name"]
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != binding["sha256"] or path.stat().st_size != binding["length"]:
                raise RuntimeError("codec output checksum differs")
        # Rust postprocess independently rehashes the helper's registered bindings too.
        # Include that work in both post windows; only Pillow decoding stays outside.
        post_ms = (time.perf_counter() - began) * 1000
        wall_ms = (time.perf_counter() - all_start) * 1000
        evidence["runs"].append({"id": phase, "pid": child.pid, "payload": payload,
            "prepare_ms": prepare_ms, "invoke_ms": invoke_ms, "post_ms": post_ms, "wall_ms": wall_ms,
            "result": json.loads(raw), "bindings": bindings, "metrics": reply.get("metrics"), "plane": reply.get("plane")})
        (root / "results.json").write_text(json.dumps(evidence, indent=2))
    phase = "shutdown"; command("shutdown")
    if child.wait() != 0:
        raise RuntimeError("stock executor shutdown failed")
    # QA follows the complete GPU sequence, avoiding extra device gaps between requests.
    checked = []
    for row in evidence["runs"]:
        for binding in row["bindings"]:
            path = root / row["id"] / binding["name"]
            with Image.open(path) as image:
                image.load()
                if image.size != (1024, 1024):
                    raise RuntimeError("encoded dimensions changed")
                checked.append({"path": str(path), "dimensions": list(image.size)})
    evidence["image_checks"] = checked
    (root / "results.json").write_text(json.dumps(evidence, indent=2))
    stream.close(); listener.close(); os.close(death); os.close(hold); events.close(); out.close(); err.close()
    address.unlink()


if __name__ == "__main__":
    main()
