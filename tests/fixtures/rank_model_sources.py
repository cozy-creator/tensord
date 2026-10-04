"""Two real ranks consume Rust-owned model bytes with the plane-only SDK.

No CUDA, Store setup, model inference, or replaced storage implementation. RankGroup
and its production follower trampoline carry the actual mid-command source exchange.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import msgspec
import tensorfs

from cozy_runtime.author._executor_requests import SealedPlan
from cozy_runtime.internal import canonical, plane, spawn
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.fill import Checkpoint, FillRefusal
from cozy_runtime.internal.model_config import construction_config
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.seam import Channel


def consume(executor: Executor, root: Path, manifest: str) -> dict[str, Any]:
    assert tensorfs.STORE is False, "this proof requires the deployed plane-only wheel"
    assert not (root / "no-store-here").exists()
    checkpoint = Checkpoint(
        root / "no-store-here", manifest,
        sourced=lambda name: executor._model_source(manifest, name),
    )
    assert checkpoint.store is None
    with_refusal = False
    try:
        checkpoint.acquire()
    except FillRefusal as exc:
        assert "reads no store" in str(exc)
        with_refusal = True
    assert with_refusal
    config = canonical.parse_canonical(construction_config(checkpoint.header["configs"]))
    assert config == {"channels": 2, "fixture": "rank-source"}
    assets = checkpoint.model_assets()
    assert assets == {"tokenizer/vocab.json": b"static tokenizer asset"}
    traversal = tuple(("allowed", key) for key in checkpoint.header["components"]["allowed"])
    read_plan = checkpoint.read_plan(traversal, 4 << 20, ("allowed",))
    all_parts = tuple(item["what"] for item in read_plan.items())
    objects = sorted({
        (item["object"].removeprefix("sha256:"), item["object_length"])
        for item in read_plan.items() if item["source"] == "object"
    })
    opened = plane.open_plane(None)
    held_files: list[tuple[str, int, int]] = []
    layouts: list[str] = []
    try:
        for grouping in [(all_parts,), tuple((part,) for part in all_parts)]:
            digest = canonical.digest({"manifest": manifest, "regions": grouping})
            doc = SealedPlan(
                manifest=manifest, name="rank-source/allowed/" + digest,
                layout=digest,
                window=4 << 20, traversal=traversal, components=("allowed",), regions=grouping,
            )
            # Production _tiers chooses the machine's sealed callback on every rank.
            tiers = executor._tiers(False, True, True)
            assert tiers is not None and tiers.seal is not None and tiers.files is not None
            fd = tiers.seal(doc)
            assert fd is not None and plane.is_sealed(fd)
            try:
                ws = opened.register_sealed(doc.name, read_plan, grouping, fd)
            finally:
                os.close(fd)
            opened.set_pinned_budget(32 << 20)
            opened.want(ws, "pinned").wait()
            actual = [
                hashlib.sha256(os.pread(ws.host_fd, part["nbytes"], part["offset"])).hexdigest()
                for part in ws.parts
            ]
            assert actual == [item["object"].removeprefix("sha256:") for item in read_plan.items()]
            layouts.append(ws.digest)
            files = tiers.files(doc, objects)
            assert len(files) == len(objects)
            for digest, length, descriptor in files:
                assert hashlib.sha256(os.pread(descriptor, length, 0)).hexdigest() == digest
                try:
                    os.pwrite(descriptor, b"x", 0)
                except OSError:
                    pass
                else:
                    raise AssertionError("the machine granted a writable object")
            held_files.extend(files)
        assert plane.stats(opened).host.fill_bytes == 0
        # Metadata presence grants no object authority: this component is in the header
        # and materialized store, but outside the machine's selected HostGrant.
        denied_plan = checkpoint.read_plan((("denied", "weight"),), 4 << 20, ("denied",))
        denied_objects = sorted({
            (item["object"].removeprefix("sha256:"), item["object_length"])
            for item in denied_plan.items()
        })
        denied = SealedPlan(
            manifest=manifest, name="rank-source/denied", layout="sha256:" + "0" * 64,
            window=4 << 20, traversal=(("denied", "weight"),), components=("denied",),
            regions=(("denied/weight#value",),),
        )
        assert executor._object_files(denied, denied_objects) == []
        refused = []
        for requested, name in [("sha256:" + "0" * 64, ""), (manifest, "../tfs.sqlite")]:
            try:
                executor._model_source(requested, name)
            except RuntimeError as exc:
                assert "model_source_refused" in str(exc)
                refused.append(name or "unselected-manifest")
            else:
                raise AssertionError("a rank acquired an unselected model source")
        return {
            "store": checkpoint.store is not None, "rank": executor.rank,
            "header": hashlib.sha256(checkpoint.header_bytes).hexdigest(),
            "config": config, "assets": sorted(assets), "layouts": layouts,
            "files": held_files, "refused": refused,
        }
    finally:
        opened.close()


def follower(args: argparse.Namespace) -> None:
    _capture_seal()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor = Executor(channel, args.root, rank=args.rank, world=args.world)
    command = channel.recv()
    assert command is not None and command["cmd"] == "consume"
    facts = consume(executor, args.root, command["manifest"])
    files = facts.pop("files")
    channel.send({"reply": "consume", "ok": True, **facts})
    command = channel.recv()
    assert command is not None and command["cmd"] == "hold"
    # This owned fault arm tests the full scope, not just the leader's group/pdeathsig.
    assert ctypes.CDLL(None).prctl(1, 0, 0, 0, 0) == 0  # PR_SET_PDEATHSIG
    os.setsid()
    channel.send({"reply": "hold", "ok": True, "pid": os.getpid()})
    with (args.root / "release-follower").open("rb") as release:
        assert release.read() == b"release"
    # Reader bytes remain correct after leader exit and an independent native GC.
    for digest, length, descriptor in files:
        assert hashlib.sha256(os.pread(descriptor, length, 0)).hexdigest() == digest
        os.close(descriptor)
    (args.root / "follower-completed.json").write_text(json.dumps({"ok": True, **facts}))


def leader(args: argparse.Namespace) -> None:
    _capture_seal()
    channel = Channel(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
    channel.sock.connect(str(args.socket))
    channel.send({"event": "fixture_ready"})
    assert channel.recv()["ok"] is True  # native custody is installed before any source ask
    executor = Executor(channel, args.root)

    def launch(argv: list[str], descriptor: int) -> spawn.Child:
        # Only the fixture program changes; production inheritance/trampoline is unchanged.
        return group._spawn_follower([__file__, *argv[2:]], descriptor)

    group = RankGroup(degree=2, backend="cpu:gloo", root=str(args.root),
                      launch=launch, relay=executor._relay)
    group.spawn()
    group.send_one(1, {"cmd": "consume", "manifest": args.manifest})
    ours = consume(executor, args.root, args.manifest)
    ours_files = ours.pop("files")
    theirs = group.reply_one(1, "consume")
    assert theirs["ok"] and theirs["rank"] == 1 and not theirs["store"]
    assert ours["header"] == theirs["header"] and ours["layouts"] == theirs["layouts"]
    assert ours["config"] == theirs["config"] and ours["assets"] == theirs["assets"]
    assert theirs["refused"] == ours["refused"]
    for _, _, descriptor in ours_files:
        os.close(descriptor)
    group.send_one(1, {"cmd": "hold"})
    held = group.reply_one(1, "hold")
    (args.root / "rank-facts.json").write_text(json.dumps({"leader": ours, "follower": theirs,
                                                        "follower_pid": held["pid"]}))
    # Deliberately bypass normal RankGroup teardown, modelling abrupt leader death.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--rank-fd", type=int)
    parser.add_argument("--leader")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    assert tensorfs.STORE is False
    follower(args) if args.rank > 0 else leader(args)
