"""Matched ordinary default-home Creator CLI runs; --plan is strictly CPU-only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


@dataclass(frozen=True)
class Arm:
    name: str
    cli: Path
    selector: Path
    service_pids: tuple[int, ...]
    generation_python: Path
    serving_uid: int | None
    initial_pinned_bytes: int
    policy_evidence: Path
    journal: Path | None

    @classmethod
    def read(cls, value: dict) -> Arm:
        return cls(value["name"], Path(value["cli"]), Path(value["selector"]),
                   tuple(value["service_pids"]), Path(value["generation_python"]),
                   value.get("serving_uid"), value["initial_pinned_bytes"],
                   Path(value["policy_evidence"]),
                   Path(value["journal"]) if value.get("journal") else None)


def load(path: Path) -> tuple[dict, dict[str, Arm]]:
    manifest = json.loads(path.read_text())
    assert manifest["format"] == "cozy.ordinary.switch-benchmark/1"
    assert manifest["transport"] == "ordinary_default_home_creator_cli"
    assert manifest["memoization"] is False
    assert manifest["gpu_reserve_bytes"] > 0
    arms = {value["name"]: Arm.read(value) for value in manifest["arms"]}
    assert len(arms) == 2 and len({arm.initial_pinned_bytes for arm in arms.values()}) == 1
    assert all(arm.initial_pinned_bytes == 4 * 1024**3 for arm in arms.values())
    for model, value in manifest["models"].items():
        assert model in ("sdxl", "anima") and value["target"] == f"paul/{model}/generate"
        assert value["shape"] == ([1024, 1024] if model == "sdxl" else [1536, 1536])
        for payload in value["payloads"]:
            assert payload["steps"] == (20 if model == "sdxl" else 30)
            assert payload["megapixels"] == (1 if model == "sdxl" else 2)
            assert payload["guidance"] == (7 if model == "sdxl" else 4.5)
            if model == "anima":
                assert payload["first_block_cache"] == 0
                assert (payload["cfg_interval_start"], payload["cfg_interval_stop"]) == (.15, .7)
            else:
                assert payload["hidiffusion"] is False
    assert all(turn["arm"] in arms for turn in manifest["sequence"])
    return manifest, arms


def topology(roots: tuple[int, ...]) -> list[dict]:
    """Boundary-only process evidence; no per-source/PSS scan in the timed hot path."""
    processes = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            processes[int(path.name)] = (int(tail[1]), int(tail[19]), path)
        except (OSError, ValueError, IndexError):
            continue
    selected = set(roots)
    while True:
        children = {pid for pid, (parent, _, _) in processes.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    result = []
    for pid in sorted(selected):
        if pid not in processes:
            continue
        parent, birth, path = processes[pid]
        try:
            uid = next(line for line in (path / "status").read_text().splitlines() if line.startswith("Uid:"))
            memory = {line.split(":")[0]: line.split(":")[1].strip()
                      for line in (path / "smaps_rollup").read_text().splitlines() if ":" in line}
            result.append({"pid": pid, "parent": parent, "start_ticks": birth, "uid": uid,
                           "argv": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                           "memory": memory})
        except OSError as error:
            result.append({"pid": pid, "start_ticks": birth, "unavailable": str(error)})
    return result


def cli(arm: Arm, root: Path, name: str, args: list[str]) -> tuple[object, float]:
    started = time.perf_counter_ns()
    process = subprocess.run([str(arm.cli), *args], capture_output=True, text=True)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    (root / f"{name}.stdout").write_text(process.stdout)
    (root / f"{name}.stderr").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"ordinary CLI {name} exited {process.returncode}; preserve logs")
    return json.loads(process.stdout), elapsed_ms


def verify(record: dict, root: Path, shape: list[int]) -> list[dict]:
    from PIL import Image, ImageStat
    outputs = []
    for index, item in enumerate(record["output"]):
        if item["media_type"] != "image/webp":
            continue
        source = Path(item["path"])
        data = source.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        assert digest == item["sha256"] and len(data) == item["length"]
        target = root / f"image-{index}.webp"
        shutil.copyfile(source, target)
        with Image.open(target) as image:
            image.load()
            assert list(image.size) == shape and image.format == "WEBP"
            variance = ImageStat.Stat(image.convert("RGB")).var
            assert any(variance)
        outputs.append({"path": str(target), "digest": digest, "bytes": len(data),
                        "shape": shape, "variance": variance})
    assert outputs, "no completed native image output; not a successful inference gate"
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    manifest, arms = load(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.plan:
        print(json.dumps({"plan_valid": True, "requests": len(manifest["sequence"]),
                          "no_gpu_or_cli_calls": True, "needs_policy_and_admission_gates": True}))
        return
    for arm in arms.values():
        assert arm.serving_uid is not None, "measure actual old/new executor UID first"
        policy = json.loads(arm.policy_evidence.read_text())
        assert policy["initial_pinned_budget_bytes"] == arm.initial_pinned_bytes
        assert policy["applied_before_model_load"] is True
        assert policy["context_and_working_memory_accounted"] is True
        assert policy["authorized_device_limit_bytes"] + manifest["gpu_reserve_bytes"] <= policy["fresh_physical_free_bytes"]
        assert policy["scope"] == "owned_benchmark_preflight", "test reserve is not a production request floor"
    observer = None
    results = []
    try:
        with (args.output / "gpu.csv").open("w") as log:
            observer = subprocess.Popen([
                "nvidia-smi", "--id=" + manifest["gpu_uuid"],
                "--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv", "--loop-ms=1000"], stdout=log, stderr=subprocess.DEVNULL)
            for index, turn in enumerate(manifest["sequence"]):
                arm = arms[turn["arm"]]
                model = manifest["models"][turn["model"]]
                root = args.output / f"{index:02}-{arm.name}-{turn['model']}"
                root.mkdir()
                payload = root / "input.json"
                payload.write_text(json.dumps(model["payloads"][turn["payload"]]) + "\n")
                before = topology(arm.service_pids)
                started = time.perf_counter_ns()
                # No HOME override, private RPC, custom credential or fallback selector.
                submission, submit_ms = cli(arm, root, "submit", [
                    "run", model["target"], "--machine-endpoint-file", str(arm.selector),
                    "--input", str(payload), "--json", "--idempotency-key",
                    f"{args.output.name}-{index}-{arm.name}"])
                reference = str(submission["run"])
                watch = subprocess.run([str(arm.cli), "run", "watch", reference, "--json"],
                                       capture_output=True, text=True)
                (root / "watch.stdout").write_text(watch.stdout)
                (root / "watch.stderr").write_text(watch.stderr)
                if watch.returncode:
                    raise RuntimeError("watch detached/failed; run may still be active; use ordinary show/watch, never automatic cancel")
                record, _ = cli(arm, root, "show", ["run", "show", reference, "--json"])
                assert record["status"] == "completed", "negative/preaccept request is not a timing sample"
                accepted = [event for event in record["events"] if event["type"] == "request.machine_accepted"]
                assert accepted and accepted[0]["payload"]["durable"] is True
                finished = time.perf_counter_ns()
                images = verify(record, root, model["shape"])
                fact = {"arm": arm.name, "model": turn["model"], "run": reference,
                        "request_id": record["request_id"], "submit_ms": submit_ms,
                        "submit_to_collected_ms": (finished - started) / 1e6,
                        "submit_to_verified_ms": (time.perf_counter_ns() - started) / 1e6,
                        "before_processes": before, "after_processes": topology(arm.service_pids),
                        "receipt": accepted[0], "images": images,
                        "model_ready_ms": None, "source_fill_hash_cache_facts": None}
                if arm.journal:
                    with sqlite3.connect(arm.journal.as_uri() + "?mode=ro", uri=True) as db:
                        rows = db.execute("SELECT record FROM executions WHERE request_id=?",
                                          (record["request_id"],)).fetchall()
                        fact["machine_execution_records"] = [json.loads(row[0]) for row in rows]
                results.append(fact)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        if observer is not None:
            observer.terminate()  # Only this observation process, never durable work.
            observer.wait()
    print(json.dumps({"ordinary_completed_requests": len(results),
                      "model_ready_and_source_phase_gate": "requires exact machine profile evidence"}))


if __name__ == "__main__":
    main()
