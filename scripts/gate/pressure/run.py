"""Run six literal original requests through the ordinary default-home Cozy CLI.

Physical observations surround every request. Disconnects leave the durable run,
holder, keeper and rental under their existing owner; this driver never ends them.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import threading
import time
import uuid

from PIL import Image


def pixels(path):
    with Image.open(path) as image:
        image.load()
        if image.size != (1024, 1024):
            raise ValueError("authored output geometry changed")
        data = image.convert("RGB").tobytes()
    return {
        "path": str(path.resolve()), "width": 1024, "height": 1024,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pixel_sha256": hashlib.sha256(data).hexdigest(),
    }, data


def quality(reference, candidate):
    ref, left = pixels(reference)
    actual, right = pixels(candidate)
    mse = sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left)
    psnr = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    return {"reference": ref, "candidate": actual, "exact": mse == 0,
            "psnr_db": psnr, "ok": mse == 0 or psnr >= 35,
            "limit": "historical single reference; stricter matched controls remain separate"}


def probe(spec, receipt, out):
    done = subprocess.run(spec["probe_command"], capture_output=True, text=True, check=True)
    actual = json.loads(done.stdout)
    for key, expected in {
        "rental_id": receipt["rental_id"], "operation": receipt["operation"],
        "gpu_uuid": receipt["gpu_uuid"], "holder_pid": receipt["pid"],
        "holder_start_ticks": receipt["start_ticks"],
    }.items():
        if actual.get(key) != expected:
            raise ValueError("physical holder changed at " + key)
    out.write_text(json.dumps(actual, indent=2) + "\n")


def monitor(spec, receipt, root, stopped, errors):
    with (root / "physical-samples.jsonl").open("a") as stream:
        while not stopped.is_set():
            try:
                done = subprocess.run(spec["probe_command"], capture_output=True, text=True, check=True)
                actual = json.loads(done.stdout)
                if actual.get("operation") != receipt["operation"]:
                    raise ValueError("sampling observed a different holder operation")
                stream.write(json.dumps(actual) + "\n")
                stream.flush()
            except BaseException as error:
                errors.append(repr(error))
                stream.write(json.dumps({"probe_error": repr(error), "unix_ns": time.time_ns()}) + "\n")
                stream.flush()
                return
            stopped.wait(2)


def final_steps(path, steps):
    positions = set()
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        progress = event.get("payload", {})
        if not isinstance(progress, dict):
            continue
        progress = progress.get("payload", progress)
        if not isinstance(progress, dict):
            continue
        if progress.get("stage") == "denoise" and progress.get("total") == steps:
            position = progress.get("position")
            if isinstance(position, int):
                positions.add(position)
    return sorted(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if "COZY_HOME" in os.environ:
        raise ValueError("qualification requires the ordinary default-home CLI")
    spec = json.loads(args.spec.read_text())
    source_path = pathlib.Path(spec["cells"])
    source = json.loads(source_path.read_text())
    requests = source["cells"][args.cell]["requests"]
    if len(requests) != 6:
        raise ValueError("each authored cell contains exactly six unchanged requests")
    receipt_path = pathlib.Path(spec["receipt"])
    receipt = json.loads(receipt_path.read_text())
    if receipt["rental"] != spec["rental"] or not receipt["fresh_before_first_model_prepare"]:
        raise ValueError("fresh rental pressure identity is not proved")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "spec.json").write_bytes(args.spec.read_bytes())
    (args.out / "holder.json").write_bytes(receipt_path.read_bytes())
    (args.out / "cells.sha256").write_text(hashlib.sha256(source_path.read_bytes()).hexdigest() + "\n")
    results = []
    for index, request in enumerate(requests):
        root = args.out / str(index)
        root.mkdir()
        payload = json.dumps(request["input"], sort_keys=True, separators=(",", ":")).encode()
        (root / "input.json").write_bytes(payload)
        reference = json.loads((pathlib.Path(spec["references"]) / args.cell / str(index) / "result.json").read_text())
        input_sha256 = hashlib.sha256(payload).hexdigest()
        if reference["input_sha256"] != input_sha256 or reference["model"] != request["model"] or not reference["ok"]:
            raise ValueError("historical control is not this same literal request")
        probe(spec, receipt, root / "physical-before.json")
        command = [spec["cli"], "run", source["targets"][request["model"]],
                   *source["target_args"][request["model"]], "--rental=" + spec["rental"],
                   "--tensorhub=" + spec["tensorhub"], "--input", str(root / "input.json"),
                   "--idempotency-key=pressure-" + uuid.uuid4().hex,
                   "--await", "--json", "--out", str(root / "artifacts")]
        (root / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        stopped = threading.Event()
        errors = []
        observer = threading.Thread(target=monitor, args=(spec, receipt, root, stopped, errors), daemon=True)
        observer.start()
        try:
            with (root / "stdout.json").open("wb") as stdout, (root / "events.jsonl").open("wb") as stderr:
                started = time.perf_counter_ns()
                done = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            paths = sorted(path for path in (root / "artifacts").rglob("*") if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            for path in paths:
                with path.open("rb") as artifact:
                    os.fsync(artifact.fileno())
            saved = time.perf_counter_ns()
        finally:
            stopped.set()
            observer.join()
        probe(spec, receipt, root / "physical-after.json")
        try:
            reply = json.loads((root / "stdout.json").read_text())
            state = reply.get("status", reply.get("state", (reply.get("run") or {}).get("state")))
        except (ValueError, AttributeError):
            reply, state = {}, None
        images = [pixels(path)[0] for path in paths]
        q = quality(pathlib.Path(reference["images"][0]["path"]), paths[0]) if len(paths) == 1 else None
        positions = final_steps(root / "events.jsonl", request["input"]["steps"])
        result = {
            "index": index, "model": request["model"], "returncode": done.returncode,
            "state": state, "wall_s": (saved - started) / 1e9,
            "timing_boundary": "first_submit_to_closed_fsynced_controller_output",
            "input_sha256": input_sha256, "images": images, "quality": q,
            "denoise_positions": positions, "final_authored_step_observed": request["input"]["steps"] in positions,
            "all_denoise_events_observed": set(range(1, request["input"]["steps"] + 1)).issubset(positions),
            "cached_result": bool(reply.get("memo") or reply.get("cached_result")),
            "physical_probe_errors": errors,
        }
        result["ok"] = (done.returncode == 0 and state in ("completed", "succeeded") and
                        len(paths) == 1 and q["ok"] and result["final_authored_step_observed"] and
                        not result["cached_result"] and not errors)
        results.append(result)
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not result["ok"]:
            break
    summary = {"cell": args.cell, "rental": spec["rental"], "rental_id": receipt["rental_id"],
               "completed": sum(row["ok"] for row in results), "requested": 6, "results": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["completed"] == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
