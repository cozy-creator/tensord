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


def native_invokes(spec):
    if "native_invoke_probe_command" not in spec:
        return []
    done = subprocess.run(spec["native_invoke_probe_command"], capture_output=True, text=True, check=True)
    return [json.loads(line) for line in done.stdout.splitlines() if line]


def source_backed_freshness(spec, model, payload, reply, positions, before, after):
    """A missing final observation needs actual source review and a fresh native receipt."""
    path = spec.get("sampling_source_proof", {}).get(model)
    if not path or len(after) != len(before) + 1 or after[:len(before)] != before:
        return None
    proof = json.loads(pathlib.Path(path).read_text())
    steps = payload["steps"]
    shape = f"height=1024,pixels=1048576,steps={steps},width=1024"
    result = reply.get("result", {})
    native = after[-1]
    if (proof.get("reviewed_full_scheduler_loop") is not True or
        not proof.get("actual_model_source_sha256") or not proof.get("actual_sdk_source_sha256") or
        proof.get("authored_steps") != steps or result.get("steps") != steps or
        result.get("width") != 1024 or result.get("height") != 1024 or
        native.get("metrics", {}).get("shape_cell") != shape or
        native.get("metrics", {}).get("handler_ms", 0) <= 0 or
        not set(range(1, steps)).issubset(positions)):
        return None
    return {"evidence_class": "SOURCE_AND_GPU", "source_proof": proof,
            "new_native_invoke": native, "final_denoise_event_observed": steps in positions,
            "limit": "full loop from reviewed source and fresh native completion; final observer event missing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--start-index", type=int, choices=range(6), default=0)
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
        if index < args.start_index:
            continue
        root = args.out / str(index)
        root.mkdir()
        payload = json.dumps(request["input"], sort_keys=True, separators=(",", ":")).encode()
        (root / "input.json").write_bytes(payload)
        reference = json.loads((pathlib.Path(spec["references"]) / args.cell / str(index) / "result.json").read_text())
        input_sha256 = hashlib.sha256(payload).hexdigest()
        if reference["input_sha256"] != input_sha256 or reference["model"] != request["model"] or not reference["ok"]:
            raise ValueError("historical control is not this same literal request")
        probe(spec, receipt, root / "physical-before.json")
        before_invokes = native_invokes(spec)
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
        after_invokes = native_invokes(spec)
        freshness = source_backed_freshness(
            spec, request["model"], request["input"], reply, positions,
            before_invokes, after_invokes,
        )
        result = {
            "index": index, "model": request["model"], "returncode": done.returncode,
            "state": state, "wall_s": (saved - started) / 1e9,
            "timing_boundary": "first_submit_to_closed_fsynced_controller_output",
            "input_sha256": input_sha256, "images": images, "quality": q,
            "denoise_positions": positions, "final_authored_step_observed": request["input"]["steps"] in positions,
            "all_denoise_events_observed": set(range(1, request["input"]["steps"] + 1)).issubset(positions),
            "cached_result": bool(reply.get("memo") or reply.get("cached_result")),
            "physical_probe_errors": errors,
            "source_backed_sampling": freshness,
        }
        result["ok"] = (done.returncode == 0 and state in ("completed", "succeeded") and
                        len(paths) == 1 and q["ok"] and
                        (result["final_authored_step_observed"] or freshness is not None) and
                        not result["cached_result"] and not errors)
        results.append(result)
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not result["ok"]:
            break
    summary = {"cell": args.cell, "rental": spec["rental"], "rental_id": receipt["rental_id"],
               "completed": sum(row["ok"] for row in results),
               "requested": 6 - args.start_index, "start_index": args.start_index, "results": results}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["completed"] == summary["requested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
