"""The R1 verdict from real report inputs: a run directory with results, a manifest and decoded images."""
import importlib.util
import json
import random
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location("gate", Path(__file__).parents[1] / "scripts/gate/gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def image(path: Path, seed: int, noise: int = 0) -> str:
    rng = random.Random(seed)
    base = [rng.randrange(256) for _ in range(64 * 64 * 3)]
    jitter = random.Random(seed + 1)
    Image.frombytes("RGB", (64, 64), bytes(min(255, max(0, v + jitter.randint(-noise, noise))) for v in base)).save(path)
    return str(path)


def run(tmp_path: Path, rust_disk=2 << 30, cold_s=23.0, failed=False, noise=0) -> dict:
    cells = {"3gib-grouped": {"budget": "3GiB", "requests": [{"model": "sdxl", "input": {"prompt": "p", "seed": 1}}] * 2},
             "cold-sdxl": {"cold": True, "requests": [{"model": "sdxl", "input": {"prompt": "c", "seed": 9}}]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"cells": cells, "r1": {"baseline": {"cold-sdxl": 22.2}}}))
    req = lambda seed, path, ok=True: {"prompt": "p", "seed": seed, "ok": ok, "submit": 0.0, "done": cold_s, "images": [{"path": path}]}
    rows = [
        {"event": "cell", "arm": "comfy", "cell": "3gib-grouped", "ok": True, "disk_read_bytes": 2 << 30, "total_s": 130, "requests": []},
        {"event": "cell", "arm": "rust", "cell": "3gib-grouped", "ok": not failed, "disk_read_bytes": rust_disk, "facts": "xid-events=0",
         "requests": [req(1, image(tmp_path / "a.png", 1, noise)), req(2, image(tmp_path / "b.png", 2), ok=not failed)]},
        {"event": "cell", "arm": "rust", "cell": "control", "ok": True, "disk_read_bytes": 0,
         "requests": [req(1, image(tmp_path / "ca.png", 1)), req(2, image(tmp_path / "cb.png", 2))]},
        {"event": "cell", "arm": "comfy", "cell": "cold-sdxl", "ok": True, "disk_read_bytes": 1 << 30, "total_s": 14, "startup_s": 8, "requests": []},
        {"event": "cell", "arm": "rust", "cell": "cold-sdxl", "ok": True, "disk_read_bytes": 1 << 30, "facts": "xid-events=0",
         "requests": [{"prompt": "c", "seed": 9, "ok": True, "submit": 100.0, "done": 100.0 + cold_s, "images": []}]},
    ]
    return gate.r1(tmp_path, rows)


def test_a_clean_run_passes_and_reports_cold_start_beside_comfyui(tmp_path):
    v = run(tmp_path)
    assert v["pass"] and not v["review"]
    cold = next(c for c in v["cells"] if c["cell"] == "cold-sdxl")
    assert cold["cold_s"] == 23.0 and cold["reference_cold_s"] == 22.0 and cold["cold_ratio"] == 1.045 and cold["previous_cold_s"] == 22.2


def test_a_failed_request_fails_the_verdict(tmp_path):
    assert not run(tmp_path, failed=True)["pass"]


def test_disk_reads_over_one_and_a_half_times_comfyui_fail(tmp_path):
    assert not run(tmp_path, rust_disk=4 << 30)["pass"]


def test_a_cold_start_over_ten_percent_slower_than_comfyui_on_the_same_pod_fails(tmp_path):
    assert not run(tmp_path, cold_s=25.0)["pass"]


def test_low_psnr_is_flagged_for_review_not_failed(tmp_path):
    v = run(tmp_path, noise=60)
    assert v["pass"] and len(v["review"]) == 1 and v["review"][0]["psnr_db"] < 30
    assert next(c for c in v["cells"] if c["cell"] == "3gib-grouped")["psnr_db"][1] == float("inf")
