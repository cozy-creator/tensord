"""CPU harness checks with actual host/CLI command failures; no inference qualification."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/gate/gate.py"
spec = importlib.util.spec_from_file_location("gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_setup_failure_exits_before_sampling_or_timed_commands(tmp_path):
    marker = tmp_path / "timed"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"salt": "setup", "work_dir": str(tmp_path / "host"),
                                    "setup": "exit 7", "on_start": f"touch {marker}", "order": []}))
    done = subprocess.run([sys.executable, str(SCRIPT), "run", str(manifest), str(tmp_path / "out")],
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert not marker.exists()
    assert not list((tmp_path / "host").glob("samples-*"))


def test_failed_anima_prime_never_submits_a_timed_request(tmp_path):
    calls = tmp_path / "calls"
    cli = tmp_path / "cli"
    cli.write_text(f"#!/bin/sh\necho \"$*\" >> {calls}\necho '{{\"error\":{{\"message\":\"anima_optimization_block_shape\"}}}}'\nexit 7\n")
    cli.chmod(0o755)
    g = gate.Gate.__new__(gate.Gate)
    g.m = {"salt": "prime", "targets": {"anima": "paul/anima/generate", "sdxl": "paul/sdxl/generate"},
           "shapes": {"anima": [64, 64]}, "continue_on_failure": True}
    g.out, g.results, g.index, g.offset = tmp_path, tmp_path / "results.jsonl", 0, 0
    payload = {"prompt": "same", "seed": 1}
    with pytest.raises(RuntimeError, match="prim"):
        g.cozy_cell("rust", "warm", {"cli": str(cli), "run_args": []},
                    [{"model": "sdxl", "input": payload}], None, ready=False,
                    prime=[{"model": "anima", "input": payload}])
    assert len(calls.read_text().splitlines()) == 1
    assert "paul/sdxl" not in calls.read_text()
    assert "anima_optimization_block_shape" in g.results.read_text()


def test_collecting_failed_cells_still_returns_a_failed_run(tmp_path, monkeypatch):
    # The observation helper is CPU-only. The engine command is a real shell exit.
    monkeypatch.setattr(gate, "POD_HELPER", '''import json,sys,time
if sys.argv[1] == "now":
 print(json.dumps({"t":time.time(),"gpu":{"temp_c":20},"load":{"load1":0},"executors":[],"cg":{"read_bytes":0,"path":"cpu"}}))
''')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"salt": "cell", "work_dir": str(tmp_path / "host"),
                                    "start_temp_c": 60, "continue_on_failure": True, "order": [],
                                    "arms": {"comfy": {"root": "echo 1", "keeps_root": True, "command": "exit 7"}},
                                    "cells": {"one": {"requests": []}}, "cell_order": [["comfy", "one"]]}))
    with pytest.raises(RuntimeError, match="fail"):
        gate.Gate(manifest, tmp_path / "out").run()
    rows = [json.loads(s) for s in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert next(r for r in rows if r.get("event") == "cell")["ok"] is False


def test_report_prints_failed_verdict_and_exits_nonzero_for_a_failed_preflight(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "samples.jsonl").write_text("")
    (tmp_path / "results.jsonl").write_text(json.dumps({"event": "preflight", "ok": False, "error": "exit 7"}) + "\n")
    done = subprocess.run([sys.executable, str(SCRIPT), "report", str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 1
    assert json.loads(done.stdout)["pass"] is False
