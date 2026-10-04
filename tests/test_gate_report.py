"""The gate must reject failed, slower, incomplete or unqualified measurements."""
from __future__ import annotations

import importlib.util
from pathlib import Path

source = Path(__file__).parents[1] / "scripts/gate/gate.py"
spec = importlib.util.spec_from_file_location("gate", source)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def measurements(ratio=0.9):
    rows = []
    for pair in range(3):
        for arm, time in (("comfy", 10), ("rust", 10 * ratio)):
            rows.append({"event": "cell", "cell": "warm", "arm": arm, "pair": str(pair),
                         "total_s": time, "ok": True, "request_digest": f"sha256:{pair:064x}",
                         "hardware_key": "same-card-and-driver", "timing_boundary": "submit_to_saved_output",
                         "output_location": "controller", "quality": {"ok": True, "method": "exact",
                                                                         "reference_digest": "sha256:reference"}})
    return rows, {"comparison": {"reference": "comfy", "candidate": "rust", "cells": ["warm"], "min_pairs": 3}}


def test_complete_faster_validated_pairs_pass():
    rows, plan = measurements()
    result = gate.comparison(rows, plan)
    assert result["pass"] and result["timing_win"]
    assert result["cells"][0]["ratio_ci95"] == [0.9, 0.9]


def test_memory_savings_do_not_turn_slower_inference_into_a_win():
    rows, plan = measurements(1.01)
    for row in rows:
        row["host_peak_gib"] = 1 if row["arm"] == "rust" else 4
    assert not gate.comparison(rows, plan)["pass"]


def test_failed_cell_cannot_be_omitted_from_the_verdict():
    rows, plan = measurements()
    rows.append({"event": "cell", "cell": "1gib-grouped", "arm": "rust", "ok": False})
    assert not gate.comparison(rows, plan)["pass"]
    assert gate.comparison(rows, plan)["status"] == "failed"


def test_missing_duplicate_or_unpaired_rows_are_inconclusive():
    rows, plan = measurements()
    for incomplete in (rows[:-1], rows + [rows[0]], [r for r in rows if r["pair"] != "2"]):
        assert not gate.comparison(incomplete, plan)["pass"]
    assert not gate.comparison(rows, {})["pass"]


def test_different_inputs_hardware_or_saved_output_boundary_cannot_pass():
    for field in ("request_digest", "hardware_key", "output_location", "timing_boundary"):
        rows, plan = measurements()
        rows[1][field] = "different"
        assert not gate.comparison(rows, plan)["pass"]


def test_smoke_checks_do_not_qualify_inference_outputs():
    rows, plan = measurements()
    for row in rows:
        row["quality"] = {"ok": True, "method": "shape_and_nonflat"}
    result = gate.comparison(rows, plan)
    assert result["timing_win"] and not result["pass"]


def test_nonfinite_timings_and_uncertain_ratios_cannot_pass():
    rows, plan = measurements()
    rows[1]["total_s"] = float("nan")
    assert not gate.comparison(rows, plan)["pass"]
    rows, plan = measurements()
    for row, ratio in zip(rows[1::2], (0.8, 0.9, 1.1), strict=True):
        row["total_s"] = 10 * ratio
    assert not gate.comparison(rows, plan)["pass"]
