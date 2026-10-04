"""CPU refusal-path proof only; never an actual physical GPU qualification."""

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/gate/pressure" / (name + ".py"))
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)
    return actual


probe = module("probe")
run = module("run")


class PhysicalRefusals(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.receipt = pathlib.Path(self.directory.name) / "holder.json"
        self.pid = os.getpid()
        start = int(pathlib.Path(f"/proc/{self.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
        self.value = {"pid": self.pid, "start_ticks": start, "gpu_uuid": "GPU-owned",
                      "rental": "owned", "rental_id": "pr-owned", "operation": "authored",
                      "driver": "fixture", "total_bytes": 8 << 30,
                      "target_free_bytes": 3 << 30, "held_bytes": 5 << 30}
        self.receipt.write_text(json.dumps(self.value))
        self.free_mib = 3072
        self.present = True

    def tearDown(self):
        self.directory.cleanup()

    def smi(self, args, **_):
        if "--query-gpu=uuid,memory.free,memory.used,utilization.gpu" in args:
            return f"GPU-owned, {self.free_mib}, 5120, 0\n"
        return f"{self.pid}, 5120\n" if self.present else ""

    def capture(self):
        with patch.object(probe.subprocess, "check_output", side_effect=self.smi):
            return probe.capture(self.receipt)

    def test_keeps_exact_holder_birth_and_ceiling(self):
        actual = self.capture()
        self.assertEqual(actual["holder_start_ticks"], self.value["start_ticks"])
        self.assertEqual(actual["free_bytes"], 3 << 30)
        self.assertIn(str(self.pid), actual["process_memory"])

    def test_recycled_pid_is_not_a_live_original_holder(self):
        self.value["start_ticks"] -= 1
        self.receipt.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(ValueError, "original physical holder"):
            self.capture()

    def test_relaxed_physical_pool_is_rejected(self):
        self.free_mib = 3075
        with self.assertRaisesRegex(ValueError, "authored ceiling"):
            self.capture()

    def test_missing_physical_holder_is_rejected(self):
        self.present = False
        with self.assertRaisesRegex(ValueError, "retain the authored ballast"):
            self.capture()

    def test_explicit_release_cannot_be_qualified(self):
        self.receipt.with_suffix(".released.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "already released"):
            self.capture()


class SamplingEvidence(unittest.TestCase):
    def test_early_observer_gap_retained_without_inventing_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            events = pathlib.Path(directory) / "events.jsonl"
            events.write_text("not-json\n" + json.dumps({"payload": {"payload": {"stage": "denoise", "position": 20, "total": 20}}}) + "\n")
            self.assertEqual(run.final_steps(events, 20), [20])
            self.assertEqual(run.final_steps(events, 30), [])

    def test_missing_final_event_needs_reviewed_source_and_one_new_native_invoke(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = pathlib.Path(directory) / "proof.json"
            proof.write_text(json.dumps({"reviewed_full_scheduler_loop": True,
                "actual_model_source_sha256": "owned-model-source",
                "actual_sdk_source_sha256": "owned-sdk-source", "authored_steps": 30}))
            spec = {"sampling_source_proof": {"anima": str(proof)}}
            reply = {"result": {"steps": 30, "width": 1024, "height": 1024}}
            native = {"metrics": {"shape_cell": "height=1024,pixels=1048576,steps=30,width=1024",
                                  "handler_ms": 43000}}
            positions = list(range(1, 30))
            self.assertIsNotNone(run.source_backed_freshness(spec, "anima", {"steps": 30},
                reply, positions, [], [native]))
            self.assertIsNone(run.source_backed_freshness(spec, "anima", {"steps": 30},
                reply, positions, [native], [native]))
            self.assertIsNone(run.source_backed_freshness(spec, "anima", {"steps": 30},
                reply, positions[:-1], [], [native]))


if __name__ == "__main__":
    unittest.main()
