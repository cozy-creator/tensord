"""Actual controller artifact/value checks; these CPU tests do not execute inference."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/gate/matched"))
from adapters import build_graph
from evidence import compare, digest, tensor_inventory, validate_quality, verify_parity


def image(path, offset=0):
    pixels = [
        ((i + offset) % 256, (i * 3 + offset) % 256, (i * 7 + offset) % 256)
        for i in range(32 * 32)
    ]
    value = Image.new("RGB", (32, 32))
    value.putdata(pixels)
    value.save(path)


def test_real_tensor_bytes_not_file_names_establish_parity(tmp_path):
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode()
    p = tmp_path / "tensor.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1, 2))
    rows = tensor_inventory(p, "denoiser")
    proof = {
        "tensors": {"cozy": rows, "comfy": rows},
        "sigmas": {"cozy": [1, 0], "comfy": [1, 0]},
        "prediction": "epsilon",
        "source_equations": ["Euler"],
        "conditioning_evidence": ["matched tokenizer source"],
    }
    assert verify_parity(proof) == verify_parity(proof)
    changed = tmp_path / "different-name.safetensors"
    changed.write_bytes(p.read_bytes()[:-4] + struct.pack("<f", 3))
    proof["tensors"]["comfy"] = tensor_inventory(changed, "denoiser")
    with pytest.raises(ValueError, match="tensor"):
        verify_parity(proof)


def test_actual_schedule_mismatch_cannot_be_declared_matched():
    proof = {
        "tensors": {"cozy": [{"value": "a"}], "comfy": [{"value": "a"}]},
        "sigmas": {"cozy": [1, 0.01, 0], "comfy": [1, 0.09, 0]},
        "prediction": "flow",
        "source_equations": ["Euler"],
        "conditioning_evidence": ["source"],
    }
    with pytest.raises(ValueError, match="scheduler grids differ"):
        verify_parity(proof)


def test_same_engine_controls_exact_pixels_and_provenance(tmp_path):
    refs = [tmp_path / str(i) for i in range(3)]
    for ref in refs:
        image(ref.with_suffix(".png"))
    refs = [str(p.with_suffix(".png")) for p in refs]
    candidate = tmp_path / "out.png"
    image(candidate)
    identity = {
        "engine": {"name": "cozy-machine", "commit": "actual"},
        "request_digest": digest({"seed": 1}),
        "hardware_key": "same",
    }
    control = [{"identity": identity, "artifacts": refs}]
    quality = validate_quality(
        [[str(candidate)]], control, {"method": "exact"}, identity
    )
    assert quality["ok"] and quality["reference_digest"]
    image(candidate, 1)
    assert not validate_quality(
        [[str(candidate)]], control, {"method": "exact"}, identity
    )["ok"]
    with pytest.raises(ValueError, match="same-engine"):
        validate_quality(
            [[str(candidate)]],
            control,
            {"method": "exact"},
            {**identity, "engine": {"name": "ComfyUI"}},
        )


def test_tolerance_requires_real_unconstrained_variance(tmp_path):
    refs = []
    for i, offset in enumerate((0, 0, 0)):
        p = tmp_path / f"ref{i}.png"
        image(p, offset)
        refs.append(str(p))
    candidate = tmp_path / "candidate.png"
    image(candidate)
    identity = {"engine": "same", "request": "same"}
    controls = [{"identity": identity, "artifacts": refs}]
    policy = {"method": "declared_tolerance", "minimum_psnr_db": 35}
    assert validate_quality([[str(candidate)]], controls, policy, identity)["ok"]
    image(Path(refs[1]), 64)
    assert not validate_quality([[str(candidate)]], controls, policy, identity)["ok"]
    assert compare(Path(refs[0]), Path(refs[1]))["psnr_db"] < 35


def test_graph_preserves_literal_request_and_verified_scheduler():
    payload = {
        "prompt": "literal",
        "negative_prompt": "negative",
        "seed": 123,
        "steps": 30,
        "guidance": 4.5,
    }
    graph = build_graph(
        {
            "model_files": {
                "anima": {"denoiser": "a", "vae": "v", "text_encoder": "t"}
            },
            "scheduler": {"anima": "normal"},
        },
        {"model": "anima", "input": payload},
    )
    assert graph["positive"]["inputs"]["text"] == "literal"
    assert (
        graph["sample"]["inputs"]["seed"] == 123
        and graph["sample"]["inputs"]["steps"] == 30
    )
    assert graph["sample"]["inputs"]["scheduler"] == "normal"
