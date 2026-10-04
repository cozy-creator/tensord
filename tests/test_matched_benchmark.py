"""Actual controller artifact/value checks; these CPU tests do not execute inference."""

from __future__ import annotations

import ast
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/gate/matched"))
from adapters import build_graph, cozy
from evidence import (
    compare,
    digest,
    native_inventory,
    tensor_inventory,
    validate_quality,
    verify_conditioning,
    verify_parity,
)


def parity(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("def generate(): pass\n")
    artifact = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "symbol": "generate",
    }
    initialization = {
        "shape": [1, 4, 128, 128],
        "distribution": "normal",
        "noise_dtype": "float32",
        "state_dtype": "float32",
        "noise_scale": 1.0,
        "latent_zero": True,
    }
    mode = {
        field: "float32"
        for field in (
            "text_encoder_compute_dtype",
            "conditioning_dtype",
            "denoiser_input_dtype",
            "denoiser_output_dtype",
            "timestep_dtype",
            "cfg_dtype",
            "sampler_compute_dtype",
            "latent_state_dtype",
            "vae_dtype",
            "latent_inverse_scale_dtype",
            "pixel_normalization_dtype",
            "image_quantization_dtype",
            "attention_precision",
            "math_flags",
        )
    }
    mode["decode_geometry"] = {"tiling": False}
    return {
        "tensors": {
            engine: [
                {
                    "name": "denoiser/w",
                    "dtype": "F32",
                    "shape": [1],
                    "bytes": 4,
                    "sha256": "0" * 64,
                }
            ]
            for engine in ("cozy", "comfy")
        },
        "component_tensors": {engine: {"denoiser": 1} for engine in ("cozy", "comfy")},
        "sigmas": {"cozy": [1, 0], "comfy": [1, 0]},
        "prediction": "epsilon",
        "source_equations": ["Euler"],
        "conditioning_evidence": {"fixture": {}},
        "initialization": {
            engine: dict(initialization) for engine in ("cozy", "comfy")
        },
        "numerical_mode": {engine: dict(mode) for engine in ("cozy", "comfy")},
        "source_artifacts": {engine: [artifact] for engine in ("cozy", "comfy")},
    }


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
    proof = parity(tmp_path)
    proof["tensors"] = {"cozy": rows, "comfy": rows}
    assert verify_parity(proof) == verify_parity(proof)
    changed = tmp_path / "different-name.safetensors"
    changed.write_bytes(p.read_bytes()[:-4] + struct.pack("<f", 3))
    proof["tensors"]["comfy"] = tensor_inventory(changed, "denoiser")
    with pytest.raises(ValueError, match="tensor"):
        verify_parity(proof)


def test_actual_schedule_mismatch_cannot_be_declared_matched(tmp_path):
    proof = parity(tmp_path)
    proof["sigmas"] = {"cozy": [1, 0.01, 0], "comfy": [1, 0.09, 0]}
    with pytest.raises(ValueError, match="scheduler grids differ"):
        verify_parity(proof)


def test_value_inventory_uses_bounded_native_reads_and_refuses_encoded_claims(tmp_path):
    import msgspec

    module = (
        Path(__file__).parents[1]
        / "scripts/gate/matched/native-source/native_source/metadata.py"
    )
    tree = ast.parse(module.read_text())
    selected = [
        node
        for node in tree.body
        if getattr(node, "name", None) in {"PartRow", "part_digest"}
        or isinstance(node, ast.Assign)
        and any(getattr(target, "id", "") == "_WIDTHS" for target in node.targets)
    ]
    scope = {"msgspec": msgspec, "math": __import__("math"), "hashlib": hashlib}
    exec(  # noqa: S102 - execute the production bounded-reader source in this CPU proof
        compile(ast.Module(body=selected, type_ignores=[]), str(module), "exec"), scope
    )
    data = bytes(range(256)) * 4097
    reads = []

    class Capability:
        def read_part_into(self, component, key, role, offset, into):
            reads.append((component, key, role, offset, len(into)))
            into[:] = data[offset : offset + len(into)]

    part = SimpleNamespace(shape=[len(data)], dtype="u8")
    result = scope["part_digest"](
        SimpleNamespace(raise_if_cancelled=lambda: None),
        Capability(),
        "vae",
        "w",
        "value",
        part,
    )
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert [row[3] for row in reads] == [0, 1 << 20]
    assert max(row[4] for row in reads) <= 1 << 20
    tensor = {
        "name": "vae/w",
        "dtype": "u8",
        "shape": [len(data)],
        "encoding": "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f",
        "parts": [msgspec.to_builtins(result)],
        "logical_value_sha256": result.sha256,
    }
    assert native_inventory({"tensors": [tensor]})[0]["sha256"] == result.sha256
    tensor["encoding"] = "quantized"
    with pytest.raises(ValueError, match="encoded tensor"):
        native_inventory({"tensors": [tensor]})


def test_matching_sigmas_do_not_hide_initial_noise_scale_or_precision(tmp_path):
    proof = parity(tmp_path)
    assert verify_parity(proof)
    proof["initialization"]["cozy"]["noise_scale"] = 11.073580517292482
    proof["initialization"]["comfy"]["noise_scale"] = 11.02833080291748
    with pytest.raises(ValueError, match="noise scales differ"):
        verify_parity(proof)
    proof["initialization"]["cozy"]["noise_scale"] = 11.02833080291748
    proof["numerical_mode"]["cozy"]["latent_state_dtype"] = "float16"
    with pytest.raises(ValueError, match="numerical modes differ"):
        verify_parity(proof)
    del proof["numerical_mode"]["comfy"]["cfg_dtype"]
    with pytest.raises(ValueError, match="mode evidence is incomplete"):
        verify_parity(proof)


def test_conditioning_is_for_the_literal_request_and_actual_token_ids():
    payload = {"prompt": "plain prompt", "seed": 123}
    measured = {
        "positive_token_ids": [12, 45],
        "negative_token_ids": [0],
        "attention_masks": [1, 1],
        "encoder_layers": [-2],
        "pooled_projection": "second_encoder",
        "microconditioning": [1024, 1024, 0, 0, 1024, 1024],
        "prompt_weight_policy": "literal",
        "tokenizer_asset_sha256": ["actual"],
    }
    proof = {
        "conditioning_evidence": {
            digest(payload): {engine: dict(measured) for engine in ("cozy", "comfy")}
        }
    }
    verify_conditioning(proof, payload)
    with pytest.raises(ValueError, match="incomplete"):
        verify_conditioning(proof, {**payload, "prompt": "new prompt"})
    proof["conditioning_evidence"][digest(payload)]["comfy"]["positive_token_ids"] = [
        12,
        44,
    ]
    with pytest.raises(ValueError, match="tokens"):
        verify_conditioning(proof, payload)


def test_unknown_cli_reply_is_retained_without_resubmitting(tmp_path):
    calls = tmp_path / "calls"
    cli = tmp_path / "fixture-cli"
    cli.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\np=Path({str(calls)!r})\np.write_text(p.read_text()+'call\\n' if p.exists() else 'call\\n')\nprint('lost reply')\n"
    )
    cli.chmod(0o755)
    request = {"model": "sdxl", "input": {"steps": 20}}
    rows, first, last = cozy(
        {
            "cli": str(cli),
            "targets": {"sdxl": "fixture"},
            "target_args": {"sdxl": []},
            "selector_args": [],
        },
        [request] * 6,
        tmp_path / "requests",
    )
    assert calls.read_text().splitlines() == ["call"]
    assert len(rows) == 6 and first <= last and not any(row["ok"] for row in rows)
    assert rows[0]["state"] == "unknown"
    assert all(row["state"] == "not_submitted" for row in rows[1:])
    assert len(list((tmp_path / "requests").glob("*/request-result.json"))) == 6


def test_failed_artifacts_and_missing_controls_still_write_the_complete_cell(
    tmp_path, monkeypatch
):
    import run as driver

    proof = parity(tmp_path)
    components = ("unet", "text_encoder", "text_encoder_2", "vae")
    tensor = proof["tensors"]["cozy"][0]
    proof["tensors"] = {
        engine: [{**tensor, "name": name + "/w"} for name in components]
        for engine in ("cozy", "comfy")
    }
    proof["component_tensors"] = {
        engine: {name: 1 for name in components} for engine in ("cozy", "comfy")
    }
    payload = {
        "prompt": "CPU fixture",
        "negative_prompt": "",
        "steps": 20,
        "guidance": 7,
        "seed": 1,
        "aspect_ratio": "1:1",
        "megapixels": 1,
        "hidiffusion": False,
    }
    inputs = {
        "positive_token_ids": [1],
        "negative_token_ids": [0],
        "attention_masks": [1],
        "encoder_layers": [-2],
        "pooled_projection": "encoder2",
        "microconditioning": [1024, 1024, 0, 0, 1024, 1024],
        "prompt_weight_policy": "literal",
        "tokenizer_asset_sha256": ["fixture"],
    }
    proof["conditioning_evidence"] = {
        digest(payload): {engine: inputs for engine in ("cozy", "comfy")}
    }
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps({"sdxl": proof}))
    manifest = {
        "arms": {
            "cozy": {
                "name": "cozy-machine",
                "cli": "/deliberately-absent-cpu-fixture-cli",
                "targets": {"sdxl": "fixture"},
                "target_args": {"sdxl": []},
                "selector_args": [],
                "commit_probe": [],
            }
        },
        "cells": {
            "degraded": {
                "kind": "degraded",
                "budget": "1GiB",
                "requests": [{"model": "sdxl", "input": payload}] * 6,
            }
        },
        "parity": str(proof_path),
        "hardware_probe": [],
        "controls": {"cozy": {"degraded": str(tmp_path / "missing-controls.json")}},
        "quality": {"method": "exact"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        driver,
        "probe",
        lambda command, path: (
            json.dumps({"revision": "cpu-fixture"})
            if path.name == "engine-probe.json"
            else json.dumps(
                {
                    "remote": True,
                    "gpu_uuid": "cpu-fixture",
                    "driver": "fixture",
                    "total_bytes": 1,
                }
            )
        ),
    )
    result = driver.run(
        manifest_path, tmp_path / "result", "cozy", "degraded", "0", False
    )
    saved = json.loads((tmp_path / "result/result.json").read_text())
    assert not result["ok"] and saved == result
    assert len(saved["requests"]) == 6 and not saved["quality"]["ok"]


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
