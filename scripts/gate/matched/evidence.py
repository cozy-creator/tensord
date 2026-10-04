"""Actual model tensor and same-engine pixel evidence for matched benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

from PIL import Image, ImageStat


def digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


def tensor_inventory(path: Path, component: str, prefix: str = ""):
    """Fingerprint actual safetensors payload bytes, never inferred from a filename."""
    rows = []
    with path.open("rb") as source:
        size = path.stat().st_size
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size > size - 8:
            raise ValueError("safetensors header leaves its file")
        header = json.loads(source.read(header_size))
        for name, row in sorted(header.items()):
            if name == "__metadata__":
                continue
            start, end = row["data_offsets"]
            if not 0 <= start <= end <= size - header_size - 8:
                raise ValueError("tensor leaves its file")
            source.seek(8 + header_size + start)
            left = end - start
            h = hashlib.sha256()
            while left:
                block = source.read(min(left, 1 << 20))
                if not block:
                    raise EOFError("short tensor payload")
                h.update(block)
                left -= len(block)
            logical = name.removeprefix(prefix) if prefix else name
            rows.append(
                {
                    "name": component + "/" + logical,
                    "dtype": row["dtype"],
                    "shape": row["shape"],
                    "bytes": end - start,
                    "sha256": h.hexdigest(),
                }
            )
    return rows


def native_inventory(value):
    """Only a verified plain value part is a dense logical tensor inventory."""
    dtypes = {
        "f64": "F64",
        "f32": "F32",
        "f16": "F16",
        "bf16": "BF16",
        "i64": "I64",
        "i32": "I32",
        "i16": "I16",
        "i8": "I8",
        "u8": "U8",
        "bool": "BOOL",
    }
    rows = []
    for tensor in value["tensors"]:
        plain = (
            tensor["encoding"]
            == "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
        )
        if not plain or len(tensor["parts"]) != 1 or not tensor["logical_value_sha256"]:
            raise ValueError(
                f"encoded tensor needs an explicit logical-value export: {tensor['name']}"
            )
        part = tensor["parts"][0]
        if (
            part["role"] != "value"
            or part["dtype"] != tensor["dtype"]
            or part["shape"] != tensor["shape"]
            or part["sha256"] != tensor["logical_value_sha256"]
        ):
            raise ValueError("native plain tensor differs from its logical value part")
        rows.append(
            {
                "name": tensor["name"],
                "dtype": dtypes[tensor["dtype"]],
                "shape": tensor["shape"],
                "bytes": part["bytes"],
                "sha256": tensor["logical_value_sha256"],
            }
        )
    return rows


def verify_parity(proof):
    """Reject missing/mismatched measured tensors or scheduler grids before submitting."""
    tensors = proof["tensors"]
    if not tensors["cozy"] or tensors["cozy"] != tensors["comfy"]:
        raise ValueError(
            "actual logical tensor name/dtype/shape/value parity is unproven"
        )
    counts = proof.get("component_tensors", {})
    for engine in ("cozy", "comfy"):
        actual = Counter(row["name"].split("/", 1)[0] for row in tensors[engine])
        if not counts.get(engine) or dict(actual) != counts[engine]:
            raise ValueError(
                "complete native source component tensor census is unproven"
            )
    if not proof.get("source_equations") or not proof.get("conditioning_evidence"):
        raise ValueError("source equation/conditioning parity evidence is absent")
    a, b = proof["sigmas"]["cozy"], proof["sigmas"]["comfy"]
    if len(a) != len(b) or len(a) < 2 or any(not math.isfinite(x) for x in a + b):
        raise ValueError("actual scheduler arrays are absent or invalid")
    tolerance = proof["sigmas"].get("absolute_tolerance", 1e-6)
    if not 0 <= tolerance <= 1e-5 or any(
        abs(x - y) > tolerance for x, y in zip(a, b, strict=True)
    ):
        raise ValueError("actual scheduler grids differ")
    initialization = proof.get("initialization", {})
    left, right = initialization.get("cozy"), initialization.get("comfy")
    fields = {
        "shape",
        "distribution",
        "noise_dtype",
        "state_dtype",
        "noise_scale",
        "latent_zero",
    }
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or not fields <= left.keys()
        or not fields <= right.keys()
    ):
        raise ValueError("actual initialization evidence is incomplete")
    scale_tolerance = initialization.get("absolute_tolerance", 1e-6)
    if not 0 <= scale_tolerance <= 1e-5:
        raise ValueError("initial noise scale tolerance exceeds the matched rule")
    if any(left[key] != right[key] for key in fields - {"noise_scale"}):
        raise ValueError("initial noise geometry, distribution or precision differs")
    if (
        not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in (left["noise_scale"], right["noise_scale"])
        )
        or abs(left["noise_scale"] - right["noise_scale"]) > scale_tolerance
    ):
        raise ValueError("initial noise scales differ")
    numerical = proof.get("numerical_mode", {})
    mode_fields = {
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
        "decode_geometry",
    }
    left, right = numerical.get("cozy"), numerical.get("comfy")
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or not mode_fields <= left.keys()
        or not mode_fields <= right.keys()
        or any(left[key] is None or right[key] is None for key in mode_fields)
    ):
        raise ValueError("actual numerical mode evidence is incomplete")
    # Pressure recovery may tile differently. Keep its actual geometry; the
    # independent same-engine reference gate decides whether it preserves quality.
    if any(left[key] != right[key] for key in mode_fields - {"decode_geometry"}):
        raise ValueError(
            "conditioning, denoising, sampler or decode numerical modes differ"
        )
    sources = proof.get("source_artifacts", {})
    for engine in ("cozy", "comfy"):
        if not sources.get(engine):
            raise ValueError("actual source artifacts are absent")
        for row in sources[engine]:
            if (
                not row.get("symbol")
                or hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                != row["sha256"]
            ):
                raise ValueError("source artifact bytes or symbol provenance differs")
    return digest(
        {
            "tensors": tensors["cozy"],
            "sigmas": a,
            "prediction": proof["prediction"],
            "source_equations": proof["source_equations"],
            "conditioning_evidence": proof["conditioning_evidence"],
            "initialization": initialization,
            "numerical_mode": numerical,
            "source_artifacts": sources,
        }
    )


def verify_conditioning(proof, payload):
    """Require measured tokens and reviewed encoder inputs for this literal request."""
    row = proof["conditioning_evidence"].get(digest(payload), {})
    fields = {
        "positive_token_ids",
        "negative_token_ids",
        "attention_masks",
        "encoder_layers",
        "pooled_projection",
        "microconditioning",
        "prompt_weight_policy",
        "tokenizer_asset_sha256",
    }
    left, right = row.get("cozy"), row.get("comfy")
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or not fields <= left.keys()
        or not fields <= right.keys()
    ):
        raise ValueError("actual request conditioning evidence is incomplete")
    if any(
        left[key] is None or right[key] is None or left[key] != right[key]
        for key in fields
    ):
        raise ValueError("actual tokens, encoder inputs or microconditioning differ")


def pixels(path: Path):
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
        data = image.tobytes()
        shape = list(image.size)
        smoke = min(ImageStat.Stat(image).stddev) > 2
    return {
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pixel_sha256": hashlib.sha256(data).hexdigest(),
        "shape": shape,
        "smoke_nonflat": smoke,
    }, data


def compare(reference: Path, candidate: Path):
    a, x = pixels(reference)
    b, y = pixels(candidate)
    if a["shape"] != b["shape"]:
        raise ValueError("reference/candidate dimensions differ")
    mse = sum((u - v) ** 2 for u, v in zip(x, y, strict=True)) / len(x)
    return {
        "reference": a,
        "candidate": b,
        "exact": x == y,
        "psnr_db": None if mse == 0 else 10 * math.log10(255**2 / mse),
    }


def validate_quality(artifacts, controls, policy, identity):
    """Compare to same-engine unconstrained controls; variance cannot lower the rule."""
    if len(artifacts) != len(controls):
        raise ValueError("one control set per actual request is required")
    method = policy["method"]
    threshold = policy.get("minimum_psnr_db")
    if method not in ("exact", "declared_tolerance"):
        raise ValueError("smoke checks are not quality")
    if method == "declared_tolerance" and (
        not isinstance(threshold, (int, float)) or threshold < 35
    ):
        raise ValueError("declare at least35dB before collecting data")
    checks = []
    all_ok = True
    for paths, control in zip(artifacts, controls, strict=True):
        if control["identity"] != identity or len(control["artifacts"]) < 3:
            raise ValueError(
                "three same-engine unconstrained repetitions with matching provenance required"
            )
        refs = [Path(x) for x in control["artifacts"]]
        if len(paths) != 1:
            raise ValueError("the declared image workload must return one image")
        variance = [compare(refs[0], other) for other in refs[1:]]
        measured = compare(refs[0], Path(paths[0]))
        if method == "exact":
            ok = measured["exact"] and all(x["exact"] for x in variance)
        else:
            baseline = all(
                x["exact"] or x["psnr_db"] >= threshold + 6 for x in variance
            )
            ok = baseline and (measured["exact"] or measured["psnr_db"] >= threshold)
        all_ok &= ok
        checks.append(
            {"ok": ok, "measurement": measured, "unconstrained_variance": variance}
        )
    reference = digest(
        {
            "identity": identity,
            "controls": [
                [pixels(Path(p))[0] for p in c["artifacts"]] for c in controls
            ],
            "policy": policy,
        }
    )
    return {
        "ok": all_ok,
        "method": method,
        "reference_digest": reference,
        "policy": policy,
        "checks": checks,
    }
