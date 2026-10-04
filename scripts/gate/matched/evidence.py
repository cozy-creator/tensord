"""Actual model tensor and same-engine pixel evidence for matched benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import struct
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


def verify_parity(proof):
    """Reject missing/mismatched measured tensors or scheduler grids before submitting."""
    tensors = proof["tensors"]
    if not tensors["cozy"] or tensors["cozy"] != tensors["comfy"]:
        raise ValueError(
            "actual logical tensor name/dtype/shape/value parity is unproven"
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
    return digest(
        {
            "tensors": tensors["cozy"],
            "sigmas": a,
            "prediction": proof["prediction"],
            "source_equations": proof["source_equations"],
            "conditioning_evidence": proof["conditioning_evidence"],
        }
    )


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
