"""Fresh paired rows from ordinary CLI/stock HTTP with controller-saved output references.

Root owns engine/ballast/rental lifecycle. This driver starts no server, ends no rental,
changes no request and kills no process. Quality/timing checks follow actual artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from adapters import comfy, cozy
from evidence import (
    digest,
    pixels,
    validate_quality,
    verify_conditioning,
    verify_parity,
)


def probe(command, path):
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    path.write_text(
        json.dumps(
            {"command": command, "stdout": done.stdout, "stderr": done.stderr}, indent=2
        )
        + "\n"
    )
    return done.stdout


def run(manifest_path, out, engine, cell, pair, reference_only):
    manifest = json.loads(manifest_path.read_text())
    spec = manifest["arms"][engine]
    definition = manifest["cells"][cell]
    inputs = definition["requests"]
    if not inputs:
        raise ValueError("an authored benchmark cell must contain requests")
    if definition.get("kind") == "degraded" and len(inputs) != 6:
        raise ValueError(
            "the declared pressure cell requires all six unchanged requests"
        )
    out.mkdir(parents=True, exist_ok=False)
    proof = json.loads(Path(manifest["parity"]).read_text())
    required_components = {
        "sdxl": {"unet", "text_encoder", "text_encoder_2", "vae"},
        "anima": {"transformer", "text_encoder", "text_conditioner", "vae"},
    }
    for name in {row["model"] for row in inputs}:
        for arm in ("cozy", "comfy"):
            if (
                set(proof[name].get("component_tensors", {}).get(arm, {}))
                != required_components[name]
            ):
                raise ValueError(
                    "all denoiser, encoder, conditioner and VAE tensors are required"
                )
    models = {
        name: verify_parity(proof[name]) for name in {row["model"] for row in inputs}
    }
    for row in inputs:
        payload = row["input"]
        model = row["model"]
        verify_conditioning(proof[model], payload)
        if payload.get("aspect_ratio") != "1:1" or payload.get("megapixels") != 1:
            raise ValueError("unsupported geometry cannot be silently resized")
        if model == "sdxl" and payload.get("hidiffusion") is not False:
            raise ValueError("Comfy graph does not implement HiDiffusion")
        if model == "anima" and (
            payload.get("quality_prefix") != ""
            or payload.get("cfg_interval_start") != 0
            or payload.get("cfg_interval_stop") != 1
            or payload.get("first_block_cache") != 0
        ):
            raise ValueError(
                "Comfy graph does not implement the requested Anima modifiers"
            )
        if (
            spec["name"] == "ComfyUI"
            and spec["scheduler"][model] != proof[model]["comfy_scheduler"]
        ):
            raise ValueError("actual graph scheduler differs from the proved grid")
    normalized = {
        "requests": inputs,
        "models": models,
        "geometry": [1024, 1024],
        "same_authored_seed_engine_specific_rng": True,
    }
    request_digest = digest(normalized)
    hardware = json.loads(
        probe(manifest["hardware_probe"], out / "hardware-probe.json")
    )
    if (
        not hardware.get("remote")
        or not hardware.get("gpu_uuid")
        or not hardware.get("driver")
    ):
        raise ValueError("actual rented hardware identity is unproven")
    hardware_key = digest(
        {k: hardware[k] for k in ("gpu_uuid", "driver", "total_bytes")}
    )
    raw = probe(spec["commit_probe"], out / "engine-probe.json").strip()
    if spec["name"] == "cozy-machine":
        version = json.loads(raw)
        revision = version.get("revision") or version.get("commit")
    else:
        revision = raw
    if not isinstance(revision, str) or len(revision) < 7:
        raise ValueError("actual engine commit is unproven")
    identity = {
        "engine": {"name": spec["name"], "commit": revision},
        "request_digest": request_digest,
        "hardware_key": hardware_key,
    }
    (out / "normalized-request.json").write_text(
        json.dumps(normalized, indent=2) + "\n"
    )
    paths, first, end = (cozy if spec["name"] == "cozy-machine" else comfy)(
        spec, inputs, out / "requests"
    )
    artifacts = [row["artifacts"] for row in paths]
    # Saving is timed equally; decode/reference comparison is outside that boundary.
    for row in paths:
        try:
            row["images"] = [pixels(Path(p))[0] for p in row["artifacts"]]
            row["smoke_ok"] = len(row["images"]) == 1 and all(
                image["shape"] == [1024, 1024] and image["smoke_nonflat"]
                for image in row["images"]
            )
        except (OSError, ValueError) as error:
            row["images"] = []
            row["smoke_ok"] = False
            row["artifact_error"] = f"{type(error).__name__}: {error}"
    result = {
        "event": "cell",
        "arm": engine,
        "cell": cell,
        "pair": pair,
        **identity,
        "total_s": (end - first) / 1e9,
        "t_first_monotonic_ns": first,
        "t_end_monotonic_ns": end,
        "timing_boundary": "submit_to_saved_output",
        "output_location": "controller",
        "saved_boundary": "closed_and_fsynced_controller_files",
        "requests": paths,
        "ok": all(row["ok"] and row["smoke_ok"] for row in paths),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "hardware": hardware,
        "normalized_request": normalized,
    }
    if reference_only:
        if manifest["cells"][cell].get("budget") != "unconstrained":
            raise ValueError("a constrained cell cannot be its own quality control")
        result["reference_identity"] = identity
        result["reference_only"] = True
    else:
        try:
            controls = json.loads(Path(manifest["controls"][engine][cell]).read_text())
            result["quality"] = validate_quality(
                artifacts, controls, manifest["quality"], identity
            )
        except (OSError, ValueError) as error:
            result["quality"] = {
                "ok": False,
                "method": manifest["quality"]["method"],
                "error": f"{type(error).__name__}: {error}",
            }
        result["ok"] &= result["quality"]["ok"]
    with (out / "result.json").open("w") as dest:
        json.dump(result, dest, indent=2)
        dest.write("\n")
        dest.flush()
        os.fsync(dest.fileno())
    print(json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--engine", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--reference-only", action="store_true")
    a = p.parse_args()
    run(a.manifest, a.out, a.engine, a.cell, a.pair, a.reference_only)


if __name__ == "__main__":
    main()
