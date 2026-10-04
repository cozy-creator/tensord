"""Measure actual Anima scheduler grids from pinned Comfy source and Diffusers.

This produces schedule evidence only. Tensor, conditioning, equation and precision
parity still require separate evidence before run.py will admit a scored row.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import diffusers
import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def symbols(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("required source symbols absent")
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<pinned-comfy>", "exec"),
        namespace,
    )


def measure(comfy, commit, config, steps):
    torch.set_num_threads(2)
    sources = {
        path: subprocess.check_output(
            ["git", "-C", str(comfy), "show", f"{commit}:{path}"], text=True
        )
        for path in ("comfy/model_sampling.py", "comfy/samplers.py")
    }
    scope = {"torch": torch, "math": math}
    symbols(
        sources["comfy/model_sampling.py"],
        ["time_snr_shift", "ModelSamplingDiscreteFlow"],
        scope,
    )
    symbols(
        sources["comfy/samplers.py"], ["normal_scheduler", "simple_scheduler"], scope
    )
    actual = json.loads(config.read_text())
    if actual.get("use_dynamic_shifting") or actual.get("num_train_timesteps") != 1000:
        raise ValueError(
            "this reviewed Anima grid requires static shift and1000training steps"
        )
    sampling = scope["ModelSamplingDiscreteFlow"](
        SimpleNamespace(sampling_settings={"shift": actual["shift"]})
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(actual)
    scheduler.set_timesteps(steps, device="cpu")
    cozy = scheduler.sigmas.tolist()
    arrays = {
        name: scope[name + "_scheduler"](sampling, steps).tolist()
        for name in ("normal", "simple")
    }
    return {
        "model": "anima",
        "steps": steps,
        "config": actual,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "diffusers": diffusers.__version__,
        "comfy_commit": commit,
        "source_sha256": {
            p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()
        },
        "sigmas": {"cozy": cozy, **arrays},
        "max_absolute_difference": {
            name: max(abs(x - y) for x, y in zip(cozy, arr, strict=True))
            for name, arr in arrays.items()
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(measure(a.comfy, a.commit, a.config, a.steps), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
