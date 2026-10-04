"""Measure actual SDXL/Anima grids from pinned Comfy source and Diffusers.

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
from diffusers import EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler


def symbols(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("required source symbols absent")
    # Like importing Comfy, this diagnostic executes the caller's reviewed Git source.
    # Keep the actual scheduler definitions instead of writing a lookalike algorithm.
    exec(  # noqa: S102
        compile(ast.Module(body=selected, type_ignores=[]), "<pinned-comfy>", "exec"),
        namespace,
    )


def measure(comfy, commit, config, steps, model="anima"):
    torch.set_num_threads(2)
    sources = {
        path: subprocess.check_output(
            ["git", "-C", str(comfy), "show", f"{commit}:{path}"], text=True
        )
        for path in ("comfy/model_sampling.py", "comfy/samplers.py")
    }
    scope = {"torch": torch, "math": math}
    symbols(
        sources["comfy/samplers.py"],
        ["normal_scheduler", "simple_scheduler", "ddim_scheduler"],
        scope,
    )
    actual = json.loads(config.read_text())
    if model == "anima":
        symbols(
            sources["comfy/model_sampling.py"],
            ["time_snr_shift", "ModelSamplingDiscreteFlow"],
            scope,
        )
        if (
            actual.get("use_dynamic_shifting")
            or actual.get("num_train_timesteps") != 1000
        ):
            raise ValueError(
                "reviewed Anima grid requires static shift and1000 training steps"
            )
        sampling = scope["ModelSamplingDiscreteFlow"](
            SimpleNamespace(sampling_settings={"shift": actual["shift"]})
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(actual)
    elif model == "sdxl":
        if (
            actual.get("beta_schedule") != "scaled_linear"
            or actual.get("trained_betas") is not None
        ):
            raise ValueError(
                "reviewed SDXL grid requires its actual scaled_linear beta configuration"
            )
        path = "comfy/ldm/modules/diffusionmodules/util.py"
        sources[path] = subprocess.check_output(
            ["git", "-C", str(comfy), "show", f"{commit}:{path}"], text=True
        )
        symbols(sources[path], ["make_beta_schedule"], scope)
        symbols(sources["comfy/model_sampling.py"], ["ModelSamplingDiscrete"], scope)
        sampling = scope["ModelSamplingDiscrete"](
            SimpleNamespace(
                sampling_settings={
                    "beta_schedule": "linear",
                    "linear_start": actual["beta_start"],
                    "linear_end": actual["beta_end"],
                    "timesteps": actual["num_train_timesteps"],
                }
            )
        )
        scheduler = EulerDiscreteScheduler.from_config(actual)
    else:
        raise ValueError("only actual SDXL/Anima grids are reviewed here")
    scheduler.set_timesteps(steps, device="cpu")
    cozy = scheduler.sigmas.tolist()
    arrays = {
        name: scope[name + "_scheduler"](sampling, steps).tolist()
        for name in ("normal", "simple", "ddim")
    }
    result = {
        "model": model,
        "steps": steps,
        "config": actual,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "diffusers": diffusers.__version__,
        "comfy_commit": commit,
        "source_sha256": {
            p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()
        },
        "sigmas": {"cozy": cozy, **arrays},
        "array_lengths": {name: len(arr) for name, arr in arrays.items()},
        "max_absolute_difference": {
            name: max(abs(x - y) for x, y in zip(cozy, arr, strict=True))
            if len(cozy) == len(arr)
            else None
            for name, arr in arrays.items()
        },
    }
    if model == "sdxl":
        symbols(sources["comfy/model_sampling.py"], ["reshape_sigma", "EPS"], scope)
        symbols(sources["comfy/samplers.py"], ["Sampler"], scope)
        model_wrap = SimpleNamespace(
            inner_model=SimpleNamespace(model_sampling=sampling)
        )
        comfy_sigmas = scope["ddim_scheduler"](sampling, steps)
        max_denoise = scope["Sampler"]().max_denoise(model_wrap, comfy_sigmas)
        ones = torch.ones((1, 4, 1, 1), dtype=torch.float32)
        scaled = scope["EPS"]().noise_scaling(
            comfy_sigmas[0], ones, torch.zeros_like(ones), max_denoise
        )
        cozy_scale, comfy_scale = (
            float(scheduler.init_noise_sigma),
            float(scaled.flatten()[0]),
        )
        result["initial_noise_scaling"] = {
            "cozy": cozy_scale,
            "comfy_ddim_uniform": comfy_scale,
            "comfy_max_denoise": max_denoise,
            "comfy_model_sigma_max": float(sampling.sigma_max),
            "absolute_difference": abs(cozy_scale - comfy_scale),
            "ratio": cozy_scale / comfy_scale,
            "matched_at_1e-5": abs(cozy_scale - comfy_scale) <= 1e-5,
            "scope": "source CPU scalar measurement, not live latent or inference parity",
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--model", choices=["sdxl", "anima"], default="anima")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(measure(a.comfy, a.commit, a.config, a.steps, a.model), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
