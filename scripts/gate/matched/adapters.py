"""Controller-side ordinary Cozy CLI and stock ComfyUI HTTP saved-output adapters."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def durable(path):
    with path.open("rb") as source:
        os.fsync(source.fileno())


def cozy(spec, requests, out):
    rows = []
    first = None
    for index, row in enumerate(requests):
        root = out / str(index)
        root.mkdir(parents=True)
        (root / "input.json").write_text(
            json.dumps(row["input"], sort_keys=True) + "\n"
        )
        args = [
            spec["cli"],
            "run",
            spec["targets"][row["model"]],
            *spec["target_args"][row["model"]],
            "--input",
            str(root / "input.json"),
            *spec["selector_args"],
            "--await",
            "--json",
            "--out",
            str(root / "artifacts"),
            "--idempotency-key=matched-" + uuid.uuid4().hex,
        ]
        (root / "command.json").write_text(json.dumps(args, indent=2) + "\n")
        with (
            (root / "stdout.json").open("wb") as stdout,
            (root / "events.jsonl").open("wb") as stderr,
        ):
            began = time.perf_counter_ns()
            first = first or began
            result = subprocess.run(args, stdout=stdout, stderr=stderr, check=False)
        paths = sorted(
            p
            for p in (root / "artifacts").rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        )
        for path in paths:
            durable(path)
        end = time.perf_counter_ns()
        reply = json.loads((root / "stdout.json").read_text())
        state = reply.get(
            "status", reply.get("state", (reply.get("run") or {}).get("state"))
        )
        cached = bool(reply.get("memo") or reply.get("cached_result"))
        positions = set()
        for line in (root / "events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            progress = event.get("payload", {})
            progress = progress.get("payload", progress)
            if (
                progress.get("stage") == "denoise"
                and progress.get("total") == row["input"]["steps"]
            ):
                position = progress.get("position")
                if isinstance(position, int):
                    positions.add(position)
        # Progress is a live observation; attaching after execution starts may miss
        # early steps. Require current step work through the authored final step,
        # retain the gap, and separately reject an explicitly cached result.
        sampling = row["input"]["steps"] in positions and bool(positions)
        rows.append(
            {
                "model": row["model"],
                "submit_ns": began,
                "saved_ns": end,
                "artifacts": [str(p) for p in paths],
                "returncode": result.returncode,
                "state": state,
                "cached_result": cached,
                "denoise_positions": sorted(positions),
                "all_denoise_events_observed": set(
                    range(1, row["input"]["steps"] + 1)
                ).issubset(positions),
                "fresh_sampling_evidence": sampling,
                "ok": result.returncode == 0
                and state in ("completed", "succeeded")
                and len(paths) == 1
                and not cached
                and sampling,
            }
        )
    return rows, first, max(r["saved_ns"] for r in rows)


def http(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def comfy(spec, requests, out):
    rows = []
    first = None
    for index, row in enumerate(requests):
        root = out / str(index)
        root.mkdir(parents=True)
        graph = build_graph(spec, row)
        (root / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
        prompt_id = str(uuid.uuid4())
        body = {"prompt": graph, "prompt_id": prompt_id, "client_id": spec["client_id"]}
        began = time.perf_counter_ns()
        first = first or began
        submitted = http(spec["base_url"], "/prompt", body)
        if submitted.get("prompt_id") != prompt_id:
            raise ValueError("Comfy changed the submitted prompt identity")
        while True:
            history = http(spec["base_url"], "/history/" + prompt_id).get(prompt_id)
            if history:
                break
            time.sleep(0.05)
        (root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        status = history["status"]["status_str"]
        paths = []
        cached = []
        for message in history["status"].get("messages", []):
            if message[0] == "execution_cached":
                cached += message[1].get("nodes", [])
        for image in history.get("outputs", {}).get("save", {}).get("images", []):
            query = urllib.parse.urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image["subfolder"],
                    "type": image["type"],
                }
            )
            path = root / Path(image["filename"]).name
            with (
                urllib.request.urlopen(spec["base_url"] + "/view?" + query) as source,
                path.open("wb") as dest,
            ):
                while block := source.read(1 << 20):
                    dest.write(block)
                dest.flush()
                os.fsync(dest.fileno())
            paths.append(str(path))
        end = time.perf_counter_ns()
        rows.append(
            {
                "model": row["model"],
                "submit_ns": began,
                "saved_ns": end,
                "artifacts": paths,
                "state": status,
                "cached_nodes": cached,
                "fresh_sampling_evidence": status == "success"
                and "sample" not in cached,
                "ok": status == "success"
                and len(paths) == 1
                and "sample" not in cached,
            }
        )
    return rows, first, max(r["saved_ns"] for r in rows)


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def build_graph(spec, row):
    model = row["model"]
    payload = row["input"]
    files = spec["model_files"][model]
    graph = {
        "load": node("UNETLoader", unet_name=files["denoiser"], weight_dtype="default"),
        "vae": node("VAELoader", vae_name=files["vae"]),
    }
    if model == "sdxl":
        graph["clip"] = node(
            "DualCLIPLoader",
            clip_name1=files["text_encoder"],
            clip_name2=files["text_encoder_2"],
            type="sdxl",
            device="default",
        )
    elif model == "anima":
        graph["clip"] = node(
            "CLIPLoader",
            clip_name=files["text_encoder"],
            type="stable_diffusion",
            device="default",
        )
    else:
        raise ValueError("only declared SDXL/Anima workloads are implemented")
    graph.update(
        positive=node("CLIPTextEncode", clip=["clip", 0], text=payload["prompt"]),
        negative=node(
            "CLIPTextEncode", clip=["clip", 0], text=payload["negative_prompt"]
        ),
        latent=node("EmptyLatentImage", width=1024, height=1024, batch_size=1),
        sample=node(
            "KSampler",
            model=["load", 0],
            seed=payload["seed"],
            steps=payload["steps"],
            cfg=payload["guidance"],
            sampler_name="euler",
            scheduler=spec["scheduler"][model],
            positive=["positive", 0],
            negative=["negative", 0],
            latent_image=["latent", 0],
            denoise=1.0,
        ),
        decode=node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        save=node(
            "SaveImage",
            images=["decode", 0],
            filename_prefix="matched-" + uuid.uuid4().hex,
        ),
    )
    return graph
