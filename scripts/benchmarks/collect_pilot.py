"""Verify and bundle an owned pilot cohort without copying raw pixel buffers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from PIL import Image, ImageStat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args()
    if args.archive.exists():
        raise SystemExit("Archive already exists; preserve it and choose a new attempt")
    files: set[Path] = set()
    cohorts = []
    for label in args.labels:
        root = args.workspace / label
        result = json.loads((root / "results.json").read_text())
        config = json.loads((args.workspace / "stage" / f"{label}.json").read_text())
        if len(result["runs"]) != len(config["payloads"]):
            raise SystemExit(f"Incomplete cohort: {label}")
        images = []
        for run in result["runs"]:
            for binding in run["bindings"]:
                path = root / run["id"] / binding["name"]
                if not path.resolve().is_relative_to(root.resolve()):
                    raise SystemExit(f"Artifact escapes owned cohort: {path}")
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                producer = "blake2b:" + hashlib.blake2b(data, digest_size=16).hexdigest()
                if (digest, producer, len(data)) != (
                    binding["sha256"], binding["producer_digest"], binding["length"]
                ):
                    raise SystemExit(f"Artifact checksum differs: {path}")
                with Image.open(path) as image:
                    image.load()
                    expected = (run["result"]["width"], run["result"]["height"])
                    variance = ImageStat.Stat(image.convert("RGB")).var
                    if image.format != "WEBP" or image.size != expected or not any(variance):
                        raise SystemExit(f"Invalid output image: {path}")
                images.append({"id": run["id"], "name": binding["name"], "sha256": digest,
                               "producer_digest": producer, "bytes": len(data),
                               "width": expected[0], "height": expected[1], "variance": variance})
                files.add(path)
        files.update(path for path in root.iterdir() if path.is_file())
        files.update(path for path in (args.workspace / f"observe-{label}").iterdir() if path.is_file())
        files.add(args.workspace / "stage" / f"{label}.json")
        cohorts.append({"label": label, "images": images, "executor_reused":
                        len({run["pid"] for run in result["runs"]}) == 1,
                        "timings": result["timings"], "sources": result["sources"]})
    verified = args.archive.with_suffix(".verified.json")
    if verified.exists():
        raise SystemExit("Verification record already exists")
    verified.write_text(json.dumps(cohorts, indent=2) + "\n")
    files.add(verified)
    with tarfile.open(args.archive, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(args.workspace)), recursive=False)
    print(json.dumps({"archive": str(args.archive), "bytes": args.archive.stat().st_size,
                      "sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                      "cohorts": len(cohorts), "verified_images": sum(len(x["images"]) for x in cohorts)}))


if __name__ == "__main__":
    main()
