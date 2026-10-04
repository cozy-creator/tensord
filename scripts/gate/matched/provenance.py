"""Hash actual safetensors values into a logical component inventory, on host or controller."""

import argparse
import json
from pathlib import Path

from evidence import digest, native_inventory, tensor_inventory

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "files",
    nargs="?",
    type=Path,
    help="[{path,component,remove_prefix}] for the actual model files",
)
p.add_argument("--native", type=Path, help="actual native-source inventory output")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if bool(a.files) == bool(a.native):
    p.error("provide safetensors files or a native-source inventory")
rows = []
if a.native:
    rows = native_inventory(json.loads(a.native.read_text()))
else:
    for file in json.loads(a.files.read_text()):
        rows += tensor_inventory(
            Path(file["path"]), file["component"], file.get("remove_prefix", "")
        )
rows = sorted(rows, key=lambda row: row["name"])
if len({row["name"] for row in rows}) != len(rows):
    raise ValueError("logical tensors overlap")
a.out.write_text(
    json.dumps({"tensors": rows, "tensor_digest": digest(rows)}, indent=2) + "\n"
)
