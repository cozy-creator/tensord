"""A fixed physical ballast for one fresh, explicitly owned rental GPU.

Run before any model preparation. No reclaim, process kills, model execution, or
pool adjustment occurs here. Release requires the exact authored operation key.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", choices=(1.0, 1.25, 1.5, 3.0), type=float, required=True)
    parser.add_argument("--rental", required=True)
    parser.add_argument("--rental-id", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--release", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.release.exists():
        raise ValueError("each holder needs new owned receipt and release paths")
    before = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if before:
        raise ValueError("fresh-pool proof refuses existing CUDA processes: " + before)
    identity = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,display_active", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    if len(identity) != 5 or "RTX 3070" not in identity[0] or identity[4] != "Disabled":
        raise ValueError("rental identity is not the declared single display-free 3070")

    import torch

    torch.set_num_threads(2)
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    target = int(args.gib * (1 << 30))
    if free < target:
        raise ValueError("fresh GPU has less available memory than its declared pool")
    held = torch.empty(free - target, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    observed, _ = torch.cuda.mem_get_info()
    if abs(observed - target) > (2 << 20):
        raise ValueError(f"physical free pool differs from declaration: {observed}, {target}")
    start_ticks = int(pathlib.Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19])
    receipt = {
        "remote": True, "fresh_before_first_model_prepare": True,
        "owner": "gpu_qualification", "rental": args.rental, "rental_id": args.rental_id,
        "operation": args.operation, "pid": os.getpid(), "start_ticks": start_ticks,
        "gpu_name": identity[0], "gpu_uuid": identity[1], "driver": identity[2],
        "total_bytes": total, "before_free_bytes": free,
        "target_free_bytes": target, "observed_free_bytes": observed,
        "held_bytes": held.numel(), "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "release_file": str(args.release), "created_unix_ns": time.time_ns(),
        "host_meminfo": pathlib.Path("/proc/meminfo").read_text(),
    }
    atomic(args.receipt, receipt)
    print(json.dumps(receipt), flush=True)
    while True:
        if args.release.exists():
            release = json.loads(args.release.read_text())
            if release != {"operation": args.operation, "rental_id": args.rental_id}:
                raise ValueError("release receipt does not name this exact owned holder")
            break
        time.sleep(0.2)
    del held
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    atomic(args.receipt.with_suffix(".released.json"), {
        "operation": args.operation, "rental_id": args.rental_id,
        "pid": os.getpid(), "start_ticks": start_ticks,
        "free_bytes": torch.cuda.mem_get_info()[0], "released_unix_ns": time.time_ns(),
    })


if __name__ == "__main__":
    main()
