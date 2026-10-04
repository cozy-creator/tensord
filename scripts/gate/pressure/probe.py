"""Read-only physical GPU, host, and exact-holder observations on a rental."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import time


def capture(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    pid = receipt["pid"]
    actual_start = int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
    if actual_start != receipt["start_ticks"]:
        raise ValueError("the original physical holder is no longer alive")
    if receipt_path.with_suffix(".released.json").exists():
        raise ValueError("the physical holder was already released")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip().split(", ")
    if len(gpu) != 4 or gpu[0] != receipt["gpu_uuid"]:
        raise ValueError("GPU identity changed under the owned holder")
    if int(gpu[1]) * (1 << 20) > receipt["target_free_bytes"] + (2 << 20):
        raise ValueError("available physical pool exceeded its authored ceiling")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
    ], text=True).strip()
    holder_rows = [line.split(", ") for line in processes.splitlines() if line.split(", ")[0] == str(pid)]
    if len(holder_rows) != 1 or int(holder_rows[0][1]) * (1 << 20) < receipt["held_bytes"]:
        raise ValueError("physical GPU process records do not retain the authored ballast")
    process_memory = {}
    for line in processes.splitlines():
        process_id = line.split(", ")[0]
        if not process_id.isdecimal():
            raise ValueError("NVIDIA returned an unreadable process identity")
        process_path = pathlib.Path("/proc") / process_id
        try:
            process_memory[process_id] = {
                "status": (process_path / "status").read_text(),
                "smaps_rollup": (process_path / "smaps_rollup").read_text(),
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            # GPU process rows and /proc are independent observations. An executor may
            # exit or become nondumpable between them. Its optional host RSS/PSS stays
            # unknown; the holder's identity, physical allocation and ceiling checks
            # above remain mandatory.
            process_memory[process_id] = {
                "observation": "host_memory_unavailable", "error": repr(error)
            }
    return {
        "remote": True, "rental": receipt["rental"], "rental_id": receipt["rental_id"],
        "operation": receipt["operation"], "gpu_uuid": gpu[0], "driver": receipt["driver"],
        "total_bytes": receipt["total_bytes"], "free_bytes": int(gpu[1]) << 20,
        "used_bytes": int(gpu[2]) << 20, "utilization_percent": int(gpu[3]),
        "holder_pid": pid, "holder_start_ticks": actual_start,
        "processes": processes, "observed_unix_ns": time.time_ns(),
        "process_memory": process_memory,
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "host_meminfo": pathlib.Path("/proc/meminfo").read_text(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.receipt)), flush=True)


if __name__ == "__main__":
    main()
