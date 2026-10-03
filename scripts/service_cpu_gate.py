"""Real Rust owner -> installed Runtime -> package -> durable output gate.

Observation limits fail this test; they never cancel accepted work implicitly.
"""
import argparse
import hashlib
import json
import os
import select
import signal
import subprocess
import time
from pathlib import Path

from cozy_machine_client.client import Client, MachineError
from cozy_machine_client.execution_client import ExecutionClient
from cozy_machine_client.protocol import Shutdown, ShutdownReply


def terminal(client, identifier, seconds=30):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        record = client.get(identifier)
        if record.state in ("completed", "failed", "canceled"):
            return record
        time.sleep(.02)
    raise AssertionError(f"observation budget exhausted: {record}")


def start(args, error_log):
    machine = subprocess.Popen([str(args.binary.resolve()), "serve", "--state",
                                str(args.output / "state"), "--generations",
                                str(args.generations.resolve()), "--host-bytes", "0"],
                               stdout=subprocess.PIPE, stderr=error_log, text=True)
    line = machine.stdout.readline().strip()
    assert line.startswith("READY "), line
    line = machine.stdout.readline().strip()
    assert line.startswith("ADMIN "), line
    return machine, line.removeprefix("ADMIN ")  # submit, cancel, results and shutdown


def killed_mid_run(args, error_log, machine, socket, inputs):
    """SIGKILL the machine during real work: the runner dies with it (parent-death), the
    restarted machine fails that started run with its reason, and new work still runs."""
    client = ExecutionClient(socket)
    run = client.submit("killed-mid-run", args.generation, "classify",
                        {**inputs, "iterations": 1000000})
    until = time.monotonic() + 30
    record = client.get(run.id)
    while record.state != "running" or record.process is None:
        assert time.monotonic() < until, record
        time.sleep(.02)
        record = client.get(run.id)
    runner = os.pidfd_open(record.process.pid)
    machine.send_signal(signal.SIGKILL)  # explicit fault injection on this owned process
    machine.wait()
    client.close()
    # No deadline decides this: the runner's exit is observed on its pidfd.
    select.select([runner], [], [])
    os.close(runner)
    restarted, socket = start(args, error_log)
    client = ExecutionClient(socket)
    failed = terminal(client, run.id)
    assert failed.state == "failed", failed
    assert "owner lost" in failed.failure, failed.failure
    assert failed.attempt == 1
    after = client.submit("after-restart", args.generation, "classify", inputs)
    assert terminal(client, after.id).state == "completed"
    client.close()
    return restarted, socket, {"killed_run": run.id, "runner_pid": record.process.pid,
                               "failure": failed.failure, "after_restart": after.id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    error_log = (args.output / "machine.stderr").open("w")
    machine, socket = start(args, error_log)
    ready_seconds = time.perf_counter() - started
    maps = Path(f"/proc/{machine.pid}/maps").read_text().lower()
    assert "libcuda" not in maps and "libnvidia-ml" not in maps
    samples = [[5.1,3.5,1.4,.2],[6.,2.7,5.1,1.6],[6.7,3.1,4.7,1.5]]
    inputs = {"samples": samples, "iterations": 2, "seed": 19}
    client = ExecutionClient(socket)
    # Explicit cancellation of real work, and a refused stop must leave both
    # halves of the owner accepting. Socket loss alone never cancels this work.
    busy = client.submit("cancel-proof", args.generation, "classify",
                         {**inputs, "iterations": 1000000})
    administrative = Client(socket)
    try:
        administrative.exchange(Shutdown(administrative.next_sequence()), ShutdownReply)
        raise AssertionError("stopped with accepted work")
    except MachineError:
        pass
    client.cancel(busy.id)
    assert terminal(client, busy.id).state == "canceled"
    client.close()
    runs = []
    for index in range(3):
        before = time.perf_counter()
        observer = ExecutionClient(socket)
        receipt = observer.submit(f"inference-{index}", args.generation, "classify", inputs)
        observer.close()
        observer = ExecutionClient(socket)
        # Replay is admission idempotency; it must not create a second attempt.
        replay = observer.submit(f"inference-{index}", args.generation, "classify", inputs)
        assert replay.id == receipt.id
        record = terminal(observer, receipt.id)
        assert record.state == "completed", record.failure
        assert record.result.value["predictions"] == [0, 2, 1]
        assert record.attempt == 1 and record.cancel_actor is None
        data = observer.read_result(record.id)
        report = json.loads(data)
        assert report["predictions"] == [0, 2, 1]
        assert report["seed"] == 19
        (args.output / f"result-{index}.json").write_bytes(data)
        runs.append({"execution_id": record.id, "seconds": time.perf_counter()-before,
                     "pid": record.process.pid if record.process is not None else None,
                     "sha256": hashlib.sha256(data).hexdigest(), "length": len(data)})
        observer.close()
    administrative.close()
    machine, socket, restart = killed_mid_run(args, error_log, machine, socket, inputs)
    administrative = Client(socket)
    administrative.exchange(Shutdown(administrative.next_sequence()), ShutdownReply)
    administrative.close()
    assert machine.wait(timeout=10) == 0
    error_log.close()
    report = {"machine_ready_seconds": ready_seconds, "runs": runs, "restart": restart,
              "runtime_dependencies": json.loads((args.generations / args.generation /
                                                   "generation.json").read_text())["dependencies"],
              "gpu_qualified": False, "ordinary_cli_qualified": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
