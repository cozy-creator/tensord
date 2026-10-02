"""Real CPU classifier component gate. This is not ordinary `cozy run` qualification."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import msgspec
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .client import Client, MachineError
from .protocol import CAPABILITIES, HelloReply, ObjectRef, Shutdown, ShutdownReply


class Fixture(msgspec.Struct):
    object: ObjectRef
    features: int
    classes: int
    samples: int
    reference_accuracy: float


class Ready(msgspec.Struct, tag="ready", tag_field="kind"):
    pid: int
    lease: int
    incarnation: str
    tier: str


class Inferred(msgspec.Struct, tag="inferred", tag_field="kind"):
    predictions: list[int]
    logits_sha256: str
    max_reference_error: float


class Stopped(msgspec.Struct, tag="stopped", tag_field="kind"):
    pid: int


class Infer(msgspec.Struct, tag="infer", tag_field="kind"):
    pass


class Stop(msgspec.Struct, tag="stop", tag_field="kind"):
    pass


class FutureHello(msgspec.Struct, tag="hello", tag_field="kind"):
    seq: int
    runtime: str
    capabilities: list[str]
    future_field: str


class UnknownOperation(msgspec.Struct, tag="future-operation", tag_field="kind"):
    seq: int
    future_field: int


class Evidence(msgspec.Struct):
    fixture: Fixture
    host_backing_bytes: int
    crash_replacements: int
    survivor_prediction_checks: int
    cold_attach_ms: float
    warm_attach_median_ms: float
    disk_fallback_prediction_checks: int
    host_lru_evictions: int
    live_view_release_checks: int
    skew_and_additive_checks: int
    unknown_operation_recoveries: int
    no_gpu_libraries_or_device_fds: bool
    fixture_scope: str = "experimental CPU component; not cozy run, CUDA, NCCL or cold-model qualification"


def encode_line(record: msgspec.Struct, pipe):
    pipe.write(msgspec.json.encode(record) + b"\n")
    pipe.flush()


def make_fixture(path: Path) -> Fixture:

    digits = load_digits()
    samples = digits.data.astype("float64") / 16
    train_x, test_x, train_y, test_y = train_test_split(
        samples, digits.target, test_size=0.25, random_state=7, stratify=digits.target,
    )
    classifier = LogisticRegression(C=3, max_iter=500, random_state=7).fit(train_x, train_y)
    assert classifier.classes_.tolist() == list(range(10))
    # A packed coefficient+bias tensor; no package code executes in the service.
    weights = np.concatenate([classifier.coef_, classifier.intercept_[:, None]], axis=1).astype("<f8")
    data = weights.tobytes()
    reference_logits = classifier.decision_function(test_x)
    reference_predictions = classifier.predict(test_x)
    accuracy = float(np.mean(reference_predictions == test_y))
    assert accuracy > 0.90, accuracy
    path.mkdir(parents=True, exist_ok=True)
    (path / "weights.bin").write_bytes(data)
    np.save(path / "samples.npy", test_x, allow_pickle=False)
    np.save(path / "reference-logits.npy", reference_logits, allow_pickle=False)
    np.save(path / "reference-predictions.npy", reference_predictions, allow_pickle=False)
    fixture = Fixture(ObjectRef(hashlib.sha256(data).hexdigest(), len(data)), 64, 10, len(test_x), accuracy)
    (path / "fixture.json").write_bytes(msgspec.json.encode(fixture))
    return fixture


def executor(socket_path: str, fixture_path: Path):
    fixture = msgspec.json.decode((fixture_path / "fixture.json").read_bytes(), type=Fixture)
    samples = np.load(fixture_path / "samples.npy", allow_pickle=False)
    reference = np.load(fixture_path / "reference-logits.npy", allow_pickle=False)
    command_decoder = msgspec.json.Decoder(Infer | Stop)
    with Client(socket_path) as client:
        client.hello()
        with client.attach(fixture.object) as borrower:
            mapping = borrower.map()
            assert hashlib.sha256(mapping).hexdigest() == fixture.object.sha256
            weights = np.frombuffer(mapping, dtype="<f8").reshape(fixture.classes, fixture.features + 1)
            encode_line(Ready(os.getpid(), borrower.reply.lease, borrower.reply.incarnation, borrower.reply.tier), sys.stdout.buffer)
            try:
                for line in sys.stdin.buffer:
                    command = command_decoder.decode(line)
                    if isinstance(command, Stop):
                        break
                    logits = samples @ weights[:, :fixture.features].T + weights[:, fixture.features]
                    assert np.allclose(logits, reference, rtol=1e-10, atol=1e-10)
                    encode_line(Inferred(np.argmax(logits, axis=1).tolist(), hashlib.sha256(logits.tobytes()).hexdigest(),
                                         float(np.max(np.abs(logits - reference)))), sys.stdout.buffer)
            finally:
                # Do not claim release while a view can still use the backing.
                del weights
    encode_line(Stopped(os.getpid()), sys.stdout.buffer)


class Child:
    decoder = msgspec.json.Decoder(Ready | Inferred | Stopped)

    def __init__(self, socket_path: str, fixture_path: Path, logs: Path, number: int):
        self.log = (logs / f"executor-{number}.log").open("wb")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cozy_machine_client.cpu_gate", "--executor", "--socket", socket_path,
             "--fixture", str(fixture_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
        )
        try:
            self.ready = self.read(Ready)
            assert self.ready.pid == self.process.pid
        except BaseException:
            self.close()
            raise

    def read(self, expected):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"executor {self.process.pid} ended; see {self.log.name}")
        reply = self.decoder.decode(line)
        assert isinstance(reply, expected), type(reply)
        return reply

    def infer(self) -> Inferred:
        encode_line(Infer(), self.process.stdin)
        return self.read(Inferred)

    def crash_after_inference(self):
        # Explicit fault injection after the parent has consumed successful output.
        os.kill(self.process.pid, signal.SIGKILL)
        assert self.process.wait() == -signal.SIGKILL
        self._close_pipes()

    def _close_pipes(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()

    def close(self):
        if self.process.poll() is None:
            try:
                encode_line(Stop(), self.process.stdin)
            except BrokenPipeError:
                pass
            self.process.wait()
        if not self.log.closed:
            self._close_pipes()


class Service:
    def __init__(self, binary: Path, state: Path, budget: int, log: Path):
        self.log = log.open("wb")
        self.process = subprocess.Popen(
            [str(binary), "serve", "--state", str(state), "--host-bytes", str(budget)],
            stdout=subprocess.PIPE, stderr=self.log,
        )
        self.socket = str(state / "machine.sock")
        ready = self.process.stdout.readline()
        if not ready.startswith(b"READY"):
            self.process.wait()
            self.log.close()
            raise RuntimeError(f"machine did not signal readiness; see {log}")
        try:
            self.assert_cpu_only()
        except BaseException:
            self.process.terminate()
            self.process.wait()
            self.process.stdout.close()
            self.log.close()
            raise

    def assert_cpu_only(self):
        maps = Path(f"/proc/{self.process.pid}/maps").read_text()
        assert "libcuda" not in maps and "libnvidia-ml" not in maps, maps
        for fd in Path(f"/proc/{self.process.pid}/fd").iterdir():
            try:
                target = fd.readlink().as_posix()
            except FileNotFoundError:
                continue
            assert not target.startswith(("/dev/nvidia", "/dev/dri/")), target

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    with Client(self.socket) as client:
                        client.hello()
                        assert client.stats().active_leases == 0
                        client.exchange(Shutdown(client.next_sequence()), ShutdownReply)
                except BaseException:
                    # Explicit teardown of this fixture's exact owned process;
                    # an assertion failure must not leave a service behind.
                    self.process.terminate()
                    self.process.wait()
                    raise
                assert self.process.wait() == 0
        finally:
            self.process.stdout.close()
            self.log.close()


def assert_predictions(result: Inferred, expected: list[int], digest: str | None = None):
    assert result.predictions == expected
    assert result.max_reference_error < 1e-10
    if digest is not None:
        assert result.logits_sha256 == digest


def compatibility(socket_path: str):
    for version in ("ancient-runtime/0.0", "future-runtime/999.0"):
        with Client(socket_path) as client:
            reply = client.exchange(FutureHello(client.next_sequence(), version,
                                     [*CAPABILITIES, "future.optional/99"], "additive"), HelloReply)
            client.peer = reply
            assert set(reply.capabilities) == set(CAPABILITIES)
            try:
                client.exchange(UnknownOperation(client.next_sequence(), 7), HelloReply)
            except MachineError:
                pass
            else:
                raise AssertionError("unknown operation unexpectedly succeeded")
            assert client.stats().active_leases == 0


def run_gate(binary: Path, output: Path, crashes: int) -> Evidence:
    assert sys.platform == "linux", "this component gate needs Linux memfd/pidfd/SCM_RIGHTS"
    assert crashes >= 10, "qualification needs at least ten killed executor incarnations"
    output.mkdir(parents=True, exist_ok=True)
    fixture_path = output / "fixture"
    fixture = make_fixture(fixture_path)
    data = (fixture_path / "weights.bin").read_bytes()
    expected = np.load(fixture_path / "reference-predictions.npy", allow_pickle=False).tolist()
    page_size = os.sysconf("SC_PAGE_SIZE")
    host_allocation = ((fixture.object.length + page_size - 1) // page_size) * page_size
    children: list[Child] = []
    number = 0
    with tempfile.TemporaryDirectory(prefix="cozy-machine-cpu-") as states:
        service = Service(binary, Path(states) / "host", host_allocation, output / "machine-host.log")
        try:
            compatibility(service.socket)
            with Client(service.socket) as control:
                control.hello()
                assert control.stats().host_bytes == 0  # clean cold state
                assert control.import_bytes(data).object == fixture.object
                start = time.perf_counter_ns()
                cold = control.attach(fixture.object)
                cold_ms = (time.perf_counter_ns() - start) / 1e6
                assert cold.reply.tier == "host"
                cold.close()
                warm_ms = []
                for _ in range(5):
                    start = time.perf_counter_ns()
                    warm = control.attach(fixture.object)
                    warm_ms.append((time.perf_counter_ns() - start) / 1e6)
                    assert warm.reply.tier == "host"
                    warm.close()
                assert control.stats().cached_objects == 1
                protected = control.attach(fixture.object)
                live_view = np.frombuffer(protected.map(), dtype="<f8")
                try:
                    protected.close()
                except BufferError:
                    pass
                else:
                    raise AssertionError("released an object with a live array view")
                assert control.stats().active_leases == 1
                del live_view
                protected.close()
                assert control.stats().active_leases == 0
                survivor = Child(service.socket, fixture_path, output, number)
                children.append(survivor)
                number += 1
                baseline = survivor.infer()
                assert_predictions(baseline, expected)
                assert control.stats().active_leases == 1
                # A known valid, active inference borrower is killed each time.
                # Server Stats reaps dead peer leases via pidfds, not a timer.
                for _ in range(crashes):
                    child = Child(service.socket, fixture_path, output, number)
                    children.append(child)
                    number += 1
                    assert control.stats().active_leases == 2
                    assert_predictions(child.infer(), expected, baseline.logits_sha256)
                    child.crash_after_inference()
                    assert control.stats().active_leases == 1
                    assert control.stats().host_bytes == host_allocation
                    assert_predictions(survivor.infer(), expected, baseline.logits_sha256)
                replacement = Child(service.socket, fixture_path, output, number)
                children.append(replacement)
                number += 1
                assert_predictions(replacement.infer(), expected, baseline.logits_sha256)
                replacement.close()
                assert control.stats().active_leases == 1
                other = bytes([data[0] ^ 1]) + data[1:]
                other_ref = control.import_bytes(other).object
                with control.attach(other_ref) as disk:
                    assert disk.reply.tier == "disk"  # no eviction of the active survivor
                    assert disk.map()[:] == other
                assert control.stats().cached_objects == 1
                survivor.close()
                assert control.stats().active_leases == 0
                with control.attach(other_ref) as host:
                    assert host.reply.tier == "host"  # idle A is now evictable
                    assert host.map()[:] == other
                assert control.stats().host_bytes == host_allocation
                assert control.stats().cached_objects == 1
                with control.attach(fixture.object) as restored:
                    assert restored.reply.tier == "host"
                    assert restored.map()[:] == data
                service.assert_cpu_only()
        finally:
            for child in children:
                child.close()
            service.close()
        disk_service = Service(binary, Path(states) / "disk", 0, output / "machine-disk.log")
        try:
            with Client(disk_service.socket) as control:
                control.hello()
                control.import_bytes(data)
                disk_child = Child(disk_service.socket, fixture_path, output, number)
                children.append(disk_child)
                try:
                    assert disk_child.ready.tier == "disk"
                    assert_predictions(disk_child.infer(), expected, baseline.logits_sha256)
                    stats = control.stats()
                    assert stats.host_bytes == stats.cached_objects == 0
                    assert stats.active_leases == 1
                finally:
                    disk_child.close()
                assert control.stats().active_leases == 0
                disk_service.assert_cpu_only()
        finally:
            disk_service.close()
    evidence = Evidence(fixture, host_allocation, crashes, crashes + 1, cold_ms, statistics.median(warm_ms), 1, 2, 1, 2, 2, True)
    (output / "results.json").write_bytes(msgspec.json.encode(evidence))
    (output / "RESULTS.md").write_text(
        f"# CPU component qualification\n\n"
        f"Passed with an actual trained sklearn digits classifier: {fixture.reference_accuracy:.2%} accuracy "
        f"on {fixture.samples} held-out samples, {fixture.object.length:,} weight bytes.\n\n"
        f"- {crashes} executor kills after successful attachment and inference; all survivor outputs and replacement outputs matched.\n"
        "- Live NumPy views prevented unmapping/release until explicitly dropped.\n"
        "- Dead leases reclaimed through pidfd-observed process death; no timeout or sleep kills.\n"
        "- Cold/warm shared host attachment, active-borrower disk fallback, zero-host-budget disk inference and two idle LRU evictions passed.\n"
        "- Older/newer runtime strings, additive fields and unknown-operation recovery passed.\n"
        "- Runtime maps and descriptors contained no CUDA/NVML library or GPU device path.\n"
        f"- Single cold attach: {cold_ms:.3f} ms; five-attach warm median: {statistics.median(warm_ms):.3f} ms. "
        "These tiny component timings are not model cold-start benchmarks.\n\n"
        "This is an experimental CPU component gate. It does not qualify ordinary `cozy run`, CUDA allocation sharing, "
        "real diffusion models, NCCL, arbitrary-low-memory recovery or interrupted GPU computation.\n"
    )
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--output", type=Path, default=Path("cpu-evidence"))
    parser.add_argument("--crashes", type=int, default=10)
    parser.add_argument("--executor", action="store_true")
    parser.add_argument("--socket")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.executor:
        executor(args.socket, args.fixture)
    else:
        if args.machine is None:
            parser.error("--machine is required")
        evidence = run_gate(args.machine.resolve(), args.output.resolve(), args.crashes)
        print(msgspec.json.encode(evidence).decode())


if __name__ == "__main__":
    main()
