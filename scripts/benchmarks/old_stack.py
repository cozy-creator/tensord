#!/usr/bin/env python3
"""Isolated released Go-agent/Python-worker benchmark; run is an explicit GPU action.

prepare/inspect are CPU-only. No Cozy controller home, Hub registration, source overlay,
SDK upgrade, CUDA probe, or package import is used by those actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def versions(python: str) -> dict[str, str]:
    return json.loads(subprocess.check_output([
        python, "-I", "-c", "import importlib.metadata as m,json; "
        "print(json.dumps({x:m.version(x) for x in ('cozy-runtime','tensorfs','sdxl')}))",
    ], text=True))


def inspect(pilot: dict) -> dict:
    measured = versions(pilot["python"])
    if measured != {"cozy-runtime": "0.18.99", "tensorfs": "0.3.90", "sdxl": "2.4.0"}:
        raise ValueError(f"benchmark package selection changed: {measured}")
    code = (
        "import tensorfs,json; s=tensorfs.Store.open(" + repr(pilot["binding"]["store"]) + "); "
        "r=" + repr(pilot["binding"]["snapshot"]) + "; m=s.manifest(r)['manifest']; "
        "print(json.dumps({'manifest_length':len(m)})); "
        "s.verify_checkpoint_source('paul/sdxl',r,len(m))"
    )
    checked = subprocess.run([pilot["python"], "-I", "-c", code], capture_output=True, text=True)
    return {"versions": measured, "gpu_started": False,
            "old_worker_checkpoint_cache_qualified": checked.returncode == 0,
            "cache_probe_stdout": checked.stdout, "cache_probe_stderr": checked.stderr,
            "requests": len(pilot["payloads"]), "payloads": pilot["payloads"],
            "snapshot": pilot["binding"]["snapshot"], "store": pilot["binding"]["store"]}


def prepare(pilot_path: Path, output: Path, port: int) -> None:
    pilot = json.loads(pilot_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    evidence = inspect(pilot)
    write(output / "cpu-inspection.json", evidence)
    root = output / "root"
    sdk = Path(pilot["python"]).parent.parent
    for directory in ("opt/cozy/bin", "usr/local/bin", "etc/cozy", "var/lib/cozy",
                      "var/lib/cozy/installs/.stage", "run/cozy/bootstrap", "tmp"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for path, target in (("opt/cozy/python", sdk),
                         ("opt/cozy/bin/cozy-runtime-worker", sdk / "bin/cozy-runtime-worker"),
                         ("usr/local/bin/tfs", sdk / "bin/tfs"),
                         ("usr/local/bin/uv", Path(shutil.which("uv") or "/usr/bin/uv"))):
        if not target.exists():
            raise ValueError(f"required released executable absent: {target}")
        (root / path).symlink_to(target)
    write(root / "etc/cozy/software-policy.json", {"startup_update": "off", "agent": "bundled"})
    # SDK's own supported existing-environment retention: the wheel/dependency closure
    # is preinstalled, shared unchanged with the pilot, not freshly solved for this run.
    installation_id = subprocess.check_output([pilot["python"], "-I", "-c",
                    "from pathlib import Path; from cozy_runtime.internal.package_installation "
                    "import retain_environment; held=retain_environment(Path(" + repr(str(root / "var/lib/cozy/installs")) + "),"
                    "Path(" + repr(pilot["python"]) + "),package='paul/sdxl',release='2.4.0',"
                    "); print(held.installation_id)"], text=True).strip()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    import base64
    key = Ed25519PrivateKey.generate()
    secret = output / "controller-key.pem"
    secret.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
    secret.chmod(0o600)
    public = base64.urlsafe_b64encode(key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode().rstrip("=")
    worker = "old-stack-a40-sdk99-baseline"
    env = {**pilot["environment"], "COZY_MACHINE_ROOT": str(root),
           "COZY_MACHINE_LIFETIME": "persistent", "COZY_TENSORFS_ROOT": pilot["binding"]["store"],
           "COZY_LISTEN_HOST": "127.0.0.1", "COZY_WORKER_ID": worker,
           "COZY_WORKER_INTERNAL_PORT": str(port), "COZY_AUTHORIZED_KEYS": public}
    if os.geteuid() == 0:
        # Only newly owned benchmark metadata is changed. SDK and store are untouched here.
        for directory in (root / "var/lib/cozy", root / "run/cozy"):
            for path, dirs, files in os.walk(directory):
                os.chown(path, 65532, 65532)
                for name in files:
                    os.chown(Path(path) / name, 65532, 65532)
    (root / "tmp").chmod(0o1777)
    write(output / "benchmark.json", {"agent": str(sdk / "bin/cozy-machine"), "root": str(root),
          "environment": env, "port": port, "worker": worker, "pilot": str(pilot_path),
          "controller_key": str(secret), "installation_id": installation_id})
    print(json.dumps({"prepared": True, "gpu_started": False, "config": str(output / "benchmark.json"),
                      "checkpoint_cache_qualified": evidence["old_worker_checkpoint_cache_qualified"]}))


def full(config_path: Path) -> None:
    """Real released PodHost -> worker -> stock executor, never an ordinary CLI gate."""
    import grpc
    from cryptography.hazmat.primitives import serialization
    from cozy_runtime.protocol import documents, worker_pb2 as pb, worker_pb2_grpc as rpc
    from cozy_runtime.internal import canonical
    from PIL import Image
    cfg = json.loads(config_path.read_text())
    pilot = json.loads(Path(cfg["pilot"]).read_text())
    output = config_path.parent
    key = serialization.load_pem_private_key(Path(cfg["controller_key"]).read_bytes(), password=None)
    rows, timings = [], {}
    evidence = {"qualification": "full released Go agent + Python worker RPC benchmark pending",
                "ordinary_cli_qualified": False, "versions": versions(pilot["python"]),
                "preinstalled_shared_environment": True, "cold_start_excludes": [
                    "pod provisioning/boot", "wheel/dependency installation", "model downloads",
                    "filesystem page-cache eviction"], "runs": rows, "timings": timings}
    started = time.perf_counter()
    log = (output / "agent.log").open("wb")
    agent = subprocess.Popen([cfg["agent"], "run"], env=cfg["environment"], stdout=log, stderr=log)
    evidence["agent_pid"] = agent.pid
    write(output / "results.json", evidence)
    # A bounded observer deadline never signals the agent or cancels accepted work.
    observation_end = time.monotonic() + 180
    root = Path(cfg["root"])
    channel = None
    try:
        while not (root / "run/cozy/bootstrap/tls.crt").exists():
            if agent.poll() is not None:
                raise RuntimeError("agent exited before identity; see agent.log")
            if time.monotonic() > observation_end:
                raise RuntimeError("identity observer budget exhausted; agent left running")
            time.sleep(0.05)
        cert = (root / "run/cozy/bootstrap/tls.crt").read_bytes()
        leaf = __import__("ssl").PEM_cert_to_DER_cert(cert.decode())
        boot = (root / "run/cozy/bootstrap/pod-boot-id").read_text().strip()
        proof, _ = documents.identity(pb.ClaimProof(record_owner_epoch=1, worker_id=cfg["worker"],
            worker_boot_id=boot, worker_tls_certificate_digest=hashlib.sha256(leaf).digest()))
        claim = pb.Claim(record_owner_epoch=1, record_owner_id="old-baseline", worker_id=cfg["worker"],
                         worker_boot_id=boot, wire_minor=70, proof=key.sign(proof))
        channel = grpc.secure_channel(f"127.0.0.1:{cfg['port']}", grpc.ssl_channel_credentials(cert),
                                      options=(("grpc.ssl_target_name_override", "cozy-worker"),))
        host = rpc.PodHostStub(channel)
        while True:
            try:
                workspace = host.GetMachineExecutionWorkspace(pb.MachineExecutionWorkspaceQuery(claim=claim), timeout=2)
                break
            except grpc.RpcError as error:
                if error.code() not in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    raise
                if agent.poll() is not None or time.monotonic() > observation_end:
                    raise RuntimeError("workspace observer budget exhausted; agent left running") from error
                time.sleep(0.05)
        timings["agent_start_to_workspace_ready_ms"] = (time.perf_counter() - started) * 1000
        write(output / "description.json", documents.body(host.DescribeMachine(pb.DescribeMachineQuery(claim=claim))))
        wheel = Path(cfg["pilot"]).parent / "sdxl-2.4.0-py3-none-any.whl"
        file = pb.LocalPackageFileRef(filename=wheel.name, digest=hashlib.sha256(wheel.read_bytes()).digest(), length=wheel.stat().st_size)
        header = pb.LocalPackageUploadHeader(claim=claim, operation_id="old-baseline-sdk99", file=file)
        def upload():
            yield pb.LocalPackageUploadFrame(header=header)
            raw = wheel.read_bytes()
            for offset in range(0, len(raw), 1 << 20):
                yield pb.LocalPackageUploadFrame(chunk=pb.LocalPackageUploadChunk(offset=offset, data=raw[offset:offset + (1 << 20)]))
        statuses = list(host.LocalPackageUpload(upload()))
        if not statuses or statuses[-1].state != pb.LOCAL_PACKAGE_FILE_STATE_VERIFIED:
            raise RuntimeError("wheel upload did not verify")
        def prepared(stream):
            last = None
            for event in stream:
                last = event
                with (output / "preparation.jsonl").open("a") as sink:
                    sink.write(json.dumps(documents.body(event)) + "\n")
            if last is None or not last.HasField("placement_set"):
                raise RuntimeError(f"preparation refused: {last}")
            return last
        began = time.perf_counter()
        installed = prepared(host.PrepareLocalPackage(pb.PrepareLocalPackageCall(claim=claim,
            local_package_set=pb.DesiredLocalPackageSet(operation_id=header.operation_id,
                package=pb.DevelopmentPackage(package="paul/sdxl", release="2.4.0", installation_id=cfg["installation_id"]),
                files=[file], python_requires=">=3.12", python_version="3.12"))))
        timings["reused_package_prepare_ms"] = (time.perf_counter() - began) * 1000
        snapshot = pilot["binding"]["snapshot"]
        delegated, _ = documents.identity(pb.DownloadDelegation(expires_at_unix=int(time.time()) + 3600,
            worker_id=cfg["worker"], worker_boot_id=boot, worker_tls_certificate_digest=hashlib.sha256(leaf).digest(),
            models=[pb.DownloadModelRef(package="paul/sdxl", slot="generate.models.model", model="paul/sdxl",
                                       release="1.0.0", manifest=snapshot)]))
        began = time.perf_counter()
        prepared(host.PreparePrivatePlacement(pb.PreparePrivatePlacementCall(claim=claim,
            private_placement_set=pb.DesiredPrivatePlacementSet(operation_id=header.operation_id,
                installation_id=cfg["installation_id"], download_delegation=delegated,
                download_delegation_signature=key.sign(delegated)))))
        timings["model_placement_prepare_ms"] = (time.perf_counter() - began) * 1000
        # Actual same checkpoint; length is observed with released TensorFS, not invented.
        import tensorfs
        manifest_length = len(tensorfs.Store.open(pilot["binding"]["store"]).manifest(snapshot)["manifest"])
        for index, payload in enumerate(pilot["payloads"]):
            began = time.perf_counter()
            request = f"old-baseline-{index}"
            submission = pb.MachineExecutionSubmit(claim=claim, submission_id=request,
                offer=pb.AttemptOffer(request_id=request), expected_execution_workspace_id=workspace.execution_workspace_id,
                payload_canonical_bytes=canonical.write(payload), release_root=pb.ReleaseRoot(package="paul/sdxl",
                    installation_id=cfg["installation_id"], entrypoint="generate", models=[pb.ModelChoice(parameter="model",
                        repository="paul/sdxl", release="1.0.0", manifest=pb.Ref(digest=bytes.fromhex(snapshot[7:]), length=manifest_length))]))
            while True:
                try:
                    receipt = host.SubmitMachineExecution(submission)
                    break
                except grpc.RpcError as error:
                    metadata = dict(error.trailing_metadata() or ())
                    if error.code() != grpc.StatusCode.UNAVAILABLE or metadata.get("cozy-error-code") != "release_root_preparing":
                        raise
                    with (output / "submission-progress.jsonl").open("a") as sink:
                        sink.write(json.dumps({"request": request, "details": error.details()}) + "\n")
                    time.sleep(0.05)
            query = pb.MachineExecutionQuery(claim=claim, request_id=receipt.request_id,
                                             expected_execution_workspace_id=workspace.execution_workspace_id)
            after, outcome, products = 0, None, []
            while outcome is None:
                page = host.ListMachineExecutionEvents(pb.MachineExecutionEventsQuery(execution=query, after=after, wait=True))
                for event in page.events:
                    with (output / "events.jsonl").open("a") as sink:
                        sink.write(json.dumps(documents.body(event)) + "\n")
                    if event.HasField("product"):
                        products.append(event.product)
                    if event.HasField("outcome"):
                        outcome = event.outcome
                after = page.next_after
            if hashlib.sha256(outcome.outcome_canonical_bytes).digest() != outcome.outcome_digest:
                raise RuntimeError("terminal checksum differs")
            terminal = documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            if terminal.status != pb.OUTCOME_STATUS_SUCCEEDED:
                raise RuntimeError(f"run failed: {terminal}")
            run = output / request
            run.mkdir()
            (run / "outcome.json").write_bytes(outcome.outcome_canonical_bytes)
            saved = []
            for product_index, product in enumerate(products):
                if product.parts or not product.HasField("source"):
                    raise RuntimeError("baseline expects an ordinary single SDXL image source")
                path = run / f"product-{product_index}.png"
                digest, length = hashlib.sha256(), 0
                with path.open("wb") as sink:
                    for chunk in host.ReadByteTreeObject(pb.NativeByteReadCall(claim=claim, source=product.source, object=product.content)):
                        if chunk.offset != length:
                            raise RuntimeError("native output cursor differs")
                        sink.write(chunk.data); digest.update(chunk.data); length += len(chunk.data)
                if length != product.content.length or digest.digest() != product.content.digest:
                    raise RuntimeError("native output checksum differs")
                saved.append({"path": str(path), "length": length, "sha256": digest.hexdigest()})
            if not saved:
                raise RuntimeError("successful run carried no verified image")
            rows.append({"request": request, "wall_ms": (time.perf_counter() - began) * 1000,
                         "outcome": documents.body(terminal), "outputs": saved})
            write(output / "results.json", evidence)
        for row in rows:
            for saved in row["outputs"]:
                with Image.open(saved["path"]) as image:
                    image.load()
                    if image.size != (1024, 1024):
                        raise RuntimeError("request image dimensions changed")
        evidence["qualification"] = "full released Go agent + Python worker + stock Executor; RPC only"
        write(output / "results.json", evidence)
        agent.terminate()  # Explicit completed benchmark cleanup; no request is left running.
        agent.wait()
    except Exception as error:
        evidence["failure"] = f"{type(error).__name__}: {error}"
        evidence["qualification"] = "full old-stack gate incomplete; no architecture comparison claimed"
        evidence["agent_left_running"] = agent.poll() is None
        write(output / "results.json", evidence)
        raise
    finally:
        if channel is not None:
            channel.close()
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "prepare", "run"))
    parser.add_argument("path", type=Path, help="pilot JSON for inspect/prepare; benchmark.json for run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=18443)
    args = parser.parse_args()
    if args.action == "inspect":
        print(json.dumps(inspect(json.loads(args.path.read_text())), indent=2))
    elif args.action == "prepare":
        if args.output is None or not args.output.is_absolute():
            parser.error("prepare requires an absolute, new --output directory")
        prepare(args.path.resolve(), args.output, args.port)
    else:
        full(args.path.resolve())


if __name__ == "__main__":
    main()
