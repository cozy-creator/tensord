#!/usr/bin/env python3
"""Matched old-stack vs Rust-machine gate on one rental, through ordinary `cozy run`.

  gate.py run MANIFEST OUT    alternate arms per the manifest; append to OUT/results.jsonl
  gate.py report OUT          medians, spread and the pre-declared verdict -> OUT/summary.json

Runs on the controller (system python3 + Pillow); reaches the pod with `cozy rental ssh-info`.
See README.md for the method, the manifest and what the numbers do and do not show.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageStat

POD_DIR = "/root/gate"
SCENARIOS = ("cold_first", "warm_first", "warm", "to_anima", "to_sdxl", "kill_next", "kill_proof")
SUBJECTS = ("a lighthouse", "a red fox", "an old tram", "a glass teapot", "a mountain hut",
            "a koi pond", "a violin", "a desert caravan", "a paper crane", "a clock tower",
            "a sailboat", "a snow owl", "a market stall", "a bonsai tree", "a steam train")
STYLES = ("at dawn, watercolor", "in fog, oil painting", "at noon, photograph", "under neon, ink",
          "in autumn rain, gouache", "at dusk, film still", "in winter light, pastel")

# Pod side: stdlib only, observation plus page-cache control. It signals nothing.
POD_HELPER = r'''
import json, os, sys, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
def cg():   # cgroup v2, else v1 (v1 rss = anon without shmem; shmem sits inside cache)
    try: CG = open(os.path.join(HERE, "cgroup")).read().strip()   # the measured arm's unit (local)
    except OSError: CG = "/sys/fs/cgroup"                          # the whole container (rental)
    if not os.path.isdir(CG): CG = "/sys/fs/cgroup"                # the arm's unit is between runs
    if os.path.exists(f"{CG}/memory.stat"):
        m = {l.split()[0]: int(l.split()[1]) for l in open(f"{CG}/memory.stat")}
        r = sum(int(f.split("=")[1]) for l in open(f"{CG}/io.stat") for f in l.split() if f.startswith("rbytes=")) \
            if os.path.exists(f"{CG}/io.stat") else unit_reads(CG)
        anon, shmem, mapped, file = m["anon"], m["shmem"], m["file_mapped"], m["file"]
    else:
        m = {l.split()[0]: int(l.split()[1]) for l in open(f"{CG}/memory/memory.stat")}
        r = sum(int(l.split()[2]) for l in open(f"{CG}/blkio/blkio.throttle.io_service_bytes_recursive")
                if len(l.split()) == 3 and l.split()[1] == "Read")
        anon, shmem, mapped, file = m["total_rss"], m["total_shmem"], m["total_mapped_file"], m["total_cache"]
    return {"anon": anon, "shmem": shmem, "file_mapped": mapped, "file": file, "host": anon + shmem, "read_bytes": r, "path": CG}
def unit_reads(cg):   # no io controller (a user unit): the storage reads of the unit's live processes
    total = 0
    for d, _, fs in os.walk(cg):
        if "cgroup.procs" in fs:
            for pid in open(os.path.join(d, "cgroup.procs")).read().split():
                try: total += next(int(l.split()[1]) for l in open(f"/proc/{pid}/io") if l.startswith("read_bytes"))
                except (OSError, StopIteration): pass
    return total
def load():
    try: psi = float(open("/proc/pressure/cpu").read().split()[1].split("=")[1])
    except (OSError, IndexError, ValueError): psi = None
    return {"load1": os.getloadavg()[0], "cpu_some_avg10": psi}
def cmd(pid):
    try: return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError: return ""
def started(pid):
    ticks = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19])
    boot = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
    return boot + ticks / os.sysconf("SC_CLK_TCK")
def executors():
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and "cozy_runtime.internal.executor" in cmd(d):
            st = {l.split(":")[0]: l.split(":")[1].split() for l in open(f"/proc/{d}/status")}
            out.append({"pid": int(d), "ppid": int(st["PPid"][0]), "uid": int(st["Uid"][0]), "started": started(d),
                        "rss": int(st["VmRSS"][0]) * 1024, "cmd": cmd(d)[:200]})
    return out
def gpu():
    q = "memory.used,temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_throttle_reasons.active"
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"], capture_output=True, text=True)
    out = dict(zip(("mem_mib", "temp_c", "sm_mhz", "power_w", "util", "throttle"), [x.strip() for x in r.stdout.split(",")]))
    for k in ("mem_mib", "temp_c", "sm_mhz", "power_w", "util"):
        try: out[k] = float(out[k])
        except (KeyError, ValueError): out[k] = None
    return out
def files(paths):   # weights and native libraries; small .py files stay as they are
    for root in paths:
        for d, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(d, f)
                if not os.path.islink(p) and os.path.isfile(p) and os.path.getsize(p) >= 1 << 20: yield p
act = sys.argv[1]
if act == "sample":
    with open(sys.argv[2], "a") as out:
        while True:
            t = time.time(); ex = executors()
            out.write(json.dumps({"t": t, "gpu": gpu(), "cg": cg(), "load": load(), "executors": len(ex)}) + "\n"); out.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - t)))
elif act == "now":
    print(json.dumps({"t": time.time(), "cg": cg(), "gpu": gpu(), "load": load(), "executors": executors()}))
elif act == "newroot":   # block until the root command names a live process other than OLD
    old, root = sys.argv[2], sys.argv[3]
    while True:
        pid = subprocess.run(root, shell=True, capture_output=True, text=True).stdout.split()
        if pid and pid[0] != old:
            print(json.dumps({"pid": int(pid[0]), "started": started(pid[0]), "cmd": cmd(pid[0])})); break
        time.sleep(0.05)
elif act == "list":
    open(sys.argv[3], "w").write("\n".join(files(json.loads(sys.argv[2]))))
    print(json.dumps({"files": len(open(sys.argv[3]).read().split())}))
elif act in ("evict", "warm"):
    n = b = 0; t = time.time()
    for p in open(sys.argv[2]).read().split("\n"):
        try:
            fd = os.open(p, os.O_RDONLY)
            if act == "evict": os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            else:
                while chunk := os.read(fd, 8 << 20): b += len(chunk)
            os.close(fd); n += 1
        except OSError: pass
    print(json.dumps({"files": n, "bytes_read": b, "s": time.time() - t, "t_end": time.time()}))
'''


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


class Host:
    """Where the machine runs: a rental (root ssh via the ordinary CLI's ssh-info) or this computer."""

    def __init__(self, m: dict):
        self.dir = m.get("work_dir", POD_DIR)
        if m.get("rental"):
            info = json.loads(subprocess.check_output(["cozy", "rental", "ssh-info", m["rental"], "--json", f"--tensorhub={m['hub']}"]))
            host, port = info["ssh_address"].rsplit(":", 1)
            self.ssh = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
                        "-o", "ControlPath=/tmp/gate-ssh-%C", "-o", "ControlMaster=auto", "-o", "ControlPersist=900",
                        f"root@{host}"]
        else:
            self.ssh = ["bash", "-c"]
        self.sh(f"mkdir -p {self.dir}")
        subprocess.run(self.ssh + [f"cat > {self.dir}/pod.py"], input=POD_HELPER, text=True, check=True)
        self.py = self.sh("command -v python3").strip()

    def sh(self, command: str, check: bool = True) -> str:
        while True:   # sshd is the machine agent's child: it is briefly absent while an arm restarts
            done = subprocess.run(self.ssh + [command], capture_output=True, text=True)
            if done.returncode != 255 or self.ssh[0] != "ssh":
                break
            time.sleep(0.2)
        if check and done.returncode:
            raise RuntimeError(f"host command failed ({done.returncode}): {command}\n{done.stderr}")
        return done.stdout

    def helper(self, *args: str) -> dict:
        return json.loads(self.sh(" ".join([self.py, f"{self.dir}/pod.py", *map(shlex.quote, args)])))


def marks(events: str) -> dict:
    """First time of each event type. Machine events carry the pod's clock, client events the controller's."""
    out = {}
    for line in events.splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            at = re.sub(r"(\.\d{6})\d*", r"\1", event["at"].rstrip("Z"))
            out.setdefault(event["type"], datetime.datetime.fromisoformat(at).replace(tzinfo=datetime.timezone.utc).timestamp())
    return out


def verify(directory: Path, shape: list[int]) -> list[dict]:
    """Decode every saved image: declared size, not flat, hashed."""
    images = []
    for path in sorted(directory.glob("*")):
        data = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            std = ImageStat.Stat(image.convert("RGB")).stddev
            images.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                           "size": list(image.size), "stddev": [round(s, 2) for s in std],
                           "ok": list(image.size) == shape and min(std) > 2.0})
    return images


class Gate:
    def __init__(self, manifest: Path, out: Path):
        self.m = json.loads(manifest.read_text())
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(self.m, indent=2) + "\n")
        self.results = out / "results.jsonl"
        rows = [json.loads(line) for line in self.results.read_text().splitlines()] if self.results.exists() else []
        self.used = {r["prompt"] for r in rows if "prompt" in r} | {r["seed"] for r in rows if "seed" in r}
        self.index = len(rows)
        self.rng = random.Random(f"{self.m['salt']}-{self.index}")
        self.pod = Host(self.m)
        self.offset = 0.0
        self.cache = f"{self.pod.dir}/cache-files"
        self.idle_since = 0.0
        self.xid = threading.Event()
        if self.m.get("xid_watch"):
            threading.Thread(target=self.watch_xid, daemon=True).start()

    def watch_xid(self) -> None:
        """Any new NVRM Xid stops the test at once: both machines are stopped, the harness raises."""
        watch = subprocess.Popen(["journalctl", "-kf", "-n0", "-o", "short-iso"], stdout=subprocess.PIPE, text=True)
        for line in watch.stdout:
            if "NVRM: Xid" in line:
                self.xid.set()
                log("NVRM Xid, stopping:", line.strip())
                self.record({"event": "xid", "line": line.strip(), "t": time.time()})
                self.pod.sh(self.m["on_xid"], check=False)
                return

    def check(self) -> None:
        if self.xid.is_set():
            raise RuntimeError("stopped on an NVRM Xid")

    def record(self, row: dict) -> None:
        with self.results.open("a") as sink:
            sink.write(json.dumps(row) + "\n")
        self.index += 1

    def unique(self) -> tuple[str, int]:
        """A prompt and a seed never used before in OUT: no engine or memo cache can answer."""
        while True:
            prompt = f"{self.rng.choice(SUBJECTS)} {self.rng.choice(STYLES)}, study {self.rng.randrange(10**6)}"
            seed = self.rng.randrange(2**31)
            if prompt not in self.used and seed not in self.used:
                self.used |= {prompt, seed}
                return prompt, seed

    def clock(self) -> dict:
        """Pod-minus-controller clock offset from an ssh round trip (error <= rtt/2)."""
        before = time.time()
        now = self.pod.helper("now")
        after = time.time()
        self.offset = now["t"] - (before + after) / 2
        return {"offset_s": self.offset, "rtt_s": after - before}

    def cool(self) -> dict:
        """Equal thermal start: at or below the manifest's ceiling, or at the card's idle floor
        (no further cooling over 60 s; a card holding a CUDA context never cools past it)."""
        soak = self.m.get("soak_s", 0) - (time.time() - self.idle_since)
        if soak > 0:
            log(f"equal soak: {soak:.0f} s of GPU idle")
            time.sleep(soak)
        temps = []
        while True:
            self.check()
            now = self.pod.helper("now")
            temps.append(now["gpu"]["temp_c"])
            cool = temps[-1] <= self.m["start_temp_c"] or (len(temps) >= 12 and min(temps[-6:]) >= min(temps[-12:-6]))
            if cool and now["load"]["load1"] <= self.m.get("max_load", float("inf")):
                return now
            log("waiting for the GPU to cool", now["gpu"])
            time.sleep(5)

    def request(self, arm: str, cycle: int, scenario: str, model: str, t0: float | None = None,
                start: float | None = None, allow_fail: bool = False) -> dict:
        prompt, seed = self.unique()
        root = self.out / "runs" / f"{self.index:03}-{arm}-{cycle}-{scenario}-{model}"
        root.mkdir(parents=True)
        (root / "input.json").write_text(json.dumps({**self.m["requests"][model], "prompt": prompt, "seed": seed}))
        self.check()
        spec = self.m["arms"][arm]
        cli = spec.get("cli", "cozy")
        place = spec.get("run_args", [f"--rental={self.m.get('rental')}", f"--tensorhub={self.m.get('hub')}"])
        before = self.pod.helper("now")
        submit = time.time()
        done = subprocess.run([cli, "run", self.m["targets"][model], f"--input={root / 'input.json'}", *place,
                               "--await", "--json", f"--out={root / 'out'}",
                               f"--idempotency-key=gate-{self.m['salt']}-{self.index}"], capture_output=True, text=True)
        finished = time.time()
        self.idle_since = finished
        self.check()
        images = verify(root / "out", self.m["shapes"][model]) if (root / "out").is_dir() else []
        verified = time.time()
        after = self.pod.helper("now")
        (root / "stdout.json").write_text(done.stdout)
        (root / "events.jsonl").write_text(done.stderr)
        run = images[0]["path"].rsplit("/", 1)[1].split("-")[0] if images else None
        show = {}
        if run:
            shown = subprocess.run([cli, "run", "show", run, "--json", "--full",
                                    *spec.get("show_args", [f"--tensorhub={self.m.get('hub')}"])], capture_output=True, text=True)
            (root / "show.json").write_text(shown.stdout or shown.stderr)
            show = json.loads(shown.stdout) if shown.returncode == 0 else {}
        if start is None:   # t0 is a pod-clock process birth; start a controller-clock event (the kill)
            start = submit if t0 is None else t0 - self.offset
        seen = marks(done.stderr)
        accepted, outcome = seen.get("request.machine_accepted"), seen.get("machine.outcome")
        row = {"arm": arm, "cycle": cycle, "scenario": scenario, "model": model, "prompt": prompt, "seed": seed,
               "run": run, "exit": done.returncode, "dir": str(root), "runtime": show.get("runtime"),
               "t_start": start, "t_submit": submit, "t_cli_done": finished, "t_verified": verified,
               "wall_s": verified - start, "cli_s": finished - submit, "offset": self.offset,
               # Pod clock only, free of controller noise: machine birth (or acceptance) to outcome.
               "machine_s": (outcome - (t0 if t0 is not None else accepted)) if outcome and accepted else None,
               # Outcome on the pod until the client noticed it: the result path (request plane, daemon, network).
               "result_lag_s": (seen["client.machine_work_finished"] - (outcome - self.offset))
               if outcome and "client.machine_work_finished" in seen else None,
               "controller_load": os.getloadavg()[0],
               "execution_ms": show.get("execution_ms"), "stages": show.get("stages"), "steps": show.get("steps"),
               "disk_read_bytes": after["cg"]["read_bytes"] - before["cg"]["read_bytes"],
               "executors_after": after["executors"], "images": images,
               "ok": done.returncode == 0 and len(images) == 1 and images[0]["ok"] and show.get("status") == "completed"}
        self.record(row)
        log(f"{arm} c{cycle} {scenario:10} {model:5} {row['wall_s']:7.2f}s  disk {row['disk_read_bytes'] / 2**30:5.2f} GiB",
            "ok" if row["ok"] else "FAILED", run)
        if not row["ok"]:
            try:
                row["error"] = json.loads(done.stdout)["error"]["message"]
            except (json.JSONDecodeError, KeyError, TypeError):
                row["error"] = done.stdout[-500:] or done.stderr[-500:]
            self.record({"arm": arm, "cycle": cycle, "event": "failed", "scenario": scenario, "run": run,
                         "error": row["error"], "dir": str(root)})
            if not allow_fail:
                raise RuntimeError(f"request failed; evidence in {root}")
        return row

    def first(self, arm: str, cycle: int, scenario: str) -> None:
        """Stopped-machine first image: restart this arm's machine, clock from the new root process's birth.

        Warm reads the cache files before the restart; cold evicts them once the old machine's
        executors (which map the libraries) are gone. Measured disk reads prove which one held."""
        spec = self.m["arms"][arm]
        cache = self.pod.helper("warm", self.cache) if scenario == "warm_first" else None
        old = (self.pod.sh(spec["root"]).split() or ["none"])[0]
        stop = self.pod.helper("now")["t"]
        self.pod.sh(spec["restart"])
        while True:   # exclusive card: every executor of the previous machine is gone
            now = self.pod.helper("now")
            if not [e for e in now["executors"] if e["started"] < stop]:
                break
            time.sleep(0.2)
        if scenario == "cold_first":
            cache = self.pod.helper("evict", self.cache)
        if spec.get("start"):   # a platform that does not relaunch by itself (this computer)
            self.pod.sh(spec["start"])
        new = self.pod.helper("newroot", old, spec["root"])
        if spec.get("after_start"):
            self.pod.sh(spec["after_start"])
        if spec.get("cgroup"):
            self.pod.sh(f"({spec['cgroup']}) > {self.pod.dir}/cgroup")
        limits = self.pod.sh(spec["limits"]) if spec.get("limits") else None
        self.record({"arm": arm, "cycle": cycle, "event": "restart", "scenario": scenario, "old_root": old, "limits": limits,
                     "new_root": new, "gpu_after_stop": now["gpu"], "stop_to_root_s": new["started"] - stop,
                     "cache": cache, "cache_done_before_root": cache["t_end"] <= new["started"]})
        self.request(arm, cycle, scenario, "sdxl", t0=new["started"])

    def cycle(self, arm: str, cycle: int) -> None:
        if self.m.get("rental"):
            subprocess.run(["cozy", "rental", "keepalive", self.m["rental"], f"--tensorhub={self.m['hub']}"],
                           capture_output=True)
        listed = self.pod.helper("list", json.dumps(self.m["cache_paths"]), self.cache)   # after any download
        self.record({"arm": arm, "cycle": cycle, "event": "begin", "clock": self.clock(), "cool": self.cool(),
                     "cache_files": listed["files"]})
        if self.m.get("first_images") == "alternate":   # one restart per cycle: a cross-arm switch
            self.first(arm, cycle, "cold_first" if cycle // 2 % 2 == 0 else "warm_first")
        elif self.m.get("first_images") == "warm":
            self.first(arm, cycle, "warm_first")
        else:
            self.first(arm, cycle, "cold_first")
            self.first(arm, cycle, "warm_first")
        for _ in range(self.m["warm"]):
            self.request(arm, cycle, "warm", "sdxl")
        self.request(arm, cycle, "to_anima", "anima")
        self.request(arm, cycle, "to_sdxl", "sdxl")
        if not self.m.get("kill", True):   # fault injection stays on rentals (the owner's laptop)
            self.record({"arm": arm, "cycle": cycle, "event": "end", "now": self.pod.helper("now")})
            return
        victims = self.pod.helper("now")["executors"]
        if not victims:
            raise RuntimeError("no idle executor to kill; the kill scenario cannot be measured")
        self.record({"arm": arm, "cycle": cycle, "event": "kill", "executors": victims})
        killed = time.time()
        self.pod.sh("kill -9 " + " ".join(str(v["pid"]) for v in victims))   # exact PIDs listed above
        for _ in range(3):   # a failed attempt is a finding; the time runs from the kill to the first image
            if self.request(arm, cycle, "kill_next", "sdxl", start=killed, allow_fail=True)["ok"]:
                break
        else:
            raise RuntimeError("three requests after an executor kill failed")
        self.record({"arm": arm, "cycle": cycle, "event": "end", "now": self.pod.helper("now")})

    def prime(self, arm: str) -> None:
        """Unmeasured: switch to the arm and pay its package install and model download once."""
        spec = self.m["arms"][arm]
        live = self.pod.sh(spec["root"]).split()
        if live and not spec.get("start"):   # already this arm's machine: no same-arm restart
            self.record({"arm": arm, "cycle": -1, "event": "prime", "root": {"pid": int(live[0]), "already": True}})
        else:
            self.pod.sh(spec["restart"])
            if spec.get("start"):
                self.pod.sh(spec["start"])
            root = self.pod.helper("newroot", "none", spec["root"])
            if spec.get("after_start"):
                self.pod.sh(spec["after_start"])
            self.record({"arm": arm, "cycle": -1, "event": "prime", "root": root})
        self.request(arm, -1, "prime", "sdxl")
        self.request(arm, -1, "prime", "anima")

    def kill_proof(self, arm: str, n: int) -> None:
        """N times: SIGKILL every executor of the idle arm, submit SDXL at once; one attempt each, failures count."""
        cycle = 1000
        if not self.pod.sh(self.m["arms"][arm]["root"]).split():
            self.first(arm, cycle, "warm_first")
        failed = 0
        for i in range(n):
            victims = self.pod.helper("now")["executors"]
            self.record({"arm": arm, "cycle": cycle + i, "event": "kill", "executors": victims})
            killed = time.time()
            if victims:
                self.pod.sh("kill -9 " + " ".join(str(v["pid"]) for v in victims))   # exact PIDs listed above
            failed += not self.request(arm, cycle + i, "kill_proof", "sdxl", start=killed, allow_fail=True)["ok"]
        self.record({"arm": arm, "event": "kill_proof", "cycles": n, "failed": failed})
        log(f"kill proof on {arm}: {n} cycles, {failed} failed")

    def run(self) -> None:
        samples = f"{self.pod.dir}/samples-{self.m['salt']}.jsonl"
        sampler = self.pod.sh(f"nohup {self.pod.py} {POD_DIR}/pod.py sample {samples} >/dev/null 2>&1 & echo $!").strip()
        try:
            for arm in self.m.get("prime", []):
                self.prime(arm)
            for cycle, arm in enumerate(self.m["order"]):
                self.cycle(arm, cycle)
            if self.m.get("kill_proof"):
                self.kill_proof(self.m["kill_proof"]["arm"], self.m["kill_proof"]["n"])
        finally:
            self.pod.sh(f"kill {sampler}", check=False)   # the sampler PID this run started
            with (self.out / "samples.jsonl").open("w") as sink:
                subprocess.run(self.pod.ssh + [f"cat {samples}"], stdout=sink, check=False)


def spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"median": round(statistics.median(values), 3), "min": round(min(values), 3),
            "max": round(max(values), 3), "n": len(values)}


def report(out: Path) -> dict:
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines() if line.strip()]
    requests = [r for r in rows if "wall_s" in r]
    rental = bool(json.loads((out / "manifest.json").read_text()).get("rental"))
    summary: dict = {"failed": {a: [(r["scenario"], r["run"], r.get("error")) for r in requests if r["arm"] == a and not r["ok"]]
                                for a in sorted({r["arm"] for r in requests})}, "arms": {}}
    for arm in sorted({r["arm"] for r in requests}):
        mine = [r for r in requests if r["arm"] == arm and r["ok"]]
        cycles = sorted({r["cycle"] for r in mine if r["cycle"] >= 0})
        cell: dict = {"runtime": sorted({r["runtime"] for r in mine if r["runtime"]}), "cycles": cycles}
        for s in SCENARIOS:
            cell[s] = spread([r["wall_s"] for r in mine if r["scenario"] == s])
            cell[s + "_from_submit"] = spread([r["cli_s"] for r in mine if r["scenario"] == s])
            cell[s + "_result_lag"] = spread([r["result_lag_s"] for r in mine if r["scenario"] == s and r.get("result_lag_s") is not None])
            cell[s + "_machine"] = spread([r["machine_s"] for r in mine if r["scenario"] == s and r.get("machine_s")])
            cell[s + "_disk_gib"] = spread([r["disk_read_bytes"] / 2**30 for r in mine if r["scenario"] == s])
        cell["switch_pair"] = spread([sum(r["wall_s"] for r in mine if r["cycle"] == c and r["scenario"] in
                                          ("to_anima", "to_sdxl")) for c in cycles])
        cell["warm_denoise_s"] = spread([sum(x["ms"] for x in r["stages"] or [] if x["name"] == "denoise") / 1000
                                         for r in mine if r["scenario"] == "warm"])
        host, gpu, clocks = [], [], []
        for c in cycles:
            span = [r for r in mine if r["cycle"] == c]
            lo = min(r["t_start"] for r in span) + span[0]["offset"]
            hi = max(r["t_verified"] for r in span) + span[0]["offset"]
            window = [s for s in samples if lo <= s["t"] <= hi and (rental or s["cg"].get("path") != "/sys/fs/cgroup")]
            if window:
                host.append(max(s["cg"]["host"] for s in window) / 2**30)
                gpu.append(max(s["gpu"]["mem_mib"] or 0 for s in window) / 1024)
                clocks += [s["gpu"]["sm_mhz"] for s in window if (s["gpu"]["util"] or 0) > 50]
        cell["host_peak_gib"] = spread(host)
        cell["gpu_peak_gib"] = spread(gpu)
        cell["busy_sm_mhz"] = spread(clocks)
        summary["arms"][arm] = cell
    old, new = summary["arms"].get("old"), summary["arms"].get("rust")
    if old and new:
        def gain(key: str) -> float | None:
            return round(1 - new[key]["median"] / old[key]["median"], 4) if old.get(key) and new.get(key) else None
        v = {"first_image_gain": gain("warm_first"), "cold_first_gain": gain("cold_first"),
             "switch_gain": gain("switch_pair"), "host_peak_gain": gain("host_peak_gib"),
             "kill_next_gain": gain("kill_next"), "gpu_peak_gain": gain("gpu_peak_gib")}
        v["warm_gain"] = gain("warm")
        v["warm_regression"] = -v["warm_gain"] if v["warm_gain"] is not None else None
        benefit = (v["first_image_gain"] or 0) >= .20 or (v["switch_gain"] or 0) >= .20 or (v["host_peak_gain"] or 0) >= .25
        v["pass"] = bool(benefit and v["warm_regression"] is not None and v["warm_regression"] <= .02
                         and not summary["failed"].get("rust"))
        summary["verdict"] = v
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "run":
        Gate(Path(sys.argv[2]), Path(sys.argv[3])).run()
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        print(json.dumps(report(Path(sys.argv[2])), indent=2))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
