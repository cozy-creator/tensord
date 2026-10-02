#!/usr/bin/env python3
"""Matched old-stack vs Rust-machine gate on one rental, through ordinary `cozy run`.

  gate.py run MANIFEST OUT    alternate arms per the manifest; append to OUT/results.jsonl
  gate.py report OUT          medians, spread and the pre-declared verdict -> OUT/summary.json

Runs on the controller (system python3 + Pillow); reaches the pod with `cozy rental ssh-info`.
See README.md for the method, the manifest and what the numbers do and do not show.
"""
from __future__ import annotations

import hashlib
import json
import random
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageStat

POD_DIR = "/root/gate"
SCENARIOS = ("cold_first", "warm_first", "warm", "to_anima", "to_sdxl", "kill_next")
SUBJECTS = ("a lighthouse", "a red fox", "an old tram", "a glass teapot", "a mountain hut",
            "a koi pond", "a violin", "a desert caravan", "a paper crane", "a clock tower",
            "a sailboat", "a snow owl", "a market stall", "a bonsai tree", "a steam train")
STYLES = ("at dawn, watercolor", "in fog, oil painting", "at noon, photograph", "under neon, ink",
          "in autumn rain, gouache", "at dusk, film still", "in winter light, pastel")

# Pod side: stdlib only, observation plus page-cache control. It signals nothing.
POD_HELPER = r'''
import json, os, sys, time, subprocess
CG = "/sys/fs/cgroup"
def cg():
    m = {l.split()[0]: int(l.split()[1]) for l in open(f"{CG}/memory.stat")}
    r = sum(int(f.split("=")[1]) for l in open(f"{CG}/io.stat") for f in l.split() if f.startswith("rbytes="))
    return {"anon": m["anon"], "shmem": m["shmem"], "file_mapped": m["file_mapped"], "file": m["file"],
            "host": m["anon"] + m["shmem"], "read_bytes": r}
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
            out.write(json.dumps({"t": t, "gpu": gpu(), "cg": cg(), "executors": len(ex)}) + "\n"); out.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - t)))
elif act == "now":
    print(json.dumps({"t": time.time(), "cg": cg(), "gpu": gpu(), "executors": executors()}))
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


class Pod:
    """Root shell on the rental through the ordinary CLI's ssh-info; one multiplexed connection."""

    def __init__(self, rental: str, hub: str):
        info = json.loads(subprocess.check_output(["cozy", "rental", "ssh-info", rental, "--json", f"--tensorhub={hub}"]))
        host, port = info["ssh_address"].rsplit(":", 1)
        self.ssh = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
                    "-o", "ControlPath=/tmp/gate-ssh-%C", "-o", "ControlMaster=auto", "-o", "ControlPersist=900",
                    f"root@{host}"]
        self.sh(f"mkdir -p {POD_DIR}")
        subprocess.run(self.ssh + [f"cat > {POD_DIR}/pod.py"], input=POD_HELPER, text=True, check=True)
        self.py = self.sh("command -v python3").strip()

    def sh(self, command: str, check: bool = True) -> str:
        while True:   # sshd is the machine agent's child: it is briefly absent while an arm restarts
            done = subprocess.run(self.ssh + [command], capture_output=True, text=True)
            if done.returncode != 255:
                break
            time.sleep(0.2)
        if check and done.returncode:
            raise RuntimeError(f"pod command failed ({done.returncode}): {command}\n{done.stderr}")
        return done.stdout

    def helper(self, *args: str) -> dict:
        return json.loads(self.sh(" ".join([self.py, f"{POD_DIR}/pod.py", *map(shlex.quote, args)])))


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
        self.pod = Pod(self.m["rental"], self.m["hub"])
        self.offset = 0.0
        self.cache = f"{POD_DIR}/cache-files"
        listed = self.pod.helper("list", json.dumps(self.m["cache_paths"]), self.cache)
        log("cache files >= 1 MiB:", listed["files"])

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
        """Equal thermal start: wait until the card is at or below the manifest's start temperature."""
        while True:
            now = self.pod.helper("now")
            if now["gpu"]["temp_c"] <= self.m["start_temp_c"]:
                return now
            log("waiting for the GPU to cool", now["gpu"])
            time.sleep(5)

    def request(self, arm: str, cycle: int, scenario: str, model: str, t0: float | None = None) -> dict:
        prompt, seed = self.unique()
        root = self.out / "runs" / f"{self.index:03}-{arm}-{cycle}-{scenario}-{model}"
        root.mkdir(parents=True)
        (root / "input.json").write_text(json.dumps({**self.m["requests"][model], "prompt": prompt, "seed": seed}))
        before = self.pod.helper("now")
        submit = time.time()
        done = subprocess.run(["cozy", "run", self.m["targets"][model], f"--input={root / 'input.json'}",
                               f"--rental={self.m['rental']}", f"--tensorhub={self.m['hub']}", "--await", "--json",
                               f"--out={root / 'out'}", f"--idempotency-key=gate-{self.m['salt']}-{self.index}"],
                              capture_output=True, text=True)
        finished = time.time()
        images = verify(root / "out", self.m["shapes"][model]) if (root / "out").is_dir() else []
        verified = time.time()
        after = self.pod.helper("now")
        (root / "stdout.json").write_text(done.stdout)
        (root / "events.jsonl").write_text(done.stderr)
        run = images[0]["path"].rsplit("/", 1)[1].split("-")[0] if images else None
        show = {}
        if run:
            shown = subprocess.run(["cozy", "run", "show", run, "--json", "--full", f"--tensorhub={self.m['hub']}"],
                                   capture_output=True, text=True)
            (root / "show.json").write_text(shown.stdout or shown.stderr)
            show = json.loads(shown.stdout) if shown.returncode == 0 else {}
        start = submit if t0 is None else t0 - self.offset   # t0 is a pod-clock process start time
        row = {"arm": arm, "cycle": cycle, "scenario": scenario, "model": model, "prompt": prompt, "seed": seed,
               "run": run, "exit": done.returncode, "dir": str(root), "runtime": show.get("runtime"),
               "t_start": start, "t_submit": submit, "t_cli_done": finished, "t_verified": verified,
               "wall_s": verified - start, "cli_s": finished - submit, "offset": self.offset,
               "execution_ms": show.get("execution_ms"), "stages": show.get("stages"), "steps": show.get("steps"),
               "disk_read_bytes": after["cg"]["read_bytes"] - before["cg"]["read_bytes"],
               "executors_after": after["executors"], "images": images,
               "ok": done.returncode == 0 and len(images) == 1 and images[0]["ok"] and show.get("status") == "completed"}
        self.record(row)
        log(f"{arm} c{cycle} {scenario:10} {model:5} {row['wall_s']:7.2f}s  disk {row['disk_read_bytes'] / 2**30:5.2f} GiB",
            "ok" if row["ok"] else "FAILED", run)
        if not row["ok"]:
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
        new = self.pod.helper("newroot", old, spec["root"])
        self.record({"arm": arm, "cycle": cycle, "event": "restart", "scenario": scenario, "old_root": old,
                     "new_root": new, "gpu_after_stop": now["gpu"], "stop_to_root_s": new["started"] - stop,
                     "cache": cache, "cache_done_before_root": cache["t_end"] <= new["started"]})
        self.request(arm, cycle, scenario, "sdxl", t0=new["started"])

    def cycle(self, arm: str, cycle: int) -> None:
        subprocess.run(["cozy", "rental", "keepalive", self.m["rental"], f"--tensorhub={self.m['hub']}"],
                       capture_output=True)
        self.record({"arm": arm, "cycle": cycle, "event": "begin", "clock": self.clock(), "cool": self.cool()})
        self.first(arm, cycle, "cold_first")
        self.first(arm, cycle, "warm_first")
        for _ in range(self.m["warm"]):
            self.request(arm, cycle, "warm", "sdxl")
        self.request(arm, cycle, "to_anima", "anima")
        self.request(arm, cycle, "to_sdxl", "sdxl")
        victims = self.pod.helper("now")["executors"]
        if not victims:
            raise RuntimeError("no idle executor to kill; the kill scenario cannot be measured")
        self.record({"arm": arm, "cycle": cycle, "event": "kill", "executors": victims})
        self.pod.sh("kill -9 " + " ".join(str(v["pid"]) for v in victims))   # exact PIDs listed above
        self.request(arm, cycle, "kill_next", "sdxl")
        self.record({"arm": arm, "cycle": cycle, "event": "end", "now": self.pod.helper("now")})

    def prime(self, arm: str) -> None:
        """Unmeasured: switch to the arm and pay its package install and model download once."""
        spec = self.m["arms"][arm]
        old = (self.pod.sh(spec["root"]).split() or ["none"])[0]
        self.pod.sh(spec["restart"])
        self.record({"arm": arm, "cycle": -1, "event": "prime", "root": self.pod.helper("newroot", old, spec["root"])})
        self.request(arm, -1, "prime", "sdxl")
        self.request(arm, -1, "prime", "anima")

    def run(self) -> None:
        samples = f"{POD_DIR}/samples-{self.m['salt']}.jsonl"
        sampler = self.pod.sh(f"nohup {self.pod.py} {POD_DIR}/pod.py sample {samples} >/dev/null 2>&1 & echo $!").strip()
        try:
            for arm in self.m.get("prime", []):
                self.prime(arm)
            for cycle, arm in enumerate(self.m["order"]):
                self.cycle(arm, cycle)
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
    summary: dict = {"failed": [r["dir"] for r in requests if not r["ok"]], "arms": {}}
    for arm in sorted({r["arm"] for r in requests}):
        mine = [r for r in requests if r["arm"] == arm and r["ok"]]
        cycles = sorted({r["cycle"] for r in mine if r["cycle"] >= 0})
        cell: dict = {"runtime": sorted({r["runtime"] for r in mine if r["runtime"]}), "cycles": cycles}
        for s in SCENARIOS:
            cell[s] = spread([r["wall_s"] for r in mine if r["scenario"] == s])
            cell[s + "_from_submit"] = spread([r["cli_s"] for r in mine if r["scenario"] == s])
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
            window = [s for s in samples if lo <= s["t"] <= hi]
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
                         and not summary["failed"])
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
