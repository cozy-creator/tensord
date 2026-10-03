"""Executor lifecycle fixture: cooperative steps, an uncooperative wedge, an abrupt exit."""
from __future__ import annotations

import os
import threading
import time

import msgspec

from cozy_runtime.author import App, Context, Telemetry

app = App()


class Steps(msgspec.Struct):
    steps: int = 3
    seconds: float = 0.05


class Done(msgspec.Struct):
    steps: int


@app.entrypoint
def steps(payload: Steps, ctx: Context, tel: Telemetry) -> Done:
    """Checks cancellation at every step and reports each one."""
    on_step = tel.step_callback(payload.steps, stage="steps")
    for index in range(payload.steps):
        ctx.raise_if_cancelled()
        time.sleep(payload.seconds)
        on_step(index)
    return Done(payload.steps)


@app.entrypoint
def wedge(payload: Steps, ctx: Context, tel: Telemetry) -> Done:
    """Reports one step, then blocks forever without checking cancellation."""
    tel.step_callback(1, stage="wedge")(0)
    threading.Event().wait()
    return Done(0)


@app.entrypoint
def exit_now(payload: Steps, ctx: Context) -> Done:
    """The process ends in the middle of a request."""
    os._exit(17)


class Probe(msgspec.Struct):
    paths: list[str] = []


class Reach(msgspec.Struct):
    uid: int
    gid: int
    groups: list[int]
    reach: dict[str, str]
    home_writable: bool
    fds: list[str]
    parent_death_signal: int = 0


@app.entrypoint
def probe(payload: Probe, ctx: Context) -> Reach:
    """What package code can reach: its identity, the named paths, its home, its fds."""
    reach = {}
    for path in payload.paths:
        try:
            if os.path.isdir(path):
                os.listdir(path)
            else:
                with open(path, "rb") as stream:
                    stream.read(1)
            reach[path] = "read"
        except PermissionError:
            reach[path] = "denied"
        except FileNotFoundError:
            reach[path] = "absent"
    home = os.environ.get("COZY_HOME", "")
    try:
        marker = os.path.join(home, "qualification-probe")
        with open(marker, "w") as stream:
            stream.write("ok")
        os.unlink(marker)
        writable = True
    except OSError:
        writable = False
    fds = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            fds.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            pass
    import ctypes
    signal = ctypes.c_int()
    ctypes.CDLL(None).prctl(2, ctypes.byref(signal), 0, 0, 0)  # PR_GET_PDEATHSIG
    return Reach(os.getuid(), os.getgid(), sorted(os.getgroups()), reach, writable, sorted(fds),
                 signal.value)


class Daemon(msgspec.Struct):
    pid: int


@app.entrypoint
def daemon(payload: Steps, ctx: Context) -> Daemon:
    """Leaves a process in its own session: outside the process group, inside the cgroup."""
    import subprocess
    child = subprocess.Popen(["sleep", "1000"], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    return Daemon(child.pid)
