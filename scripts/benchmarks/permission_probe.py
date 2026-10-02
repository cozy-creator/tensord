"""CPU-only installed trampoline and filesystem probe; writes only a fresh fixture."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import selectors
import socket
import struct
import subprocess


@dataclass(frozen=True)
class Config:
    python: str
    uid: int
    gid: int
    hold: str

    @classmethod
    def read(cls, path: Path) -> Config:
        value = json.loads(path.read_text())
        assert value["format"] == "cozy.permission.probe/1"
        result = cls(**{key: value[key] for key in cls.__annotations__})
        assert all(type(number) is int and number >= 0 for number in (result.uid, result.gid))
        assert Path(result.python).is_absolute() and Path(result.hold).is_absolute()
        return result


# Trusted probe code, not a package description or substitute executor implementation.
CHILD = r'''
import ctypes,fcntl,json,os,pathlib,socket,sys
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());checks=[]
def check(name, action):
 try: value=action();checks.append({'name':name,'ok':True,'value':value})
 except Exception as error: checks.append({'name':name,'ok':False,'error':str(error),'errno':getattr(error,'errno',None)})
def fresh_hold():
 with open(cfg['hold'],'rb') as file:fcntl.flock(file,fcntl.LOCK_SH|fcntl.LOCK_NB)
 return 'shared hold acquired'
def inherited_hold():
 fd=cfg['hold_fd'];fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
 return {'inode':os.fstat(fd).st_ino,'cloexec':bool(fcntl.fcntl(fd,fcntl.F_GETFD)&fcntl.FD_CLOEXEC)}
def spool():
 path=pathlib.Path(cfg['spool'])/'result.json';path.write_text('{"probe":true}\n')
 with path.open('rb') as file:os.fsync(file.fileno())
 return str(path)
check('fresh_generation_hold',fresh_hold);check('inherited_generation_hold',inherited_hold)
check('peer_spool_write',spool)
check('private_journal_read',lambda:pathlib.Path(cfg['private']).read_bytes().decode())
check('legacy_store_open',lambda:__import__('tensorfs').Store.open(cfg['legacy_store']).root)
libc=ctypes.CDLL(None);death=ctypes.c_int();assert libc.prctl(2,ctypes.byref(death),0,0,0)==0
status=pathlib.Path('/proc/self/status').read_text().splitlines()
before=death.value
if os.geteuid()==0:
 libc.setfsuid(cfg['fsuid_probe']);assert libc.prctl(2,ctypes.byref(death),0,0,0)==0
 after=death.value;libc.setfsuid(0)
else:after=None
result={'format':'cozy.permission.result/1','uid':os.getuid(),'euid':os.geteuid(),'gid':os.getegid(),'groups':os.getgroups(),'pid':os.getpid(),'parent':os.getppid(),'status':{row.split(':')[0]:row.split(':',1)[1].strip() for row in status if row.startswith(('Uid:','Gid:','NoNewPrivs:'))},'parent_death_signal':before,'parent_death_after_foreign_fsuid':after,'checks':checks,'gpu_library_mapped':any(name in pathlib.Path('/proc/self/maps').read_text() for name in ('libcuda.so','libnvidia-ml.so'))}
peer=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);peer.connect(cfg['socket']);peer.sendall(json.dumps(result).encode());peer.shutdown(socket.SHUT_WR);peer.close()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = Config.read(args.config)
    root = args.output.absolute()
    root.mkdir(mode=0o711)  # Existing ancestors are never chowned/chmodded.
    if os.geteuid() != 0 and (config.uid, config.gid) != (os.geteuid(), os.getegid()):
        (root / "result.json").write_text(json.dumps({
            "qualified": False, "reason": "foreign identity requires an explicit privileged CPU fixture",
            "config": asdict(config)}, indent=2) + "\n")
        return
    os.chmod(root, 0o711)
    peer = root / "peer"
    peer.mkdir(mode=0o700)
    if os.geteuid() == 0:
        os.chown(peer, config.uid, config.gid)
    private = root / "private"
    private.mkdir(mode=0o700)
    (private / "journal").write_text("owner-private\n")
    legacy = root / "legacy-store"
    # Only this new owned store may be opened/mutated by the fixture.
    init = subprocess.run([config.python, "-I", "-c",
                           "import sys,tensorfs;tensorfs.Store.init(sys.argv[1])", str(legacy)],
                          capture_output=True)
    (root / "store-init.stderr").write_bytes(init.stderr)
    if init.returncode:
        raise RuntimeError("owned native store initialization failed; preserve stderr")
    os.chmod(legacy, 0o700)
    child = root / "probe-child.py"
    child.write_text(CHILD)
    os.chmod(child, 0o444)
    hold = open(config.hold, "rb")
    fcntl.flock(hold, fcntl.LOCK_SH)
    endpoint = root / "peer.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    os.chmod(endpoint, 0o660)
    if os.geteuid() == 0:
        os.chown(endpoint, os.geteuid(), config.gid)
    listener.listen(1)
    request = {**asdict(config), "hold_fd": hold.fileno(), "spool": str(peer),
               "private": str(private / "journal"), "legacy_store": str(legacy),
               "socket": str(endpoint), "fsuid_probe": config.uid if config.uid else 12345}
    wire = root / "child-config.json"
    wire.write_text(json.dumps(request))
    os.chmod(wire, 0o444)
    command = [config.python, "-I", "-m", "cozy_runtime.internal.trampoline",
               "--expect-parent", str(os.getpid()), "--oom-adj", "1000",
               "--scope-backend", "inherit", "--uid", str(config.uid),
               "--gid", str(config.gid), "--", config.python, "-I", str(child), str(wire)]
    with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                                   pass_fds=(hold.fileno(),), stdout=stdout, stderr=stderr)
        pidfd = os.pidfd_open(process.pid)
        observed = selectors.DefaultSelector()
        observed.register(listener, selectors.EVENT_READ, "peer")
        observed.register(pidfd, selectors.EVENT_READ, "exit")
        events = observed.select()  # Exact exit or socket activity; no elapsed-time kill.
        if any(key.data == "peer" for key, _ in events):
            connection, _ = listener.accept()
            credential = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            data = b""
            while chunk := connection.recv(65536):
                data += chunk
                assert len(data) <= 65536
            result = json.loads(data)
            result["socket_peer_matches"] = credential == (process.pid, config.uid, config.gid)
            connection.close()
        else:
            result = {"qualified": False, "reason": "installed trampoline/interpreter exited before probe connection"}
        result["exit_code"] = process.wait()
        result["configured_identity"] = asdict(config)
        result["identity_is_measured_old_serving_identity"] = False
        result["gpu_devices_present"] = list(map(str, Path("/dev").glob("nvidia*")))
        result["owner_can_read_peer_output"] = (peer / "result.json").exists()
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        observed.close()
        os.close(pidfd)
    listener.close()
    hold.close()
    print(json.dumps({"result": str(root / "result.json"), "exit_code": result["exit_code"]}))


if __name__ == "__main__":
    main()
