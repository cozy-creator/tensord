"""Linux descriptor primitives on interpreters built without optional os APIs."""
import ctypes
import errno
import os

MFD_CLOEXEC = getattr(os, "MFD_CLOEXEC", 0x0001)
MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)

_libc = ctypes.CDLL(None, use_errno=True)
_libc_memfd_create = getattr(_libc, "memfd_create", None)
if _libc_memfd_create is not None:
    _libc_memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    _libc_memfd_create.restype = ctypes.c_int
_os_memfd_create = getattr(os, "memfd_create", None)


def libc_memfd_create(name: str, flags: int) -> int:
    if _libc_memfd_create is None:
        raise OSError(errno.ENOSYS, "libc does not expose memfd_create")
    encoded = os.fsencode(name)
    if b"\0" in encoded:
        raise ValueError("memfd name contains a NUL byte")
    fd = _libc_memfd_create(encoded, flags)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), name)
    return fd


def memfd_create(name: str, flags: int = MFD_CLOEXEC | MFD_ALLOW_SEALING) -> int:
    if _os_memfd_create is not None:
        return _os_memfd_create(name, flags)
    return libc_memfd_create(name, flags)
