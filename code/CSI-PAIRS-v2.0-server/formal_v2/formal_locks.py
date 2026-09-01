"""Advisory file locks with a Windows-safe import path.

POSIX uses fcntl. Windows cannot run formal evidence hosts, but the CLI and
tests must still import. The Windows fallback is process-local and keys by
the opened path (or inode), not by file descriptor, so two handles to the
same lock file still exclude each other. A closed/stale handle is treated
as released so crash-recovery tests and dead owners can proceed.
"""

from __future__ import annotations

import errno
import os
import threading

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

if _fcntl is not None:
    LOCK_EX = _fcntl.LOCK_EX
    LOCK_SH = _fcntl.LOCK_SH
    LOCK_UN = _fcntl.LOCK_UN
    LOCK_NB = _fcntl.LOCK_NB
else:
    LOCK_EX = 2
    LOCK_SH = 1
    LOCK_UN = 8
    LOCK_NB = 4

_local_locks: dict[str, int] = {}
_local_guard = threading.Lock()


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(int(fd))
    except OSError:
        return False
    return True


def _windows_final_path(fd: int) -> str | None:
    try:
        import ctypes
        import msvcrt
    except ImportError:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    get_final_path.restype = ctypes.c_uint
    handle = msvcrt.get_osfhandle(int(fd))
    length = get_final_path(handle, ctypes.create_unicode_buffer(0), 0, 0)
    if length == 0:
        return None
    buffer = ctypes.create_unicode_buffer(length + 1)
    written = get_final_path(handle, buffer, length + 1, 0)
    if written == 0:
        return None
    return os.path.normcase(os.path.realpath(buffer.value))


def _lock_identity(fd: int) -> str:
    fd = int(fd)
    if os.name == "nt":
        path = _windows_final_path(fd)
        if path:
            return f"path:{path}"
    else:
        proc = f"/proc/self/fd/{fd}"
        try:
            return f"path:{os.path.realpath(proc)}"
        except OSError:
            pass
    try:
        stat = os.fstat(fd)
    except OSError:
        return f"fd:{fd}"
    if int(getattr(stat, "st_ino", 0) or 0) > 0:
        return f"ino:{stat.st_dev}:{stat.st_ino}"
    return f"fd:{fd}"


def flock(fd: int, flags: int) -> None:
    if _fcntl is not None:
        _fcntl.flock(fd, flags)
        return
    identity = _lock_identity(fd)
    with _local_guard:
        if flags & LOCK_UN:
            current = _local_locks.get(identity)
            if current == int(fd):
                _local_locks.pop(identity, None)
            return
        current = _local_locks.get(identity)
        if current is not None and current != int(fd):
            if _fd_is_open(current):
                if flags & LOCK_NB:
                    raise BlockingIOError(
                        errno.EAGAIN, "non-blocking flock would block"
                    )
                raise BlockingIOError(errno.EAGAIN, "flock would block")
            _local_locks.pop(identity, None)
        _local_locks[identity] = int(fd)


def posix_locks_available() -> bool:
    return _fcntl is not None
