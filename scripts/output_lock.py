"""Fail-fast cross-thread and cross-process ownership for output paths."""

import errno
import fcntl
import hashlib
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union


PathValue = Union[str, os.PathLike]
_LOCK_DIRECTORY = Path(tempfile.gettempdir()) / "ai-image-to-ppt-output-locks"
_REGISTRY_GUARD = threading.Lock()
_THREAD_LOCKS = {}


class OutputLockError(OSError):
    """Raised when output ownership cannot be established safely."""


class OutputLockBusy(OutputLockError):
    """Raised when another live caller owns the same output."""


def _identity(target: PathValue, namespace: str) -> str:
    try:
        canonical = str(Path(target).expanduser().resolve(strict=False))
    except (TypeError, ValueError, OSError) as error:
        raise OutputLockError(f"invalid output lock target: {error}") from error
    return f"{namespace}\0{canonical}"


def _lock_file_for_identity(identity: str) -> Path:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return _LOCK_DIRECTORY / f"{digest}.lock"


def lock_file_for(target: PathValue, namespace: str = "output") -> Path:
    """Return the stable lock-file path for an output ownership key."""
    return _lock_file_for_identity(_identity(target, namespace))


def _thread_lock(identity: str) -> threading.Lock:
    with _REGISTRY_GUARD:
        lock = _THREAD_LOCKS.get(identity)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[identity] = lock
        return lock


def _open_lock_file(path: Path) -> int:
    try:
        _LOCK_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        return os.open(str(path), flags, 0o600)
    except OSError as error:
        raise OutputLockError(f"cannot open output lock: {error}") from error


@contextmanager
def output_lock(
    target: PathValue,
    namespace: str = "output",
) -> Iterator[None]:
    """Own one output key until the context exits, failing fast if it is busy."""
    identity = _identity(target, namespace)
    thread_lock = _thread_lock(identity)
    if not thread_lock.acquire(blocking=False):
        raise OutputLockBusy(f"output is busy: {Path(target)}")

    descriptor = None
    file_locked = False
    try:
        descriptor = _open_lock_file(_lock_file_for_identity(identity))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            file_locked = True
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise OutputLockBusy(f"output is busy: {Path(target)}") from error
            raise OutputLockError(f"cannot acquire output lock: {error}") from error
        yield
    finally:
        try:
            if descriptor is not None:
                if file_locked:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(descriptor)
        finally:
            thread_lock.release()
