"""Fail-fast cross-thread and cross-process ownership for output paths."""

import errno
import hashlib
import os
import tempfile
import threading
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple, Union

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by isolated import tests
    _fcntl = None

if _fcntl is None:
    try:
        import msvcrt as _msvcrt
    except ImportError:  # pragma: no cover - exercised by isolated import tests
        _msvcrt = None
else:
    _msvcrt = None


PathValue = Union[str, os.PathLike]
_LOCK_DIRECTORY = Path(tempfile.gettempdir()) / "ai-image-to-ppt-output-locks"
_REGISTRY_GUARD = threading.Lock()
_THREAD_LOCKS = {}
_THREAD_LOCK_REFS = {}
_SEMANTICS_GUARD = threading.Lock()
_FILESYSTEM_SEMANTICS = {}


class OutputLockError(OSError):
    """Raised when output ownership cannot be established safely."""


class OutputLockBusy(OutputLockError):
    """Raised when another live caller owns the same output."""


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.parent
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    if not candidate.is_dir():
        raise OutputLockError(
            f"cannot determine output filesystem semantics: {candidate} "
            "is not a directory"
        )
    return candidate


def _probe_filesystem_semantics(directory: Path) -> Tuple[bool, bool]:
    descriptor = None
    probe_path = None
    try:
        descriptor, probe_path = tempfile.mkstemp(
            prefix=".ai-image-to-ppt-case-probe-é-",
            dir=str(directory),
        )
        os.close(descriptor)
        descriptor = None
        probe = Path(probe_path)
        case_alias = probe.with_name(
            probe.name.replace("case-probe", "CASE-PROBE", 1)
        )
        unicode_alias = probe.with_name(unicodedata.normalize("NFD", probe.name))

        def aliases_same_file(alias: Path) -> bool:
            try:
                return alias.exists() and os.path.samefile(str(probe), str(alias))
            except OSError:
                return False

        return aliases_same_file(case_alias), aliases_same_file(unicode_alias)
    except OSError as error:
        raise OutputLockError(
            f"cannot determine output filesystem semantics: {error}"
        ) from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if probe_path is not None:
            try:
                os.unlink(probe_path)
            except OSError:
                pass


def _filesystem_semantics(path: Path) -> Tuple[bool, bool]:
    directory = _nearest_existing_directory(path)
    try:
        device = directory.stat().st_dev
    except OSError as error:
        raise OutputLockError(
            f"cannot determine output filesystem identity: {error}"
        ) from error
    with _SEMANTICS_GUARD:
        semantics = _FILESYSTEM_SEMANTICS.get(device)
        if semantics is None:
            semantics = _probe_filesystem_semantics(directory)
            _FILESYSTEM_SEMANTICS[device] = semantics
        return semantics


def _identity(target: PathValue, namespace: str) -> str:
    try:
        canonical_path = Path(target).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError) as error:
        raise OutputLockError(f"invalid output lock target: {error}") from error
    case_insensitive, normalization_insensitive = _filesystem_semantics(
        canonical_path
    )
    canonical = str(canonical_path)
    if normalization_insensitive:
        canonical = unicodedata.normalize("NFC", canonical)
    if case_insensitive:
        canonical = canonical.casefold()
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
            _THREAD_LOCK_REFS[identity] = 0
        _THREAD_LOCK_REFS[identity] += 1
        return lock


def _release_thread_lock_reference(identity: str, lock: threading.Lock) -> None:
    """Forget an idle registry entry after its final caller is finished."""
    with _REGISTRY_GUARD:
        if _THREAD_LOCKS.get(identity) is not lock:
            return
        remaining = _THREAD_LOCK_REFS[identity] - 1
        if remaining <= 0 and not lock.locked():
            _THREAD_LOCK_REFS.pop(identity, None)
            _THREAD_LOCKS.pop(identity, None)
        else:
            _THREAD_LOCK_REFS[identity] = remaining


def _open_lock_file(path: Path) -> int:
    try:
        _LOCK_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags, 0o600)
        if _msvcrt is not None:
            if os.fstat(descriptor).st_size < 1:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except OSError as error:
        raise OutputLockError(f"cannot open output lock: {error}") from error


def _busy_lock_error(error: OSError) -> bool:
    return error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK) or getattr(
        error, "winerror", None
    ) in (33, 36)


def _acquire_process_lock(descriptor: int) -> None:
    if _fcntl is not None:
        _fcntl.flock(descriptor, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return
    if _msvcrt is not None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        _msvcrt.locking(descriptor, _msvcrt.LK_NBLCK, 1)
        return
    raise OutputLockError("no supported process lock backend is available")


def _release_process_lock(descriptor: int) -> None:
    if _fcntl is not None:
        _fcntl.flock(descriptor, _fcntl.LOCK_UN)
    elif _msvcrt is not None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        _msvcrt.locking(descriptor, _msvcrt.LK_UNLCK, 1)


@contextmanager
def output_lock(
    target: PathValue,
    namespace: str = "output",
) -> Iterator[None]:
    """Own one output key until the context exits, failing fast if it is busy."""
    identity = _identity(target, namespace)
    thread_lock = _thread_lock(identity)
    try:
        acquired = thread_lock.acquire(blocking=False)
    except BaseException:
        _release_thread_lock_reference(identity, thread_lock)
        raise
    if not acquired:
        _release_thread_lock_reference(identity, thread_lock)
        raise OutputLockBusy(f"output is busy: {Path(target)}")

    descriptor = None
    file_locked = False
    try:
        if _fcntl is None and _msvcrt is None:
            raise OutputLockError("no supported process lock backend is available")
        descriptor = _open_lock_file(_lock_file_for_identity(identity))
        try:
            _acquire_process_lock(descriptor)
            file_locked = True
        except OSError as error:
            if _busy_lock_error(error):
                raise OutputLockBusy(f"output is busy: {Path(target)}") from error
            raise OutputLockError(f"cannot acquire output lock: {error}") from error
        yield
    finally:
        try:
            if descriptor is not None:
                if file_locked:
                    try:
                        _release_process_lock(descriptor)
                    except OSError:
                        pass
                os.close(descriptor)
        finally:
            thread_lock.release()
            _release_thread_lock_reference(identity, thread_lock)
