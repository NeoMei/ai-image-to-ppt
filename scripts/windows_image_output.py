"""Identity-bound image publication using native Windows file handles."""

import contextlib
import ctypes
import os
import secrets
from dataclasses import replace
from pathlib import Path

import image_output


_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_DELETE = 0x00010000
_FILE_READ_ATTRIBUTES = 0x00000080
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_CREATE_NEW = 1
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_RENAME_INFO_CLASS = 3
_FILE_DISPOSITION_INFO_CLASS = 4
_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3


def _error(message, error):
    if isinstance(error, image_output.ImageOutputError):
        return error
    return image_output.ImageOutputError(f"{message}: {error}")


def _validate_path(path):
    path = Path(path)
    text = str(path)
    if not path.is_absolute() or not path.drive or text.startswith(("\\\\", "//")):
        raise image_output.ImageOutputError(
            "secure Windows publication requires a local absolute drive path"
        )
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{index}" for index in range(1, 10))
    reserved.update(f"LPT{index}" for index in range(1, 10))
    for component in path.parts[1:]:
        if not component or component.endswith((" ", ".")) or ":" in component:
            raise image_output.ImageOutputError("ambiguous Windows output path is unsafe")
        stem = component.split(".", 1)[0].rstrip(" .").upper()
        stem = stem.translate(str.maketrans({"¹": "1", "²": "2", "³": "3"}))
        if stem in reserved:
            raise image_output.ImageOutputError("reserved Windows output path is unsafe")
    return path


def _verify_prepared_parent(prepared):
    image_output.verify_parent_identity(prepared.parent)


def _publication_checkpoint(_phase):
    """Test boundary after installation while all identity handles remain open."""


class _NativeApi:
    def __init__(self):
        if os.name != "nt":
            raise OSError("Win32 APIs are available only on Windows")
        import msvcrt
        from ctypes import wintypes

        self._msvcrt = msvcrt
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._invalid = ctypes.c_void_p(-1).value

        class FileTime(ctypes.Structure):
            _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

        class ByHandleInfo(ctypes.Structure):
            _fields_ = (
                ("attributes", wintypes.DWORD),
                ("creation", FileTime),
                ("access", FileTime),
                ("write", FileTime),
                ("volume", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD),
                ("index_low", wintypes.DWORD),
            )

        class RenameHeader(ctypes.Structure):
            _fields_ = (
                ("replace", wintypes.BOOLEAN),
                ("root", wintypes.HANDLE),
                ("name_length", wintypes.DWORD),
            )

        class DispositionInfo(ctypes.Structure):
            _fields_ = (("delete", wintypes.BOOLEAN),)

        self._ByHandleInfo = ByHandleInfo
        self._RenameHeader = RenameHeader
        self._DispositionInfo = DispositionInfo

        self._create_file = self._kernel32.CreateFileW
        self._create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self._create_file.restype = wintypes.HANDLE
        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL
        self._get_info = self._kernel32.GetFileInformationByHandle
        self._get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(ByHandleInfo))
        self._get_info.restype = wintypes.BOOL
        self._set_info = self._kernel32.SetFileInformationByHandle
        self._set_info.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        self._set_info.restype = wintypes.BOOL
        self._get_process = self._kernel32.GetCurrentProcess
        self._get_process.argtypes = ()
        self._get_process.restype = wintypes.HANDLE
        self._duplicate = self._kernel32.DuplicateHandle
        self._duplicate.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        self._duplicate.restype = wintypes.BOOL

    def _raise_last(self):
        raise ctypes.WinError(ctypes.get_last_error())

    def _open(self, path, access, share, creation, flags):
        handle = self._create_file(
            str(path), access, share, None, creation, flags, None
        )
        if handle == self._invalid:
            self._raise_last()
        return handle

    def close(self, handle):
        if handle is not None and not self._close_handle(handle):
            self._raise_last()

    def info(self, handle):
        result = self._ByHandleInfo()
        if not self._get_info(handle, ctypes.byref(result)):
            self._raise_last()
        return result

    def attributes(self, handle):
        return self.info(handle).attributes

    def _duplicate_fd(self, handle, flags):
        duplicate = self._wintypes.HANDLE()
        process = self._get_process()
        if not self._duplicate(
            process, handle, process, ctypes.byref(duplicate), 0, False, 2
        ):
            self._raise_last()
        try:
            return self._msvcrt.open_osfhandle(duplicate.value, flags)
        except Exception:
            self._close_handle(duplicate)
            raise

    def identity(self, handle):
        fd = self._duplicate_fd(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            current = os.fstat(fd)
            return current.st_dev, current.st_ino
        finally:
            os.close(fd)

    def open_directory(self, path, delete=False):
        access = _FILE_READ_ATTRIBUTES | (_DELETE if delete else 0)
        handle = self._open(
            path,
            access,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            attributes = self.attributes(handle)
            if (
                not attributes & _FILE_ATTRIBUTE_DIRECTORY
                or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise image_output.ImageOutputError(
                    "Windows output ancestor is not a plain directory"
                )
        except Exception:
            try:
                self.close(handle)
            except Exception:
                pass
            raise
        return handle

    @contextlib.contextmanager
    def pin_ancestors(self, target):
        handles = []
        try:
            parent = Path(target).parent
            chain = []
            while True:
                chain.append(parent)
                if parent == Path(parent.anchor):
                    break
                parent = parent.parent
            chain.reverse()
            for directory in chain:
                handles.append(self.open_directory(directory))
            yield
        finally:
            for handle in reversed(handles):
                try:
                    self.close(handle)
                except OSError:
                    pass

    def open_existing(self, path, read=False):
        access = _DELETE | _FILE_READ_ATTRIBUTES | (_GENERIC_READ if read else 0)
        try:
            handle = self._open(
                path,
                access,
                _FILE_SHARE_READ,
                _OPEN_EXISTING,
                _FILE_FLAG_OPEN_REPARSE_POINT,
            )
        except OSError as error:
            if getattr(error, "winerror", None) in (
                _ERROR_FILE_NOT_FOUND,
                _ERROR_PATH_NOT_FOUND,
            ):
                return None
            raise
        try:
            attributes = self.attributes(handle)
            if attributes & (
                _FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise image_output.ImageOutputError(
                    "Windows output path exists and is not a plain regular file"
                )
        except Exception:
            try:
                self.close(handle)
            except Exception:
                pass
            raise
        return handle

    def create_new(self, path):
        return self._open(
            path,
            _GENERIC_READ | _GENERIC_WRITE | _DELETE | _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ,
            _CREATE_NEW,
            _FILE_ATTRIBUTE_NORMAL,
        )

    def write(self, handle, data):
        fd = self._duplicate_fd(
            handle, os.O_WRONLY | getattr(os, "O_BINARY", 0)
        )
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

    def read(self, handle):
        fd = self._duplicate_fd(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as source:
            source.seek(0)
            return image_output.read_bounded_image_stream(source)

    def rename(self, handle, destination):
        encoded = str(destination).encode("utf-16-le")
        offset = self._RenameHeader.name_length.offset + ctypes.sizeof(
            self._wintypes.DWORD
        )
        # Win32 can inspect FileName as a NUL-terminated wide string even
        # though FileNameLength excludes the terminator. Keep one zero WCHAR
        # after the encoded path so adjacent memory cannot become its suffix.
        buffer = ctypes.create_string_buffer(
            offset + len(encoded) + ctypes.sizeof(self._wintypes.WCHAR)
        )
        header = self._RenameHeader.from_buffer(buffer)
        header.replace = False
        header.root = None
        header.name_length = len(encoded)
        ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
        if not self._set_info(
            handle, _FILE_RENAME_INFO_CLASS, buffer, len(buffer)
        ):
            self._raise_last()

    def delete(self, handle):
        disposition = self._DispositionInfo(True)
        if not self._set_info(
            handle,
            _FILE_DISPOSITION_INFO_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            self._raise_last()


_NATIVE_API = None


def _api():
    global _NATIVE_API
    if _NATIVE_API is None:
        _NATIVE_API = _NativeApi()
    return _NATIVE_API


def _open_identity(api, path, read=False):
    handle = api.open_existing(path, read=read)
    if handle is None:
        return None, None
    try:
        return handle, api.identity(handle)
    except Exception:
        api.close(handle)
        raise


def _new_temp(api, prepared):
    for _attempt in range(100):
        path = prepared.parent.path / (
            f".{prepared.name}.{secrets.token_hex(16)}.tmp"
        )
        try:
            return path, api.create_new(path)
        except FileExistsError:
            continue
        except OSError as error:
            if getattr(error, "winerror", None) == 80:
                continue
            raise
    raise image_output.ImageOutputError("failed to allocate unique output temporary file")


def _displace_to_new_recovery(api, prepared, old):
    for _attempt in range(100):
        name = f".image-output-recovery-{secrets.token_hex(16)}.entry"
        path = prepared.parent.path / name
        try:
            api.rename(old, path)
        except FileExistsError:
            continue
        except OSError as error:
            if getattr(error, "winerror", None) in (80, 183):
                continue
            raise
        return name, path
    raise image_output.ImageOutputError("failed to allocate unique output recovery file")


def _warn_recovery(prepared, name, context, error):
    image_output._warn_retained_recovery(
        prepared.parent, name, context, error, entry=False
    )


def preflight_output(prepared, overwrite):
    _validate_path(prepared.path)
    api = _api()
    probe = existing = None
    probe_path = None
    try:
        with api.pin_ancestors(prepared.path):
            _verify_prepared_parent(prepared)
            existing, identity = _open_identity(api, prepared.path)
            if identity is not None and not overwrite:
                raise image_output.ImageOutputError(
                    f"output already exists; refusing to overwrite: {prepared.path}"
                )
            if existing is not None:
                api.close(existing)
                existing = None
            probe_path, probe = _new_temp(api, prepared)
            api.delete(probe)
            api.close(probe)
            probe = None
            current, final_identity = _open_identity(api, prepared.path)
            if current is not None:
                api.close(current)
            if final_identity != identity:
                raise image_output.ImageOutputError("output identity changed during preflight")
            _verify_prepared_parent(prepared)
            return replace(
                prepared,
                expected_identity=final_identity,
                expectation_captured=True,
            )
    except image_output.ImageOutputError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise _error("output path is not writable", error) from error
    finally:
        if probe is not None:
            try:
                api.delete(probe)
            except Exception as error:
                image_output._warn_retained_temp(
                    "write-test cleanup was incomplete", str(probe_path), error
                )
            try:
                api.close(probe)
            except Exception as error:
                image_output._warn_retained_temp(
                    "write-test handle cleanup was incomplete", str(probe_path), error
                )
        if existing is not None:
            try:
                api.close(existing)
            except Exception:
                pass


def capture_identity(prepared):
    _validate_path(prepared.path)
    api = _api()
    try:
        with api.pin_ancestors(prepared.path):
            _verify_prepared_parent(prepared)
            handle, identity = _open_identity(api, prepared.path)
            if handle is not None:
                api.close(handle)
            return identity
    except image_output.ImageOutputError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise _error("cannot inspect output identity securely", error) from error


def snapshot_output(prepared):
    _validate_path(prepared.path)
    api = _api()
    handle = None
    try:
        with api.pin_ancestors(prepared.path):
            _verify_prepared_parent(prepared)
            handle, identity = _open_identity(api, prepared.path, read=True)
            if handle is None:
                return None, None
            data = api.read(handle)
            if api.identity(handle) != identity:
                raise image_output.ImageOutputError(
                    "output identity changed during reading"
                )
            return data, identity
    except image_output.ImageOutputError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise _error("could not snapshot transaction output", error) from error
    finally:
        if handle is not None:
            try:
                api.close(handle)
            except Exception:
                pass


def remove_output_with_identity(prepared, expected):
    _validate_path(prepared.path)
    api = _api()
    handle = None
    try:
        with api.pin_ancestors(prepared.path):
            _verify_prepared_parent(prepared)
            handle, identity = _open_identity(api, prepared.path)
            if handle is None:
                raise image_output.ImageOutputError(
                    "owned output disappeared before removal"
                )
            if identity != expected:
                raise image_output.ImageOutputError(
                    "output identity changed during removal"
                )
            api.delete(handle)
    except image_output.ImageOutputError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise _error("could not remove transaction output", error) from error
    finally:
        if handle is not None:
            try:
                api.close(handle)
            except Exception:
                pass


def publish_bytes(data, prepared, overwrite, validator):
    _validate_path(prepared.path)
    if not prepared.expectation_captured:
        raise image_output.ImageOutputError(
            "output publication expectation was not captured"
        )
    validated = validator(data, prepared)
    api = _api()
    old = temporary = None
    temporary_path = None
    recovery_name = recovery_path = None
    installed = False
    old_displaced = False
    installed_identity = None
    try:
        with api.pin_ancestors(prepared.path):
            _verify_prepared_parent(prepared)
            old, old_identity = _open_identity(api, prepared.path, read=True)
            if old_identity != prepared.expected_identity:
                raise image_output.ImageOutputError(
                    "output identity changed before publication"
                )
            if old is not None and not overwrite:
                raise image_output.ImageOutputError(
                    "existing output replacement was not authorized"
                )
            temporary_path, temporary = _new_temp(api, prepared)
            api.write(temporary, data)
            installed_identity = api.identity(temporary)
            if installed_identity is None:
                raise image_output.ImageOutputError(
                    "output temporary file identity changed"
                )

            if old is not None:
                recovery_name, recovery_path = _displace_to_new_recovery(
                    api, prepared, old
                )
                old_displaced = True

            try:
                _publication_checkpoint("before_install")
                api.rename(temporary, prepared.path)
                installed = True
            except Exception as publish_error:
                restore_error = None
                if old_displaced:
                    try:
                        api.rename(old, prepared.path)
                        old_displaced = False
                    except Exception as error:
                        restore_error = error
                        _warn_recovery(
                            prepared,
                            recovery_name,
                            "previous output retained after concurrent target creation",
                            error,
                        )
                if restore_error is not None:
                    raise image_output.ImageOutputError(
                        "failed to publish image without clobbering; previous output "
                        f"retained at {recovery_path}"
                    ) from publish_error
                raise _error(
                    "failed to publish image without clobbering", publish_error
                ) from publish_error

            _publication_checkpoint("installed")
            _verify_prepared_parent(prepared)
            if old_displaced:
                try:
                    api.delete(old)
                    api.close(old)
                    old = None
                    old_displaced = False
                except Exception as cleanup_error:
                    _warn_recovery(
                        prepared,
                        recovery_name,
                        "previous output cleanup was incomplete",
                        cleanup_error,
                    )
            return image_output.PublishedOutput(
                validated.byte_count, *installed_identity
            )
    except image_output.PublishedOutputError:
        raise
    except image_output.ImageOutputError as error:
        if installed and installed_identity is not None:
            try:
                api.delete(temporary)
                api.close(temporary)
                temporary = None
                installed = False
                if old_displaced:
                    api.rename(old, prepared.path)
                    api.close(old)
                    old = None
                    old_displaced = False
                raise image_output.RemovedPublishedOutputError(
                    str(error), *installed_identity
                ) from error
            except image_output.RemovedPublishedOutputError:
                raise
            except Exception as recovery_error:
                if old_displaced and recovery_name is not None:
                    _warn_recovery(
                        prepared,
                        recovery_name,
                        "previous output retained after post-publication failure",
                        recovery_error,
                    )
                raise image_output.PublishedOutputError(
                    str(error), *installed_identity
                ) from recovery_error
        raise
    except (OSError, ValueError, TypeError) as error:
        if installed and installed_identity is not None:
            try:
                api.delete(temporary)
                api.close(temporary)
                temporary = None
                installed = False
                if old_displaced:
                    api.rename(old, prepared.path)
                    api.close(old)
                    old = None
                    old_displaced = False
                raise image_output.RemovedPublishedOutputError(
                    "published output could not be verified", *installed_identity
                ) from error
            except image_output.RemovedPublishedOutputError:
                raise
            except Exception as recovery_error:
                if old_displaced and recovery_name is not None:
                    _warn_recovery(
                        prepared,
                        recovery_name,
                        "previous output retained after post-publication failure",
                        recovery_error,
                    )
                raise image_output.PublishedOutputError(
                    "published output could not be verified", *installed_identity
                ) from recovery_error
        raise _error("failed to write image", error) from error
    finally:
        if temporary is not None and not installed:
            try:
                api.delete(temporary)
            except Exception as cleanup_error:
                image_output._warn_retained_temp(
                    "output failed", str(temporary_path), cleanup_error
                )
        for handle in (old, temporary):
            if handle is not None:
                try:
                    api.close(handle)
                except Exception as cleanup_error:
                    if handle is temporary:
                        image_output._warn_retained_temp(
                            "output handle cleanup was incomplete",
                            str(temporary_path if not installed else prepared.path),
                            cleanup_error,
                        )
