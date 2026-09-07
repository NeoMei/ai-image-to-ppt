"""Diagnostic only: test a NUL-terminated Win32 rename buffer in memory."""
import ctypes
import re
import sys
import unittest
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import windows_image_output


def terminated_rename(self, handle, destination):
    encoded = str(destination).encode("utf-16-le")
    offset = self._RenameHeader.name_length.offset + ctypes.sizeof(self._wintypes.DWORD)
    # Only experimental difference from the candidate: room for one zero WCHAR.
    buffer = ctypes.create_string_buffer(offset + len(encoded) + 2)
    header = self._RenameHeader.from_buffer(buffer)
    header.replace = False
    header.root = None
    header.name_length = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
    if not self._set_info(handle, windows_image_output._FILE_RENAME_INFO_CLASS, buffer, len(buffer)):
        self._raise_last()


windows_image_output._NativeApi.rename = terminated_rename
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "test_windows_image_output",
    "test_import_host_image.HostImageImportTests.test_local_file_is_validated_and_published_inside_workspace",
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
doc = (ROOT / "docs/windows-validation.md").read_text(encoding="utf-8")
block = re.search(r"@'\r?\n(.*?)\r?\n'@ \| ", doc, re.S)
assert block is not None
exec(compile(block.group(1), "documented-artifact-flow", "exec"), {"__name__": "__main__"})
