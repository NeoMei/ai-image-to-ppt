import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


class ExportDirectorySyncTests(unittest.TestCase):
    def test_windows_ignores_failure_to_open_directory_handle(self):
        open_error = OSError("directory handles are unsupported")
        directory = Path("output")
        with mock.patch.object(export_images.os, "name", "nt"), mock.patch.object(
            export_images.os,
            "open",
            side_effect=open_error,
        ), mock.patch.object(export_images.os, "fsync") as fsync:
            export_images._sync_directory(directory)

        fsync.assert_not_called()

    def test_windows_propagates_fsync_error_after_successful_open(self):
        fsync_error = OSError("directory fsync failed")
        directory = Path("output")
        with mock.patch.object(export_images.os, "name", "nt"), mock.patch.object(
            export_images.os,
            "open",
            return_value=41,
        ), mock.patch.object(
            export_images.os,
            "fsync",
            side_effect=fsync_error,
        ), mock.patch.object(export_images.os, "close") as close:
            with self.assertRaisesRegex(OSError, "directory fsync failed"):
                export_images._sync_directory(directory)

        close.assert_called_once_with(41)

    def test_posix_propagates_failure_to_open_directory_handle(self):
        open_error = OSError("cannot open directory")
        with mock.patch.object(export_images.os, "name", "posix"), mock.patch.object(
            export_images.os,
            "open",
            side_effect=open_error,
        ), mock.patch.object(export_images.os, "fsync") as fsync:
            with self.assertRaisesRegex(OSError, "cannot open directory"):
                export_images._sync_directory(Path("output"))

        fsync.assert_not_called()

    def test_cli_help_states_platform_recovery_boundary(self):
        help_text = export_images._parser().format_help()

        self.assertIn("POSIX", help_text)
        self.assertIn("power-loss metadata", help_text)
        self.assertIn("Windows covers process crashes only", help_text)


if __name__ == "__main__":
    unittest.main()
