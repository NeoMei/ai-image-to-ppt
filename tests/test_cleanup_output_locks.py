import contextlib
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cleanup_output_locks


class CleanupOutputLocksTests(unittest.TestCase):
    def test_cleanup_rejects_symlink_directory_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "real-locks"
            target.mkdir()
            stale = target / f"{'e' * 64}.lock"
            stale.write_bytes(b"")
            old = time.time() - 2 * 86400
            os.utime(stale, (old, old))
            link = root / "lock-directory-link"
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(
                cleanup_output_locks.CleanupOutputLockError,
                "must be a real directory",
            ):
                cleanup_output_locks.cleanup_stale_lock_files(
                    lock_directory=link,
                    max_age_seconds=86400,
                )

            self.assertTrue(stale.exists())

    def test_cleanup_rejects_non_directory_lock_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "not-a-directory"
            lock_path.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(
                cleanup_output_locks.CleanupOutputLockError,
                "must be a real directory",
            ):
                cleanup_output_locks.cleanup_stale_lock_files(
                    lock_directory=lock_path,
                    max_age_seconds=0,
                )

    def test_cleanup_removes_only_stale_hashed_lock_files_at_high_cardinality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_directory = Path(temp_dir)
            now = time.time()
            stale_paths = []
            for index in range(500):
                path = lock_directory / f"{index:064x}.lock"
                path.write_bytes(b"")
                os.utime(path, (now - 3600, now - 3600))
                stale_paths.append(path)

            recent = lock_directory / f"{999:064x}.lock"
            recent.write_bytes(b"")
            os.utime(recent, (now, now))
            unrelated = lock_directory / "notes.txt"
            unrelated.write_text("keep", encoding="utf-8")
            nested = lock_directory / f"{'a' * 64}.lock"
            nested.mkdir()
            symlink = lock_directory / f"{'b' * 64}.lock"
            symlink.symlink_to(unrelated)

            removed = cleanup_output_locks.cleanup_stale_lock_files(
                lock_directory=lock_directory,
                max_age_seconds=60,
                now=now,
            )

            self.assertEqual(removed, 500)
            self.assertTrue(all(not path.exists() for path in stale_paths))
            self.assertTrue(recent.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(nested.is_dir())
            self.assertTrue(symlink.is_symlink())

    def test_cleanup_rejects_invalid_age_without_removing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_directory = Path(temp_dir)
            stale = lock_directory / f"{'c' * 64}.lock"
            stale.write_bytes(b"")

            for invalid in (-1, float("inf"), True):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        cleanup_output_locks.cleanup_stale_lock_files(
                            lock_directory=lock_directory,
                            max_age_seconds=invalid,
                        )
                    self.assertTrue(stale.exists())

    def test_cli_cleans_configured_directory_and_reports_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_directory = Path(temp_dir)
            stale = lock_directory / f"{'d' * 64}.lock"
            stale.write_bytes(b"")
            old = time.time() - 2 * 86400
            os.utime(stale, (old, old))
            output = io.StringIO()

            with mock.patch.object(
                cleanup_output_locks,
                "LOCK_DIRECTORY",
                lock_directory,
            ), contextlib.redirect_stdout(output):
                result = cleanup_output_locks.main(["--older-than-days", "1"])

            self.assertEqual(result, 0)
            self.assertFalse(stale.exists())
            self.assertIn("Removed 1 stale output lock file", output.getvalue())

    def test_cli_help_states_strict_offline_precondition(self):
        help_text = cleanup_output_locks.build_parser().format_help()

        self.assertIn("--older-than-days", help_text)
        self.assertIn("ONLY when no generation or export process is running", help_text)
        self.assertIn("never runs automatically", help_text)
        self.assertIn(
            "offline precondition is the safety boundary for pathname races",
            help_text,
        )
        self.assertIn(
            "Filesystem failures stop cleanup, return exit 1 without a "
            "traceback, and do not roll back earlier removals.",
            help_text,
        )

    def test_cli_returns_one_without_traceback_when_directory_scan_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = io.StringIO()
            with mock.patch.object(
                cleanup_output_locks,
                "LOCK_DIRECTORY",
                Path(temp_dir),
            ), mock.patch.object(
                Path,
                "iterdir",
                side_effect=PermissionError("forced scan denial"),
            ), contextlib.redirect_stderr(stderr):
                result = cleanup_output_locks.main([])

        self.assertEqual(result, 1)
        self.assertIn("ERR: output lock cleanup failed", stderr.getvalue())
        self.assertIn("forced scan denial", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_returns_one_without_traceback_when_lock_removal_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_directory = Path(temp_dir)
            stale = lock_directory / f"{'f' * 64}.lock"
            stale.write_bytes(b"")
            old = time.time() - 2 * 86400
            os.utime(stale, (old, old))
            stderr = io.StringIO()
            with mock.patch.object(
                cleanup_output_locks,
                "LOCK_DIRECTORY",
                lock_directory,
            ), mock.patch.object(
                Path,
                "unlink",
                side_effect=PermissionError("forced unlink denial"),
            ), contextlib.redirect_stderr(stderr):
                result = cleanup_output_locks.main(["--older-than-days", "1"])

            self.assertTrue(stale.exists())

        self.assertEqual(result, 1)
        self.assertIn("ERR: output lock cleanup failed", stderr.getvalue())
        self.assertIn("forced unlink denial", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_negative_days(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cleanup_output_locks.main(["--older-than-days", "-1"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
