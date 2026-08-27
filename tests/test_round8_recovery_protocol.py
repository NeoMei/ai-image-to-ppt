import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


class CommitDecisionTests(unittest.TestCase):
    def test_installed_marker_failure_never_rolls_back_committed_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            Image.new("RGB", (160, 90), "navy").save(source)
            real_write_marker = export_images._write_commit_marker

            def installed_then_raise(*args, **kwargs):
                real_write_marker(*args, **kwargs)
                raise OSError("forced failure after marker installation")

            stderr = io.StringIO()
            with mock.patch.object(
                export_images,
                "_write_commit_marker",
                side_effect=installed_then_raise,
            ), redirect_stderr(stderr):
                first_result = export_images.export_deck(
                    [str(source)], str(prefix)
                )

            self.assertTrue(first_result, stderr.getvalue())
            self.assertTrue(prefix.with_suffix(".pdf").is_file())
            self.assertTrue(prefix.with_suffix(".pptx").is_file())
            self.assertFalse(export_images._journal_path(prefix).exists())
            self.assertFalse(
                export_images._commit_marker_path(
                    export_images._journal_path(prefix)
                ).exists()
            )


class ForceRestoreCrashTests(unittest.TestCase):
    def test_crash_when_backup_name_is_retired_recovers_on_next_call(self):
        program = textwrap.dedent(
            """
            import os
            import sys
            from pathlib import Path

            sys.path.insert(0, sys.argv[1])
            import export_images

            source = sys.argv[2]
            prefix = sys.argv[3]
            real_link = export_images.os.link
            real_rename = export_images.os.rename
            failed_publish = False

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                global failed_publish
                if (
                    str(target_path).endswith(".pptx")
                    and ".tmp.pptx" in str(source_path)
                    and not failed_publish
                ):
                    failed_publish = True
                    raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            def crash_after_backup_name_retired(source_path, target_path):
                result = real_rename(source_path, target_path)
                if str(source_path).endswith(".backup"):
                    os._exit(77)
                return result

            export_images.os.link = fail_second_publish
            export_images.os.rename = crash_after_backup_name_retired
            export_images.export_deck([source], prefix, force=True)
            raise SystemExit(3)
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            Image.new("RGB", (160, 90), "navy").save(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")

            crashed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(ROOT / "scripts"),
                    str(source),
                    str(prefix),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(crashed.returncode, 77, crashed.stderr)

            with redirect_stderr(io.StringIO()):
                result = export_images.export_deck(
                    [str(root / "missing.png")], str(prefix), force=True
                )

            self.assertFalse(result)
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertFalse(export_images._journal_path(prefix).exists())
            self.assertEqual(list(root.glob(".*.backup")), [])
            self.assertEqual(
                list(root.glob(".ai-image-to-ppt-recovery-*.quarantine")),
                [],
            )


class InitialJournalInstallationTests(unittest.TestCase):
    def test_crash_before_initial_journal_install_is_cleaned_on_next_call(self):
        program = textwrap.dedent(
            """
            import os
            import sys

            sys.path.insert(0, sys.argv[1])
            import export_images

            source = sys.argv[2]
            prefix = sys.argv[3]
            journal_suffix = export_images.JOURNAL_SUFFIX
            real_link = export_images.os.link

            def crash_before_journal_install(source_path, target_path, *args, **kwargs):
                if str(target_path).endswith(journal_suffix):
                    os._exit(77)
                return real_link(source_path, target_path, *args, **kwargs)

            export_images.os.link = crash_before_journal_install
            export_images.export_deck([source], prefix)
            raise SystemExit(3)
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            Image.new("RGB", (160, 90), "navy").save(source)

            crashed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(ROOT / "scripts"),
                    str(source),
                    str(prefix),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(crashed.returncode, 77, crashed.stderr)
            self.assertFalse(export_images._journal_path(prefix).exists())
            self.assertTrue(list(root.rglob("*.journal-write")))

            with redirect_stderr(io.StringIO()):
                result = export_images.export_deck(
                    [str(root / "missing.png")], str(prefix)
                )

            self.assertFalse(result)
            self.assertFalse(export_images._journal_path(prefix).exists())
            self.assertEqual(list(root.rglob("*.journal-write")), [])
            self.assertEqual(list(root.rglob(".*.tmp.*")), [])


class TemporaryIdentityPublicationTests(unittest.TestCase):
    def test_replaced_serialized_temp_is_not_left_at_final_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            Image.new("RGB", (160, 90), "navy").save(source)
            real_sync_file = export_images._sync_file
            sync_count = 0

            def replace_first_temp_after_sync(path):
                nonlocal sync_count
                real_sync_file(path)
                sync_count += 1
                if sync_count == 1:
                    Path(path).unlink()
                    Path(path).write_bytes(b"external-temp-replacement")

            stderr = io.StringIO()
            with mock.patch.object(
                export_images,
                "_sync_file",
                side_effect=replace_first_temp_after_sync,
            ), redirect_stderr(stderr):
                result = export_images.export_deck(
                    [str(source)], str(prefix)
                )

            self.assertFalse(result)
            pdf = prefix.with_suffix(".pdf")
            self.assertFalse(pdf.exists())
            self.assertNotIn(
                b"external-temp-replacement",
                pdf.read_bytes() if pdf.exists() else b"",
            )


class AggregateLoadedByteTests(unittest.TestCase):
    def test_actual_loaded_bytes_enforce_aggregate_limit_before_serializers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.img"
            second = root / "second.img"
            prefix = root / "deck"
            Image.new("RGB", (160, 90), "navy").save(first, format="PNG")
            Image.new("RGB", (160, 90), "navy").save(second, format="PNG")
            aggregate_limit = first.stat().st_size + second.stat().st_size + 1

            def replacing_inputs():
                yield str(first)
                yield str(second)
                Image.effect_noise((1000, 1000), 100).convert("RGB").save(
                    first, format="BMP"
                )
                Image.effect_noise((1000, 1000), 100).convert("RGB").save(
                    second, format="BMP"
                )

            stderr = io.StringIO()
            with mock.patch.object(
                export_images,
                "MAX_DECK_SOURCE_BYTES",
                aggregate_limit,
            ), mock.patch.object(
                export_images,
                "_save_pdf",
            ) as save_pdf, mock.patch.object(
                export_images,
                "_save_pptx",
            ) as save_pptx, redirect_stderr(stderr):
                result = export_images.export_deck(
                    replacing_inputs(), str(prefix)
                )

            self.assertFalse(result)
            self.assertIn("aggregate source bytes", stderr.getvalue())
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
