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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


class RecoveryIdentityRaceTests(unittest.TestCase):
    def test_owned_cleanup_preserves_replacement_after_identity_check(self):
        for context in ("published output", "temporary output", "backup"):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "owned.bin"
                path.write_bytes(b"owned")
                expected = export_images._file_identity(path)
                real_identity = export_images._file_identity
                injected = False

                def replace_after_check(candidate):
                    nonlocal injected
                    identity = real_identity(candidate)
                    if Path(candidate) == path and not injected:
                        injected = True
                        path.unlink()
                        path.write_bytes(b"external")
                    return identity

                with mock.patch.object(
                    export_images, "_file_identity", side_effect=replace_after_check
                ):
                    error = export_images._remove_owned_file(path, expected, context)

                self.assertTrue(injected)
                self.assertIsNotNone(error)
                self.assertEqual(path.read_bytes(), b"external")
                quarantines = list(
                    path.parent.glob(".ai-image-to-ppt-recovery-*.quarantine")
                )
                self.assertEqual(len(quarantines), 1)
                self.assertEqual(quarantines[0].read_bytes(), b"external")

    def test_backup_restore_never_clobbers_target_created_after_inspection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            target = root / "deck.pdf"
            temporary = root / ".deck.pdf.owned.tmp.pdf"
            backup = root / ".deck.pdf.owned.backup"
            temporary.write_bytes(b"new")
            backup.write_bytes(b"old")
            original_identity = export_images._file_identity(backup)
            temporary_identity = export_images._file_identity(temporary)
            journal = root / (".deck" + export_images.JOURNAL_SUFFIX)
            journal.write_bytes(b"journal")
            journal_identity = export_images._file_identity(journal)
            parent = export_images.prepare_target(target).parent
            state = {
                "version": export_images.JOURNAL_VERSION,
                "decision": "rollback",
                "records": [{
                    "target": str(target),
                    "temporary": str(temporary),
                    "temporary_identity": export_images._identity_list(temporary_identity),
                    "existed": True,
                    "original_identity": export_images._identity_list(original_identity),
                    "backup": str(backup),
                }],
            }
            real_identity = export_images._file_identity
            injected = False

            def create_after_absence_check(candidate):
                nonlocal injected
                identity = real_identity(candidate)
                if Path(candidate) == target and identity is None and not injected:
                    injected = True
                    target.write_bytes(b"external")
                return identity

            with mock.patch.object(
                export_images, "_file_identity", side_effect=create_after_absence_check
            ):
                errors = export_images._recover_state(
                    state,
                    journal,
                    parent,
                    journal_identity=journal_identity,
                )

            self.assertTrue(injected)
            self.assertTrue(errors)
            self.assertEqual(target.read_bytes(), b"external")
            self.assertEqual(backup.read_bytes(), b"old")
            self.assertTrue(journal.exists())

    def test_loaded_journal_replacement_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prefix = root / "deck"
            targets = (root / "deck.pdf", root / "deck.pptx")
            records = []
            for target in targets:
                temporary = root / f".{target.name}.owned.tmp{target.suffix}"
                temporary.write_bytes(b"owned")
                records.append({
                    "target": str(target),
                    "temporary": str(temporary),
                    "temporary_identity": export_images._identity_list(
                        export_images._file_identity(temporary)
                    ),
                    "existed": False,
                    "original_identity": None,
                    "backup": None,
                })
            journal = export_images._journal_path(prefix)
            journal.write_text(json.dumps({
                "version": export_images.JOURNAL_VERSION,
                "decision": "rollback",
                "records": records,
            }), encoding="utf-8")
            parent = export_images.prepare_target(targets[0]).parent
            loaded = export_images._load_journal(journal, targets, parent)
            journal.unlink()
            journal.write_bytes(b"external-journal")

            errors = export_images._recover_state(
                loaded.state,
                journal,
                parent,
                journal_identity=loaded.identity,
            )

            self.assertTrue(errors)
            self.assertEqual(journal.read_bytes(), b"external-journal")
            self.assertEqual(
                list(root.glob(f".{journal.name}.*.quarantine")), []
            )

    def test_commit_marker_never_replaces_external_journal(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            journal = export_images._journal_path(prefix)
            marker = export_images._commit_marker_path(journal).resolve(
                strict=False
            )
            Image.new("RGB", (160, 90), "navy").save(source)
            real_write_marker = export_images._write_commit_marker
            replaced = False

            def replace_journal_then_mark(marker_path, state, parent):
                nonlocal replaced
                journal.unlink()
                journal.write_bytes(b"external-journal")
                replaced = True
                return real_write_marker(marker_path, state, parent)

            stderr = io.StringIO()
            with mock.patch.object(
                export_images,
                "_write_commit_marker",
                side_effect=replace_journal_then_mark,
            ), redirect_stderr(stderr):
                result = export_images.export_deck(
                    [str(source)], str(prefix)
                )

            self.assertTrue(result)
            self.assertTrue(replaced)
            self.assertEqual(journal.read_bytes(), b"external-journal")
            self.assertTrue(marker.exists())
            self.assertIn("journal ownership changed", stderr.getvalue())

            with redirect_stderr(io.StringIO()):
                self.assertFalse(
                    export_images.export_deck(
                        [str(root / "missing.png")], str(prefix)
                    )
                )
            self.assertEqual(journal.read_bytes(), b"external-journal")
            self.assertTrue(marker.exists())

    def test_commit_marker_replacement_during_cleanup_is_preserved(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            journal = export_images._journal_path(prefix)
            marker = export_images._commit_marker_path(journal).resolve(
                strict=False
            )
            Image.new("RGB", (160, 90), "navy").save(source)
            real_identity = export_images._file_identity
            injected = False

            def replace_marker_after_check(candidate):
                nonlocal injected
                identity = real_identity(candidate)
                if Path(candidate) == marker and identity is not None and not injected:
                    injected = True
                    marker.unlink()
                    marker.write_bytes(b"external-marker")
                return identity

            stderr = io.StringIO()
            with mock.patch.object(
                export_images,
                "_file_identity",
                side_effect=replace_marker_after_check,
            ), redirect_stderr(stderr):
                result = export_images.export_deck(
                    [str(source)], str(prefix)
                )

            self.assertTrue(result)
            self.assertTrue(injected)
            self.assertFalse(journal.exists())
            self.assertEqual(marker.read_bytes(), b"external-marker")
            quarantines = list(
                root.glob(".ai-image-to-ppt-recovery-*.quarantine")
            )
            self.assertEqual(len(quarantines), 1)
            self.assertEqual(quarantines[0].read_bytes(), b"external-marker")
            self.assertIn("commit marker ownership changed", stderr.getvalue())


class PreparationJournalCrashTests(unittest.TestCase):
    def _crash(self, phase: str, source: Path, prefix: Path) -> subprocess.CompletedProcess:
        program = textwrap.dedent(
            """
            import os
            import sys
            from pathlib import Path

            sys.path.insert(0, sys.argv[1])
            import export_images

            phase = sys.argv[2]
            source = sys.argv[3]
            prefix = sys.argv[4]
            real_write_journal = export_images._write_journal
            real_save_pdf = export_images._save_pdf
            real_save_pptx = export_images._save_pptx
            real_fsync = export_images.os.fsync
            crashed_on_fsync = False

            def write_then_crash(*args, **kwargs):
                result = real_write_journal(*args, **kwargs)
                if phase == "journal-write":
                    os._exit(77)
                return result

            def save_pdf_then_crash(images, path):
                real_save_pdf(images, path)
                if phase == "pdf-serialize":
                    os._exit(77)

            def save_pptx_then_crash(images, path):
                real_save_pptx(images, path)
                if phase == "pptx-serialize":
                    os._exit(77)

            def fsync_then_crash(descriptor):
                global crashed_on_fsync
                real_fsync(descriptor)
                if phase == "journal-fsync" and not crashed_on_fsync:
                    crashed_on_fsync = True
                    os._exit(77)

            export_images._write_journal = write_then_crash
            export_images._save_pdf = save_pdf_then_crash
            export_images._save_pptx = save_pptx_then_crash
            export_images.os.fsync = fsync_then_crash
            raise SystemExit(
                0 if export_images.export_deck([source], prefix) else 2
            )
            """
        )
        return subprocess.run(
            [sys.executable, "-c", program, str(ROOT / "scripts"), phase,
             str(source), str(prefix)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_next_call_recovers_temps_after_prepublication_crash(self):
        from PIL import Image

        for phase in (
            "journal-fsync",
            "journal-write",
            "pdf-serialize",
            "pptx-serialize",
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "slide.png"
                prefix = root / "deck"
                Image.new("RGB", (160, 90), "navy").save(source)
                crashed = self._crash(phase, source, prefix)
                self.assertEqual(crashed.returncode, 77, crashed.stderr)
                journal = export_images._journal_path(prefix)
                preparation = export_images._preparation_path(journal)
                self.assertTrue(preparation.is_dir())
                if phase == "journal-fsync":
                    self.assertFalse(journal.exists())
                else:
                    self.assertTrue(journal.is_file())
                    self.assertEqual(
                        len([
                            path for path in root.rglob(".*.tmp.*")
                            if not path.name.endswith(".owner")
                        ]),
                        2,
                    )

                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    result = export_images.export_deck(
                        [str(root / "missing.png")], str(prefix)
                    )

                self.assertFalse(result)
                self.assertFalse(journal.exists())
                self.assertFalse(preparation.exists())
                self.assertEqual(list(root.rglob(".*.tmp.*")), [])
                self.assertEqual(list(root.glob(".*.backup")), [])
                self.assertFalse(prefix.with_suffix(".pdf").exists())
                self.assertFalse(prefix.with_suffix(".pptx").exists())

    def test_recovery_does_not_remove_external_temp_replacement(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            prefix = root / "deck"
            Image.new("RGB", (160, 90), "navy").save(source)
            crashed = self._crash("pdf-serialize", source, prefix)
            self.assertEqual(crashed.returncode, 77, crashed.stderr)
            temporary = sorted(
                path for path in root.rglob(".*.tmp.*")
                if not path.name.endswith(".owner")
            )[0]
            temporary.unlink()
            temporary.write_bytes(b"external-temp")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = export_images.export_deck(
                    [str(root / "missing.png")], str(prefix)
                )

            self.assertFalse(result)
            self.assertEqual(temporary.read_bytes(), b"external-temp")
            self.assertTrue(export_images._journal_path(prefix).exists())
            self.assertIn("external file preserved", stderr.getvalue())

    def test_commit_marker_crash_completes_instead_of_rolling_back(self):
        from PIL import Image

        program = textwrap.dedent(
            """
            import os
            import sys
            from pathlib import Path

            sys.path.insert(0, sys.argv[1])
            import export_images

            source = sys.argv[2]
            prefix = sys.argv[3]
            force = sys.argv[4] == "true"
            real_link = export_images.os.link

            def link_then_crash(source_path, target_path, *args, **kwargs):
                result = real_link(source_path, target_path, *args, **kwargs)
                if str(target_path).endswith(
                    export_images.COMMIT_MARKER_SUFFIX
                ):
                    os._exit(77)
                return result

            export_images.os.link = link_then_crash
            raise SystemExit(
                0 if export_images.export_deck(
                    [source], prefix, force=force
                ) else 2
            )
            """
        )
        for force in (False, True):
            with self.subTest(force=force), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "slide.png"
                prefix = root / "deck"
                pdf = prefix.with_suffix(".pdf")
                pptx = prefix.with_suffix(".pptx")
                Image.new("RGB", (160, 90), "navy").save(source)
                if force:
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
                        "true" if force else "false",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(crashed.returncode, 77, crashed.stderr)
                journal = export_images._journal_path(prefix)
                marker = export_images._commit_marker_path(journal)
                self.assertTrue(journal.exists())
                self.assertTrue(marker.exists())
                if not force:
                    # Models a crash after identity-bound journal cleanup but
                    # before commit-marker cleanup. The marker is a complete,
                    # independently discoverable recovery record.
                    journal.unlink()

                with redirect_stderr(io.StringIO()):
                    self.assertFalse(
                        export_images.export_deck(
                            [str(root / "missing.png")], str(prefix)
                        )
                    )

                self.assertTrue(pdf.is_file())
                self.assertTrue(pptx.is_file())
                self.assertNotEqual(pdf.read_bytes(), b"old-pdf")
                self.assertNotEqual(pptx.read_bytes(), b"old-pptx")
                self.assertFalse(journal.exists())
                self.assertFalse(marker.exists())
                self.assertEqual(list(root.glob(".*.backup")), [])
                self.assertEqual(list(root.glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
