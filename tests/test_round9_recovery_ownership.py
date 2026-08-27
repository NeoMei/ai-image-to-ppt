import io
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


class PreparationOwnershipTests(unittest.TestCase):
    def test_journalless_preexisting_preparation_preserves_external_sentinel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prefix = root / "deck"
            preparation = export_images._preparation_path(
                export_images._journal_path(prefix)
            )
            preparation.mkdir()
            sentinel = preparation / "external-important.txt"
            sentinel.write_text("keep me", encoding="utf-8")
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = export_images.export_deck(
                    [str(root / "missing.png")], str(prefix)
                )

            self.assertFalse(result)
            self.assertTrue(sentinel.is_file())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
            self.assertTrue(preparation.is_dir())
            self.assertIn("preparation ownership", stderr.getvalue())
            self.assertIn("preserved", stderr.getvalue())

    def test_owned_preparation_with_unknown_entry_is_preserved_whole(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prefix = root / "deck"
            targets = (prefix.with_suffix(".pdf"), prefix.with_suffix(".pptx"))
            parent = export_images.prepare_target(targets[0]).parent
            journal = export_images._journal_path(prefix)
            preparation, _identity = export_images._begin_preparation(journal, parent)
            temporary, owner, _temporary_identity = (
                export_images._create_prepared_temporary(
                    targets[0], parent, preparation
                )
            )
            Path(temporary).write_bytes(b"owned")
            sentinel = preparation / "external-important.txt"
            sentinel.write_text("keep me", encoding="utf-8")

            with self.assertRaisesRegex(OSError, "preparation.*preserved"):
                export_images._recover_transaction(prefix, targets, parent)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
            self.assertEqual(Path(temporary).read_bytes(), b"owned")
            self.assertTrue(Path(owner).is_file())
            self.assertTrue(preparation.is_dir())

    def test_replaced_preparation_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            prefix = root / "deck"
            targets = (prefix.with_suffix(".pdf"), prefix.with_suffix(".pptx"))
            parent = export_images.prepare_target(targets[0]).parent
            journal = export_images._journal_path(prefix)
            preparation, _identity = export_images._begin_preparation(journal, parent)
            displaced = root / "owned-preparation-displaced"
            preparation.rename(displaced)
            preparation.mkdir()
            sentinel = preparation / "external-important.txt"
            sentinel.write_text("keep me", encoding="utf-8")

            with self.assertRaisesRegex(OSError, "preparation ownership"):
                export_images._recover_transaction(prefix, targets, parent)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
            self.assertTrue(displaced.is_dir())

    def test_preparation_replaced_after_scan_preserves_valid_looking_external_pair(self):
        for replacement_kind in ("directory", "symlink"):
            with self.subTest(replacement_kind=replacement_kind), \
                 tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir).resolve()
                prefix = root / "deck"
                targets = (
                    prefix.with_suffix(".pdf"),
                    prefix.with_suffix(".pptx"),
                )
                parent = export_images.prepare_target(targets[0]).parent
                journal = export_images._journal_path(prefix)
                preparation, _identity = export_images._begin_preparation(
                    journal, parent
                )
                temporary, owner, _temporary_identity = (
                    export_images._create_prepared_temporary(
                        targets[0], parent, preparation
                    )
                )
                Path(temporary).write_bytes(b"owned")
                displaced = root / "owned-preparation-displaced"
                external = root / "external-preparation"
                external.mkdir()
                external_file = external / Path(temporary).name
                external_owner = external / Path(owner).name
                external_file.write_bytes(b"external-important")
                os.link(external_file, external_owner)
                real_iterdir = Path.iterdir
                swapped = False

                def swap_after_scan(path):
                    nonlocal swapped
                    children = list(real_iterdir(path))
                    if Path(path) == preparation and not swapped:
                        swapped = True
                        preparation.rename(displaced)
                        if replacement_kind == "directory":
                            preparation.mkdir()
                            for child in external.iterdir():
                                os.link(child, preparation / child.name)
                        else:
                            preparation.symlink_to(
                                external, target_is_directory=True
                            )
                    return iter(children)

                with mock.patch.object(
                    Path, "iterdir", autospec=True, side_effect=swap_after_scan
                ), self.assertRaisesRegex(OSError, "preparation"):
                    export_images._recover_transaction(prefix, targets, parent)

                self.assertTrue(swapped)
                self.assertTrue(external_file.is_file())
                self.assertTrue(external_owner.is_file())
                self.assertEqual(
                    external_file.read_bytes(), b"external-important"
                )
                self.assertEqual(
                    external_owner.read_bytes(), b"external-important"
                )
                if replacement_kind == "directory":
                    self.assertTrue((preparation / Path(temporary).name).is_file())
                    self.assertTrue((preparation / Path(owner).name).is_file())

    def test_partial_journal_writer_crash_cleans_only_owned_structure(self):
        program = textwrap.dedent(
            """
            import os
            import sys

            sys.path.insert(0, sys.argv[1])
            import export_images

            real_fdopen = export_images.os.fdopen
            binary_writer_count = 0

            class CrashWriter:
                def __init__(self, descriptor, mode):
                    self.stream = real_fdopen(descriptor, mode)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.stream.close()

                def write(self, data):
                    self.stream.write(data[: max(1, len(data) // 2)])
                    self.stream.flush()
                    os.fsync(self.stream.fileno())
                    os._exit(77)

                def flush(self):
                    return self.stream.flush()

                def fileno(self):
                    return self.stream.fileno()

            def crash_fdopen(descriptor, mode, *args, **kwargs):
                global binary_writer_count
                if mode == "wb":
                    binary_writer_count += 1
                if mode == "wb" and binary_writer_count == 2:
                    return CrashWriter(descriptor, mode)
                return real_fdopen(descriptor, mode, *args, **kwargs)

            export_images.os.fdopen = crash_fdopen
            export_images.export_deck([sys.argv[2]], sys.argv[3])
            raise SystemExit(3)
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
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
            journal = export_images._journal_path(prefix)
            preparation = export_images._preparation_path(journal)
            self.assertFalse(journal.exists())
            self.assertTrue(preparation.is_dir())
            self.assertEqual(
                len(list(root.glob(f"{preparation.name}.owner*"))),
                2,
                "a journals-free preparation needs a parent-level owner pair",
            )

            with redirect_stderr(io.StringIO()):
                result = export_images.export_deck(
                    [str(root / "missing.png")], str(prefix)
                )

            self.assertFalse(result)
            self.assertFalse(preparation.exists())
            self.assertEqual(list(root.glob(f"{preparation.name}.owner*")), [])
            self.assertEqual(list(root.rglob("*.journal-write*")), [])
            self.assertEqual(list(root.rglob(".*.tmp.*")), [])


class ForceNoClobberTests(unittest.TestCase):
    def test_force_backup_install_never_replaces_external_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            source = root / "slide.png"
            prefix = root / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            Image.new("RGB", (160, 90), "navy").save(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_replace = export_images.os.replace
            real_link = export_images.os.link
            collision = None

            def create_collision(destination):
                nonlocal collision
                path = Path(destination)
                if path.name.endswith(".backup") and collision is None:
                    path.write_bytes(b"external-backup")
                    collision = path

            def collide_before_replace(source_path, target_path, *args, **kwargs):
                create_collision(target_path)
                return real_replace(source_path, target_path, *args, **kwargs)

            def collide_before_link(source_path, target_path, *args, **kwargs):
                create_collision(target_path)
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os, "replace", side_effect=collide_before_replace
            ), mock.patch.object(
                export_images.os, "link", side_effect=collide_before_link
            ), redirect_stderr(io.StringIO()):
                result = export_images.export_deck(
                    [str(source)], str(prefix), force=True
                )

            self.assertFalse(result)
            self.assertIsNotNone(collision)
            self.assertEqual(collision.read_bytes(), b"external-backup")
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")

    def test_retired_install_never_replaces_external_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            backup = root / ".deck.pdf.random.backup"
            retired = export_images._retired_backup_path(backup)
            backup.write_bytes(b"owned-old")
            expected = export_images._file_identity(backup)
            real_rename = export_images.os.rename
            real_link = export_images.os.link
            injected = False

            def create_collision(destination):
                nonlocal injected
                if Path(destination) == retired and not injected:
                    retired.write_bytes(b"external-retired")
                    injected = True

            def collide_before_rename(source_path, target_path, *args, **kwargs):
                create_collision(target_path)
                return real_rename(source_path, target_path, *args, **kwargs)

            def collide_before_link(source_path, target_path, *args, **kwargs):
                create_collision(target_path)
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os, "rename", side_effect=collide_before_rename
            ), mock.patch.object(
                export_images.os, "link", side_effect=collide_before_link
            ):
                error = export_images._retire_backup(backup, expected)

            self.assertTrue(injected)
            self.assertIsNotNone(error)
            self.assertEqual(backup.read_bytes(), b"owned-old")
            self.assertEqual(retired.read_bytes(), b"external-retired")

    def test_crash_after_backup_link_recovers_original_pair(self):
        program = textwrap.dedent(
            """
            import os
            import sys

            sys.path.insert(0, sys.argv[1])
            import export_images

            real_link = export_images.os.link

            def crash_after_backup_link(source_path, target_path, *args, **kwargs):
                result = real_link(source_path, target_path, *args, **kwargs)
                if str(target_path).endswith(".backup"):
                    os._exit(77)
                return result

            export_images.os.link = crash_after_backup_link
            export_images.export_deck([sys.argv[2]], sys.argv[3], force=True)
            raise SystemExit(3)
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
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
            self.assertEqual(list(root.glob(".*.backup*")), [])

    def test_crash_after_retired_link_recovers_original_pair(self):
        program = textwrap.dedent(
            """
            import os
            import sys

            sys.path.insert(0, sys.argv[1])
            import export_images

            real_link = export_images.os.link
            failed_publish = False

            def crash_during_rollback(source_path, target_path, *args, **kwargs):
                global failed_publish
                if (
                    str(target_path).endswith(".pptx")
                    and ".tmp.pptx" in str(source_path)
                    and not failed_publish
                ):
                    failed_publish = True
                    raise OSError("forced second publish failure")
                result = real_link(source_path, target_path, *args, **kwargs)
                if str(target_path).endswith(".backup.retired"):
                    os._exit(77)
                return result

            export_images.os.link = crash_during_rollback
            export_images.export_deck([sys.argv[2]], sys.argv[3], force=True)
            raise SystemExit(3)
            """
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
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
            self.assertEqual(list(root.glob(".*.backup*")), [])


if __name__ == "__main__":
    unittest.main()
