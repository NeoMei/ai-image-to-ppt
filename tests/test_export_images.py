import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from PIL import Image
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


class ExportImagesTests(unittest.TestCase):
    def _image(self, path: Path, size=(1600, 900), color="navy") -> None:
        Image.new("RGB", size, color).save(path)

    def test_empty_input_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "deck.pdf"

            with self.assertRaisesRegex(ValueError, "at least one"):
                export_images.export_pdf([], str(target))

            self.assertFalse(target.exists())

    def test_corrupt_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "bad.png"
            target = Path(temp_dir) / "deck.pdf"
            source.write_bytes(b"not an image")
            target.write_bytes(b"existing")

            with self.assertRaises(OSError):
                export_images.export_pdf([str(source)], str(target))

            self.assertEqual(target.read_bytes(), b"existing")

    def test_nested_output_parent_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            target = Path(temp_dir) / "nested" / "deck.pptx"
            self._image(source)

            export_images.export_pptx([str(source)], str(target))

            self.assertTrue(target.is_file())

    def test_relative_output_prefix_is_not_redirected_when_cwd_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            (redirected / "output").mkdir()
            source = root / "slide.png"
            self._image(source)
            real_load_all = export_images._load_all
            original_cwd = Path.cwd()

            def change_cwd_then_load(files):
                os.chdir(redirected)
                return real_load_all(files)

            try:
                os.chdir(approved)
                with mock.patch.object(
                    export_images,
                    "_load_all",
                    side_effect=change_cwd_then_load,
                ):
                    result = export_images.main(
                        ["output/deck", str(source)]
                    )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result, 0)
            self.assertTrue((approved / "output" / "deck.pdf").is_file())
            self.assertTrue((approved / "output" / "deck.pptx").is_file())
            self.assertFalse((redirected / "output" / "deck.pdf").exists())
            self.assertFalse((redirected / "output" / "deck.pptx").exists())

    def test_pptx_uses_exact_sixteen_by_nine_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            target = Path(temp_dir) / "deck.pptx"
            self._image(source)

            export_images.export_pptx([str(source)], str(target))
            presentation = Presentation(target)

            self.assertEqual(presentation.slide_width, 12_192_000)
            self.assertEqual(presentation.slide_height, 6_858_000)
            self.assertEqual(
                presentation.slide_width * 9,
                presentation.slide_height * 16,
            )

    def test_transparency_is_composited_onto_cream_not_black(self):
        image = Image.new("RGBA", (1600, 900), (255, 0, 0, 0))

        normalized = export_images.normalize(image)

        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.getpixel((100, 100)), (248, 245, 240))

    def test_cli_help_exits_zero(self):
        with self.assertRaises(SystemExit) as caught:
            export_images.main(["--help"])

        self.assertEqual(caught.exception.code, 0)

    def test_cli_missing_image_is_controlled_and_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir) / "nested" / "deck"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = export_images.main([str(prefix), "missing.png"])

            self.assertEqual(result, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())

    def test_cli_pair_publication_rolls_back_if_second_publish_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            self._image(source)
            real_link = export_images.os.link
            publish_count = 0

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                if str(target_path).endswith((".pdf", ".pptx")):
                    publish_count += 1
                    if publish_count == 2:
                        raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_publish,
            ):
                result = export_images.main([str(prefix), str(source)])

            self.assertEqual(result, 1)
            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())

    def test_cli_force_failure_restores_both_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            new_publish_count = 0

            def fail_second_new_publish(source_path, target_path, *args, **kwargs):
                nonlocal new_publish_count
                if str(target_path).endswith((".pdf", ".pptx")):
                    new_publish_count += 1
                    if new_publish_count == 2:
                        raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_new_publish,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

    def test_cli_default_pdf_race_preserves_external_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            sentinel = b"external-pdf-writer"
            self._image(source)
            real_link = export_images.os.link
            injected = False

            def inject_pdf_before_publish(source_path, target_path, *args, **kwargs):
                nonlocal injected
                if Path(target_path) == pdf and not injected:
                    pdf.write_bytes(sentinel)
                    injected = True
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=inject_pdf_before_publish,
            ):
                result = export_images.main([str(prefix), str(source)])

            self.assertTrue(injected)
            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), sentinel)
            self.assertFalse(pptx.exists())

    def test_cli_default_pptx_race_preserves_external_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            pdf_sentinel = b"external-pdf-replacement"
            pptx_sentinel = b"external-pptx-writer"
            self._image(source)
            real_link = export_images.os.link
            pdf_replaced = False
            pptx_injected = False

            def race_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal pdf_replaced, pptx_injected
                target = Path(target_path)
                if target == pdf:
                    result = real_link(source_path, target_path, *args, **kwargs)
                    pdf.unlink()
                    pdf.write_bytes(pdf_sentinel)
                    pdf_replaced = True
                    return result
                if target == pptx and not pptx_injected:
                    pptx.write_bytes(pptx_sentinel)
                    pptx_injected = True
                return real_link(source_path, target_path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=race_second_publish,
            ), redirect_stderr(stderr):
                result = export_images.main([str(prefix), str(source)])

            self.assertTrue(pdf_replaced)
            self.assertTrue(pptx_injected)
            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), pdf_sentinel)
            self.assertEqual(pptx.read_bytes(), pptx_sentinel)
            self.assertIn("ownership changed", stderr.getvalue())

    def test_force_rollback_unlink_failure_still_restores_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            real_replace = export_images.os.replace
            real_unlink = export_images.os.unlink
            publish_count = 0
            restore_attempts = []

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                if publish_count == 2:
                    raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            def fail_published_cleanup(path, *args, **kwargs):
                if Path(path) == pdf:
                    raise OSError("forced published unlink failure")
                return real_unlink(path, *args, **kwargs)

            def record_restore(source_path, target_path, *args, **kwargs):
                if str(source_path).endswith(".backup"):
                    restore_attempts.append(Path(target_path))
                return real_replace(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_publish,
            ), mock.patch.object(
                export_images.os,
                "unlink",
                side_effect=fail_published_cleanup,
            ), mock.patch.object(
                export_images.os,
                "replace",
                side_effect=record_restore,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(restore_attempts, [pptx])
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertTrue(pdf.is_file())
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

    def test_force_rollback_restores_backed_targets_without_unlinking_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            real_unlink = export_images.os.unlink
            publish_count = 0
            target_unlinks = []

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                if publish_count == 2:
                    raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            def reject_target_unlink(path, *args, **kwargs):
                target = Path(path)
                if target in (pdf, pptx):
                    target_unlinks.append(target)
                    raise OSError("backed target must be restored directly")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_publish,
            ), mock.patch.object(
                export_images.os,
                "unlink",
                side_effect=reject_target_unlink,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(target_unlinks, [])
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

    def test_force_rollback_attempts_every_restore_and_preserves_failed_backup(self):
        for failed_name in ("deck.pdf", "deck.pptx"):
            with self.subTest(failed_name=failed_name), tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / "slide.png"
                prefix = Path(temp_dir) / "deck"
                pdf = prefix.with_suffix(".pdf")
                pptx = prefix.with_suffix(".pptx")
                self._image(source)
                pdf.write_bytes(b"old-pdf")
                pptx.write_bytes(b"old-pptx")
                real_link = export_images.os.link
                real_replace = export_images.os.replace
                publish_count = 0
                restore_attempts = []

                def fail_second_publish(source_path, target_path, *args, **kwargs):
                    nonlocal publish_count
                    publish_count += 1
                    if publish_count == 2:
                        raise OSError("forced second publish failure")
                    return real_link(source_path, target_path, *args, **kwargs)

                def fail_selected_restore(source_path, target_path, *args, **kwargs):
                    if str(source_path).endswith(".backup"):
                        target = Path(target_path)
                        restore_attempts.append(target.name)
                        if target.name == failed_name:
                            raise OSError(f"forced restore failure for {failed_name}")
                    return real_replace(source_path, target_path, *args, **kwargs)

                stderr = io.StringIO()
                with mock.patch.object(
                    export_images.os,
                    "link",
                    side_effect=fail_second_publish,
                ), mock.patch.object(
                    export_images.os,
                    "replace",
                    side_effect=fail_selected_restore,
                ), redirect_stderr(stderr):
                    result = export_images.main(
                        ["--force", str(prefix), str(source)]
                    )

                self.assertEqual(result, 1)
                self.assertEqual(restore_attempts, ["deck.pdf", "deck.pptx"])
                failed_backups = [
                    path
                    for path in Path(temp_dir).glob(".*.backup")
                    if f".{failed_name}." in path.name
                ]
                self.assertEqual(len(failed_backups), 1)
                expected = b"old-pdf" if failed_name.endswith(".pdf") else b"old-pptx"
                self.assertEqual(failed_backups[0].read_bytes(), expected)
                restored = pptx if failed_name.endswith(".pdf") else pdf
                restored_expected = b"old-pptx" if restored == pptx else b"old-pdf"
                self.assertEqual(restored.read_bytes(), restored_expected)
                self.assertIn(str(failed_backups[0]), stderr.getvalue())

    def test_second_temporary_file_setup_failure_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            self._image(source)
            real_mkstemp = export_images.tempfile.mkstemp
            calls = 0

            def fail_second_mkstemp(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("forced temp setup failure")
                return real_mkstemp(*args, **kwargs)

            with mock.patch.object(
                export_images.tempfile,
                "mkstemp",
                side_effect=fail_second_mkstemp,
            ):
                result = export_images.main([str(prefix), str(source)])

            self.assertEqual(result, 1)
            self.assertEqual(list(Path(temp_dir).glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
