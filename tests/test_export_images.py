import io
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
            real_replace = export_images.os.replace
            publish_count = 0

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                if str(target_path).endswith((".pdf", ".pptx")):
                    publish_count += 1
                    if publish_count == 2:
                        raise OSError("forced second publish failure")
                return real_replace(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "replace",
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
            real_replace = export_images.os.replace
            new_publish_count = 0

            def fail_second_new_publish(source_path, target_path, *args, **kwargs):
                nonlocal new_publish_count
                if ".tmp" in str(source_path) and str(target_path).endswith(
                    (".pdf", ".pptx")
                ):
                    new_publish_count += 1
                    if new_publish_count == 2:
                        raise OSError("forced second publish failure")
                return real_replace(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "replace",
                side_effect=fail_second_new_publish,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

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
