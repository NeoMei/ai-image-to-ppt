import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_editable_input


class PrepareEditableInputTests(unittest.TestCase):
    def test_converts_16_by_9_jpeg_to_exact_1280_by_720_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "editable" / "slide.png"
            Image.new("RGB", (2048, 1152), "navy").save(source, format="JPEG")
            source_before = source.read_bytes()

            self.assertTrue(prepare_editable_input.prepare(str(source), str(target)))

            self.assertEqual(source.read_bytes(), source_before)
            with Image.open(target) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1280, 720))
                self.assertEqual(image.mode, "RGB")

    def test_rejects_non_16_by_9_source_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "square.png"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1024, 1024), "white").save(source)

            self.assertFalse(prepare_editable_input.prepare(str(source), str(target)))
            self.assertFalse(target.exists())

    def test_rejects_non_png_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.jpg"
            Image.new("RGB", (1920, 1080), "white").save(source)

            self.assertFalse(prepare_editable_input.prepare(str(source), str(target)))
            self.assertFalse(target.exists())

    def test_rejects_source_and_target_as_the_same_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1280, 720), "white").save(source)
            source_before = source.read_bytes()

            self.assertFalse(prepare_editable_input.prepare(str(source), str(source)))
            self.assertEqual(source.read_bytes(), source_before)

    def test_existing_target_is_preserved_and_no_temp_file_remains(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (2560, 1440), "white").save(source)
            target.write_bytes(b"existing-output")

            self.assertFalse(prepare_editable_input.prepare(str(source), str(target)))
            self.assertEqual(target.read_bytes(), b"existing-output")
            self.assertEqual(list(Path(temp_dir).glob(f".{target.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
