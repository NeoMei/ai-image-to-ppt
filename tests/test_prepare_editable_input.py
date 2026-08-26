import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_editable_input


class PrepareEditableInputTests(unittest.TestCase):
    def test_transparency_is_composited_onto_cream_not_black(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.png"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGBA", (1600, 900), (255, 0, 0, 0)).save(source)

            self.assertTrue(prepare_editable_input.prepare(str(source), str(target)))

            with Image.open(target) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((100, 100)), (248, 245, 240))

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

    def test_prepare_freezes_relative_target_before_lock_yields_and_cwd_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            source = root / "master.jpg"
            Image.new("RGB", (1600, 900), "navy").save(source, format="JPEG")
            original_cwd = Path.cwd()

            @contextmanager
            def change_cwd_before_yield(_target, namespace="output"):
                os.chdir(redirected)
                yield

            try:
                os.chdir(approved)
                with mock.patch.object(
                    prepare_editable_input,
                    "output_lock",
                    change_cwd_before_yield,
                ):
                    self.assertTrue(
                        prepare_editable_input.prepare(
                            str(source), "output/slide.png"
                        )
                    )
            finally:
                os.chdir(original_cwd)

            self.assertTrue((approved / "output" / "slide.png").is_file())
            self.assertFalse((redirected / "output" / "slide.png").exists())

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

    def test_output_setup_failure_returns_false_without_damage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            blocked_parent = Path(temp_dir) / "not-a-directory"
            target = blocked_parent / "slide.png"
            Image.new("RGB", (1920, 1080), "navy").save(source, format="JPEG")
            source_before = source.read_bytes()
            blocked_parent.write_bytes(b"existing-parent-file")

            try:
                result = prepare_editable_input.prepare(str(source), str(target))
            except OSError as error:
                self.fail(f"prepare raised instead of returning False: {error}")

            self.assertFalse(result)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(blocked_parent.read_bytes(), b"existing-parent-file")

    def test_invalid_output_target_is_controlled_before_reading_source(self):
        output = io.StringIO()
        with mock.patch.object(prepare_editable_input, "load_image") as load_image, \
             redirect_stdout(output):
            self.assertFalse(
                prepare_editable_input.prepare("ignored.jpg", "bad\0target.png")
            )

        load_image.assert_not_called()
        self.assertIn("invalid output target", output.getvalue())

    def test_png_save_failure_removes_created_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1920, 1080), "navy").save(source, format="JPEG")
            source_before = source.read_bytes()

            with mock.patch.object(
                Image.Image,
                "save",
                side_effect=OSError("forced PNG save failure"),
            ):
                self.assertFalse(
                    prepare_editable_input.prepare(str(source), str(target))
                )

            self.assertEqual(source.read_bytes(), source_before)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temp_dir).glob(f".{target.name}.*.tmp")), [])

    def test_publish_race_preserves_target_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1920, 1080), "navy").save(source, format="JPEG")

            def create_racing_target(_temp_path, output_path):
                Path(output_path).write_bytes(b"racing-output")
                raise FileExistsError("forced publish race")

            with mock.patch.object(
                prepare_editable_input.os,
                "link",
                side_effect=create_racing_target,
            ):
                self.assertFalse(
                    prepare_editable_input.prepare(str(source), str(target))
                )

            self.assertEqual(target.read_bytes(), b"racing-output")
            self.assertEqual(list(Path(temp_dir).glob(f".{target.name}.*.tmp")), [])

    def test_transient_post_publish_cleanup_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1920, 1080), "navy").save(source, format="JPEG")
            real_unlink = prepare_editable_input.os.unlink
            attempts = 0

            def fail_once_then_unlink(path):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("transient cleanup failure")
                real_unlink(path)

            with mock.patch.object(
                prepare_editable_input.os,
                "unlink",
                side_effect=fail_once_then_unlink,
            ):
                self.assertTrue(
                    prepare_editable_input.prepare(str(source), str(target))
                )

            self.assertEqual(attempts, 2)
            with Image.open(target) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1280, 720))
                self.assertEqual(image.mode, "RGB")
            self.assertEqual(list(Path(temp_dir).glob(f".{target.name}.*.tmp")), [])

    def test_unavoidable_post_publish_cleanup_failure_warns_but_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1920, 1080), "navy").save(source, format="JPEG")
            output = io.StringIO()

            with mock.patch.object(
                prepare_editable_input.os,
                "unlink",
                side_effect=OSError("persistent cleanup failure"),
            ), redirect_stdout(output):
                try:
                    result = prepare_editable_input.prepare(str(source), str(target))
                except OSError as error:
                    self.fail(f"prepare raised after publishing target: {error}")

            self.assertTrue(result)
            self.assertTrue(target.exists())
            self.assertIn("WARN:", output.getvalue())
            self.assertNotIn("failed to write normalized PNG", output.getvalue())


if __name__ == "__main__":
    unittest.main()
