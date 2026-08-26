import base64
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images
import gen_slide_openai
import image_output
import prepare_editable_input


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class JsonResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")
        self.stream = io.BytesIO(self.payload)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def swap_parent(parent, redirected):
    original = parent.with_name(f"{parent.name}-original")
    parent.rename(original)
    parent.symlink_to(redirected, target_is_directory=True)
    return original


def assert_no_transaction_files(test_case, directory):
    test_case.assertEqual(list(directory.glob(".*.tmp*")), [])
    test_case.assertEqual(list(directory.glob(".*.backup")), [])


class SharedParentIdentityTests(unittest.TestCase):
    def test_preflight_rejects_a_symlink_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_parent = root / "real"
            real_parent.mkdir()
            alias_parent = root / "alias"
            alias_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "parent directory identity",
            ):
                image_output.preflight_output(str(alias_parent / "slide.jpg"))

            self.assertEqual(list(real_parent.iterdir()), [])

    def test_replaced_parent_inode_is_rejected_before_temp_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            approved.mkdir()
            target = approved / "slide.jpg"
            prepared = image_output.preflight_output(str(target))
            original = approved.with_name("approved-original")
            approved.rename(original)
            approved.mkdir()

            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "parent directory identity changed",
            ):
                image_output.publish_bytes(image_bytes(), prepared)

            self.assertFalse((approved / "slide.jpg").exists())
            self.assertFalse((original / "slide.jpg").exists())
            assert_no_transaction_files(self, approved)
            assert_no_transaction_files(self, original)

    def test_provider_parent_swap_during_network_fails_closed(self):
        generated = image_bytes()
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            target = approved / "slide.jpg"
            original = None

            def swap_then_respond(_request, timeout):
                nonlocal original
                self.assertEqual(timeout, 180)
                original = swap_parent(approved, redirected)
                return response

            with mock.patch.object(
                gen_slide_openai,
                "_load_api_key",
                return_value="key",
            ), mock.patch.object(
                gen_slide_openai.urllib.request,
                "urlopen",
                side_effect=swap_then_respond,
            ), redirect_stdout(io.StringIO()):
                self.assertFalse(
                    gen_slide_openai.gen("prompt", str(target), retries=0)
                )

            self.assertIsNotNone(original)
            self.assertFalse((redirected / "slide.jpg").exists())
            self.assertFalse((original / "slide.jpg").exists())
            assert_no_transaction_files(self, redirected)
            assert_no_transaction_files(self, original)


class LocalParentIdentityTests(unittest.TestCase):
    def test_forced_deck_parent_swap_preserves_original_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            Image.new("RGB", (1600, 900), "navy").save(source)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            prefix = approved / "deck"
            prefix.with_suffix(".pdf").write_bytes(b"old-pdf")
            prefix.with_suffix(".pptx").write_bytes(b"old-pptx")
            real_load_all = export_images._load_all
            original = None

            def swap_then_load(files):
                nonlocal original
                original = swap_parent(approved, redirected)
                return real_load_all(files)

            with mock.patch.object(
                export_images,
                "_load_all",
                side_effect=swap_then_load,
            ), redirect_stderr(io.StringIO()):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual((original / "deck.pdf").read_bytes(), b"old-pdf")
            self.assertEqual((original / "deck.pptx").read_bytes(), b"old-pptx")
            self.assertFalse((redirected / "deck.pdf").exists())
            self.assertFalse((redirected / "deck.pptx").exists())
            assert_no_transaction_files(self, redirected)
            assert_no_transaction_files(self, original)

    def test_prepare_parent_swap_during_source_load_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            Image.new("RGB", (1600, 900), "navy").save(source)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            target = approved / "slide.png"
            real_load_image = prepare_editable_input.load_image
            original = None

            def swap_then_load(path):
                nonlocal original
                original = swap_parent(approved, redirected)
                return real_load_image(path)

            with mock.patch.object(
                prepare_editable_input,
                "load_image",
                side_effect=swap_then_load,
            ), redirect_stdout(io.StringIO()):
                self.assertFalse(
                    prepare_editable_input.prepare(str(source), str(target))
                )

            self.assertFalse((redirected / "slide.png").exists())
            self.assertFalse((original / "slide.png").exists())
            assert_no_transaction_files(self, redirected)
            assert_no_transaction_files(self, original)


if __name__ == "__main__":
    unittest.main()
