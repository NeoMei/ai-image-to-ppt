import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


def loaded_image(byte_count):
    image = Image.new("RGB", (160, 90), "navy")
    image.info["ai_image_to_ppt_source_bytes"] = byte_count
    return image


class LoadedByteShortCircuitTests(unittest.TestCase):
    def test_python_api_stops_loading_when_second_image_exceeds_actual_byte_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [root / f"{index}.img" for index in range(3)]
            for source in sources:
                source.write_bytes(b"x")
            prefix = root / "deck"
            stderr = io.StringIO()

            with mock.patch.object(
                export_images,
                "MAX_DECK_SOURCE_BYTES",
                5,
            ), mock.patch.object(
                export_images,
                "_load",
                side_effect=[
                    loaded_image(3),
                    loaded_image(3),
                    loaded_image(1),
                ],
            ) as load, mock.patch.object(
                export_images,
                "_save_pdf",
            ) as save_pdf, mock.patch.object(
                export_images,
                "_save_pptx",
            ) as save_pptx, redirect_stderr(stderr):
                result = export_images.export_deck(
                    [str(source) for source in sources],
                    str(prefix),
                )

            self.assertFalse(result)
            self.assertEqual(
                load.call_args_list,
                [
                    mock.call(str(sources[0].resolve())),
                    mock.call(str(sources[1].resolve())),
                ],
            )
            self.assertIn("aggregate source bytes", stderr.getvalue())
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()
            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())

    def test_cli_returns_one_and_stops_loading_after_actual_byte_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [root / f"{index}.img" for index in range(3)]
            for source in sources:
                source.write_bytes(b"x")
            stderr = io.StringIO()

            with mock.patch.object(
                export_images,
                "MAX_DECK_SOURCE_BYTES",
                5,
            ), mock.patch.object(
                export_images,
                "_load",
                side_effect=[
                    loaded_image(3),
                    loaded_image(3),
                    loaded_image(1),
                ],
            ) as load, mock.patch.object(
                export_images,
                "_save_pdf",
            ) as save_pdf, mock.patch.object(
                export_images,
                "_save_pptx",
            ) as save_pptx, redirect_stderr(stderr):
                result = export_images.main(
                    [str(root / "deck"), *[str(source) for source in sources]]
                )

            self.assertEqual(result, 1)
            self.assertEqual(
                load.call_args_list,
                [
                    mock.call(str(sources[0].resolve())),
                    mock.call(str(sources[1].resolve())),
                ],
            )
            self.assertIn("aggregate source bytes", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()

    def test_toctou_replacements_stop_before_loading_third_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = [root / f"{index}.img" for index in range(3)]
            for source in sources:
                Image.new("RGB", (160, 90), "navy").save(source, format="PNG")
            replacements = [root / f"replacement-{index}.bmp" for index in range(2)]
            for replacement in replacements:
                Image.effect_noise((1000, 1000), 100).convert("RGB").save(
                    replacement,
                    format="BMP",
                )
            limit = sum(path.stat().st_size for path in replacements) - 1

            def replacing_inputs():
                yield str(sources[0])
                yield str(sources[1])
                yield str(sources[2])
                shutil.copyfile(replacements[0], sources[0])
                shutil.copyfile(replacements[1], sources[1])

            stderr = io.StringIO()
            with mock.patch.object(
                export_images,
                "MAX_DECK_SOURCE_BYTES",
                limit,
            ), mock.patch.object(
                export_images,
                "_load",
                wraps=export_images._load,
            ) as load, mock.patch.object(
                export_images,
                "_save_pdf",
            ) as save_pdf, mock.patch.object(
                export_images,
                "_save_pptx",
            ) as save_pptx, redirect_stderr(stderr):
                result = export_images.export_deck(
                    replacing_inputs(),
                    str(root / "deck"),
                )

            self.assertFalse(result)
            self.assertEqual(
                load.call_args_list,
                [
                    mock.call(str(sources[0].resolve())),
                    mock.call(str(sources[1].resolve())),
                ],
            )
            self.assertIn("aggregate source bytes", stderr.getvalue())
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()


class LoadedByteDocumentationTests(unittest.TestCase):
    def test_deck_byte_limit_documents_preflight_and_post_preflight_checks(self):
        expected = (
            "Clearly over-limit manifests are rejected during path preflight, "
            "before image decoding. If source files change after preflight, "
            "actual loaded bytes are accumulated after each image load and "
            "rejected before PDF/PPTX serialization."
        )
        for document_name in ("README.md", "SKILL.md"):
            with self.subTest(document=document_name):
                document = (ROOT / document_name).read_text(encoding="utf-8")
                self.assertIn(expected, " ".join(document.split()))


if __name__ == "__main__":
    unittest.main()
