import io
import os
import sys
import tempfile
import threading
import urllib.error
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images
import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import image_output
import output_lock
import prepare_editable_input


class DeckInputBoundaryTests(unittest.TestCase):
    def test_deck_resource_limits_are_documented(self):
        for document_name in ("README.md", "SKILL.md"):
            with self.subTest(document=document_name):
                document = (ROOT / document_name).read_text(encoding="utf-8")
                normalized = " ".join(document.split())
                self.assertIn("128 slides", normalized)
                self.assertIn("512 MiB", normalized)

    def test_generator_at_slide_and_byte_limits_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = (root / "one.bin", root / "two.bin")
            sources[0].write_bytes(b"12")
            sources[1].write_bytes(b"345")
            prefix = root / "deck"
            consumed = []

            def generated_sources():
                for source in sources:
                    consumed.append(source)
                    yield str(source)

            def save(_images, path):
                Path(path).write_bytes(b"artifact")

            with mock.patch.object(export_images, "MAX_DECK_SLIDES", 2, create=True), \
                 mock.patch.object(export_images, "MAX_DECK_SOURCE_BYTES", 5, create=True), \
                 mock.patch.object(export_images, "_load_all", return_value=["a", "b"]) as load_all, \
                 mock.patch.object(export_images, "_save_pdf", side_effect=save), \
                 mock.patch.object(export_images, "_save_pptx", side_effect=save), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = export_images.export_deck(generated_sources(), str(prefix))

            self.assertTrue(result)
            self.assertEqual(consumed, list(sources))
            load_all.assert_called_once()

    def test_slide_limit_stops_generator_before_decode_or_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = []
            for index in range(4):
                source = root / f"{index}.bin"
                source.write_bytes(b"x")
                sources.append(source)
            consumed = []

            def generated_sources():
                for source in sources:
                    consumed.append(source)
                    yield str(source)

            stderr = io.StringIO()
            with mock.patch.object(export_images, "MAX_DECK_SLIDES", 2, create=True), \
                 mock.patch.object(export_images, "MAX_DECK_SOURCE_BYTES", 100, create=True), \
                 mock.patch.object(export_images, "_load") as decode, \
                 mock.patch.object(export_images, "_save_pdf") as save_pdf, \
                 mock.patch.object(export_images, "_save_pptx") as save_pptx, \
                 redirect_stderr(stderr):
                result = export_images.export_deck(
                    generated_sources(), str(root / "deck")
                )

            self.assertFalse(result)
            self.assertEqual(consumed, sources[:3])
            self.assertIn("at most 2", stderr.getvalue())
            decode.assert_not_called()
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()

    def test_aggregate_byte_limit_rejects_before_decode_or_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "one.bin"
            second = root / "two.bin"
            first.write_bytes(b"123")
            second.write_bytes(b"456")
            stderr = io.StringIO()

            with mock.patch.object(export_images, "MAX_DECK_SLIDES", 10, create=True), \
                 mock.patch.object(export_images, "MAX_DECK_SOURCE_BYTES", 5, create=True), \
                 mock.patch.object(export_images, "_load") as decode, \
                 mock.patch.object(export_images, "_save_pdf") as save_pdf, \
                 mock.patch.object(export_images, "_save_pptx") as save_pptx, \
                 redirect_stderr(stderr):
                result = export_images.export_deck(
                    (str(path) for path in (first, second)), str(root / "deck")
                )

            self.assertFalse(result)
            self.assertIn("source bytes", stderr.getvalue())
            decode.assert_not_called()
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()

    def test_cli_maps_slide_limit_to_exit_one_without_traceback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = []
            for index in range(3):
                source = root / f"{index}.bin"
                source.write_bytes(b"x")
                sources.append(str(source))
            stderr = io.StringIO()

            with mock.patch.object(export_images, "MAX_DECK_SLIDES", 2, create=True), \
                 mock.patch.object(export_images, "MAX_DECK_SOURCE_BYTES", 100, create=True), \
                 mock.patch.object(export_images, "_load") as decode, \
                 redirect_stderr(stderr):
                result = export_images.main([str(root / "deck"), *sources])

            self.assertEqual(result, 1)
            self.assertIn("at most 2", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            decode.assert_not_called()


class NarrowNormalizeTests(unittest.TestCase):
    def test_one_by_two_red_image_retains_content(self):
        normalized = export_images.normalize(Image.new("RGB", (1, 2), "red"))

        self.assertEqual(normalized.size, export_images.TARGET_SIZE)
        self.assertEqual(normalized.getpixel((960, 540)), (255, 0, 0))


class PreparedTargetLockBoundaryTests(unittest.TestCase):
    @staticmethod
    def _swap_missing_ancestor(missing: Path, redirected: Path) -> None:
        if missing.exists() and not missing.is_symlink():
            missing.rename(missing.with_name(f"{missing.name}-captured"))
        missing.symlink_to(redirected, target_is_directory=True)

    def test_preflight_reuses_captured_prepared_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.prepare_target(
                Path(temp_dir) / "nested" / "slide.jpg"
            )
            with mock.patch.object(
                image_output,
                "prepare_target",
                side_effect=AssertionError("target identity was recaptured"),
            ):
                result = image_output.preflight_output(prepared)

            self.assertIs(result, prepared)

    def test_preflight_verifies_parent_before_inspecting_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "captured"
            redirected = root / "redirected"
            parent.mkdir()
            redirected.mkdir()
            prepared = image_output.prepare_target(parent / "slide.jpg")
            parent.rename(root / "captured-original")
            parent.symlink_to(redirected, target_is_directory=True)

            with mock.patch.object(image_output.os.path, "lexists") as lexists:
                with self.assertRaisesRegex(
                    image_output.ImageOutputError,
                    "parent directory identity changed",
                ):
                    image_output.preflight_output(prepared)

            lexists.assert_not_called()

    def test_all_providers_reject_ancestor_swap_before_credentials_or_network(self):
        providers = (gen_slide_openai, gen_slide_gemini, gen_slide_doubao)
        for provider in providers:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                missing = root / "missing"
                redirected = root / "redirected"
                (redirected / "deep").mkdir(parents=True)
                target = missing / "deep" / "slide.jpg"

                @contextmanager
                def swap_before_yield(_target, namespace="output"):
                    self._swap_missing_ancestor(missing, redirected)
                    yield

                with mock.patch.object(provider, "output_lock", swap_before_yield), \
                     mock.patch.object(provider, "_load_api_key") as load_key, \
                     mock.patch.object(provider.urllib.request, "urlopen") as urlopen, \
                     redirect_stdout(io.StringIO()):
                    result = provider.gen("prompt", str(target), retries=0)

                self.assertFalse(result)
                load_key.assert_not_called()
                urlopen.assert_not_called()

    def test_swapped_first_provider_and_second_actual_target_do_not_both_pay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing"
            redirected = root / "redirected"
            (redirected / "deep").mkdir(parents=True)
            target = missing / "deep" / "slide.jpg"
            swapped = threading.Event()
            entered_network = threading.Event()
            release_network = threading.Event()
            both_network = threading.Event()
            calls_guard = threading.Lock()
            network_calls = []
            real_output_lock = output_lock.output_lock

            @contextmanager
            def swap_after_lock(lock_target, namespace="output"):
                with real_output_lock(lock_target, namespace=namespace):
                    if not swapped.is_set():
                        self._swap_missing_ancestor(missing, redirected)
                        swapped.set()
                    yield

            def fail_network(_request, timeout):
                self.assertEqual(timeout, 180)
                with calls_guard:
                    network_calls.append(threading.current_thread().name)
                    if len(network_calls) == 2:
                        both_network.set()
                entered_network.set()
                self.assertTrue(release_network.wait(2))
                raise urllib.error.URLError("offline test")

            results = []
            with mock.patch.object(gen_slide_openai, "output_lock", swap_after_lock), \
                 mock.patch.object(gen_slide_openai, "_load_api_key", return_value="key"), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     side_effect=fail_network,
                 ), redirect_stdout(io.StringIO()):
                first = threading.Thread(
                    name="first",
                    target=lambda: results.append(
                        gen_slide_openai.gen("first", str(target), retries=0)
                    ),
                )
                first.start()
                self.assertTrue(swapped.wait(2))
                second = threading.Thread(
                    name="second",
                    target=lambda: results.append(
                        gen_slide_openai.gen("second", str(target), retries=0)
                    ),
                )
                second.start()
                self.assertTrue(entered_network.wait(2))
                both_network.wait(0.5)
                release_network.set()
                first.join(timeout=2)
                second.join(timeout=2)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(network_calls, ["second"])
            self.assertEqual(results, [False, False])

    def test_prepare_rejects_ancestor_swap_before_reading_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            source.write_bytes(b"not read")
            missing = root / "missing"
            redirected = root / "redirected"
            (redirected / "deep").mkdir(parents=True)
            target = missing / "deep" / "slide.png"

            @contextmanager
            def swap_before_yield(_target, namespace="output"):
                self._swap_missing_ancestor(missing, redirected)
                yield

            with mock.patch.object(
                prepare_editable_input, "output_lock", swap_before_yield
            ), mock.patch.object(
                prepare_editable_input, "load_image"
            ) as load_image, redirect_stdout(io.StringIO()):
                result = prepare_editable_input.prepare(str(source), str(target))

            self.assertFalse(result)
            load_image.assert_not_called()

    def test_deck_rejects_ancestor_swap_before_decode_or_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing"
            redirected = root / "redirected"
            (redirected / "deep").mkdir(parents=True)
            prefix = missing / "deep" / "deck"

            @contextmanager
            def swap_before_yield(_target, namespace="output"):
                self._swap_missing_ancestor(missing, redirected)
                yield

            with mock.patch.object(export_images, "output_lock", swap_before_yield), \
                 mock.patch.object(export_images, "_load_all") as load_all, \
                 mock.patch.object(export_images, "_save_pdf") as save_pdf, \
                 mock.patch.object(export_images, "_save_pptx") as save_pptx, \
                 redirect_stderr(io.StringIO()):
                result = export_images.export_deck(["ignored"], str(prefix))

            self.assertFalse(result)
            load_all.assert_not_called()
            save_pdf.assert_not_called()
            save_pptx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
