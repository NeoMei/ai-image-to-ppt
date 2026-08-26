import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import output_lock
import export_images
import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import prepare_editable_input


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (160, 90), "navy").save(buffer, format="JPEG")
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


class OutputLockModuleTests(unittest.TestCase):
    def test_output_lock_module_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("output_lock"))

    def test_same_target_thread_contender_fails_fast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            result = []
            finished = threading.Event()

            def contend():
                try:
                    with output_lock.output_lock(target):
                        result.append("acquired")
                except output_lock.OutputLockBusy:
                    result.append("busy")
                finally:
                    finished.set()

            with output_lock.output_lock(target):
                thread = threading.Thread(target=contend)
                thread.start()
                self.assertTrue(finished.wait(0.5), "contender did not fail fast")
                thread.join()

            self.assertEqual(result, ["busy"])

    def test_same_target_process_contender_fails_fast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; from output_lock import output_lock; "
                        "\nwith output_lock(sys.argv[1]):"
                        "\n print('locked', flush=True); sys.stdin.readline()"
                    ),
                    str(target),
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                started = time.monotonic()
                with self.assertRaises(output_lock.OutputLockBusy):
                    with output_lock.output_lock(target):
                        pass
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                child.communicate("\n", timeout=5)

    def test_process_crash_releases_lock_and_lock_file_is_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            first_path = output_lock.lock_file_for(target)
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os, sys; from output_lock import output_lock; "
                        "\nwith output_lock(sys.argv[1]):"
                        "\n print('locked', flush=True); os._exit(17)"
                    ),
                    str(target),
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(child.returncode, 17)
            self.assertEqual(child.stdout.strip(), "locked")

            with output_lock.output_lock(target):
                self.assertEqual(output_lock.lock_file_for(target), first_path)
                self.assertTrue(first_path.is_file())

            self.assertTrue(first_path.is_file())


class OutputOwnershipIntegrationTests(unittest.TestCase):
    def test_busy_provider_returns_false_before_preflight_credentials_or_network(self):
        providers = (gen_slide_openai, gen_slide_gemini, gen_slide_doubao)
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            for provider in providers:
                with self.subTest(provider=provider.__name__), \
                     output_lock.output_lock(target), \
                     mock.patch.object(provider, "preflight_output") as preflight, \
                     mock.patch.object(provider, "_load_api_key") as load_key, \
                     mock.patch.object(provider.urllib.request, "urlopen") as urlopen, \
                     redirect_stdout(io.StringIO()):
                    self.assertFalse(
                        provider.gen("prompt", str(target), retries=0)
                    )
                preflight.assert_not_called()
                load_key.assert_not_called()
                urlopen.assert_not_called()

    def test_concurrent_openai_generation_calls_network_once_and_one_succeeds(self):
        generated = image_bytes()
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}]
        })
        entered_network = threading.Event()
        release_network = threading.Event()
        results = []

        def fake_urlopen(_request, timeout):
            self.assertEqual(timeout, 180)
            entered_network.set()
            self.assertTrue(release_network.wait(2))
            return response

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(gen_slide_openai, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 side_effect=fake_urlopen,
             ) as urlopen, redirect_stdout(io.StringIO()):
            target = Path(temp_dir) / "slide.jpg"

            first = threading.Thread(
                target=lambda: results.append(
                    gen_slide_openai.gen("first", str(target), retries=0)
                )
            )
            second = threading.Thread(
                target=lambda: results.append(
                    gen_slide_openai.gen("second", str(target), retries=0)
                )
            )
            first.start()
            self.assertTrue(entered_network.wait(2))
            second.start()
            second.join(timeout=2)
            self.assertFalse(second.is_alive(), "busy generator did not fail fast")
            release_network.set()
            first.join(timeout=2)

            self.assertCountEqual(results, [True, False])
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(target.read_bytes(), generated)
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_busy_prepare_returns_false_before_reading_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.jpg"
            target = Path(temp_dir) / "slide.png"
            with output_lock.output_lock(target), \
                 mock.patch.object(prepare_editable_input.Image, "open") as image_open, \
                 redirect_stdout(io.StringIO()):
                self.assertFalse(
                    prepare_editable_input.prepare(str(source), str(target))
                )
            image_open.assert_not_called()

    def test_busy_export_returns_false_and_cli_returns_one_before_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir) / "deck"
            with output_lock.output_lock(prefix, namespace="deck"), \
                 mock.patch.object(export_images, "_load_all") as load_all, \
                 redirect_stderr(io.StringIO()):
                self.assertFalse(export_images.export_deck(["slide.png"], str(prefix)))
                self.assertEqual(
                    export_images.main([str(prefix), "slide.png"]),
                    1,
                )
            load_all.assert_not_called()

    def test_concurrent_same_prefix_exports_one_consistent_pair(self):
        entered_save = threading.Event()
        release_save = threading.Event()
        results = []

        def fake_load_all(files):
            return [Path(path).name for path in files]

        def fake_save_pdf(images, path):
            Path(path).write_bytes(f"{images[0]}-PDF".encode("ascii"))
            if images[0] == "A":
                entered_save.set()
                self.assertTrue(release_save.wait(2))

        def fake_save_pptx(images, path):
            Path(path).write_bytes(f"{images[0]}-PPTX".encode("ascii"))

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(export_images, "_load_all", side_effect=fake_load_all), \
             mock.patch.object(export_images, "_save_pdf", side_effect=fake_save_pdf), \
             mock.patch.object(export_images, "_save_pptx", side_effect=fake_save_pptx), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            prefix = Path(temp_dir) / "deck"
            first_source = Path(temp_dir) / "A"
            second_source = Path(temp_dir) / "B"
            first_source.write_bytes(b"A")
            second_source.write_bytes(b"B")
            first = threading.Thread(
                target=lambda: results.append(
                    export_images.export_deck([str(first_source)], str(prefix))
                )
            )
            second = threading.Thread(
                target=lambda: results.append(
                    export_images.export_deck([str(second_source)], str(prefix))
                )
            )

            first.start()
            self.assertTrue(entered_save.wait(2))
            second.start()
            second.join(timeout=2)
            self.assertFalse(second.is_alive(), "busy exporter did not fail fast")
            release_save.set()
            first.join(timeout=2)

            self.assertCountEqual(results, [True, False])
            self.assertEqual(prefix.with_suffix(".pdf").read_bytes(), b"A-PDF")
            self.assertEqual(prefix.with_suffix(".pptx").read_bytes(), b"A-PPTX")
            self.assertEqual(list(Path(temp_dir).glob(".*.tmp.*")), [])
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])
            self.assertEqual(list(Path(temp_dir).glob("*.lock")), [])


if __name__ == "__main__":
    unittest.main()
