import base64
import builtins
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import types
import unicodedata
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_images
import gen_slide_openai
import image_output
import output_lock


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

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def aliases_share_file(parent, first_name, second_name):
    first = parent / first_name
    second = parent / second_name
    first.write_bytes(b"probe")
    try:
        return second.exists() and os.path.samefile(first, second)
    finally:
        first.unlink()


def load_output_lock_without(*blocked_names, fake_msvcrt=None):
    module_name = "output_lock_isolated_test"
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPTS / "output_lock.py"
    )
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_names:
            raise ImportError(f"blocked test import: {name}")
        if name == "msvcrt" and fake_msvcrt is not None:
            return fake_msvcrt
        return real_import(name, globals, locals, fromlist, level)

    try:
        with mock.patch.object(builtins, "__import__", side_effect=controlled_import):
            spec.loader.exec_module(module)
    except ImportError as error:
        return None, error
    return module, None


class FilesystemAliasLockTests(unittest.TestCase):
    def test_lock_key_normalization_follows_detected_filesystem_semantics(self):
        composed = Path("slide-é.jpg")
        decomposed = Path(unicodedata.normalize("NFD", str(composed)))

        with mock.patch.object(
            output_lock,
            "_filesystem_semantics",
            return_value=(True, False),
        ):
            self.assertEqual(
                output_lock.lock_file_for("slide.jpg"),
                output_lock.lock_file_for("SLIDE.JPG"),
            )
            self.assertNotEqual(
                output_lock.lock_file_for(composed),
                output_lock.lock_file_for(decomposed),
            )

        with mock.patch.object(
            output_lock,
            "_filesystem_semantics",
            return_value=(False, True),
        ):
            self.assertNotEqual(
                output_lock.lock_file_for("slide.jpg"),
                output_lock.lock_file_for("SLIDE.JPG"),
            )
            self.assertEqual(
                output_lock.lock_file_for(composed),
                output_lock.lock_file_for(decomposed),
            )

        with mock.patch.object(
            output_lock,
            "_filesystem_semantics",
            return_value=(False, False),
        ):
            self.assertNotEqual(
                output_lock.lock_file_for("slide.jpg"),
                output_lock.lock_file_for("SLIDE.JPG"),
            )
            self.assertNotEqual(
                output_lock.lock_file_for(composed),
                output_lock.lock_file_for(decomposed),
            )

    def test_filesystem_semantics_probe_failure_removes_probe_file(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            with mock.patch.object(
                output_lock.Path,
                "with_name",
                side_effect=OSError("forced probe failure"),
            ):
                with self.assertRaisesRegex(
                    output_lock.OutputLockError,
                    "filesystem semantics",
                ):
                    output_lock._probe_filesystem_semantics(parent)
            self.assertEqual(list(parent.iterdir()), [])

    def _assert_provider_aliases_share_one_owner(self, first_name, second_name):
        generated = image_bytes()
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}]
        })
        entered_network = threading.Event()
        release_network = threading.Event()
        results = []
        call_guard = threading.Lock()
        call_count = 0

        def fake_urlopen(_request, timeout):
            nonlocal call_count
            self.assertEqual(timeout, 180)
            with call_guard:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                entered_network.set()
                self.assertTrue(release_network.wait(2))
            return response

        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            first_target = parent / first_name
            second_target = parent / second_name
            with mock.patch.object(
                gen_slide_openai, "_load_api_key", return_value="key"
            ), mock.patch.object(
                gen_slide_openai.urllib.request,
                "urlopen",
                side_effect=fake_urlopen,
            ) as urlopen, redirect_stdout(io.StringIO()):
                first = threading.Thread(
                    target=lambda: results.append(
                        gen_slide_openai.gen(
                            "first", str(first_target), retries=0
                        )
                    )
                )
                second = threading.Thread(
                    target=lambda: results.append(
                        gen_slide_openai.gen(
                            "second", str(second_target), retries=0
                        )
                    )
                )
                first.start()
                self.assertTrue(entered_network.wait(2))
                second.start()
                second.join(timeout=2)
                self.assertFalse(second.is_alive(), "alias contender did not fail fast")
                release_network.set()
                first.join(timeout=2)

            self.assertCountEqual(results, [True, False])
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(first_target.read_bytes(), generated)
            self.assertEqual(list(parent.glob(".*.tmp")), [])

    def test_case_aliases_share_provider_lock_on_case_insensitive_filesystem(self):
        with tempfile.TemporaryDirectory() as td:
            if not aliases_share_file(Path(td), "case-probe", "CASE-PROBE"):
                self.skipTest("temporary filesystem is case-sensitive")
        self._assert_provider_aliases_share_one_owner("slide.jpg", "SLIDE.JPG")

    def test_unicode_aliases_share_provider_lock_on_normalizing_filesystem(self):
        composed = "slide-é.jpg"
        decomposed = unicodedata.normalize("NFD", composed)
        with tempfile.TemporaryDirectory() as td:
            if not aliases_share_file(Path(td), composed, decomposed):
                self.skipTest("temporary filesystem distinguishes Unicode normalization")
        self._assert_provider_aliases_share_one_owner(composed, decomposed)

    def test_case_sensitive_filesystem_keeps_distinct_lock_keys(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            if aliases_share_file(parent, "case-probe", "CASE-PROBE"):
                self.skipTest("temporary filesystem is case-insensitive")
            self.assertNotEqual(
                output_lock.lock_file_for(parent / "slide.jpg"),
                output_lock.lock_file_for(parent / "SLIDE.JPG"),
            )

    def test_case_aliases_share_one_deck_owner_and_consistent_pair(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            if not aliases_share_file(parent, "case-probe", "CASE-PROBE"):
                self.skipTest("temporary filesystem is case-sensitive")

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

            with mock.patch.object(
                export_images, "_load_all", side_effect=fake_load_all
            ) as load_all, mock.patch.object(
                export_images, "_save_pdf", side_effect=fake_save_pdf
            ), mock.patch.object(
                export_images, "_save_pptx", side_effect=fake_save_pptx
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                first_source = parent / "A"
                second_source = parent / "B"
                first_source.write_bytes(b"A")
                second_source.write_bytes(b"B")
                first = threading.Thread(
                    target=lambda: results.append(
                        export_images.export_deck(
                            [str(first_source)], str(parent / "deck")
                        )
                    )
                )
                second = threading.Thread(
                    target=lambda: results.append(
                        export_images.export_deck(
                            [str(second_source)], str(parent / "DECK")
                        )
                    )
                )
                first.start()
                self.assertTrue(entered_save.wait(2))
                second.start()
                second.join(timeout=2)
                self.assertFalse(second.is_alive(), "alias deck contender did not fail fast")
                release_save.set()
                first.join(timeout=2)

            self.assertCountEqual(results, [True, False])
            self.assertEqual(load_all.call_count, 1)
            self.assertEqual((parent / "deck.pdf").read_bytes(), b"A-PDF")
            self.assertEqual((parent / "deck.pptx").read_bytes(), b"A-PPTX")
            self.assertEqual(list(parent.glob(".*.tmp.*")), [])
            self.assertEqual(list(parent.glob(".*.backup")), [])


class AbsolutePreparedTargetTests(unittest.TestCase):
    def test_prepared_target_path_is_immediately_absolute(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            approved = Path(td) / "approved"
            approved.mkdir()
            try:
                os.chdir(approved)
                prepared = image_output.preflight_output("slide.jpg")
            finally:
                os.chdir(original_cwd)
        self.assertEqual(
            prepared.path,
            (approved / "slide.jpg").resolve(strict=False),
        )
        self.assertTrue(prepared.path.is_absolute())

    def test_provider_cwd_change_publishes_to_original_approved_parent(self):
        generated = image_bytes()
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}]
        })
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()

            def change_cwd_then_respond(_request, timeout):
                self.assertEqual(timeout, 180)
                os.chdir(redirected)
                return response

            try:
                os.chdir(approved)
                with mock.patch.object(
                    gen_slide_openai, "_load_api_key", return_value="key"
                ), mock.patch.object(
                    gen_slide_openai.urllib.request,
                    "urlopen",
                    side_effect=change_cwd_then_respond,
                ), redirect_stdout(io.StringIO()):
                    self.assertTrue(
                        gen_slide_openai.gen("prompt", "slide.jpg", retries=0)
                    )
            finally:
                os.chdir(original_cwd)

            self.assertTrue((approved / "slide.jpg").exists())
            self.assertEqual((approved / "slide.jpg").read_bytes(), generated)
            self.assertFalse((redirected / "slide.jpg").exists())


class PortableLockBackendTests(unittest.TestCase):
    def test_module_imports_without_fcntl_and_uses_msvcrt_backend(self):
        fake_msvcrt = types.SimpleNamespace(
            LK_NBLCK=1,
            LK_UNLCK=2,
            locking=mock.Mock(),
        )
        module, error = load_output_lock_without(
            "fcntl", fake_msvcrt=fake_msvcrt
        )
        self.assertIsNone(error, f"output_lock failed to import: {error}")

        with tempfile.TemporaryDirectory() as td:
            module._LOCK_DIRECTORY = Path(td) / "locks"
            with module.output_lock(Path(td) / "slide.jpg"):
                pass

        self.assertEqual(fake_msvcrt.locking.call_count, 2)
        self.assertEqual(fake_msvcrt.locking.call_args_list[0].args[1:], (1, 1))
        self.assertEqual(fake_msvcrt.locking.call_args_list[1].args[1:], (2, 1))

    def test_module_imports_without_any_process_backend_and_fails_closed(self):
        module, error = load_output_lock_without("fcntl", "msvcrt")
        self.assertIsNone(error, f"output_lock failed to import: {error}")
        with tempfile.TemporaryDirectory() as td:
            module._LOCK_DIRECTORY = Path(td) / "locks"
            with self.assertRaisesRegex(
                module.OutputLockError, "process lock backend"
            ):
                with module.output_lock(Path(td) / "slide.jpg"):
                    pass


if __name__ == "__main__":
    unittest.main()
