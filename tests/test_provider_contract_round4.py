import base64
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import vision_check_gemini
from generation_result import GenerationResult, GenerationStatus


def image_bytes(image_format):
    buffer = io.BytesIO()
    Image.new("RGB", (160, 90), "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class JsonResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(json.dumps(payload).encode("utf-8"))
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def gemini_response(image_format, mime_type):
    return JsonResponse({
        "candidates": [{
            "content": {
                "parts": [{
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(
                            image_bytes(image_format)
                        ).decode("ascii"),
                    }
                }]
            }
        }]
    })


class ProviderCredentialBoundaryTests(unittest.TestCase):
    def test_invalid_keys_fail_before_request_without_echo_or_retry(self):
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )
        invalid_keys = ("", "bad key", "bad\nkey", "bad\tkey", "key-é")

        for provider, filename in providers:
            for invalid_key in invalid_keys:
                with self.subTest(provider=provider.__name__, key=repr(invalid_key)), \
                     tempfile.TemporaryDirectory() as temp_dir, \
                     mock.patch.object(
                         provider, "_load_api_key", return_value=invalid_key
                     ), \
                     mock.patch.object(
                         provider.urllib.request, "urlopen"
                     ) as urlopen, \
                     mock.patch.object(provider.time, "sleep") as sleep:
                    output = io.StringIO()
                    with redirect_stdout(output):
                        try:
                            result = provider.gen(
                                "prompt",
                                str(Path(temp_dir) / filename),
                                retries=2,
                            )
                        except Exception as error:
                            self.fail(
                                "invalid API key escaped provider boundary: "
                                f"{type(error).__name__}"
                            )

                self.assertFalse(result)
                urlopen.assert_not_called()
                sleep.assert_not_called()
                if invalid_key:
                    self.assertNotIn(invalid_key, output.getvalue())
                self.assertIn("API key", output.getvalue())

    def test_four_direct_clis_report_invalid_keys_without_traceback_or_network(self):
        invalid_key = "bad key"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            image_path = temp / "input.png"
            Image.new("RGB", (160, 90), "navy").save(image_path)
            cases = (
                (gen_slide_openai, [str(temp / "openai.jpg"), "prompt"]),
                (gen_slide_gemini, [str(temp / "gemini.png"), "prompt"]),
                (gen_slide_doubao, [str(temp / "doubao.jpg"), "prompt"]),
            )
            for module, argv in cases:
                with self.subTest(module=module.__name__), \
                     mock.patch.object(
                         module, "_load_api_key", return_value=invalid_key
                     ), \
                     mock.patch.object(
                         module.urllib.request, "urlopen"
                     ) as urlopen:
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        exit_code = module.main(argv)
                self.assertEqual(exit_code, 1)
                self.assertNotIn("Traceback", stdout.getvalue())
                self.assertNotIn(invalid_key, stdout.getvalue())
                urlopen.assert_not_called()

            with mock.patch.object(
                vision_check_gemini, "_load_api_key", return_value=invalid_key
            ), mock.patch.object(
                vision_check_gemini.urllib.request, "urlopen"
            ) as urlopen:
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = vision_check_gemini.main([str(image_path)])
            self.assertEqual(exit_code, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertNotIn(invalid_key, stderr.getvalue())
            self.assertIn("API key", stderr.getvalue())
            urlopen.assert_not_called()


class ProviderAbsoluteOutputTests(unittest.TestCase):
    def test_relative_symlinked_output_is_resolved_once_before_lock_and_cwd_change(self):
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )
        original_cwd = Path.cwd()
        for provider, filename in providers:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                approved = temp / "approved"
                redirected = temp / "redirected"
                approved.mkdir()
                redirected.mkdir()
                (temp / "alias").symlink_to(approved, target_is_directory=True)
                relative = Path("alias") / filename
                expected = (approved / filename).resolve(strict=False)
                observed_locks = []

                @contextmanager
                def changing_cwd_lock(target, *args, **kwargs):
                    observed_locks.append(Path(target))
                    os.chdir(redirected)
                    try:
                        yield
                    finally:
                        os.chdir(temp)

                os.chdir(temp)
                try:
                    owned_result = GenerationResult(
                        GenerationStatus.SUCCESS,
                        (
                            "openai" if provider is gen_slide_openai
                            else "gemini" if provider is gen_slide_gemini
                            else "doubao"
                        ),
                        "api",
                        str(expected),
                        "generated",
                    )
                    with mock.patch.object(
                        provider, "output_lock", side_effect=changing_cwd_lock
                    ), mock.patch.object(
                        provider, "_gen_owned", return_value=owned_result
                    ) as owned:
                        self.assertTrue(provider.gen("prompt", str(relative)))
                finally:
                    os.chdir(original_cwd)

                self.assertEqual(observed_locks, [expected])
                self.assertEqual(Path(owned.call_args.args[1]), expected)

    def test_final_output_symlink_is_rejected_before_credentials_or_network(self):
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )
        for provider, filename in providers:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                referent = temp / f"referent-{filename}"
                referent.write_bytes(b"keep-me")
                target = temp / filename
                target.symlink_to(referent)

                with mock.patch.object(provider, "_load_api_key") as load_key, \
                     mock.patch.object(
                         provider.urllib.request, "urlopen"
                     ) as urlopen:
                    output = io.StringIO()
                    with redirect_stdout(output):
                        result = provider.gen(
                            "prompt", str(target), retries=0, overwrite=True
                        )

                self.assertFalse(result)
                self.assertIn("unable to prepare output target", output.getvalue())
                load_key.assert_not_called()
                urlopen.assert_not_called()
                self.assertEqual(referent.read_bytes(), b"keep-me")


class GeminiOutputContractTests(unittest.TestCase):
    def test_official_default_png_response_publishes_to_png(self):
        response = gemini_response("PNG", "image/png")
        environment = {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_IMAGE_MODEL": "custom/model",
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, environment, clear=True), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=response,
             ) as urlopen:
            target = Path(temp_dir) / "slide.png"
            self.assertTrue(
                gen_slide_gemini.gen("prompt", str(target), retries=0)
            )

            with Image.open(target) as image:
                self.assertEqual(image.format, "PNG")

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1/models/"
            "custom%2Fmodel:generateContent",
        )
        image_config = json.loads(request.data)["generationConfig"][
            "responseFormat"
        ]["image"]
        self.assertEqual(
            image_config,
            {"aspectRatio": "16:9", "imageSize": "2K"},
        )

    def test_jpeg_target_requests_official_jpeg_mime_and_publishes_jpeg(self):
        response = gemini_response("JPEG", "image/jpeg")
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(
                 os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True
             ), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=response,
             ) as urlopen:
            target = Path(temp_dir) / "slide.jpg"
            self.assertTrue(
                gen_slide_gemini.gen("prompt", str(target), retries=0)
            )

            with Image.open(target) as image:
                self.assertEqual(image.format, "JPEG")

        image_config = json.loads(urlopen.call_args.args[0].data)[
            "generationConfig"
        ]["responseFormat"]["image"]
        self.assertIn("mimeType", image_config)
        self.assertEqual(image_config["mimeType"], "IMAGE_JPEG")

    def test_gif_and_webp_are_rejected_before_key_or_network(self):
        for suffix in (".gif", ".webp"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(
                     gen_slide_gemini, "_load_api_key", return_value="test-key"
                 ) as load_key, \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     return_value=gemini_response("PNG", "image/png"),
                 ) as urlopen:
                result = gen_slide_gemini.gen(
                    "prompt", str(Path(temp_dir) / f"slide{suffix}")
                )
            self.assertFalse(result)
            load_key.assert_not_called()
            urlopen.assert_not_called()


class ProviderDocumentationTests(unittest.TestCase):
    def test_provider_extensions_and_gemini_png_default_are_documented(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        for document in (readme, skill):
            self.assertIn("Gemini", document)
            self.assertIn("`.png`, `.jpg`, and `.jpeg`", document)
            self.assertIn("PNG by default", document)
            self.assertNotIn(
                "All providers support output extensions `.jpg`, `.jpeg`, `.png`, and `.webp`",
                document,
            )
            self.assertIn(
                'out/slide_01.png "<prompt>" --engine gemini',
                readme,
            )
        self.assertIn(
            'out/slide_01.png", engine="gemini"',
            skill,
        )

    def test_plan_current_count_is_not_the_pre_round4_137(self):
        plan = (
            ROOT
            / "docs/superpowers/plans/2026-08-26-gpt-image-default-engine.md"
        ).read_text(encoding="utf-8")
        execution_status = plan.split("## Execution status ", 1)[1]
        self.assertNotIn("**137 tests passed**", execution_status)


if __name__ == "__main__":
    unittest.main()
