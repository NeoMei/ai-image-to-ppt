import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vision_check_gemini.py"
sys.path.insert(0, str(ROOT / "scripts"))

import vision_check_gemini


class FakeResponse:
    def __init__(self, payload):
        if isinstance(payload, bytes):
            self.payload = payload
        else:
            self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def make_image(path: Path, image_format: str = "PNG") -> None:
    Image.new("RGB", (32, 18), "navy").save(path, format=image_format)


def successful_response(text="looks good"):
    return FakeResponse({
        "candidates": [{"content": {"parts": [{"text": text}]}}]
    })


class GeminiVisionCredentialTests(unittest.TestCase):
    def test_environment_key_has_priority_over_secret_file(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True), \
             mock.patch.object(Path, "read_text", return_value="file-key") as read_text:
            self.assertEqual(vision_check_gemini._load_api_key(), "env-key")
        read_text.assert_not_called()

    def test_secret_file_is_used_when_environment_key_is_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(Path, "read_text", return_value=" file-key\n"):
            self.assertEqual(vision_check_gemini._load_api_key(), "file-key")

    def test_unreadable_secret_file_is_reported_only_when_check_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "slide.png"
            make_image(image_path)
            stderr = io.StringIO()
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(Path, "read_text", side_effect=OSError("missing")), \
                 mock.patch.object(vision_check_gemini.urllib.request, "urlopen") as urlopen, \
                 redirect_stderr(stderr):
                exit_code = vision_check_gemini.main([str(image_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("GEMINI_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        urlopen.assert_not_called()


class GeminiVisionInputTests(unittest.TestCase):
    def test_png_jpeg_and_webp_are_detected_from_content(self):
        cases = [("mystery.bin", "PNG", "image/png"),
                 ("mystery.data", "JPEG", "image/jpeg"),
                 ("mystery.slide", "WEBP", "image/webp")]
        for filename, image_format, expected_mime in cases:
            with self.subTest(image_format=image_format), tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / filename
                make_image(image_path, image_format)
                with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                     mock.patch.object(
                         vision_check_gemini.urllib.request,
                         "urlopen",
                         return_value=successful_response(),
                     ) as urlopen:
                    self.assertEqual(
                        vision_check_gemini.check(str(image_path), retries=0),
                        "looks good",
                    )

                payload = json.loads(urlopen.call_args.args[0].data)
                inline = payload["contents"][0]["parts"][1]["inline_data"]
                self.assertEqual(inline["mime_type"], expected_mime)
                self.assertTrue(inline["data"])

    def test_missing_empty_non_image_unsupported_and_oversized_inputs_stop_before_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            empty = temp / "empty.png"
            empty.write_bytes(b"")
            text = temp / "fake.png"
            text.write_text("not an image", encoding="utf-8")
            gif = temp / "slide.gif"
            make_image(gif, "GIF")
            oversized = temp / "large.png"
            make_image(oversized)
            cases = [temp / "missing.png", empty, text, gif, oversized]

            for image_path in cases:
                with self.subTest(image_path=image_path.name), \
                     mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                     mock.patch.object(vision_check_gemini, "MAX_IMAGE_BYTES", 1 if image_path == oversized else 1024 * 1024), \
                     mock.patch.object(vision_check_gemini.urllib.request, "urlopen") as urlopen:
                    with self.assertRaises(vision_check_gemini.VisionCheckError):
                        vision_check_gemini.check(str(image_path), retries=0)
                    urlopen.assert_not_called()

    def test_decompression_bomb_is_a_controlled_input_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "slide.png"
            make_image(image_path)
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     vision_check_gemini.Image,
                     "open",
                     side_effect=Image.DecompressionBombError("too many pixels"),
                 ), \
                 mock.patch.object(vision_check_gemini.urllib.request, "urlopen") as urlopen:
                with self.assertRaises(vision_check_gemini.VisionCheckError):
                    vision_check_gemini.check(str(image_path), retries=0)
                urlopen.assert_not_called()


class GeminiVisionRequestTests(unittest.TestCase):
    def test_default_model_key_header_question_and_timeout_are_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "slide.png"
            make_image(image_path)
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key"}, clear=True), \
                 mock.patch.object(
                     vision_check_gemini.urllib.request,
                     "urlopen",
                     return_value=successful_response("answer"),
                 ) as urlopen:
                result = vision_check_gemini.check(
                    str(image_path), "count the cards", retries=0
                )

        self.assertEqual(result, "answer")
        request = urlopen.call_args.args[0]
        self.assertIn("gemini-3.6-flash:generateContent", request.full_url)
        self.assertNotIn("secret-key", request.full_url)
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "secret-key")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 60)
        payload = json.loads(request.data)
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "count the cards")

    def test_environment_can_override_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "slide.png"
            make_image(image_path)
            environment = {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_VISION_MODEL": "custom/model",
            }
            with mock.patch.dict(os.environ, environment, clear=True), \
                 mock.patch.object(
                     vision_check_gemini.urllib.request,
                     "urlopen",
                     return_value=successful_response(),
                 ) as urlopen:
                vision_check_gemini.check(str(image_path), retries=0)

        self.assertIn("custom%2Fmodel:generateContent", urlopen.call_args.args[0].full_url)


class GeminiVisionResponseTests(unittest.TestCase):
    def _check_with_response(self, response, retries=0):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        image_path = Path(temp_dir.name) / "slide.png"
        make_image(image_path)
        environment = {"GEMINI_API_KEY": "known-secret-key"}
        patch_env = mock.patch.dict(os.environ, environment, clear=True)
        response_options = (
            {"side_effect": response}
            if isinstance(response, (list, BaseException))
            else {"return_value": response}
        )
        patch_url = mock.patch.object(
            vision_check_gemini.urllib.request, "urlopen", **response_options
        )
        self.addCleanup(patch_env.stop)
        self.addCleanup(patch_url.stop)
        patch_env.start()
        urlopen = patch_url.start()
        return image_path, urlopen

    def test_text_is_found_in_any_candidate_part(self):
        response = FakeResponse({
            "candidates": [
                {"content": {"parts": [{"thought": True}]}},
                {"content": {"parts": [{"inlineData": {}}, {"text": " found "}]}},
            ]
        })
        image_path, _ = self._check_with_response(response)
        self.assertEqual(
            vision_check_gemini.check(str(image_path), retries=0), "found"
        )

    def test_malformed_json_empty_candidates_empty_parts_and_safety_block_are_controlled(self):
        responses = [
            FakeResponse(b"{not-json"),
            FakeResponse({"candidates": []}),
            FakeResponse({"candidates": [{"content": {"parts": []}}]}),
            FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}}),
        ]
        for response in responses:
            with self.subTest(payload=response.payload):
                image_path, urlopen = self._check_with_response(response)
                with self.assertRaises(vision_check_gemini.VisionCheckError):
                    vision_check_gemini.check(str(image_path), retries=2)
                self.assertEqual(urlopen.call_count, 1)

    def test_safety_feedback_cannot_echo_api_key(self):
        response = FakeResponse({
            "promptFeedback": {"blockReason": "known-secret-key"}
        })
        image_path, _ = self._check_with_response(response)
        with self.assertRaises(vision_check_gemini.VisionCheckError) as raised:
            vision_check_gemini.check(str(image_path), retries=0)
        self.assertNotIn("known-secret-key", str(raised.exception))

    def test_http_400_401_and_403_are_not_retried_and_secrets_are_redacted(self):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://example.invalid",
                    status,
                    "failed",
                    {},
                    io.BytesIO(json.dumps({
                        "error": {
                            "message": "bad known-secret-key and AIzaSyRemoteSecret1234567890"
                        }
                    }).encode("utf-8")),
                )
                image_path, urlopen = self._check_with_response(error)
                with self.assertRaises(vision_check_gemini.VisionCheckError) as raised:
                    vision_check_gemini.check(str(image_path), retries=2)
                message = str(raised.exception)
                self.assertIn(f"HTTP {status}", message)
                self.assertNotIn("known-secret-key", message)
                self.assertNotIn("AIzaSyRemoteSecret1234567890", message)
                self.assertEqual(urlopen.call_count, 1)

    def test_429_and_5xx_are_retried_then_succeed(self):
        for status in (429, 500, 503):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://example.invalid", status, "transient", {}, io.BytesIO(b"{}")
                )
                image_path, urlopen = self._check_with_response(
                    [error, successful_response("recovered")]
                )
                with mock.patch.object(vision_check_gemini.time, "sleep") as sleep:
                    self.assertEqual(
                        vision_check_gemini.check(str(image_path), retries=1),
                        "recovered",
                    )
                self.assertEqual(urlopen.call_count, 2)
                sleep.assert_called_once_with(2)

    def test_network_error_and_timeout_are_retried_then_succeed(self):
        errors = [urllib.error.URLError("offline"), TimeoutError("timed out")]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                image_path, urlopen = self._check_with_response(
                    [error, successful_response("recovered")]
                )
                with mock.patch.object(vision_check_gemini.time, "sleep") as sleep:
                    self.assertEqual(
                        vision_check_gemini.check(str(image_path), retries=1),
                        "recovered",
                    )
                self.assertEqual(urlopen.call_count, 2)
                sleep.assert_called_once_with(2)


class GeminiVisionCliTests(unittest.TestCase):
    def test_main_prints_success_and_returns_zero(self):
        stdout = io.StringIO()
        with mock.patch.object(vision_check_gemini, "check", return_value="all good") as check, \
             redirect_stdout(stdout):
            exit_code = vision_check_gemini.main(["slide.webp", "inspect", "this"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "all good")
        check.assert_called_once_with("slide.webp", "inspect this", retries=2)

    def test_main_reports_operational_error_without_traceback(self):
        stderr = io.StringIO()
        with mock.patch.object(
            vision_check_gemini, "check", side_effect=vision_check_gemini.VisionCheckError("safe failure")
        ), redirect_stderr(stderr):
            exit_code = vision_check_gemini.main(["slide.png"])
        self.assertEqual(exit_code, 1)
        self.assertIn("safe failure", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_help_and_missing_argument_have_standard_argparse_exit_codes(self):
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            env={**os.environ, "GEMINI_API_KEY": ""},
            check=False,
        )
        missing_result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            env={**os.environ, "GEMINI_API_KEY": ""},
            check=False,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("usage:", help_result.stdout.lower())
        self.assertNotIn("Traceback", help_result.stderr)
        self.assertEqual(missing_result.returncode, 2)
        self.assertIn("usage:", missing_result.stderr.lower())
        self.assertNotIn("Traceback", missing_result.stderr)

    def test_negative_retries_are_rejected_as_usage_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised, \
             mock.patch.object(vision_check_gemini, "check") as check:
            vision_check_gemini.main(["slide.png", "--retries", "-1"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("non-negative", stderr.getvalue())
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
