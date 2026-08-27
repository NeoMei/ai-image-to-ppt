import base64
import http.client
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide_openai
from generation_result import GenerationStatus
from image_output import ImageOutputError


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
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


OPENAI_CASES = (
    (401, "invalid_api_key", GenerationStatus.AUTH_UNAVAILABLE),
    (403, "permission_denied", GenerationStatus.AUTH_UNAVAILABLE),
    (400, "content_policy_violation", GenerationStatus.POLICY_REFUSED),
    (400, "moderation_blocked", GenerationStatus.POLICY_REFUSED),
    (429, "rate_limit_exceeded", GenerationStatus.RETRYABLE_EXHAUSTED),
    (400, "insufficient_quota", GenerationStatus.RETRYABLE_EXHAUSTED),
    (500, "server_error", GenerationStatus.RETRYABLE_EXHAUSTED),
    (400, "invalid_size", GenerationStatus.INVALID_INPUT),
)


def http_error(status, code, message="safe"):
    return urllib.error.HTTPError(
        gen_slide_openai.API_URL,
        status,
        "request failed",
        {},
        io.BytesIO(json.dumps({"error": {"code": code, "message": message}}).encode()),
    )


class OpenAIResultTests(unittest.TestCase):
    def test_http_error_codes_produce_the_structured_result_matrix(self):
        for status, code, expected in OPENAI_CASES:
            with self.subTest(status=status, code=code), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     side_effect=http_error(status, code),
                 ):
                target = Path(temp_dir) / "slide.jpg"
                result = gen_slide_openai.generate_result("prompt", str(target), retries=0)
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.output_path)
                self.assertFalse(target.exists())

    def test_policy_error_code_overrides_a_retryable_http_status(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 side_effect=http_error(500, "content_policy_violation"),
             ) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            result = gen_slide_openai.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
            )
        self.assertEqual(result.status, GenerationStatus.POLICY_REFUSED)
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_non_result_owned_return_fails_closed_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(gen_slide_openai, "_gen_owned", return_value=True):
            target = Path(temp_dir) / "slide.jpg"
            result = gen_slide_openai.generate_result("prompt", str(target), retries=0)
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
        self.assertIsNone(result.output_path)
        self.assertFalse(target.exists())

    def test_early_local_failures_redact_environment_key_before_networking(self):
        key = "known-test-key"
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / key / "slide.jpg"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai,
                     "prepare_target",
                     side_effect=ImageOutputError(f"cannot prepare {target}"),
                 ), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen:
                prepared_failure = gen_slide_openai.generate_result(
                    "prompt", str(target), retries=0
                )
            self.assertEqual(prepared_failure.status, GenerationStatus.LOCAL_FAILURE)
            self.assertNotIn(key, prepared_failure.safe_message)
            urlopen.assert_not_called()

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai,
                     "output_lock",
                     side_effect=gen_slide_openai.OutputLockError(f"locked {target}"),
                 ), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen:
                lock_failure = gen_slide_openai.generate_result(
                    "prompt", str(target), retries=0
                )
            self.assertEqual(lock_failure.status, GenerationStatus.LOCAL_FAILURE)
            self.assertNotIn(key, lock_failure.safe_message)
            urlopen.assert_not_called()

    def test_missing_or_invalid_key_stops_before_network(self):
        for key in ("", "bad key"):
            with self.subTest(key=repr(key)), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(gen_slide_openai, "_load_api_key", return_value=key), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen:
                result = gen_slide_openai.generate_result(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.AUTH_UNAVAILABLE)
                if key:
                    self.assertNotIn(key, result.safe_message)
                urlopen.assert_not_called()

    def test_exhausted_transport_failure_is_retryable_exhausted(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 side_effect=urllib.error.URLError("temporary outage"),
             ), \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            result = gen_slide_openai.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
            )
        self.assertEqual(result.status, GenerationStatus.RETRYABLE_EXHAUSTED)
        sleep.assert_called_once_with(2)

    def test_malformed_json_is_invalid_output(self):
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(b"{not-json")
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response):
            result = gen_slide_openai.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)

    def test_invalid_base64_is_invalid_output(self):
        response = JsonResponse({"data": [{"b64_json": "%%%invalid%%%"}]})
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response):
            result = gen_slide_openai.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)

    def test_wrong_format_and_non_16_by_9_bytes_are_invalid_output(self):
        cases = (image_bytes("PNG"), image_bytes("JPEG", (160, 100)))
        for data in cases:
            with self.subTest(byte_count=len(data)), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     return_value=JsonResponse({"data": [{"b64_json": base64.b64encode(data).decode()}]}),
                 ):
                result = gen_slide_openai.generate_result(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)

    def test_publication_failure_after_valid_bytes_is_local_failure(self):
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes()).decode()}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response), \
             mock.patch.object(
                 gen_slide_openai,
                 "publish_bytes",
                 side_effect=ImageOutputError("replace failed"),
             ):
            result = gen_slide_openai.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)

    def test_success_returns_only_an_absolute_published_path(self):
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes()).decode()}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response):
            target = Path(temp_dir) / "slide.jpg"
            result = gen_slide_openai.generate_result("prompt", str(target), retries=0)
            self.assertTrue(target.exists())
        self.assertEqual(result.status, GenerationStatus.SUCCESS)
        self.assertEqual(result.output_path, str(target.resolve()))

    def test_safe_messages_redact_known_and_key_like_secrets(self):
        key = "known-test-key"
        message = f"request contained {key} and sk-remoteSecret123"
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 side_effect=http_error(400, "invalid_size", message),
             ):
            result = gen_slide_openai.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertNotIn(key, result.safe_message)
        self.assertNotIn("sk-remoteSecret123", result.safe_message)
        self.assertIn("[REDACTED]", result.safe_message)


if __name__ == "__main__":
    unittest.main()
