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

import gen_slide_gemini
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


def gemini_payload(data, mime_type="image/jpeg"):
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }]
            }
        }]
    }


POLICY_PAYLOADS = (
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": [{"finishReason": "SAFETY"}]},
    {"candidates": [{"finishReason": "BLOCKLIST"}]},
    {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]},
    {"candidates": [{"finishReason": "IMAGE_SAFETY"}]},
)


GEMINI_CASES = (
    (401, "invalid_api_key", GenerationStatus.AUTH_UNAVAILABLE),
    (403, "permission_denied", GenerationStatus.AUTH_UNAVAILABLE),
    (400, "blocked", GenerationStatus.POLICY_REFUSED),
    (400, "safety", GenerationStatus.POLICY_REFUSED),
    (429, "rate_limit_exceeded", GenerationStatus.RETRYABLE_EXHAUSTED),
    (400, "quota_exceeded", GenerationStatus.RETRYABLE_EXHAUSTED),
    (500, "server_error", GenerationStatus.RETRYABLE_EXHAUSTED),
    (400, "invalid_argument", GenerationStatus.INVALID_INPUT),
)


def http_error(status, code, message="safe"):
    return urllib.error.HTTPError(
        "https://provider.invalid",
        status,
        "request failed",
        {},
        io.BytesIO(json.dumps({"error": {"code": code, "message": message}}).encode()),
    )


class GeminiResultTests(unittest.TestCase):
    def test_http_error_codes_produce_the_structured_result_matrix(self):
        for status, code, expected in GEMINI_CASES:
            with self.subTest(status=status, code=code), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     side_effect=http_error(status, code),
                 ):
                target = Path(temp_dir) / "slide.jpg"
                result = gen_slide_gemini.generate_result("prompt", str(target), retries=0)
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.output_path)
                self.assertFalse(target.exists())

    def test_policy_payloads_stop_without_retry_or_publication(self):
        for payload in POLICY_PAYLOADS:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     return_value=JsonResponse(payload),
                 ) as urlopen, \
                 mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
                target = Path(temp_dir) / "slide.jpg"
                result = gen_slide_gemini.generate_result("prompt", str(target), retries=2)
                self.assertEqual(result.status, GenerationStatus.POLICY_REFUSED)
                self.assertIsNone(result.output_path)
                self.assertFalse(target.exists())
                self.assertEqual(urlopen.call_count, 1)
                sleep.assert_not_called()

    def test_policy_payload_takes_priority_over_missing_image(self):
        payload = {
            "promptFeedback": {"blockReason": "SAFETY"},
            "candidates": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=JsonResponse(payload),
             ):
            result = gen_slide_gemini.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertEqual(result.status, GenerationStatus.POLICY_REFUSED)

    def test_non_result_owned_return_fails_closed_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(gen_slide_gemini, "_gen_owned", return_value=True):
            target = Path(temp_dir) / "slide.jpg"
            result = gen_slide_gemini.generate_result("prompt", str(target), retries=0)
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
        self.assertIsNone(result.output_path)
        self.assertFalse(target.exists())

    def test_early_validation_and_target_failures_never_read_environment_key(self):
        cases = (
            ("invalid prompt", "", "slide.jpg", 0, False, None),
            ("invalid retries", "prompt", "slide.jpg", -1, False, None),
            ("invalid overwrite", "prompt", "slide.jpg", 0, "yes", None),
            ("invalid suffix", "prompt", "slide.gif", 0, False, None),
            (
                "target failure",
                "prompt",
                "slide.jpg",
                0,
                False,
                mock.patch.object(
                    gen_slide_gemini,
                    "prepare_target",
                    side_effect=ImageOutputError("key-like-target-detail"),
                ),
            ),
            (
                "lock failure",
                "prompt",
                "slide.jpg",
                0,
                False,
                mock.patch.object(
                    gen_slide_gemini,
                    "output_lock",
                    side_effect=gen_slide_gemini.OutputLockError("key-like-lock-detail"),
                ),
            ),
        )
        for name, prompt, target, retries, overwrite, boundary_patch in cases:
            with self.subTest(name=name), \
                 mock.patch.object(
                     gen_slide_gemini,
                     "_environment_redaction_secrets",
                     return_value=("known-test-key",),
                 ) as redact_key, \
                 mock.patch.object(gen_slide_gemini, "_load_api_key") as load_key, \
                 mock.patch.object(gen_slide_gemini.urllib.request, "urlopen") as urlopen:
                if boundary_patch is None:
                    result = gen_slide_gemini.generate_result(
                        prompt, target, retries=retries, overwrite=overwrite
                    )
                else:
                    with boundary_patch:
                        result = gen_slide_gemini.generate_result(
                            prompt, target, retries=retries, overwrite=overwrite
                        )
            self.assertFalse(result.ok)
            redact_key.assert_not_called()
            load_key.assert_not_called()
            urlopen.assert_not_called()

    def test_missing_or_invalid_key_stops_before_network(self):
        for key in ("", "bad key"):
            with self.subTest(key=repr(key)), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(gen_slide_gemini, "_load_api_key", return_value=key), \
                 mock.patch.object(gen_slide_gemini.urllib.request, "urlopen") as urlopen:
                result = gen_slide_gemini.generate_result(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.AUTH_UNAVAILABLE)
                if key:
                    self.assertNotIn(key, result.safe_message)
                urlopen.assert_not_called()

    def test_exhausted_transport_failure_is_retryable_exhausted(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 side_effect=urllib.error.URLError("temporary outage"),
             ), \
             mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
            result = gen_slide_gemini.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
            )
        self.assertEqual(result.status, GenerationStatus.RETRYABLE_EXHAUSTED)
        sleep.assert_called_once_with(2)

    def test_malformed_envelope_and_missing_image_are_invalid_output(self):
        for payload in ({"not_candidates": []}, {"candidates": [{"content": {"parts": []}}]}):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     return_value=JsonResponse(payload),
                 ):
                result = gen_slide_gemini.generate_result(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)

    def test_declared_mime_mismatch_and_invalid_bytes_are_invalid_output(self):
        payloads = (
            gemini_payload(image_bytes(), "image/png"),
            gemini_payload(b"not an image"),
            gemini_payload(image_bytes("JPEG", (160, 100))),
        )
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     return_value=JsonResponse(payload),
                 ):
                result = gen_slide_gemini.generate_result(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)

    def test_publication_failure_after_valid_bytes_is_local_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=JsonResponse(gemini_payload(image_bytes())),
             ), \
             mock.patch.object(
                 gen_slide_gemini,
                 "publish_bytes",
                 side_effect=ImageOutputError("replace failed"),
             ):
            result = gen_slide_gemini.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)

    def test_success_returns_only_an_absolute_published_path(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=JsonResponse(gemini_payload(image_bytes())),
             ):
            target = Path(temp_dir) / "slide.jpg"
            result = gen_slide_gemini.generate_result("prompt", str(target), retries=0)
            self.assertTrue(target.exists())
        self.assertEqual(result.status, GenerationStatus.SUCCESS)
        self.assertEqual(result.output_path, str(target.resolve()))

    def test_safe_messages_redact_known_and_key_like_secrets(self):
        key = "known-test-key"
        message = f"request contained {key} and sk-remoteSecret123"
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"GEMINI_API_KEY": key}, clear=True), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 side_effect=http_error(400, "invalid_argument", message),
             ):
            result = gen_slide_gemini.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertNotIn(key, result.safe_message)
        self.assertNotIn("sk-remoteSecret123", result.safe_message)
        self.assertIn("[REDACTED]", result.safe_message)


if __name__ == "__main__":
    unittest.main()
