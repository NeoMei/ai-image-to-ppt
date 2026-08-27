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

import gen_slide_doubao
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


class StreamResponse:
    def __init__(self, data, headers=None):
        self.stream = io.BytesIO(data)
        self.headers = (
            {
                "Content-Length": str(len(data)),
                "Content-Type": "image/jpeg",
            }
            if headers is None
            else headers
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


DOUBAO_GENERATION_CASES = (
    (401, "invalid_api_key", GenerationStatus.AUTH_UNAVAILABLE, True),
    (403, "permission_denied", GenerationStatus.AUTH_UNAVAILABLE, True),
    (400, "content_filter", GenerationStatus.POLICY_REFUSED, False),
    (400, "content_policy_violation", GenerationStatus.POLICY_REFUSED, False),
    (400, "input_text_risk", GenerationStatus.POLICY_REFUSED, False),
    (400, "output_image_risk", GenerationStatus.POLICY_REFUSED, False),
    (429, "rate_limit_exceeded", GenerationStatus.RETRYABLE_EXHAUSTED, True),
    (500, "server_error", GenerationStatus.RETRYABLE_EXHAUSTED, True),
    (400, "invalid_prompt", GenerationStatus.INVALID_INPUT, False),
)

DOUBAO_DOWNLOAD_CASES = (
    (401, "invalid_signature", GenerationStatus.AUTH_UNAVAILABLE, True),
    (403, "cdn_denied", GenerationStatus.AUTH_UNAVAILABLE, True),
    (400, "content_filter", GenerationStatus.POLICY_REFUSED, False),
    (429, "rate_limit_exceeded", GenerationStatus.RETRYABLE_EXHAUSTED, True),
    (503, "upstream_unavailable", GenerationStatus.RETRYABLE_EXHAUSTED, True),
    (404, "not_found", GenerationStatus.INVALID_INPUT, False),
)


def http_error(status, code, message="safe"):
    return urllib.error.HTTPError(
        "https://provider.invalid/request",
        status,
        "request failed",
        {},
        io.BytesIO(json.dumps({"error": {"code": code, "message": message}}).encode()),
    )


def generated_url(url="https://cdn.invalid/image"):
    return JsonResponse({"data": [{"url": url}]})


class DoubaoResultTests(unittest.TestCase):
    def _generate(self, responses, target, retries=0, overwrite=False):
        urlopen_kwargs = (
            {"side_effect": responses}
            if isinstance(responses, (list, tuple, BaseException))
            else {"return_value": responses}
        )
        with mock.patch.dict(os.environ, {"DOUBAO_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request, "urlopen", **urlopen_kwargs
             ) as urlopen:
            result = gen_slide_doubao.generate_result(
                "prompt", str(target), retries=retries, overwrite=overwrite
            )
        return result, urlopen

    def test_generation_http_errors_produce_the_structured_result_matrix(self):
        for status, code, expected, can_fallback in DOUBAO_GENERATION_CASES:
            with self.subTest(status=status, code=code), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "slide.jpg"
                result, urlopen = self._generate(http_error(status, code), target)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.can_fallback, can_fallback)
                self.assertIsNone(result.output_path)
                self.assertFalse(target.exists())
                self.assertEqual(urlopen.call_count, 1)

    def test_generation_policy_code_wins_over_retryable_http_status(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"DOUBAO_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=http_error(500, "content_filter"),
             ) as urlopen, \
             mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            result = gen_slide_doubao.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
            )
        self.assertEqual(result.status, GenerationStatus.POLICY_REFUSED)
        self.assertFalse(result.can_fallback)
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_generation_transport_exhaustion_is_retryable_without_download(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"DOUBAO_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=urllib.error.URLError("temporary outage"),
             ) as urlopen, \
             mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            result = gen_slide_doubao.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
            )
        self.assertEqual(result.status, GenerationStatus.RETRYABLE_EXHAUSTED)
        self.assertTrue(result.can_fallback)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_missing_or_invalid_key_stops_before_network(self):
        for key in ("", "bad key"):
            with self.subTest(key=repr(key)), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(gen_slide_doubao, "_load_api_key", return_value=key), \
                 mock.patch.object(gen_slide_doubao.urllib.request, "urlopen") as urlopen:
                result = gen_slide_doubao.generate_result(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.AUTH_UNAVAILABLE)
                self.assertTrue(result.can_fallback)
                if key:
                    self.assertNotIn(key, result.safe_message)
                urlopen.assert_not_called()

    def test_arguments_target_and_lock_fail_before_redaction_key_or_network(self):
        cases = (
            ("invalid prompt", "", "slide.jpg", 0, False, None),
            ("invalid retries", "prompt", "slide.jpg", -1, False, None),
            ("invalid overwrite", "prompt", "slide.jpg", 0, "yes", None),
            ("invalid suffix", "prompt", "slide.gif", 0, False, None),
            (
                "target failure", "prompt", "slide.jpg", 0, False,
                mock.patch.object(
                    gen_slide_doubao,
                    "prepare_target",
                    side_effect=ImageOutputError("target failure"),
                ),
            ),
            (
                "lock failure", "prompt", "slide.jpg", 0, False,
                mock.patch.object(
                    gen_slide_doubao,
                    "output_lock",
                    side_effect=gen_slide_doubao.OutputLockError("lock failure"),
                ),
            ),
        )
        for name, prompt, out_path, retries, overwrite, boundary_patch in cases:
            with self.subTest(name=name), \
                 mock.patch.object(
                     gen_slide_doubao, "_environment_redaction_secrets", create=True
                 ) as redact, \
                 mock.patch.object(gen_slide_doubao, "_load_api_key") as load_key, \
                 mock.patch.object(gen_slide_doubao.urllib.request, "urlopen") as urlopen:
                if boundary_patch is None:
                    result = gen_slide_doubao.generate_result(
                        prompt, out_path, retries=retries, overwrite=overwrite
                    )
                else:
                    with boundary_patch:
                        result = gen_slide_doubao.generate_result(
                            prompt, out_path, retries=retries, overwrite=overwrite
                        )
            expected = (
                GenerationStatus.LOCAL_FAILURE
                if boundary_patch is not None
                else GenerationStatus.INVALID_INPUT
            )
            self.assertEqual(result.status, expected)
            self.assertFalse(result.can_fallback)
            redact.assert_not_called()
            load_key.assert_not_called()
            urlopen.assert_not_called()

    def test_malformed_generation_envelope_or_url_is_invalid_output(self):
        payloads = (
            {},
            {"data": []},
            {"data": [{"url": "http://cdn.invalid/image"}]},
        )
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "slide.jpg"
                result, urlopen = self._generate(JsonResponse(payload), target)
                self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
                self.assertFalse(result.can_fallback)
                self.assertFalse(target.exists())
                self.assertEqual(urlopen.call_count, 1)

    def test_download_http_errors_produce_the_structured_result_matrix(self):
        for status, code, expected, can_fallback in DOUBAO_DOWNLOAD_CASES:
            with self.subTest(status=status, code=code), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "slide.jpg"
                result, urlopen = self._generate(
                    [generated_url(), http_error(status, code)], target
                )
                self.assertEqual(result.status, expected)
                self.assertEqual(result.can_fallback, can_fallback)
                self.assertIsNone(result.output_path)
                self.assertFalse(target.exists())
                self.assertEqual(urlopen.call_count, 2)

    def test_download_transport_exhaustion_is_retryable_without_regenerating(self):
        cdn_url = "https://cdn.invalid/image"
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"DOUBAO_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated_url(cdn_url), TimeoutError("slow CDN"), TimeoutError("slow CDN")],
             ) as urlopen, \
             mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            result = gen_slide_doubao.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
            )
        self.assertEqual(result.status, GenerationStatus.RETRYABLE_EXHAUSTED)
        self.assertTrue(result.can_fallback)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(
            [call.args[0].full_url for call in urlopen.call_args_list],
            [gen_slide_doubao.URL, cdn_url, cdn_url],
        )
        sleep.assert_called_once_with(2)

    def test_invalid_download_bytes_are_invalid_output_and_force_preserves_target(self):
        invalid_downloads = (
            StreamResponse(b"not an image"),
            StreamResponse(image_bytes("PNG")),
            StreamResponse(image_bytes("JPEG", (160, 100))),
            StreamResponse(
                b"x",
                {"Content-Length": str(gen_slide_doubao.MAX_IMAGE_BYTES + 1)},
            ),
        )
        for downloaded in invalid_downloads:
            with self.subTest(headers=downloaded.headers), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "slide.jpg"
                target.write_bytes(b"existing")
                result, urlopen = self._generate(
                    [generated_url(), downloaded], target, overwrite=True
                )
                self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
                self.assertFalse(result.can_fallback)
                self.assertEqual(target.read_bytes(), b"existing")
                self.assertEqual(urlopen.call_count, 2)
                self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_download_content_type_must_match_the_target_before_publication(self):
        rejected_headers = (
            {"Content-Length": str(len(image_bytes()))},
            {"Content-Length": str(len(image_bytes())), "Content-Type": ""},
            {"Content-Length": str(len(image_bytes())), "Content-Type": 3},
            {"Content-Length": str(len(image_bytes())), "Content-Type": "text/html"},
            {"Content-Length": str(len(image_bytes())), "Content-Type": "image/png"},
        )
        for headers in rejected_headers:
            with self.subTest(headers=headers), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
                target = Path(temp_dir) / "slide.jpg"
                target.write_bytes(b"existing")
                result, urlopen = self._generate(
                    [generated_url(), StreamResponse(image_bytes(), headers)],
                    target,
                    retries=2,
                    overwrite=True,
                )
                self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
                self.assertFalse(result.can_fallback)
                self.assertEqual(target.read_bytes(), b"existing")
                self.assertEqual(urlopen.call_count, 2)
                sleep.assert_not_called()

    def test_download_accepts_a_matching_content_type_with_parameters(self):
        headers = {
            "Content-Length": str(len(image_bytes())),
            "Content-Type": "image/jpeg; charset=binary",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            result, urlopen = self._generate(
                [generated_url(), StreamResponse(image_bytes(), headers)], target
            )
            self.assertTrue(target.exists())
        self.assertEqual(result.status, GenerationStatus.SUCCESS)
        self.assertEqual(urlopen.call_count, 2)

    def test_content_type_rejection_redacts_the_api_key(self):
        key = "known-test-key"
        headers = {
            "Content-Length": str(len(image_bytes())),
            "Content-Type": f"image/png; credential={key}",
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"DOUBAO_API_KEY": key}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated_url(), StreamResponse(image_bytes(), headers)],
             ):
            result = gen_slide_doubao.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
        self.assertNotIn(key, result.safe_message)

    def test_download_publication_failure_is_local_failure_and_preserves_target(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"DOUBAO_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated_url(), StreamResponse(image_bytes())],
             ), \
             mock.patch.object(
                 gen_slide_doubao,
                 "publish_bytes",
                 side_effect=ImageOutputError("replace failed"),
                 create=True,
             ):
            target = Path(temp_dir) / "slide.jpg"
            target.write_bytes(b"existing")
            result = gen_slide_doubao.generate_result(
                "prompt", str(target), retries=0, overwrite=True
            )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse(result.can_fallback)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_success_returns_an_absolute_published_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            result, urlopen = self._generate(
                [generated_url(), StreamResponse(image_bytes())], target
            )
            self.assertTrue(target.exists())
        self.assertEqual(result.status, GenerationStatus.SUCCESS)
        self.assertEqual(result.output_path, str(target.resolve()))
        self.assertFalse(result.can_fallback)
        self.assertEqual(urlopen.call_count, 2)

    def test_non_result_owned_return_fails_closed_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(gen_slide_doubao, "_gen_owned", return_value=True):
            target = Path(temp_dir) / "slide.jpg"
            result = gen_slide_doubao.generate_result("prompt", str(target), retries=0)
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
        self.assertFalse(result.can_fallback)
        self.assertIsNone(result.output_path)
        self.assertFalse(target.exists())

    def test_safe_messages_redact_known_and_key_like_secrets(self):
        key = "known-test-key"
        message = f"request contained {key} and sk-remoteSecret123"
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"DOUBAO_API_KEY": key}, clear=True), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=http_error(400, "invalid_prompt", message),
             ):
            result = gen_slide_doubao.generate_result(
                "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
            )
        self.assertNotIn(key, result.safe_message)
        self.assertNotIn("sk-remoteSecret123", result.safe_message)
        self.assertIn("[REDACTED]", result.safe_message)


if __name__ == "__main__":
    unittest.main()
