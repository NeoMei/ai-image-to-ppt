import base64
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import vision_check_gemini


RETRY_DELAY_PATH = SCRIPTS / "retry_delay.py"
if RETRY_DELAY_PATH.exists():
    spec = importlib.util.spec_from_file_location("retry_delay", RETRY_DELAY_PATH)
    retry_delay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retry_delay)
else:
    retry_delay = None


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class JsonResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")
        self.stream = io.BytesIO(self.payload)
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class StreamResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
        self.headers = {
            "Content-Length": str(len(payload)),
            "Content-Type": "image/jpeg",
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def http_error(status, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(
        "https://provider.invalid",
        status,
        "transient",
        headers,
        io.BytesIO(b'{"error":{"message":"retry later"}}'),
    )


class RetryDelayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(retry_delay, "shared retry_delay module is missing")

    def test_delta_seconds_is_parsed_and_clamped(self):
        self.assertEqual(retry_delay.retry_delay(0, {"Retry-After": "7"}), 7)
        self.assertEqual(retry_delay.retry_delay(0, {"Retry-After": "9999"}), 60)

    def test_http_date_is_relative_to_now_and_clamped(self):
        now = datetime(2026, 8, 26, 0, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(
            retry_delay.retry_delay(
                0,
                {"Retry-After": "Wed, 26 Aug 2026 00:00:12 GMT"},
                now=now,
            ),
            12,
        )
        self.assertEqual(
            retry_delay.retry_delay(
                0,
                {"Retry-After": "Tue, 25 Aug 2026 23:59:00 GMT"},
                now=now,
            ),
            0,
        )

    def test_missing_or_invalid_header_uses_bounded_exponential_backoff(self):
        self.assertEqual(retry_delay.retry_delay(0, {}), 2)
        self.assertEqual(
            retry_delay.retry_delay(1, {"Retry-After": "not-a-delay"}),
            4,
        )
        self.assertEqual(retry_delay.retry_delay(20, None), 60)

    def test_non_ascii_digit_retry_after_uses_fallback(self):
        self.assertEqual(
            retry_delay.retry_delay(0, {"Retry-After": "²"}),
            2,
        )


class RetryAfterIntegrationTests(unittest.TestCase):
    def test_openai_generation_uses_retry_after(self):
        generated = base64.b64encode(image_bytes()).decode("ascii")
        success = JsonResponse({"data": [{"b64_json": generated}]})
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_openai, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 side_effect=[http_error(429, "7"), success],
             ), mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertTrue(
                gen_slide_openai.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=1
                )
            )
        sleep.assert_called_once_with(7)

    def test_gemini_generation_uses_retry_after(self):
        generated = base64.b64encode(image_bytes()).decode("ascii")
        success = JsonResponse({
            "candidates": [{
                "content": {"parts": [{
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": generated,
                    }
                }]}
            }]
        })
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_gemini, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 side_effect=[http_error(503, "9"), success],
             ), mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
            self.assertTrue(
                gen_slide_gemini.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=1
                )
            )
        sleep.assert_called_once_with(9)

    def test_doubao_generation_uses_retry_after(self):
        generated = JsonResponse({"data": [{"url": "https://cdn.invalid/image"}]})
        downloaded = StreamResponse(image_bytes())
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[http_error(500, "11"), generated, downloaded],
             ), mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            self.assertTrue(
                gen_slide_doubao.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=1
                )
            )
        sleep.assert_called_once_with(11)

    def test_doubao_download_uses_retry_after(self):
        generated = JsonResponse({"data": [{"url": "https://cdn.invalid/image"}]})
        downloaded = StreamResponse(image_bytes())
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated, http_error(503, "13"), downloaded],
             ), mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            self.assertTrue(
                gen_slide_doubao.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=1
                )
            )
        sleep.assert_called_once_with(13)

    def test_vision_uses_retry_after(self):
        success = JsonResponse({
            "candidates": [{"content": {"parts": [{"text": "recovered"}]}}]
        })
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / "slide.png"
            Image.new("RGB", (32, 18), "navy").save(image_path)
            with mock.patch.dict(
                os.environ, {"GEMINI_API_KEY": "key"}, clear=True
            ), mock.patch.object(
                vision_check_gemini.urllib.request,
                "urlopen",
                side_effect=[http_error(429, "17"), success],
            ), mock.patch.object(vision_check_gemini.time, "sleep") as sleep:
                self.assertEqual(
                    vision_check_gemini.check(str(image_path), retries=1),
                    "recovered",
                )
        sleep.assert_called_once_with(17)

    def test_non_retryable_download_does_not_sleep(self):
        generated = JsonResponse({"data": [{"url": "https://cdn.invalid/image"}]})
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated, http_error(404, "30")],
             ), mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            self.assertFalse(
                gen_slide_doubao.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=2
                )
            )
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
