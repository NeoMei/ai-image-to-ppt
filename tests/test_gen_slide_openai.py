import base64
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide_openai


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


class OpenAIConfigTests(unittest.TestCase):
    def test_environment_key_has_priority_over_secret_file(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True), \
             mock.patch.object(Path, "read_text", return_value="file-key") as read_text:
            self.assertEqual(gen_slide_openai._load_api_key(), "env-key")
            read_text.assert_not_called()

    def test_secret_file_is_used_when_environment_key_is_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(Path, "read_text", return_value=" file-key\n"):
            self.assertEqual(gen_slide_openai._load_api_key(), "file-key")

    def test_output_format_matches_supported_file_extension(self):
        self.assertEqual(gen_slide_openai._output_format("slide.jpg"), "jpeg")
        self.assertEqual(gen_slide_openai._output_format("slide.jpeg"), "jpeg")
        self.assertEqual(gen_slide_openai._output_format("slide.png"), "png")
        self.assertEqual(gen_slide_openai._output_format("slide.webp"), "webp")
        with self.assertRaisesRegex(ValueError, "Unsupported output extension"):
            gen_slide_openai._output_format("slide.gif")


class OpenAIGenerationTests(unittest.TestCase):
    def test_success_uses_confirmed_defaults_and_writes_decoded_image(self):
        image_bytes = b"fake-jpeg-bytes"
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes).decode("ascii")}]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "slide.jpg"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response) as urlopen:
                ok = gen_slide_openai.gen("draw a slide", str(out_path), retries=0)

            self.assertTrue(ok)
            self.assertEqual(out_path.read_bytes(), image_bytes)
            request = urlopen.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, gen_slide_openai.API_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(payload, {
                "model": "gpt-image-2",
                "prompt": "draw a slide",
                "size": "2048x1152",
                "quality": "medium",
                "output_format": "jpeg",
            })

    def test_environment_overrides_model_size_and_quality(self):
        response = FakeResponse({"data": [{"b64_json": base64.b64encode(b"image").decode("ascii")}]})
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_IMAGE_MODEL": "gpt-image-test",
            "OPENAI_IMAGE_SIZE": "1536x1024",
            "OPENAI_IMAGE_QUALITY": "high",
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, environment, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.png"), retries=0))

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "gpt-image-test")
        self.assertEqual(payload["size"], "1536x1024")
        self.assertEqual(payload["quality"], "high")
        self.assertEqual(payload["output_format"], "png")

    def test_invalid_base64_does_not_overwrite_existing_file(self):
        response = FakeResponse({"data": [{"b64_json": "%%%invalid%%%"}]})
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "slide.jpg"
            out_path.write_bytes(b"existing")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response):
                self.assertFalse(gen_slide_openai.gen("prompt", str(out_path), retries=0))
            self.assertEqual(out_path.read_bytes(), b"existing")
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_429_retries_then_succeeds(self):
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"error":{"message":"slow down"}}'),
        )
        response = FakeResponse({"data": [{"b64_json": base64.b64encode(b"image").decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=[error, response]) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=1))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_non_retryable_400_stops_after_one_request(self):
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            400,
            "bad request",
            {},
            io.BytesIO(b'{"error":{"message":"invalid size"}}'),
        )
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=error) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertFalse(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=2))
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_unsupported_extension_fails_before_loading_key_or_network(self):
        with mock.patch.object(gen_slide_openai, "_load_api_key") as load_key, \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen:
            self.assertFalse(gen_slide_openai.gen("prompt", "slide.gif"))
        load_key.assert_not_called()
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
