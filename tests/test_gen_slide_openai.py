import base64
import http.client
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide_openai


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")
        self.stream = io.BytesIO(self.payload)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class ReadFailureResponse:
    def __init__(self, error):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size=-1):
        raise self.error


class RawResponse:
    def __init__(self, payload):
        self.payload = payload
        self.returned = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size=-1):
        if self.returned:
            return b""
        self.returned = True
        return self.payload


class OpenAIConfigTests(unittest.TestCase):
    def test_environment_key_has_priority_over_secret_file(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True), \
             mock.patch.object(Path, "read_text", return_value="file-key") as read_text:
            self.assertEqual(gen_slide_openai._load_api_key(), "env-key")
            read_text.assert_not_called()

    def test_secret_file_is_used_when_environment_key_is_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(Path, "read_text", return_value="file-key\n"):
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
        generated_image = image_bytes("JPEG")
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(generated_image).decode("ascii")}]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "slide.jpg"
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response) as urlopen, \
                 redirect_stdout(output):
                ok = gen_slide_openai.gen("draw a slide", str(out_path), retries=0)

            self.assertTrue(ok)
            self.assertEqual(out_path.read_bytes(), generated_image)
            self.assertIn("(0KB, OpenAI gpt-image-2)", output.getvalue())
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

    def test_gen_redacts_environment_key_from_success_and_final_failure_paths(self):
        key = "known-test-key"
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / key / "slide.jpg"
            response = FakeResponse({
                "data": [{"b64_json": base64.b64encode(image_bytes()).decode()}]
            })
            success_output = io.StringIO()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     return_value=response,
                 ), \
                 redirect_stdout(success_output):
                self.assertTrue(gen_slide_openai.gen("prompt", str(target), retries=0))
            self.assertNotIn(key, success_output.getvalue())
            self.assertIn("(0KB, OpenAI gpt-image-2)", success_output.getvalue())

            failure_output = io.StringIO()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai,
                     "output_lock",
                     side_effect=gen_slide_openai.OutputLockError(f"locked {target}"),
                 ), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen, \
                 redirect_stdout(failure_output):
                self.assertFalse(gen_slide_openai.gen("prompt", str(target), retries=0))
            self.assertNotIn(key, failure_output.getvalue())
            self.assertEqual(failure_output.getvalue().count("  ERR:"), 1)
            urlopen.assert_not_called()

    def test_terminal_http_and_transport_errors_print_only_the_final_error(self):
        failures = (
            urllib.error.HTTPError(
                gen_slide_openai.API_URL,
                400,
                "bad request",
                {},
                io.BytesIO(b'{"error":{"message":"bad input"}}'),
            ),
            urllib.error.URLError("temporary outage"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     side_effect=failure,
                 ):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertFalse(
                        gen_slide_openai.gen(
                            "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                        )
                    )
                self.assertEqual(output.getvalue().count("  ERR:"), 1)
                self.assertNotIn("  HTTP", output.getvalue())

    def test_environment_overrides_model_size_and_quality(self):
        response = FakeResponse({
            "data": [{
                "b64_json": base64.b64encode(image_bytes("PNG")).decode("ascii")
            }]
        })
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
                self.assertFalse(
                    gen_slide_openai.gen(
                        "prompt", str(out_path), retries=0, overwrite=True
                    )
                )
            self.assertEqual(out_path.read_bytes(), b"existing")
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_malformed_json_does_not_overwrite_existing_file(self):
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(b"{not-json")
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "slide.jpg"
            out_path.write_bytes(b"existing")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response):
                self.assertFalse(
                    gen_slide_openai.gen(
                        "prompt", str(out_path), retries=0, overwrite=True
                    )
                )
            self.assertEqual(out_path.read_bytes(), b"existing")

    def test_429_retries_then_succeeds(self):
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"error":{"message":"slow down"}}'),
        )
        response = FakeResponse({"data": [{"b64_json": base64.b64encode(image_bytes("JPEG")).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=[error, response]) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=1))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_500_retries_then_succeeds(self):
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            500,
            "server error",
            {},
            io.BytesIO(b'{"error":{"message":"try later"}}'),
        )
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes("JPEG")).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=[error, response]) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=1))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_url_error_retries_then_succeeds(self):
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes("JPEG")).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 side_effect=[urllib.error.URLError("temporary outage"), response],
             ) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=1))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_truncated_successful_read_is_retryable(self):
        truncated = ReadFailureResponse(http.client.IncompleteRead(b'{"data":', 20))
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes("JPEG")).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=[truncated, response]) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            try:
                result = gen_slide_openai.gen(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=1
                )
            except http.client.IncompleteRead as error:
                self.fail(f"truncated response escaped instead of being retried: {error}")
            self.assertTrue(result)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_unexpected_success_body_type_returns_false(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 return_value=RawResponse(None),
             ):
            try:
                result = gen_slide_openai.gen(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
            except TypeError as error:
                self.fail(f"successful response body type escaped: {error}")
            self.assertFalse(result)

    def test_non_object_success_envelopes_return_false(self):
        payloads = [[], "text", {"data": {}}, {"data": ["not-an-object"]}]
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     return_value=FakeResponse(payload),
                 ):
                self.assertFalse(
                    gen_slide_openai.gen(
                        "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                    )
                )

    def test_malformed_and_non_object_http_error_bodies_return_false(self):
        bodies = [b"{not-json", b"[]", b'{"error":"not-an-object"}', "plain text"]
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temp_dir:
                stream = io.StringIO(body) if isinstance(body, str) else io.BytesIO(body)
                error = urllib.error.HTTPError(
                    gen_slide_openai.API_URL, 400, "bad request", {}, stream
                )
                with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                     mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=error):
                    try:
                        result = gen_slide_openai.gen(
                            "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                        )
                    except (AttributeError, TypeError) as escaped:
                        self.fail(f"malformed HTTP error body escaped: {escaped}")
                    self.assertFalse(result)

    def test_http_error_body_read_failure_returns_false(self):
        body = mock.Mock()
        body.read.side_effect = http.client.IncompleteRead(b"partial", 20)
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL, 400, "bad request", {}, body
        )
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=error):
            try:
                result = gen_slide_openai.gen(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
            except http.client.IncompleteRead as escaped:
                self.fail(f"HTTP error body read failure escaped: {escaped}")
            self.assertFalse(result)

    def test_authentication_errors_preserve_safe_remote_message_and_redact_secrets(self):
        key = "known-test-key"
        remote_message = (
            "Your organization must be verified to use this model; "
            f"request included {key}; also sk-remoteEcho123"
        )
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            401,
            "unauthorized",
            {},
            io.BytesIO(json.dumps({"error": {"message": remote_message}}).encode()),
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=error), \
             redirect_stdout(output):
            self.assertFalse(
                gen_slide_openai.gen(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
            )
        self.assertIn("authentication failed", output.getvalue().lower())
        self.assertIn("organization must be verified", output.getvalue())
        self.assertNotIn(key, output.getvalue())
        self.assertNotIn("sk-remoteEcho123", output.getvalue())

    def test_other_http_errors_redact_known_and_key_like_secrets(self):
        key = "known-test-key"
        remote_message = f"request included {key} and sk-remoteEcho123"
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            400,
            "bad request",
            {},
            io.BytesIO(json.dumps({"error": {"message": remote_message}}).encode()),
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=error), \
             redirect_stdout(output):
            self.assertFalse(
                gen_slide_openai.gen(
                    "prompt", str(Path(temp_dir) / "slide.jpg"), retries=0
                )
            )
        self.assertIn("[REDACTED]", output.getvalue())
        self.assertNotIn(key, output.getvalue())
        self.assertNotIn("sk-remoteEcho123", output.getvalue())

    def test_missing_nested_parent_is_created_before_request(self):
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes("JPEG")).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "missing" / "nested"
            out_path = parent / "slide.jpg"

            def observe_parent(*_args, **_kwargs):
                self.assertTrue(parent.is_dir())
                return response

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     side_effect=observe_parent,
                 ):
                self.assertTrue(gen_slide_openai.gen("prompt", str(out_path), retries=0))
            with Image.open(out_path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (160, 90))

    def test_invalid_parent_fails_before_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocked_parent = Path(temp_dir) / "not-a-directory"
            blocked_parent.write_bytes(b"existing")
            out_path = blocked_parent / "slide.jpg"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen:
                try:
                    result = gen_slide_openai.gen("prompt", str(out_path), retries=0)
                except TypeError as error:
                    self.fail(f"invalid parent did not stop before networking: {error}")
                self.assertFalse(result)
            urlopen.assert_not_called()
            self.assertEqual(blocked_parent.read_bytes(), b"existing")

    def test_negative_retries_fail_before_network(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen, \
             redirect_stdout(output):
            self.assertFalse(gen_slide_openai.gen("prompt", "slide.jpg", retries=-1))
        urlopen.assert_not_called()
        self.assertIn("invalid generation arguments", output.getvalue())

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
