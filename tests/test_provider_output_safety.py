import base64
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

import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai


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

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class RawResponse(JsonResponse):
    def __init__(self, payload):
        self.payload = payload
        self.stream = io.BytesIO(payload)
        self.headers = {}


class StreamResponse:
    def __init__(self, data, headers=None):
        self.stream = io.BytesIO(data)
        self.headers = headers or {
            "Content-Length": str(len(data)),
            "Content-Type": "image/jpeg",
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
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


class ProviderOutputContractTests(unittest.TestCase):
    def test_existing_output_refuses_before_credentials_and_network_by_default(self):
        for provider in (gen_slide_openai, gen_slide_gemini, gen_slide_doubao):
            with self.subTest(provider=provider.__name__), tempfile.TemporaryDirectory() as td:
                target = Path(td) / "slide.jpg"
                target.write_bytes(b"existing")
                with mock.patch.object(provider, "_load_api_key") as load_key, \
                     mock.patch.object(provider.urllib.request, "urlopen") as urlopen:
                    self.assertFalse(provider.gen("prompt", str(target), retries=0))
                load_key.assert_not_called()
                urlopen.assert_not_called()
                self.assertEqual(target.read_bytes(), b"existing")

    def test_invalid_retry_values_fail_before_credentials_and_network(self):
        invalid_values = (-1, "2", 1.5, None, True, 11)
        for provider in (gen_slide_openai, gen_slide_gemini, gen_slide_doubao):
            for retries in invalid_values:
                with self.subTest(provider=provider.__name__, retries=retries):
                    with mock.patch.object(provider, "_load_api_key") as load_key, \
                         mock.patch.object(
                             provider, "resolve_output_path"
                         ) as resolve_output_path, \
                         mock.patch.object(provider, "output_lock") as output_lock, \
                         mock.patch.object(provider.urllib.request, "urlopen") as urlopen:
                        self.assertFalse(
                            provider.gen("prompt", "slide.jpg", retries=retries)
                        )
                    resolve_output_path.assert_not_called()
                    output_lock.assert_not_called()
                    load_key.assert_not_called()
                    urlopen.assert_not_called()

    def test_deep_success_json_is_controlled_by_each_direct_cli(self):
        deep_json = b"[" * 1500 + b"0" + b"]" * 1500
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )
        for provider, filename in providers:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(
                     provider, "_load_api_key", return_value="mock-key"
                 ), mock.patch.object(
                     provider.urllib.request,
                     "urlopen",
                     return_value=RawResponse(deep_json),
                 ) as urlopen:
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = provider.main(
                        [str(Path(temp_dir) / filename), "prompt"]
                    )

            self.assertEqual(exit_code, 1)
            self.assertNotIn("Traceback", output.getvalue())
            self.assertIn("invalid", output.getvalue().lower())
            self.assertEqual(urlopen.call_count, 1)

    def test_openai_rejects_non_image_wrong_ratio_and_wrong_format(self):
        cases = (
            (b"<html>not an image</html>", "slide.jpg"),
            (image_bytes(size=(100, 100)), "slide.jpg"),
            (image_bytes("PNG"), "slide.jpg"),
        )
        for response_data, filename in cases:
            with self.subTest(filename=filename, data_size=len(response_data)), \
                 tempfile.TemporaryDirectory() as td:
                target = Path(td) / filename
                target.write_bytes(b"existing")
                response = JsonResponse({
                    "data": [{
                        "b64_json": base64.b64encode(response_data).decode("ascii")
                    }]
                })
                with mock.patch.object(gen_slide_openai, "_load_api_key", return_value="key"), \
                     mock.patch.object(
                         gen_slide_openai.urllib.request,
                         "urlopen",
                         return_value=response,
                     ):
                    self.assertFalse(
                        gen_slide_openai.gen(
                            "prompt", str(target), retries=2, overwrite=True
                        )
                    )
                self.assertEqual(target.read_bytes(), b"existing")
                self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_openai_atomic_replace_failure_preserves_existing_target_and_cleans_temp(self):
        target_image = image_bytes("JPEG")
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(target_image).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "slide.jpg"
            target.write_bytes(b"existing")
            with mock.patch.object(gen_slide_openai, "_load_api_key", return_value="key"), \
                 mock.patch.object(
                     gen_slide_openai.urllib.request,
                     "urlopen",
                     return_value=response,
                ), mock.patch("image_output.os.rename", side_effect=OSError("replace failed")):
                self.assertFalse(
                    gen_slide_openai.gen(
                        "prompt", str(target), retries=0, overwrite=True
                    )
                )
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])


class GeminiContractTests(unittest.TestCase):
    def test_stable_model_header_and_16_by_9_2k_request(self):
        response = JsonResponse(gemini_payload(image_bytes("JPEG")))
        environment = {"GEMINI_IMAGE_MODEL": "gemini-test-image"}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.dict(os.environ, environment, clear=True), \
             mock.patch.object(gen_slide_gemini, "_load_api_key", return_value="secret-key"), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=response,
             ) as urlopen:
            target = Path(td) / "nested" / "slide.jpg"
            self.assertTrue(
                gen_slide_gemini.gen("prompt", str(target), retries=0)
            )

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertIn("/models/gemini-test-image:generateContent", request.full_url)
        self.assertNotIn("secret-key", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "secret-key")
        self.assertEqual(
            body["generationConfig"]["responseFormat"]["image"],
            {
                "aspectRatio": "16:9",
                "imageSize": "2K",
                "mimeType": "IMAGE_JPEG",
            },
        )

    def test_default_model_is_current_stable_image_model(self):
        self.assertEqual(
            gen_slide_gemini.DEFAULT_MODEL, "gemini-3.1-flash-image"
        )

    def test_invalid_base64_is_not_retried_or_published(self):
        response = JsonResponse({
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": "!!!!",
                        }
                    }]
                }
            }]
        })
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_gemini, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=response,
             ) as urlopen, mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
            target = Path(td) / "slide.jpg"
            self.assertFalse(
                gen_slide_gemini.gen("prompt", str(target), retries=2)
            )
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()
        self.assertFalse(target.exists())

    def test_non_text_mime_type_is_a_controlled_non_retryable_failure(self):
        response = JsonResponse({
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": {"unexpected": "object"},
                            "data": base64.b64encode(image_bytes("JPEG")).decode("ascii"),
                        }
                    }]
                }
            }]
        })
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_gemini, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_gemini.urllib.request,
                 "urlopen",
                 return_value=response,
             ) as urlopen, mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
            try:
                result = gen_slide_gemini.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=2
                )
            except AttributeError as error:
                self.fail(f"malformed MIME type escaped provider boundary: {error}")
        self.assertFalse(result)
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_http_400_is_not_retried_but_429_is(self):
        error_400 = urllib.error.HTTPError(
            "https://provider.invalid",
            400,
            "bad request",
            {},
            io.BytesIO(b'{"error":{"message":"bad input"}}'),
        )
        error_429 = urllib.error.HTTPError(
            "https://provider.invalid",
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"error":{"message":"try later"}}'),
        )
        success = JsonResponse(gemini_payload(image_bytes("JPEG")))
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "slide.jpg"
            with mock.patch.object(gen_slide_gemini, "_load_api_key", return_value="key"), \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     side_effect=error_400,
                 ) as urlopen, mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
                self.assertFalse(
                    gen_slide_gemini.gen("prompt", str(target), retries=2)
                )
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

            with mock.patch.object(gen_slide_gemini, "_load_api_key", return_value="key"), \
                 mock.patch.object(
                     gen_slide_gemini.urllib.request,
                     "urlopen",
                     side_effect=[error_429, success],
                 ) as urlopen, mock.patch.object(gen_slide_gemini.time, "sleep") as sleep:
                self.assertTrue(
                    gen_slide_gemini.gen("prompt", str(target), retries=1)
                )
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once()


class DoubaoContractTests(unittest.TestCase):
    def test_requested_suffix_sets_output_format_and_download_is_bounded(self):
        generated = JsonResponse({"data": [{"url": "https://cdn.invalid/image"}]})
        downloaded = StreamResponse(
            image_bytes("PNG"),
            {"Content-Length": str(len(image_bytes("PNG"))), "Content-Type": "image/png"},
        )
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated, downloaded],
             ) as urlopen:
            target = Path(td) / "nested" / "slide.png"
            self.assertTrue(gen_slide_doubao.gen("prompt", str(target), retries=0))

        post_request = urlopen.call_args_list[0].args[0]
        post_body = json.loads(post_request.data)
        self.assertEqual(post_body["output_format"], "png")
        self.assertEqual(urlopen.call_args_list[0].kwargs["timeout"], 180)
        self.assertEqual(urlopen.call_args_list[1].kwargs["timeout"], 120)

    def test_html_and_oversized_downloads_are_not_published(self):
        for downloaded in (
            StreamResponse(b"<html>upstream error</html>"),
            StreamResponse(
                b"x",
                {"Content-Length": str(gen_slide_doubao.MAX_IMAGE_BYTES + 1)},
            ),
        ):
            with self.subTest(headers=downloaded.headers), tempfile.TemporaryDirectory() as td:
                target = Path(td) / "slide.jpg"
                target.write_bytes(b"existing")
                generated = JsonResponse({
                    "data": [{"url": "https://cdn.invalid/image"}]
                })
                with mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
                     mock.patch.object(
                         gen_slide_doubao.urllib.request,
                         "urlopen",
                         side_effect=[generated, downloaded],
                     ) as urlopen:
                    self.assertFalse(
                        gen_slide_doubao.gen(
                            "prompt", str(target), retries=2, overwrite=True
                        )
                    )
                self.assertEqual(urlopen.call_count, 2)
                self.assertEqual(target.read_bytes(), b"existing")
                self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_download_timeout_retries_same_cdn_url_without_regenerating(self):
        generated = JsonResponse({"data": [{"url": "https://cdn.invalid/image"}]})
        downloaded = StreamResponse(image_bytes("JPEG"))
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=[generated, TimeoutError("slow CDN"), downloaded],
             ) as urlopen, mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            target = Path(td) / "slide.jpg"
            self.assertTrue(gen_slide_doubao.gen("prompt", str(target), retries=1))

        self.assertEqual(urlopen.call_count, 3)
        post_urls = [
            call.args[0].full_url
            for call in urlopen.call_args_list
            if call.args[0].data is not None
        ]
        self.assertEqual(post_urls, [gen_slide_doubao.URL])
        sleep.assert_called_once()

    def test_http_400_is_not_retried(self):
        error = urllib.error.HTTPError(
            gen_slide_doubao.URL,
            400,
            "bad request",
            {},
            io.BytesIO(b'{"error":{"message":"invalid prompt"}}'),
        )
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(gen_slide_doubao, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_doubao.urllib.request,
                 "urlopen",
                 side_effect=error,
             ) as urlopen, mock.patch.object(gen_slide_doubao.time, "sleep") as sleep:
            self.assertFalse(
                gen_slide_doubao.gen(
                    "prompt", str(Path(td) / "slide.jpg"), retries=2
                )
            )
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()


class DirectProviderCliTests(unittest.TestCase):
    def test_force_flag_enables_overwrite_for_each_provider(self):
        for provider in (gen_slide_openai, gen_slide_gemini, gen_slide_doubao):
            with self.subTest(provider=provider.__name__), \
                 mock.patch.object(provider, "gen", return_value=True) as gen:
                self.assertEqual(
                    provider.main(["slide.jpg", "prompt", "--force"]), 0
                )
            gen.assert_called_once_with(
                "prompt", "slide.jpg", overwrite=True
            )

    def test_provider_failure_maps_to_cli_exit_one(self):
        for provider in (gen_slide_openai, gen_slide_gemini, gen_slide_doubao):
            with self.subTest(provider=provider.__name__), \
                 mock.patch.object(provider, "gen", return_value=False):
                self.assertEqual(provider.main(["slide.jpg", "prompt"]), 1)


if __name__ == "__main__":
    unittest.main()
