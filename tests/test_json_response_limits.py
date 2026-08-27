import gc
import io
import json
import os
import sys
import tempfile
import tracemalloc
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import image_output
import vision_check_gemini


class RawResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class JsonResponseLimitTests(unittest.TestCase):
    def test_duplicate_object_keys_are_rejected_instead_of_collapsed(self):
        payload = '{"data": 0, "data": 1, "data": 2}'

        with self.assertRaisesRegex(
            image_output.ImageOutputError,
            "provider response is not valid JSON",
        ):
            image_output.parse_json_response(payload)

    def test_container_item_limit_is_enforced_before_json_decode(self):
        limit = image_output.MAX_JSON_CONTAINER_ITEMS
        payload = b"[" + b",".join([b"null"] * (limit + 1)) + b"]"

        with mock.patch.object(
            image_output.json,
            "loads",
            side_effect=AssertionError("oversized JSON reached json.loads"),
        ) as loads, self.assertRaisesRegex(
            image_output.ImageOutputError,
            "provider response is not valid JSON",
        ):
            image_output.parse_json_response(payload)

        loads.assert_not_called()

    def test_total_node_limit_is_enforced_before_json_decode(self):
        items_per_group = min(image_output.MAX_JSON_CONTAINER_ITEMS, 1000)
        group_count = image_output.MAX_JSON_NODES // (items_per_group + 1) + 1
        group = b"[" + b",".join([b"null"] * items_per_group) + b"]"
        payload = b"[" + b",".join([group] * group_count) + b"]"

        with mock.patch.object(
            image_output.json,
            "loads",
            side_effect=AssertionError("oversized JSON reached json.loads"),
        ) as loads, self.assertRaisesRegex(
            image_output.ImageOutputError,
            "provider response is not valid JSON",
        ):
            image_output.parse_json_response(payload)

        loads.assert_not_called()

    def test_predecode_scan_ignores_escaped_structure_inside_large_strings(self):
        encoded = "W10sOnt9LFxcXFwi" * (1024 * 1024 // 16)
        payload = json.dumps({
            "data": [{
                "b64_json": encoded,
                "note": "escaped quote: \\\" and brackets: [ ] { } , :",
            }]
        })

        parsed = image_output.parse_json_response(payload)

        self.assertEqual(parsed["data"][0]["b64_json"], encoded)
        self.assertIn("[ ] { } , :", parsed["data"][0]["note"])

    def test_wide_array_and_object_exceeding_item_limit_are_controlled(self):
        limit = image_output.MAX_JSON_CONTAINER_ITEMS
        payloads = (
            json.dumps([None] * (limit + 1)),
            json.dumps({str(index): None for index in range(limit + 1)}),
        )

        for payload in payloads:
            with self.subTest(prefix=payload[:1]), self.assertRaisesRegex(
                image_output.ImageOutputError,
                "provider response is not valid JSON",
            ):
                image_output.parse_json_response(payload)

    def test_total_node_limit_applies_across_narrow_containers(self):
        items_per_group = min(image_output.MAX_JSON_CONTAINER_ITEMS, 1000)
        group_count = image_output.MAX_JSON_NODES // (items_per_group + 1) + 1
        self.assertLessEqual(group_count, image_output.MAX_JSON_CONTAINER_ITEMS)
        payload = json.dumps(
            [[None] * items_per_group for _ in range(group_count)]
        )

        with self.assertRaisesRegex(
            image_output.ImageOutputError,
            "provider response is not valid JSON",
        ):
            image_output.parse_json_response(payload)

    def test_nesting_boundary_remains_deterministic(self):
        allowed = image_output.MAX_JSON_NESTING + 1
        accepted = "[" * allowed + "null" + "]" * allowed
        rejected = "[" * (allowed + 1) + "null" + "]" * (allowed + 1)

        self.assertIsNotNone(image_output.parse_json_response(accepted))
        with self.assertRaises(image_output.ImageOutputError):
            image_output.parse_json_response(rejected)

    def test_wide_traversal_has_no_per_child_pending_frame_amplification(self):
        width = 120_000
        payload = "[" + ",".join(["null"] * width) + "]"
        gc.collect()
        tracemalloc.start()
        try:
            with mock.patch.object(
                image_output, "MAX_JSON_CONTAINER_ITEMS", width
            ), mock.patch.object(image_output, "MAX_JSON_NODES", width + 1):
                parsed = image_output.parse_json_response(payload)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertEqual(len(parsed), width)
        self.assertLess(
            peak - current,
            4 * 1024 * 1024,
            "traversal retained one pending frame per sibling",
        )


class ProviderJsonResponseLimitTests(unittest.TestCase):
    def test_generator_http_error_json_limits_fall_back_without_leaking(self):
        key = "known-http-error-key"
        unsafe = f"should-not-surface {key} sk-httpErrorEcho123"
        bodies = (
            (
                "wide",
                json.dumps({
                    "error": {"message": unsafe},
                    "wide": [0, 1, 2],
                }).encode("utf-8"),
            ),
            (
                "deep",
                json.dumps({
                    "error": {"message": unsafe},
                    "deep": [[[[0]]]],
                }).encode("utf-8"),
            ),
            (
                "duplicate",
                (
                    '{"error":{"message":"first"},'
                    f'"error":{{"message":{json.dumps(unsafe)}}}}}'
                ).encode("utf-8"),
            ),
        )
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )

        with mock.patch.object(
            image_output, "MAX_JSON_CONTAINER_ITEMS", 2
        ), mock.patch.object(image_output, "MAX_JSON_NESTING", 2):
            for provider, filename in providers:
                for label, body in bodies:
                    with self.subTest(provider=provider.__name__, body=label), \
                         tempfile.TemporaryDirectory() as temp_dir, \
                         mock.patch.dict(os.environ, {}, clear=True), \
                         mock.patch.object(
                             provider, "_load_api_key", return_value=key
                         ), mock.patch.object(
                             provider.urllib.request,
                             "urlopen",
                             side_effect=urllib.error.HTTPError(
                                 "https://provider.invalid",
                                 400,
                                 "bad request",
                                 {},
                                 io.BytesIO(body),
                             ),
                         ):
                        output = io.StringIO()
                        with redirect_stdout(output), redirect_stderr(output):
                            exit_code = provider.main(
                                [str(Path(temp_dir) / filename), "prompt"]
                            )

                    rendered = output.getvalue()
                    self.assertEqual(exit_code, 1)
                    self.assertIn("bad request", rendered)
                    self.assertNotIn("should-not-surface", rendered)
                    self.assertNotIn(key, rendered)
                    self.assertNotIn("sk-httpErrorEcho123", rendered)
                    self.assertNotIn("Traceback", rendered)

    def test_generator_http_error_preserves_safe_bounded_message(self):
        key = "known-http-error-key"
        message = f"provider details include {key} and sk-httpErrorEcho123"
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )

        for provider, filename in providers:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(provider, "_load_api_key", return_value=key), \
                 mock.patch.object(
                     provider.urllib.request,
                     "urlopen",
                     side_effect=urllib.error.HTTPError(
                         "https://provider.invalid",
                         400,
                         "bad request",
                         {},
                         io.BytesIO(body),
                     ),
                 ):
                    output = io.StringIO()
                    with redirect_stdout(output), redirect_stderr(output):
                        exit_code = provider.main(
                            [str(Path(temp_dir) / filename), "prompt"]
                        )

            rendered = output.getvalue()
            self.assertEqual(exit_code, 1)
            self.assertIn("provider details include", rendered)
            self.assertIn("[REDACTED]", rendered)
            self.assertNotIn(key, rendered)
            self.assertNotIn("sk-httpErrorEcho123", rendered)
            self.assertNotIn("Traceback", rendered)

    def test_generator_clis_control_wide_json_with_mocked_transport(self):
        response_body = json.dumps({"wide": [None] * 5}).encode("utf-8")
        providers = (
            (gen_slide_openai, "slide.jpg"),
            (gen_slide_gemini, "slide.png"),
            (gen_slide_doubao, "slide.jpg"),
        )

        with mock.patch.object(image_output, "MAX_JSON_CONTAINER_ITEMS", 4):
            for provider, filename in providers:
                with self.subTest(provider=provider.__name__), \
                     tempfile.TemporaryDirectory() as temp_dir, \
                     mock.patch.object(
                         provider, "_load_api_key", return_value="mock-key"
                     ), mock.patch.object(
                         provider.urllib.request,
                         "urlopen",
                         return_value=RawResponse(response_body),
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

    def test_vision_cli_controls_wide_json_with_mocked_transport(self):
        response_body = json.dumps({"wide": [None] * 5}).encode("utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "slide.png"
            Image.new("RGB", (160, 90), "navy").save(image_path)
            stderr = io.StringIO()
            with mock.patch.object(
                image_output, "MAX_JSON_CONTAINER_ITEMS", 4
            ), mock.patch.object(
                vision_check_gemini, "_load_api_key", return_value="mock-key"
            ), mock.patch.object(
                vision_check_gemini.urllib.request,
                "urlopen",
                return_value=RawResponse(response_body),
            ) as urlopen, redirect_stderr(stderr):
                exit_code = vision_check_gemini.main(
                    [str(image_path), "--retries", "0"]
                )

        self.assertEqual(exit_code, 1)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertIn("invalid JSON", stderr.getvalue())
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
