import base64
import http.client
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import image_output
import export_images
import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import prepare_editable_input


def png_header(width, height):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    result = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")
    if len(result) != 45:
        raise AssertionError("crafted PNG header must stay tiny")
    return result


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


class StreamResponse:
    def __init__(self, payload):
        self.stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class RecordingOversizedResponse:
    def __init__(self):
        self.read_sizes = []
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        self.read_sizes.append(size)
        return b"x" * max(size, 0)


class ChunkedResponse:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.read_sizes = []

    def read(self, size):
        self.read_sizes.append(size)
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > size:
            raise AssertionError("test chunk exceeded the requested read size")
        return chunk


class UnsizedOnlyResponse:
    def __init__(self):
        self.sized_reads = 0
        self.unsized_reads = 0

    def read(self, *args):
        if args:
            self.sized_reads += 1
            raise TypeError("read() takes no arguments")
        self.unsized_reads += 1
        return b"x" * 1024


class EndlessTinyResponse:
    def __init__(self):
        self.reads = 0

    def read(self, _size):
        self.reads += 1
        return b"x"


class SharedImageLimitTests(unittest.TestCase):
    def test_shared_limits_are_50_mib_and_64_megapixels(self):
        self.assertEqual(
            getattr(image_output, "MAX_IMAGE_BYTES", None),
            50 * 1024 * 1024,
        )
        self.assertEqual(
            getattr(image_output, "MAX_IMAGE_PIXELS", None),
            64_000_000,
        )

    def test_base64_encoded_limit_rejects_before_decoding(self):
        with mock.patch.object(image_output, "MAX_IMAGE_BYTES", 3, create=True), \
             mock.patch.object(
                 image_output.base64,
                 "b64decode",
                 return_value=b"AAAA",
             ) as decode:
            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "Base64 exceeds",
            ):
                image_output.decode_base64("QUFBQQ==")
        decode.assert_not_called()

    def test_base64_decoded_limit_is_checked_defensively(self):
        with mock.patch.object(image_output, "MAX_IMAGE_BYTES", 3, create=True), \
             mock.patch.object(
                 image_output.base64,
                 "b64decode",
                 return_value=b"AAAA",
             ):
            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "image data exceeds",
            ):
                image_output.decode_base64("QUFB")

    def test_local_byte_limit_rejects_before_pillow_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "oversized.png"
            with source.open("wb") as stream:
                stream.truncate(image_output.MAX_IMAGE_BYTES + 1)
            with mock.patch.object(image_output.Image, "open") as image_open:
                with self.assertRaisesRegex(
                    image_output.ImageOutputError,
                    "maximum size",
                ):
                    image_output.load_image(source)
            image_open.assert_not_called()

    def test_64_megapixel_limit_rejects_tiny_bomb_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "bomb.png"
            source.write_bytes(png_header(17_920, 10_080))
            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "dimensions are too large",
            ):
                image_output.load_image(source)

    def test_publish_bytes_warns_and_succeeds_when_published_temp_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            prepared = image_output.preflight_output(str(target))
            buffer = io.BytesIO()
            Image.new("RGB", (160, 90), "navy").save(buffer, format="JPEG")
            real_unlink = image_output.os.unlink
            attempts = 0

            def deny_temp(path, *args, **kwargs):
                nonlocal attempts
                candidate = Path(path)
                if candidate.parent == root and candidate.name.endswith(".tmp"):
                    attempts += 1
                    raise PermissionError("forced temp cleanup denial")
                return real_unlink(path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                image_output.os, "unlink", side_effect=deny_temp
            ), redirect_stderr(stderr):
                byte_count = image_output.publish_bytes(buffer.getvalue(), prepared)

            self.assertGreater(byte_count, 0)
            with Image.open(target) as image:
                self.assertEqual(image.size, (160, 90))
            stale = list(root.glob(".slide.jpg.*.tmp"))
            self.assertEqual(len(stale), 1)
            self.assertLessEqual(attempts, 2)
            self.assertIn("WARN:", stderr.getvalue())
            self.assertIn(str(stale[0]), stderr.getvalue())

    def test_publish_stream_warns_and_succeeds_when_published_temp_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            prepared = image_output.preflight_output(str(target))
            buffer = io.BytesIO()
            Image.new("RGB", (160, 90), "navy").save(buffer, format="JPEG")
            response = StreamResponse(buffer.getvalue())
            real_unlink = image_output.os.unlink
            attempts = 0

            def deny_temp(path, *args, **kwargs):
                nonlocal attempts
                candidate = Path(path)
                if candidate.parent == root and candidate.name.endswith(".tmp"):
                    attempts += 1
                    raise PermissionError("forced temp cleanup denial")
                return real_unlink(path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                image_output.os, "unlink", side_effect=deny_temp
            ), redirect_stderr(stderr):
                byte_count = image_output.publish_stream(response, prepared)

            self.assertGreater(byte_count, 0)
            with Image.open(target) as image:
                self.assertEqual(image.size, (160, 90))
            stale = list(root.glob(".slide.jpg.*.tmp"))
            self.assertEqual(len(stale), 1)
            self.assertLessEqual(attempts, 2)
            self.assertIn("WARN:", stderr.getvalue())
            self.assertIn(str(stale[0]), stderr.getvalue())

    def test_invalid_bytes_survive_persistent_failure_cleanup_denial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.png"
            real_unlink = image_output.os.unlink
            attempts = 0

            def deny_temp(path, *args, **kwargs):
                nonlocal attempts
                candidate = Path(path)
                if candidate.parent == root and candidate.name.endswith(".tmp"):
                    attempts += 1
                    raise PermissionError("cleanup-denied")
                return real_unlink(path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                image_output.os, "unlink", side_effect=deny_temp
            ), redirect_stderr(stderr), self.assertRaisesRegex(
                image_output.ImageOutputError, "not a valid image"
            ):
                image_output.publish_bytes(b"not-an-image", target)

            stale = list(root.glob(".slide.png.*.tmp"))
            self.assertEqual(attempts, 2)
            self.assertEqual(len(stale), 1)
            self.assertIn("WARN:", stderr.getvalue())
            self.assertIn(str(stale[0]), stderr.getvalue())

    def test_stream_read_error_survives_persistent_failure_cleanup_denial(self):
        class FailingResponse:
            headers = {}

            def read(self, _size):
                raise OSError("stream-root-cause")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.png"
            real_unlink = image_output.os.unlink
            attempts = 0

            def deny_temp(path, *args, **kwargs):
                nonlocal attempts
                candidate = Path(path)
                if candidate.parent == root and candidate.name.endswith(".tmp"):
                    attempts += 1
                    raise PermissionError("cleanup-denied")
                return real_unlink(path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                image_output.os, "unlink", side_effect=deny_temp
            ), redirect_stderr(stderr), self.assertRaisesRegex(
                image_output.ImageStreamError, "stream-root-cause"
            ):
                image_output.publish_stream(FailingResponse(), target)

            stale = list(root.glob(".slide.png.*.tmp"))
            self.assertEqual(attempts, 2)
            self.assertEqual(len(stale), 1)
            self.assertIn("WARN:", stderr.getvalue())
            self.assertIn(str(stale[0]), stderr.getvalue())

    def test_publish_error_survives_persistent_failure_cleanup_denial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            buffer = io.BytesIO()
            Image.new("RGB", (160, 90), "navy").save(buffer, format="JPEG")
            real_unlink = image_output.os.unlink
            attempts = 0

            def deny_temp(path, *args, **kwargs):
                nonlocal attempts
                candidate = Path(path)
                if candidate.parent == root and candidate.name.endswith(".tmp"):
                    attempts += 1
                    raise PermissionError("cleanup-denied")
                return real_unlink(path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                image_output.os, "link", side_effect=OSError("publish-root-cause")
            ), mock.patch.object(
                image_output.os, "unlink", side_effect=deny_temp
            ), redirect_stderr(stderr), self.assertRaisesRegex(
                image_output.ImageOutputError, "publish-root-cause"
            ):
                image_output.publish_bytes(buffer.getvalue(), target)

            stale = list(root.glob(".slide.jpg.*.tmp"))
            self.assertEqual(attempts, 2)
            self.assertEqual(len(stale), 1)
            self.assertIn("WARN:", stderr.getvalue())
            self.assertIn(str(stale[0]), stderr.getvalue())


class ProviderResourceBoundaryTests(unittest.TestCase):
    def test_response_reader_never_falls_back_to_unbounded_read(self):
        response = UnsizedOnlyResponse()
        with mock.patch.object(image_output, "MAX_PROVIDER_RESPONSE_BYTES", 16), \
             self.assertRaisesRegex(
                 image_output.ImageOutputError,
                 "bounded reads",
             ):
            image_output.read_response_body(response)

        self.assertEqual(response.sized_reads, 1)
        self.assertEqual(response.unsized_reads, 0)

    def test_response_reader_accumulates_real_chunked_stream_within_limit(self):
        response = ChunkedResponse([b"ab", b"cd", b"e", b""])
        with mock.patch.object(image_output, "MAX_PROVIDER_RESPONSE_BYTES", 5):
            self.assertEqual(image_output.read_response_body(response), b"abcde")

        self.assertEqual(response.read_sizes, [6, 4, 2, 1])

    def test_response_reader_rejects_chunked_stream_at_max_plus_one(self):
        response = ChunkedResponse([b"ab", b"cde"])
        with mock.patch.object(image_output, "MAX_PROVIDER_RESPONSE_BYTES", 4), \
             self.assertRaisesRegex(
                 image_output.ImageOutputError,
                 "exceeds maximum size",
             ):
            image_output.read_response_body(response)

        self.assertEqual(response.read_sizes, [5, 3])

    def test_response_reader_rejects_non_bytes_chunk(self):
        response = ChunkedResponse(["not bytes"])
        with self.assertRaisesRegex(
            image_output.ImageOutputError,
            "body must be bytes",
        ):
            image_output.read_response_body(response)

    def test_response_reader_classifies_transport_failures_as_retryable(self):
        for error in (OSError("socket reset"), http.client.IncompleteRead(b"x", 2)):
            with self.subTest(error=type(error).__name__):
                response = mock.Mock()
                response.read.side_effect = error
                with self.assertRaises(image_output.ImageStreamError):
                    image_output.read_response_body(response)

    def test_response_reader_rejects_stream_that_never_reaches_eof(self):
        response = EndlessTinyResponse()
        with mock.patch.object(image_output, "MAX_PROVIDER_RESPONSE_BYTES", 100), \
             mock.patch.object(
                 image_output,
                 "MAX_PROVIDER_RESPONSE_READS",
                 3,
                 create=True,
             ), self.assertRaisesRegex(
                 image_output.ImageOutputError,
                 "too many chunks",
             ):
            image_output.read_response_body(response)

        self.assertEqual(response.reads, 3)

    def test_bomb_header_is_controlled_for_all_three_providers(self):
        bomb = png_header(17_920, 10_080)
        cases = (
            (
                gen_slide_openai,
                [JsonResponse({
                    "data": [{
                        "b64_json": base64.b64encode(bomb).decode("ascii")
                    }]
                })],
            ),
            (
                gen_slide_gemini,
                [JsonResponse({
                    "candidates": [{
                        "content": {"parts": [{
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(bomb).decode("ascii"),
                            }
                        }]}
                    }]
                })],
            ),
            (
                gen_slide_doubao,
                [
                    JsonResponse({"data": [{"url": "https://cdn.invalid/bomb"}]}),
                    StreamResponse(bomb),
                ],
            ),
        )
        for provider, responses in cases:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(provider, "_load_api_key", return_value="key"), \
                 mock.patch.object(
                     provider.urllib.request,
                     "urlopen",
                     side_effect=responses,
                 ), redirect_stdout(io.StringIO()):
                target = Path(temp_dir) / "slide.png"
                try:
                    result = provider.gen("prompt", str(target), retries=0)
                except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                    self.fail(f"Pillow bomb escaped {provider.__name__}: {error}")
                self.assertFalse(result)
                self.assertFalse(target.exists())
                self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_provider_success_bodies_use_bounded_max_plus_one_read(self):
        providers = (gen_slide_openai, gen_slide_gemini, gen_slide_doubao)
        for provider in providers:
            with self.subTest(provider=provider.__name__), \
                 tempfile.TemporaryDirectory() as temp_dir, \
                 mock.patch.object(image_output, "MAX_PROVIDER_RESPONSE_BYTES", 16, create=True), \
                 mock.patch.object(provider, "_load_api_key", return_value="key"), \
                 redirect_stdout(io.StringIO()):
                response = RecordingOversizedResponse()
                with mock.patch.object(
                    provider.urllib.request,
                    "urlopen",
                    return_value=response,
                ):
                    target = Path(temp_dir) / "slide.png"
                    self.assertFalse(
                        provider.gen(
                            "prompt",
                            str(target),
                            retries=0,
                        )
                    )
                self.assertEqual(response.read_sizes, [17])
                self.assertFalse(target.exists())
                self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_warning_band_is_rejected_without_leaking_pillow_warning(self):
        warning_header = png_header(12_000, 8_000)
        response = JsonResponse({
            "data": [{
                "b64_json": base64.b64encode(warning_header).decode("ascii")
            }]
        })
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(gen_slide_openai, "_load_api_key", return_value="key"), \
             mock.patch.object(
                 gen_slide_openai.urllib.request,
                 "urlopen",
                 return_value=response,
             ), warnings.catch_warnings(record=True) as caught, \
             redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            target = Path(temp_dir) / "slide.png"
            self.assertFalse(
                gen_slide_openai.gen("prompt", str(target), retries=0)
            )
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])
        self.assertFalse(
            any(isinstance(item.message, Image.DecompressionBombWarning) for item in caught)
        )


class LocalResourceBoundaryTests(unittest.TestCase):
    def test_local_byte_limit_is_controlled_by_export_and_prepare(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "oversized.png"
            with source.open("wb") as stream:
                stream.truncate(image_output.MAX_IMAGE_BYTES + 1)
            prefix = root / "deck"
            prepared = root / "prepared.png"

            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    export_images.main([str(prefix), str(source)]),
                    1,
                )
            with redirect_stdout(io.StringIO()):
                self.assertFalse(
                    prepare_editable_input.prepare(str(source), str(prepared))
                )

            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())
            self.assertFalse(prepared.exists())
            self.assertEqual(list(root.glob(".*.tmp*")), [])

    def test_bomb_header_is_controlled_by_export_and_prepare(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "bomb.png"
            source.write_bytes(png_header(17_920, 10_080))
            prefix = root / "deck"
            prepared = root / "prepared.png"
            stderr = io.StringIO()
            stdout = io.StringIO()

            try:
                with redirect_stderr(stderr):
                    export_result = export_images.main([str(prefix), str(source)])
                with redirect_stdout(stdout):
                    prepare_result = prepare_editable_input.prepare(
                        str(source),
                        str(prepared),
                    )
            except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                self.fail(f"Pillow bomb escaped local public boundary: {error}")

            self.assertEqual(export_result, 1)
            self.assertFalse(prepare_result)
            self.assertNotIn("Traceback", stderr.getvalue() + stdout.getvalue())
            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())
            self.assertFalse(prepared.exists())
            self.assertEqual(list(root.glob(".*.tmp*")), [])


if __name__ == "__main__":
    unittest.main()
