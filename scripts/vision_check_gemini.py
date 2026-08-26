#!/usr/bin/env python3
"""Inspect a local slide image with Gemini Vision."""

import argparse
import base64
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from retry_delay import retry_delay


DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_QUESTION = "详细描述这张图片: 配色、布局、文字内容、任何渲染问题。"
SECRET_PATH = Path("~/.secrets/gemini_api_key").expanduser()
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
REQUEST_TIMEOUT = 60
# Gemini text inspections should stay compact. Keep response and diagnostic
# bodies bounded independently so an upstream service cannot exhaust memory.
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ERROR_BODY_BYTES = 64 * 1024
RESPONSE_READ_CHUNK_BYTES = 64 * 1024
# Inline request bodies have a 20 MB limit. Four-thirds Base64 expansion and
# JSON overhead make 14 MiB a safe raw-image ceiling.
MAX_IMAGE_BYTES = 14 * 1024 * 1024
MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
GOOGLE_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{16,}")


class VisionCheckError(RuntimeError):
    """An expected, user-facing vision-check failure."""


def _load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _redact(value: object, key: str) -> str:
    text = str(value)
    if key:
        text = text.replace(key, "[REDACTED]")
    return GOOGLE_KEY_PATTERN.sub("[REDACTED]", text)[:300]


def _read_bounded_body(stream: object, limit: int, label: str) -> bytes:
    """Read a response stream incrementally, rejecting anything over ``limit``."""
    chunks = []
    total = 0
    while True:
        read_size = min(RESPONSE_READ_CHUNK_BYTES, limit - total + 1)
        try:
            chunk = stream.read(read_size)
        except TypeError as error:
            raise VisionCheckError(
                f"{label} reader does not support bounded reads"
            ) from error
        if not isinstance(chunk, (bytes, bytearray)):
            raise VisionCheckError(f"{label} reader returned non-bytes data")
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise VisionCheckError(f"{label} exceeds {limit} bytes")
        chunks.append(bytes(chunk))


def _read_image(img_path: str) -> tuple[bytes, str]:
    path = Path(img_path)
    try:
        if not path.is_file():
            raise VisionCheckError(f"input image is not a regular file: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise VisionCheckError(f"input image is empty: {path}")
        if size > MAX_IMAGE_BYTES:
            raise VisionCheckError(
                f"input image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit: {path}"
            )
        with path.open("rb") as image_file:
            data = image_file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise VisionCheckError(
                f"input image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit: {path}"
            )
    except VisionCheckError:
        raise
    except OSError as error:
        raise VisionCheckError(f"cannot read input image: {error}") from error

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format
                width, height = image.size
                image.verify()
        if width <= 0 or height <= 0:
            raise VisionCheckError("input image has invalid dimensions")
        try:
            mime = MIME_TYPES[image_format]
        except KeyError as error:
            supported = ", ".join(MIME_TYPES)
            raise VisionCheckError(
                f"unsupported input image format {image_format!r}; use {supported}"
            ) from error
    except VisionCheckError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise VisionCheckError(f"input is not a valid image: {error}") from error
    except Image.DecompressionBombError as error:
        raise VisionCheckError(f"input image dimensions are too large: {error}") from error
    except Image.DecompressionBombWarning as error:
        raise VisionCheckError(f"input image dimensions are too large: {error}") from error

    return data, mime


def _http_error_message(error: urllib.error.HTTPError, key: str) -> str:
    try:
        raw = _read_bounded_body(error, MAX_ERROR_BODY_BYTES, "Gemini error body")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise TypeError("error body is not text")
        payload = json.loads(raw)
        details = payload.get("error") if isinstance(payload, dict) else None
        message = details.get("message") if isinstance(details, dict) else None
        if isinstance(message, str) and message.strip():
            return _redact(message.strip(), key)
    except VisionCheckError:
        raise
    except Exception:
        pass
    return _redact(error.reason or "request failed", key)


def _parse_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise VisionCheckError("Gemini response envelope is not an object")

    feedback = payload.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise VisionCheckError("Gemini blocked the request via promptFeedback")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise VisionCheckError("Gemini response has no candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    raise VisionCheckError("Gemini response contains no text")


def check(img_path: str, question: str = None, retries: int = 2) -> str:
    """Return Gemini's textual inspection of a PNG, JPEG, or WebP image."""
    if retries < 0:
        raise VisionCheckError("retries must be non-negative")

    image_data, mime = _read_image(img_path)
    key = _load_api_key()
    if not key:
        raise VisionCheckError(
            "Gemini API key not found; set GEMINI_API_KEY or create "
            "~/.secrets/gemini_api_key"
        )

    model = os.environ.get("GEMINI_VISION_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    encoded_model = urllib.parse.quote(model, safe="")
    url = f"{API_ROOT}/{encoded_model}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [
            {"text": question or DEFAULT_QUESTION},
            {"inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(image_data).decode("ascii"),
            }},
        ]}],
    }).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                raw_response = _read_bounded_body(
                    response, MAX_RESPONSE_BYTES, "Gemini response body"
                )
        except urllib.error.HTTPError as error:
            try:
                message = _http_error_message(error, key)
            except VisionCheckError as body_error:
                raise VisionCheckError(
                    f"Gemini HTTP {error.code}: {body_error}"
                ) from error
            failure = VisionCheckError(f"Gemini HTTP {error.code}: {message}")
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(retry_delay(attempt, error.headers))
                continue
            raise failure from error
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as error:
            failure = VisionCheckError(
                f"Gemini network request failed: {_redact(error, key)}"
            )
            if attempt < retries:
                time.sleep(retry_delay(attempt))
                continue
            raise failure from error

        try:
            payload = json.loads(raw_response)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise VisionCheckError("Gemini returned invalid JSON") from error
        return _parse_text(payload)

    raise VisionCheckError("Gemini request failed")


def _non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use Gemini to inspect a local PNG, JPEG, or WebP slide image."
    )
    parser.add_argument("image", help="path to the local image")
    parser.add_argument("question", nargs="*", help="optional inspection question")
    parser.add_argument(
        "--retries",
        type=_non_negative,
        default=2,
        help="transient retry count (default: 2)",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    question = " ".join(args.question) or None
    try:
        result = check(args.image, question, retries=args.retries)
    except VisionCheckError as error:
        print(f"ERR: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
