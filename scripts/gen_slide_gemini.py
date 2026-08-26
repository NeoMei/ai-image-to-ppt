#!/usr/bin/env python3
"""Gemini image generation (explicit fallback engine)."""

import argparse
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from image_output import (
    ImageOutputError,
    ImageStreamError,
    decode_base64,
    parse_json_response,
    prepare_target,
    prepared_lock_target,
    preflight_output,
    publish_bytes,
    read_response_body,
    resolve_output_path,
    validate_retries,
)
from output_lock import OutputLockError, output_lock
from provider_credentials import APIKeyError, load_api_key, validate_api_key
from retry_delay import retry_delay

SECRET_PATH = Path("~/.secrets/gemini_api_key").expanduser()
DEFAULT_MODEL = "gemini-3.1-flash-image"
URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "{}:generateContent"
)
KEY_LIKE_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+")
TRANSPORT_ERRORS = (
    ImageStreamError,
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
)


def _load_api_key() -> str:
    return load_api_key("GEMINI_API_KEY", SECRET_PATH)


def _output_config(out_path: str):
    extension = Path(out_path).suffix.lower()
    image_config = {"aspectRatio": "16:9", "imageSize": "2K"}
    if extension == ".png":
        return "image/png", image_config
    if extension in {".jpg", ".jpeg"}:
        image_config["mimeType"] = "IMAGE_JPEG"
        return "image/jpeg", image_config
    raise ImageOutputError(
        f"Unsupported Gemini output extension '{extension or '<none>'}'. "
        "Use: .jpeg, .jpg, .png"
    )


def _redact(message: object, key: str) -> str:
    text = str(message)
    if key:
        text = text.replace(key, "[REDACTED]")
    return KEY_LIKE_PATTERN.sub("[REDACTED]", text)[:300]


def _http_error_message(error: urllib.error.HTTPError, key: str) -> str:
    if error.code in (401, 403):
        return "authentication failed; check GEMINI_API_KEY or ~/.secrets/gemini_api_key"
    try:
        raw_body = read_response_body(error)
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        payload = json.loads(raw_body)
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = payload["error"].get("message")
            if isinstance(message, (str, int, float, bool)) and message:
                return _redact(message, key)
    except Exception:
        pass
    return _redact(error.reason or "request failed", key)


def _extract_image(payload: object, target: Path, expected_mime: str) -> bytes:
    if not isinstance(payload, dict):
        raise ImageOutputError("response envelope must be an object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ImageOutputError("response candidates must be a non-empty list")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ImageOutputError("response candidate must be an object")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ImageOutputError("response content must be an object")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ImageOutputError("response parts must be a list")

    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if not isinstance(inline, dict) or not inline.get("data"):
            continue
        declared_mime = inline.get("mimeType") or inline.get("mime_type")
        if not isinstance(declared_mime, str) or not declared_mime:
            raise ImageOutputError("response MIME type must be non-empty text")
        if declared_mime.lower() != expected_mime:
            raise ImageOutputError(
                f"response MIME type {declared_mime} does not match {target.suffix.lower()}"
            )
        return decode_base64(inline.get("data"))
    raise ImageOutputError("response contains no image data")


def _gen_owned(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    """Generate one strict 16:9 image. Return True only after atomic publication."""
    try:
        retries = validate_retries(retries)
        expected_mime, image_config = _output_config(out_path)
        target = preflight_output(out_path, overwrite=overwrite)
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False

    key = _load_api_key()
    if not key:
        print(
            "  ERR: Gemini API key not found. Set GEMINI_API_KEY or create "
            "~/.secrets/gemini_api_key (see README.md)"
        )
        return False
    try:
        key = validate_api_key(key)
    except APIKeyError as error:
        print(f"  ERR: Gemini API key is invalid: {error}")
        return False

    model = os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    encoded_model = urllib.parse.quote(model, safe="")
    url = URL_TEMPLATE.format(encoded_model)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "responseFormat": {
                "image": image_config
            },
        },
    }).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw_response = read_response_body(response)
        except urllib.error.HTTPError as error:
            print(
                f"  HTTP {error.code}: {_http_error_message(error, key)} "
                f"(attempt {attempt + 1})"
            )
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(retry_delay(attempt, error.headers))
                continue
            return False
        except TRANSPORT_ERRORS as error:
            print(f"  ERR: {_redact(error, key)} (attempt {attempt + 1})")
            if attempt < retries:
                time.sleep(retry_delay(attempt))
                continue
            return False
        except ImageOutputError as error:
            print(f"  ERR: invalid provider response: {_redact(error, key)}")
            return False

        try:
            payload = parse_json_response(raw_response)
            image = _extract_image(payload, target, expected_mime)
            byte_count = publish_bytes(image, target, overwrite=overwrite)
        except (TypeError, ValueError, UnicodeError, OSError) as error:
            print(f"  ERR: invalid image response or output failure: {_redact(error, key)}")
            return False

        print(
            f"  OK: {_redact(target, key)} "
            f"({byte_count // 1024}KB, Gemini {_redact(model, key)})"
        )
        return True

    return False


def gen(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    try:
        retries = validate_retries(retries)
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False
    try:
        target = prepare_target(resolve_output_path(out_path))
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False
    try:
        with output_lock(prepared_lock_target(target)):
            return _gen_owned(prompt, target, retries, overwrite)
    except OutputLockError as error:
        print(f"  ERR: {error}")
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="Output .png (default), .jpg, or .jpeg")
    parser.add_argument("prompt", help="Slide image prompt")
    parser.add_argument(
        "--force", action="store_true", help="Atomically replace an existing output"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return 0 if gen(args.prompt, args.output_path, overwrite=args.force) else 1


if __name__ == "__main__":
    raise SystemExit(main())
