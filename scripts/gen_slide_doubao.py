#!/usr/bin/env python3
"""Volcano Ark Doubao Seedream image generation (explicit fallback engine)."""

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
    MAX_IMAGE_BYTES,
    ImageOutputError,
    ImageStreamError,
    output_format,
    preflight_output,
    publish_stream,
    read_response_body,
    resolve_output_path,
    validate_retries,
)
from output_lock import OutputLockError, output_lock
from provider_credentials import APIKeyError, load_api_key, validate_api_key
from retry_delay import retry_delay

SECRET_PATH = Path("~/.secrets/doubao_api_key").expanduser()
URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MODEL = "doubao-seedream-5-0-260128"
SIZE = "2560x1440"
DOWNLOAD_TIMEOUT = 120
KEY_LIKE_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+")
TRANSPORT_ERRORS = (
    ImageStreamError,
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
)


def _load_api_key() -> str:
    return load_api_key("DOUBAO_API_KEY", SECRET_PATH)


def _redact(message: object, key: str) -> str:
    text = str(message)
    if key:
        text = text.replace(key, "[REDACTED]")
    return KEY_LIKE_PATTERN.sub("[REDACTED]", text)[:300]


def _http_error_message(error: urllib.error.HTTPError, key: str) -> str:
    if error.code in (401, 403):
        return "authentication failed; check DOUBAO_API_KEY or ~/.secrets/doubao_api_key"
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


def _extract_image_url(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ImageOutputError("response envelope must be an object")
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ImageOutputError("response data must contain an image object")
    image_url = data[0].get("url")
    if not isinstance(image_url, str) or not image_url:
        raise ImageOutputError("response image URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(image_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ImageOutputError("response image URL must use HTTPS")
    return image_url


def _retryable_http(error: urllib.error.HTTPError) -> bool:
    return error.code == 429 or error.code >= 500


def _download_image(
    image_url: str,
    target: Path,
    retries: int,
    overwrite: bool,
    key: str,
) -> Optional[int]:
    for attempt in range(retries + 1):
        request = urllib.request.Request(image_url)
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                return publish_stream(
                    response,
                    target,
                    overwrite=overwrite,
                    max_bytes=MAX_IMAGE_BYTES,
                )
        except urllib.error.HTTPError as error:
            print(
                f"  HTTP {error.code}: image download failed: "
                f"{_http_error_message(error, key)} (attempt {attempt + 1})"
            )
            if _retryable_http(error) and attempt < retries:
                time.sleep(retry_delay(attempt, error.headers))
                continue
            return None
        except (ImageStreamError,) + TRANSPORT_ERRORS as error:
            print(f"  ERR: {_redact(error, key)} (attempt {attempt + 1})")
            if attempt < retries:
                time.sleep(retry_delay(attempt))
                continue
            return None
        except (ImageOutputError, TypeError, ValueError) as error:
            print(f"  ERR: invalid downloaded image or output failure: {_redact(error, key)}")
            return None
    return None


def _gen_owned(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    """Generate one strict 16:9 image. Return True only after atomic publication."""
    try:
        retries = validate_retries(retries)
        requested_format = output_format(out_path)
        target = preflight_output(out_path, overwrite=overwrite)
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False

    key = _load_api_key()
    if not key:
        print(
            "  ERR: Doubao API key not found. Set DOUBAO_API_KEY or create "
            "~/.secrets/doubao_api_key (see README.md)"
        )
        return False
    try:
        key = validate_api_key(key)
    except APIKeyError as error:
        print(f"  ERR: Doubao API key is invalid: {error}")
        return False

    body = json.dumps({
        "model": os.environ.get("DOUBAO_IMAGE_MODEL", MODEL),
        "prompt": prompt,
        "size": SIZE,
        "response_format": "url",
        "output_format": requested_format,
        "watermark": False,
        "sequential_image_generation": "disabled",
    }).encode("utf-8")

    image_url = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_response = read_response_body(response)
        except urllib.error.HTTPError as error:
            print(
                f"  HTTP {error.code}: {_http_error_message(error, key)} "
                f"(attempt {attempt + 1})"
            )
            if _retryable_http(error) and attempt < retries:
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
            payload = json.loads(raw_response)
            image_url = _extract_image_url(payload)
        except (TypeError, ValueError, UnicodeError) as error:
            print(f"  ERR: invalid image response: {_redact(error, key)}")
            return False
        break

    if image_url is None:
        return False

    byte_count = _download_image(image_url, target, retries, overwrite, key)
    if byte_count is None:
        return False
    print(f"  OK: {_redact(target, key)} ({byte_count // 1024}KB)")
    return True


def gen(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    try:
        target = str(resolve_output_path(out_path))
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False
    try:
        with output_lock(target):
            return _gen_owned(prompt, target, retries, overwrite)
    except OutputLockError as error:
        print(f"  ERR: {error}")
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="Output .jpg, .jpeg, .png, or .webp")
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
