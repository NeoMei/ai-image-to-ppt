#!/usr/bin/env python3
"""Generate one slide image with OpenAI GPT Image (default engine)."""

import argparse
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from image_output import (
    ImageOutputError,
    ImageStreamError,
    decode_base64,
    output_format,
    preflight_output,
    publish_bytes,
    read_response_body,
    validate_retries,
)
from output_lock import OutputLockError, output_lock
from retry_delay import retry_delay

API_URL = "https://api.openai.com/v1/images/generations"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "medium"
SECRET_PATH = Path("~/.secrets/openai_api_key").expanduser()
KEY_LIKE_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+")


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _output_format(out_path: str) -> str:
    return output_format(out_path)


def _redact(message: object, key: str) -> str:
    text = str(message)
    if key:
        text = text.replace(key, "[REDACTED]")
    return KEY_LIKE_PATTERN.sub("[REDACTED]", text)[:300]


def _error_message(error: urllib.error.HTTPError, key: str) -> str:
    try:
        raw_body = read_response_body(error)
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        if not isinstance(raw_body, str):
            raise TypeError("HTTP error body must be text or bytes")
        payload = json.loads(raw_body)
        if not isinstance(payload, dict):
            raise TypeError("HTTP error envelope must be an object")
        details = payload.get("error")
        if not isinstance(details, dict):
            raise TypeError("HTTP error details must be an object")
        message = details.get("message")
        if isinstance(message, (str, int, float, bool)) and message:
            return _redact(message, key)
    except Exception:
        pass
    return _redact(error.reason or "request failed", key)


def _preflight_output(out_path: str, key: str, overwrite: bool = False) -> bool:
    try:
        preflight_output(out_path, overwrite=overwrite)
    except ImageOutputError as error:
        print(f"  ERR: {_redact(error, key)}")
        return False
    return True


def _decode_image_response(result: object) -> bytes:
    if not isinstance(result, dict):
        raise ValueError("response envelope must be an object")
    data = result.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("response data must be a non-empty list")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("response image entry must be an object")
    encoded = first.get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("response b64_json must be a non-empty string")
    return decode_base64(encoded)


def _gen_owned(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    try:
        retries = validate_retries(retries)
        output_format_value = _output_format(out_path)
        target = preflight_output(out_path, overwrite=overwrite)
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False

    key = _load_api_key()
    if not key:
        print(
            "  ERR: OpenAI API key not found. Set OPENAI_API_KEY or create "
            "~/.secrets/openai_api_key"
        )
        return False

    payload = {
        "model": os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL),
        "prompt": prompt,
        "size": os.environ.get("OPENAI_IMAGE_SIZE", DEFAULT_SIZE),
        "quality": os.environ.get("OPENAI_IMAGE_QUALITY", DEFAULT_QUALITY),
        "output_format": output_format_value,
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            API_URL,
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
            if error.code in (401, 403):
                message = (
                    "authentication failed; check OPENAI_API_KEY; provider says: "
                    f"{_error_message(error, key)}"
                )
            else:
                message = _error_message(error, key)
            print(f"  HTTP {error.code}: {message} (attempt {attempt + 1})")
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(retry_delay(attempt, error.headers))
                continue
            return False
        except (
            ImageStreamError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as error:
            print(f"  ERR: {_redact(error, key)} (attempt {attempt + 1})")
            if attempt < retries:
                time.sleep(retry_delay(attempt))
                continue
            return False
        except ImageOutputError as error:
            print(f"  ERR: invalid provider response: {_redact(error, key)}")
            return False

        try:
            result = json.loads(raw_response)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            print("  ERR: invalid JSON response from OpenAI")
            return False

        try:
            image = _decode_image_response(result)
            byte_count = publish_bytes(image, target, overwrite=overwrite)
        except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
            print(
                "  ERR: invalid image response or output failure: "
                f"{_redact(error, key)}"
            )
            return False

        size_kb = byte_count // 1024
        safe_path = _redact(out_path, key)
        safe_model = _redact(payload["model"], key)
        print(f"  OK: {safe_path} ({size_kb}KB, OpenAI {safe_model})")
        return True

    return False


def gen(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    try:
        with output_lock(out_path):
            return _gen_owned(prompt, out_path, retries, overwrite)
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
