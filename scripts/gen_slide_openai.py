#!/usr/bin/env python3
"""Generate one slide image with OpenAI GPT Image (default engine)."""

import base64
import binascii
import http.client
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.openai.com/v1/images/generations"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "medium"
SECRET_PATH = Path("~/.secrets/openai_api_key").expanduser()
OUTPUT_FORMATS = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}
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
    extension = Path(out_path).suffix.lower()
    try:
        return OUTPUT_FORMATS[extension]
    except KeyError as exc:
        supported = ", ".join(sorted(OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output extension '{extension or '<none>'}'. Use: {supported}"
        ) from exc


def _atomic_write(data: bytes, out_path: str) -> None:
    target = Path(out_path)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, target)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _redact(message: object, key: str) -> str:
    text = str(message)
    if key:
        text = text.replace(key, "[REDACTED]")
    return KEY_LIKE_PATTERN.sub("[REDACTED]", text)[:300]


def _error_message(error: urllib.error.HTTPError, key: str) -> str:
    try:
        raw_body = error.read()
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


def _preflight_output(out_path: str, key: str) -> bool:
    target = Path(out_path)
    probe_path = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_file():
            raise OSError("output path exists and is not a regular file")
        fd, probe_path = tempfile.mkstemp(prefix=".write-test.", dir=str(target.parent))
        os.close(fd)
        os.unlink(probe_path)
        probe_path = None
    except (OSError, ValueError) as error:
        if probe_path is not None:
            try:
                os.unlink(probe_path)
            except OSError:
                pass
        print(f"  ERR: output path is not writable: {_redact(error, key)}")
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
    image = base64.b64decode(encoded, validate=True)
    if not image:
        raise ValueError("empty image data")
    return image


def gen(prompt: str, out_path: str, retries: int = 2) -> bool:
    if retries < 0:
        print("  ERR: retries must be non-negative")
        return False

    try:
        output_format = _output_format(out_path)
    except ValueError as error:
        print(f"  ERR: {error}")
        return False

    key = _load_api_key()
    if not key:
        print(
            "  ERR: OpenAI API key not found. Set OPENAI_API_KEY or create "
            "~/.secrets/openai_api_key"
        )
        return False

    if not _preflight_output(out_path, key):
        return False

    payload = {
        "model": os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL),
        "prompt": prompt,
        "size": os.environ.get("OPENAI_IMAGE_SIZE", DEFAULT_SIZE),
        "quality": os.environ.get("OPENAI_IMAGE_QUALITY", DEFAULT_QUALITY),
        "output_format": output_format,
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
                raw_response = response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                message = "authentication failed; check OPENAI_API_KEY"
            else:
                message = _error_message(error, key)
            print(f"  HTTP {error.code}: {message} (attempt {attempt + 1})")
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(2)
                continue
            return False
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as error:
            print(f"  ERR: {_redact(error, key)} (attempt {attempt + 1})")
            if attempt < retries:
                time.sleep(2)
                continue
            return False

        try:
            result = json.loads(raw_response)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            print("  ERR: invalid JSON response from OpenAI")
            return False

        try:
            image = _decode_image_response(result)
            _atomic_write(image, out_path)
        except (KeyError, IndexError, TypeError, ValueError, binascii.Error, OSError) as error:
            print(
                "  ERR: invalid image response or output failure: "
                f"{_redact(error, key)}"
            )
            return False

        size_kb = len(image) // 1024
        safe_path = _redact(out_path, key)
        safe_model = _redact(payload["model"], key)
        print(f"  OK: {safe_path} ({size_kb}KB, OpenAI {safe_model})")
        return True

    return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        success = gen(sys.argv[2], sys.argv[1])
        sys.exit(0 if success else 1)
    print("用法: gen_slide_openai.py <输出路径.jpg|png|webp> <prompt>")
