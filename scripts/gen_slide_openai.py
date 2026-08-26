#!/usr/bin/env python3
"""Generate one slide image with OpenAI GPT Image (default engine)."""

import base64
import binascii
import json
import os
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


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        message = payload.get("error", {}).get("message")
        if message:
            return str(message)[:300]
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return str(error.reason or error)[:300]


def gen(prompt: str, out_path: str, retries: int = 2) -> bool:
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
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            print(f"  HTTP {error.code}: {_error_message(error)} (attempt {attempt + 1})")
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(2)
                continue
            return False
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  ERR: {error} (attempt {attempt + 1})")
            if attempt < retries:
                time.sleep(2)
                continue
            return False
        except (OSError, UnicodeError, json.JSONDecodeError):
            print("  ERR: invalid JSON response from OpenAI")
            return False

        try:
            encoded = result["data"][0]["b64_json"]
            image = base64.b64decode(encoded, validate=True)
            if not image:
                raise ValueError("empty image data")
            _atomic_write(image, out_path)
        except (KeyError, IndexError, TypeError, ValueError, binascii.Error, OSError) as error:
            print(f"  ERR: invalid image response or output failure: {error}")
            return False

        size_kb = len(image) // 1024
        print(f"  OK: {out_path} ({size_kb}KB, OpenAI {payload['model']})")
        return True

    return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        success = gen(sys.argv[2], sys.argv[1])
        sys.exit(0 if success else 1)
    print("用法: gen_slide_openai.py <输出路径.jpg|png|webp> <prompt>")
