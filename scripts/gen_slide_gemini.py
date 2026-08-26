#!/usr/bin/env python3
"""Gemini nano banana 2 图像生成（显式备用引擎）.

依赖:
  - ~/.secrets/gemini_api_key  (从 https://aistudio.google.com/apikey 获取)

用法:
  # 单张
  python3 gen_slide_gemini.py <输出路径.jpg> "<prompt>"
  # 脚本内调用
  from gen_slide_gemini import gen
  gen("<prompt>", "<输出路径.jpg>")
"""
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SECRET_PATH = Path("~/.secrets/gemini_api_key").expanduser()
URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.1-flash-image-preview:generateContent?key={}"
)
KEY_LIKE_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+")


def _load_api_key() -> str:
    try:
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _redact(message: object, key: str) -> str:
    text = str(message)
    if key:
        text = text.replace(key, "[REDACTED]")
    return KEY_LIKE_PATTERN.sub("[REDACTED]", text)[:200]


def _http_error_message(error: urllib.error.HTTPError, key: str) -> str:
    if error.code in (401, 403):
        return "authentication failed; check ~/.secrets/gemini_api_key"
    try:
        raw_body = error.read()
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


def gen(prompt: str, out_path: str, retries: int = 2) -> bool:
    """生成单张图. 成功返回 True."""
    if retries < 0:
        print("  ERR: retries must be non-negative")
        return False

    key = _load_api_key()
    if not key:
        print(
            "  ERR: Gemini API key not found. Create "
            "~/.secrets/gemini_api_key (see README.md)"
        )
        return False

    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode()
    req = urllib.request.Request(
        URL_TEMPLATE.format(key),
        data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            parts = d.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for p in parts:
                data = p.get("inlineData") or p.get("inline_data")
                if data and data.get("data"):
                    b = base64.b64decode(data["data"])
                    with open(out_path, "wb") as f:
                        f.write(b)
                    print(f"  OK: {out_path} ({len(b)//1024}KB)")
                    return True
            print(f"  无图片数据 (attempt {attempt+1})")
        except urllib.error.HTTPError as e:
            print(
                f"  HTTP {e.code}: {_http_error_message(e, key)} "
                f"(attempt {attempt+1})"
            )
        except Exception as e:
            print(f"  ERR: {_redact(e, key)} (attempt {attempt+1})")
        if attempt < retries:
            time.sleep(2)
    return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ok = gen(sys.argv[2], sys.argv[1])
        sys.exit(0 if ok else 1)
    print("用法: gen_slide_gemini.py <输出路径.jpg> <prompt>")
