#!/usr/bin/env python3
"""方舟豆包 Seedream 图像生成（显式备用引擎，OpenAI 兼容接口）.

依赖:
  - ~/.secrets/doubao_api_key  (从 https://console.volcengine.com/ark 获取)

可用模型:
  - doubao-seedream-4-0-250828  (基础版, 便宜)
  - doubao-seedream-4-5-251128  (增强版)
  - doubao-seedream-5-0-260128  (最新 5.0, 默认)

用法:
  # 单张
  python3 gen_slide_doubao.py <输出路径.jpg> "<prompt>"
  # 脚本内调用
  from gen_slide_doubao import gen
  gen("<prompt>", "<输出路径.jpg>")
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SECRET_PATH = Path("~/.secrets/doubao_api_key").expanduser()
URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MODEL = "doubao-seedream-5-0-260128"
SIZE = "2560x1440"  # 16:9
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
        return "authentication failed; check ~/.secrets/doubao_api_key"
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
            "  ERR: Doubao API key not found. Create "
            "~/.secrets/doubao_api_key (see README.md)"
        )
        return False

    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "response_format": "url",
        "watermark": False,
        "sequential_image_generation": "disabled",
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            img_url = d["data"][0]["url"]
            urllib.request.urlretrieve(img_url, out_path)
            size_kb = os.path.getsize(out_path) // 1024
            print(f"  OK: {out_path} ({size_kb}KB)")
            return True
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
    print("用法: gen_slide_doubao.py <输出路径.jpg> <prompt>")
