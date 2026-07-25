#!/usr/bin/env python3
"""Gemini 视觉自检 (用 gemini-2.0-flash 分析图片内容).

依赖:
  - ~/.secrets/gemini_api_key  (跟 gen_slide_gemini.py 共用)

用法:
  python3 vision_check_gemini.py <图片路径> [问题]
"""
import sys, base64, json, os, urllib.request

KEY = open(os.path.expanduser("~/.secrets/gemini_api_key")).read().strip()
MODEL = "gemini-2.0-flash"


def check(img_path, question=None):
    mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({
        "contents": [{"parts": [
            {"text": question or "详细描述这张图片: 配色、布局、文字内容、任何渲染问题。"},
            {"inline_data": {"mime_type": mime, "data": b64}}
        ]}]
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: vision_check_gemini.py <图片路径> [问题]")
        sys.exit(1)
    print(check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
