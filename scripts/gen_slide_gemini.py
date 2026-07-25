#!/usr/bin/env python3
"""Gemini nano banana 2 图像生成 (教科书米白风首选).

依赖:
  - ~/.secrets/gemini_api_key  (从 https://aistudio.google.com/apikey 获取)

用法:
  # 单张
  python3 gen_slide_gemini.py <输出路径.jpg> "<prompt>"
  # 脚本内调用
  from gen_slide_gemini import gen
  gen("<prompt>", "<输出路径.jpg>")
"""
import json, os, sys, time, base64, urllib.request, urllib.error

KEY = open(os.path.expanduser("~/.secrets/gemini_api_key")).read().strip()
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={KEY}"


def gen(prompt: str, out_path: str, retries: int = 2) -> bool:
    """生成单张图. 成功返回 True."""
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
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
            err = json.loads(e.read()).get("error", {}).get("message", str(e))
            print(f"  HTTP {e.code}: {err[:150]} (attempt {attempt+1})")
        except Exception as e:
            print(f"  ERR: {e} (attempt {attempt+1})")
        time.sleep(2)
    return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ok = gen(sys.argv[2], sys.argv[1])
        sys.exit(0 if ok else 1)
    print("用法: gen_slide_gemini.py <输出路径.jpg> <prompt>")
