#!/usr/bin/env python3
"""方舟豆包 Seedream 图像生成 (Gemini 配额耗尽时的备选, OpenAI 兼容接口).

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
import json, os, sys, time, urllib.request, urllib.error

KEY = open(os.path.expanduser("~/.secrets/doubao_api_key")).read().strip()
URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MODEL = "doubao-seedream-5-0-260128"
SIZE = "2560x1440"  # 16:9


def gen(prompt: str, out_path: str, retries: int = 2) -> bool:
    """生成单张图. 成功返回 True. 接口跟 gen_slide_gemini.gen 一致, 可直接替换."""
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "size": SIZE,
        "response_format": "url",
        "watermark": False,
        "sequential_image_generation": "disabled",
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}",
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
            err = e.read().decode()[:200]
            print(f"  HTTP {e.code}: {err} (attempt {attempt+1})")
        except Exception as e:
            print(f"  ERR: {e} (attempt {attempt+1})")
        time.sleep(2)
    return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ok = gen(sys.argv[2], sys.argv[1])
        sys.exit(0 if ok else 1)
    print("用法: gen_slide_doubao.py <输出路径.jpg> <prompt>")
