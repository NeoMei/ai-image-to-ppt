# ai-image-to-ppt

Generate 16:9 educational textbook-style slide images from any topic using AI image generation models (Gemini **nano banana** primary, Volcano **doubao-seedream** fallback), then package into PDF + PPTX.

用 AI 图像生成模型把主题/大纲批量变成教科书风幻灯片图片，再打包成 PDF / PPTX。

## Features

- 🎨 教科书米白风 16:9 幻灯片批量生成（中英混排友好）
- 🔁 双引擎可互换：Gemini nano banana（免费额度）/ 方舟 doubao-seedream（付费备用）
- 👁 视觉自检脚本（Gemini 视觉模型检查错字、数量错误）
- 📦 一键导出 PDF + PPTX（自动归一化到 1920×1080）

## Setup

Two API keys (any one suffices; both recommended for redundancy):

```bash
# Gemini (https://aistudio.google.com/apikey)
echo "YOUR_GEMINI_KEY" > ~/.secrets/gemini_api_key

# 方舟 doubao (https://console.volcengine.com/ark)
echo "YOUR_DOUBAO_KEY" > ~/.secrets/doubao_api_key

chmod 600 ~/.secrets/*_api_key
pip install Pillow python-pptx
```

> Keys are read from `~/.secrets/` at runtime only. Never commit real keys.

## Quick Start

```bash
# 单张生成
python3 scripts/gen_slide_gemini.py out/slide_01.jpg "<prompt>"
# 或备用引擎
python3 scripts/gen_slide_doubao.py out/slide_01.jpg "<prompt>"

# 视觉自检
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"

# 打包导出
python3 scripts/export_images.py "deck_name" out/*.jpg
```

脚本内调用（批量并发生成见 [SKILL.md](SKILL.md)）：

```python
import sys
sys.path.insert(0, "scripts")
from gen_slide_gemini import gen  # or gen_slide_doubao

gen("<detailed prompt>", "out/slide_01.jpg")
```

## Scripts

| Script | Engine | API | Cost |
|---|---|---|---|
| `gen_slide_gemini.py` | Gemini nano banana | `gemini-3.1-flash-image-preview` | 免费额度 |
| `gen_slide_doubao.py` | 方舟 doubao-seedream | OpenAI 兼容 | 按量付费 |
| `vision_check_gemini.py` | Gemini 2.0 Flash 视觉理解 | `gemini-2.0-flash` | 免费 |
| `export_images.py` | Pillow + python-pptx | 本地 | 免费 |

Both generators share the identical interface `gen(prompt, out_path) -> bool`, swappable.

## As an Agent Skill

[SKILL.md](SKILL.md) is written in the agent-skill format: it contains the full workflow
(style template, batch concurrency pattern, quantity-constraining tricks, pitfalls table).
Drop the folder into your agent's skills directory (e.g. `~/.agents/skills/`) to let an
AI agent generate decks autonomously.

## License

[MIT](LICENSE)
