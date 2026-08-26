# ai-image-to-ppt

Generate 16:9 educational textbook-style slide images from any topic using OpenAI **GPT Image 2** by default, with Gemini **nano banana** and Volcano **doubao-seedream** as explicit fallbacks, then package them into PDF + PPTX.

用 AI 图像生成模型把主题/大纲批量变成教科书风幻灯片图片，再打包成 PDF / PPTX。

## Features

- 🎨 教科书米白风 16:9 幻灯片批量生成（中英混排友好）
- 🤖 默认使用 OpenAI GPT Image 2；Gemini nano banana / 方舟 doubao-seedream 可显式选作备用引擎
- 👁 视觉自检脚本（Gemini 视觉模型检查错字、数量错误）
- 📦 一键导出 PDF + PPTX（自动归一化到 1920×1080）
- 🧩 可选生成精确 1280×720 PNG，作为单页 `image-to-editable-pptx` 转换器的输入

## Setup

One OpenAI API key is sufficient for default generation:

```bash
# Standard OpenAI credential (preferred)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

# Or use this project's secret-file convention
mkdir -p ~/.secrets
printf '%s\n' "YOUR_OPENAI_KEY" > ~/.secrets/openai_api_key
chmod 600 ~/.secrets/openai_api_key

# Gemini fallback and visual self-check (only if used)
printf '%s\n' "YOUR_GEMINI_KEY" > ~/.secrets/gemini_api_key
chmod 600 ~/.secrets/gemini_api_key

# Doubao fallback (only if used)
printf '%s\n' "YOUR_DOUBAO_KEY" > ~/.secrets/doubao_api_key
chmod 600 ~/.secrets/doubao_api_key

pip install 'Pillow>=9.1' python-pptx
```

Gemini and Doubao credentials are only needed when selecting those fallback engines. A Gemini key is also required for the optional visual self-check. The commands above restore the exact secret-file paths expected by the legacy scripts. Never commit real keys.

## Quick Start

```bash
mkdir -p out out/editable

# Default: OpenAI GPT Image 2, 2048x1152, medium
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>"

# Explicit fallback engines
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>" --engine gemini
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>" --engine doubao

# Optional handoff: prepare the exact single-slide input required by
# image-to-editable-pptx. This PNG is converter input, not an editable PPTX.
python3 scripts/prepare_editable_input.py \
  out/slide_01.jpg \
  out/editable/slide_01.png

# 视觉自检
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"

# 打包导出
python3 scripts/export_images.py "deck_name" out/*.jpg
```

脚本内调用（批量并发生成见 [SKILL.md](SKILL.md)）：

```python
import sys
sys.path.insert(0, "scripts")
from gen_slide import gen

gen("<detailed prompt>", "out/slide_01.jpg")
gen("<detailed prompt>", "out/slide_01.jpg", engine="gemini")
```

## Scripts

| Script | Role / Engine | API / Output |
|---|---|---|
| `gen_slide.py` | Default provider router | OpenAI by default; `--engine gemini\|doubao` for explicit fallback |
| `gen_slide_openai.py` | OpenAI GPT Image 2 | `gpt-image-2`; defaults to 2048×1152, medium quality |
| `gen_slide_gemini.py` | Gemini nano banana | `gemini-3.1-flash-image-preview` |
| `gen_slide_doubao.py` | 方舟 doubao-seedream | OpenAI-compatible API |
| `prepare_editable_input.py` | Editable-converter handoff | Deterministic 1280x720 PNG |
| `vision_check_gemini.py` | Gemini visual self-check | `gemini-2.0-flash` |
| `export_images.py` | Local export | PDF + PPTX |

The router does not automatically switch providers: choose fallback engines explicitly. Provider pricing can change; consult the [OpenAI API pricing documentation](https://developers.openai.com/api/docs/pricing) instead of relying on a fixed per-image estimate.

OpenAI output extensions `.jpg`, `.jpeg`, `.png`, and `.webp` select matching API formats. The legacy Gemini and Doubao engines keep provider-returned encoding; conventionally use `.jpg` for them and do not assume that changing the suffix transcodes the response.

`prepare_editable_input.py` preserves the original high-resolution master, rejects input that is not strictly 16:9, and creates a new exact 1280x720 PNG. That PNG is input for the downstream `image-to-editable-pptx` converter; the standardizer does not itself create an editable PPTX, and the converter currently handles one slide at a time.

## As an Agent Skill

[SKILL.md](SKILL.md) is written in the agent-skill format: it contains the full workflow
(style template, batch concurrency pattern, quantity-constraining tricks, pitfalls table).
Drop the folder into your agent's skills directory (e.g. `~/.agents/skills/`) to let an
AI agent generate decks autonomously.

## License

[MIT](LICENSE)
