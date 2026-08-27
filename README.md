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

Python 3.9 or newer and one OpenAI API key are sufficient for default generation:

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

python3 -m pip install -r requirements.txt
```

Gemini and Doubao credentials are only needed when selecting those fallback engines. A Gemini key is also required for the optional visual self-check. The commands above restore the exact secret-file paths expected by the legacy scripts. Never commit real keys.

Provider API keys must be non-empty printable ASCII on one line, with no whitespace or control characters. Invalid credentials fail before any provider request and are never printed in errors.

## Quick Start

```bash
mkdir -p out out/editable

# Default: OpenAI GPT Image 2, 2048x1152, medium
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>"

# Existing outputs are protected. Regenerate explicitly only when intended.
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>" --force

# Explicit fallback engines. Gemini defaults to PNG.
python3 scripts/gen_slide.py out/slide_01.png "<prompt>" --engine gemini
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>" --engine doubao

# Optional handoff: prepare the exact single-slide input required by
# image-to-editable-pptx. This PNG is converter input, not an editable PPTX.
python3 scripts/prepare_editable_input.py \
  out/slide_01.jpg \
  out/editable/slide_01.png

# 视觉自检
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"

# 打包导出：使用下方显式清单，兼容混合图片格式并固定页序
```

The export command validates every image before publishing either artifact,
creates missing parent directories, and uses a same-directory recovery journal
so interrupted PDF/PPTX publication can be repaired on the next export. On
POSIX, file and directory fsync cover process crashes and power-loss metadata
recovery. On Windows, recovery covers process crashes only and does not promise
power-loss durability. It refuses existing outputs by default; add `--force` to
replace an existing pair with rollback and crash-recovery protection. One deck
accepts at most 128 slides and 512 MiB of aggregate source image bytes; larger
Python or CLI requests fail before image decoding or PDF/PPTX serialization.

脚本内调用（批量并发生成见 [SKILL.md](SKILL.md)）：

```python
import sys
sys.path.insert(0, "scripts")
from gen_slide import gen

gen("<detailed prompt>", "out/slide_01.jpg")
gen("<detailed prompt>", "out/slide_01.png", engine="gemini")
```

For a deck, use one explicit manifest as the source of truth. Its order is the
slide order, mixed suffixes are supported, and duplicate slide IDs fail before
export instead of producing duplicate pages. Update the paths to match the
artifacts you actually generated; do not use a single-suffix glob.

```python
from pathlib import Path
import subprocess
import sys

SLIDES = [
    "out/slide_01.jpg",
    "out/slide_02.png",
    "out/slide_03.webp",
]
if len(SLIDES) != len(set(SLIDES)):
    raise SystemExit("duplicate slide paths in SLIDES")
SLIDE_IDS = [Path(path).stem for path in SLIDES]
if len(SLIDE_IDS) != len(set(SLIDE_IDS)):
    raise SystemExit("duplicate slide IDs across output suffixes")
missing = [path for path in SLIDES if not Path(path).is_file()]
if missing:
    raise SystemExit(f"missing slide files: {missing}")

# Keep True for the CLI route; set False to use the Python API instead.
USE_CLI = True
if USE_CLI:
    subprocess.run(
        [sys.executable, "scripts/export_images.py", "deck_name", *SLIDES],
        check=True,
    )
else:
    sys.path.insert(0, "scripts")
    from export_images import export_deck

    if not export_deck(SLIDES, "deck_name"):
        raise SystemExit("deck export failed")
```

## Scripts

| Script | Role / Engine | API / Output |
|---|---|---|
| `gen_slide.py` | Default provider router | OpenAI by default; `--engine gemini\|doubao` for explicit fallback |
| `gen_slide_openai.py` | OpenAI GPT Image 2 | `gpt-image-2`; defaults to 2048×1152, medium quality |
| `gen_slide_gemini.py` | Gemini nano banana | `gemini-3.1-flash-image`; 16:9, 2K; PNG by default or JPEG by suffix |
| `gen_slide_doubao.py` | 方舟 doubao-seedream | OpenAI-compatible API |
| `prepare_editable_input.py` | Editable-converter handoff | Deterministic 1280x720 PNG |
| `vision_check_gemini.py` | Gemini visual self-check | `gemini-3.6-flash` |
| `export_images.py` | Local export | PDF + PPTX |
| `cleanup_output_locks.py` | Explicit offline maintenance | Removes old stable hashed lock files |

The router does not automatically switch providers: choose fallback engines explicitly. Provider pricing can change; consult the [OpenAI API pricing documentation](https://developers.openai.com/api/docs/pricing) instead of relying on a fixed per-image estimate.

OpenAI and Doubao support output extensions `.jpg`, `.jpeg`, `.png`, and `.webp`.
Gemini supports `.png`, `.jpg`, and `.jpeg`: PNG by default, while a JPEG suffix
requests `IMAGE_JPEG`. Unsupported Gemini suffixes fail before credential lookup or
network access. Only Gemini validates the provider-declared response MIME type.
All providers validate the actual image encoding and strict 16:9 dimensions before
publication. Generation refuses existing
outputs by default and makes no provider request; pass `overwrite=True` in Python
or `--force` on the CLI only when replacement is intentional.

Concurrent work on the same output fails fast before contacting a provider.
Stable hashed lock files intentionally persist because automatically unlinking a
lock file can split ownership between processes. Optional cleanup never runs
automatically. Run it ONLY while no generation or export process is running:

```bash
python3 scripts/cleanup_output_locks.py --older-than-days 30
```

Because cleanup uses pathname age checks and removal, the offline precondition
is the safety boundary for pathname races. Filesystem failures stop cleanup,
return exit 1 without a traceback, and do not roll back earlier removals.

Generation outputs and inputs to ordinary export or editable preparation are
capped at 50 MiB and 64 megapixels. Vision-check inputs have a separate 14 MiB
limit. Parent-directory replacement is detected before publication, and
transient HTTP retries honor a bounded `Retry-After` value when supplied.

`prepare_editable_input.py` preserves the original high-resolution master, rejects input that is not strictly 16:9, and creates a new exact 1280x720 PNG. That PNG is input for the downstream `image-to-editable-pptx` converter; the standardizer does not itself create an editable PPTX, and the converter currently handles one slide at a time.

## As an Agent Skill

[SKILL.md](SKILL.md) is written in the agent-skill format: it contains the full workflow
(style template, batch concurrency pattern, quantity-constraining tricks, pitfalls table).
Drop the folder into your agent's skills directory (e.g. `~/.agents/skills/`) to let an
AI agent generate decks autonomously.

For development and package validation:

```bash
python3 -m pip install -r requirements-dev.txt
python3 scripts/validate_skill.py .
python3 -m unittest discover -s tests -v
```

## License

[MIT](LICENSE)
