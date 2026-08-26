---
name: ai-image-to-ppt
description: Use when generating 16:9 presentation slide images from a topic using AI image generation models, then packaging them into PDF/PPTX. Triggers include "用 AI 生成 PPT", "nano banana 生成幻灯片", "方舟 doubao-seedream 生成图", "把主题/大纲变成图文 PPT", "批量生成教科书风插图", "图片打包成 PDF PPTX", "16:9 slide generation". Especially useful for converting book outlines, course materials, or technical documentation into visual decks.
---

# AI Image to PPT

Generate 16:9 educational textbook-style slide images from any topic using OpenAI **GPT Image 2** as the primary engine, with Gemini **nano banana** and Volcano **doubao-seedream** as explicit fallback engines, then package them into PDF + PPTX.

## When to Use

- Convert book outlines / course notes / technical docs into visual slide decks
- Batch generate slide illustrations with Chinese + English mixed typography
- Need consistent 16:9 visual style across many slides

## When NOT to Use

- Need precise text layout editing → use PowerPoint/Keynote directly
- Need real-time interactive slides
- Single one-off image → call API directly

## Prerequisites

Python 3.9 or newer and one OpenAI key are sufficient for default generation:

```bash
# Preferred
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

# Or the project's secret-file convention
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

Add a Gemini credential only for Gemini image generation or the visual self-check. Add a Doubao credential only when explicitly selecting Doubao. The exact secret-file paths above are what the legacy scripts read. Provider selection is explicit; generation does not automatically fall back.

Provider API keys must be non-empty printable ASCII on one line, with no whitespace or control characters. Invalid credentials fail before any provider request and are never printed in errors.

## Scripts

All in `scripts/` directory. Copy to project or add to `PYTHONPATH`.

| Script | Role / Engine | API / Output |
|---|---|---|
| `gen_slide.py` | Default provider router | OpenAI by default; Gemini and Doubao are explicit choices |
| `gen_slide_openai.py` | OpenAI GPT Image 2 | `gpt-image-2`; 2048×1152, medium by default |
| `gen_slide_gemini.py` | Gemini nano banana | `gemini-3.1-flash-image`; 16:9, 2K; PNG by default or JPEG by suffix |
| `gen_slide_doubao.py` | 方舟 doubao-seedream-5-0 | OpenAI-compatible API |
| `prepare_editable_input.py` | Editable-converter handoff | Exact 1280x720 PNG |
| `vision_check_gemini.py` | Gemini visual self-check | `gemini-3.6-flash` |
| `export_images.py` | Local export | PDF + PPTX |

## Workflow

### Step 1: Generate single slide

```python
import sys
sys.path.insert(0, "<skill_path>/scripts")
from gen_slide import gen

gen("<detailed prompt>", "out/slide_01.jpg")

# Explicit fallbacks when requested
gen("<detailed prompt>", "out/slide_01.png", engine="gemini")
gen("<detailed prompt>", "out/slide_01.jpg", engine="doubao")
```

### Step 2: Batch generate with concurrency

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_slide import gen

PROMPTS = {
    "slide_01": "<prompt 1>",
    "slide_02": "<prompt 2>",
    # ...
}

def gen_one(name, prompt):
    path = f"out/{name}.jpg"
    if __import__('os').path.exists(path):
        return name, "cached"
    ok = gen(prompt, path)
    return name, "done" if ok else "failed"

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(gen_one, n, p): n for n, p in PROMPTS.items()}
    for f in as_completed(futures):
        name, status = f.result()
        print(name, status)
```

### Step 3: Optionally prepare one editable-converter input

The downstream `image-to-editable-pptx` converter currently processes one slide at a time. Preserve the generated high-resolution master and create a separate standardized artifact:

```bash
python3 scripts/prepare_editable_input.py \
  out/slide_01.jpg \
  out/editable/slide_01.png
```

Before invoking the converter, verify that the new output is an exact 1280x720 PNG. The standardizer rejects non-16:9 input and existing output paths; it does not create an editable PPTX. Its PNG is only the converter's input artifact.

### Step 4: Vision self-check (recommended for quantity-sensitive prompts)

```bash
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"
```

### Step 5: Export to PDF + PPTX

```bash
# CLI (glob accepts space-separated paths)
python3 scripts/export_images.py "deck_name" out/*.jpg

# Python
from export_images import export_pdf, export_pptx
files = sorted(__import__('glob').glob("out/*.jpg"))
export_pdf(files, "deck.pdf")
export_pptx(files, "deck.pptx")
```

The CLI validates all source images before publishing, creates missing output
directories, and treats the PDF/PPTX files as one transactional pair. Existing
outputs are preserved unless `--force` is supplied. Transparent pixels are
composited onto the style's cream `#f8f5f0` background rather than black.

## Style Template (Educational Textbook)

Always append this `STYLE` block to visual prompts to keep slides consistent:

```
Style: 16:9 educational textbook infographic slide.
Background: soft warm cream off-white (#f8f5f0), subtle paper-like texture.
Typography: clean modern Chinese sans-serif (Noto Sans / Inter),
  dark navy-charcoal (#2c3e50) main text, gray (#6b7280) secondary.
Accent color: terracotta orange (#e67e22) for highlights, arrows, key icons.
Icons: hand-drawn cartoon style, friendly, muted palette.
Cards: rounded rectangles, thin light gray borders, very subtle shadow.
Layout: generous whitespace, NO top/bottom dark banners, NO strips at edges.
Mood: warm, professional, calm, educational.
No photos, no neon, no glow effects, no gradients, no 3D renders.
```

## Quantity-Sensitive Prompts (CRITICAL)

Image models miscount items (8 cards → 9). Always over-constrain:

```
CRITICAL: Render EXACTLY 8 quote cards, no more, no less.
Layout: strict 2 columns × 4 rows = EXACTLY 8 cells.
Do NOT add a 9th card. Do NOT add summary/conclusion card.
[list all 8 items explicitly]
Final check before output: total cards MUST be 8.
```

## Chapter Metadata Pattern

For multi-chapter decks (e.g. book summaries), define metadata once and templatize:

```python
META = [
    {
        "num": "01", "slug": "ch01_intro",
        "zh": "提示链", "en": "Prompt Chaining",
        "one_liner": "拆线性流水线",
        "problem": "...", "solution": "...",
        "key_points": ["...", "..."],
        "frameworks": [("LangChain", "LCEL", "pipe"), ...],
        "scenarios": ["..."], "pitfalls": ["..."],
        "quote": "...",
    },
    # ... more chapters
]

# 8 standard pages per chapter:
# 01_cover, 02_glance, 03_what, 04_why,
# 05_mechanism, 06_frameworks, 07_rule, 08_summary
```

See `examples/chapters_meta.py` for a filled-in example.

## Critical Pitfalls

| Symptom | Fix |
|---|---|
| OpenAI key missing | Set `OPENAI_API_KEY` or create `~/.secrets/openai_api_key` |
| OpenAI HTTP 403 / organization verification required | Complete the required OpenAI organization verification, then retry |
| OpenAI HTTP 429 / rate limit | Wait for capacity or quota, then retry; select another engine explicitly if desired |
| Unsupported OpenAI output extension | Use `.jpg`, `.jpeg`, `.png`, or `.webp` |
| Gemini HTTP 429 (配额超) | Wait for quota or explicitly use `engine="doubao"` |
| Items miscounted (8→9) | Add `CRITICAL: EXACTLY N` constraint, list items explicitly |
| PDF 页大小不一致 | `export_images.py` auto-normalizes to 1920×1080 |
| Chinese 错字/糊字 | Run `vision_check_gemini.py` on samples, regenerate failures |
| 风格不统一 | Always append the same `STYLE` block to every prompt |
| Output not工整 (misaligned) | Don't embed raw images of varying dimensions; always go through normalize |
| Gemini vision model unavailable | Use the current default `gemini-3.6-flash` or set `GEMINI_VISION_MODEL` to an enabled compatible model |
| doubao chat 404 | Vision models need separate endpoint activation; image gen works |

## Real-World Reference

`agentic-design-patterns` project: 28 chapters × 8 pages = 224 slides generated in **276 seconds** (8 concurrent), plus 17 overview slides in 44s. All exported to 58 individual PDF + PPTX files. ~12 sample slides passed Gemini visual self-check.

## Tips

- **Concurrency sweet spot**: 4-8 workers. Higher triggers rate limits.
- **OpenAI defaults and overrides**: Confirmed defaults are `gpt-image-2`, `2048x1152`, and `medium`. Override them with `OPENAI_IMAGE_MODEL`, `OPENAI_IMAGE_SIZE`, and `OPENAI_IMAGE_QUALITY` when needed.
- **Retry behavior**: Providers retry some transient failures internally; surface final failures for manual handling. There is no automatic provider fallback.
- **Output extensions**: OpenAI and Doubao accept `.jpg`, `.jpeg`, `.png`, and `.webp`. Gemini accepts `.png`, `.jpg`, and `.jpeg`: PNG by default, while a JPEG suffix requests `IMAGE_JPEG`. Unsupported Gemini suffixes fail before credential lookup or network access. Only Gemini validates the provider-declared response MIME type. All providers validate the actual image encoding and strict 16:9 dimensions before publication.
- **Safe reruns**: Generation refuses existing outputs without contacting a provider. Treat that as cached in batch jobs; pass `overwrite=True` or `--force` only for intentional regeneration.
- **Safety boundaries**: Same-output work fails fast before provider access; images are limited to 50 MiB and 64 megapixels; directory replacement fails closed; transient HTTP retries honor bounded `Retry-After` guidance.
- **First-page validation**: Generate 1 sample, visually confirm style, then batch.
- **Pricing**: Provider pricing can change. Check the [OpenAI API pricing documentation](https://developers.openai.com/api/docs/pricing) instead of assuming a fixed per-image cost.
