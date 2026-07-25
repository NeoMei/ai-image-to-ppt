---
name: ai-image-to-ppt
description: Use when generating 16:9 presentation slide images from a topic using AI image generation models, then packaging them into PDF/PPTX. Triggers include "用 AI 生成 PPT", "nano banana 生成幻灯片", "方舟 doubao-seedream 生成图", "把主题/大纲变成图文 PPT", "批量生成教科书风插图", "图片打包成 PDF PPTX", "16:9 slide generation". Especially useful for converting book outlines, course materials, or technical documentation into visual decks.
---

# AI Image to PPT

Generate 16:9 educational textbook-style slide images from any topic using AI image generation models (Gemini **nano banana 2** primary, Volcano **doubao-seedream** fallback), then package into PDF + PPTX.

## When to Use

- Convert book outlines / course notes / technical docs into visual slide decks
- Batch generate slide illustrations with Chinese + English mixed typography
- Need consistent 16:9 visual style across many slides

## When NOT to Use

- Need precise text layout editing → use PowerPoint/Keynote directly
- Need real-time interactive slides
- Single one-off image → call API directly

## Prerequisites

Two API keys (any one suffices; both recommended for redundancy):

```bash
# Gemini (https://aistudio.google.com/apikey)
echo "YOUR_GEMINI_KEY" > ~/.secrets/gemini_api_key

# 方舟 doubao (https://console.volcengine.com/ark)
echo "YOUR_DOUBAO_KEY" > ~/.secrets/doubao_api_key
chmod 600 ~/.secrets/*_api_key

pip install Pillow python-pptx
```

## Scripts

All in `scripts/` directory. Copy to project or add to `PYTHONPATH`.

| Script | Engine | API | Cost |
|---|---|---|---|
| `gen_slide_gemini.py` | Gemini nano banana 2 | `gemini-3.1-flash-image-preview` | 免费 500/天 |
| `gen_slide_doubao.py` | 方舟 doubao-seedream-5-0 | OpenAI 兼容 | 按量付费 |
| `vision_check_gemini.py` | Gemini 2.0 Flash 视觉理解 | `gemini-2.0-flash` | 免费 |
| `export_images.py` | Pillow + python-pptx | 本地 | 免费 |

Both generators share identical interface `gen(prompt, out_path) -> bool`, swappable.

## Workflow

### Step 1: Generate single slide

```python
import sys
sys.path.insert(0, "<skill_path>/scripts")
from gen_slide_gemini import gen  # or gen_slide_doubao

gen("<detailed prompt>", "out/slide_01.jpg")
```

### Step 2: Batch generate with concurrency

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_slide_gemini import gen

PROMPTS = {
    "slide_01": "<prompt 1>",
    "slide_02": "<prompt 2>",
    # ...
}

def gen_one(name, prompt):
    gen(prompt, f"out/{name}.jpg")
    return name

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(gen_one, n, p): n for n, p in PROMPTS.items()}
    for f in as_completed(futures):
        print(f.result(), "done")
```

### Step 3: Vision self-check (recommended for quantity-sensitive prompts)

```bash
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"
```

### Step 4: Export to PDF + PPTX

```bash
# CLI (glob accepts space-separated paths)
python3 scripts/export_images.py "deck_name" out/*.jpg

# Python
from export_images import export_pdf, export_pptx
files = sorted(__import__('glob').glob("out/*.jpg"))
export_pdf(files, "deck.pdf")
export_pptx(files, "deck.pptx")
```

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
| Gemini HTTP 429 (配额超) | Switch to `gen_slide_doubao` |
| Items miscounted (8→9) | Add `CRITICAL: EXACTLY N` constraint, list items explicitly |
| PDF 页大小不一致 | `export_images.py` auto-normalizes to 1920×1080 |
| Chinese 错字/糊字 | Run `vision_check_gemini.py` on samples, regenerate failures |
| 风格不统一 | Always append the same `STYLE` block to every prompt |
| Output not工整 (misaligned) | Don't embed raw images of varying dimensions; always go through normalize |
| Gemini key 报 404 on vision | Use `gemini-2.0-flash` not `gemini-2.5-flash` (not all keys enabled) |
| doubao chat 404 | Vision models need separate endpoint activation; image gen works |

## Real-World Reference

`agentic-design-patterns` project: 28 chapters × 8 pages = 224 slides generated in **276 seconds** (8 concurrent), plus 17 overview slides in 44s. All exported to 58 individual PDF + PPTX files. ~12 sample slides passed Gemini visual self-check.

## Tips

- **Concurrency sweet spot**: 4-8 workers. Higher triggers rate limits.
- **Retry once on failure**: Both generators retry internally; surface failures for manual retry.
- **Cache by file existence**: Skip already-generated files when re-running batch jobs.
- **First-page validation**: Generate 1 sample, visually confirm style, then batch.
- **Cost**: nano banana 2 free tier covers ~500 images/day. doubao-seedream-5-0 paid.
