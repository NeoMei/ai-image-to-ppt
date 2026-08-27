---
name: ai-image-to-ppt
description: Generate 16:9 slide images through the host's OpenAI image capability first, with optional API fallbacks, then package them as PDF/PPTX. Use for topic-to-deck, course, and illustrated presentation requests.
---

# AI Image to PPT

Create educational 16:9 slide-image decks. Use the host-first route for image
generation, then use the local scripts for deterministic validation, editable
input preparation, visual checking, and export.

## Image generation routing

Before generating any slide image, read
[references/host-image-routing.md](references/host-image-routing.md) and follow
its candidate order, failure boundaries, artifact-import contract, and serial
sticky batch rules. That reference uses
[`scripts/host_routing_policy.py`](scripts/host_routing_policy.py) for the
deterministic batch state; the Skill retains host discovery and host-tool calls.

- Prefer the host's callable OpenAI image tool; it does not require an
  `OPENAI_API_KEY`.
- Then try the matching OpenAI API adapter only when configured or when the
  user chooses to configure it. Configuration may be skipped.
- Apply the same host-then-key rule to Gemini and Doubao.
- Do not use browser automation, cookies, or extracted host tokens.
- Generate page images serially. Cached pages do not change routing; a fallback
  moves only forward, and a successful page is never regenerated.
- The host importer preserves an absolute workspace raw copy under
  `raw/<filename>`. Only host artifacts within the 0.5% integer-checked
  near-16:9 tolerance may be center-cropped (never stretched) into a strict
  16:9 master; API outputs remain strict 16:9 without this exception.
- Keep generated masters. Create exact `1280×720 PNG` files only for the
  editable-converter handoff.

## Optional API/CLI-only credentials

The recommended host route needs no API key. `scripts/gen_slide.py` is an
**API/CLI-only** adapter for a selected provider; it cannot call host tools.
Each supported key may be skipped. Use Python 3.9+ and install dependencies:

```bash
mkdir -p ~/.secrets
printf '%s\n' "YOUR_OPENAI_KEY" > ~/.secrets/openai_api_key
chmod 600 ~/.secrets/openai_api_key
printf '%s\n' "YOUR_GEMINI_KEY" > ~/.secrets/gemini_api_key
chmod 600 ~/.secrets/gemini_api_key
printf '%s\n' "YOUR_DOUBAO_KEY" > ~/.secrets/doubao_api_key
chmod 600 ~/.secrets/doubao_api_key
python3 -m pip install -r requirements.txt
```

Never commit or print credentials. Keys must be one-line printable ASCII with
no whitespace or control characters. Gemini also needs a key for the optional
vision self-check.

For an intentional low-level API call, retain explicit provider selection:

```bash
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>"
python3 scripts/gen_slide.py out/slide_01.png "<prompt>" --engine gemini
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>" --engine doubao
```

Gemini accepts `.png`, `.jpg`, and `.jpeg` and uses PNG by default. For a
low-level Python API call, keep the same explicit selection:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
from gen_slide import gen

gen("<prompt>", "out/slide_01.png", engine="gemini")
```

## Prompt style

Append this style block to visual prompts unless the user asks for another
art direction:

```
Style: 16:9 educational textbook infographic slide.
Background: soft warm cream off-white (#f8f5f0), subtle paper-like texture.
Typography: clean modern Chinese sans-serif (Noto Sans / Inter), dark navy-charcoal.
Accent: terracotta orange (#e67e22); friendly hand-drawn icons and rounded cards.
Layout: generous whitespace; no dark edge banners, photos, neon, gradients, or 3D.
```

For counted objects, make the count, grid, and exclusions explicit:

```
CRITICAL: Render EXACTLY 8 quote cards, no more, no less.
Layout: strict 2 columns × 4 rows = EXACTLY 8 cells.
Do NOT add a 9th card. Do NOT add a summary/conclusion card.
Final check before output: total cards MUST be 8.
```

## Validate, convert, and export

Run a visual self-check for quantity-sensitive pages:

```bash
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"
```

For `image-to-editable-pptx`, preserve the strict-16:9 master and its unchanged
raw host copy, then create a separate exact `1280×720 PNG`; this command does
not create an editable PPTX:

```bash
python3 scripts/prepare_editable_input.py \
  out/slide_01.jpg \
  out/editable/slide_01.png
```

Use an explicit, ordered manifest for export. It supports mixed suffixes and
prevents duplicate slide IDs:

```python
from pathlib import Path
import subprocess
import sys

SLIDES = ["out/slide_01.jpg", "out/slide_02.png", "out/slide_03.webp"]
if len(SLIDES) != len(set(SLIDES)):
    raise SystemExit("duplicate slide paths in SLIDES")
SLIDE_IDS = [Path(path).stem for path in SLIDES]
if len(SLIDE_IDS) != len(set(SLIDE_IDS)):
    raise SystemExit("duplicate slide IDs across output suffixes")
subprocess.run([sys.executable, "scripts/export_images.py", "deck_name", *SLIDES], check=True)
```

Generation refuses existing outputs unless `--force` is explicit. Inputs and
generated images are capped at 50 MiB and 64 MP; vision inputs have a separate
14 MiB cap. Export validates input, writes PDF/PPTX with recovery on the next
export, and accepts at most 128 slides or 512 MiB aggregate source bytes.
Generation and host import require POSIX secure publication primitives
(directory-descriptor/no-follow checks, hard links, and same-directory rename);
unsupported platforms fail closed as `local_failure` instead of weakening
output ownership checks. Forced replacement first moves the current pathname
into a private same-directory recovery area and verifies the inode actually
moved before installing without clobbering; this is not a single-syscall atomic
replacement. Concurrent unknown files and failed recovery cleanup are retained
under `.image-output-recovery-*/entry` and reported with a bounded warning.
Cooperating processes may race public output names. A same-UID actor that
deliberately discovers and mutates the private recovery namespace (random and
mode 0700) between syscalls is outside the portable guarantee because POSIX has
no unlink-if-inode primitive; treat reported retained paths as sensitive and
recover them manually.
Clearly over-limit manifests are rejected during path preflight, before image
decoding. If source files change after preflight, actual loaded bytes are
accumulated after each image load and rejected before PDF/PPTX serialization.

Stable output lock files intentionally persist. Run cleanup only while no
generation or export process is active:

```bash
python3 scripts/cleanup_output_locks.py --older-than-days 30
```

## Historical reference

The 2026-07-25 nano banana / Doubao provider period generated 224 slides in
276 seconds with concurrent API work. It is historical only and does not
represent current host-first GPT Image 2 performance; never reuse that
concurrency pattern for a sticky routed batch.
