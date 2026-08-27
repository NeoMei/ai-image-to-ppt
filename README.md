# ai-image-to-ppt

Generate educational 16:9 slide images from a topic or outline, package them
into PDF/PPTX, and optionally prepare a single-page editable-converter input.

## Recommended: host-first Skill

In a supported agent host, use this repository's `ai-image-to-ppt` Skill. It
checks the host's callable OpenAI image capability first, so no API key is
required for the recommended path. It then follows the same host-then-API rule
for OpenAI, Gemini, and Doubao in this fixed order: host OpenAI, OpenAI API,
host Gemini, Gemini API, host Doubao, Doubao API.

Only currently callable host tools or connected connectors count. A browser
login is not a capability. The Skill processes slide images serially with a
sticky route, advances only for unavailable/auth/retryable outcomes, stops on
safety or validation failures, and reports page-level switches. Cached pages do
not establish or change that route.

See [host routing details](references/host-image-routing.md) for artifact
acceptance, import, status handling, and the final redacted report contract.

## Optional API/CLI-only credentials

Python 3.9+ is required for local scripts. API keys are optional and may be
skipped: they are only for the **API/CLI-only** adapters, not a requirement for
the host-first Skill. These exact secret-file commands remain supported:

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

Do not commit keys. Credentials must be non-empty, printable ASCII on one line,
with no whitespace or control characters. Gemini also needs a key for the
optional visual self-check.

## API/CLI-only generation

`scripts/gen_slide.py` calls one selected API provider; it does not discover or
call host capabilities and does not perform the six-candidate route. Use it
when an explicit API-only call is intended:

```bash
mkdir -p out out/editable
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>"
python3 scripts/gen_slide.py out/slide_01.png "<prompt>" --engine gemini
python3 scripts/gen_slide.py out/slide_01.jpg "<prompt>" --engine doubao
```

OpenAI uses `gpt-image-2` at 2048×1152/medium by default. Gemini uses
`gemini-3.1-flash-image`, targets 16:9 2K, and uses PNG by default; the optional
vision check uses `gemini-3.6-flash`. Doubao uses the current
`doubao-seedream-5-0-260128` default model. OpenAI and Doubao accept `.jpg`, `.jpeg`,
`.png`, and `.webp`; Gemini accepts `.png`, `.jpg`, and `.jpeg`.

All API sources are decoded and checked for actual format and strict 16:9 before
publication. A host artifact is first retained unchanged at the absolute
workspace sibling path `raw/<filename>`. The host importer alone may accept a
near-16:9 image within a 0.5% integer cross-product tolerance, center-crop it
(never stretch it) to the largest strict 16:9 master, and strictly revalidate
the encoded master. API adapters never receive that tolerance or crop. Existing
outputs are protected unless `--force` is explicit; use `--force` only for
intentional replacement with rollback and crash-recovery protection. Image
inputs and generation outputs are capped at 50 MiB and 64 MP. Vision-check
inputs have a separate 14 MiB limit. Transient provider retries honor bounded
`Retry-After` guidance. Parent-directory replacement is detected before
publication and fails closed. The paired host raw/master publication uses
compensating rollback for ordinary process-time failures while both output locks
are held; it is not a crash-atomic two-file commit.

## Editable-converter handoff

Keep the high-resolution strict 16:9 master and unchanged raw host copy. For
`image-to-editable-pptx`, make a separate real exact `1280×720 PNG`; this is
converter input, not an editable PPTX:

```bash
python3 scripts/prepare_editable_input.py \
  out/slide_01.jpg \
  out/editable/slide_01.png
```

## Visual check and export

```bash
python3 scripts/vision_check_gemini.py out/slide_01.jpg "精确数一下卡片数量是否正确?"
python3 scripts/export_images.py deck_name out/slide_01.jpg out/slide_02.png
```

Export normalizes images to 1920×1080, validates every source before publishing,
and uses a same-directory recovery journal after interrupted output. A deck is
limited to 128 slides and 512 MiB aggregate source bytes. Clearly over-limit
manifests are rejected during path preflight, before image decoding. If source
files change after preflight, actual loaded bytes are accumulated after each
image load and rejected before PDF/PPTX serialization. On POSIX, file and
directory fsync cover process crashes and power-loss metadata recovery. On
Windows, recovery covers process crashes only and does not promise power-loss
durability.

Stable output locks persist intentionally. Cleanup never runs automatically;
run it ONLY while no generation or export process is active:

```bash
python3 scripts/cleanup_output_locks.py --older-than-days 30
```

Because cleanup uses pathname age checks and removal, the offline precondition
is the safety boundary for pathname races. Filesystem failures stop cleanup,
return exit 1 without a traceback, and do not roll back earlier removals.

## Development

```bash
python3 -m pip install -r requirements-dev.txt
python3 scripts/validate_skill.py .
python3 -m unittest discover -s tests -v
```

## License

[MIT](LICENSE)
