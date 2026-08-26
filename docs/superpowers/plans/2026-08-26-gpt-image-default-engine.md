# GPT Image Default Engine Implementation Plan

> Historical implementation plan. The unchecked boxes and red/green test counts
> below preserve the original execution recipe; they are not the current completion
> state. See **Execution status — 2026-08-26** at the end of this document.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI GPT Image 2 generation as the default and provide a deterministic bridge from any 16:9 generated master image to the exact `1280x720 PNG` required by `image-to-editable-pptx`.

**Architecture:** Add one focused OpenAI provider module and one lazy-loading router module. The provider owns credential/config resolution, Images API requests, retry policy, Base64 validation, and atomic output; the router owns engine selection and defaults. A separate Pillow-based standardizer preserves high-resolution master images while producing the converter's exact input contract; legacy provider public `gen(prompt, out_path, retries=2) -> bool` interfaces remain compatible while credential loading and expected-failure handling may be hardened.

**Tech Stack:** Python 3 standard library (`argparse`, `base64`, `json`, `pathlib`, `tempfile`, `urllib`, `unittest`), Markdown skill documentation.

## Global Constraints

- Default model is exactly `gpt-image-2`.
- Default size is exactly `2048x1152`.
- Default quality is exactly `medium`.
- Credential order is `OPENAI_API_KEY`, then `~/.secrets/openai_api_key`.
- Existing `scripts/gen_slide_gemini.py` and `scripts/gen_slide_doubao.py` interfaces remain compatible.
- Automatic tests must not contact a real API or incur image-generation charges.
- Do not migrate Gemini visual self-checking in this change.
- Keep OpenAI master generation at `2048x1152` and normal PDF/PPTX export at `1920x1080`.
- Editable-converter input must be a real PNG at exactly `1280x720` pixels.
- Reject non-16:9 input instead of silently cropping or padding it.

## File Structure

- Create `scripts/gen_slide_openai.py`: OpenAI credential/config resolution, request/retry logic, response decoding, atomic output.
- Create `scripts/gen_slide.py`: lazy engine router and CLI; OpenAI is the default.
- Create `tests/test_gen_slide_openai.py`: isolated provider behavior tests with fake HTTP responses.
- Create `tests/test_gen_slide.py`: default selection, explicit selection, and CLI routing tests.
- Create `scripts/prepare_editable_input.py`: strict 16:9 validation and atomic `1280x720 PNG` preparation.
- Create `tests/test_prepare_editable_input.py`: format, dimensions, rejection, and preservation tests.
- Modify `README.md`: user-facing setup, quick start, engine table, and fallback instructions.
- Modify `SKILL.md`: agent workflow defaults, prerequisites, examples, costs, and troubleshooting.

---

### Task 1: OpenAI GPT Image provider

**Files:**
- Create: `scripts/gen_slide_openai.py`
- Create: `tests/test_gen_slide_openai.py`

**Interfaces:**
- Consumes: `OPENAI_API_KEY`, optional `~/.secrets/openai_api_key`, optional `OPENAI_IMAGE_MODEL`, `OPENAI_IMAGE_SIZE`, and `OPENAI_IMAGE_QUALITY`.
- Produces: `gen(prompt: str, out_path: str, retries: int = 2) -> bool`.
- Produces for tests: `_load_api_key() -> str` and `_output_format(out_path: str) -> str`.

- [ ] **Step 1: Write failing tests for credentials, format selection, and a successful default request**

Create `tests/test_gen_slide_openai.py` with the following initial content:

```python
import base64
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide_openai


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


class OpenAIConfigTests(unittest.TestCase):
    def test_environment_key_has_priority_over_secret_file(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True), \
             mock.patch.object(Path, "read_text", return_value="file-key") as read_text:
            self.assertEqual(gen_slide_openai._load_api_key(), "env-key")
            read_text.assert_not_called()

    def test_secret_file_is_used_when_environment_key_is_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(Path, "read_text", return_value=" file-key\n"):
            self.assertEqual(gen_slide_openai._load_api_key(), "file-key")

    def test_output_format_matches_supported_file_extension(self):
        self.assertEqual(gen_slide_openai._output_format("slide.jpg"), "jpeg")
        self.assertEqual(gen_slide_openai._output_format("slide.jpeg"), "jpeg")
        self.assertEqual(gen_slide_openai._output_format("slide.png"), "png")
        self.assertEqual(gen_slide_openai._output_format("slide.webp"), "webp")
        with self.assertRaisesRegex(ValueError, "Unsupported output extension"):
            gen_slide_openai._output_format("slide.gif")


class OpenAIGenerationTests(unittest.TestCase):
    def test_success_uses_confirmed_defaults_and_writes_decoded_image(self):
        image_bytes = b"fake-jpeg-bytes"
        response = FakeResponse({
            "data": [{"b64_json": base64.b64encode(image_bytes).decode("ascii")}]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "slide.jpg"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response) as urlopen:
                ok = gen_slide_openai.gen("draw a slide", str(out_path), retries=0)

            self.assertTrue(ok)
            self.assertEqual(out_path.read_bytes(), image_bytes)
            request = urlopen.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, gen_slide_openai.API_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(payload, {
                "model": "gpt-image-2",
                "prompt": "draw a slide",
                "size": "2048x1152",
                "quality": "medium",
                "output_format": "jpeg",
            })

    def test_environment_overrides_model_size_and_quality(self):
        response = FakeResponse({"data": [{"b64_json": base64.b64encode(b"image").decode("ascii")}]})
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_IMAGE_MODEL": "gpt-image-test",
            "OPENAI_IMAGE_SIZE": "1536x1024",
            "OPENAI_IMAGE_QUALITY": "high",
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, environment, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.png"), retries=0))

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "gpt-image-test")
        self.assertEqual(payload["size"], "1536x1024")
        self.assertEqual(payload["quality"], "high")
        self.assertEqual(payload["output_format"], "png")

    def test_invalid_base64_does_not_overwrite_existing_file(self):
        response = FakeResponse({"data": [{"b64_json": "%%%invalid%%%"}]})
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "slide.jpg"
            out_path.write_bytes(b"existing")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                 mock.patch.object(gen_slide_openai.urllib.request, "urlopen", return_value=response):
                self.assertFalse(gen_slide_openai.gen("prompt", str(out_path), retries=0))
            self.assertEqual(out_path.read_bytes(), b"existing")
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_429_retries_then_succeeds(self):
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"error":{"message":"slow down"}}'),
        )
        response = FakeResponse({"data": [{"b64_json": base64.b64encode(b"image").decode("ascii")}]})
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=[error, response]) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertTrue(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=1))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_non_retryable_400_stops_after_one_request(self):
        error = urllib.error.HTTPError(
            gen_slide_openai.API_URL,
            400,
            "bad request",
            {},
            io.BytesIO(b'{"error":{"message":"invalid size"}}'),
        )
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen", side_effect=error) as urlopen, \
             mock.patch.object(gen_slide_openai.time, "sleep") as sleep:
            self.assertFalse(gen_slide_openai.gen("prompt", str(Path(temp_dir) / "slide.jpg"), retries=2))
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_unsupported_extension_fails_before_loading_key_or_network(self):
        with mock.patch.object(gen_slide_openai, "_load_api_key") as load_key, \
             mock.patch.object(gen_slide_openai.urllib.request, "urlopen") as urlopen:
            self.assertFalse(gen_slide_openai.gen("prompt", "slide.gif"))
        load_key.assert_not_called()
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the provider tests and verify the missing module failure**

Run:

```bash
python3 -m unittest tests/test_gen_slide_openai.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gen_slide_openai'`.

- [ ] **Step 3: Implement credentials, output format, default request, and atomic writing**

Create `scripts/gen_slide_openai.py`:

```python
#!/usr/bin/env python3
"""Generate one slide image with OpenAI GPT Image (default engine)."""

import base64
import binascii
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.openai.com/v1/images/generations"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "medium"
SECRET_PATH = Path("~/.secrets/openai_api_key").expanduser()
OUTPUT_FORMATS = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _output_format(out_path: str) -> str:
    extension = Path(out_path).suffix.lower()
    try:
        return OUTPUT_FORMATS[extension]
    except KeyError as exc:
        supported = ", ".join(sorted(OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output extension '{extension or '<none>'}'. Use: {supported}"
        ) from exc


def _atomic_write(data: bytes, out_path: str) -> None:
    target = Path(out_path)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, target)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        message = payload.get("error", {}).get("message")
        if message:
            return str(message)[:300]
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return str(error.reason or error)[:300]


def gen(prompt: str, out_path: str, retries: int = 2) -> bool:
    try:
        output_format = _output_format(out_path)
    except ValueError as error:
        print(f"  ERR: {error}")
        return False

    key = _load_api_key()
    if not key:
        print(
            "  ERR: OpenAI API key not found. Set OPENAI_API_KEY or create "
            "~/.secrets/openai_api_key"
        )
        return False

    payload = {
        "model": os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL),
        "prompt": prompt,
        "size": os.environ.get("OPENAI_IMAGE_SIZE", DEFAULT_SIZE),
        "quality": os.environ.get("OPENAI_IMAGE_QUALITY", DEFAULT_QUALITY),
        "output_format": output_format,
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            print(f"  HTTP {error.code}: {_error_message(error)} (attempt {attempt + 1})")
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(2)
                continue
            return False
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  ERR: {error} (attempt {attempt + 1})")
            if attempt < retries:
                time.sleep(2)
                continue
            return False

        try:
            encoded = result["data"][0]["b64_json"]
            image = base64.b64decode(encoded, validate=True)
            if not image:
                raise ValueError("empty image data")
            _atomic_write(image, out_path)
        except (KeyError, IndexError, TypeError, ValueError, binascii.Error, OSError) as error:
            print(f"  ERR: invalid image response or output failure: {error}")
            return False

        size_kb = len(image) // 1024
        print(f"  OK: {out_path} ({size_kb}KB, OpenAI {payload['model']})")
        return True

    return False


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        success = gen(sys.argv[2], sys.argv[1])
        sys.exit(0 if success else 1)
    print("用法: gen_slide_openai.py <输出路径.jpg|png|webp> <prompt>")
```

- [ ] **Step 4: Run the provider tests and verify the provider is green**

Run:

```bash
python3 -m unittest tests/test_gen_slide_openai.py -v
```

Expected: 9 tests PASS and no real network request occurs.

- [ ] **Step 5: Run checks and commit the provider**

Run:

```bash
python3 -m unittest tests/test_gen_slide_openai.py -v
git diff --check
```

Expected: 9 tests PASS; `git diff --check` produces no output.

Commit:

```bash
git add scripts/gen_slide_openai.py tests/test_gen_slide_openai.py
git commit -m "feat: add GPT Image slide generator"
```

---

### Task 2: Default engine router and CLI

**Files:**
- Create: `scripts/gen_slide.py`
- Create: `tests/test_gen_slide.py`

**Interfaces:**
- Consumes: provider functions named `gen(prompt: str, out_path: str, retries: int = 2) -> bool` from `gen_slide_openai`, `gen_slide_gemini`, and `gen_slide_doubao`.
- Produces: `gen(prompt: str, out_path: str, engine: str = "openai", retries: int = 2) -> bool`.
- Produces: `main(argv: list[str] | None = None) -> int` and CLI `gen_slide.py <out_path> <prompt> [--engine ...] [--retries N]`.

- [ ] **Step 1: Write failing router tests**

Create `tests/test_gen_slide.py`:

```python
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide


class RouterTests(unittest.TestCase):
    def test_openai_is_the_default_engine(self):
        provider_gen = mock.Mock(return_value=True)
        with mock.patch.object(
            gen_slide.importlib,
            "import_module",
            return_value=SimpleNamespace(gen=provider_gen),
        ) as import_module:
            self.assertTrue(gen_slide.gen("prompt", "slide.jpg"))

        import_module.assert_called_once_with("gen_slide_openai")
        provider_gen.assert_called_once_with("prompt", "slide.jpg", retries=2)

    def test_each_named_engine_is_lazy_loaded(self):
        expected = {
            "openai": "gen_slide_openai",
            "gemini": "gen_slide_gemini",
            "doubao": "gen_slide_doubao",
        }
        for engine, module_name in expected.items():
            with self.subTest(engine=engine):
                provider_gen = mock.Mock(return_value=True)
                with mock.patch.object(
                    gen_slide.importlib,
                    "import_module",
                    return_value=SimpleNamespace(gen=provider_gen),
                ) as import_module:
                    self.assertTrue(gen_slide.gen("prompt", "slide.jpg", engine=engine, retries=4))
                import_module.assert_called_once_with(module_name)
                provider_gen.assert_called_once_with("prompt", "slide.jpg", retries=4)

    def test_unknown_engine_fails_without_importing(self):
        with mock.patch.object(gen_slide.importlib, "import_module") as import_module:
            self.assertFalse(gen_slide.gen("prompt", "slide.jpg", engine="unknown"))
        import_module.assert_not_called()

    def test_cli_defaults_to_openai(self):
        with mock.patch.object(gen_slide, "gen", return_value=True) as generate:
            exit_code = gen_slide.main(["slide.jpg", "draw a slide"])
        self.assertEqual(exit_code, 0)
        generate.assert_called_once_with(
            "draw a slide", "slide.jpg", engine="openai", retries=2
        )

    def test_cli_passes_explicit_engine_and_retry_count(self):
        with mock.patch.object(gen_slide, "gen", return_value=False) as generate:
            exit_code = gen_slide.main([
                "slide.png",
                "draw a slide",
                "--engine",
                "gemini",
                "--retries",
                "5",
            ])
        self.assertEqual(exit_code, 1)
        generate.assert_called_once_with(
            "draw a slide", "slide.png", engine="gemini", retries=5
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run router tests and verify the missing module failure**

Run:

```bash
python3 -m unittest tests/test_gen_slide.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gen_slide'`.

- [ ] **Step 3: Implement the minimal lazy router and CLI**

Create `scripts/gen_slide.py`:

```python
#!/usr/bin/env python3
"""Unified slide image generator; OpenAI GPT Image is the default engine."""

import argparse
import importlib
from typing import Optional, Sequence

ENGINE_MODULES = {
    "openai": "gen_slide_openai",
    "gemini": "gen_slide_gemini",
    "doubao": "gen_slide_doubao",
}
DEFAULT_ENGINE = "openai"


def gen(
    prompt: str,
    out_path: str,
    engine: str = DEFAULT_ENGINE,
    retries: int = 2,
) -> bool:
    module_name = ENGINE_MODULES.get(engine)
    if module_name is None:
        choices = ", ".join(ENGINE_MODULES)
        print(f"  ERR: unknown engine '{engine}'. Choose: {choices}")
        return False
    provider = importlib.import_module(module_name)
    return bool(provider.gen(prompt, out_path, retries=retries))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 16:9 slide image (default: OpenAI GPT Image 2)."
    )
    parser.add_argument("out_path", help="Output .jpg, .jpeg, .png, or .webp path")
    parser.add_argument("prompt", help="Image generation prompt")
    parser.add_argument(
        "--engine",
        choices=tuple(ENGINE_MODULES),
        default=DEFAULT_ENGINE,
        help="Image engine (default: openai)",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retry count (default: 2)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    success = gen(
        args.prompt,
        args.out_path,
        engine=args.engine,
        retries=args.retries,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run router and provider tests**

Run:

```bash
python3 -m unittest tests/test_gen_slide.py tests/test_gen_slide_openai.py -v
```

Expected: 14 tests PASS and no real provider module is imported by the mocked router tests.

- [ ] **Step 5: Run CLI help smoke test**

Run:

```bash
python3 scripts/gen_slide.py --help
```

Expected: exit 0; output contains `default: OpenAI GPT Image 2`, `--engine`, and the three engine choices.

- [ ] **Step 6: Commit the router**

Run:

```bash
git diff --check
git add scripts/gen_slide.py tests/test_gen_slide.py
git commit -m "feat: default slide generation to OpenAI"
```

---

### Task 3: Prepare exact input for Image to Editable PPTX

**Files:**
- Create: `scripts/prepare_editable_input.py`
- Create: `tests/test_prepare_editable_input.py`

**Interfaces:**
- Consumes: one Pillow-decodable source image whose pixel dimensions are strictly 16:9.
- Produces: `prepare(input_path: str, output_path: str) -> bool`.
- Produces: a real RGB PNG at exactly `1280x720`, without changing the source or an existing target.
- Produces: CLI `prepare_editable_input.py <input_image> <output.png>`.

- [ ] **Step 1: Write failing standardization tests**

Create `tests/test_prepare_editable_input.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_editable_input


class PrepareEditableInputTests(unittest.TestCase):
    def test_converts_16_by_9_jpeg_to_exact_1280_by_720_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "editable" / "slide.png"
            Image.new("RGB", (2048, 1152), "navy").save(source, format="JPEG")
            source_before = source.read_bytes()

            self.assertTrue(prepare_editable_input.prepare(str(source), str(target)))

            self.assertEqual(source.read_bytes(), source_before)
            with Image.open(target) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1280, 720))
                self.assertEqual(image.mode, "RGB")

    def test_rejects_non_16_by_9_source_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "square.png"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1024, 1024), "white").save(source)

            self.assertFalse(prepare_editable_input.prepare(str(source), str(target)))
            self.assertFalse(target.exists())

    def test_rejects_non_png_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.jpg"
            Image.new("RGB", (1920, 1080), "white").save(source)

            self.assertFalse(prepare_editable_input.prepare(str(source), str(target)))
            self.assertFalse(target.exists())

    def test_rejects_source_and_target_as_the_same_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            Image.new("RGB", (1280, 720), "white").save(source)
            source_before = source.read_bytes()

            self.assertFalse(prepare_editable_input.prepare(str(source), str(source)))
            self.assertEqual(source.read_bytes(), source_before)

    def test_existing_target_is_preserved_and_no_temp_file_remains(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.jpg"
            target = Path(temp_dir) / "slide.png"
            Image.new("RGB", (2560, 1440), "white").save(source)
            target.write_bytes(b"existing-output")

            self.assertFalse(prepare_editable_input.prepare(str(source), str(target)))
            self.assertEqual(target.read_bytes(), b"existing-output")
            self.assertEqual(list(Path(temp_dir).glob(f".{target.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the standardization tests and verify the missing module failure**

Run:

```bash
python3 -m unittest tests/test_prepare_editable_input.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_editable_input'`.

- [ ] **Step 3: Implement strict, non-overwriting PNG standardization**

Create `scripts/prepare_editable_input.py`:

```python
#!/usr/bin/env python3
"""Prepare one exact 1280x720 PNG for image-to-editable-pptx."""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, UnidentifiedImageError

TARGET_SIZE = (1280, 720)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def prepare(input_path: str, output_path: str) -> bool:
    source = Path(input_path)
    target = Path(output_path)

    if target.suffix.lower() != ".png":
        print("  ERR: editable converter input must use a .png output path")
        return False
    if _same_path(source, target):
        print("  ERR: source and output paths must be different")
        return False
    if os.path.lexists(target):
        print(f"  ERR: output already exists; refusing to overwrite: {target}")
        return False

    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            if width * 9 != height * 16:
                print(
                    f"  ERR: source must be exactly 16:9; received {width}x{height}"
                )
                return False
            prepared = image.convert("RGB").resize(
                TARGET_SIZE,
                Image.Resampling.LANCZOS,
            )
    except (OSError, UnidentifiedImageError) as error:
        print(f"  ERR: cannot read source image: {error}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as temp_file:
            prepared.save(temp_file, format="PNG", optimize=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.link(temp_path, target)
        os.unlink(temp_path)
    except FileExistsError:
        print(f"  ERR: output already exists; refusing to overwrite: {target}")
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        return False
    except OSError as error:
        print(f"  ERR: failed to write normalized PNG: {error}")
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        return False

    print(f"  OK: {target} (1280x720 PNG, editable-converter input)")
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an exact 1280x720 PNG for image-to-editable-pptx."
    )
    parser.add_argument("input_path", help="Strict 16:9 source image")
    parser.add_argument("output_path", help="New .png output path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return 0 if prepare(args.input_path, args.output_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run standardizer tests and CLI help**

Run:

```bash
python3 -m unittest tests/test_prepare_editable_input.py -v
python3 scripts/prepare_editable_input.py --help
git diff --check
```

Expected: 5 tests PASS; CLI help exits 0 and states `1280x720 PNG`; whitespace check produces no output.

- [ ] **Step 5: Commit the standardizer**

Run:

```bash
git add scripts/prepare_editable_input.py tests/test_prepare_editable_input.py
git commit -m "feat: prepare editable PPTX converter input"
```

---

### Task 4: Make GPT Image the documented skill default and verify the package

**Files:**
- Modify: `README.md`
- Modify: `SKILL.md`

**Interfaces:**
- Consumes: `scripts/gen_slide.py`, its default `openai` engine, the provider configuration from Task 1, and `scripts/prepare_editable_input.py` from Task 3.
- Produces: copy-pasteable default and fallback instructions for humans and agents.

- [ ] **Step 1: Update README setup and default quick start**

Change the opening description and feature list so they explicitly say GPT Image 2 is the default, with Gemini and Doubao as fallbacks. Replace setup and quick-start examples with these exact command patterns:

```bash
# Standard OpenAI credential (preferred)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

# Or use this project's secret-file convention
mkdir -p ~/.secrets
printf '%s\n' "YOUR_OPENAI_KEY" > ~/.secrets/openai_api_key
chmod 600 ~/.secrets/openai_api_key

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
```

Use the unified Python import in default and batch examples:

```python
import sys
sys.path.insert(0, "scripts")
from gen_slide import gen

gen("<detailed prompt>", "out/slide_01.jpg")
gen("<detailed prompt>", "out/slide_01.jpg", engine="gemini")
```

Update the script table to list `gen_slide.py` first as the default router, `gen_slide_openai.py` as GPT Image 2, and `prepare_editable_input.py` as the deterministic `1280x720 PNG` handoff tool. Do not label Gemini as primary or free-default. State that pricing can change and link to the OpenAI pricing/documentation rather than hard-coding a stale per-image promise. State that the standardizer preserves the high-resolution master, rejects non-16:9 input, and does not itself create an editable PPTX.

- [ ] **Step 2: Update SKILL prerequisites and workflow**

Make these semantic changes in `SKILL.md`:

- Opening summary: OpenAI GPT Image 2 is primary; Gemini nano banana and Doubao Seedream are fallback engines.
- Prerequisites: one OpenAI key is sufficient for default generation; Gemini is additionally required only for Gemini generation or the existing visual self-check.
- Scripts table: add `gen_slide.py` and `gen_slide_openai.py`; preserve export and visual-check rows.
- Step 1 and batch generation: import `gen` from `gen_slide` without an engine argument.
- Fallback examples: use `engine="gemini"` or `engine="doubao"`.
- Pitfall table: add missing OpenAI key, OpenAI organization verification/403, rate limit/429, and unsupported output extension; do not promise automatic fallback.
- Tips: describe `OPENAI_IMAGE_MODEL`, `OPENAI_IMAGE_SIZE`, and `OPENAI_IMAGE_QUALITY` overrides and keep the confirmed defaults explicit.
- Add an optional step after generation that runs `prepare_editable_input.py` and verifies the output is an exact `1280x720 PNG` before invoking `image-to-editable-pptx`.
- Preserve the downstream converter boundary: it currently handles one slide at a time, and a standardized PNG is only its input artifact.

The default batch pattern must use:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_slide import gen

def gen_one(name, prompt):
    ok = gen(prompt, f"out/{name}.jpg")
    return name, ok

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(gen_one, name, prompt) for name, prompt in PROMPTS.items()]
    for future in as_completed(futures):
        name, ok = future.result()
        status = "done" if ok else "failed"
        print(name, status)
```

- [ ] **Step 3: Verify default/fallback wording and inspect the complete diff**

Run:

```bash
rg -n "GPT Image 2|gpt-image-2|gen_slide.py|OPENAI_API_KEY|engine=\"gemini\"|engine=\"doubao\"|prepare_editable_input|1280x720 PNG" README.md SKILL.md
rg -n "Gemini.*primary|nano banana.*primary|Gemini.*首选|免费额度.*主" README.md SKILL.md
git diff -- README.md SKILL.md
git diff --check
```

Expected: the first command shows both documents contain default and fallback instructions; the stale-primary search produces no matches; the diff contains no accidental unrelated rewrite; `git diff --check` produces no output.

- [ ] **Step 4: Run all automated and static verification**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/gen_slide.py --help
python3 scripts/prepare_editable_input.py --help
python3 scripts/validate_skill.py .
git status --short
```

Expected:

- All 19 unit tests PASS with no real API calls.
- Compile check exits 0.
- Both CLI help commands exit 0; generation identifies OpenAI as default and standardization identifies exact `1280x720 PNG` output.
- Skill validation reports success.
- Git status lists only the intended README/SKILL changes before their commit.

- [ ] **Step 5: Commit documentation**

Run:

```bash
git add README.md SKILL.md
git commit -m "docs: make GPT Image the default engine"
```

- [ ] **Step 6: Final repository verification**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/gen_slide.py --help
python3 scripts/prepare_editable_input.py --help
python3 scripts/validate_skill.py .
git diff --check origin/main...HEAD
git status --short --branch
git log --oneline --decorate origin/main..HEAD
```

Expected: every validation exits 0; working tree is clean; local `main` is ahead only by the design, revised plan, provider, router, editable-input standardizer, and documentation commits. Do not push without separate user authorization.

## Execution status — 2026-08-26

The feature and subsequent comprehensive audit were executed on local branches.
The original task steps above are retained as historical TDD instructions; later
hardening expanded their test counts and commit history.

- OpenAI GPT Image 2 is the default router engine with `gpt-image-2`,
  `2048x1152`, and `medium` defaults.
- Gemini `gemini-3.1-flash-image` and Doubao remain explicit fallbacks;
  Gemini visual inspection uses `gemini-3.6-flash`.
- Normal PDF/PPTX export remains `1920x1080`; editable handoff remains a
  separate exact `1280x720 PNG` operation.
- Provider responses and local images now have strict format, 16:9, byte, pixel,
  concurrency, directory-identity, retry, atomic-publication, and rollback tests.
- Current automated suite: **137 tests passed** on Python 3.9 and Python 3.12;
  compilation and `python3 scripts/validate_skill.py .` also passed. Tests used
  mocked provider responses and made no real paid API request.
- Audit work was performed on local branch `audit/comprehensive-completion`.
  Local `main`, audit HEAD, and `origin/main` must be reported separately after
  final integration. No push is implied by this execution record.
