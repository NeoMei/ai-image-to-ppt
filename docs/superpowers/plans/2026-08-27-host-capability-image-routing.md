# Host-First Image Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ai-image-to-ppt` Skill prefer host-provided OpenAI, Gemini, and Doubao image capabilities before the matching API-key adapters, with deterministic fallback, structured failure results, sticky deck routing, safe workspace publication, and exact `1280×720 PNG` editable-converter inputs.

**Architecture:** The Skill remains the only layer allowed to discover and invoke host tools. Python modules expose structured API-adapter results and a deterministic host-artifact importer, but never impersonate or reach into host authentication. The Skill follows one six-candidate, serial, monotonic routing protocol; every successful host artifact is materialized into a same-directory temporary file, validated, and atomically published before downstream standardization or export.

**Tech Stack:** Python 3.9+, standard library (`argparse`, `dataclasses`, `enum`, `json`, `urllib`, `pathlib`), Pillow 9.1+, `unittest`, Markdown Agent Skill instructions.

**Spec:** `docs/superpowers/specs/2026-08-27-host-capability-image-routing-design.md`

## Global Constraints

- Keep Python 3.9 compatibility; use `Optional[str]`, `Union[str, bytes]`, and `Sequence[str]`, not PEP 604 unions.
- The host candidate order is exactly: host OpenAI, OpenAI API, host Gemini, Gemini API, host Doubao, Doubao API.
- Only `unavailable`, `auth_unavailable`, and `retryable_exhausted` permit trying the next candidate.
- `policy_refused`, `invalid_input`, `invalid_output`, and `local_failure` stop the task immediately.
- API-key configuration is optional and every configuration prompt must allow skipping.
- Do not read browser cookies, scrape web sessions, extract host tokens, or let Python call Codex built-in tools.
- Image generation runs serially by page so sticky routing can only move forward and never regenerate a successful page.
- A cached page never establishes or changes the current candidate.
- Host output is capped at 50 MiB and 64 MP, must decode successfully, must match its final suffix, and must be strict 16:9 before publication.
- Resolve and capture the workspace root once. Reject lexical `..`, workspace escape, ancestor-symlink escape, final-target symlinks, and parent identity replacement.
- Validate a same-directory temporary file before no-clobber or `force` atomic publication; an invalid artifact must not create or replace the final target.
- Preserve high-resolution masters. Only `image-to-editable-pptx` handoff files are converted into separate exact `1280×720 PNG` artifacts.
- Keep existing `gen(prompt, out_path, retries=2, overwrite=False) -> bool` interfaces and default human CLI exit codes compatible.
- Default and automated tests must not use real API keys, network requests, or paid API calls.

## File Map

- Create `scripts/generation_result.py`: shared statuses, `GenerationResult`, fallback predicate, safe serialization, and HTTP-status classification primitives.
- Modify `scripts/image_output.py`: validate image bytes before publication, read bounded image streams, and prepare workspace-contained targets.
- Create `scripts/import_host_image.py`: materialize supported host artifacts and safely publish them into the captured workspace.
- Modify `scripts/gen_slide_openai.py`: return structured OpenAI API results while preserving `gen()`.
- Modify `scripts/gen_slide_gemini.py`: return structured Gemini API results, including structured safety refusals, while preserving `gen()`.
- Modify `scripts/gen_slide_doubao.py`: return structured request/download results while preserving `gen()`.
- Modify `scripts/gen_slide.py`: expose provider `generate_result()` and `--json`, while remaining an API-only router.
- Create `references/host-image-routing.md`: detailed six-candidate host orchestration, artifact handoff, status, batching, and reporting protocol.
- Modify `SKILL.md`: make host-first routing the recommended path and link the detailed reference only when generating images.
- Modify `README.md`: distinguish the host Skill workflow from the API/CLI workflow and document optional credentials.
- Create focused tests for each new module and provider result path; update existing compatibility and documentation tests.

---

### Task 1: Shared Structured Generation Result Contract

**Files:**
- Create: `scripts/generation_result.py`
- Create: `tests/test_generation_result.py`

**Interfaces:**
- Consumes: provider/channel names and already-redacted diagnostic text.
- Produces: `GenerationStatus`, `GenerationResult`, `FALLBACK_STATUSES`, `FATAL_STATUSES`, `classify_http_failure(status_code, error_code, policy_codes, retryable_codes)`, and `safe_message(message, secrets=())`.

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_generation_result.py` with these concrete cases:

```python
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generation_result import (
    FATAL_STATUSES,
    FALLBACK_STATUSES,
    GenerationResult,
    GenerationStatus,
    classify_http_failure,
    safe_message,
)


class GenerationResultTests(unittest.TestCase):
    def test_status_partition_is_complete_and_disjoint(self):
        all_failures = set(GenerationStatus) - {GenerationStatus.SUCCESS}
        self.assertEqual(FALLBACK_STATUSES | FATAL_STATUSES, all_failures)
        self.assertFalse(FALLBACK_STATUSES & FATAL_STATUSES)

    def test_result_serializes_stable_public_fields(self):
        result = GenerationResult(
            GenerationStatus.SUCCESS,
            "openai",
            "host",
            "/workspace/out/slide.png",
            "generated",
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.can_fallback)
        self.assertEqual(
            json.loads(result.to_json()),
            {
                "status": "success",
                "provider": "openai",
                "channel": "host",
                "output_path": "/workspace/out/slide.png",
                "safe_message": "generated",
            },
        )

    def test_only_fallback_statuses_allow_fallback(self):
        for status in GenerationStatus:
            result = GenerationResult(status, "gemini", "api")
            self.assertEqual(result.can_fallback, status in FALLBACK_STATUSES)

    def test_http_classifier_uses_codes_before_messages(self):
        policy = {"content_policy_violation"}
        retryable = {"insufficient_quota"}
        self.assertEqual(
            classify_http_failure(400, "content_policy_violation", policy, retryable),
            GenerationStatus.POLICY_REFUSED,
        )
        self.assertEqual(
            classify_http_failure(400, "insufficient_quota", policy, retryable),
            GenerationStatus.RETRYABLE_EXHAUSTED,
        )
        self.assertEqual(
            classify_http_failure(401, None, policy, retryable),
            GenerationStatus.AUTH_UNAVAILABLE,
        )
        self.assertEqual(
            classify_http_failure(429, None, policy, retryable),
            GenerationStatus.RETRYABLE_EXHAUSTED,
        )
        self.assertEqual(
            classify_http_failure(400, None, policy, retryable),
            GenerationStatus.INVALID_INPUT,
        )

    def test_safe_message_removes_known_and_key_like_secrets(self):
        result = safe_message(
            "Bearer known-key and sk-remoteSecret123",
            secrets=("known-key",),
        )
        self.assertNotIn("known-key", result)
        self.assertNotIn("sk-remoteSecret123", result)
        self.assertLessEqual(len(result), 300)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract tests and confirm RED**

Run:

```bash
python3 -m unittest tests/test_generation_result.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'generation_result'`.

- [ ] **Step 3: Implement the minimal shared contract**

Create `scripts/generation_result.py` with this public shape:

```python
from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
from typing import AbstractSet, Iterable, Optional


class GenerationStatus(str, Enum):
    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    AUTH_UNAVAILABLE = "auth_unavailable"
    RETRYABLE_EXHAUSTED = "retryable_exhausted"
    POLICY_REFUSED = "policy_refused"
    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    LOCAL_FAILURE = "local_failure"


FALLBACK_STATUSES = frozenset({
    GenerationStatus.UNAVAILABLE,
    GenerationStatus.AUTH_UNAVAILABLE,
    GenerationStatus.RETRYABLE_EXHAUSTED,
})
FATAL_STATUSES = frozenset(set(GenerationStatus) - FALLBACK_STATUSES - {
    GenerationStatus.SUCCESS,
})


@dataclass(frozen=True)
class GenerationResult:
    status: GenerationStatus
    provider: str
    channel: str
    output_path: Optional[str] = None
    safe_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status is GenerationStatus.SUCCESS

    @property
    def can_fallback(self) -> bool:
        return self.status in FALLBACK_STATUSES

    def to_json(self) -> str:
        payload = asdict(self)
        payload["status"] = self.status.value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def classify_http_failure(
    status_code: int,
    error_code: Optional[str],
    policy_codes: AbstractSet[str],
    retryable_codes: AbstractSet[str],
) -> GenerationStatus:
    normalized = error_code.strip().lower() if isinstance(error_code, str) else ""
    if normalized in policy_codes:
        return GenerationStatus.POLICY_REFUSED
    if normalized in retryable_codes:
        return GenerationStatus.RETRYABLE_EXHAUSTED
    if status_code in (401, 403):
        return GenerationStatus.AUTH_UNAVAILABLE
    if status_code == 429 or status_code >= 500:
        return GenerationStatus.RETRYABLE_EXHAUSTED
    return GenerationStatus.INVALID_INPUT


def safe_message(message: object, secrets: Iterable[str] = ()) -> str:
    text = str(message)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer[ ]+[A-Za-z0-9._~+/-]+", "Bearer [REDACTED]", text)
    return text[:300]
```

Also validate in `GenerationResult.__post_init__` that `provider` is one of `openai`, `gemini`, `doubao`; `channel` is `host` or `api`; only success may carry `output_path`; and success must carry an absolute output path. Add exact rejection tests before adding those checks.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run:

```bash
python3 -m unittest tests/test_generation_result.py -v
python3 -m unittest tests/test_gen_slide.py tests/test_legacy_generators.py -v
git diff --check
```

Expected: all tests PASS and whitespace check produces no output.

- [ ] **Step 5: Commit the shared result contract**

```bash
git add scripts/generation_result.py tests/test_generation_result.py
git commit -m "feat: add structured generation results"
```

---

### Task 2: Safe Host Artifact Import and Workspace Containment

**Files:**
- Modify: `scripts/image_output.py`
- Create: `scripts/import_host_image.py`
- Create: `tests/test_import_host_image.py`
- Modify: `tests/test_resource_boundaries.py`
- Modify: `tests/test_parent_identity.py`

**Interfaces:**
- Consumes: `GenerationResult`, existing `PreparedTarget`, `preflight_output`, `publish_bytes`, output locks, and host artifacts already returned or materialized by a host tool.
- Produces: `validate_image_bytes(data, target) -> LoadedImage`, `read_bounded_image_stream(stream, max_bytes=MAX_IMAGE_BYTES) -> bytes`, `prepare_workspace_target(target, workspace_root) -> PreparedTarget`, `HostArtifact`, `HostArtifactKind`, and `import_host_artifact(artifact, out_path, workspace_root, provider, overwrite=False) -> GenerationResult`.

- [ ] **Step 1: Add failing byte-validation and containment tests**

Add these focused cases to `tests/test_resource_boundaries.py` and `tests/test_parent_identity.py`:

```python
def test_validate_image_bytes_requires_real_suffix_and_strict_16_by_9(self):
    jpeg = image_bytes("JPEG", (160, 90))
    loaded = image_output.validate_image_bytes(jpeg, "slide.jpg")
    self.assertEqual((loaded.width, loaded.height, loaded.image_format), (160, 90, "JPEG"))
    with self.assertRaisesRegex(image_output.ImageOutputError, "does not match"):
        image_output.validate_image_bytes(jpeg, "slide.png")
    with self.assertRaisesRegex(image_output.ImageOutputError, "exactly 16:9"):
        image_output.validate_image_bytes(image_bytes("JPEG", (160, 100)), "slide.jpg")

def test_workspace_target_rejects_escape_before_file_creation(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "workspace"
        root.mkdir()
        with self.assertRaisesRegex(image_output.ImageOutputError, "workspace"):
            image_output.prepare_workspace_target("../escape.jpg", root)
        self.assertFalse((Path(temp_dir) / "escape.jpg").exists())
```

Add separate tests for an ancestor symlink resolving outside the root, final-target symlink, CWD change after root capture, and parent directory device/inode replacement before publication.

- [ ] **Step 2: Run the new helper tests and confirm RED**

Run:

```bash
python3 -m unittest \
  tests/test_resource_boundaries.py \
  tests/test_parent_identity.py -v
```

Expected: FAIL because the three new helper functions do not exist.

- [ ] **Step 3: Implement reusable image and workspace helpers**

Refactor `scripts/image_output.py` so byte and temporary-file validation share one exact format/dimension check:

```python
def validate_image_bytes(data: bytes, target: object) -> LoadedImage:
    if not isinstance(data, bytes) or not data:
        raise ImageOutputError("image data must be non-empty bytes")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageOutputError(
            f"image data exceeds maximum size of {MAX_IMAGE_BYTES} bytes"
        )
    loaded = _load_image_data(data, verify=True, copy_image=False)
    expected = PIL_FORMATS[output_format(str(target))]
    if loaded.image_format != expected:
        raise ImageOutputError(
            f"image format {loaded.image_format or '<unknown>'} does not match "
            f"{Path(str(target)).suffix.lower()}"
        )
    if loaded.width * 9 != loaded.height * 16:
        raise ImageOutputError(
            f"generated image must be exactly 16:9; received "
            f"{loaded.width}x{loaded.height}"
        )
    return loaded
```

Extract the existing Pillow loading body into `_load_image_data(data, verify=False, copy_image=True) -> LoadedImage`, make `load_image(path, verify=False, copy_image=True)` call it, and make `_validate_temp(path, target)` call `validate_image_bytes(data, target)`. Add `read_bounded_image_stream(stream, max_bytes=MAX_IMAGE_BYTES) -> bytes` using only sized reads, the existing 50 MiB limit, `Content-Length` preflight, and a bounded read count.

Implement workspace containment before `prepare_target(target)` creates directories:

```python
def prepare_workspace_target(target: object, workspace_root: object) -> PreparedTarget:
    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ImageOutputError("workspace root must be an existing directory")
    raw = Path(target).expanduser()
    if ".." in raw.parts:
        raise ImageOutputError("output target must not traverse outside workspace")
    absolute = raw if raw.is_absolute() else root / raw
    resolved = resolve_output_path(absolute, base=root)
    if resolved.parent != root and root not in resolved.parent.parents:
        raise ImageOutputError("output target escapes workspace root")
    prepared = prepare_target(resolved)
    if prepared.parent.real_path != root and root not in prepared.parent.real_path.parents:
        raise ImageOutputError("resolved output parent escapes workspace root")
    return prepared
```

Reject a final symlink in preflight, retain parent identity checks at temporary creation, validation, and publication, and preserve all existing output-lock behavior.

- [ ] **Step 4: Write the failing host importer tests**

Create `tests/test_import_host_image.py` covering the four supported artifact kinds and fatal failure boundaries:

```python
def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class HostImageImportTests(unittest.TestCase):
    def test_local_file_is_validated_and_published_inside_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "host-source.jpg"
            source.write_bytes(image_bytes("JPEG", (160, 90)))
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(source)),
                "out/slide.jpg",
                str(root),
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertEqual(Path(result.output_path), root / "out/slide.jpg")

    def test_invalid_force_input_does_not_replace_existing_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            target.write_bytes(image_bytes("JPEG", (160, 90)))
            before = target.read_bytes()
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(b"not-an-image", "image/jpeg"),
                "slide.jpg",
                str(root),
                provider="openai",
                overwrite=True,
            )
            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(root.glob(".slide.jpg.*.tmp")), [])
```

Also add explicit cases for Base64, `data:` URL, MIME mismatch, oversized encoded data, ambiguous/unsupported artifact kinds, local source read failure, workspace escape, target lock contention, parent replacement, and JSON CLI output with no secret echo.

- [ ] **Step 5: Run importer tests and confirm RED**

Run:

```bash
python3 -m unittest tests/test_import_host_image.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'import_host_image'`.

- [ ] **Step 6: Implement the host importer and local-path CLI**

Create `scripts/import_host_image.py` with these exact public interfaces:

```python
class HostArtifactKind(str, Enum):
    LOCAL_PATH = "local_path"
    INLINE_BYTES = "inline_bytes"
    BASE64 = "base64"
    DATA_URL = "data_url"


@dataclass(frozen=True)
class HostArtifact:
    kind: HostArtifactKind
    value: Union[str, bytes]
    mime_type: Optional[str] = None

    @classmethod
    def local_path(cls, path: str) -> "HostArtifact":
        return cls(HostArtifactKind.LOCAL_PATH, path)

    @classmethod
    def inline_bytes(cls, data: bytes, mime_type: str) -> "HostArtifact":
        return cls(HostArtifactKind.INLINE_BYTES, data, mime_type)
```

Define `HostArtifact.base64(encoded, mime_type)` and `HostArtifact.data_url(value)` with the same concrete constructor pattern. Implement `import_host_artifact(artifact, out_path, workspace_root, provider, overwrite=False) -> GenerationResult` using the ordered algorithm below.

Implementation order inside `import_host_artifact(artifact, out_path, workspace_root, provider, overwrite=False)`:

1. Validate `provider`, `overwrite`, workspace root, target path, suffix, and output lock before reading the host source.
2. Convert the supported artifact to bytes with the shared 50 MiB limit. For a local path, freeze the absolute source, require a regular file, read through an opened descriptor with a bounded length, and reject identity changes.
3. Validate MIME when supplied, actual encoding, suffix, 64 MP, and strict 16:9 before publication.
4. Call `publish_bytes(data, target, overwrite=overwrite)` with the already-validated bytes and captured target.
5. Return `invalid_output` for artifact decoding/format problems and `local_failure` for source I/O, lock, workspace, temporary-write, or publication problems.

The CLI accepts only a local source path so Base64 never appears in process arguments:

```text
import_host_image.py SOURCE OUTPUT --workspace-root ROOT --provider openai|gemini|doubao [--force] [--json]
```

Default mode prints a short human message and returns 0/1. `--json` writes exactly one `GenerationResult` JSON object to stdout and sends no diagnostic text to stdout.

- [ ] **Step 7: Run importer and existing publication tests**

Run:

```bash
python3 -m unittest \
  tests/test_import_host_image.py \
  tests/test_resource_boundaries.py \
  tests/test_parent_identity.py \
  tests/test_provider_output_safety.py -v
python3 scripts/import_host_image.py --help
git diff --check
```

Expected: all tests PASS; help exits 0; whitespace check is empty.

- [ ] **Step 8: Commit the host import boundary**

```bash
git add \
  scripts/image_output.py \
  scripts/import_host_image.py \
  tests/test_import_host_image.py \
  tests/test_resource_boundaries.py \
  tests/test_parent_identity.py
git commit -m "feat: safely import host generated images"
```

---

### Task 3: Structured OpenAI API Adapter

**Files:**
- Modify: `scripts/gen_slide_openai.py`
- Create: `tests/test_gen_slide_openai_result.py`
- Modify: `tests/test_gen_slide_openai.py`

**Interfaces:**
- Consumes: `GenerationResult`, shared HTTP classifier, byte validation/publication helpers, existing credential lookup, retry delay, and output locks.
- Produces: `generate_result(prompt, out_path, retries=2, overwrite=False, progress=None) -> GenerationResult`; preserves `gen(prompt, out_path, retries=2, overwrite=False) -> bool`.

- [ ] **Step 1: Write the failing OpenAI result matrix**

Create a table-driven `tests/test_gen_slide_openai_result.py` that asserts:

```python
OPENAI_CASES = (
    (401, "invalid_api_key", GenerationStatus.AUTH_UNAVAILABLE),
    (403, "permission_denied", GenerationStatus.AUTH_UNAVAILABLE),
    (400, "content_policy_violation", GenerationStatus.POLICY_REFUSED),
    (400, "moderation_blocked", GenerationStatus.POLICY_REFUSED),
    (429, "rate_limit_exceeded", GenerationStatus.RETRYABLE_EXHAUSTED),
    (400, "insufficient_quota", GenerationStatus.RETRYABLE_EXHAUSTED),
    (500, "server_error", GenerationStatus.RETRYABLE_EXHAUSTED),
    (400, "invalid_size", GenerationStatus.INVALID_INPUT),
)
```

For every case, construct an `HTTPError` with `{"error":{"code": error_code,"message":"safe"}}`, patch `urlopen`, call `generate_result("prompt", output_path, retries=0)`, assert the status, and assert no output file. Add separate tests for missing/invalid key without network, exhausted `URLError`, malformed JSON, invalid Base64, valid but wrong-format bytes, non-16:9 bytes, local publication failure after valid byte validation, success, and secret redaction.

- [ ] **Step 2: Run the OpenAI result tests and confirm RED**

Run:

```bash
python3 -m unittest tests/test_gen_slide_openai_result.py -v
```

Expected: FAIL because `generate_result` does not exist.

- [ ] **Step 3: Refactor OpenAI around a silent structured core**

Implement these constants and signatures:

```python
from typing import Callable, Optional

OPENAI_POLICY_CODES = frozenset({
    "content_policy_violation",
    "moderation_blocked",
})
OPENAI_RETRYABLE_CODES = frozenset({
    "billing_hard_limit_reached",
    "insufficient_quota",
    "rate_limit_exceeded",
})


def generate_result(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    return _generate_result_with_lock(
        prompt,
        out_path,
        retries,
        overwrite,
        progress,
    )
```

Define `_generate_result_with_lock(prompt, out_path, retries, overwrite, progress) -> GenerationResult` in the same module as the lock-owning implementation of the seven ordered requirements below; it is not a second public interface.

Required implementation order:

1. Validate prompt is non-empty text, retries, overwrite, suffix, target, and lock. Map argument errors to `invalid_input`; map target/lock/filesystem errors to `local_failure`.
2. Load and validate the key. Missing, malformed, 401, and non-policy 403 map to `auth_unavailable` without echoing the key.
3. Parse structured `error.code`; classify it before falling back to HTTP status.
4. Retry 429, 5xx, and transport failures internally. Return `retryable_exhausted` only after the configured attempts are exhausted.
5. Map malformed JSON/Base64 or an image failing shared validation to `invalid_output`.
6. Validate bytes before `publish_bytes(image, target, overwrite=overwrite)`; any later target write or atomic publication failure is `local_failure`.
7. Return an absolute `output_path` only after atomic publication succeeds.

Make `gen(prompt, out_path, retries=2, overwrite=False)` call `generate_result(prompt, out_path, retries=retries, overwrite=overwrite, progress=print)`, print the final safe message in the existing human style, and return `result.ok`. Keep `main()` returning 0/1.

- [ ] **Step 4: Run OpenAI focused and compatibility tests**

Run:

```bash
python3 -m unittest \
  tests/test_gen_slide_openai_result.py \
  tests/test_gen_slide_openai.py \
  tests/test_retry_after.py \
  tests/test_provider_output_safety.py \
  tests/test_provider_contract_round4.py -v
git diff --check
```

Expected: all tests PASS; existing boolean output protection and retry behavior remain intact.

- [ ] **Step 5: Commit the OpenAI adapter**

```bash
git add \
  scripts/gen_slide_openai.py \
  tests/test_gen_slide_openai_result.py \
  tests/test_gen_slide_openai.py
git commit -m "feat: classify openai generation results"
```

---

### Task 4: Structured Gemini API Adapter

**Files:**
- Modify: `scripts/gen_slide_gemini.py`
- Create: `tests/test_gen_slide_gemini_result.py`
- Modify: `tests/test_legacy_generators.py`
- Modify: `tests/test_provider_output_safety.py`

**Interfaces:**
- Consumes: the Task 1 result contract and Task 2 byte validation/publication helpers.
- Produces: Gemini `generate_result(prompt, out_path, retries=2, overwrite=False, progress=None) -> GenerationResult`; preserves `gen(prompt, out_path, retries=2, overwrite=False) -> bool` and current PNG/JPEG MIME rules.

- [ ] **Step 1: Write the failing Gemini result matrix**

Create `tests/test_gen_slide_gemini_result.py` with HTTP cases parallel to OpenAI plus these payload refusals:

```python
POLICY_PAYLOADS = (
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": [{"finishReason": "SAFETY"}]},
    {"candidates": [{"finishReason": "BLOCKLIST"}]},
    {"candidates": [{"finishReason": "PROHIBITED_CONTENT"}]},
    {"candidates": [{"finishReason": "IMAGE_SAFETY"}]},
)
```

Assert every payload returns `policy_refused`, stops without trying another request, and publishes nothing. Test missing key, 401/403, 429/5xx, network exhaustion, unsupported suffix before key/network, malformed envelope, missing image, declared MIME mismatch, invalid image bytes, local publication failure, success, and redaction.

- [ ] **Step 2: Run Gemini tests and confirm RED**

Run:

```bash
python3 -m unittest tests/test_gen_slide_gemini_result.py -v
```

Expected: FAIL because the Gemini provider has no structured entrypoint or policy-result handling.

- [ ] **Step 3: Implement Gemini structured classification**

Add:

```python
GEMINI_POLICY_REASONS = frozenset({
    "BLOCKLIST",
    "IMAGE_SAFETY",
    "PROHIBITED_CONTENT",
    "SAFETY",
})
GEMINI_POLICY_CODES = frozenset({"blocked", "safety"})
GEMINI_RETRYABLE_CODES = frozenset({
    "quota_exceeded",
    "rate_limit_exceeded",
    "resource_exhausted",
})
```

Add `generate_result(prompt, out_path, retries=2, overwrite=False, progress=None) -> GenerationResult` with the same ordering as Task 3. Check `promptFeedback.blockReason` and `candidate.finishReason` before treating a missing image as invalid output. Keep provider-declared MIME validation. Split byte validation from publication so invalid provider content maps to `invalid_output` and local publication maps to `local_failure`. Keep `gen(prompt, out_path, retries=2, overwrite=False) -> bool` as the human-rendering boolean wrapper.

- [ ] **Step 4: Run Gemini focused and shared provider tests**

Run:

```bash
python3 -m unittest \
  tests/test_gen_slide_gemini_result.py \
  tests/test_legacy_generators.py \
  tests/test_provider_output_safety.py \
  tests/test_retry_after.py -v
git diff --check
```

Expected: all tests PASS with no network access outside mocks.

- [ ] **Step 5: Commit the Gemini adapter**

```bash
git add \
  scripts/gen_slide_gemini.py \
  tests/test_gen_slide_gemini_result.py \
  tests/test_legacy_generators.py \
  tests/test_provider_output_safety.py
git commit -m "feat: classify gemini generation results"
```

---

### Task 5: Structured Doubao API and Download Adapter

**Files:**
- Modify: `scripts/gen_slide_doubao.py`
- Create: `tests/test_gen_slide_doubao_result.py`
- Modify: `tests/test_retry_after.py`
- Modify: `tests/test_provider_output_safety.py`

**Interfaces:**
- Consumes: Task 1 results, Task 2 bounded stream/byte validation, existing Doubao request/download behavior.
- Produces: Doubao `generate_result(prompt, out_path, retries=2, overwrite=False, progress=None) -> GenerationResult`; preserves `gen(prompt, out_path, retries=2, overwrite=False) -> bool`.

- [ ] **Step 1: Write the failing two-stage Doubao matrix**

Create `tests/test_gen_slide_doubao_result.py` covering both generation and returned-image download:

```python
DOUBAO_POLICY_CODES = (
    "content_filter",
    "content_policy_violation",
    "input_text_risk",
    "output_image_risk",
)
```

For the generation request, assert missing/invalid key and 401/403 are `auth_unavailable`; 429/5xx/network exhaustion are `retryable_exhausted`; listed structured policy codes are `policy_refused`; other 4xx are `invalid_input`; malformed URL/envelope is `invalid_output`.

For image download, assert 401/403 is `auth_unavailable`, 429/5xx/network exhaustion is `retryable_exhausted`, a policy code is `policy_refused`, invalid/oversized/non-16:9/wrong-format bytes are `invalid_output`, and temporary/publish failures are `local_failure`. Each case must verify whether fallback is allowed and that invalid `force` content preserves the existing output.

- [ ] **Step 2: Run Doubao tests and confirm RED**

Run:

```bash
python3 -m unittest tests/test_gen_slide_doubao_result.py -v
```

Expected: FAIL because `_download_image` and `_gen_owned` return only optional byte counts/booleans.

- [ ] **Step 3: Implement one structured result across both remote stages**

Add provider-specific policy/retryable code sets and `generate_result(prompt, out_path, retries=2, overwrite=False, progress=None) -> GenerationResult`. Replace the optional integer download result with:

```python
from typing import Callable, Optional

from image_output import PreparedTarget


def _download_image_result(
    image_url: str,
    target: PreparedTarget,
    retries: int,
    overwrite: bool,
    key: str,
    progress: Optional[Callable[[str], None]],
) -> GenerationResult:
    return _download_with_retries(
        image_url,
        target,
        retries,
        overwrite,
        key,
        progress,
    )
```

Define `_download_with_retries(image_url, target, retries, overwrite, key, progress) -> GenerationResult` in the provider as the private implementation that applies the status mapping in Step 1 and returns the final structured result.

Download with `read_bounded_image_stream(response, max_bytes=MAX_IMAGE_BYTES)`, validate the bytes before publication, and then publish. This makes remote read failures retryable, invalid bytes fatal `invalid_output`, and local write failures fatal `local_failure`. Return one final Doubao API result from generation through download. Preserve safe redaction and the existing boolean wrapper.

- [ ] **Step 4: Run Doubao focused and regression tests**

Run:

```bash
python3 -m unittest \
  tests/test_gen_slide_doubao_result.py \
  tests/test_legacy_generators.py \
  tests/test_retry_after.py \
  tests/test_provider_output_safety.py \
  tests/test_resource_boundaries.py -v
git diff --check
```

Expected: all tests PASS, including bounded download behavior and existing output preservation.

- [ ] **Step 5: Commit the Doubao adapter**

```bash
git add \
  scripts/gen_slide_doubao.py \
  tests/test_gen_slide_doubao_result.py \
  tests/test_retry_after.py \
  tests/test_provider_output_safety.py
git commit -m "feat: classify doubao generation results"
```

---

### Task 6: API-Only Router and Machine-Readable CLI

**Files:**
- Modify: `scripts/gen_slide.py`
- Modify: `tests/test_gen_slide.py`

**Interfaces:**
- Consumes: each provider's `generate_result(prompt, out_path, retries=2, overwrite=False, progress=None)` and preserved `gen(prompt, out_path, retries=2, overwrite=False)`.
- Produces: router `generate_result(prompt, out_path, engine="openai", retries=2, overwrite=False, progress=None) -> GenerationResult`, preserved `gen(prompt, out_path, engine="openai", retries=2, overwrite=False) -> bool`, and `--json`.

- [ ] **Step 1: Write failing router result and JSON tests**

Add tests that assert:

```python
def test_generate_result_lazy_loads_provider_structured_entrypoint(self):
    expected = GenerationResult(
        GenerationStatus.AUTH_UNAVAILABLE,
        "openai",
        "api",
        safe_message="missing key",
    )
    provider = SimpleNamespace(generate_result=mock.Mock(return_value=expected))
    with mock.patch.object(gen_slide.importlib, "import_module", return_value=provider):
        result = gen_slide.generate_result("prompt", "slide.jpg")
    self.assertEqual(result, expected)

def test_json_cli_emits_one_object_and_keeps_default_exit_compatibility(self):
    result = GenerationResult(
        GenerationStatus.AUTH_UNAVAILABLE,
        "openai",
        "api",
        safe_message="missing key",
    )
    stdout = io.StringIO()
    with mock.patch.object(gen_slide, "generate_result", return_value=result), redirect_stdout(stdout):
        exit_code = gen_slide.main(["slide.jpg", "prompt", "--json"])
    self.assertEqual(exit_code, 1)
    self.assertEqual(json.loads(stdout.getvalue())["status"], "auth_unavailable")
```

Also test unknown engine → `invalid_input`, provider import failure → `unavailable`, provider exception → `local_failure` without exception text or secrets, success JSON includes absolute output path, human mode still exits 0/1, and CLI help explicitly says “API/CLI adapter” and “does not use host capabilities.”

- [ ] **Step 2: Run router tests and confirm RED**

Run:

```bash
python3 -m unittest tests/test_gen_slide.py -v
```

Expected: FAIL because `generate_result` and `--json` are absent.

- [ ] **Step 3: Implement the structured API-only router**

Implement `generate_result(prompt, out_path, engine="openai", retries=2, overwrite=False, progress=None) -> GenerationResult` with the same argument validation as the current router. Lazy-load `ENGINE_MODULES[engine]`, call its structured entrypoint, and convert import/unexpected failures into safe results. Keep `gen(prompt, out_path, engine="openai", retries=2, overwrite=False) -> bool` as a boolean compatibility wrapper around the structured result.

Add:

```python
parser.add_argument(
    "--json",
    action="store_true",
    help="emit one structured API-adapter result as JSON",
)
```

Human mode prints one safe summary and returns 0/1. JSON mode prints only `result.to_json()` and also returns 0/1, so existing shell success/failure semantics remain compatible. The description must call this an API/CLI adapter, not the host default router.

- [ ] **Step 4: Run router, provider, and CLI regression tests**

Run:

```bash
python3 -m unittest \
  tests/test_gen_slide.py \
  tests/test_gen_slide_openai_result.py \
  tests/test_gen_slide_gemini_result.py \
  tests/test_gen_slide_doubao_result.py \
  tests/test_legacy_generators.py -v
python3 scripts/gen_slide.py --help
git diff --check
```

Expected: all tests PASS; help states the API-only boundary and exits 0.

- [ ] **Step 5: Commit the API router contract**

```bash
git add scripts/gen_slide.py tests/test_gen_slide.py
git commit -m "feat: expose structured api generation results"
```

---

### Task 7: Host-First Skill Workflow, Sticky Batch Routing, and Documentation

**Files:**
- Create: `references/host-image-routing.md`
- Modify: `SKILL.md`
- Modify: `README.md`
- Modify: `tests/test_documentation.py`
- Create: `tests/test_host_routing_contract.py`

**Interfaces:**
- Consumes: host tool registry, host image-generation results, `import_host_image.py --json`, and `gen_slide.py --json`.
- Produces: the user-facing six-candidate routing behavior, optional key setup, page-level switch report, and documented API/CLI escape hatch.

- [ ] **Step 1: Write failing Skill contract tests**

Create `tests/test_host_routing_contract.py` to parse the ordered candidate list and failure table from the reference, then assert the actual invariants rather than isolated marketing phrases:

```python
import json
import re


def load_contract_block(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match is None:
        raise AssertionError("host routing reference has no JSON contract")
    return json.loads(match.group(1))


EXPECTED_CANDIDATES = [
    "host-openai",
    "api-openai",
    "host-gemini",
    "api-gemini",
    "host-doubao",
    "api-doubao",
]
EXPECTED_FALLBACK = {
    "unavailable",
    "auth_unavailable",
    "retryable_exhausted",
}
EXPECTED_FATAL = {
    "policy_refused",
    "invalid_input",
    "invalid_output",
    "local_failure",
}

def test_reference_has_one_ordered_machine_checkable_contract(self):
    contract = load_contract_block(ROOT / "references/host-image-routing.md")
    self.assertEqual(contract["candidates"], EXPECTED_CANDIDATES)
    self.assertEqual(set(contract["fallback_statuses"]), EXPECTED_FALLBACK)
    self.assertEqual(set(contract["fatal_statuses"]), EXPECTED_FATAL)
    self.assertEqual(contract["batch_mode"], "serial-sticky-monotonic")
    self.assertEqual(contract["cache_changes_route"], False)
```

The reference must contain one fenced JSON contract block that `load_contract_block(path)` extracts with `json.loads`; prose links back to that single block rather than duplicating a second order table.

Update `tests/test_documentation.py` to assert:

- `SKILL.md` links `references/host-image-routing.md` and says to read it before image generation.
- Host OpenAI is the recommended first candidate and does not require `OPENAI_API_KEY`.
- Each key is optional/skippable.
- `scripts/gen_slide.py` is labelled API/CLI-only in both documents.
- The old `ThreadPoolExecutor` generation example and “batch generate with concurrency” heading are removed.
- Batch generation is serial and cached pages do not change routing.
- Editable handoff remains a separate exact `1280×720 PNG` step.
- The three API-key setup commands remain valid but are under an optional API/CLI section.

- [ ] **Step 2: Run documentation tests and confirm RED**

Run:

```bash
python3 -m unittest \
  tests/test_host_routing_contract.py \
  tests/test_documentation.py -v
```

Expected: FAIL because the reference does not exist and current docs still require an OpenAI key/default concurrent API generation.

- [ ] **Step 3: Write the detailed host routing reference**

Create `references/host-image-routing.md` with these sections and exact decisions:

1. One fenced JSON contract with the candidate/status/batch/cache values above.
2. Runtime capability discovery: only a currently callable tool/connector counts; browser login, documentation, and guessed account state do not. A missing or disconnected tool produces `unavailable` without invoking an API.
3. Optional credential setup: offer once when useful, every key can be skipped, and non-interactive missing keys return `auth_unavailable`.
4. Host execution: call the host tool directly. Accept its explicit primary/result image, or its only image when exactly one exists; multiple unmarked images are `invalid_output`. Accept an absolute local path, MIME-labelled inline data, or a resource that a host export/download tool can materialize. Reject bare URLs, UI-only previews, ambiguous text, and unavailable resource handles. Import the resulting local artifact through `import_host_image.py --json`.
5. API execution: run `python3 scripts/gen_slide.py OUTPUT PROMPT --engine PROVIDER --json`; do not inspect or print key contents.
6. Result handling: continue only for the three fallback statuses; stop for all fatal statuses. During resource materialization, authorization failure is `auth_unavailable`, exhausted timeout/429/5xx is `retryable_exhausted`, an explicit safety refusal is `policy_refused`, missing/invalid returned content is `invalid_output`, and workspace writing/publication failure is `local_failure`.
7. Sticky deck state: process page numbers serially; first success selects the candidate; later fallback moves only forward; never regenerate a success; cached files never change state.
8. Final report: list pages, chosen channel/provider, every switch page, and an ordered redacted summary if all candidates are exhausted.
9. Editable handoff: keep the master and invoke `prepare_editable_input.py` for a separate exact `1280×720 PNG`.

Include the concrete API command template:

```bash
python3 scripts/gen_slide.py \
  "${slide_output}" \
  "${slide_prompt}" \
  --engine "${provider}" \
  --json
```

Include the concrete host-import template using previously captured absolute workspace and source paths, not unresolved `$HOME` or globs:

```bash
python3 scripts/import_host_image.py \
  "/absolute/host/generated/image.png" \
  "out/slide_01.png" \
  --workspace-root "/absolute/workspace" \
  --provider openai \
  --json
```

- [ ] **Step 4: Update the Skill entrypoint with progressive disclosure**

Change the frontmatter description so it identifies host-first GPT image generation and API fallbacks without claiming an API key is required.

Replace the existing single-slide and concurrent-batch sections with a compact workflow:

```markdown
## Image generation routing

Before generating any slide image, read
[references/host-image-routing.md](references/host-image-routing.md) and follow
its candidate order, failure boundaries, artifact-import contract, and serial
sticky batch rules.

- Prefer the host's callable OpenAI image tool; it does not require an
  `OPENAI_API_KEY`.
- Then try the matching OpenAI API adapter only when configured or after the
  user chooses to configure it. Configuration may always be skipped.
- Apply the same host-then-key rule to Gemini and Doubao.
- Do not use browser automation, cookies, or extracted host tokens.
- Keep generated masters; create exact 1280×720 PNG files only for the
  editable-converter handoff.
```

Keep the prompt style, quantity constraints, visual-check, export, safety, and historical sections that remain relevant. Remove concurrency guidance for image generation; do not remove export recovery or output lock documentation.

- [ ] **Step 5: Update README for two explicit entrypoints**

Rewrite Setup and Quick Start so the recommended path is the Skill using host capabilities with no required key. Move API-key commands under “Optional API/CLI credentials” and label `gen_slide.py` an API-only adapter. Explain that the Skill handles automatic six-candidate fallback while the low-level CLI calls only the selected API provider.

Keep the exact credential file commands, provider formats/models, output limits, export recovery guarantees, and editable standardization documentation.

- [ ] **Step 6: Run Skill/documentation tests and validator**

Run:

```bash
python3 -m unittest \
  tests/test_host_routing_contract.py \
  tests/test_documentation.py \
  tests/test_validate_skill.py -v
python3 scripts/validate_skill.py .
git diff --check
```

Expected: all tests PASS; validator prints `Skill is valid!`; whitespace check is empty.

- [ ] **Step 7: Commit the host-first Skill workflow**

```bash
git add \
  references/host-image-routing.md \
  SKILL.md \
  README.md \
  tests/test_documentation.py \
  tests/test_host_routing_contract.py
git commit -m "feat: route slide images through host capabilities"
```

---

### Task 8: Full Regression, Mocked Six-Candidate E2E, and Host Forward-Test

**Files:**
- Create: `tests/test_host_routing_e2e.py`
- Modify only if a test exposes a real defect: the smallest owning source/test/document file from Tasks 1–7.

**Interfaces:**
- Consumes: the completed structured adapters, importer, Skill reference, and host built-in image generation capability.
- Produces: fresh evidence that API routing, host import, sticky behavior, converter standardization, CLI surfaces, and the real no-key host path work together.

- [ ] **Step 1: Write a mocked six-candidate transcript test**

Create `tests/test_host_routing_e2e.py` with a small test-only executor that reads the reference contract and feeds controlled `GenerationResult` objects through it. Cover these exact transcripts:

```python
from generation_result import FALLBACK_STATUSES, GenerationStatus
from tests.test_host_routing_contract import load_contract_block


def run_page(candidates, outcomes, start_index=0):
    calls = []
    for index in range(start_index, len(candidates)):
        candidate = candidates[index]
        status = GenerationStatus(outcomes[candidate])
        calls.append((candidate, status.value))
        if status is GenerationStatus.SUCCESS:
            return "success", index, calls
        if status not in FALLBACK_STATUSES:
            return "fatal", index, calls
    return "exhausted", len(candidates), calls


TRANSCRIPTS = {
    "host_openai_wins_over_configured_key": [
        ("host-openai", "success"),
    ],
    "openai_api_after_missing_host": [
        ("host-openai", "unavailable"),
        ("api-openai", "success"),
    ],
    "gemini_host_after_openai_exhaustion": [
        ("host-openai", "unavailable"),
        ("api-openai", "auth_unavailable"),
        ("host-gemini", "success"),
    ],
    "doubao_api_last_resort": [
        ("host-openai", "unavailable"),
        ("api-openai", "auth_unavailable"),
        ("host-gemini", "unavailable"),
        ("api-gemini", "retryable_exhausted"),
        ("host-doubao", "unavailable"),
        ("api-doubao", "success"),
    ],
}
```

For each transcript, build the `outcomes` mapping, call `run_page(candidates, outcomes, start_index)`, and assert `calls` equals the listed ordered pairs. Add a three-page sticky case where page 1 succeeds on host OpenAI, page 2 gets `retryable_exhausted` then succeeds on API OpenAI, and page 3 calls `run_page(candidates, outcomes, api_openai_index)` with the API OpenAI index. Assert page 1 is never invoked again and the switch report identifies page 2. Add fatal cases for each fatal status and an all-exhausted aggregate with six redacted reasons.

- [ ] **Step 2: Run the E2E contract test and confirm behavior**

Run:

```bash
python3 -m unittest tests/test_host_routing_e2e.py -v
```

Expected: all transcript and sticky/fatal cases PASS. If the test fails because the reference is ambiguous, fix the reference first rather than weakening the assertions.

- [ ] **Step 3: Run the complete automated regression on both supported Python runtimes**

Run with the configured Python 3.9 and Python 3.12 executables discovered locally:

```bash
python3.9 -m unittest discover -s tests -v
python3.12 -m unittest discover -s tests -v
```

Expected: the complete suite passes on both; only already-documented platform/version skips are allowed. Do not claim exact counts until these fresh runs finish.

- [ ] **Step 4: Run all CLI and package validation smoke checks**

Run:

```bash
python3 scripts/validate_skill.py .
python3 scripts/gen_slide.py --help
python3 scripts/gen_slide_openai.py --help
python3 scripts/gen_slide_gemini.py --help
python3 scripts/gen_slide_doubao.py --help
python3 scripts/import_host_image.py --help
python3 scripts/prepare_editable_input.py --help
python3 scripts/vision_check_gemini.py --help
python3 scripts/export_images.py --help
python3 scripts/cleanup_output_locks.py --help
git diff --check
git status --short --branch
```

Expected: validator succeeds; all help commands exit 0 without traceback; whitespace check is empty; only intentional implementation files are changed.

This repository has no browser frontend or separate web backend. Its interaction surfaces are the host Skill/tool flow, Python library calls, and CLIs; the host forward-test below is the UI-level interaction check for this feature.

- [ ] **Step 5: Perform an independent no-key host OpenAI forward-test**

Use a fresh subagent and an isolated `mktemp -d` workspace outside the repository. Give it only this realistic request and the Skill path:

```text
Use the ai-image-to-ppt Skill at
/Users/neomei/项目/codexprojects/AI-Image-to-ppt to generate one simple 16:9
educational slide image in an isolated workspace. Do not configure or use any
API key. Use the host's built-in OpenAI image capability, import the selected
artifact into the isolated workspace, then create the separate exact 1280x720
PNG editable-converter input. Report the host tool used, master path and actual
format/dimensions, standardized path and actual format/dimensions, and whether
any API adapter ran.
```

Inspect the actual files with Pillow after the subagent returns. Expected:

- Built-in host OpenAI runs without `OPENAI_API_KEY`.
- No API adapter is invoked.
- The master is inside the isolated workspace, decodes, matches its suffix, and is strict 16:9.
- The separate converter input is a real PNG at exactly `1280×720`.
- No artifact is left referenced only under `$CODEX_HOME/generated_images`.

If the built-in host returns a non-16:9 or non-materializable artifact, preserve that evidence and stop: the confirmed fatal `invalid_output` boundary forbids silent crop, browser download, or cross-provider fallback. Resolve the product contract explicitly before claiming host support.

- [ ] **Step 6: Repeat review and regression after every discovered fix**

For each real defect found in Steps 2–5:

1. Add the smallest failing regression test.
2. Run it and observe the expected failure.
3. Fix only the owning code or instruction.
4. Re-run the focused test.
5. Re-run Steps 2–5 from the first affected layer onward.
6. Commit the focused fix with a `fix:` message that names the failed contract only after the relevant suite is green.

Stop only after one complete fresh pass finds no remaining defect worth fixing.

- [ ] **Step 7: Commit the final E2E test and report local state**

```bash
git add tests/test_host_routing_e2e.py
git commit -m "test: cover host-first routing end to end"
git status --short --branch
git log --oneline --decorate -10
```

Expected: working tree clean; local branch and its ahead/behind state are reported separately from remote publication. Do not push or publish without explicit authorization.
