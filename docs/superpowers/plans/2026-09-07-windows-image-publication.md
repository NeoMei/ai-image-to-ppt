# Windows Image Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Enable native Windows API generation and host artifact import without weakening output ownership protection or regressing POSIX execution.

**Architecture:** Keep the existing POSIX backend unchanged. Add a Windows backend using native handles for identity-bound rename/delete and directory pinning. Dispatch at the existing publication boundary; keep image validation, GenerationResult, routing, and paired host rollback contracts.

**Tech Stack:** Python 3.9+, standard-library ctypes/msvcrt, Pillow, python-pptx, unittest.

**Spec:** User-approved scope in this task: implement locally, preserve Mac behavior, validate Windows separately; only produce PPT files, no Office/WPS app integration.

## Global Constraints

- Python 3.9+; no new runtime dependency.
- Preserve image limits, strict API 16:9 validation, host 0.5% crop tolerance, raw artifact preservation and output format contracts.
- Never silently overwrite or delete an externally replaced file, including during cleanup and rollback.
- Unknown or unrecoverable files must remain intact with a complete recovery locator.
- POSIX publication continues through the existing implementation.
- Native Windows support uses handle-bound mutations; no check-then-pathname-replace/unlink fallback.
- File generation and host raw/master import have process-time rollback, not a crash-atomic or power-loss guarantee.
- Windows native tests must be explicitly skipped on other platforms, never reported as Windows verification.
- No Office/WPS-specific generation path, UI, add-in, new provider, push, or release.

## Task 1: Windows publication backend and integration

**Files:** Create `scripts/windows_image_output.py` (split native handle wrapper if necessary), `tests/test_windows_image_output.py`; modify `scripts/image_output.py`, and only if needed `scripts/import_host_image.py`, `scripts/run_windows_tests.py`.

**Interfaces:** Existing `preflight_output`, `_publish_image_bytes`, `remove_output_with_identity`, `PublishedOutput`, and `GenerationResult` stay compatible. A lazily imported Windows backend provides preflight, validated publication and identity-bound removal. Read current host rollback callers before choosing a narrow backend interface.

- [x] Write a failing regression for selecting a Windows backend when POSIX primitives are unavailable; execute it and record expected failure.
- [x] Add Windows native integration tests using TemporaryDirectory and real bytes: create, refuse overwrite, explicit overwrite, mismatched identity, concurrent target creation, controlled write/publish failure, rollback retention, Unicode paths, and parent/reparse protection. Keep these Windows-only and supplement branch/transaction tests runnable on Mac.
- [x] Implement CreateFileW pinning and SetFileInformationByHandle rename/delete with explicit ctypes signatures. Reject unexpected reparse points and unsupported native operations. Hold handles until mutation/recovery completes. Use non-replacing publication and recovery. On Windows retain old output in a unique sibling `.image-output-recovery-<32hex>.entry` file, atomically claimed by handle rename; do not create/delete a recovery directory. POSIX keeps its directory/entry layout.
- [x] Route the existing publication API and host rollback through the backend without editing provider behavior.
- [x] Run focused tests, then the existing complete suite. Update the Windows runner so the new native tests and provider/import happy paths actually run there, retaining justified POSIX-only exclusions.
- [x] Write RED/GREEN evidence and limitations to the assigned report. GitHub handoff authorized: commit the reviewed candidate using the repository's existing author identity via per-command configuration, without changing global Git configuration; push the isolated branch for Windows validation.
- [x] Independent task review: spec compliance and code quality; fix important findings and re-review the fixes.

## Task 2: Documentation and final acceptance

**Files:** README.md, SKILL.md, references/host-image-routing.md if required; `docs/windows-validation.md`.

**Interfaces:** Documentation describes the actual platform dispatch and failure behavior delivered in Task 1.

- [x] Remove obsolete blanket Windows fail-closed statements; retain explicit unsupported-filesystem and open-file failure behavior.
- [x] Document PowerShell installation/test commands and Windows verification: API adapter with fixture response, real host import, raw preservation, image preparation, export, conflicts and recovery. Label fixture tests separately from live provider evidence.
- [x] Run documentation validation and complete regression on final code, inspect git diff/check, and produce a real PNG-to-PPTX smoke artifact in temporary storage.
- [x] Perform whole-branch independent review using the immutable diff package and test report.
- [x] Report local implementation evidence and Windows-native acceptance still pending if no Windows runner is accessible. Keep changes on the isolated branch for review.
