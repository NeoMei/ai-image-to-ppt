# Windows image-generation validation

This project produces PPTX files without controlling PowerPoint or WPS.
Windows execution and Office/WPS file viewing are separate checks.

## Setup

Use the candidate branch/commit supplied with the fix, in a writable local
NTFS directory. The native backend accepts absolute drive-letter paths; UNC
paths, unresolved junctions/symlinks/reparse points, reserved device names,
alternate data streams and trailing-dot/space aliases are rejected. Use an
ordinary local NTFS output folder for acceptance; mapped network drives and
cloud-backed folders require separate validation and are not established by
a local NTFS run. Record the exact commit (or source-archive manifest) and
interpreter before testing.
The commands below run from the repository root in PowerShell and do not
require activation scripts or administrator privileges.

```powershell
if (Test-Path .git) { git rev-parse HEAD } else { Get-Content WINDOWS-CANDIDATE.txt }
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe scripts/validate_skill.py .
.\.venv\Scripts\python.exe scripts/run_windows_tests.py
if ($LASTEXITCODE -ne 0) { throw 'Windows tests failed' }
```

The Windows runner must execute the native Windows publication tests. A run
on macOS that skips those tests is not Windows acceptance. The runner also
validates the static capability manifest. Its FIFO test is POSIX-only; symbolic-link
fixtures explicitly skip when Windows denies link-creation privileges. Record
these skips separately from successful rejection checks. Provider tests
using fixture HTTP responses prove the adapter/write path, not live access
to an image-generation service.

## Local artifact flow, without an API key

This deterministic check creates a fresh temporary workspace, imports a
near-16:9 PNG, verifies the unchanged raw copy and normalized master, checks
no-overwrite protection and explicit overwrite, prepares an editable-converter
input, and exports/reopens a PPTX. It leaves its output directory for inspection.
The image is a test fixture, not evidence of an actual host image tool call.
The Python block uses Unicode escapes for the Chinese directory name so it
also survives Windows PowerShell 5.1's default ASCII pipeline encoding.

```powershell
@'
import hashlib
import sys
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation

sys.path.insert(0, str(Path.cwd() / "scripts"))
from import_host_image import HostArtifact, import_host_artifact
from prepare_editable_input import prepare
from export_images import export_deck

workspace = Path(tempfile.mkdtemp(prefix="ai-image-windows-"))
out = workspace / "\u4e2d\u6587 \u7a7a\u683c"
out.mkdir()
source = workspace / "source.png"
Image.new("RGB", (1672, 941), (30, 80, 150)).save(source)
source_bytes = source.read_bytes()
master = out / "slide.png"
raw = out / "raw" / "slide.png"

result = import_host_artifact(HostArtifact.local_path(str(source)), master,
                              workspace, "openai")
assert result.ok, result.to_json()
assert raw.read_bytes() == source_bytes
assert Image.open(master).size == (1664, 936)
before = hashlib.sha256(master.read_bytes()).hexdigest()
refused = import_host_artifact(HostArtifact.local_path(str(source)), master,
                               workspace, "openai")
assert not refused.ok, refused.to_json()
assert hashlib.sha256(master.read_bytes()).hexdigest() == before
assert raw.read_bytes() == source_bytes

Image.new("RGB", (1672, 941), (150, 80, 30)).save(source)
replaced = import_host_artifact(HostArtifact.local_path(str(source)), master,
                                workspace, "openai", overwrite=True)
assert replaced.ok, replaced.to_json()
assert raw.read_bytes() == source.read_bytes()
assert hashlib.sha256(master.read_bytes()).hexdigest() != before
editable = out / "editable.png"
assert prepare(str(master), str(editable))
assert Image.open(editable).size == (1280, 720)
assert export_deck([str(master)], str(out / "deck"))
deck = Presentation(out / "deck.pptx")
assert len(deck.slides) == 1
assert len(deck.slides[0].shapes) == 1
assert deck.slide_width * 9 == deck.slide_height * 16
assert (out / "deck.pdf").stat().st_size > 0
print("PASS: import, raw preservation, overwrite protection, replacement, editable input, PPTX/PDF")
print("Artifacts:", out)
'@ | .\.venv\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Artifact flow failed' }
```

## Native failure and recovery checks

The automated Windows publication tests cover actual Win32 handles, file
identity, non-replacing publication, safe removal/rollback, open-file sharing
conflicts and reparse protection. Preserve any recovery locator reported by a
failed run. Windows retains old outputs as unique sibling files named
`.image-output-recovery-<random>.entry`; POSIX retains its existing directory/entry
layout. Do not remove retained files until the original and current
outputs have been compared. Generation/import rollback operates during the
process; only deck export has an interruption-recovery journal. Neither native
Windows path promises power-loss durability.

For a live acceptance run, use an available host image tool or a configured API
provider and follow the normal Skill/CLI flow in the README. Record the provider,
channel, GenerationResult status and resulting artifacts; omit credentials.
A lack of credentials or host capability is distinct from a publication failure.

## Evidence to report

- Windows version, Python version, filesystem, branch and commit.
- Native test result, including every skip and its reason.
- Artifact-flow result and output paths, plus any failure/recovery locator.
- Live provider or host-tool result separately, if available.
- Optionally open the produced deck in PowerPoint/WPS and verify display,
  playback, save and reopen. This proves viewing compatibility only.

## Native API references

The Windows backend relies on [CreateFileW sharing and reparse semantics](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew),
[handle-bound rename](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_rename_info),
and [handle-bound deletion](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_disposition_info).
Unsupported operations remain failures; the backend does not fall back to
a pathname-only replacement.
