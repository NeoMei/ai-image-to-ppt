# Host-first image routing

Read this reference before generating any slide image. The Skill performs host
capability discovery and calls; `scripts/host_routing_policy.py` is the
executable policy for the resulting classified attempts.

## Candidate policy

The fixed candidate order is `host-openai`, `api-openai`, `host-gemini`,
`api-gemini`, `host-doubao`, then `api-doubao`. The only fallback statuses are
`unavailable`, `auth_unavailable`, and `retryable_exhausted`. Stop immediately
for `policy_refused`, `invalid_input`, `invalid_output`, or `local_failure`.

Only a tool or connector that is currently registered and callable by this
host counts as a host capability. A login in a browser, provider documentation,
or guessed account state does not. A missing or disconnected host tool is
`unavailable`; do not attempt browser automation, cookies, extracted tokens, or
a hidden web fallback. The host OpenAI capability is the first candidate and
does not require `OPENAI_API_KEY`.

Offer each API key only when useful, and let the user skip it. In noninteractive
work, a missing key is `auth_unavailable` and routing continues. Never print,
persist, or put a key in a prompt or report.

## Host artifact handoff

Call an available host tool directly. Accept one explicitly marked primary or
result image, or the sole image when exactly one is returned. Multiple unmarked
images are `invalid_output`.

Accept only a readable absolute local path, MIME-labelled inline bytes/Base64 or
`data:` URL, or a resource handle for which this host exposes an explicit export
or download call. Materialize the accepted artifact into the workspace through
the importer and its same local validation/publish contract. Reject bare
HTTP(S) URLs, UI-only previews, ambiguous text, unavailable handles, and any
unmaterializable resource as `invalid_output`; do not fetch URLs yourself.

For a captured absolute host-local path, use:

```bash
python3 scripts/import_host_image.py \
  "/absolute/host/generated/image.png" \
  "out/slide_01.png" \
  --workspace-root "/absolute/workspace" \
  --provider openai \
  --json
```

The importer first copies the accepted host bytes unchanged to the absolute
workspace path `out/raw/<filename>` (for this example,
`/absolute/workspace/out/raw/slide_01.png`). That raw artifact is preserved on a
successful import; it is never resized by the master-normalization step.

The host raw/master pair has compensating rollback only for ordinary failures
while the importer process and both locks remain alive; it is not a crash-atomic
two-file transaction. Import and API publication use platform-specific secure
output primitives: POSIX directory-descriptor/no-follow checks, hard links and
same-directory rename, or Windows native file handles with directory pinning
and handle-bound rename/deletion. Unsupported operations, unsafe paths and
incompatible file-sharing permissions fail closed as `local_failure`; do not
substitute a pathname-only write. See [Windows validation](../docs/windows-validation.md)
for setup and native acceptance commands.

Host artifacts alone may be within a 0.5% relative 16:9 error, evaluated with
integer cross-products. Exact 16:9 host bytes retain the existing master
publication behavior. A qualifying near-ratio artifact is center-cropped; do
not stretch it. Crop to the largest contained `16*k × 9*k` master, then strict 16:9
validation is applied to the newly encoded PNG/JPEG before safe publication.
For example, a 1672×941 PNG becomes a 1664×936 PNG master. MIME and actual
format must still match the requested suffix. The OpenAI, Gemini, and Doubao
API adapters remain strict 16:9: they do not receive this tolerance or crop.
Forced generation/import publication is ownership-preserving rather than a
single-syscall atomic replacement. POSIX isolates the current pathname and
verifies the inode actually moved; Windows opens and verifies the existing
file while denying conflicting writes/deletes, then moves it through that
handle to a unique sibling recovery file. Both paths install the new image without clobbering another file. An
unexpected file or failed cleanup is retained under
`.image-output-recovery-*/entry` on POSIX or the sibling file
`.image-output-recovery-*.entry` on Windows, and reported instead of being deleted.
Cooperating processes may race public target names. On POSIX, a same-UID actor that
deliberately discovers and mutates the private recovery namespace (random and
mode 0700) between syscalls is outside the portable ownership guarantee because
POSIX has no unlink-if-inode primitive. Treat a reported retained path as
sensitive and recover it manually.
Recovery warnings begin with a complete ASCII locator such as
`recovery=.image-output-recovery-<random>/entry relative-to-target-parent`
on POSIX, or `recovery=.image-output-recovery-<random>.entry relative-to-target-parent`
on Windows. Resolve it against the known raw or master target parent. It is never truncated;
only the following absolute-path diagnostic may be shortened.

A materialization authorization failure is `auth_unavailable`; an exhausted
timeout, 429, network error, or 5xx is `retryable_exhausted`; an explicit
safety refusal is `policy_refused`; missing, undecodable, too-small,
MIME/format-mismatched, or over-tolerance content is `invalid_output`; and
workspace write or publication failure is `local_failure`.

## API/CLI-only adapter

`scripts/gen_slide.py` is an **API/CLI-only** adapter. It cannot discover or
call host tools, and explicit CLI selection calls only that one API provider.
Use it only for the API candidate selected by this policy:

```bash
python3 scripts/gen_slide.py \
  "${slide_output}" \
  "${slide_prompt}" \
  --engine "${provider}" \
  --json
```

## Serial sticky batches and report

Generate pages serially. The first actual generation success becomes sticky.
For later pages start at that candidate; on a fallback status, try only later
candidates and make the first success sticky. Never move backward, never
regenerate a successful page, and stop the batch on a fatal result. Existing
validated cached pages are reported as cached but never establish or change
routing state. `SerialStickyRouter` records each switch page, old/new candidate,
and a safe reason; it also returns an ordered redacted all-candidates-exhausted
summary.

Report each page's selected provider/channel, cached/success/fatal outcome,
and every switch page. On exhaustion, show the ordered redacted candidate/status
summary only—never raw provider responses or credentials.

## Editable handoff

Keep the validated high-resolution master. Only when using
`image-to-editable-pptx`, create a separate, real exact `1280×720 PNG`:

```bash
python3 scripts/prepare_editable_input.py \
  out/slide_01.jpg \
  out/editable/slide_01.png
```

Do not overwrite the master or raw artifact. The host importer is the only
place that can center-crop a qualifying near-16:9 host artifact; this handoff
accepts the resulting strict 16:9 master and never stretches it.
