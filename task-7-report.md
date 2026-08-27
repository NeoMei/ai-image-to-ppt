# Task 7 SDD report — routing policy fix round 1/5

## State and decisions

- `SerialStickyRouter` now separates its forward-only search cursor from the
  optional selected sticky candidate. An all-cached batch has no selection, and
  the first actual success is a selection rather than a switch.
- Route execution is transactional. Executor exceptions, non-results, and
  provider/channel-mismatched results fail closed without consuming a page,
  changing routing state, or creating a report page; the same page can retry.
- An explicit active-call guard rejects nested `route_page` calls from an
  executor even though the serializing lock is reentrant.
- Exhaustion and switch reasons are redacted, stripped of ANSI/C0/C1 control
  characters, whitespace-normalized, and held to one report line per candidate.

## RED → GREEN evidence

The new policy tests first failed against the prior implementation for executor
reentry, transactional retry, unset cached selection, and line-injectable
exhaustion reasons. They pass after the policy change.

- Policy suite: 11 passed.
- Documentation suite: 7 passed.
- Provider/export documentation checks: 13 passed.
- Full suite: 380 passed, 1 skipped.
- `py_compile`, Skill validator, and `git diff --check`: passed.

## Documentation recovery

README again names the active Doubao model and preserves the 14 MiB vision cap,
bounded Retry-After behavior, parent-directory replacement failure boundary,
POSIX/Windows durability split, force rollback/recovery, and explicit offline
cleanup race/partial-failure limitations. It does not restore concurrent image
generation guidance.

## Cost

This round used only local mocked tests. No host image tool or provider API ran;
external image-generation cost was $0.
