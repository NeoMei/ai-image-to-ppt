#!/usr/bin/env python3
"""Explicit offline maintenance for persistent output-lock files."""

import argparse
import math
import re
import stat
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from output_lock import _LOCK_DIRECTORY


LOCK_DIRECTORY = _LOCK_DIRECTORY
_LOCK_FILE_NAME = re.compile(r"[0-9a-f]{64}\.lock\Z")
_SECONDS_PER_DAY = 24 * 60 * 60


class CleanupOutputLockError(OSError):
    """Raised when offline output-lock maintenance cannot proceed safely."""


def _nonnegative_days(value: str) -> float:
    try:
        days = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(days) or days < 0:
        raise argparse.ArgumentTypeError("must be a finite number at least 0")
    return days


def cleanup_stale_lock_files(
    max_age_seconds: float,
    lock_directory: Path = LOCK_DIRECTORY,
    now: Optional[float] = None,
) -> int:
    """Remove stale hashed lock files under an explicitly offline directory."""
    if (
        isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, (int, float))
        or not math.isfinite(max_age_seconds)
        or max_age_seconds < 0
    ):
        raise ValueError("max_age_seconds must be a finite number at least 0")

    directory = Path(lock_directory)
    try:
        directory_metadata = directory.lstat()
    except FileNotFoundError:
        return 0
    except OSError as error:
        raise CleanupOutputLockError(
            f"cannot inspect lock directory {directory}: {error}"
        ) from error
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise CleanupOutputLockError(
            f"lock directory must be a real directory, not a symlink or file: "
            f"{directory}"
        )

    cutoff = (time.time() if now is None else now) - max_age_seconds
    removed = 0
    try:
        for path in directory.iterdir():
            if _LOCK_FILE_NAME.fullmatch(path.name) is None:
                continue
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            except OSError as error:
                raise CleanupOutputLockError(
                    f"cannot inspect output lock file {path}: {error}"
                ) from error
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mtime >= cutoff:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as error:
                raise CleanupOutputLockError(
                    f"cannot remove output lock file {path}: {error}"
                ) from error
            removed += 1
    except CleanupOutputLockError:
        raise
    except OSError as error:
        raise CleanupOutputLockError(
            f"cannot scan lock directory {directory}: {error}"
        ) from error
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove old persistent output-lock files.",
        epilog=(
            "Safety: run this command ONLY when no generation or export process "
            "is running. It never runs automatically because unlinking a live "
            "lock file can split cross-process ownership. The offline "
            "precondition is the safety boundary for pathname races. "
            "Filesystem failures stop cleanup, return exit 1 without a "
            "traceback, and do not roll back earlier removals."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--older-than-days",
        type=_nonnegative_days,
        default=30.0,
        metavar="DAYS",
        help="remove hashed lock files older than DAYS (default: 30)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        removed = cleanup_stale_lock_files(
            lock_directory=LOCK_DIRECTORY,
            max_age_seconds=args.older_than_days * _SECONDS_PER_DAY,
        )
    except CleanupOutputLockError as error:
        print(f"ERR: output lock cleanup failed: {error}", file=sys.stderr)
        return 1
    suffix = "" if removed == 1 else "s"
    print(f"Removed {removed} stale output lock file{suffix} from {LOCK_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
