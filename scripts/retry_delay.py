"""Shared Retry-After parsing and bounded retry backoff."""

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping, Optional


MAX_RETRY_DELAY = 60.0
BASE_RETRY_DELAY = 2.0


def _clamp(delay: float) -> float:
    return min(MAX_RETRY_DELAY, max(0.0, delay))


def _retry_after_value(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
        if value is None:
            value = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_retry_after(value: str, now: datetime) -> Optional[float]:
    if value.isascii() and value.isdecimal():
        try:
            return _clamp(float(value))
        except (ValueError, OverflowError):
            return None
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    try:
        delay = (retry_at - now).total_seconds()
    except (TypeError, OverflowError):
        return None
    if not math.isfinite(delay):
        return None
    return _clamp(delay)


def retry_delay(
    attempt: int,
    headers: Optional[Mapping[str, str]] = None,
    *,
    now: Optional[datetime] = None,
) -> float:
    """Return Retry-After delay or a deterministic exponential fallback.

    ``attempt`` is the zero-based failed-attempt index. All returned delays are
    bounded to the inclusive range 0..60 seconds.
    """
    current_time = now or datetime.now(timezone.utc)
    value = _retry_after_value(headers)
    if value is not None:
        parsed = _parse_retry_after(value, current_time)
        if parsed is not None:
            return parsed
    exponent = min(max(attempt, 0), 5)
    return _clamp(BASE_RETRY_DELAY * (2 ** exponent))
