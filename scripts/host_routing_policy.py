"""Deterministic host-first routing policy for slide-image batches.

The Skill owns host capability discovery and calls.  This module deliberately
does not inspect a host, read credentials, or invoke a provider: it turns those
already-classified attempts into the fixed six-candidate, serial sticky policy.
"""

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Dict, Optional, Tuple

from generation_result import (
    FATAL_STATUSES as _FATAL_STATUSES,
    FALLBACK_STATUSES as _FALLBACK_STATUSES,
    GenerationResult,
    GenerationStatus,
    safe_message,
)


@dataclass(frozen=True)
class RoutingCandidate:
    """One host or API candidate in the policy's immutable order."""

    index: int
    channel: str
    provider: str

    @property
    def key(self) -> str:
        return f"{self.channel}-{self.provider}"


CANDIDATES: Tuple[RoutingCandidate, ...] = (
    RoutingCandidate(0, "host", "openai"),
    RoutingCandidate(1, "api", "openai"),
    RoutingCandidate(2, "host", "gemini"),
    RoutingCandidate(3, "api", "gemini"),
    RoutingCandidate(4, "host", "doubao"),
    RoutingCandidate(5, "api", "doubao"),
)
CANDIDATE_KEYS = tuple(candidate.key for candidate in CANDIDATES)
FALLBACK_STATUSES = _FALLBACK_STATUSES
FATAL_STATUSES = _FATAL_STATUSES


@dataclass(frozen=True)
class AttemptRecord:
    page_number: int
    candidate: RoutingCandidate
    result: GenerationResult


@dataclass(frozen=True)
class SwitchRecord:
    page_number: int
    from_candidate: RoutingCandidate
    to_candidate: RoutingCandidate
    reason: str


@dataclass(frozen=True)
class PageRoutingResult:
    """The complete, safe-to-report decision for one ordered page."""

    page_number: int
    outcome: str
    candidate: Optional[RoutingCandidate]
    attempts: Tuple[AttemptRecord, ...]
    summary: str = ""


AttemptExecutor = Callable[[RoutingCandidate], GenerationResult]


def _ordered_redacted_summary(attempts: Tuple[AttemptRecord, ...]) -> str:
    """Return an ordered user-safe exhaustion summary without raw responses."""
    return "\n".join(
        f"{attempt.candidate.key}: {attempt.result.status.value}: "
        f"{safe_message(attempt.result.safe_message) or 'no safe reason provided'}"
        for attempt in attempts
    )


class SerialStickyRouter:
    """Route pages serially, pinning the first successful candidate.

    `route_page` holds one lock through the supplied executor, so callers cannot
    accidentally make parallel generation calls through the same batch state.
    Page numbers must strictly increase; successful and cached pages are never
    retried.  A fallback can only advance the sticky candidate, while a fatal
    result or full exhaustion closes the batch.
    """

    def __init__(self) -> None:
        self._sticky_index = 0
        self._last_page_number = 0
        self._stopped = False
        self._pages: Dict[int, PageRoutingResult] = {}
        self._switches: list[SwitchRecord] = []
        self._lock = RLock()

    @property
    def sticky_candidate(self) -> RoutingCandidate:
        return CANDIDATES[self._sticky_index]

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def switches(self) -> Tuple[SwitchRecord, ...]:
        return tuple(self._switches)

    @property
    def pages(self) -> Tuple[PageRoutingResult, ...]:
        return tuple(self._pages[number] for number in sorted(self._pages))

    def route_page(
        self,
        page_number: int,
        execute: AttemptExecutor,
        *,
        cached: bool = False,
    ) -> PageRoutingResult:
        """Route exactly one page through the current sticky candidate onward."""
        with self._lock:
            self._validate_page(page_number, execute, cached)
            self._last_page_number = page_number
            if cached:
                page = PageRoutingResult(page_number, "cached", None, ())
                self._pages[page_number] = page
                return page

            start_index = self._sticky_index
            attempts = []
            for candidate in CANDIDATES[start_index:]:
                result = execute(candidate)
                self._validate_result(candidate, result)
                attempt = AttemptRecord(page_number, candidate, result)
                attempts.append(attempt)
                immutable_attempts = tuple(attempts)
                if result.status is GenerationStatus.SUCCESS:
                    if candidate.index > start_index:
                        self._switches.append(
                            SwitchRecord(
                                page_number,
                                CANDIDATES[start_index],
                                candidate,
                                "; ".join(
                                    f"{prior.candidate.key}: "
                                    f"{prior.result.status.value}: "
                                    f"{safe_message(prior.result.safe_message)}"
                                    for prior in attempts[:-1]
                                ),
                            )
                        )
                    self._sticky_index = candidate.index
                    page = PageRoutingResult(
                        page_number, "success", candidate, immutable_attempts
                    )
                    self._pages[page_number] = page
                    return page
                if result.status in FATAL_STATUSES:
                    self._stopped = True
                    page = PageRoutingResult(
                        page_number,
                        "fatal",
                        candidate,
                        immutable_attempts,
                        safe_message(result.safe_message),
                    )
                    self._pages[page_number] = page
                    return page
                if result.status not in FALLBACK_STATUSES:
                    raise RuntimeError(f"unclassified generation status: {result.status}")

            immutable_attempts = tuple(attempts)
            self._stopped = True
            page = PageRoutingResult(
                page_number,
                "exhausted",
                None,
                immutable_attempts,
                _ordered_redacted_summary(immutable_attempts),
            )
            self._pages[page_number] = page
            return page

    def report(self) -> dict:
        """Return a JSON-ready record for the Skill's final batch report."""
        return {
            "batch_mode": "serial-sticky-monotonic",
            "stopped": self.stopped,
            "sticky_candidate": self.sticky_candidate.key,
            "pages": [
                {
                    "page": page.page_number,
                    "outcome": page.outcome,
                    "candidate": page.candidate.key if page.candidate else None,
                    "summary": page.summary,
                }
                for page in self.pages
            ],
            "switches": [
                {
                    "page": switch.page_number,
                    "from": switch.from_candidate.key,
                    "to": switch.to_candidate.key,
                    "reason": switch.reason,
                }
                for switch in self.switches
            ],
        }

    def _validate_page(
        self, page_number: int, execute: AttemptExecutor, cached: bool
    ) -> None:
        if self._stopped:
            raise RuntimeError("batch routing has stopped")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if page_number <= self._last_page_number or page_number in self._pages:
            raise ValueError("pages must be routed once in strictly increasing order")
        if not callable(execute):
            raise TypeError("execute must be callable")
        if not isinstance(cached, bool):
            raise TypeError("cached must be a bool")

    @staticmethod
    def _validate_result(
        candidate: RoutingCandidate, result: GenerationResult
    ) -> None:
        if not isinstance(result, GenerationResult):
            raise TypeError("execute must return a GenerationResult")
        if result.provider != candidate.provider or result.channel != candidate.channel:
            raise ValueError("attempt result does not match its routing candidate")
