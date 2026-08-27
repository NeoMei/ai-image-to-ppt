"""Deterministic host-first routing policy for slide-image batches.

The Skill owns host capability discovery and calls. This module receives only
classified attempts and makes the six-candidate serial routing decision.
"""

import re
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
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])?")
_UNSAFE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")


def _safe_reason(message: object) -> str:
    """Make a user-safe reason incapable of creating another report line."""
    value = _ANSI_ESCAPE.sub(" ", safe_message(message))
    value = _UNSAFE_CONTROL.sub(" ", value)
    value = _WHITESPACE.sub(" ", value).strip()
    return value.replace(":", ";")


def _ordered_redacted_summary(attempts: Tuple[AttemptRecord, ...]) -> str:
    """Return an ordered, one-line-per-candidate exhaustion summary."""
    return "\n".join(
        f"{attempt.candidate.key}: {attempt.result.status.value}: "
        f"{_safe_reason(attempt.result.safe_message) or 'no safe reason provided'}"
        for attempt in attempts
    )


class SerialStickyRouter:
    """Route pages serially with transactional, forward-only sticky state.

    The search cursor starts at host OpenAI, while the selected sticky candidate
    remains ``None`` until a page actually succeeds. A lock serializes calls and
    an explicit active-call guard rejects executor reentry even on this thread's
    reentrant lock. Executor failures are not route results: they leave all
    state untouched, fail closed, and permit retrying the same page.
    """

    def __init__(self) -> None:
        self._search_index = 0
        self._selected_index: Optional[int] = None
        self._last_page_number = 0
        self._stopped = False
        self._active_call = False
        self._pages: Dict[int, PageRoutingResult] = {}
        self._switches: list[SwitchRecord] = []
        self._lock = RLock()

    @property
    def sticky_candidate(self) -> Optional[RoutingCandidate]:
        if self._selected_index is None:
            return None
        return CANDIDATES[self._selected_index]

    @property
    def search_candidate(self) -> RoutingCandidate:
        return CANDIDATES[self._search_index]

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
        """Route exactly one page and commit state only after a decision."""
        with self._lock:
            if self._active_call:
                raise RuntimeError("route_page cannot be called from an active executor")
            self._validate_page(page_number, execute, cached)
            self._active_call = True
            try:
                if cached:
                    return self._commit_page(PageRoutingResult(page_number, "cached", None, ()))

                start_index = self._search_index
                selected_before = self._selected_index
                attempts = []
                for candidate in CANDIDATES[start_index:]:
                    try:
                        result = execute(candidate)
                        self._validate_result(candidate, result)
                    except Exception as error:
                        raise RuntimeError(
                            "candidate executor failed; route state was not committed"
                        ) from error
                    attempt = AttemptRecord(page_number, candidate, result)
                    attempts.append(attempt)
                    immutable_attempts = tuple(attempts)
                    if result.status is GenerationStatus.SUCCESS:
                        switch = None
                        if selected_before is not None and candidate.index > selected_before:
                            switch = SwitchRecord(
                                page_number,
                                CANDIDATES[selected_before],
                                candidate,
                                "; ".join(
                                    f"{prior.candidate.key}: {prior.result.status.value}: "
                                    f"{_safe_reason(prior.result.safe_message)}"
                                    for prior in attempts[:-1]
                                ),
                            )
                        page = PageRoutingResult(
                            page_number, "success", candidate, immutable_attempts
                        )
                        return self._commit_page(
                            page,
                            search_index=candidate.index,
                            selected_index=candidate.index,
                            switch=switch,
                        )
                    if result.status in FATAL_STATUSES:
                        return self._commit_page(
                            PageRoutingResult(
                                page_number,
                                "fatal",
                                candidate,
                                immutable_attempts,
                                _safe_reason(result.safe_message),
                            ),
                            stopped=True,
                        )
                    if result.status not in FALLBACK_STATUSES:
                        raise RuntimeError(f"unclassified generation status: {result.status}")

                immutable_attempts = tuple(attempts)
                return self._commit_page(
                    PageRoutingResult(
                        page_number,
                        "exhausted",
                        None,
                        immutable_attempts,
                        _ordered_redacted_summary(immutable_attempts),
                    ),
                    stopped=True,
                )
            finally:
                self._active_call = False

    def report(self) -> dict:
        """Return a JSON-ready record for the Skill's final batch report."""
        return {
            "batch_mode": "serial-sticky-monotonic",
            "stopped": self.stopped,
            "search_candidate": self.search_candidate.key,
            "sticky_candidate": self.sticky_candidate.key if self.sticky_candidate else None,
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

    def _commit_page(
        self,
        page: PageRoutingResult,
        *,
        search_index: Optional[int] = None,
        selected_index: Optional[int] = None,
        stopped: bool = False,
        switch: Optional[SwitchRecord] = None,
    ) -> PageRoutingResult:
        """Commit one terminal decision after all of its work has succeeded."""
        if search_index is not None:
            self._search_index = search_index
        if selected_index is not None:
            self._selected_index = selected_index
        if switch is not None:
            self._switches.append(switch)
        self._pages[page.page_number] = page
        self._last_page_number = page.page_number
        self._stopped = stopped
        return page

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
