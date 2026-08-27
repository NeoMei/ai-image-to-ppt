import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generation_result import GenerationResult, GenerationStatus
from host_routing_policy import (
    CANDIDATE_KEYS,
    FATAL_STATUSES,
    FALLBACK_STATUSES,
    SerialStickyRouter,
)


def result(candidate, status, message="safe reason"):
    channel, provider = candidate.split("-", 1)
    return GenerationResult(
        GenerationStatus(status),
        provider,
        channel,
        "/workspace/out/slide.png" if status == "success" else None,
        message,
    )


class HostRoutingPolicyTests(unittest.TestCase):
    def test_candidate_order_and_status_partitions_are_fixed(self):
        self.assertEqual(CANDIDATE_KEYS, ("host-openai", "api-openai", "host-gemini", "api-gemini", "host-doubao", "api-doubao"))
        self.assertEqual(FALLBACK_STATUSES, frozenset({GenerationStatus.UNAVAILABLE, GenerationStatus.AUTH_UNAVAILABLE, GenerationStatus.RETRYABLE_EXHAUSTED}))
        self.assertEqual(FATAL_STATUSES, frozenset({GenerationStatus.POLICY_REFUSED, GenerationStatus.INVALID_INPUT, GenerationStatus.INVALID_OUTPUT, GenerationStatus.LOCAL_FAILURE}))

    def test_fallback_moves_only_forward_until_a_candidate_succeeds(self):
        router = SerialStickyRouter()
        calls = []
        outcomes = {"host-openai": "unavailable", "api-openai": "success"}
        page = router.route_page(1, lambda candidate: calls.append(candidate.key) or result(candidate.key, outcomes[candidate.key]))
        self.assertEqual(calls, ["host-openai", "api-openai"])
        self.assertEqual(page.outcome, "success")
        self.assertEqual(page.candidate.key, "api-openai")
        self.assertEqual(router.sticky_candidate.key, "api-openai")

    def test_fatal_status_stops_immediately_and_closes_the_batch(self):
        for status in FATAL_STATUSES:
            with self.subTest(status=status):
                router = SerialStickyRouter()
                calls = []
                page = router.route_page(1, lambda candidate: calls.append(candidate.key) or result(candidate.key, status.value))
                self.assertEqual(page.outcome, "fatal")
                self.assertEqual(calls, ["host-openai"])
                self.assertTrue(router.stopped)
                with self.assertRaises(RuntimeError):
                    router.route_page(2, lambda candidate: result(candidate.key, "success"))

    def test_batch_is_serial_sticky_monotonic_and_records_switch_page(self):
        router = SerialStickyRouter()
        calls = []
        per_page = {1: {"host-openai": "success"}, 2: {"host-openai": "retryable_exhausted", "api-openai": "success"}, 3: {"api-openai": "success"}}
        for page_number in (1, 2, 3):
            router.route_page(page_number, lambda candidate, page_number=page_number: calls.append((page_number, candidate.key)) or result(candidate.key, per_page[page_number][candidate.key]))
        self.assertEqual(calls, [(1, "host-openai"), (2, "host-openai"), (2, "api-openai"), (3, "api-openai")])
        self.assertEqual(router.sticky_candidate.key, "api-openai")
        self.assertEqual(len(router.switches), 1)
        switch = router.switches[0]
        self.assertEqual((switch.page_number, switch.from_candidate.key, switch.to_candidate.key), (2, "host-openai", "api-openai"))

    def test_cache_never_calls_a_candidate_or_changes_route(self):
        router = SerialStickyRouter()
        cached = router.route_page(1, lambda candidate: self.fail("must not run"), cached=True)
        self.assertEqual(cached.outcome, "cached")
        self.assertEqual(router.sticky_candidate.key, "host-openai")
        generated = router.route_page(2, lambda candidate: result(candidate.key, "success"))
        self.assertEqual(generated.candidate.key, "host-openai")

    def test_successful_page_cannot_be_regenerated_and_pages_are_monotonic(self):
        router = SerialStickyRouter()
        router.route_page(2, lambda candidate: result(candidate.key, "success"))
        with self.assertRaises(ValueError):
            router.route_page(2, lambda candidate: result(candidate.key, "success"))
        with self.assertRaises(ValueError):
            router.route_page(1, lambda candidate: result(candidate.key, "success"))

    def test_all_candidates_exhausted_returns_ordered_redacted_summary(self):
        router = SerialStickyRouter()
        statuses = ("unavailable", "auth_unavailable", "retryable_exhausted", "unavailable", "auth_unavailable", "retryable_exhausted")
        page = router.route_page(1, lambda candidate: result(candidate.key, statuses[candidate.index], "Bearer top-secret sk-testSecret123"))
        self.assertEqual(page.outcome, "exhausted")
        self.assertEqual([attempt.candidate.key for attempt in page.attempts], list(CANDIDATE_KEYS))
        self.assertNotIn("top-secret", page.summary)
        self.assertNotIn("sk-testSecret123", page.summary)
        self.assertEqual([line.split(":", 1)[0] for line in page.summary.splitlines()], list(CANDIDATE_KEYS))
        self.assertTrue(router.stopped)


if __name__ == "__main__":
    unittest.main()
