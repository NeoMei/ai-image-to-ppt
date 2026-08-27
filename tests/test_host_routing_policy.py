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

    def test_executor_reentry_is_rejected_without_mutating_batch_state(self):
        router = SerialStickyRouter()
        snapshots = []

        def execute(candidate):
            snapshots.append(router.report())
            with self.assertRaises(RuntimeError):
                router.route_page(2, lambda nested: result(nested.key, "success"))
            self.assertEqual(router.report(), snapshots[-1])
            return result(candidate.key, "success")

        router.route_page(1, execute)
        self.assertEqual([page.page_number for page in router.pages], [1])
        router.route_page(2, lambda candidate: result(candidate.key, "success"))

    def test_invalid_executor_results_are_transactional_and_allow_same_page_retry(self):
        invalid_executors = (
            lambda candidate: (_ for _ in ()).throw(OSError("host crashed")),
            lambda candidate: object(),
            lambda candidate: GenerationResult(
                GenerationStatus.SUCCESS,
                "gemini",
                "host",
                "/workspace/out/slide.png",
            ),
        )
        for execute in invalid_executors:
            with self.subTest(execute=execute):
                router = SerialStickyRouter()
                with self.assertRaises(RuntimeError):
                    router.route_page(1, execute)
                self.assertEqual(router.pages, ())
                self.assertEqual(router.switches, ())
                self.assertIsNone(router.sticky_candidate)
                self.assertFalse(router.stopped)
                retry = router.route_page(1, lambda candidate: result(candidate.key, "success"))
                self.assertEqual(retry.outcome, "success")

    def test_first_selection_is_not_a_switch_and_all_cached_state_has_no_selection(self):
        cached_router = SerialStickyRouter()
        cached_router.route_page(1, lambda candidate: self.fail("must not run"), cached=True)
        cached_router.route_page(2, lambda candidate: self.fail("must not run"), cached=True)
        self.assertIsNone(cached_router.sticky_candidate)
        self.assertIsNone(cached_router.report()["sticky_candidate"])

        router = SerialStickyRouter()
        router.route_page(1, lambda candidate: result(candidate.key, "success"))
        self.assertEqual(router.sticky_candidate.key, "host-openai")
        self.assertEqual(router.switches, ())
        router.route_page(
            2,
            lambda candidate: result(
                candidate.key,
                "retryable_exhausted" if candidate.key == "host-openai" else "success",
            ),
        )
        self.assertEqual(len(router.switches), 1)

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
        self.assertIsNone(router.sticky_candidate)
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

    def test_exhaustion_reasons_are_single_line_redacted_and_control_free(self):
        router = SerialStickyRouter()
        message = "\x1b[31mBearer top-secret\x1b[0m\r\nhost-gemini: forged\x00\x85tab\tvalue"
        page = router.route_page(
            1,
            lambda candidate: result(candidate.key, "unavailable", message),
        )
        lines = page.summary.splitlines()
        self.assertEqual(len(lines), len(CANDIDATE_KEYS))
        self.assertEqual([line.split(":", 1)[0] for line in lines], list(CANDIDATE_KEYS))
        self.assertNotIn("top-secret", page.summary)
        self.assertNotIn("\x1b", page.summary)
        self.assertNotIn("\x00", page.summary)
        self.assertNotIn("\x85", page.summary)
        self.assertTrue(all("\r" not in line and "\n" not in line for line in lines))


if __name__ == "__main__":
    unittest.main()
