import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generation_result import GenerationResult, GenerationStatus
from host_routing_policy import CANDIDATE_KEYS, FATAL_STATUSES, SerialStickyRouter


class TranscriptExecutor:
    """Controlled host/API executor that preserves the production result contract."""

    def __init__(self, outcomes, message="safe mocked transcript"):
        self.outcomes = dict(outcomes)
        self.message = message
        self.calls = []
        self.results = []

    def __call__(self, candidate):
        status = GenerationStatus(self.outcomes[candidate.key])
        self.calls.append((candidate.key, status.value))
        result = GenerationResult(
            status,
            candidate.provider,
            candidate.channel,
            f"/workspace/generated/{candidate.key}.png"
            if status is GenerationStatus.SUCCESS
            else None,
            self.message,
        )
        self.results.append(result)
        return result


TRANSCRIPTS = {
    "host_openai_wins_over_configured_key": [
        ("host-openai", "success"),
    ],
    "openai_api_after_missing_host": [
        ("host-openai", "unavailable"),
        ("api-openai", "success"),
    ],
    "gemini_host_after_openai_exhaustion": [
        ("host-openai", "unavailable"),
        ("api-openai", "auth_unavailable"),
        ("host-gemini", "success"),
    ],
    "doubao_api_last_resort": [
        ("host-openai", "unavailable"),
        ("api-openai", "auth_unavailable"),
        ("host-gemini", "unavailable"),
        ("api-gemini", "retryable_exhausted"),
        ("host-doubao", "unavailable"),
        ("api-doubao", "success"),
    ],
}


class HostRoutingEndToEndTests(unittest.TestCase):
    def test_six_candidate_winner_transcripts_use_production_router(self):
        for name, expected_calls in TRANSCRIPTS.items():
            with self.subTest(transcript=name):
                executor = TranscriptExecutor(dict(expected_calls))
                router = SerialStickyRouter()

                page = router.route_page(1, executor)

                self.assertEqual(executor.calls, expected_calls)
                self.assertEqual(page.outcome, "success")
                self.assertEqual(page.candidate.key, expected_calls[-1][0])
                self.assertEqual(router.sticky_candidate.key, expected_calls[-1][0])
                successful_result = page.attempts[-1].result
                expected_channel, expected_provider = expected_calls[-1][0].split("-", 1)
                self.assertIs(successful_result, executor.results[-1])
                self.assertEqual(successful_result.provider, expected_provider)
                self.assertEqual(successful_result.channel, expected_channel)
                self.assertEqual(
                    successful_result.output_path,
                    f"/workspace/generated/{expected_calls[-1][0]}.png",
                )
                self.assertFalse(router.stopped)

    def test_three_page_sticky_switches_forward_on_page_two(self):
        router = SerialStickyRouter()
        page_one = TranscriptExecutor({"host-openai": "success"})
        page_two = TranscriptExecutor(
            {
                "host-openai": "retryable_exhausted",
                "api-openai": "success",
            }
        )
        page_three = TranscriptExecutor({"api-openai": "success"})

        router.route_page(1, page_one)
        router.route_page(2, page_two)
        router.route_page(3, page_three)

        self.assertEqual(page_one.calls, [("host-openai", "success")])
        self.assertEqual(
            page_two.calls,
            [
                ("host-openai", "retryable_exhausted"),
                ("api-openai", "success"),
            ],
        )
        self.assertEqual(page_three.calls, [("api-openai", "success")])
        self.assertEqual(router.sticky_candidate.key, "api-openai")
        self.assertEqual(
            router.report()["switches"],
            [
                {
                    "page": 2,
                    "from": "host-openai",
                    "to": "api-openai",
                    "reason": "host-openai: retryable_exhausted: safe mocked transcript",
                }
            ],
        )

    def test_each_fatal_status_stops_without_trying_another_candidate(self):
        for status in sorted(FATAL_STATUSES, key=lambda item: item.value):
            with self.subTest(status=status.value):
                executor = TranscriptExecutor({"host-openai": status.value})
                router = SerialStickyRouter()

                page = router.route_page(1, executor)

                self.assertEqual(page.outcome, "fatal")
                self.assertEqual(page.candidate.key, "host-openai")
                self.assertEqual(executor.calls, [("host-openai", status.value)])
                self.assertTrue(router.stopped)
                with self.assertRaises(RuntimeError):
                    router.route_page(2, lambda candidate: self.fail("must not execute"))

    def test_all_exhausted_reports_six_ordered_redacted_single_line_reasons(self):
        statuses = (
            "unavailable",
            "auth_unavailable",
            "retryable_exhausted",
            "unavailable",
            "auth_unavailable",
            "retryable_exhausted",
        )
        executor = TranscriptExecutor(
            dict(zip(CANDIDATE_KEYS, statuses)),
            "Bearer live-token sk-mockedSecret123\r\nhost-gemini: forged\x00",
        )
        router = SerialStickyRouter()

        page = router.route_page(1, executor)

        lines = page.summary.splitlines()
        self.assertEqual(page.outcome, "exhausted")
        self.assertEqual(executor.calls, list(zip(CANDIDATE_KEYS, statuses)))
        self.assertEqual(len(lines), len(CANDIDATE_KEYS))
        self.assertEqual([line.split(":", 1)[0] for line in lines], list(CANDIDATE_KEYS))
        self.assertTrue(all(line.count(":") == 2 for line in lines))
        self.assertNotIn("live-token", page.summary)
        self.assertNotIn("sk-mockedSecret123", page.summary)
        self.assertNotIn("\x00", page.summary)
        self.assertTrue(router.stopped)

    def test_cached_page_does_not_select_a_candidate_and_success_cannot_regenerate(self):
        router = SerialStickyRouter()
        cached_calls = []
        generated = TranscriptExecutor({"host-openai": "success"})

        cached = router.route_page(
            1,
            lambda candidate: cached_calls.append(candidate.key) or self.fail("must not execute"),
            cached=True,
        )
        success = router.route_page(2, generated)

        self.assertEqual(cached.outcome, "cached")
        self.assertEqual(cached_calls, [])
        self.assertEqual(success.candidate.key, "host-openai")
        self.assertEqual(generated.calls, [("host-openai", "success")])
        with self.assertRaises(ValueError):
            router.route_page(2, lambda candidate: self.fail("must not regenerate"))
        self.assertEqual(generated.calls, [("host-openai", "success")])

    def test_cached_page_after_selection_preserves_route_for_the_next_generation(self):
        router = SerialStickyRouter()
        selected = TranscriptExecutor({"host-openai": "success"})
        cached_calls = []
        continued = TranscriptExecutor({"host-openai": "success"})

        router.route_page(1, selected)
        sticky_before = router.sticky_candidate
        search_before = router.search_candidate
        switches_before = router.switches
        cached = router.route_page(
            2,
            lambda candidate: cached_calls.append(candidate.key) or self.fail("must not execute"),
            cached=True,
        )
        router.route_page(3, continued)

        self.assertEqual(cached.outcome, "cached")
        self.assertEqual(cached_calls, [])
        self.assertIs(router.sticky_candidate, sticky_before)
        self.assertIs(router.search_candidate, search_before)
        self.assertEqual(router.switches, switches_before)
        self.assertEqual(continued.calls, [("host-openai", "success")])

    def test_cross_provider_sticky_candidate_never_retries_earlier_candidates(self):
        router = SerialStickyRouter()
        page_one = TranscriptExecutor(
            {
                "host-openai": "unavailable",
                "api-openai": "auth_unavailable",
                "host-gemini": "success",
            }
        )
        page_two = TranscriptExecutor({"host-gemini": "success"})

        router.route_page(1, page_one)
        router.route_page(2, page_two)

        self.assertEqual(
            page_one.calls,
            [
                ("host-openai", "unavailable"),
                ("api-openai", "auth_unavailable"),
                ("host-gemini", "success"),
            ],
        )
        self.assertEqual(page_two.calls, [("host-gemini", "success")])
        self.assertEqual(router.sticky_candidate.key, "host-gemini")

    def test_channel_only_result_mismatch_fails_closed_without_committing_page(self):
        router = SerialStickyRouter()
        router.route_page(1, TranscriptExecutor({"host-openai": "success"}))
        report_before = router.report()

        def wrong_channel(candidate):
            return GenerationResult(
                GenerationStatus.SUCCESS,
                candidate.provider,
                "api" if candidate.channel == "host" else "host",
                "/workspace/generated/mismatched.png",
            )

        with self.assertRaises(RuntimeError):
            router.route_page(2, wrong_channel)

        self.assertEqual(router.report(), report_before)
        retry = TranscriptExecutor({"host-openai": "success"})
        router.route_page(2, retry)
        self.assertEqual(retry.calls, [("host-openai", "success")])


if __name__ == "__main__":
    unittest.main()
