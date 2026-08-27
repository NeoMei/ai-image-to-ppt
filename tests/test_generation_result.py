import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generation_result import (
    FATAL_STATUSES,
    FALLBACK_STATUSES,
    GenerationResult,
    GenerationStatus,
    classify_http_failure,
    safe_message,
)


class GenerationResultTests(unittest.TestCase):
    def test_status_partition_is_complete_and_disjoint(self):
        all_failures = set(GenerationStatus) - {GenerationStatus.SUCCESS}
        self.assertEqual(FALLBACK_STATUSES | FATAL_STATUSES, all_failures)
        self.assertFalse(FALLBACK_STATUSES & FATAL_STATUSES)

    def test_result_serializes_stable_public_fields(self):
        result = GenerationResult(
            GenerationStatus.SUCCESS,
            "openai",
            "host",
            "/workspace/out/slide.png",
            "generated",
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.can_fallback)
        self.assertEqual(
            json.loads(result.to_json()),
            {
                "status": "success",
                "provider": "openai",
                "channel": "host",
                "output_path": "/workspace/out/slide.png",
                "safe_message": "generated",
            },
        )

    def test_only_fallback_statuses_allow_fallback(self):
        for status in GenerationStatus:
            output_path = "/workspace/out/slide.png" if status is GenerationStatus.SUCCESS else None
            result = GenerationResult(status, "gemini", "api", output_path)
            self.assertEqual(result.can_fallback, status in FALLBACK_STATUSES)

    def test_http_classifier_uses_codes_before_messages(self):
        policy = {"content_policy_violation"}
        retryable = {"insufficient_quota"}
        self.assertEqual(
            classify_http_failure(400, "content_policy_violation", policy, retryable),
            GenerationStatus.POLICY_REFUSED,
        )
        self.assertEqual(
            classify_http_failure(400, "insufficient_quota", policy, retryable),
            GenerationStatus.RETRYABLE_EXHAUSTED,
        )
        self.assertEqual(
            classify_http_failure(401, None, policy, retryable),
            GenerationStatus.AUTH_UNAVAILABLE,
        )
        self.assertEqual(
            classify_http_failure(429, None, policy, retryable),
            GenerationStatus.RETRYABLE_EXHAUSTED,
        )
        self.assertEqual(
            classify_http_failure(400, None, policy, retryable),
            GenerationStatus.INVALID_INPUT,
        )

    def test_safe_message_removes_known_and_key_like_secrets(self):
        result = safe_message(
            "Bearer known-key and sk-remoteSecret123",
            secrets=("known-key",),
        )
        self.assertNotIn("known-key", result)
        self.assertNotIn("sk-remoteSecret123", result)
        self.assertLessEqual(len(result), 300)

    def test_result_rejects_invalid_contract_values(self):
        cases = (
            (GenerationStatus.SUCCESS, "other", "host", "/workspace/out/slide.png"),
            (GenerationStatus.SUCCESS, "openai", "other", "/workspace/out/slide.png"),
            (GenerationStatus.INVALID_INPUT, "openai", "host", "/workspace/out/slide.png"),
            (GenerationStatus.SUCCESS, "openai", "host", None),
            (GenerationStatus.SUCCESS, "openai", "host", "relative/slide.png"),
        )
        for status, provider, channel, output_path in cases:
            with self.subTest(status=status, provider=provider, channel=channel):
                with self.assertRaises(ValueError):
                    GenerationResult(status, provider, channel, output_path)

    def test_result_rejects_non_enum_status_values(self):
        for status in ("unknown", "success"):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    GenerationResult(status, "openai", "host")

    def test_safe_message_redacts_overlapping_known_secrets(self):
        result = safe_message("token=abcdef", secrets=("abc", "abcdef"))
        self.assertNotIn("abcdef", result)
        self.assertNotIn("def", result)


if __name__ == "__main__":
    unittest.main()
