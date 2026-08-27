import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    def test_skill_discovers_and_requires_the_host_routing_reference(self):
        self.assertTrue((ROOT / "references/host-image-routing.md").is_file())
        self.assertIn("references/host-image-routing.md", self.skill)
        self.assertIn("Before generating any slide image, read", self.skill)

    def test_host_openai_is_the_default_without_a_key_requirement(self):
        skill = " ".join(self.skill.split())
        readme = " ".join(self.readme.split())
        self.assertIn("host's callable OpenAI image tool", skill)
        self.assertIn("does not require an `OPENAI_API_KEY`", skill)
        self.assertIn("no API key is required", readme)

    def test_all_api_credentials_remain_optional_and_copy_pasteable(self):
        for document in (self.readme, self.skill):
            self.assertIn("may be skipped", " ".join(document.split()))
            for secret_name, placeholder in (
                ("openai_api_key", "YOUR_OPENAI_KEY"),
                ("gemini_api_key", "YOUR_GEMINI_KEY"),
                ("doubao_api_key", "YOUR_DOUBAO_KEY"),
            ):
                self.assertIn(f'printf \'%s\\n\' "{placeholder}" > ~/.secrets/{secret_name}', document)
                self.assertIn(f"chmod 600 ~/.secrets/{secret_name}", document)

    def test_cli_is_explicit_api_only_and_optional_commands_remain(self):
        for document in (self.readme, self.skill):
            self.assertIn("API/CLI-only", document)
            self.assertIn("scripts/gen_slide.py", document)
            self.assertIn("--engine gemini", document)
            self.assertIn("--engine doubao", document)

    def test_batch_is_serial_sticky_and_never_uses_concurrent_generation(self):
        normalized = " ".join(self.skill.split()).lower()
        self.assertIn("serial", normalized)
        self.assertIn("sticky", normalized)
        self.assertIn("cached pages do not change routing", normalized)
        self.assertNotIn("ThreadPoolExecutor", self.skill)
        self.assertNotIn("Batch generate with concurrency", self.skill)

    def test_editable_handoff_remains_a_separate_exact_png(self):
        for document in (self.readme, self.skill):
            self.assertIn("1280×720 PNG", document)
            self.assertIn("prepare_editable_input.py", document)

    def test_host_near_ratio_normalization_contract_is_discoverable(self):
        reference = (ROOT / "references/host-image-routing.md").read_text(
            encoding="utf-8"
        )
        for document in (reference, self.readme, self.skill):
            normalized = " ".join(document.split()).lower()
            self.assertIn("0.5%", normalized)
            self.assertIn("raw", normalized)
            self.assertIn("center-crop", normalized)
            self.assertIn("strict 16:9", normalized)
        self.assertIn("raw/<filename>", reference)
        self.assertIn("do not stretch", " ".join(reference.split()).lower())

    def test_readme_keeps_provider_and_safety_recovery_boundaries(self):
        normalized = " ".join(self.readme.split())
        for expected in (
            "`doubao-seedream-5-0-260128`",
            "14 MiB",
            "bounded `Retry-After`",
            "Parent-directory replacement",
            "On POSIX, file and directory fsync cover process crashes and power-loss metadata recovery.",
            "On Windows, recovery covers process crashes only and does not promise power-loss durability.",
            "rollback and crash-recovery protection",
            "never runs automatically",
            "offline precondition is the safety boundary for pathname races",
            "Filesystem failures stop cleanup, return exit 1 without a traceback, and do not roll back earlier removals.",
        ):
            self.assertIn(expected, normalized)


if __name__ == "__main__":
    unittest.main()
