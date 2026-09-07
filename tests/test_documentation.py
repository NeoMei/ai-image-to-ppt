import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_capability_manifest_matches_the_superppt_v3_contract(self):
        manifest = json.loads((ROOT / "references" / "capabilities.json").read_text(encoding="utf-8"))
        self.assertIn("references/capabilities.json", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["skill"], "ai-image-to-ppt")
        self.assertEqual(
            manifest["contracts"],
            {
                "generationResult": 1,
                "serialStickyRouterReport": 1,
                "hostImageImport": 1,
                "editableInput": 1,
            },
        )
        self.assertEqual(
            set(manifest["scripts"].values()),
            {
                "scripts/generation_result.py",
                "scripts/host_routing_policy.py",
                "scripts/import_host_image.py",
                "scripts/prepare_editable_input.py",
                "scripts/gen_slide.py",
                "scripts/export_images.py",
            },
        )
        for relative_path in manifest["scripts"].values():
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_windows_test_gate_is_documented_and_present(self):
        self.assertIn("scripts/run_windows_tests.py", self.readme)
        self.assertTrue((ROOT / "scripts" / "run_windows_tests.py").is_file())

    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    def test_skill_discovers_and_requires_the_host_routing_reference(self):
        self.assertTrue((ROOT / "references/host-image-routing.md").is_file())
        self.assertIn("references/host-image-routing.md", self.skill)
        self.assertIn("Before generating any slide image, read", self.skill)

    def test_manifest_is_documented_as_static_authority_not_live_availability(self):
        for document in (self.readme, self.skill):
            normalized = " ".join(document.split()).lower()
            self.assertIn("references/capabilities.json", document)
            self.assertIn("static machine-readable capability contract", normalized)
            self.assertIn("authoritative", normalized)
            self.assertIn("credentials", normalized)
            self.assertIn("real-time host/api availability", normalized)

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

    def test_host_raw_master_failure_semantics_are_not_claimed_as_crash_atomic(self):
        normalized = " ".join(self.readme.split()).lower()
        self.assertIn("compensating rollback", normalized)
        self.assertIn("not a crash-atomic two-file commit", normalized)

    def test_readme_keeps_provider_and_safety_recovery_boundaries(self):
        normalized = " ".join(self.readme.split())
        for expected in (
            "`doubao-seedream-5-0-260128`",
            "14 MiB",
            "bounded `Retry-After`",
            "Parent-directory replacement",
            "On POSIX, file and directory fsync cover process crashes and power-loss metadata recovery.",
            "On Windows, recovery covers process crashes only and does not promise power-loss durability.",
            "API single-image outputs use conditional, ownership-preserving publication",
            "This is not a single-syscall atomic replacement",
            "Only deck export has an export journal",
            "secure publication primitives",
            "never runs automatically",
            "offline precondition is the safety boundary for pathname races",
            "Filesystem failures stop cleanup, return exit 1 without a traceback, and do not roll back earlier removals.",
        ):
            self.assertIn(expected, normalized)


if __name__ == "__main__":
    unittest.main()
