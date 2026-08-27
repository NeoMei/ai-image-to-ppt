import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    def test_install_commands_pin_minimum_pillow_version(self):
        for document in (self.readme, self.skill):
            self.assertIn("python3 -m pip install -r requirements.txt", document)
            self.assertNotIn("\npip install", document)

        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("Pillow>=9.1", requirements)
        self.assertIn("python-pptx>=1.0", requirements)

    def test_development_validator_dependency_and_command_are_documented(self):
        dev_requirements = (ROOT / "requirements-dev.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("PyYAML>=6.0", dev_requirements)
        self.assertIn(
            "python3 -m pip install -r requirements-dev.txt",
            self.readme,
        )
        self.assertIn("python3 scripts/validate_skill.py .", self.readme)
        self.assertNotIn("/Users/neomei/", self.readme)

    def test_fallback_secret_file_setup_commands_are_copy_pasteable(self):
        for document in (self.readme, self.skill):
            for secret_name, placeholder in (
                ("gemini_api_key", "YOUR_GEMINI_KEY"),
                ("doubao_api_key", "YOUR_DOUBAO_KEY"),
            ):
                self.assertIn(
                    f'printf \'%s\\n\' "{placeholder}" > ~/.secrets/{secret_name}',
                    document,
                )
                self.assertIn(f"chmod 600 ~/.secrets/{secret_name}", document)

    def test_quick_start_creates_output_directory(self):
        self.assertIn("mkdir -p out", self.readme)

    def test_export_examples_use_one_explicit_mixed_format_manifest(self):
        for document in (self.readme, self.skill):
            self.assertNotIn("out/*.jpg", document)
            self.assertIn("SLIDES = [", document)
            self.assertIn('"out/slide_01.jpg"', document)
            self.assertIn('"out/slide_02.png"', document)
            self.assertIn('"out/slide_03.webp"', document)
            self.assertIn("len(SLIDES) != len(set(SLIDES))", document)
            self.assertIn("*SLIDES", document)
            self.assertIn("USE_CLI = True", document)
            self.assertIn("if USE_CLI:", document)
            self.assertIn("else:", document)
            self.assertIn("from export_images import export_deck", document)
            self.assertIn('if not export_deck(SLIDES, "deck_name"):', document)
            self.assertIn('raise SystemExit("deck export failed")', document)
            self.assertNotIn("from export_images import export_pdf", document)
            self.assertNotIn('export_pdf(SLIDES, "deck.pdf")', document)
            self.assertNotIn('export_pptx(SLIDES, "deck.pptx")', document)

    def test_export_manifest_rejects_duplicate_slide_ids_across_suffixes(self):
        for document in (self.readme, self.skill):
            self.assertIn(
                "SLIDE_IDS = [Path(path).stem for path in SLIDES]",
                document,
            )
            self.assertIn(
                "len(SLIDE_IDS) != len(set(SLIDE_IDS))",
                document,
            )

    def test_skill_discovery_names_the_primary_openai_engine(self):
        frontmatter = self.skill.split("---", 2)[1]
        self.assertIn("OpenAI", frontmatter)
        self.assertIn("GPT Image 2", frontmatter)
        self.assertIn("gpt-image-2", frontmatter)

    def test_batch_example_reports_boolean_result(self):
        self.assertIn('return name, "cached"', self.skill)
        self.assertIn('return name, "done" if ok else "failed"', self.skill)
        self.assertNotIn('print(f.result(), "done")', self.skill)

    def test_output_extension_claim_is_provider_specific(self):
        for document in (self.readme, self.skill):
            normalized = " ".join(document.split())
            self.assertIn("`.jpg`, `.jpeg`, `.png`, and `.webp`", document)
            self.assertIn(
                "Only Gemini validates the provider-declared response MIME type.",
                normalized,
            )
            self.assertIn(
                "All providers validate the actual image encoding and strict "
                "16:9 dimensions before publication.",
                normalized,
            )
            self.assertNotIn("every provider's returned MIME type", document.lower())
            self.assertNotIn("provider-returned encoding", document)

    def test_current_gemini_models_are_consistent(self):
        for document in (self.readme, self.skill):
            self.assertIn("`gemini-3.1-flash-image`", document)
            self.assertIn("`gemini-3.6-flash`", document)
            self.assertNotIn("gemini-3.1-flash-image-preview", document)
            self.assertNotIn("gemini-2.0-flash", document)

    def test_safe_existing_output_and_force_contract_is_documented(self):
        for document in (self.readme, self.skill):
            self.assertIn("--force", document)
            self.assertIn("refuses existing", document)
            normalized = " ".join(document.split())
            self.assertIn(
                "Generation outputs and inputs to ordinary export or editable "
                "preparation are capped at 50 MiB and 64 megapixels.",
                normalized,
            )
            self.assertIn(
                "Vision-check inputs have a separate 14 MiB limit.",
                normalized,
            )
            self.assertNotIn("images are limited to 50 MiB", normalized)
            self.assertNotIn("Generated and local inputs are capped", normalized)
            self.assertIn("Retry-After", document)

    def test_pair_export_is_documented_as_crash_recoverable(self):
        for document in (self.readme, self.skill):
            normalized = " ".join(document.split())
            self.assertIn("recover", normalized.lower())
            self.assertIn("next export", normalized.lower())
            self.assertIn(
                "On POSIX, file and directory fsync cover process crashes and "
                "power-loss metadata recovery.",
                normalized,
            )
            self.assertIn(
                "On Windows, recovery covers process crashes only and does not "
                "promise power-loss durability.",
                normalized,
            )
            self.assertNotIn("transactionally", normalized.lower())
            self.assertNotIn("transactional pair", normalized.lower())

    def test_stable_output_lock_cleanup_is_explicitly_offline(self):
        for document in (self.readme, self.skill):
            normalized = " ".join(document.split())
            self.assertIn("cleanup_output_locks.py --older-than-days 30", document)
            self.assertIn(
                "Stable hashed lock files intentionally persist",
                normalized,
            )
            self.assertIn(
                "ONLY while no generation or export process is running",
                normalized,
            )
            self.assertIn("never runs automatically", normalized)
            self.assertIn(
                "offline precondition is the safety boundary for pathname races",
                normalized,
            )
            self.assertIn(
                "Filesystem failures stop cleanup, return exit 1 without a "
                "traceback, and do not roll back earlier removals.",
                normalized,
            )


if __name__ == "__main__":
    unittest.main()
