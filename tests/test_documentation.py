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
        self.assertIn("quick_validate.py", self.readme)

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

    def test_batch_example_reports_boolean_result(self):
        self.assertIn('return name, "cached"', self.skill)
        self.assertIn('return name, "done" if ok else "failed"', self.skill)
        self.assertNotIn('print(f.result(), "done")', self.skill)

    def test_output_extension_claim_is_provider_specific(self):
        for document in (self.readme, self.skill):
            self.assertIn("`.jpg`, `.jpeg`, `.png`, and `.webp`", document)
            self.assertIn("validated before publication", document)
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


if __name__ == "__main__":
    unittest.main()
