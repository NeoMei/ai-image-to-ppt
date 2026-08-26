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
            self.assertIn("Pillow>=9.1", document)
            self.assertNotIn("pip install Pillow python-pptx", document)

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
        self.assertIn("return name, ok", self.skill)
        self.assertIn('status = "done" if ok else "failed"', self.skill)
        self.assertNotIn('print(f.result(), "done")', self.skill)

    def test_output_extension_claim_is_provider_specific(self):
        for document in (self.readme, self.skill):
            self.assertIn("OpenAI output extensions", document)
            self.assertIn("provider-returned encoding", document)


if __name__ == "__main__":
    unittest.main()
