import sys
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide


class RouterTests(unittest.TestCase):
    def test_openai_is_the_default_engine(self):
        provider_gen = mock.Mock(return_value=True)
        with mock.patch.object(
            gen_slide.importlib,
            "import_module",
            return_value=SimpleNamespace(gen=provider_gen),
        ) as import_module:
            self.assertTrue(gen_slide.gen("prompt", "slide.jpg"))

        import_module.assert_called_once_with("gen_slide_openai")
        provider_gen.assert_called_once_with("prompt", "slide.jpg", retries=2)

    def test_each_named_engine_is_lazy_loaded(self):
        expected = {
            "openai": "gen_slide_openai",
            "gemini": "gen_slide_gemini",
            "doubao": "gen_slide_doubao",
        }
        for engine, module_name in expected.items():
            with self.subTest(engine=engine):
                provider_gen = mock.Mock(return_value=True)
                with mock.patch.object(
                    gen_slide.importlib,
                    "import_module",
                    return_value=SimpleNamespace(gen=provider_gen),
                ) as import_module:
                    self.assertTrue(gen_slide.gen("prompt", "slide.jpg", engine=engine, retries=4))
                import_module.assert_called_once_with(module_name)
                provider_gen.assert_called_once_with("prompt", "slide.jpg", retries=4)

    def test_unknown_engine_fails_without_importing(self):
        with mock.patch.object(gen_slide.importlib, "import_module") as import_module:
            self.assertFalse(gen_slide.gen("prompt", "slide.jpg", engine="unknown"))
        import_module.assert_not_called()

    def test_cli_defaults_to_openai(self):
        with mock.patch.object(gen_slide, "gen", return_value=True) as generate:
            exit_code = gen_slide.main(["slide.jpg", "draw a slide"])
        self.assertEqual(exit_code, 0)
        generate.assert_called_once_with(
            "draw a slide", "slide.jpg", engine="openai", retries=2
        )

    def test_cli_passes_explicit_engine_and_retry_count(self):
        with mock.patch.object(gen_slide, "gen", return_value=False) as generate:
            exit_code = gen_slide.main([
                "slide.png",
                "draw a slide",
                "--engine",
                "gemini",
                "--retries",
                "5",
            ])
        self.assertEqual(exit_code, 1)
        generate.assert_called_once_with(
            "draw a slide", "slide.png", engine="gemini", retries=5
        )

    def test_python_router_rejects_negative_retries_before_importing(self):
        with mock.patch.object(gen_slide.importlib, "import_module") as import_module:
            self.assertFalse(
                gen_slide.gen("prompt", "slide.jpg", engine="openai", retries=-1)
            )
        import_module.assert_not_called()

    def test_cli_rejects_negative_retries_with_usage_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised, \
             mock.patch.object(gen_slide, "gen") as generate:
            gen_slide.main(["slide.jpg", "draw a slide", "--retries", "-1"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("non-negative", stderr.getvalue())
        generate.assert_not_called()

    def test_cli_help_describes_provider_specific_output_behavior(self):
        help_text = gen_slide._parser().format_help()
        self.assertIn("OpenAI", help_text)
        self.assertIn("legacy", help_text.lower())
        self.assertIn("provider-returned", help_text)


if __name__ == "__main__":
    unittest.main()
