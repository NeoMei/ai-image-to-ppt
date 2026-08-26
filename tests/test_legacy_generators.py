import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class LegacyGeneratorCredentialTests(unittest.TestCase):
    def _load_without_credentials(self, script_name, home):
        module_name = f"test_{script_name}_{id(home)}"
        spec = importlib.util.spec_from_file_location(
            module_name, SCRIPTS / f"{script_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        try:
            with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=True):
                spec.loader.exec_module(module)
        except (OSError, UnicodeError) as error:
            self.fail(f"{script_name} raised while importing without credentials: {error}")
        return module

    def test_import_and_call_without_fallback_credentials_returns_false(self):
        for script_name, secret_name in (
            ("gen_slide_gemini", "gemini_api_key"),
            ("gen_slide_doubao", "doubao_api_key"),
        ):
            with self.subTest(script_name=script_name), tempfile.TemporaryDirectory() as temp_dir:
                home = Path(temp_dir)
                module = self._load_without_credentials(script_name, home)
                output = io.StringIO()
                with mock.patch.object(module.urllib.request, "urlopen") as urlopen, \
                     redirect_stdout(output):
                    result = module.gen("prompt", str(home / "slide.jpg"), retries=0)

                self.assertFalse(result)
                urlopen.assert_not_called()
                self.assertIn(str(Path("~/.secrets") / secret_name), output.getvalue())
                self.assertNotIn("Traceback", output.getvalue())

    def test_direct_calls_reject_negative_retries_before_network(self):
        for script_name in ("gen_slide_gemini", "gen_slide_doubao"):
            with self.subTest(script_name=script_name), tempfile.TemporaryDirectory() as temp_dir:
                home = Path(temp_dir)
                module = self._load_without_credentials(script_name, home)
                with mock.patch.object(module.urllib.request, "urlopen") as urlopen:
                    self.assertFalse(
                        module.gen("prompt", str(home / "slide.jpg"), retries=-1)
                    )
                urlopen.assert_not_called()

    def test_legacy_provider_docstrings_do_not_claim_preference_or_quota_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            gemini = self._load_without_credentials("gen_slide_gemini", home)
            doubao = self._load_without_credentials("gen_slide_doubao", home)

        self.assertNotIn("首选", gemini.__doc__)
        self.assertNotIn("配额耗尽", doubao.__doc__)

    def test_remote_errors_do_not_print_fallback_provider_keys(self):
        for script_name, secret_name in (
            ("gen_slide_gemini", "gemini_api_key"),
            ("gen_slide_doubao", "doubao_api_key"),
        ):
            with self.subTest(script_name=script_name), tempfile.TemporaryDirectory() as temp_dir:
                home = Path(temp_dir)
                secret_dir = home / ".secrets"
                secret_dir.mkdir()
                key = "known-fallback-key"
                (secret_dir / secret_name).write_text(key, encoding="utf-8")
                module = self._load_without_credentials(script_name, home)
                message = f"request exposed {key} and sk-remoteLegacy123"
                error = urllib.error.HTTPError(
                    "https://provider.invalid",
                    400,
                    "bad request",
                    {},
                    io.BytesIO(
                        json.dumps({"error": {"message": message}}).encode("utf-8")
                    ),
                )
                output = io.StringIO()
                with mock.patch.object(
                    module.urllib.request, "urlopen", side_effect=error
                ), mock.patch.object(module.time, "sleep"), redirect_stdout(output):
                    self.assertFalse(
                        module.gen("prompt", str(home / "slide.jpg"), retries=0)
                    )

                self.assertNotIn(key, output.getvalue())
                self.assertNotIn("sk-remoteLegacy123", output.getvalue())


if __name__ == "__main__":
    unittest.main()
