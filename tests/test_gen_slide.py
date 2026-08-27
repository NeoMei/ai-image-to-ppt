import sys
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide
from generation_result import GenerationResult, GenerationStatus


class RouterTests(unittest.TestCase):
    def test_generate_result_lazy_loads_provider_structured_entrypoint(self):
        expected = GenerationResult(
            GenerationStatus.AUTH_UNAVAILABLE,
            "openai",
            "api",
            safe_message="missing key",
        )
        provider = SimpleNamespace(generate_result=mock.Mock(return_value=expected))
        with mock.patch.object(
            gen_slide.importlib,
            "import_module",
            return_value=provider,
        ) as import_module:
            result = gen_slide.generate_result("prompt", "slide.jpg")

        self.assertEqual(result, expected)
        import_module.assert_called_once_with("gen_slide_openai")
        provider.generate_result.assert_called_once_with(
            "prompt", "slide.jpg", retries=2, overwrite=False, progress=None
        )

    def test_boolean_wrapper_uses_structured_result_for_each_named_engine(self):
        expected = {
            "openai": "gen_slide_openai",
            "gemini": "gen_slide_gemini",
            "doubao": "gen_slide_doubao",
        }
        for engine, module_name in expected.items():
            with self.subTest(engine=engine):
                provider_result = GenerationResult(
                    GenerationStatus.SUCCESS,
                    engine,
                    "api",
                    output_path="/workspace/slide.jpg",
                )
                generate_result = mock.Mock(return_value=provider_result)
                with mock.patch.object(
                    gen_slide.importlib,
                    "import_module",
                    return_value=SimpleNamespace(generate_result=generate_result),
                ) as import_module:
                    self.assertTrue(
                        gen_slide.gen("prompt", "slide.jpg", engine=engine, retries=4)
                    )
                import_module.assert_called_once_with(module_name)
                generate_result.assert_called_once_with(
                    "prompt", "slide.jpg", retries=4, overwrite=False, progress=None
                )

    def test_unknown_engine_raises_for_structured_call_and_boolean_wrapper_fails_closed(self):
        with mock.patch.object(gen_slide.importlib, "import_module") as import_module:
            with self.assertRaises(ValueError):
                gen_slide.generate_result("prompt", "slide.jpg", engine="unknown")
            self.assertFalse(gen_slide.gen("prompt", "slide.jpg", engine="unknown"))
        import_module.assert_not_called()

    def test_cli_human_mode_prints_one_safe_terminal_summary_and_keeps_exit_compatibility(self):
        result = GenerationResult(
            GenerationStatus.SUCCESS,
            "openai",
            "api",
            output_path="/workspace/slide.jpg",
            safe_message="done",
        )
        stdout = io.StringIO()
        with mock.patch.object(gen_slide, "generate_result", return_value=result) as generate, \
             redirect_stdout(stdout):
            exit_code = gen_slide.main(["slide.jpg", "draw a slide"])
        self.assertEqual(exit_code, 0)
        generate.assert_called_once_with(
            "draw a slide",
            "slide.jpg",
            engine="openai",
            retries=2,
            overwrite=False,
            progress=None,
        )
        self.assertEqual(stdout.getvalue().count("\n"), 1)
        self.assertIn("OK:", stdout.getvalue())

    def test_json_cli_emits_one_object_and_keeps_default_exit_compatibility(self):
        result = GenerationResult(
            GenerationStatus.AUTH_UNAVAILABLE,
            "openai",
            "api",
            safe_message="missing key",
        )
        stdout = io.StringIO()
        with mock.patch.object(gen_slide, "generate_result", return_value=result), \
             redirect_stdout(stdout):
            exit_code = gen_slide.main(["slide.jpg", "prompt", "--json"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "auth_unavailable")
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)

    def test_cli_passes_explicit_engine_and_retry_count(self):
        result = GenerationResult(
            GenerationStatus.POLICY_REFUSED,
            "gemini",
            "api",
            safe_message="refused",
        )
        with mock.patch.object(gen_slide, "generate_result", return_value=result) as generate:
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
            "draw a slide",
            "slide.png",
            engine="gemini",
            retries=5,
            overwrite=False,
            progress=None,
        )

    def test_cli_force_requests_explicit_overwrite(self):
        result = GenerationResult(
            GenerationStatus.SUCCESS,
            "openai",
            "api",
            output_path="/workspace/slide.png",
        )
        with mock.patch.object(gen_slide, "generate_result", return_value=result) as generate:
            exit_code = gen_slide.main(["slide.png", "prompt", "--force"])

        self.assertEqual(exit_code, 0)
        generate.assert_called_once_with(
            "prompt",
            "slide.png",
            engine="openai",
            retries=2,
            overwrite=True,
            progress=None,
        )

    def test_structured_router_rejects_invalid_retries_before_importing(self):
        with mock.patch.object(gen_slide.importlib, "import_module") as import_module:
            result = gen_slide.generate_result(
                "prompt", "slide.jpg", engine="openai", retries=-1
            )
        self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
        import_module.assert_not_called()

    def test_structured_router_rejects_invalid_retry_types_before_importing(self):
        for retries in ("2", 1.5, None, True, [], -1, 11):
            with self.subTest(retries=retries), mock.patch.object(
                gen_slide.importlib, "import_module"
            ) as import_module:
                result = gen_slide.generate_result(
                    "prompt", "slide.jpg", engine="openai", retries=retries
                )
                self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
                import_module.assert_not_called()

    def test_structured_router_rejects_invalid_overwrite_before_importing(self):
        with mock.patch.object(gen_slide.importlib, "import_module") as import_module:
            result = gen_slide.generate_result(
                "prompt", "slide.jpg", overwrite="yes"
            )
        self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
        import_module.assert_not_called()

    def test_import_failure_is_unavailable_for_the_requested_provider(self):
        with mock.patch.object(
            gen_slide.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("missing dependency"),
        ):
            result = gen_slide.generate_result("prompt", "slide.jpg", engine="gemini")
        self.assertEqual(result.status, GenerationStatus.UNAVAILABLE)
        self.assertEqual(result.provider, "gemini")

    def test_unexpected_provider_failure_is_local_failure_without_secret_echo(self):
        provider = SimpleNamespace(
            generate_result=mock.Mock(side_effect=OSError("failed with sk-secret-value"))
        )
        with mock.patch.object(
            gen_slide.importlib, "import_module", return_value=provider
        ):
            result = gen_slide.generate_result("prompt", "slide.jpg")

        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
        self.assertNotIn("sk-secret-value", result.safe_message)

    def test_non_result_provider_return_fails_closed(self):
        provider = SimpleNamespace(generate_result=mock.Mock(return_value=True))
        with mock.patch.object(gen_slide.importlib, "import_module", return_value=provider):
            result = gen_slide.generate_result("prompt", "slide.jpg")
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)

    def test_json_cli_success_includes_absolute_output_path(self):
        result = GenerationResult(
            GenerationStatus.SUCCESS,
            "doubao",
            "api",
            output_path="/workspace/slide.jpg",
            safe_message="done",
        )
        stdout = io.StringIO()
        with mock.patch.object(gen_slide, "generate_result", return_value=result), \
             redirect_stdout(stdout):
            exit_code = gen_slide.main(["slide.jpg", "prompt", "--json"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["output_path"], "/workspace/slide.jpg")

    def test_cli_rejects_negative_retries_with_usage_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised, \
             mock.patch.object(gen_slide, "generate_result") as generate:
            gen_slide.main(["slide.jpg", "draw a slide", "--retries", "-1"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("non-negative", stderr.getvalue())
        generate.assert_not_called()

    def test_cli_rejects_retries_above_shared_limit_with_usage_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised, \
             mock.patch.object(gen_slide, "generate_result") as generate:
            gen_slide.main(["slide.jpg", "draw a slide", "--retries", "11"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("at most 10", stderr.getvalue())
        generate.assert_not_called()

    def test_cli_rejects_unknown_engine_before_calling_structured_router(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised, \
             mock.patch.object(gen_slide, "generate_result") as generate:
            gen_slide.main(["slide.jpg", "draw a slide", "--engine", "unknown"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())
        generate.assert_not_called()

    def test_cli_help_describes_provider_specific_output_behavior(self):
        help_text = " ".join(gen_slide._parser().format_help().split())
        self.assertIn("OpenAI/Doubao: .jpg/.jpeg/.png/.webp", help_text)
        self.assertIn("Gemini: .png (default), .jpg/.jpeg", help_text)
        self.assertIn("JPEG suffix requests JPEG", help_text)
        self.assertIn("actual encoding and strict 16:9", help_text)
        self.assertIn(f"0..{gen_slide.MAX_RETRIES}", help_text)
        self.assertIn("API/CLI adapter", help_text)
        self.assertIn("does not use host capabilities", help_text)
        self.assertNotIn("legacy", help_text.lower())
        self.assertNotIn("provider-returned", help_text)


if __name__ == "__main__":
    unittest.main()
