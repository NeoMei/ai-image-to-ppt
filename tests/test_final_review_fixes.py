import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_slide
import gen_slide_doubao
import gen_slide_gemini
import gen_slide_openai
import image_output
import import_host_image
from generation_result import GenerationResult, GenerationStatus, safe_message


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class JsonResponse:
    def __init__(self, payload, content_type=None):
        self.stream = io.BytesIO(json.dumps(payload).encode("utf-8"))
        self.headers = {}
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class BytesResponse:
    def __init__(self, data, content_type):
        self.stream = io.BytesIO(data)
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


class ConditionalPublicationTests(unittest.TestCase):
    def test_api_force_preserves_external_replacement_in_actual_rename_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            target.write_bytes(image_bytes())
            external = b"external-at-api-rename-boundary"
            external_identity = None
            raced = False
            real_rename = image_output.os.rename

            def race_inside_rename(source, destination, *args, **kwargs):
                nonlocal external_identity, raced
                if not raced:
                    raced = True
                    target.unlink()
                    target.write_bytes(external)
                    current = target.stat()
                    external_identity = (current.st_dev, current.st_ino)
                return real_rename(source, destination, *args, **kwargs)

            generated = image_bytes()
            response = JsonResponse({
                "data": [{
                    "b64_json": base64.b64encode(generated).decode("ascii")
                }]
            })
            with mock.patch.object(
                gen_slide_openai, "_load_api_key", return_value="test-key"
            ), mock.patch.object(
                gen_slide_openai.urllib.request,
                "urlopen",
                return_value=response,
            ), mock.patch.object(
                image_output.os,
                "rename",
                side_effect=race_inside_rename,
            ):
                result = gen_slide_openai.generate_result(
                    "prompt", str(target), retries=0, overwrite=True
                )

            self.assertTrue(raced)
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(target.read_bytes(), external)
            self.assertEqual(
                (target.stat().st_dev, target.stat().st_ino), external_identity
            )

    def test_host_force_preserves_master_replacement_inside_actual_rename_and_compensates_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            original_master = image_bytes("PNG")
            original_raw = image_bytes("PNG")
            master.write_bytes(original_master)
            raw.write_bytes(original_raw)
            external = b"external-at-host-master-rename-boundary"
            external_identity = None
            raced = False
            raw_published = False
            real_rename = image_output.os.rename
            master_parent = master.parent.stat()

            def record_checkpoint(phase, _snapshot):
                nonlocal raw_published
                if phase == "raw":
                    raw_published = True

            def race_master_rename(source, destination, *args, **kwargs):
                nonlocal external_identity, raced
                source_parent = os.fstat(kwargs["src_dir_fd"])
                if (
                    raw_published
                    and not raced
                    and source_parent.st_dev == master_parent.st_dev
                    and source_parent.st_ino == master_parent.st_ino
                    and (source == master.name or destination == master.name)
                ):
                    raced = True
                    master.unlink()
                    master.write_bytes(external)
                    current = master.stat()
                    external_identity = (current.st_dev, current.st_ino)
                return real_rename(source, destination, *args, **kwargs)

            with mock.patch.object(
                image_output.os,
                "rename",
                side_effect=race_master_rename,
            ), mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=record_checkpoint,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        image_bytes("PNG"), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertTrue(raced)
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), external)
            self.assertEqual(
                (master.stat().st_dev, master.stat().st_ino), external_identity
            )
            self.assertEqual(raw.read_bytes(), original_raw)

    def test_host_force_preserves_external_raw_replacement_before_first_member(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            original_master = image_bytes("PNG")
            master.write_bytes(original_master)
            raw.write_bytes(image_bytes("PNG"))
            external = b"external-raw-replacement"
            real_publish = import_host_image._publish_transaction_member

            def race_then_publish(data, target, overwrite, strict):
                if not strict:
                    target.path.unlink()
                    target.path.write_bytes(external)
                return real_publish(data, target, overwrite, strict)

            with mock.patch.object(
                import_host_image,
                "_publish_transaction_member",
                side_effect=race_then_publish,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        image_bytes("PNG"), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(raw.read_bytes(), external)
            self.assertEqual(master.read_bytes(), original_master)

    def test_host_force_preserves_external_master_replacement_and_compensates_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            original_master = image_bytes("PNG")
            original_raw = image_bytes("PNG")
            master.write_bytes(original_master)
            raw.write_bytes(original_raw)
            external = b"external-master-replacement"

            def race_after_raw(phase, _snapshot):
                if phase == "raw":
                    master.unlink()
                    master.write_bytes(external)

            with mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=race_after_raw,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        image_bytes("PNG"), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), external)
            self.assertEqual(raw.read_bytes(), original_raw)

    def test_host_force_preserves_external_master_creation_and_compensates_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "out" / "raw" / "slide.png"
            master = root / "out" / "slide.png"
            external = b"external-master-creation"

            def race_after_raw(phase, _snapshot):
                if phase == "raw":
                    master.write_bytes(external)

            with mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=race_after_raw,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        image_bytes("PNG"), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), external)
            self.assertFalse(raw.exists())

    def _openai_result_with_publish_race(self, target, create_external):
        generated = image_bytes()
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}]
        })
        real_publish = gen_slide_openai.publish_bytes
        external = b"external-api-output"

        def race_then_publish(data, prepared, overwrite=False):
            if prepared.path.exists():
                prepared.path.unlink()
            prepared.path.write_bytes(external)
            return real_publish(data, prepared, overwrite=overwrite)

        if not create_external:
            target.write_bytes(image_bytes())
        with mock.patch.object(
            gen_slide_openai, "_load_api_key", return_value="test-key"
        ), mock.patch.object(
            gen_slide_openai.urllib.request,
            "urlopen",
            return_value=response,
        ), mock.patch.object(
            gen_slide_openai,
            "publish_bytes",
            side_effect=race_then_publish,
        ):
            result = gen_slide_openai.generate_result(
                "prompt", str(target), retries=0, overwrite=True
            )
        return result, external

    def test_api_force_preserves_external_replacement_after_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            result, external = self._openai_result_with_publish_race(
                target, create_external=False
            )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(target.read_bytes(), external)

    def test_api_force_preserves_external_creation_after_missing_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            result, external = self._openai_result_with_publish_race(
                target, create_external=True
            )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(target.read_bytes(), external)


class UnsupportedPublicationPlatformTests(unittest.TestCase):
    def test_missing_secure_primitive_capability_fails_closed_before_temp_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            image_output, "_SECURE_PUBLICATION_SUPPORTED", False
        ), mock.patch.object(image_output, "_temporary_path") as temporary:
            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "secure output publication primitives",
            ):
                image_output.publish_bytes(
                    image_bytes(), Path(temp_dir) / "slide.jpg"
                )
        temporary.assert_not_called()

    def test_shared_publication_wraps_unsupported_hard_link(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.jpg"
            with mock.patch.object(
                image_output.os,
                "link",
                side_effect=NotImplementedError("hard links unsupported"),
            ), self.assertRaises(image_output.ImageOutputError):
                image_output.publish_bytes(image_bytes(), target)

    def test_host_import_returns_structured_local_failure_for_unsupported_link(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            image_output.os,
            "link",
            side_effect=NotImplementedError("hard links unsupported"),
        ):
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    image_bytes("PNG"), "image/png"
                ),
                "slide.png",
                temp_dir,
                provider="openai",
            )
        self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)

    def test_all_direct_api_results_map_unsupported_publication_to_local_failure(self):
        providers = (gen_slide_openai, gen_slide_gemini, gen_slide_doubao)
        for provider in providers:
            with self.subTest(provider=provider.__name__), tempfile.TemporaryDirectory() as td, \
                 mock.patch.object(provider, "_gen_owned", side_effect=NotImplementedError("unsupported")):
                result = provider.generate_result(
                    "prompt", str(Path(td) / "slide.jpg"), retries=0
                )
                self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
                self.assertIsInstance(result, GenerationResult)

    def test_all_api_providers_map_post_image_unsupported_publish_to_local_failure(self):
        generated = image_bytes()
        provider_cases = (
            (
                gen_slide_openai,
                [JsonResponse({
                    "data": [{
                        "b64_json": base64.b64encode(generated).decode("ascii")
                    }]
                })],
            ),
            (
                gen_slide_gemini,
                [JsonResponse({
                    "candidates": [{
                        "content": {"parts": [{"inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(generated).decode("ascii"),
                        }}]}
                    }]
                })],
            ),
            (
                gen_slide_doubao,
                [
                    JsonResponse({"data": [{"url": "https://cdn.invalid/image"}]}),
                    BytesResponse(generated, "image/jpeg"),
                ],
            ),
        )
        for provider, responses in provider_cases:
            with self.subTest(provider=provider.__name__), tempfile.TemporaryDirectory() as td, \
                 mock.patch.object(provider, "_load_api_key", return_value="test-key"), \
                 mock.patch.object(provider.urllib.request, "urlopen", side_effect=responses), \
                 mock.patch.object(
                     provider,
                     "publish_bytes",
                     side_effect=NotImplementedError("unsupported publication"),
                 ):
                result = provider.generate_result(
                    "prompt", str(Path(td) / "slide.jpg"), retries=0
                )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertIsNone(result.output_path)


class DestructiveBoundaryTests(unittest.TestCase):
    def test_transaction_owned_removal_preserves_replacement_at_rename_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(image_bytes("PNG"))
            prepared = image_output.preflight_output(target, overwrite=True)
            expected = (target.stat().st_dev, target.stat().st_ino)
            external = b"external-at-owned-remove-unlink"
            external_identity = None
            real_rename = image_output.os.rename
            raced = False

            def race_rename(source, destination, *args, **kwargs):
                nonlocal external_identity, raced
                if not raced and source == target.name:
                    raced = True
                    os.unlink(source, dir_fd=kwargs["src_dir_fd"])
                    descriptor = os.open(
                        source,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=kwargs["src_dir_fd"],
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(external)
                    current = target.stat()
                    external_identity = (current.st_dev, current.st_ino)
                return real_rename(source, destination, *args, **kwargs)

            with mock.patch.object(
                image_output.os,
                "rename",
                side_effect=race_rename,
            ), self.assertRaises(image_output.ImageOutputError):
                import_host_image._remove_owned_output(prepared, expected)

            self.assertTrue(raced)
            self.assertEqual(target.read_bytes(), external)
            self.assertEqual(
                (target.stat().st_dev, target.stat().st_ino), external_identity
            )

    def test_opaque_restore_preserves_replacement_inside_actual_rename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"owned-current")
            prepared = image_output.preflight_output(target, overwrite=True)
            expected = (target.stat().st_dev, target.stat().st_ino)
            external = b"external-at-restore-rename"
            real_rename = image_output.os.rename
            raced = False

            def race_rename(source, destination, *args, **kwargs):
                nonlocal raced
                if not raced:
                    raced = True
                    target.unlink()
                    target.write_bytes(external)
                return real_rename(source, destination, *args, **kwargs)

            with mock.patch.object(
                image_output.os,
                "rename",
                side_effect=race_rename,
            ), self.assertRaises(image_output.ImageOutputError):
                image_output.publish_opaque_bytes_with_identity(
                    b"snapshot",
                    prepared,
                    expected_existing_identity=expected,
                )

            self.assertTrue(raced)
            self.assertEqual(target.read_bytes(), external)

    def test_temp_cleanup_preserves_replacement_at_rename_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.preflight_output(Path(temp_dir) / "slide.jpg")
            temporary = image_output._temporary_path(prepared)
            external = b"external-temp-replacement"
            external_identity = None
            real_rename = image_output.os.rename
            raced = False

            def race_rename(source, destination, *args, **kwargs):
                nonlocal external_identity, raced
                if not raced and source == temporary.name:
                    raced = True
                    os.unlink(source, dir_fd=kwargs["src_dir_fd"])
                    descriptor = os.open(
                        source,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=kwargs["src_dir_fd"],
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(external)
                    current = (Path(temp_dir) / temporary.name).stat()
                    external_identity = (current.st_dev, current.st_ino)
                return real_rename(source, destination, *args, **kwargs)

            try:
                with mock.patch.object(
                    image_output.os,
                    "rename",
                    side_effect=race_rename,
                ), self.assertRaises(image_output.ImageOutputError):
                    image_output._remove_temp(temporary)
                self.assertTrue(raced)
                self.assertEqual(
                    (Path(temp_dir) / temporary.name).read_bytes(), external
                )
                current = (Path(temp_dir) / temporary.name).stat()
                self.assertEqual(
                    (current.st_dev, current.st_ino), external_identity
                )
            finally:
                os.close(temporary.descriptor)
                os.close(temporary.parent_fd)

    def test_published_cleanup_preserves_replacement_at_rename_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.preflight_output(Path(temp_dir) / "slide.jpg")
            temporary = image_output._temporary_path(prepared)
            os.link(
                temporary.name,
                prepared.name,
                src_dir_fd=temporary.parent_fd,
                dst_dir_fd=temporary.parent_fd,
                follow_symlinks=False,
            )
            external = b"external-published-replacement"
            external_identity = None
            real_rename = image_output.os.rename
            raced = False

            def race_rename(source, destination, *args, **kwargs):
                nonlocal external_identity, raced
                if not raced and source == prepared.name:
                    raced = True
                    os.unlink(source, dir_fd=kwargs["src_dir_fd"])
                    descriptor = os.open(
                        source,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=kwargs["src_dir_fd"],
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(external)
                    current = prepared.path.stat()
                    external_identity = (current.st_dev, current.st_ino)
                return real_rename(source, destination, *args, **kwargs)

            try:
                with mock.patch.object(
                    image_output.os,
                    "rename",
                    side_effect=race_rename,
                ):
                    removed = image_output._remove_published_target(
                        temporary, prepared.name
                    )
                self.assertTrue(raced)
                self.assertFalse(removed)
                self.assertEqual(prepared.path.read_bytes(), external)
                current = prepared.path.stat()
                self.assertEqual(
                    (current.st_dev, current.st_ino), external_identity
                )
            finally:
                if prepared.path.exists():
                    prepared.path.unlink()
                temp_path = Path(temp_dir) / temporary.name
                if temp_path.exists():
                    temp_path.unlink()
                os.close(temporary.descriptor)
                os.close(temporary.parent_fd)

    def test_private_unlink_failure_retains_owned_cleanup_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.preflight_output(Path(temp_dir) / "slide.jpg")
            temporary = image_output._temporary_path(prepared)
            real_unlink = image_output.os.unlink
            attempted = False

            def fail_private_unlink(name, *args, **kwargs):
                nonlocal attempted
                if name == "entry" and kwargs.get("dir_fd") is not None:
                    attempted = True
                    raise PermissionError("injected private cleanup failure")
                return real_unlink(name, *args, **kwargs)

            stderr = io.StringIO()
            try:
                with mock.patch.object(
                    image_output.os,
                    "unlink",
                    side_effect=fail_private_unlink,
                ), redirect_stderr(stderr), self.assertRaises(
                    image_output.ImageOutputError
                ):
                    image_output._remove_temp(temporary)
                retained = list(Path(temp_dir).glob(
                    ".image-output-recovery-*/entry"
                ))
                self.assertTrue(attempted)
                self.assertEqual(len(retained), 1)
                self.assertEqual(retained[0].stat().st_ino, temporary.inode)
                self.assertEqual(len(stderr.getvalue().splitlines()), 1)
            finally:
                os.close(temporary.descriptor)
                os.close(temporary.parent_fd)


class SafeDiagnosticTests(unittest.TestCase):
    ADVERSARIAL = "useful\r\n\x1b[31mRED\x1b[0m\x00\x85\ud800 tail"

    def test_safe_message_removes_line_terminal_and_unicode_hazards(self):
        value = safe_message(self.ADVERSARIAL)
        self.assertIn("useful", value)
        self.assertIn("tail", value)
        self.assertNotIn("\r", value)
        self.assertNotIn("\n", value)
        self.assertNotIn("\x1b", value)
        self.assertNotIn("\x00", value)
        self.assertNotIn("\x85", value)
        value.encode("utf-8")

    def test_generation_result_normalizes_safe_message_and_ascii_json(self):
        result = GenerationResult(
            GenerationStatus.LOCAL_FAILURE,
            "openai",
            "api",
            safe_message=self.ADVERSARIAL,
        )
        self.assertEqual(result.safe_message, safe_message(self.ADVERSARIAL))
        serialized = result.to_json()
        serialized.encode("ascii")
        self.assertEqual(json.loads(serialized)["safe_message"], result.safe_message)
        with self.assertRaises(ValueError):
            GenerationResult(
                GenerationStatus.LOCAL_FAILURE,
                "openai",
                "api",
                safe_message=object(),
            )

    def test_direct_provider_clis_emit_exactly_one_safe_line(self):
        for provider in (gen_slide_openai, gen_slide_gemini, gen_slide_doubao):
            with self.subTest(provider=provider.__name__):
                result = GenerationResult(
                    GenerationStatus.LOCAL_FAILURE,
                    provider.__name__.replace("gen_slide_", ""),
                    "api",
                    safe_message=self.ADVERSARIAL,
                )
                stdout = io.StringIO()
                with mock.patch.object(
                    provider, "generate_result", return_value=result
                ), redirect_stdout(stdout):
                    exit_code = provider.main(["slide.jpg", "prompt"])
                output = stdout.getvalue()
                self.assertEqual(exit_code, 1)
                self.assertEqual(len(output.splitlines()), 1)
                self.assertNotIn("\x1b", output)
                self.assertNotIn("\ud800", output)

    def test_unified_json_cli_is_one_ascii_serializable_object(self):
        result = GenerationResult(
            GenerationStatus.LOCAL_FAILURE,
            "openai",
            "api",
            safe_message=self.ADVERSARIAL,
        )
        stdout = io.StringIO()
        with mock.patch.object(
            gen_slide, "generate_result", return_value=result
        ), redirect_stdout(stdout):
            exit_code = gen_slide.main(["--json", "slide.jpg", "prompt"])
        self.assertEqual(exit_code, 1)
        output = stdout.getvalue()
        self.assertEqual(len(output.splitlines()), 1)
        output.encode("ascii")
        self.assertEqual(json.loads(output)["safe_message"], result.safe_message)

    def test_host_import_human_success_path_is_one_safe_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            source.write_bytes(image_bytes("PNG"))
            unsafe_output = "out/slide\n\x1b[31m.png"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = import_host_image.main([
                    str(source),
                    unsafe_output,
                    "--workspace-root",
                    str(root),
                    "--provider",
                    "openai",
                ])
            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(output.splitlines()), 1)
            self.assertNotIn("\x1b", output)


class DocumentationCorrectionTests(unittest.TestCase):
    def test_readme_scopes_publication_guarantees(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("API single-image", readme)
        self.assertIn("conditional", readme)
        self.assertIn("host raw/master", readme)
        self.assertIn("not a crash-atomic", readme)
        self.assertIn("export journal", readme)
        self.assertIn("secure publication primitives", readme)

    def test_skill_root_python_example_executes_from_repo_root(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("sys.path.insert(0, str(Path.cwd() / \"scripts\"))", skill)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "sys.path.insert(0, str(Path.cwd() / 'scripts')); "
                    "from gen_slide import gen; assert callable(gen)"
                ),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
