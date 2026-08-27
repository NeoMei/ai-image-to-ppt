import base64
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import image_output
import import_host_image
from generation_result import GenerationStatus
from output_lock import OutputLockBusy


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


def assert_no_temp_files(test_case, root, target_name="slide.jpg"):
    test_case.assertEqual(list(root.glob(f".{target_name}.*.tmp")), [])


class HostImageImportTests(unittest.TestCase):
    def test_local_file_is_validated_and_published_inside_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "host-source.jpg"
            source.write_bytes(image_bytes("JPEG", (160, 90)))

            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(source)),
                "out/slide.jpg",
                str(root),
                provider="openai",
            )

            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertEqual(
                Path(result.output_path), (root / "out/slide.jpg").resolve()
            )
            self.assertEqual((root / "out/slide.jpg").read_bytes(), source.read_bytes())

    def test_inline_bytes_is_validated_and_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    image_bytes(), "image/jpeg"
                ),
                "slide.jpg",
                root,
                provider="gemini",
            )
            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertTrue((root / "slide.jpg").is_file())

    def test_base64_is_decoded_and_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            encoded = base64.b64encode(image_bytes()).decode("ascii")
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.base64(encoded, "image/jpeg"),
                "slide.jpg",
                root,
                provider="doubao",
            )
            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertTrue((root / "slide.jpg").is_file())

    def test_data_url_is_decoded_and_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            encoded = base64.b64encode(image_bytes()).decode("ascii")
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.data_url(
                    f"data:image/jpeg;base64,{encoded}"
                ),
                "slide.jpg",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertTrue((root / "slide.jpg").is_file())

    def test_invalid_force_input_does_not_replace_existing_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            target.write_bytes(image_bytes("JPEG", (160, 90)))
            before = target.read_bytes()

            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    b"not-an-image", "image/jpeg"
                ),
                "slide.jpg",
                str(root),
                provider="openai",
                overwrite=True,
            )

            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertEqual(target.read_bytes(), before)
            assert_no_temp_files(self, root)

    def test_mime_mismatch_is_invalid_output_without_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(image_bytes(), "image/png"),
                "slide.jpg",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertFalse((root / "slide.jpg").exists())
            assert_no_temp_files(self, root)

    def test_oversized_encoded_data_is_invalid_output_without_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            image_output, "MAX_IMAGE_BYTES", 3
        ):
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.base64("QUFBQQ==", "image/jpeg"),
                "slide.jpg",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertFalse((root / "slide.jpg").exists())

    def test_ambiguous_or_unsupported_artifact_is_invalid_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = import_host_image.HostArtifact("unknown", b"data")
            result = import_host_image.import_host_artifact(
                artifact, "slide.jpg", temp_dir, provider="openai"
            )
            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertFalse((Path(temp_dir) / "slide.jpg").exists())

    def test_local_source_read_failure_is_local_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(root / "missing.jpg")),
                "slide.jpg",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse((root / "slide.jpg").exists())

    def test_workspace_escape_is_local_failure_before_source_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(root / "missing.jpg")),
                "../slide.jpg",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse((root.parent / "slide.jpg").exists())

    def test_unsupported_suffix_is_rejected_before_host_source_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            read_attempted = False

            def source_read_should_not_run(_artifact):
                nonlocal read_attempted
                read_attempted = True
                raise OSError("source read should not occur")

            with mock.patch.object(
                import_host_image,
                "_artifact_bytes",
                side_effect=source_read_should_not_run,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.local_path("missing.jpg"),
                    "slide.gif",
                    temp_dir,
                    provider="openai",
                )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse(read_attempted)

    def test_target_lock_contention_is_local_failure_before_source_read(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            import_host_image, "output_lock", side_effect=OutputLockBusy("busy")
        ):
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(root / "missing.jpg")),
                "slide.jpg",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse((root / "slide.jpg").exists())

    def test_parent_replacement_before_publication_is_local_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            source.write_bytes(image_bytes())
            target_parent = root / "out"
            original_validate = import_host_image.validate_image_bytes

            def validate_then_replace(data, target):
                loaded = original_validate(data, target)
                target_parent.rename(root / "out-original")
                target_parent.mkdir()
                return loaded

            with mock.patch.object(
                import_host_image,
                "validate_image_bytes",
                side_effect=validate_then_replace,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.local_path(str(source)),
                    "out/slide.jpg",
                    root,
                    provider="openai",
                )
            self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse((root / "out" / "slide.jpg").exists())
            self.assertFalse((root / "out-original" / "slide.jpg").exists())
            assert_no_temp_files(self, root / "out")

    def test_json_cli_emits_one_result_without_echoing_source_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source-sk-not-a-secret.jpg"
            source.write_bytes(image_bytes())
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = import_host_image.main([
                    str(source), "slide.jpg", "--workspace-root", str(root),
                    "--provider", "openai", "--json",
                ])
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(stdout.getvalue().splitlines()), 1)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "success")
            self.assertNotIn(source.name, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
