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
import prepare_editable_input
from generation_result import GenerationStatus
from output_lock import OutputLockBusy


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


def patterned_image_bytes(image_format="PNG", size=(1672, 941), mode="RGB"):
    """Create a near-16:9 image whose cropped center is observable."""
    image = Image.new(mode, size, (20, 30, 40, 128) if mode == "RGBA" else (20, 30, 40))

    def pixel(red, green, blue, alpha=255):
        return (red, green, blue, alpha) if mode == "RGBA" else (red, green, blue)

    for y, color in (
        (0, pixel(255, 0, 0, 64)),
        (2, pixel(0, 255, 0, 128)),
        (937, pixel(0, 0, 255, 192)),
        (940, pixel(255, 255, 0)),
    ):
        if y < size[1]:
            for x in range(size[0]):
                image.putpixel((x, y), color)
    for x, color in (
        (0, pixel(255, 0, 0, 64)),
        (4, pixel(0, 255, 255, 96)),
        (1667, pixel(255, 0, 255, 160)),
        (1671, pixel(255, 255, 0)),
    ):
        if x < size[0]:
            for y in range(size[1]):
                image.putpixel((x, y), color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def assert_no_temp_files(test_case, root, target_name="slide.jpg"):
    test_case.assertEqual(list(root.glob(f".{target_name}.*.tmp")), [])


class HostImageImportTests(unittest.TestCase):
    def test_near_ratio_png_is_central_cropped_after_raw_workspace_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "host-source.png"
            source_bytes = patterned_image_bytes(mode="RGBA")
            source.write_bytes(source_bytes)

            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(source)),
                "out/slide.png",
                root,
                provider="openai",
            )

            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertEqual(Path(result.output_path), master.resolve())
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(raw.read_bytes(), source_bytes)
            self.assertNotEqual(master.read_bytes(), source_bytes)
            self.assertTrue(master.is_absolute())
            with Image.open(master) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1664, 936))
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((800, 0)), (0, 255, 0, 128))
                self.assertEqual(image.getpixel((800, 935)), (0, 0, 255, 192))
                self.assertEqual(image.getpixel((0, 500)), (0, 255, 255, 96))
                self.assertEqual(image.getpixel((1663, 500)), (255, 0, 255, 160))
            image_output.validate_image_bytes(master.read_bytes(), master)

    def test_master_publication_failure_removes_new_raw_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = import_host_image.HostArtifact.inline_bytes(
                patterned_image_bytes(), "image/png"
            )
            real_publish = import_host_image._publish_transaction_member

            def fail_master(data, target, overwrite, strict):
                if strict:
                    raise image_output.ImageOutputError("forced master failure")
                return real_publish(data, target, overwrite, strict)

            with mock.patch.object(
                import_host_image,
                "_publish_transaction_member",
                side_effect=fail_master,
            ):
                failed = import_host_image.import_host_artifact(
                    artifact, "out/slide.png", root, provider="openai"
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertFalse((root / "out" / "slide.png").exists())
            self.assertFalse((root / "out" / "raw" / "slide.png").exists())
            self.assertEqual(list(root.rglob(".*.tmp")), [])
            self.assertEqual(list(root.rglob("*.backup")), [])

            retried = import_host_image.import_host_artifact(
                artifact, "out/slide.png", root, provider="openai"
            )
            self.assertEqual(retried.status, GenerationStatus.SUCCESS)

    def test_force_master_failure_restores_existing_raw_and_master(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            master.write_bytes(image_bytes("PNG", (160, 90)))
            raw.write_bytes(patterned_image_bytes())
            previous_master = master.read_bytes()
            previous_raw = raw.read_bytes()
            phases = []

            def fail_after_master_publish(phase, _snapshot):
                phases.append(phase)
                if phase == "master":
                    raise image_output.ImageOutputError("forced master failure")

            with mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=fail_after_master_publish,
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(mode="RGBA"), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), previous_master)
            self.assertEqual(raw.read_bytes(), previous_raw)
            self.assertEqual(phases, ["raw", "master"])
            self.assertEqual(list(root.rglob(".*.tmp")), [])
            self.assertEqual(list(root.rglob("*.backup")), [])

    def test_force_prewrite_master_failure_restores_invalid_existing_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            master.write_bytes(image_bytes("PNG", (160, 90)))
            previous_master = master.read_bytes()
            previous_raw = b"old raw is deliberately not an image"
            raw.write_bytes(previous_raw)
            real_publish = import_host_image._publish_transaction_member

            def fail_master(data, target, overwrite, strict):
                if strict:
                    raise image_output.ImageOutputError("forced pre-write failure")
                return real_publish(data, target, overwrite, strict)

            with mock.patch.object(
                import_host_image,
                "_publish_transaction_member",
                side_effect=fail_master,
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), previous_master)
            self.assertEqual(raw.read_bytes(), previous_raw)
            self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_force_prewrite_master_failure_restores_empty_existing_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            previous_master = image_bytes("PNG", (160, 90))
            master.write_bytes(previous_master)
            raw.write_bytes(b"")
            real_publish = import_host_image._publish_transaction_member

            def fail_master(data, target, overwrite, strict):
                if strict:
                    raise image_output.ImageOutputError("forced pre-write failure")
                return real_publish(data, target, overwrite, strict)

            with mock.patch.object(
                import_host_image,
                "_publish_transaction_member",
                side_effect=fail_master,
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), previous_master)
            self.assertEqual(raw.read_bytes(), b"")
            self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_force_postwrite_failure_restores_invalid_existing_master_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            previous_master = b"old master is deliberately not an image"
            previous_raw = patterned_image_bytes()
            master.write_bytes(previous_master)
            raw.write_bytes(previous_raw)

            def fail_after_master_publish(phase, _snapshot):
                if phase == "master":
                    raise image_output.ImageOutputError("forced post-write failure")

            with mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=fail_after_master_publish,
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), previous_master)
            self.assertEqual(raw.read_bytes(), previous_raw)
            self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_force_postwrite_failure_restores_empty_existing_master_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            previous_raw = patterned_image_bytes()
            master.write_bytes(b"")
            raw.write_bytes(previous_raw)

            def fail_after_master_publish(phase, _snapshot):
                if phase == "master":
                    raise image_output.ImageOutputError("forced post-write failure")

            with mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=fail_after_master_publish,
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), b"")
            self.assertEqual(raw.read_bytes(), previous_raw)
            self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_post_rename_stat_failure_restores_existing_raw_and_master(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            previous_master = image_bytes("PNG", (160, 90))
            previous_raw = patterned_image_bytes()
            master.write_bytes(previous_master)
            raw.write_bytes(previous_raw)
            real_rename = image_output.os.rename
            real_stat = image_output.os.stat
            rename_count = 0
            stat_failed = False

            def count_rename(*args, **kwargs):
                nonlocal rename_count
                rename_count += 1
                return real_rename(*args, **kwargs)

            def fail_second_target_stat(name, *args, **kwargs):
                nonlocal stat_failed
                if (
                    rename_count == 2
                    and not stat_failed
                    and name == "slide.png"
                    and kwargs.get("dir_fd") is not None
                ):
                    stat_failed = True
                    raise OSError("forced post-rename stat failure")
                return real_stat(name, *args, **kwargs)

            with mock.patch.object(
                image_output.os, "rename", side_effect=count_rename
            ), mock.patch.object(
                image_output.os, "stat", side_effect=fail_second_target_stat
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertTrue(stat_failed)
            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertEqual(master.read_bytes(), previous_master)
            self.assertEqual(raw.read_bytes(), previous_raw)
            self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_opaque_restore_rejects_malformed_identity_before_temp_allocation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.png"
            original = b"old arbitrary regular-file bytes"
            target.write_bytes(original)
            prepared = image_output.preflight_output(target, overwrite=True)

            for malformed in (None, (1,), (True, 1), (1, "inode")):
                with self.subTest(expected_existing_identity=malformed):
                    target.write_bytes(original)
                    with self.assertRaises(image_output.ImageOutputError):
                        image_output.publish_opaque_bytes_with_identity(
                            b"replacement",
                            prepared,
                            expected_existing_identity=malformed,
                        )
                    self.assertEqual(target.read_bytes(), original)
                    self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_same_byte_external_replacement_is_not_treated_as_transaction_owned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            master = root / "out" / "slide.png"
            master.parent.mkdir(parents=True)
            master.write_bytes(image_bytes("PNG", (160, 90)))
            replacement_identity = None
            replacement_bytes = None

            def externally_replace_master(phase, snapshot):
                nonlocal replacement_bytes, replacement_identity
                if phase != "master":
                    return
                published = snapshot.target.path.read_bytes()
                snapshot.target.path.unlink()
                snapshot.target.path.write_bytes(published)
                replacement_bytes = published
                replacement_identity = (
                    snapshot.target.path.stat().st_dev,
                    snapshot.target.path.stat().st_ino,
                )
                raise image_output.ImageOutputError("external replacement after publish")

            with mock.patch.object(
                import_host_image,
                "_transaction_checkpoint",
                side_effect=externally_replace_master,
            ):
                failed = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        patterned_image_bytes(), "image/png"
                    ),
                    "out/slide.png",
                    root,
                    provider="openai",
                    overwrite=True,
                )

            self.assertEqual(failed.status, GenerationStatus.LOCAL_FAILURE)
            self.assertIn("rollback was incomplete", failed.safe_message)
            self.assertIsNotNone(replacement_identity)
            self.assertEqual(master.read_bytes(), replacement_bytes)
            self.assertEqual(
                (master.stat().st_dev, master.stat().st_ino), replacement_identity
            )
            self.assertFalse((root / "out" / "raw" / "slide.png").exists())
            self.assertEqual(list(root.rglob(".*.tmp")), [])

    def test_near_ratio_is_normalized_before_strict_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_bytes = patterned_image_bytes()
            calls = []
            original_validate = import_host_image.validate_image_bytes

            def capture_strict_validation(data, target):
                calls.append(data)
                return original_validate(data, target)

            with mock.patch.object(
                import_host_image,
                "validate_image_bytes",
                side_effect=capture_strict_validation,
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(
                        source_bytes, "image/png"
                    ),
                    "slide.png",
                    root,
                    provider="openai",
                )

            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertEqual(len(calls), 1)
            self.assertNotEqual(calls[0], source_bytes)

    def test_exact_ratio_host_image_retains_existing_master_bytes_and_raw_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_bytes = image_bytes("JPEG", (160, 90))
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(source_bytes, "image/jpeg"),
                "out/slide.jpg",
                root,
                provider="openai",
            )

            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertEqual((root / "out" / "slide.jpg").read_bytes(), source_bytes)
            self.assertEqual((root / "out" / "raw" / "slide.jpg").read_bytes(), source_bytes)

    def test_host_ratio_threshold_is_inclusive_by_integer_cross_product(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # abs(3216 * 9 - 1800 * 16) / (1800 * 16) == 0.005 exactly.
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    image_bytes("JPEG", (3216, 1800)), "image/jpeg"
                ),
                "slide.jpg",
                root,
                provider="openai",
            )

            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            with Image.open(root / "slide.jpg") as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (3200, 1800))

    def test_host_ratio_just_over_threshold_is_invalid_and_force_preserves_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            target.write_bytes(image_bytes("JPEG", (160, 90)))
            before = target.read_bytes()

            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    image_bytes("JPEG", (3217, 1800)), "image/jpeg"
                ),
                "slide.jpg",
                root,
                provider="openai",
                overwrite=True,
            )

            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse((root / "raw" / "slide.jpg").exists())

    def test_host_image_too_small_to_contain_a_16_by_9_rectangle_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = image_output.prepare_workspace_target("slide.png", root)
            with self.assertRaisesRegex(image_output.ImageOutputError, "too small"):
                import_host_image._normalize_host_image(
                    image_bytes("PNG", (15, 9)), target
                )
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    image_bytes("PNG", (15, 9)), "image/png"
                ),
                "slide.png",
                root,
                provider="openai",
            )
            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertFalse((root / "slide.png").exists())
            self.assertFalse((root / "raw" / "slide.png").exists())

    def test_normalized_master_can_be_prepared_as_a_distinct_editable_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.inline_bytes(
                    patterned_image_bytes(), "image/png"
                ),
                "out/master.png",
                root,
                provider="openai",
            )
            master = Path(result.output_path)
            editable = root / "out" / "editable" / "slide.png"

            self.assertEqual(result.status, GenerationStatus.SUCCESS)
            self.assertTrue(prepare_editable_input.prepare(str(master), str(editable)))
            self.assertNotEqual(editable, master)
            with Image.open(editable) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1280, 720))
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

    def test_oversized_local_file_is_invalid_output_not_local_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            image_output, "MAX_IMAGE_BYTES", 3
        ):
            root = Path(temp_dir)
            source = root / "host-source.jpg"
            source.write_bytes(b"four")
            result = import_host_image.import_host_artifact(
                import_host_image.HostArtifact.local_path(str(source)),
                "slide.jpg",
                root,
                provider="openai",
            )

            self.assertEqual(result.status, GenerationStatus.INVALID_OUTPUT)
            self.assertFalse((root / "slide.jpg").exists())
            self.assertFalse((root / "raw" / "slide.jpg").exists())

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

    def test_source_path_parse_errors_are_local_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.subTest("embedded NUL"):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.local_path("bad\0source.jpg"),
                    "slide.jpg",
                    root,
                    provider="openai",
                )
                self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)

            with self.subTest("current directory capture"):
                with mock.patch.object(
                    import_host_image,
                    "resolve_input_path",
                    side_effect=image_output.ImageOutputError("cannot capture CWD"),
                ):
                    result = import_host_image.import_host_artifact(
                        import_host_image.HostArtifact.local_path("source.jpg"),
                        "other.jpg",
                        root,
                        provider="openai",
                    )
                self.assertEqual(result.status, GenerationStatus.LOCAL_FAILURE)

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
