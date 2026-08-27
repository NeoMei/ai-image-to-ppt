import base64
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images
import gen_slide_openai
import image_output
import prepare_editable_input


def image_bytes(image_format="JPEG", size=(160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "navy").save(buffer, format=image_format)
    return buffer.getvalue()


class JsonResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")
        self.stream = io.BytesIO(self.payload)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def swap_parent(parent, redirected):
    original = parent.with_name(f"{parent.name}-original")
    parent.rename(original)
    parent.symlink_to(redirected, target_is_directory=True)
    return original


def assert_no_transaction_files(test_case, directory):
    test_case.assertEqual(list(directory.glob(".*.tmp*")), [])
    test_case.assertEqual(list(directory.glob(".*.backup")), [])


class SharedParentIdentityTests(unittest.TestCase):
    def test_temp_fstat_failure_does_not_unlink_replacement_after_name_stat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = image_output.preflight_output(root / "slide.jpg")
            candidate = root / ".slide.jpg.fstat-race.tmp"
            real_open = image_output.os.open
            real_fstat = image_output.os.fstat
            real_stat = image_output.os.stat
            real_close = image_output.os.close
            real_unlink = image_output.os.unlink
            opened = []
            closed = []
            fstat_calls = 0
            replaced = False

            def record_open(*args, **kwargs):
                descriptor = real_open(*args, **kwargs)
                opened.append(descriptor)
                return descriptor

            def fail_first_temp_fstat(descriptor):
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 2:
                    raise OSError("injected temporary fstat failure")
                return real_fstat(descriptor)

            def replace_after_name_stat(path, *args, **kwargs):
                nonlocal replaced
                result = real_stat(path, *args, **kwargs)
                if (
                    not replaced
                    and path == candidate.name
                    and kwargs.get("dir_fd") is not None
                ):
                    replaced = True
                    real_unlink(path, dir_fd=kwargs["dir_fd"])
                    candidate.write_bytes(b"external replacement")
                return result

            def record_close(descriptor):
                closed.append(descriptor)
                return real_close(descriptor)

            with mock.patch.object(
                image_output.secrets,
                "token_hex",
                return_value="fstat-race",
            ), mock.patch.object(
                image_output.os,
                "open",
                side_effect=record_open,
            ), mock.patch.object(
                image_output.os,
                "fstat",
                side_effect=fail_first_temp_fstat,
            ), mock.patch.object(
                image_output.os,
                "stat",
                side_effect=replace_after_name_stat,
            ), mock.patch.object(
                image_output.os,
                "close",
                side_effect=record_close,
            ), self.assertRaisesRegex(
                image_output.ImageOutputError,
                "failed to create output temporary file",
            ):
                image_output._temporary_path(prepared)

            self.assertTrue(replaced)
            self.assertGreaterEqual(len(opened), 2)
            self.assertTrue(set(opened).issubset(set(closed)))
            self.assertEqual(candidate.read_bytes(), b"external replacement")

    def test_temp_fstat_failure_retains_replacement_when_restore_name_is_taken(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = image_output.preflight_output(root / "slide.jpg")
            candidate = root / ".slide.jpg.fstat-restore-race.tmp"
            real_fstat = image_output.os.fstat
            real_stat = image_output.os.stat
            real_unlink = image_output.os.unlink
            real_link = image_output.os.link
            fstat_calls = 0
            replaced = False

            def fail_first_temp_fstat(descriptor):
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 2:
                    raise OSError("injected temporary fstat failure")
                return real_fstat(descriptor)

            def replace_after_name_stat(path, *args, **kwargs):
                nonlocal replaced
                result = real_stat(path, *args, **kwargs)
                if (
                    not replaced
                    and path == candidate.name
                    and kwargs.get("dir_fd") is not None
                ):
                    replaced = True
                    real_unlink(path, dir_fd=kwargs["dir_fd"])
                    candidate.write_bytes(b"first external replacement")
                return result

            def occupy_name_before_restore(*args, **kwargs):
                candidate.write_bytes(b"second external replacement")
                return real_link(*args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                image_output.secrets,
                "token_hex",
                return_value="fstat-restore-race",
            ), mock.patch.object(
                image_output.os,
                "fstat",
                side_effect=fail_first_temp_fstat,
            ), mock.patch.object(
                image_output.os,
                "stat",
                side_effect=replace_after_name_stat,
            ), mock.patch.object(
                image_output.os,
                "link",
                side_effect=occupy_name_before_restore,
            ), redirect_stderr(stderr), self.assertRaisesRegex(
                image_output.ImageOutputError,
                "failed to create output temporary file",
            ):
                image_output._temporary_path(prepared)

            quarantines = list(root.glob(".image-output-cleanup-*"))
            self.assertTrue(replaced)
            self.assertEqual(
                candidate.read_bytes(),
                b"second external replacement",
            )
            self.assertEqual(len(quarantines), 1)
            self.assertEqual(
                (quarantines[0] / "entry").read_bytes(),
                b"first external replacement",
            )
            self.assertIn("could not be restored safely", stderr.getvalue())

    def test_temp_fstat_failure_cleans_owned_name_and_closes_fd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = image_output.preflight_output(root / "slide.jpg")
            candidate = root / ".slide.jpg.fstat-failure.tmp"
            real_open = image_output.os.open
            real_fstat = image_output.os.fstat
            real_close = image_output.os.close
            opened = []
            closed = []
            fstat_calls = 0

            def record_open(*args, **kwargs):
                descriptor = real_open(*args, **kwargs)
                opened.append(descriptor)
                return descriptor

            def fail_first_temp_fstat(descriptor):
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 2:
                    raise OSError("injected temporary fstat failure")
                return real_fstat(descriptor)

            def record_close(descriptor):
                closed.append(descriptor)
                return real_close(descriptor)

            with mock.patch.object(
                image_output.secrets,
                "token_hex",
                return_value="fstat-failure",
            ), mock.patch.object(
                image_output.os,
                "open",
                side_effect=record_open,
            ), mock.patch.object(
                image_output.os,
                "fstat",
                side_effect=fail_first_temp_fstat,
            ), mock.patch.object(
                image_output.os,
                "close",
                side_effect=record_close,
            ), self.assertRaisesRegex(
                image_output.ImageOutputError,
                "failed to create output temporary file",
            ):
                image_output._temporary_path(prepared)

            self.assertGreaterEqual(len(opened), 2)
            self.assertTrue(set(opened).issubset(set(closed)))
            self.assertFalse(candidate.exists())

    def test_temp_name_collisions_do_not_delete_external_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            prepared = image_output.preflight_output(target)
            candidate = root / ".slide.jpg.collision.tmp"
            candidate.write_bytes(b"external collision")

            with mock.patch.object(
                image_output.secrets,
                "token_hex",
                return_value="collision",
            ), self.assertRaisesRegex(
                image_output.ImageOutputError,
                "allocate unique",
            ):
                image_output._temporary_path(prepared)

            self.assertEqual(candidate.read_bytes(), b"external collision")

    def test_temp_replacement_before_return_preserves_external_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            prepared = image_output.preflight_output(target)
            candidate = root / ".slide.jpg.owned.tmp"
            real_verify = image_output.verify_parent_identity
            verify_count = 0

            def replace_then_fail(parent):
                nonlocal verify_count
                real_verify(parent)
                verify_count += 1
                if verify_count == 2:
                    candidate.unlink()
                    candidate.write_bytes(b"external replacement")
                    raise image_output.ImageOutputError("forced return failure")

            with mock.patch.object(
                image_output.secrets,
                "token_hex",
                return_value="owned",
            ), mock.patch.object(
                image_output,
                "verify_parent_identity",
                side_effect=replace_then_fail,
            ), self.assertRaisesRegex(
                image_output.ImageOutputError,
                "forced return failure",
            ):
                image_output._temporary_path(prepared)

            self.assertEqual(candidate.read_bytes(), b"external replacement")
            self.assertEqual(list(root.glob(".slide.jpg.*.tmp")), [candidate])

    def test_temp_path_replacement_after_validation_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "slide.jpg"
            prepared = image_output.preflight_output(target)
            real_validate = image_output._validate_temp
            replacement = None

            def validate_then_replace(temp_path, prepared_target):
                nonlocal replacement
                result = real_validate(temp_path, prepared_target)
                replacement = Path(temp_path)
                replacement.unlink()
                replacement.write_bytes(b"not-an-image")
                return result

            with mock.patch.object(
                image_output,
                "_validate_temp",
                side_effect=validate_then_replace,
            ), self.assertRaisesRegex(image_output.ImageOutputError, "temporary"):
                image_output.publish_bytes(image_bytes(), prepared)

            self.assertFalse(target.exists())
            self.assertIsNotNone(replacement)
            self.assertEqual(replacement.read_bytes(), b"not-an-image")

    def test_parent_replacement_after_publish_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "approved"
            parent.mkdir()
            prepared = image_output.preflight_output(parent / "slide.jpg")
            original = parent.with_name("approved-original")
            rename_called = False
            real_rename = image_output.os.rename

            def swap_then_rename(*args, **kwargs):
                nonlocal rename_called
                rename_called = True
                real_rename(str(parent), str(original))
                parent.mkdir()
                return real_rename(*args, **kwargs)

            with mock.patch.object(
                image_output.os,
                "rename",
                side_effect=swap_then_rename,
            ), self.assertRaisesRegex(image_output.ImageOutputError, "parent directory"):
                image_output.publish_bytes(image_bytes(), prepared, overwrite=True)

            self.assertTrue(rename_called)
            self.assertFalse((parent / "slide.jpg").exists())
            self.assertFalse((original / "slide.jpg").exists())
            assert_no_transaction_files(self, parent)
            assert_no_transaction_files(self, original)

    def test_failed_cleanup_preserves_replacement_directory_external_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "approved"
            parent.mkdir()
            prepared = image_output.preflight_output(parent / "slide.jpg")
            original = parent.with_name("approved-original")
            external_temp = None

            def replace_parent_then_fail(temp_path, _target, _overwrite):
                nonlocal external_temp
                temporary = Path(temp_path)
                parent.rename(original)
                parent.mkdir()
                external_temp = parent / temporary.name
                external_temp.write_bytes(b"external")
                raise image_output.ImageOutputError("forced publication failure")

            with mock.patch.object(
                image_output,
                "_publish_temp",
                side_effect=replace_parent_then_fail,
            ), self.assertRaisesRegex(image_output.ImageOutputError, "forced publication"):
                image_output.publish_bytes(image_bytes(), prepared)

            self.assertIsNotNone(external_temp)
            self.assertEqual(external_temp.read_bytes(), b"external")
            self.assertEqual(list(original.glob(".slide.jpg.*.tmp")), [])

    def test_workspace_target_rejects_escape_before_file_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "workspace"
            root.mkdir()
            with self.assertRaisesRegex(image_output.ImageOutputError, "workspace"):
                image_output.prepare_workspace_target("../escape.jpg", root)
            self.assertFalse((Path(temp_dir) / "escape.jpg").exists())

    def test_workspace_target_rejects_ancestor_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "workspace"
            root.mkdir()
            outside = temp / "outside"
            outside.mkdir()
            (root / "nested").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(image_output.ImageOutputError, "workspace"):
                image_output.prepare_workspace_target("nested/slide.jpg", root)
            self.assertFalse((outside / "slide.jpg").exists())

    def test_workspace_target_rejects_final_symlink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "workspace"
            root.mkdir()
            target = root / "slide.jpg"
            target.symlink_to(root / "elsewhere.jpg")

            with self.assertRaisesRegex(
                image_output.ImageOutputError, "not a regular file"
            ):
                image_output.preflight_output(
                    image_output.prepare_workspace_target("slide.jpg", root),
                    overwrite=True,
                )

    def test_workspace_target_uses_captured_root_after_cwd_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "workspace"
            root.mkdir()
            original_cwd = Path.cwd()
            try:
                prepared = image_output.prepare_workspace_target("out/slide.jpg", root)
                os.chdir(tempfile.gettempdir())
                image_output.preflight_output(prepared)
            finally:
                os.chdir(original_cwd)
            self.assertEqual(prepared.path, (root / "out/slide.jpg").resolve())

    def test_workspace_target_parent_replacement_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "workspace"
            root.mkdir()
            prepared = image_output.prepare_workspace_target("out/slide.jpg", root)
            original = root / "out-original"
            (root / "out").rename(original)
            (root / "out").mkdir()

            with self.assertRaisesRegex(
                image_output.ImageOutputError, "parent directory identity changed"
            ):
                image_output.publish_bytes(image_bytes(), prepared)
            self.assertFalse((root / "out" / "slide.jpg").exists())
            self.assertFalse((original / "slide.jpg").exists())

    def test_preflight_rejects_a_symlink_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_parent = root / "real"
            real_parent.mkdir()
            alias_parent = root / "alias"
            alias_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "parent directory identity",
            ):
                image_output.preflight_output(str(alias_parent / "slide.jpg"))

            self.assertEqual(list(real_parent.iterdir()), [])

    def test_replaced_parent_inode_is_rejected_before_temp_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            approved.mkdir()
            target = approved / "slide.jpg"
            prepared = image_output.preflight_output(str(target))
            original = approved.with_name("approved-original")
            approved.rename(original)
            approved.mkdir()

            with self.assertRaisesRegex(
                image_output.ImageOutputError,
                "parent directory identity changed",
            ):
                image_output.publish_bytes(image_bytes(), prepared)

            self.assertFalse((approved / "slide.jpg").exists())
            self.assertFalse((original / "slide.jpg").exists())
            assert_no_transaction_files(self, approved)
            assert_no_transaction_files(self, original)

    def test_provider_parent_swap_during_network_fails_closed(self):
        generated = image_bytes()
        response = JsonResponse({
            "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}]
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            target = approved / "slide.jpg"
            original = None

            def swap_then_respond(_request, timeout):
                nonlocal original
                self.assertEqual(timeout, 180)
                original = swap_parent(approved, redirected)
                return response

            with mock.patch.object(
                gen_slide_openai,
                "_load_api_key",
                return_value="key",
            ), mock.patch.object(
                gen_slide_openai.urllib.request,
                "urlopen",
                side_effect=swap_then_respond,
            ), redirect_stdout(io.StringIO()):
                self.assertFalse(
                    gen_slide_openai.gen("prompt", str(target), retries=0)
                )

            self.assertIsNotNone(original)
            self.assertFalse((redirected / "slide.jpg").exists())
            self.assertFalse((original / "slide.jpg").exists())
            assert_no_transaction_files(self, redirected)
            assert_no_transaction_files(self, original)


class LocalParentIdentityTests(unittest.TestCase):
    def test_forced_deck_parent_swap_preserves_original_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "slide.png"
            Image.new("RGB", (1600, 900), "navy").save(source)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            prefix = approved / "deck"
            prefix.with_suffix(".pdf").write_bytes(b"old-pdf")
            prefix.with_suffix(".pptx").write_bytes(b"old-pptx")
            real_load_all = export_images._load_all
            original = None

            def swap_then_load(files, **kwargs):
                nonlocal original
                original = swap_parent(approved, redirected)
                return real_load_all(files, **kwargs)

            with mock.patch.object(
                export_images,
                "_load_all",
                side_effect=swap_then_load,
            ), redirect_stderr(io.StringIO()):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual((original / "deck.pdf").read_bytes(), b"old-pdf")
            self.assertEqual((original / "deck.pptx").read_bytes(), b"old-pptx")
            self.assertFalse((redirected / "deck.pdf").exists())
            self.assertFalse((redirected / "deck.pptx").exists())
            assert_no_transaction_files(self, redirected)
            assert_no_transaction_files(self, original)

    def test_prepare_parent_swap_during_source_load_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jpg"
            Image.new("RGB", (1600, 900), "navy").save(source)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            target = approved / "slide.png"
            real_load_image = prepare_editable_input.load_image
            original = None

            def swap_then_load(path):
                nonlocal original
                original = swap_parent(approved, redirected)
                return real_load_image(path)

            with mock.patch.object(
                prepare_editable_input,
                "load_image",
                side_effect=swap_then_load,
            ), redirect_stdout(io.StringIO()):
                self.assertFalse(
                    prepare_editable_input.prepare(str(source), str(target))
                )

            self.assertFalse((redirected / "slide.png").exists())
            self.assertFalse((original / "slide.png").exists())
            assert_no_transaction_files(self, redirected)
            assert_no_transaction_files(self, original)


if __name__ == "__main__":
    unittest.main()
