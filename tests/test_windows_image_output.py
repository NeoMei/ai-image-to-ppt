import io
import base64
import json
import os
import sys
import tempfile
import unittest
import contextlib
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_output
import import_host_image
import windows_image_output
import gen_slide_openai
import gen_slide_gemini
import gen_slide_doubao


class _FakeHandle:
    def __init__(self, path):
        self.path = Path(path)


class _FilesystemHandleApi:
    def __init__(self):
        self.before_install = None

    @staticmethod
    @contextlib.contextmanager
    def pin_ancestors(_target):
        yield

    @staticmethod
    def close(_handle):
        return None

    @staticmethod
    def open_directory(path, delete=False):
        del delete
        if not Path(path).is_dir() or Path(path).is_symlink():
            raise image_output.ImageOutputError("unsafe directory")
        return _FakeHandle(path)

    @staticmethod
    def open_existing(path, read=False):
        del read
        path = Path(path)
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise image_output.ImageOutputError("unsafe file")
        return _FakeHandle(path)

    @staticmethod
    def create_new(path):
        path = Path(path)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        return _FakeHandle(path)

    @staticmethod
    def identity(handle):
        current = handle.path.stat()
        return current.st_dev, current.st_ino

    @staticmethod
    def write(handle, data):
        handle.path.write_bytes(data)

    @staticmethod
    def read(handle):
        return handle.path.read_bytes()

    def rename(self, handle, destination):
        destination = Path(destination)
        if self.before_install is not None and destination.name == "slide.png":
            callback, self.before_install = self.before_install, None
            callback(destination)
        if destination.exists():
            raise FileExistsError(destination)
        handle.path.rename(destination)
        handle.path = destination

    @staticmethod
    def delete(handle):
        if handle.path.is_dir():
            handle.path.rmdir()
        else:
            handle.path.unlink()


class WindowsDispatchTests(unittest.TestCase):
    def test_preflight_selects_windows_backend_without_posix_primitives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.prepare_target(Path(temp_dir) / "slide.png")
            expected = object()
            backend = mock.Mock()
            backend.preflight_output.return_value = expected

            with mock.patch.object(image_output, "_SECURE_PUBLICATION_SUPPORTED", False), mock.patch.object(
                image_output, "_WINDOWS_PUBLICATION_SUPPORTED", True
            , create=True), mock.patch.object(
                image_output, "_windows_backend", return_value=backend, create=True
            ):
                try:
                    actual = image_output.preflight_output(prepared, overwrite=True)
                except image_output.ImageOutputError as error:
                    self.fail(f"Windows backend was not selected: {error}")

        self.assertIs(actual, expected)
        backend.preflight_output.assert_called_once_with(prepared, True)

    def test_byte_publication_dispatches_validation_and_identity_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.prepare_target(Path(temp_dir) / "slide.png")
            prepared = image_output.replace(
                prepared, expected_identity=None, expectation_captured=True
            )
            backend = mock.Mock()
            backend.publish_bytes.return_value = image_output.PublishedOutput(7, 4, 9)

            with mock.patch.object(image_output, "_WINDOWS_PUBLICATION_SUPPORTED", True), mock.patch.object(
                image_output, "_windows_backend", return_value=backend
            ):
                result = image_output._publish_image_bytes(
                    b"payload", prepared, False, image_output._validate_opaque_bytes,
                    expected_missing=True, allow_empty=False,
                )

        self.assertEqual(result, image_output.PublishedOutput(7, 4, 9))
        backend.publish_bytes.assert_called_once_with(
            b"payload", prepared, False, image_output._validate_opaque_bytes
        )

    def test_stream_uses_windows_publication_after_bounded_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.prepare_target(Path(temp_dir) / "slide.png")
            prepared = image_output.replace(
                prepared, expected_identity=None, expectation_captured=True
            )
            backend = mock.Mock()
            backend.publish_bytes.return_value = image_output.PublishedOutput(7, 4, 9)

            with mock.patch.object(image_output, "_WINDOWS_PUBLICATION_SUPPORTED", True), mock.patch.object(
                image_output, "_windows_backend", return_value=backend
            ):
                count = image_output.publish_stream(io.BytesIO(b"payload"), prepared)

        self.assertEqual(count, 7)
        backend.publish_bytes.assert_called_once_with(
            b"payload", prepared, False, image_output.validate_image_bytes
        )

    def test_host_snapshot_and_identity_dispatch_to_windows_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = image_output.prepare_target(Path(temp_dir) / "slide.png")
            backend = mock.Mock()
            backend.capture_identity.return_value = (11, 12)
            backend.snapshot_output.return_value = (b"old", (11, 12))

            with mock.patch.object(image_output, "_WINDOWS_PUBLICATION_SUPPORTED", True), mock.patch.object(
                image_output, "_windows_backend", return_value=backend
            ):
                identity = image_output.capture_output_identity(prepared)
                snapshot = image_output.snapshot_output_bytes(prepared)

        self.assertEqual(identity, (11, 12))
        self.assertEqual(snapshot, (b"old", (11, 12)))


class WindowsTransactionStateTests(unittest.TestCase):
    def publish(self, data, prepared, overwrite, api):
        with mock.patch.object(windows_image_output, "_validate_path", side_effect=lambda path: path), mock.patch.object(
            windows_image_output, "_api", return_value=api
        ):
            return windows_image_output.publish_bytes(
                data, prepared, overwrite, image_output._validate_opaque_bytes
            )

    def test_overwrite_removes_recovery_after_displaced_handle_closes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"old")
            prepared = image_output.preflight_output(target, overwrite=True)

            result = self.publish(b"new", prepared, True, _FilesystemHandleApi())

            self.assertEqual(result.byte_count, 3)
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(list(target.parent.glob(".image-output-recovery-*")), [])

    def test_recovery_name_collision_preserves_unowned_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"old")
            prepared = image_output.preflight_output(target, overwrite=True)
            collision_token = "a" * 32
            unique_token = "b" * 32
            collision = target.parent / (
                f".image-output-recovery-{collision_token}.entry"
            )
            collision.write_bytes(b"external")

            with mock.patch.object(
                windows_image_output.secrets,
                "token_hex",
                side_effect=["c" * 32, collision_token, unique_token],
            ):
                result = self.publish(
                    b"new", prepared, True, _FilesystemHandleApi()
                )

            self.assertEqual(result.byte_count, 3)
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(collision.read_bytes(), b"external")
            self.assertFalse(
                (target.parent / f".image-output-recovery-{unique_token}.entry").exists()
            )

    def test_recovery_never_creates_or_opens_a_recovery_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"old")
            prepared = image_output.preflight_output(target, overwrite=True)
            api = _FilesystemHandleApi()

            with mock.patch.object(
                windows_image_output.os,
                "mkdir",
                side_effect=AssertionError("recovery directory created"),
            ):
                self.publish(b"new", prepared, True, api)

            self.assertEqual(target.read_bytes(), b"new")

    def test_refusing_overwrite_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"external")
            prepared = image_output.replace(
                image_output.prepare_target(target),
                expected_identity=None,
                expectation_captured=True,
            )

            with self.assertRaises(image_output.ImageOutputError):
                self.publish(b"new", prepared, False, _FilesystemHandleApi())

            self.assertEqual(target.read_bytes(), b"external")

    def test_preflight_refusal_never_deletes_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"external")
            prepared = image_output.prepare_target(target)
            with mock.patch.object(
                windows_image_output, "_validate_path", side_effect=lambda path: path
            ), mock.patch.object(
                windows_image_output, "_api", return_value=_FilesystemHandleApi()
            ):
                with self.assertRaises(image_output.ImageOutputError):
                    windows_image_output.preflight_output(prepared, False)

            self.assertEqual(target.read_bytes(), b"external")

    def test_concurrent_target_creation_preserves_both_external_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"old")
            prepared = image_output.preflight_output(target, overwrite=True)
            api = _FilesystemHandleApi()
            api.before_install = lambda destination: destination.write_bytes(b"external")

            with self.assertRaisesRegex(
                image_output.ImageOutputError, r"retained at .*entry"
            ):
                self.publish(b"new", prepared, True, api)

            self.assertEqual(target.read_bytes(), b"external")
            retained = list(target.parent.glob(".image-output-recovery-*.entry"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"old")

    def test_post_install_failure_removes_owned_output_and_restores_old(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(b"old")
            prepared = image_output.preflight_output(target, overwrite=True)

            with mock.patch.object(
                windows_image_output,
                "_publication_checkpoint",
                side_effect=lambda phase: (
                    (_ for _ in ()).throw(OSError("injected post-install failure"))
                    if phase == "installed"
                    else None
                ),
                create=True,
            ):
                with self.assertRaises(image_output.RemovedPublishedOutputError):
                    self.publish(b"new", prepared, True, _FilesystemHandleApi())

            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(target.parent.glob(".image-output-recovery-*")), [])

    def test_ambiguous_windows_names_fail_before_native_access(self):
        unsafe = (
            Path("C:/output/NUL.png"),
            Path("C:/output/COM¹.jpg"),
            Path("C:/output/slide.png:secret"),
            Path("C:/output/trailing./slide.png"),
        )
        for path in unsafe:
            with self.subTest(path=path), self.assertRaises(
                image_output.ImageOutputError
            ):
                windows_image_output._validate_path(path)


@unittest.skipUnless(os.name == "nt", "requires real Win32 filesystem APIs")
class WindowsNativePublicationTests(unittest.TestCase):
    @staticmethod
    def png_bytes(color="navy"):
        from PIL import Image

        output = io.BytesIO()
        Image.new("RGB", (160, 90), color).save(output, format="PNG")
        return output.getvalue()

    def test_create_refuse_overwrite_and_explicit_overwrite_unicode(self):
        data = self.png_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "幻灯片-α.png"
            first = image_output.publish_bytes_with_identity(data, target)
            self.assertEqual(target.read_bytes(), data)
            with self.assertRaises(image_output.ImageOutputError):
                image_output.publish_bytes(data, target)
            second = image_output.publish_bytes_with_identity(data, target, overwrite=True)
            self.assertNotEqual((first.device, first.inode), (second.device, second.inode))

    def test_identity_removal_refuses_mismatch_then_removes_owned_file(self):
        data = self.png_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            published = image_output.publish_bytes_with_identity(data, target)
            with self.assertRaises(image_output.ImageOutputError):
                image_output.remove_output_with_identity(target, (0, 0))
            self.assertTrue(target.exists())
            image_output.remove_output_with_identity(
                target, (published.device, published.inode)
            )
            self.assertFalse(target.exists())

    def test_competing_reader_without_delete_share_blocks_then_releases_overwrite(self):
        old = self.png_bytes("navy")
        replacement = self.png_bytes("gold")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(old)
            prepared = image_output.preflight_output(target, overwrite=True)
            native = windows_image_output._api()
            competing = native._open(
                target,
                windows_image_output._GENERIC_READ,
                windows_image_output._FILE_SHARE_READ
                | windows_image_output._FILE_SHARE_WRITE,
                windows_image_output._OPEN_EXISTING,
                windows_image_output._FILE_ATTRIBUTE_NORMAL,
            )
            try:
                with self.assertRaises(image_output.ImageOutputError):
                    image_output.publish_bytes(replacement, prepared, overwrite=True)
                self.assertEqual(target.read_bytes(), old)
            finally:
                native.close(competing)

            published = image_output.publish_bytes_with_identity(
                replacement, prepared, overwrite=True
            )
            self.assertEqual(published.byte_count, len(replacement))
            self.assertEqual(target.read_bytes(), replacement)

    def test_reparse_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actual = root / "actual"
            actual.mkdir()
            link = root / "linked"
            try:
                link.symlink_to(actual, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"cannot create Windows symlink: {error}")
            with self.assertRaises(image_output.ImageOutputError):
                image_output.publish_bytes(self.png_bytes(), link / "slide.png")

    def test_host_pair_rolls_back_after_master_checkpoint_failure(self):
        data = self.png_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "out" / "slide.png"
            raw = root / "out" / "raw" / "slide.png"
            raw.parent.mkdir(parents=True)
            old_master = data
            old_raw = b"old raw"
            target.write_bytes(old_master)
            raw.write_bytes(old_raw)

            def fail_master(phase, _snapshot):
                if phase == "master":
                    raise image_output.ImageOutputError("injected failure")

            with mock.patch.object(
                import_host_image, "_transaction_checkpoint", side_effect=fail_master
            ):
                result = import_host_image.import_host_artifact(
                    import_host_image.HostArtifact.inline_bytes(data, "image/png"),
                    "out/slide.png", root, "openai", overwrite=True,
                )

            self.assertFalse(result.ok)
            self.assertEqual(target.read_bytes(), old_master)
            self.assertEqual(raw.read_bytes(), old_raw)

    def test_controlled_write_failure_preserves_existing_output(self):
        data = self.png_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            target.write_bytes(data)
            prepared = image_output.preflight_output(target, overwrite=True)
            native = windows_image_output._api()

            with mock.patch.object(
                native, "write", side_effect=OSError("injected write failure")
            ):
                with self.assertRaises(image_output.ImageOutputError):
                    image_output.publish_bytes(data, prepared, overwrite=True)

            self.assertEqual(target.read_bytes(), data)

    def test_concurrent_target_creation_preserves_external_and_old_recovery(self):
        data = self.png_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "slide.png"
            old = data + b"old"
            target.write_bytes(old)
            prepared = image_output.preflight_output(target, overwrite=True)

            def create_external(phase):
                if phase == "before_install":
                    target.write_bytes(b"external")

            with mock.patch.object(
                windows_image_output, "_publication_checkpoint", side_effect=create_external
            ):
                with self.assertRaises(image_output.ImageOutputError):
                    image_output.publish_bytes(data, prepared, overwrite=True)

            self.assertEqual(target.read_bytes(), b"external")
            retained = list(target.parent.glob(".image-output-recovery-*.entry"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), old)

    def test_all_provider_fixture_responses_publish_through_native_backend(self):
        data = self.png_bytes()

        class Response:
            def __init__(self, payload, headers=None):
                self.stream = io.BytesIO(payload)
                self.headers = headers or {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, size=-1):
                return self.stream.read(size)

        fixtures = (
            (
                gen_slide_openai,
                [Response(json.dumps({"data": [{"b64_json": base64.b64encode(data).decode("ascii")}]}).encode())],
            ),
            (
                gen_slide_gemini,
                [Response(json.dumps({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(data).decode("ascii")}}]}}]}).encode())],
            ),
            (
                gen_slide_doubao,
                [
                    Response(json.dumps({"data": [{"url": "https://fixture.invalid/image"}]}).encode()),
                    Response(data, {"Content-Type": "image/png", "Content-Length": str(len(data))}),
                ],
            ),
        )
        for provider, responses in fixtures:
            with self.subTest(provider=provider.__name__), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "slide.png"
                with mock.patch.object(provider, "_load_api_key", return_value="fixture-key"), mock.patch.object(
                    provider.urllib.request, "urlopen", side_effect=responses
                ):
                    result = provider.generate_result("fixture", str(target), retries=0)
                self.assertTrue(result.ok, result.to_json())
                self.assertEqual(target.read_bytes(), data)


class NativeHandleCleanupTests(unittest.TestCase):
    def api_with_failed_attributes(self):
        api = object.__new__(windows_image_output._NativeApi)
        handle = object()
        api._open = mock.Mock(return_value=handle)
        api.attributes = mock.Mock(side_effect=OSError("metadata failed"))
        api.close = mock.Mock()
        return api, handle

    def test_open_directory_closes_handle_when_metadata_query_fails(self):
        api, handle = self.api_with_failed_attributes()

        with self.assertRaises(OSError):
            api.open_directory(Path("directory"))

        api.close.assert_called_once_with(handle)

    def test_open_existing_closes_handle_when_metadata_query_fails(self):
        api, handle = self.api_with_failed_attributes()

        with self.assertRaises(OSError):
            api.open_existing(Path("file"))

        api.close.assert_called_once_with(handle)

    def test_open_directory_closes_invalid_object_once(self):
        api, handle = self.api_with_failed_attributes()
        api.attributes.side_effect = None
        api.attributes.return_value = windows_image_output._FILE_ATTRIBUTE_REPARSE_POINT

        with self.assertRaises(image_output.ImageOutputError):
            api.open_directory(Path("directory"))

        api.close.assert_called_once_with(handle)

    def test_open_existing_closes_invalid_object_once(self):
        api, handle = self.api_with_failed_attributes()
        api.attributes.side_effect = None
        api.attributes.return_value = windows_image_output._FILE_ATTRIBUTE_DIRECTORY

        with self.assertRaises(image_output.ImageOutputError):
            api.open_existing(Path("file"))

        api.close.assert_called_once_with(handle)


if __name__ == "__main__":
    unittest.main()
