import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest import mock

from PIL import Image
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_images


class ExportImagesTests(unittest.TestCase):
    def _image(self, path: Path, size=(1600, 900), color="navy") -> None:
        Image.new("RGB", size, color).save(path)

    def test_empty_input_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "deck.pdf"

            with self.assertRaisesRegex(ValueError, "at least one"):
                export_images.export_pdf([], str(target))

            self.assertFalse(target.exists())

    def test_corrupt_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "bad.png"
            target = Path(temp_dir) / "deck.pdf"
            source.write_bytes(b"not an image")
            target.write_bytes(b"existing")

            with self.assertRaises(OSError):
                export_images.export_pdf([str(source)], str(target))

            self.assertEqual(target.read_bytes(), b"existing")

    def test_standalone_failure_cleanup_warns_without_masking_primary_error(self):
        for phase in ("serializer", "sync", "publish"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                target = root / "deck.pdf"
                resolved_target = export_images.resolve_output_path(target)
                real_sync = export_images._sync_file
                real_replace = export_images.os.replace
                real_unlink = export_images.os.unlink
                cleanup_attempts = 0

                def save(_images, path):
                    if phase == "serializer":
                        raise OSError("serializer-root-cause")
                    Path(path).write_bytes(b"pdf")

                def sync(path):
                    if phase == "sync":
                        raise OSError("sync-root-cause")
                    return real_sync(path)

                def replace(source, destination, *args, **kwargs):
                    if phase == "publish" and Path(destination) == resolved_target:
                        raise OSError("publish-root-cause")
                    return real_replace(source, destination, *args, **kwargs)

                def deny_temp_cleanup(path, *args, **kwargs):
                    nonlocal cleanup_attempts
                    candidate = Path(path)
                    if (
                        candidate.parent.resolve(strict=False)
                        == root.resolve(strict=False)
                        and ".tmp.pdf" in candidate.name
                    ):
                        cleanup_attempts += 1
                        raise PermissionError("cleanup-denied")
                    return real_unlink(path, *args, **kwargs)

                stderr = io.StringIO()
                with mock.patch.object(
                    export_images, "_load_all", return_value=["image"]
                ), mock.patch.object(
                    export_images, "_save_pdf", side_effect=save
                ), mock.patch.object(
                    export_images, "_sync_file", side_effect=sync
                ), mock.patch.object(
                    export_images.os, "replace", side_effect=replace
                ), mock.patch.object(
                    export_images.os, "unlink", side_effect=deny_temp_cleanup
                ), redirect_stderr(stderr), self.assertRaisesRegex(
                    OSError, f"{phase}-root-cause"
                ):
                    export_images.export_pdf(["ignored"], str(target))

                stale = list(root.glob(".deck.pdf.*.tmp.pdf"))
                self.assertEqual(cleanup_attempts, 2)
                self.assertEqual(len(stale), 1)
                self.assertIn("WARN:", stderr.getvalue())
                self.assertIn(str(stale[0]), stderr.getvalue())

    def test_nested_output_parent_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            target = Path(temp_dir) / "nested" / "deck.pptx"
            self._image(source)

            export_images.export_pptx([str(source)], str(target))

            self.assertTrue(target.is_file())

    def test_relative_output_prefix_is_not_redirected_when_cwd_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            (redirected / "output").mkdir()
            source = root / "slide.png"
            self._image(source)
            real_load_all = export_images._load_all
            original_cwd = Path.cwd()

            def change_cwd_then_load(files):
                os.chdir(redirected)
                return real_load_all(files)

            try:
                os.chdir(approved)
                with mock.patch.object(
                    export_images,
                    "_load_all",
                    side_effect=change_cwd_then_load,
                ):
                    result = export_images.main(
                        ["output/deck", str(source)]
                    )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result, 0)
            self.assertTrue((approved / "output" / "deck.pdf").is_file())
            self.assertTrue((approved / "output" / "deck.pptx").is_file())
            self.assertFalse((redirected / "output" / "deck.pdf").exists())
            self.assertFalse((redirected / "output" / "deck.pptx").exists())

    def test_export_pdf_freezes_relative_target_before_input_loading_changes_cwd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            original_cwd = Path.cwd()

            def change_cwd_then_return(_files):
                os.chdir(redirected)
                return ["image"]

            def save_pdf(_images, path):
                Path(path).write_bytes(b"pdf")

            try:
                os.chdir(approved)
                with mock.patch.object(
                    export_images, "_load_all", side_effect=change_cwd_then_return
                ), mock.patch.object(export_images, "_save_pdf", side_effect=save_pdf):
                    export_images.export_pdf(["ignored"], "output/deck.pdf")
            finally:
                os.chdir(original_cwd)

            self.assertEqual((approved / "output" / "deck.pdf").read_bytes(), b"pdf")
            self.assertFalse((redirected / "output" / "deck.pdf").exists())

    def test_export_pptx_freezes_relative_target_before_saving_changes_cwd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            original_cwd = Path.cwd()

            def save_then_change_cwd(_images, path):
                Path(path).write_bytes(b"pptx")
                os.chdir(redirected)

            try:
                os.chdir(approved)
                with mock.patch.object(
                    export_images, "_load_all", return_value=["image"]
                ), mock.patch.object(
                    export_images, "_save_pptx", side_effect=save_then_change_cwd
                ):
                    export_images.export_pptx(["ignored"], "output/deck.pptx")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(
                (approved / "output" / "deck.pptx").read_bytes(), b"pptx"
            )
            self.assertFalse((redirected / "output" / "deck.pptx").exists())

    def test_standalone_exports_freeze_relative_input_symlink_before_callbacks(self):
        cases = (
            (export_images.export_pdf, "_save_pdf", ".pdf"),
            (export_images.export_pptx, "_save_pptx", ".pptx"),
        )
        for export, serializer_name, suffix in cases:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                approved = root / "approved"
                redirected = root / "redirected"
                approved.mkdir()
                redirected.mkdir()
                original_source = approved / "original.png"
                redirected_source = redirected / "redirected.png"
                self._image(original_source, color="red")
                self._image(redirected_source, color="blue")
                source_alias = approved / "source.png"
                source_alias.symlink_to(original_source)
                observed_files = []
                original_cwd = Path.cwd()

                def load_after_freeze(files):
                    observed_files.extend(files)
                    source_alias.unlink()
                    source_alias.symlink_to(redirected_source)
                    os.chdir(redirected)
                    return ["image"]

                def save(_images, path):
                    Path(path).write_bytes(b"artifact")

                try:
                    os.chdir(approved)
                    with mock.patch.object(
                        export_images, "_load_all", side_effect=load_after_freeze
                    ), mock.patch.object(
                        export_images, serializer_name, side_effect=save
                    ):
                        export(["source.png"], f"output/deck{suffix}")
                finally:
                    os.chdir(original_cwd)

                self.assertEqual(
                    observed_files,
                    [str(original_source.resolve(strict=True))],
                )
                self.assertEqual(
                    (approved / "output" / f"deck{suffix}").read_bytes(),
                    b"artifact",
                )

    def test_export_deck_freezes_all_relative_inputs_before_lock_cwd_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            approved.mkdir()
            redirected.mkdir()
            self._image(approved / "one.png", color="red")
            self._image(approved / "two.png", color="green")
            self._image(redirected / "one.png", color="blue")
            self._image(redirected / "two.png", color="yellow")
            original_cwd = Path.cwd()

            @contextmanager
            def change_cwd_before_yield(_target, namespace="output"):
                os.chdir(redirected)
                yield

            def save_markers(images, path):
                markers = b"".join(bytes(image.getpixel((0, 0))) for image in images)
                Path(path).write_bytes(markers)

            try:
                os.chdir(approved)
                with mock.patch.object(
                    export_images, "output_lock", change_cwd_before_yield
                ), mock.patch.object(
                    export_images, "_save_pdf", side_effect=save_markers
                ), mock.patch.object(
                    export_images, "_save_pptx", side_effect=save_markers
                ):
                    self.assertTrue(
                        export_images.export_deck(
                            ["one.png", "two.png"], "output/deck"
                        )
                    )
            finally:
                os.chdir(original_cwd)

            expected = bytes((255, 0, 0)) + bytes((0, 128, 0))
            self.assertEqual(
                (approved / "output" / "deck.pdf").read_bytes(), expected
            )
            self.assertEqual(
                (approved / "output" / "deck.pptx").read_bytes(), expected
            )

    def test_export_pdf_resolves_parent_but_does_not_follow_final_symlink(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            referent = root / "referent.pdf"
            target = root / "deck.pdf"
            referent.write_bytes(b"referent")
            target.symlink_to(referent)

            def save_pdf(_images, path):
                Path(path).write_bytes(b"new-pdf")

            with mock.patch.object(
                export_images, "_load_all", return_value=["image"]
            ), mock.patch.object(export_images, "_save_pdf", side_effect=save_pdf):
                export_images.export_pdf(["ignored"], str(target))

            self.assertEqual(referent.read_bytes(), b"referent")
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), b"new-pdf")

    def test_export_deck_freezes_ancestor_symlink_before_lock_yields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "approved"
            redirected = root / "redirected"
            (approved / "sub").mkdir(parents=True)
            (redirected / "sub").mkdir(parents=True)
            alias = root / "current"
            alias.symlink_to(approved, target_is_directory=True)
            real_output_lock = export_images.output_lock

            @contextmanager
            def swap_after_lock(target, namespace="output"):
                with real_output_lock(target, namespace=namespace):
                    alias.unlink()
                    alias.symlink_to(redirected, target_is_directory=True)
                    yield

            def save_pdf(_images, path):
                Path(path).write_bytes(b"pdf")

            def save_pptx(_images, path):
                Path(path).write_bytes(b"pptx")

            with mock.patch.object(export_images, "output_lock", swap_after_lock), \
                 mock.patch.object(export_images, "_load_all", return_value=["image"]), \
                 mock.patch.object(export_images, "_save_pdf", side_effect=save_pdf), \
                 mock.patch.object(export_images, "_save_pptx", side_effect=save_pptx):
                self.assertTrue(
                    export_images.export_deck(
                        ["ignored"], str(alias / "sub" / "deck")
                    )
                )

            self.assertEqual((approved / "sub" / "deck.pdf").read_bytes(), b"pdf")
            self.assertEqual(
                (approved / "sub" / "deck.pptx").read_bytes(), b"pptx"
            )
            self.assertFalse((redirected / "sub" / "deck.pdf").exists())
            self.assertFalse((redirected / "sub" / "deck.pptx").exists())

    def test_pptx_uses_exact_sixteen_by_nine_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            target = Path(temp_dir) / "deck.pptx"
            self._image(source)

            export_images.export_pptx([str(source)], str(target))
            presentation = Presentation(target)

            self.assertEqual(presentation.slide_width, 12_192_000)
            self.assertEqual(presentation.slide_height, 6_858_000)
            self.assertEqual(
                presentation.slide_width * 9,
                presentation.slide_height * 16,
            )

    def test_transparency_is_composited_onto_cream_not_black(self):
        image = Image.new("RGBA", (1600, 900), (255, 0, 0, 0))

        normalized = export_images.normalize(image)

        self.assertEqual(normalized.mode, "RGB")
        self.assertEqual(normalized.getpixel((100, 100)), (248, 245, 240))

    def test_cli_help_exits_zero(self):
        with self.assertRaises(SystemExit) as caught:
            export_images.main(["--help"])

        self.assertEqual(caught.exception.code, 0)

    def test_cli_missing_image_is_controlled_and_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir) / "nested" / "deck"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                result = export_images.main([str(prefix), "missing.png"])

            self.assertEqual(result, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())

    def test_invalid_output_prefix_is_controlled_before_loading_inputs(self):
        stderr = io.StringIO()
        with mock.patch.object(export_images, "_load_all") as load_all, \
             redirect_stderr(stderr):
            self.assertFalse(export_images.export_deck(["ignored"], "bad\0prefix"))

        load_all.assert_not_called()
        self.assertIn("invalid output target", stderr.getvalue())

    def test_cli_pair_publication_rolls_back_if_second_publish_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            self._image(source)
            real_link = export_images.os.link
            publish_count = 0

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                if str(target_path).endswith((".pdf", ".pptx")):
                    publish_count += 1
                    if publish_count == 2:
                        raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_publish,
            ):
                result = export_images.main([str(prefix), str(source)])

            self.assertEqual(result, 1)
            self.assertFalse(prefix.with_suffix(".pdf").exists())
            self.assertFalse(prefix.with_suffix(".pptx").exists())

    def test_cli_force_failure_restores_both_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            new_publish_count = 0

            def fail_second_new_publish(source_path, target_path, *args, **kwargs):
                nonlocal new_publish_count
                if str(target_path).endswith((".pdf", ".pptx")):
                    new_publish_count += 1
                    if new_publish_count == 2:
                        raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_new_publish,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

    def test_force_backup_identity_race_preserves_external_member_and_aborts(self):
        for raced_suffix in (".pdf", ".pptx"):
            with self.subTest(raced_suffix=raced_suffix), \
                 tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / "slide.png"
                prefix = Path(temp_dir) / "deck"
                pdf = prefix.with_suffix(".pdf")
                pptx = prefix.with_suffix(".pptx")
                self._image(source)
                pdf.write_bytes(b"old-pdf")
                pptx.write_bytes(b"old-pptx")
                raced_target = export_images.resolve_output_path(
                    prefix.with_suffix(raced_suffix)
                )
                external = f"external-{raced_suffix[1:]}".encode()
                real_replace = export_images.os.replace
                injected = False

                def replace_after_identity_check(
                    source_path, destination, *args, **kwargs
                ):
                    nonlocal injected
                    if Path(source_path) == raced_target and not injected:
                        injected = True
                        raced_target.unlink()
                        raced_target.write_bytes(external)
                    return real_replace(source_path, destination, *args, **kwargs)

                stderr = io.StringIO()
                with mock.patch.object(
                    export_images.os,
                    "replace",
                    side_effect=replace_after_identity_check,
                ), redirect_stderr(stderr):
                    result = export_images.main(
                        ["--force", str(prefix), str(source)]
                    )

                self.assertTrue(injected)
                self.assertEqual(result, 1)
                self.assertEqual(raced_target.read_bytes(), external)
                untouched = (
                    export_images.resolve_output_path(pptx)
                    if raced_suffix == ".pdf"
                    else export_images.resolve_output_path(pdf)
                )
                untouched_expected = (
                    b"old-pptx" if raced_suffix == ".pdf" else b"old-pdf"
                )
                self.assertEqual(untouched.read_bytes(), untouched_expected)
                preserved = [
                    path
                    for path in Path(temp_dir).glob(".*.backup")
                    if path.read_bytes() == external
                ]
                self.assertEqual(len(preserved), 1)
                self.assertIn("identity changed", stderr.getvalue())
                self.assertIn(str(preserved[0]), stderr.getvalue())

    def test_force_rollback_preserves_external_replacement_of_published_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            publish_count = 0

            def replace_pdf_then_fail_pptx(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                target = Path(target_path)
                if publish_count == 1:
                    result = real_link(source_path, target_path, *args, **kwargs)
                    target.unlink()
                    target.write_bytes(b"external-pdf")
                    return result
                raise OSError("forced second publish failure")

            stderr = io.StringIO()
            with mock.patch.object(
                export_images.os, "link", side_effect=replace_pdf_then_fail_pptx
            ), redirect_stderr(stderr):
                result = export_images.main(["--force", str(prefix), str(source)])

            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), b"external-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            backups = list(Path(temp_dir).glob(".*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"old-pdf")
            self.assertIn("ownership changed", stderr.getvalue())
            self.assertIn(str(backups[0]), stderr.getvalue())

    def test_force_rollback_preserves_external_creation_at_unpublished_pptx(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            publish_count = 0

            def create_external_pptx(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                if publish_count == 2:
                    Path(target_path).write_bytes(b"external-pptx")
                return real_link(source_path, target_path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                export_images.os, "link", side_effect=create_external_pptx
            ), redirect_stderr(stderr):
                result = export_images.main(["--force", str(prefix), str(source)])

            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"external-pptx")
            backups = list(Path(temp_dir).glob(".*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"old-pptx")
            self.assertIn("ownership changed", stderr.getvalue())
            self.assertIn(str(backups[0]), stderr.getvalue())

    def test_successful_deck_warns_and_returns_true_when_temp_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            self._image(source)
            real_unlink = export_images.os.unlink

            def deny_transaction_temp(path, *args, **kwargs):
                candidate = Path(path)
                if (
                    candidate.parent.resolve(strict=False)
                    == Path(temp_dir).resolve(strict=False)
                    and ".tmp." in candidate.name
                ):
                    raise PermissionError("forced temp cleanup denial")
                return real_unlink(path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                export_images.os, "unlink", side_effect=deny_transaction_temp
            ), redirect_stderr(stderr):
                self.assertTrue(
                    export_images.export_deck([str(source)], str(prefix))
                )

            self.assertTrue(prefix.with_suffix(".pdf").is_file())
            self.assertTrue(prefix.with_suffix(".pptx").is_file())
            stale = list(Path(temp_dir).glob(".*.tmp.*"))
            self.assertEqual(len(stale), 2)
            self.assertIn("WARN:", stderr.getvalue())
            for path in stale:
                self.assertIn(str(path), stderr.getvalue())

    def test_cli_default_pdf_race_preserves_external_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            resolved_pdf = export_images.resolve_output_path(pdf)
            sentinel = b"external-pdf-writer"
            self._image(source)
            real_link = export_images.os.link
            injected = False

            def inject_pdf_before_publish(source_path, target_path, *args, **kwargs):
                nonlocal injected
                if Path(target_path) == resolved_pdf and not injected:
                    pdf.write_bytes(sentinel)
                    injected = True
                return real_link(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=inject_pdf_before_publish,
            ):
                result = export_images.main([str(prefix), str(source)])

            self.assertTrue(injected)
            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), sentinel)
            self.assertFalse(pptx.exists())

    def test_cli_default_pptx_race_preserves_external_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            resolved_pdf = export_images.resolve_output_path(pdf)
            resolved_pptx = export_images.resolve_output_path(pptx)
            pdf_sentinel = b"external-pdf-replacement"
            pptx_sentinel = b"external-pptx-writer"
            self._image(source)
            real_link = export_images.os.link
            pdf_replaced = False
            pptx_injected = False

            def race_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal pdf_replaced, pptx_injected
                target = Path(target_path)
                if target == resolved_pdf:
                    result = real_link(source_path, target_path, *args, **kwargs)
                    pdf.unlink()
                    pdf.write_bytes(pdf_sentinel)
                    pdf_replaced = True
                    return result
                if target == resolved_pptx and not pptx_injected:
                    pptx.write_bytes(pptx_sentinel)
                    pptx_injected = True
                return real_link(source_path, target_path, *args, **kwargs)

            stderr = io.StringIO()
            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=race_second_publish,
            ), redirect_stderr(stderr):
                result = export_images.main([str(prefix), str(source)])

            self.assertTrue(pdf_replaced)
            self.assertTrue(pptx_injected)
            self.assertEqual(result, 1)
            self.assertEqual(pdf.read_bytes(), pdf_sentinel)
            self.assertEqual(pptx.read_bytes(), pptx_sentinel)
            self.assertIn("ownership changed", stderr.getvalue())

    def test_force_rollback_unlink_failure_still_restores_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            resolved_pdf = export_images.resolve_output_path(pdf)
            self._image(source)
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            real_replace = export_images.os.replace
            real_unlink = export_images.os.unlink
            publish_count = 0
            restore_attempts = []

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                if publish_count == 2:
                    raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            def fail_published_cleanup(path, *args, **kwargs):
                if Path(path) == resolved_pdf:
                    raise OSError("forced published unlink failure")
                return real_unlink(path, *args, **kwargs)

            def record_restore(source_path, target_path, *args, **kwargs):
                if str(source_path).endswith(".backup"):
                    restore_attempts.append(Path(target_path))
                return real_replace(source_path, target_path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_publish,
            ), mock.patch.object(
                export_images.os,
                "unlink",
                side_effect=fail_published_cleanup,
            ), mock.patch.object(
                export_images.os,
                "replace",
                side_effect=record_restore,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(
                restore_attempts,
                [export_images.resolve_output_path(pptx)],
            )
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertTrue(pdf.is_file())
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

    def test_force_rollback_restores_backed_targets_without_unlinking_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            pdf = prefix.with_suffix(".pdf")
            pptx = prefix.with_suffix(".pptx")
            self._image(source)
            pdf.write_bytes(b"old-pdf")
            pptx.write_bytes(b"old-pptx")
            real_link = export_images.os.link
            real_unlink = export_images.os.unlink
            publish_count = 0
            target_unlinks = []

            def fail_second_publish(source_path, target_path, *args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                if publish_count == 2:
                    raise OSError("forced second publish failure")
                return real_link(source_path, target_path, *args, **kwargs)

            def reject_target_unlink(path, *args, **kwargs):
                target = Path(path)
                if target in (pdf, pptx):
                    target_unlinks.append(target)
                    raise OSError("backed target must be restored directly")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(
                export_images.os,
                "link",
                side_effect=fail_second_publish,
            ), mock.patch.object(
                export_images.os,
                "unlink",
                side_effect=reject_target_unlink,
            ):
                result = export_images.main(
                    ["--force", str(prefix), str(source)]
                )

            self.assertEqual(result, 1)
            self.assertEqual(target_unlinks, [])
            self.assertEqual(pdf.read_bytes(), b"old-pdf")
            self.assertEqual(pptx.read_bytes(), b"old-pptx")
            self.assertEqual(list(Path(temp_dir).glob(".*.backup")), [])

    def test_force_rollback_attempts_every_restore_and_preserves_failed_backup(self):
        for failed_name in ("deck.pdf", "deck.pptx"):
            with self.subTest(failed_name=failed_name), tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / "slide.png"
                prefix = Path(temp_dir) / "deck"
                pdf = prefix.with_suffix(".pdf")
                pptx = prefix.with_suffix(".pptx")
                self._image(source)
                pdf.write_bytes(b"old-pdf")
                pptx.write_bytes(b"old-pptx")
                real_link = export_images.os.link
                real_replace = export_images.os.replace
                publish_count = 0
                restore_attempts = []

                def fail_second_publish(source_path, target_path, *args, **kwargs):
                    nonlocal publish_count
                    publish_count += 1
                    if publish_count == 2:
                        raise OSError("forced second publish failure")
                    return real_link(source_path, target_path, *args, **kwargs)

                def fail_selected_restore(source_path, target_path, *args, **kwargs):
                    if str(source_path).endswith(".backup"):
                        target = Path(target_path)
                        restore_attempts.append(target.name)
                        if target.name == failed_name:
                            raise OSError(f"forced restore failure for {failed_name}")
                    return real_replace(source_path, target_path, *args, **kwargs)

                stderr = io.StringIO()
                with mock.patch.object(
                    export_images.os,
                    "link",
                    side_effect=fail_second_publish,
                ), mock.patch.object(
                    export_images.os,
                    "replace",
                    side_effect=fail_selected_restore,
                ), redirect_stderr(stderr):
                    result = export_images.main(
                        ["--force", str(prefix), str(source)]
                    )

                self.assertEqual(result, 1)
                self.assertEqual(restore_attempts, ["deck.pdf", "deck.pptx"])
                failed_backups = [
                    path
                    for path in Path(temp_dir).glob(".*.backup")
                    if f".{failed_name}." in path.name
                ]
                self.assertEqual(len(failed_backups), 1)
                expected = b"old-pdf" if failed_name.endswith(".pdf") else b"old-pptx"
                self.assertEqual(failed_backups[0].read_bytes(), expected)
                restored = pptx if failed_name.endswith(".pdf") else pdf
                restored_expected = b"old-pptx" if restored == pptx else b"old-pdf"
                self.assertEqual(restored.read_bytes(), restored_expected)
                self.assertIn(str(failed_backups[0]), stderr.getvalue())

    def test_second_temporary_file_setup_failure_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "slide.png"
            prefix = Path(temp_dir) / "deck"
            self._image(source)
            real_mkstemp = export_images.tempfile.mkstemp
            calls = 0

            def fail_second_mkstemp(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("forced temp setup failure")
                return real_mkstemp(*args, **kwargs)

            with mock.patch.object(
                export_images.tempfile,
                "mkstemp",
                side_effect=fail_second_mkstemp,
            ):
                result = export_images.main([str(prefix), str(source)])

            self.assertEqual(result, 1)
            self.assertEqual(list(Path(temp_dir).glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
