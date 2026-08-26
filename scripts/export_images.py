#!/usr/bin/env python3
"""Normalize slide images and export transactional PDF/PPTX artifacts.

Install runtime dependencies with ``python3 -m pip install -r requirements.txt``.
"""

import argparse
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PIL import Image
from pptx import Presentation
from pptx.util import Emu
from image_output import (
    ImageOutputError,
    ParentIdentity,
    load_image,
    prepare_target,
    verify_parent_identity,
)
from output_lock import OutputLockError, output_lock

TARGET_SIZE = (1920, 1080)
CREAM_BACKGROUND = (248, 245, 240)
SLIDE_H = Emu(6_858_000)  # exact 7.5 inches
SLIDE_W = Emu(int(SLIDE_H) * 16 // 9)


def _flatten_transparency(image: Image.Image) -> Image.Image:
    """Return RGB pixels, compositing transparency on the deck's cream canvas."""
    has_transparency = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_transparency:
        return image.convert("RGB") if image.mode != "RGB" else image.copy()
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, CREAM_BACKGROUND + (255,))
    return Image.alpha_composite(background, rgba).convert("RGB")


def normalize(image: Image.Image) -> Image.Image:
    """Cover-crop an image to an exact 1920x1080 RGB slide."""
    image = _flatten_transparency(image)
    if image.size == TARGET_SIZE:
        return image
    target_width, target_height = TARGET_SIZE
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if width * target_height > height * target_width:
        new_width = height * target_width // target_height
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    elif width * target_height < height * target_width:
        new_height = width * target_height // target_width
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    return image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def _load(path: str) -> Image.Image:
    try:
        return normalize(load_image(Path(path)).image)
    except ImageOutputError as error:
        raise OSError(str(error)) from error


def _load_all(files: Iterable[str]) -> List[Image.Image]:
    paths = list(files)
    if not paths:
        raise ValueError("at least one input image is required")
    return [_load(path) for path in paths]


def _save_pdf(images: Sequence[Image.Image], path: str) -> None:
    images[0].save(
        path,
        save_all=True,
        append_images=list(images[1:]),
        resolution=150.0,
        quality=85,
        optimize=True,
    )


def _save_pptx(images: Sequence[Image.Image], path: str) -> None:
    presentation = Presentation()
    presentation.slide_width = SLIDE_W
    presentation.slide_height = SLIDE_H
    blank = presentation.slide_layouts[6]
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        slide = presentation.slides.add_slide(blank)
        slide.shapes.add_picture(
            buffer,
            0,
            0,
            width=SLIDE_W,
            height=SLIDE_H,
        )
    presentation.save(path)


def _temporary_path(
    target: Path,
    parent: Optional[ParentIdentity] = None,
) -> str:
    prepared = prepare_target(target) if parent is None else None
    identity = prepared.parent if prepared is not None else parent
    path = None
    try:
        verify_parent_identity(identity)
        descriptor, path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=f".tmp{target.suffix}",
            dir=str(identity.path),
        )
        os.close(descriptor)
        verify_parent_identity(identity)
        return path
    except Exception:
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


def _sync_file(path: str) -> None:
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def _atomic_save(target_path: str, save: Callable[[str], None]) -> None:
    target = Path(target_path)
    temporary = _temporary_path(target)
    try:
        save(temporary)
        _sync_file(temporary)
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def export_pdf(files: Sequence[str], pdf_path: str) -> None:
    """Export a normalized multi-page PDF without exposing partial output."""
    images = _load_all(files)
    _atomic_save(pdf_path, lambda path: _save_pdf(images, path))
    size_mb = os.path.getsize(pdf_path) / 1024 / 1024
    print(f"  PDF: {pdf_path} ({len(images)} pages, {size_mb:.1f}MB)")


def export_pptx(files: Sequence[str], pptx_path: str) -> None:
    """Export an exact 16:9 PPTX without exposing partial output."""
    images = _load_all(files)
    _atomic_save(pptx_path, lambda path: _save_pptx(images, path))
    size_mb = os.path.getsize(pptx_path) / 1024 / 1024
    print(f"  PPTX: {pptx_path} ({len(images)} slides, {size_mb:.1f}MB)")


def _backup_path(
    target: Path,
    parent: Optional[ParentIdentity] = None,
) -> str:
    identity = prepare_target(target).parent if parent is None else parent
    descriptor = None
    path = None
    try:
        verify_parent_identity(identity)
        descriptor, path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".backup",
            dir=str(identity.path),
        )
        os.close(descriptor)
        descriptor = None
        verify_parent_identity(identity)
        os.unlink(path)
        return path
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


def _publish_pair(
    temporary_paths: Sequence[str],
    targets: Sequence[Path],
    force: bool,
    parent: Optional[ParentIdentity] = None,
) -> None:
    if parent is not None:
        verify_parent_identity(parent)
    existing = [target for target in targets if os.path.lexists(target)]
    if existing and not force:
        names = ", ".join(str(target) for target in existing)
        raise FileExistsError(f"output already exists; use --force: {names}")

    backups = {}
    published = []
    publication_succeeded = False
    try:
        for target in existing:
            if parent is not None:
                verify_parent_identity(parent)
            if target.is_symlink() or not target.is_file():
                raise OSError(f"refusing non-regular output target: {target}")
            backup = _backup_path(target, parent=parent)
            os.replace(target, backup)
            backups[target] = backup
        for temporary, target in zip(temporary_paths, targets):
            if parent is not None:
                verify_parent_identity(parent)
            temporary_stat = os.stat(temporary, follow_symlinks=False)
            os.link(temporary, target)
            published.append(
                (target, temporary_stat.st_dev, temporary_stat.st_ino)
            )
        publication_succeeded = True
    except Exception as publication_error:
        cleanup_errors = []
        for target, temporary_device, temporary_inode in reversed(published):
            if target in backups:
                continue
            try:
                target_stat = os.stat(target, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as error:
                cleanup_errors.append((target, error))
                continue
            if (
                temporary_device != target_stat.st_dev
                or temporary_inode != target_stat.st_ino
            ):
                cleanup_errors.append(
                    (
                        target,
                        OSError(
                            "published output ownership changed; "
                            "external target preserved"
                        ),
                    )
                )
                continue
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
            except OSError as error:
                cleanup_errors.append((target, error))

        restore_errors = []
        for target, backup in backups.items():
            try:
                os.replace(backup, target)
            except OSError as error:
                restore_errors.append((backup, target, error))

        if cleanup_errors or restore_errors:
            details = []
            if cleanup_errors:
                failures = ", ".join(
                    f"{path} ({error})" for path, error in cleanup_errors
                )
                details.append(f"could not remove published outputs: {failures}")
            if restore_errors:
                preserved = ", ".join(item[0] for item in restore_errors)
                details.append(f"backups preserved at: {preserved}")
            raise OSError(
                "publication failed and rollback was incomplete; "
                + "; ".join(details)
            ) from publication_error
        raise
    finally:
        if publication_succeeded:
            for backup in backups.values():
                try:
                    os.unlink(backup)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    print(
                        f"  WARN: stale export backup remains at {backup}: {error}",
                        file=sys.stderr,
                    )


def _export_deck_owned(
    files: Sequence[str], output_prefix: str, force: bool = False
) -> None:
    """Build PDF and PPTX completely, then publish them as one logical pair."""
    output_prefix = os.path.abspath(output_prefix)
    targets = (Path(f"{output_prefix}.pdf"), Path(f"{output_prefix}.pptx"))
    parent = prepare_target(targets[0]).parent
    images = _load_all(files)
    temporary_paths = []
    try:
        for target in targets:
            temporary_paths.append(_temporary_path(target, parent=parent))
        _save_pdf(images, temporary_paths[0])
        _sync_file(temporary_paths[0])
        _save_pptx(images, temporary_paths[1])
        _sync_file(temporary_paths[1])
        _publish_pair(temporary_paths, targets, force=force, parent=parent)
    finally:
        for temporary in temporary_paths:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    for label, target in zip(("PDF", "PPTX"), targets):
        verify_parent_identity(parent)
        size_mb = target.stat().st_size / 1024 / 1024
        print(f"  {label}: {target} ({len(images)} slides, {size_mb:.1f}MB)")


def export_deck(
    files: Sequence[str], output_prefix: str, force: bool = False
) -> bool:
    """Build and publish one pair while owning the complete output prefix."""
    try:
        with output_lock(output_prefix, namespace="deck"):
            _export_deck_owned(files, output_prefix, force=force)
        return True
    except OutputLockError as error:
        print(f"  ERR: export failed: {error}", file=sys.stderr)
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize images and export a PDF/PPTX pair."
    )
    parser.add_argument("output_prefix", help="Output path without .pdf/.pptx")
    parser.add_argument("images", nargs="+", help="Ordered slide image paths")
    parser.add_argument(
        "--force",
        action="store_true",
        help="transactionally replace an existing PDF/PPTX pair",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not export_deck(args.images, args.output_prefix, force=args.force):
            return 1
    except (OSError, ValueError) as error:
        print(f"  ERR: export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
