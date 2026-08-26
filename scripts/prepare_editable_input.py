#!/usr/bin/env python3
"""Prepare one exact 1280x720 PNG for image-to-editable-pptx."""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, UnidentifiedImageError

TARGET_SIZE = (1280, 720)
CREAM_BACKGROUND = (248, 245, 240)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _remove_temp(temp_path: str) -> Optional[OSError]:
    last_error = None
    for _attempt in range(2):
        try:
            os.unlink(temp_path)
            return None
        except FileNotFoundError:
            return None
        except OSError as error:
            last_error = error
    return last_error


def _flatten_transparency(image: Image.Image) -> Image.Image:
    has_transparency = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_transparency:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, CREAM_BACKGROUND + (255,))
    return Image.alpha_composite(background, rgba).convert("RGB")


def prepare(input_path: str, output_path: str) -> bool:
    source = Path(input_path)
    target = Path(output_path)

    if target.suffix.lower() != ".png":
        print("  ERR: editable converter input must use a .png output path")
        return False
    if _same_path(source, target):
        print("  ERR: source and output paths must be different")
        return False
    if os.path.lexists(target):
        print(f"  ERR: output already exists; refusing to overwrite: {target}")
        return False

    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            if width * 9 != height * 16:
                print(
                    f"  ERR: source must be exactly 16:9; received {width}x{height}"
                )
                return False
            prepared = _flatten_transparency(image).resize(
                TARGET_SIZE,
                Image.Resampling.LANCZOS,
            )
    except (OSError, UnidentifiedImageError) as error:
        print(f"  ERR: cannot read source image: {error}")
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
    except OSError as error:
        print(f"  ERR: failed to prepare output path: {error}")
        return False

    try:
        with os.fdopen(fd, "wb") as temp_file:
            prepared.save(temp_file, format="PNG", optimize=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())
    except OSError as error:
        print(f"  ERR: failed to write normalized PNG: {error}")
        cleanup_error = _remove_temp(temp_path)
        if cleanup_error is not None:
            print(f"  WARN: could not remove temporary file: {cleanup_error}")
        return False

    try:
        os.link(temp_path, target)
    except FileExistsError:
        print(f"  ERR: output already exists; refusing to overwrite: {target}")
        cleanup_error = _remove_temp(temp_path)
        if cleanup_error is not None:
            print(f"  WARN: could not remove temporary file: {cleanup_error}")
        return False
    except OSError as error:
        print(f"  ERR: failed to publish normalized PNG: {error}")
        cleanup_error = _remove_temp(temp_path)
        if cleanup_error is not None:
            print(f"  WARN: could not remove temporary file: {cleanup_error}")
        return False

    cleanup_error = _remove_temp(temp_path)
    if cleanup_error is not None:
        print(
            "  WARN: output was published but temporary file cleanup failed: "
            f"{cleanup_error}"
        )
    print(f"  OK: {target} (1280x720 PNG, editable-converter input)")
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an exact 1280x720 PNG for image-to-editable-pptx."
    )
    parser.add_argument("input_path", help="Strict 16:9 source image")
    parser.add_argument("output_path", help="New .png output path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return 0 if prepare(args.input_path, args.output_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
