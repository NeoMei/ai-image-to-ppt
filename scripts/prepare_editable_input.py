#!/usr/bin/env python3
"""Prepare one exact 1280x720 PNG for image-to-editable-pptx."""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, UnidentifiedImageError

TARGET_SIZE = (1280, 720)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


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
            prepared = image.convert("RGB").resize(
                TARGET_SIZE,
                Image.Resampling.LANCZOS,
            )
    except (OSError, UnidentifiedImageError) as error:
        print(f"  ERR: cannot read source image: {error}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as temp_file:
            prepared.save(temp_file, format="PNG", optimize=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.link(temp_path, target)
        os.unlink(temp_path)
    except FileExistsError:
        print(f"  ERR: output already exists; refusing to overwrite: {target}")
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        return False
    except OSError as error:
        print(f"  ERR: failed to write normalized PNG: {error}")
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        return False

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
