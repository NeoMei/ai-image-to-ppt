#!/usr/bin/env python3
"""Unified slide image generator; OpenAI GPT Image is the default engine."""

import argparse
import importlib
from typing import Optional, Sequence

from image_output import MAX_RETRIES, ImageOutputError, validate_retries

ENGINE_MODULES = {
    "openai": "gen_slide_openai",
    "gemini": "gen_slide_gemini",
    "doubao": "gen_slide_doubao",
}
DEFAULT_ENGINE = "openai"


def _non_negative_int(value: str) -> int:
    try:
        return validate_retries(int(value))
    except (ValueError, ImageOutputError) as error:
        raise argparse.ArgumentTypeError(
            f"must be non-negative and at most {MAX_RETRIES}"
        ) from error


def gen(
    prompt: str,
    out_path: str,
    engine: str = DEFAULT_ENGINE,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    try:
        retries = validate_retries(retries)
    except ImageOutputError as error:
        print(f"  ERR: {error}")
        return False
    if not isinstance(overwrite, bool):
        print("  ERR: overwrite must be a boolean")
        return False
    module_name = ENGINE_MODULES.get(engine)
    if module_name is None:
        choices = ", ".join(ENGINE_MODULES)
        print(f"  ERR: unknown engine '{engine}'. Choose: {choices}")
        return False
    try:
        provider = importlib.import_module(module_name)
    except ImportError:
        print(f"  ERR: engine '{engine}' is unavailable ({module_name})")
        return False
    try:
        return bool(
            provider.gen(
                prompt,
                out_path,
                retries=retries,
                overwrite=overwrite,
            )
        )
    except Exception as error:
        print(f"  ERR: engine '{engine}' failed ({type(error).__name__})")
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 16:9 slide image (default: OpenAI GPT Image 2)."
    )
    parser.add_argument(
        "out_path",
        help=(
            "Output path. OpenAI/Doubao: .jpg/.jpeg/.png/.webp; "
            "Gemini: .png (default), .jpg/.jpeg (JPEG suffix requests JPEG). "
            "All providers validate actual encoding and strict 16:9 dimensions"
        ),
    )
    parser.add_argument("prompt", help="Image generation prompt")
    parser.add_argument(
        "--engine",
        choices=tuple(ENGINE_MODULES),
        default=DEFAULT_ENGINE,
        help="Image engine (default: openai)",
    )
    parser.add_argument(
        "--retries",
        type=_non_negative_int,
        default=2,
        help=f"Retry count, 0..{MAX_RETRIES} (default: 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output only after a valid image is ready",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    success = gen(
        args.prompt,
        args.out_path,
        engine=args.engine,
        retries=args.retries,
        overwrite=args.force,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
