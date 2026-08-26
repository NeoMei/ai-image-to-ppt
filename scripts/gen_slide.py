#!/usr/bin/env python3
"""Unified slide image generator; OpenAI GPT Image is the default engine."""

import argparse
import importlib
from typing import Optional, Sequence

ENGINE_MODULES = {
    "openai": "gen_slide_openai",
    "gemini": "gen_slide_gemini",
    "doubao": "gen_slide_doubao",
}
DEFAULT_ENGINE = "openai"


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def gen(
    prompt: str,
    out_path: str,
    engine: str = DEFAULT_ENGINE,
    retries: int = 2,
) -> bool:
    if retries < 0:
        print("  ERR: retries must be non-negative")
        return False
    module_name = ENGINE_MODULES.get(engine)
    if module_name is None:
        choices = ", ".join(ENGINE_MODULES)
        print(f"  ERR: unknown engine '{engine}'. Choose: {choices}")
        return False
    provider = importlib.import_module(module_name)
    return bool(provider.gen(prompt, out_path, retries=retries))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 16:9 slide image (default: OpenAI GPT Image 2)."
    )
    parser.add_argument(
        "out_path",
        help=(
            "Output path. OpenAI matches .jpg/.jpeg/.png/.webp; legacy "
            "engines keep provider-returned encoding (conventionally use .jpg)"
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
        help="Non-negative retry count (default: 2)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    success = gen(
        args.prompt,
        args.out_path,
        engine=args.engine,
        retries=args.retries,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
