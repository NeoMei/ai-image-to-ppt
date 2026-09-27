#!/usr/bin/env python3
"""API/CLI adapter for slide generation; does not use host capabilities."""

import argparse
import importlib
import re
from contextlib import redirect_stdout
from typing import Callable, Optional, Sequence

from generation_result import GenerationResult, GenerationStatus, safe_message
from image_output import MAX_RETRIES, ImageOutputError, validate_retries

ENGINE_MODULES = {
    "openai": "gen_slide_openai",
    "gemini": "gen_slide_gemini",
    "doubao": "gen_slide_doubao",
}
DEFAULT_ENGINE = "openai"
_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\][^\x1b\x07]*(?:\x07|\x1b\\)|[@-_][0-?]*[ -/]*[@-~]|[0-?]*[ -/]*[@-~])"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


class _DiscardTextSink:
    """A text stream that accepts output without retaining it."""

    encoding = "utf-8"

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


def _non_negative_int(value: str) -> int:
    try:
        return validate_retries(int(value))
    except (ValueError, ImageOutputError) as error:
        raise argparse.ArgumentTypeError(
            f"must be non-negative and at most {MAX_RETRIES}"
        ) from error


def _module_for_engine(engine: object) -> str:
    try:
        module_name = ENGINE_MODULES.get(engine)
    except TypeError:
        module_name = None
    if module_name is None:
        raise ValueError("engine must be openai, gemini, or doubao")
    return module_name


def _result(
    status: GenerationStatus,
    engine: str,
    message: str,
) -> GenerationResult:
    return GenerationResult(status, engine, "api", safe_message=message)


def generate_result(
    prompt: str,
    out_path: str,
    engine: str = DEFAULT_ENGINE,
    retries: int = 2,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    reference_images: Optional[Sequence[str]] = None,
) -> GenerationResult:
    """Generate through one API provider and return its structured result."""
    try:
        retries = validate_retries(retries)
    except ImageOutputError:
        _module_for_engine(engine)
        return _result(
            GenerationStatus.INVALID_INPUT,
            engine,
            "invalid generation arguments",
        )
    if not isinstance(overwrite, bool):
        _module_for_engine(engine)
        return _result(
            GenerationStatus.INVALID_INPUT,
            engine,
            "invalid generation arguments",
        )

    module_name = _module_for_engine(engine)
    if reference_images is not None and not isinstance(reference_images, (list, tuple)):
        return _result(GenerationStatus.INVALID_INPUT, engine, "invalid reference image list")
    if reference_images and engine != "doubao":
        return _result(GenerationStatus.UNAVAILABLE, engine, "requested API adapter cannot preserve reference images")
    try:
        provider = importlib.import_module(module_name)
    except Exception:
        return _result(
            GenerationStatus.UNAVAILABLE,
            engine,
            "requested API provider is unavailable",
        )
    try:
        result = provider.generate_result(
            prompt,
            out_path,
            retries=retries,
            overwrite=overwrite,
            progress=progress,
            **({"reference_images": reference_images} if reference_images else {}),
        )
    except Exception:
        return _result(
            GenerationStatus.LOCAL_FAILURE,
            engine,
            "requested API provider failed locally",
        )
    if (
        not isinstance(result, GenerationResult)
        or result.provider != engine
        or result.channel != "api"
    ):
        return _result(
            GenerationStatus.LOCAL_FAILURE,
            engine,
            "requested API provider returned an invalid result",
        )
    return result


def _single_line_safe_text(value: object) -> str:
    text = _ANSI_ESCAPE.sub("", str(value))
    text = _CONTROL_CHARACTERS.sub(" ", text)
    return " ".join(safe_message(text).split())


def _human_summary(result: GenerationResult) -> str:
    message = _single_line_safe_text(result.safe_message)
    if result.ok:
        output_path = _single_line_safe_text(result.output_path)
        detail = f" ({message})" if message else ""
        return f"  OK: {output_path}{detail}"
    return f"  ERR: {message}"


def gen(
    prompt: str,
    out_path: str,
    engine: str = DEFAULT_ENGINE,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    try:
        result = generate_result(
            prompt,
            out_path,
            engine=engine,
            retries=retries,
            overwrite=overwrite,
            progress=None,
        )
    except ValueError:
        return False
    return result.ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 16:9 slide image through an API/CLI adapter; "
            "does not use host capabilities (default: OpenAI GPT Image 2)."
        )
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
    parser.add_argument("--reference-image", action="append", help="Local reference image; repeatable, currently supported by Doubao")
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one structured API-adapter result as JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.json:
        with redirect_stdout(_DiscardTextSink()):
            result = generate_result(
                args.prompt,
                args.out_path,
                engine=args.engine,
                retries=args.retries,
                overwrite=args.force,
                progress=None,
                **({"reference_images": args.reference_image} if args.reference_image else {}),
            )
        print(result.to_json())
    else:
        result = generate_result(
            args.prompt,
            args.out_path,
            engine=args.engine,
            retries=args.retries,
            overwrite=args.force,
            progress=None,
            **({"reference_images": args.reference_image} if args.reference_image else {}),
        )
        print(_human_summary(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
