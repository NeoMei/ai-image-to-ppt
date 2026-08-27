#!/usr/bin/env python3
"""Generate one slide image with OpenAI GPT Image (default engine)."""

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

from generation_result import (
    GenerationResult,
    GenerationStatus,
    classify_http_failure,
    safe_message,
)
from image_output import (
    ImageOutputError,
    ImageStreamError,
    decode_base64,
    output_format,
    parse_json_response,
    prepare_target,
    prepared_lock_target,
    preflight_output,
    publish_bytes,
    read_response_body,
    resolve_output_path,
    validate_image_bytes,
    validate_retries,
)
from output_lock import OutputLockError, output_lock
from provider_credentials import APIKeyError, load_api_key, validate_api_key
from retry_delay import retry_delay

API_URL = "https://api.openai.com/v1/images/generations"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "medium"
SECRET_PATH = Path("~/.secrets/openai_api_key").expanduser()
OPENAI_POLICY_CODES = frozenset({
    "content_policy_violation",
    "moderation_blocked",
})
OPENAI_RETRYABLE_CODES = frozenset({
    "billing_hard_limit_reached",
    "insufficient_quota",
    "rate_limit_exceeded",
})


def _load_api_key() -> str:
    return load_api_key("OPENAI_API_KEY", SECRET_PATH)


def _output_format(out_path: str) -> str:
    return output_format(out_path)


def _redact(message: object, key: str) -> str:
    return safe_message(message, (key,))


def _environment_redaction_secrets() -> Sequence[str]:
    """Collect the configured environment key for credential-stage diagnostics."""
    key = os.environ.get("OPENAI_API_KEY")
    return (key,) if isinstance(key, str) and key else ()


def _error_details(error: urllib.error.HTTPError, key: str):
    code = None
    message = None
    try:
        raw_body = read_response_body(error)
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        if not isinstance(raw_body, str):
            raise TypeError("HTTP error body must be text or bytes")
        payload = parse_json_response(raw_body)
        if not isinstance(payload, dict):
            raise TypeError("HTTP error envelope must be an object")
        details = payload.get("error")
        if not isinstance(details, dict):
            raise TypeError("HTTP error details must be an object")
        candidate_code = details.get("code")
        if isinstance(candidate_code, str) and candidate_code:
            code = candidate_code
        candidate_message = details.get("message")
        if isinstance(candidate_message, (str, int, float, bool)) and candidate_message:
            message = candidate_message
    except Exception:
        pass
    return code, _redact(message or error.reason or "request failed", key)


def _error_message(error: urllib.error.HTTPError, key: str) -> str:
    return _error_details(error, key)[1]


def _decode_image_response(result: object) -> bytes:
    if not isinstance(result, dict):
        raise ValueError("response envelope must be an object")
    data = result.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("response data must be a non-empty list")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("response image entry must be an object")
    encoded = first.get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("response b64_json must be a non-empty string")
    return decode_base64(encoded)


def _result(
    status: GenerationStatus,
    message: object,
    output_path: Optional[str] = None,
    secrets: Sequence[str] = (),
) -> GenerationResult:
    return GenerationResult(
        status,
        "openai",
        "api",
        output_path=output_path,
        safe_message=safe_message(message, secrets),
    )


def _gen_owned(
    prompt: str,
    target,
    retries: int = 2,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    """Generate while the caller owns a prepared output lock."""
    try:
        output_format_value = _output_format(str(target))
        preflight_output(target, overwrite=overwrite)
    except ImageOutputError:
        return _result(GenerationStatus.LOCAL_FAILURE, "unable to prepare output target")

    redaction_secrets = _environment_redaction_secrets()
    key = _load_api_key()
    secrets = tuple(redaction_secrets) + ((key,) if isinstance(key, str) and key else ())
    if not key:
        return _result(
            GenerationStatus.AUTH_UNAVAILABLE,
            "OpenAI API key not found. Set OPENAI_API_KEY or create "
            "~/.secrets/openai_api_key",
            secrets=secrets,
        )
    try:
        key = validate_api_key(key)
    except APIKeyError as error:
        return _result(
            GenerationStatus.AUTH_UNAVAILABLE,
            f"OpenAI API key is invalid: {error}",
            secrets=secrets,
        )

    payload = {
        "model": os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_MODEL),
        "prompt": prompt,
        "size": os.environ.get("OPENAI_IMAGE_SIZE", DEFAULT_SIZE),
        "quality": os.environ.get("OPENAI_IMAGE_QUALITY", DEFAULT_QUALITY),
        "output_format": output_format_value,
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_response = read_response_body(response)
        except urllib.error.HTTPError as error:
            error_code, remote_message = _error_details(error, key)
            status = classify_http_failure(
                error.code,
                error_code,
                OPENAI_POLICY_CODES,
                OPENAI_RETRYABLE_CODES,
            )
            if status is GenerationStatus.AUTH_UNAVAILABLE:
                message = (
                    "authentication failed; check OPENAI_API_KEY; provider says: "
                    f"{remote_message}"
                )
            else:
                message = remote_message
            should_retry = (
                status is GenerationStatus.RETRYABLE_EXHAUSTED
                and (error.code == 429 or error.code >= 500)
                and attempt < retries
            )
            if should_retry:
                if progress is not None:
                    progress(f"  HTTP {error.code}: {message} (attempt {attempt + 1})")
                time.sleep(retry_delay(attempt, error.headers))
                continue
            return _result(status, message, secrets=secrets)
        except (
            ImageStreamError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as error:
            message = _redact(error, key)
            if attempt < retries:
                if progress is not None:
                    progress(f"  ERR: {message} (attempt {attempt + 1})")
                time.sleep(retry_delay(attempt))
                continue
            return _result(
                GenerationStatus.RETRYABLE_EXHAUSTED,
                message,
                secrets=secrets,
            )
        except ImageOutputError as error:
            return _result(
                GenerationStatus.INVALID_OUTPUT,
                f"invalid provider response: {_redact(error, key)}",
                secrets=secrets,
            )

        try:
            result = parse_json_response(raw_response)
        except ImageOutputError:
            return _result(
                GenerationStatus.INVALID_OUTPUT,
                "invalid JSON response from OpenAI",
                secrets=secrets,
            )

        try:
            image = _decode_image_response(result)
            validate_image_bytes(image, target)
        except (ImageOutputError, KeyError, IndexError, TypeError, ValueError) as error:
            return _result(
                GenerationStatus.INVALID_OUTPUT,
                f"invalid image response: {_redact(error, key)}",
                secrets=secrets,
            )
        try:
            byte_count = publish_bytes(image, target, overwrite=overwrite)
        except (ImageOutputError, OSError) as error:
            return _result(
                GenerationStatus.LOCAL_FAILURE,
                f"output failure: {_redact(error, key)}",
                secrets=secrets,
            )

        return _result(
            GenerationStatus.SUCCESS,
            f"{byte_count // 1024}KB, OpenAI {_redact(payload['model'], key)}",
            str(target.path),
            secrets=secrets,
        )

    return _result(
        GenerationStatus.RETRYABLE_EXHAUSTED,
        "request retries exhausted",
        secrets=secrets,
    )


def _generate_result_with_lock(
    prompt: str,
    out_path: str,
    retries: int,
    overwrite: bool,
    progress: Optional[Callable[[str], None]],
) -> GenerationResult:
    try:
        retries = validate_retries(retries)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageOutputError("prompt must be non-empty text")
        if not isinstance(overwrite, bool):
            raise ImageOutputError("overwrite must be a boolean")
        _output_format(out_path)
    except (ImageOutputError, TypeError, ValueError):
        return _result(GenerationStatus.INVALID_INPUT, "invalid generation arguments")

    try:
        target = prepare_target(resolve_output_path(out_path))
        with output_lock(prepared_lock_target(target)):
            result = _gen_owned(
                prompt,
                target,
                retries,
                overwrite,
                progress,
            )
    except (ImageOutputError, OutputLockError, OSError, TypeError, ValueError):
        return _result(GenerationStatus.LOCAL_FAILURE, "unable to prepare output target")

    if not isinstance(result, GenerationResult):
        return _result(
            GenerationStatus.LOCAL_FAILURE,
            "internal generator returned an invalid result",
        )
    return result


def generate_result(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    return _generate_result_with_lock(prompt, out_path, retries, overwrite, progress)


def gen(
    prompt: str,
    out_path: str,
    retries: int = 2,
    overwrite: bool = False,
) -> bool:
    result = generate_result(
        prompt,
        out_path,
        retries=retries,
        overwrite=overwrite,
        progress=print,
    )
    if result.ok:
        redaction_secrets = _environment_redaction_secrets()
        print(
            f"  OK: {safe_message(result.output_path, redaction_secrets)} "
            f"({result.safe_message})"
        )
    else:
        print(f"  ERR: {result.safe_message}")
    return result.ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="Output .jpg, .jpeg, .png, or .webp")
    parser.add_argument("prompt", help="Slide image prompt")
    parser.add_argument(
        "--force", action="store_true", help="Atomically replace an existing output"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return 0 if gen(args.prompt, args.output_path, overwrite=args.force) else 1


if __name__ == "__main__":
    raise SystemExit(main())
