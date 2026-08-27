#!/usr/bin/env python3
"""Gemini image generation (explicit fallback engine)."""

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
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

SECRET_PATH = Path("~/.secrets/gemini_api_key").expanduser()
DEFAULT_MODEL = "gemini-3.1-flash-image"
URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "{}:generateContent"
)
GEMINI_POLICY_REASONS = frozenset({
    "BLOCKLIST",
    "IMAGE_SAFETY",
    "PROHIBITED_CONTENT",
    "SAFETY",
})
GEMINI_POLICY_CODES = frozenset({"blocked", "safety"})
GEMINI_RETRYABLE_CODES = frozenset({
    "quota_exceeded",
    "rate_limit_exceeded",
    "resource_exhausted",
})
TRANSPORT_ERRORS = (
    ImageStreamError,
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
)


def _load_api_key() -> str:
    return load_api_key("GEMINI_API_KEY", SECRET_PATH)


def _output_config(out_path: str):
    extension = Path(out_path).suffix.lower()
    image_config = {"aspectRatio": "16:9", "imageSize": "2K"}
    if extension == ".png":
        return "image/png", image_config
    if extension in {".jpg", ".jpeg"}:
        image_config["mimeType"] = "IMAGE_JPEG"
        return "image/jpeg", image_config
    raise ImageOutputError(
        f"Unsupported Gemini output extension '{extension or '<none>'}'. "
        "Use: .jpeg, .jpg, .png"
    )


def _redact(message: object, key: str) -> str:
    return safe_message(message, (key,))


def _environment_redaction_secrets() -> Sequence[str]:
    """Collect the configured environment key after local validation succeeds."""
    key = os.environ.get("GEMINI_API_KEY")
    return (key,) if isinstance(key, str) and key else ()


def _result(
    status: GenerationStatus,
    message: object,
    output_path: Optional[str] = None,
    secrets: Sequence[str] = (),
) -> GenerationResult:
    return GenerationResult(
        status,
        "gemini",
        "api",
        output_path=output_path,
        safe_message=safe_message(message, secrets),
    )


def _http_error_details(error: urllib.error.HTTPError, key: str):
    code = None
    message = None
    try:
        raw_body = read_response_body(error)
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        payload = parse_json_response(raw_body)
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            details = payload["error"]
            candidate_code = details.get("code")
            if isinstance(candidate_code, str) and candidate_code:
                code = candidate_code
            candidate_message = details.get("message")
            if isinstance(candidate_message, (str, int, float, bool)) and candidate_message:
                message = candidate_message
    except Exception:
        pass
    return code, _redact(message or error.reason or "request failed", key)


def _http_error_message(error: urllib.error.HTTPError, key: str) -> str:
    return _http_error_details(error, key)[1]


def _policy_reason(payload: object) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    feedback = payload.get("promptFeedback")
    if isinstance(feedback, dict):
        reason = feedback.get("blockReason")
        if isinstance(reason, str) and reason.upper() in GEMINI_POLICY_REASONS:
            return reason.upper()
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            reason = candidate.get("finishReason")
            if isinstance(reason, str) and reason.upper() in GEMINI_POLICY_REASONS:
                return reason.upper()
    return None


def _extract_image(payload: object, target: Path, expected_mime: str) -> bytes:
    if not isinstance(payload, dict):
        raise ImageOutputError("response envelope must be an object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ImageOutputError("response candidates must be a non-empty list")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ImageOutputError("response candidate must be an object")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ImageOutputError("response content must be an object")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ImageOutputError("response parts must be a list")

    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if not isinstance(inline, dict) or not inline.get("data"):
            continue
        declared_mime = inline.get("mimeType") or inline.get("mime_type")
        if not isinstance(declared_mime, str) or not declared_mime:
            raise ImageOutputError("response MIME type must be non-empty text")
        if declared_mime.lower() != expected_mime:
            raise ImageOutputError(
                f"response MIME type {declared_mime} does not match {target.suffix.lower()}"
            )
        return decode_base64(inline.get("data"))
    raise ImageOutputError("response contains no image data")


def _gen_owned(
    prompt: str,
    target,
    retries: int = 2,
    overwrite: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    """Generate while the caller owns a prepared output lock."""
    try:
        expected_mime, image_config = _output_config(str(target))
        target = preflight_output(target, overwrite=overwrite)
    except ImageOutputError:
        return _result(GenerationStatus.LOCAL_FAILURE, "unable to prepare output target")

    redaction_secrets = _environment_redaction_secrets()
    key = _load_api_key()
    secrets = tuple(redaction_secrets) + ((key,) if isinstance(key, str) and key else ())
    if not key:
        return _result(
            GenerationStatus.AUTH_UNAVAILABLE,
            "Gemini API key not found. Set GEMINI_API_KEY or create "
            "~/.secrets/gemini_api_key (see README.md)",
            secrets=secrets,
        )
    try:
        key = validate_api_key(key)
    except APIKeyError as error:
        return _result(
            GenerationStatus.AUTH_UNAVAILABLE,
            f"Gemini API key is invalid: {error}",
            secrets=secrets,
        )

    model = os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    encoded_model = urllib.parse.quote(model, safe="")
    url = URL_TEMPLATE.format(encoded_model)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "responseFormat": {
                "image": image_config
            },
        },
    }).encode("utf-8")

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw_response = read_response_body(response)
        except urllib.error.HTTPError as error:
            error_code, remote_message = _http_error_details(error, key)
            status = classify_http_failure(
                error.code,
                error_code,
                GEMINI_POLICY_CODES,
                GEMINI_RETRYABLE_CODES,
            )
            if status is GenerationStatus.AUTH_UNAVAILABLE:
                message = (
                    "authentication failed; check GEMINI_API_KEY; provider says: "
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
        except TRANSPORT_ERRORS as error:
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
            payload = parse_json_response(raw_response)
        except ImageOutputError:
            return _result(
                GenerationStatus.INVALID_OUTPUT,
                "invalid JSON response from Gemini",
                secrets=secrets,
            )

        if _policy_reason(payload) is not None:
            return _result(
                GenerationStatus.POLICY_REFUSED,
                "Gemini declined the image request due to policy",
                secrets=secrets,
            )

        try:
            image = _extract_image(payload, target, expected_mime)
            validate_image_bytes(image, target)
        except (ImageOutputError, KeyError, IndexError, TypeError, ValueError) as error:
            return _result(
                GenerationStatus.INVALID_OUTPUT,
                f"invalid image response: {_redact(error, key)}",
                secrets=secrets,
            )
        try:
            byte_count = publish_bytes(image, target, overwrite=overwrite)
        except (ImageOutputError, OSError, TypeError, NotImplementedError) as error:
            return _result(
                GenerationStatus.LOCAL_FAILURE,
                f"output failure: {_redact(error, key)}",
                secrets=secrets,
            )

        return _result(
            GenerationStatus.SUCCESS,
            f"{byte_count // 1024}KB, Gemini {_redact(model, key)}",
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
        _output_config(out_path)
    except (ImageOutputError, TypeError, ValueError):
        return _result(GenerationStatus.INVALID_INPUT, "invalid generation arguments")

    try:
        target = prepare_target(resolve_output_path(out_path))
        with output_lock(prepared_lock_target(target)):
            result = _gen_owned(prompt, target, retries, overwrite, progress)
    except (
        ImageOutputError,
        OutputLockError,
        OSError,
        TypeError,
        ValueError,
        NotImplementedError,
    ):
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
        progress=None,
    )
    if result.ok:
        redaction_secrets = _environment_redaction_secrets()
        print(
            f"  OK: {safe_message(result.output_path, redaction_secrets)} "
            f"({safe_message(result.safe_message)})"
        )
    else:
        print(f"  ERR: {safe_message(result.safe_message)}")
    return result.ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="Output .png (default), .jpg, or .jpeg")
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
