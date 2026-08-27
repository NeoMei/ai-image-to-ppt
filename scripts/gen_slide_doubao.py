#!/usr/bin/env python3
"""Volcano Ark Doubao Seedream image generation (explicit fallback engine)."""

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

from generation_result import GenerationResult, GenerationStatus, classify_http_failure, safe_message
from image_output import (
    MAX_IMAGE_BYTES, ImageOutputError, ImageStreamError, PreparedTarget,
    expected_mime_type, output_format, parse_json_response, prepare_target, prepared_lock_target,
    preflight_output, publish_bytes, read_bounded_image_stream,
    read_response_body, resolve_output_path, validate_image_bytes, validate_retries,
)
from output_lock import OutputLockError, output_lock
from provider_credentials import APIKeyError, load_api_key, validate_api_key
from retry_delay import retry_delay

SECRET_PATH = Path("~/.secrets/doubao_api_key").expanduser()
URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MODEL = "doubao-seedream-5-0-260128"
SIZE = "2560x1440"
DOWNLOAD_TIMEOUT = 120
DOUBAO_POLICY_CODES = frozenset({"content_filter", "content_policy_violation", "input_text_risk", "output_image_risk"})
DOUBAO_RETRYABLE_CODES = frozenset({"quota_exceeded", "rate_limit_exceeded", "resource_exhausted"})
TRANSPORT_ERRORS = (ImageStreamError, urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException)


def _load_api_key() -> str:
    return load_api_key("DOUBAO_API_KEY", SECRET_PATH)


def _redact(message: object, key: str) -> str:
    return safe_message(message, (key,))


def _environment_redaction_secrets() -> Sequence[str]:
    """Collect the configured environment key only after local setup succeeds."""
    key = os.environ.get("DOUBAO_API_KEY")
    return (key,) if isinstance(key, str) and key else ()


def _result(status: GenerationStatus, message: object, output_path: Optional[str] = None, secrets: Sequence[str] = ()) -> GenerationResult:
    return GenerationResult(status, "doubao", "api", output_path=output_path, safe_message=safe_message(message, secrets))


def _error_details(error: urllib.error.HTTPError, key: str):
    code = None
    message = None
    try:
        raw_body = read_response_body(error)
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
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


def _http_error_message(error: urllib.error.HTTPError, key: str) -> str:
    return _error_details(error, key)[1]


def _extract_image_url(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ImageOutputError("response envelope must be an object")
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ImageOutputError("response data must contain an image object")
    image_url = data[0].get("url")
    if not isinstance(image_url, str) or not image_url:
        raise ImageOutputError("response image URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(image_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ImageOutputError("response image URL must use HTTPS")
    return image_url


def _remote_http_result(error: urllib.error.HTTPError, key: str) -> tuple[GenerationStatus, str]:
    error_code, message = _error_details(error, key)
    status = classify_http_failure(error.code, error_code, DOUBAO_POLICY_CODES, DOUBAO_RETRYABLE_CODES)
    if status is GenerationStatus.AUTH_UNAVAILABLE:
        message = f"authentication failed; check DOUBAO_API_KEY; provider says: {message}"
    return status, message


def _validate_download_content_type(response, target: PreparedTarget) -> None:
    headers = getattr(response, "headers", {})
    try:
        content_type = headers.get("Content-Type")
    except AttributeError as error:
        raise ImageOutputError("download Content-Type must be non-empty text") from error
    if not isinstance(content_type, str) or not content_type.strip():
        raise ImageOutputError("download Content-Type must be non-empty text")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != expected_mime_type(str(target)):
        raise ImageOutputError("download Content-Type does not match output target")


def _download_with_retries(image_url: str, target: PreparedTarget, retries: int, overwrite: bool, key: str, progress: Optional[Callable[[str], None]]) -> GenerationResult:
    """Download one generated URL; retries remain inside this exact URL."""
    secrets = (key,) if key else ()
    for attempt in range(retries + 1):
        request = urllib.request.Request(image_url)
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                _validate_download_content_type(response, target)
                image = read_bounded_image_stream(response, max_bytes=MAX_IMAGE_BYTES)
        except urllib.error.HTTPError as error:
            status, message = _remote_http_result(error, key)
            if status is GenerationStatus.RETRYABLE_EXHAUSTED and (error.code == 429 or error.code >= 500) and attempt < retries:
                if progress is not None:
                    progress(f"  HTTP {error.code}: image download failed: {message} (attempt {attempt + 1})")
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
            return _result(GenerationStatus.RETRYABLE_EXHAUSTED, message, secrets=secrets)
        except (ImageOutputError, TypeError, ValueError) as error:
            return _result(GenerationStatus.INVALID_OUTPUT, f"invalid downloaded image: {_redact(error, key)}", secrets=secrets)

        try:
            validate_image_bytes(image, target)
        except (ImageOutputError, TypeError, ValueError) as error:
            return _result(GenerationStatus.INVALID_OUTPUT, f"invalid downloaded image: {_redact(error, key)}", secrets=secrets)
        try:
            byte_count = publish_bytes(image, target, overwrite=overwrite)
        except (ImageOutputError, OSError) as error:
            return _result(GenerationStatus.LOCAL_FAILURE, f"output failure: {_redact(error, key)}", secrets=secrets)
        return _result(GenerationStatus.SUCCESS, f"{byte_count // 1024}KB, Doubao", str(target.path), secrets=secrets)
    return _result(GenerationStatus.RETRYABLE_EXHAUSTED, "download retries exhausted", secrets=secrets)


def _download_image_result(image_url: str, target: PreparedTarget, retries: int, overwrite: bool, key: str, progress: Optional[Callable[[str], None]]) -> GenerationResult:
    return _download_with_retries(image_url, target, retries, overwrite, key, progress)


def _gen_owned(prompt: str, target: PreparedTarget, retries: int = 2, overwrite: bool = False, progress: Optional[Callable[[str], None]] = None) -> GenerationResult:
    """Generate while the caller owns a prepared output lock."""
    try:
        requested_format = output_format(str(target))
        preflight_output(target, overwrite=overwrite)
    except ImageOutputError:
        return _result(GenerationStatus.LOCAL_FAILURE, "unable to prepare output target")

    redaction_secrets = _environment_redaction_secrets()
    key = _load_api_key()
    secrets = tuple(redaction_secrets) + ((key,) if isinstance(key, str) and key else ())
    if not key:
        return _result(GenerationStatus.AUTH_UNAVAILABLE, "Doubao API key not found. Set DOUBAO_API_KEY or create ~/.secrets/doubao_api_key (see README.md)", secrets=secrets)
    try:
        key = validate_api_key(key)
    except APIKeyError as error:
        return _result(GenerationStatus.AUTH_UNAVAILABLE, f"Doubao API key is invalid: {error}", secrets=secrets)

    body = json.dumps({
        "model": os.environ.get("DOUBAO_IMAGE_MODEL", MODEL), "prompt": prompt,
        "size": SIZE, "response_format": "url", "output_format": requested_format,
        "watermark": False, "sequential_image_generation": "disabled",
    }).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_response = read_response_body(response)
        except urllib.error.HTTPError as error:
            status, message = _remote_http_result(error, key)
            if status is GenerationStatus.RETRYABLE_EXHAUSTED and (error.code == 429 or error.code >= 500) and attempt < retries:
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
            return _result(GenerationStatus.RETRYABLE_EXHAUSTED, message, secrets=secrets)
        except ImageOutputError as error:
            return _result(GenerationStatus.INVALID_OUTPUT, f"invalid provider response: {_redact(error, key)}", secrets=secrets)

        try:
            payload = parse_json_response(raw_response)
            image_url = _extract_image_url(payload)
        except (ImageOutputError, TypeError, ValueError, UnicodeError) as error:
            return _result(GenerationStatus.INVALID_OUTPUT, f"invalid image response: {_redact(error, key)}", secrets=secrets)
        return _download_image_result(image_url, target, retries, overwrite, key, progress)
    return _result(GenerationStatus.RETRYABLE_EXHAUSTED, "request retries exhausted", secrets=secrets)


def _generate_result_with_lock(prompt: str, out_path: str, retries: int, overwrite: bool, progress: Optional[Callable[[str], None]]) -> GenerationResult:
    try:
        retries = validate_retries(retries)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageOutputError("prompt must be non-empty text")
        if not isinstance(overwrite, bool):
            raise ImageOutputError("overwrite must be a boolean")
        output_format(out_path)
    except (ImageOutputError, TypeError, ValueError):
        return _result(GenerationStatus.INVALID_INPUT, "invalid generation arguments")
    try:
        target = prepare_target(resolve_output_path(out_path))
        with output_lock(prepared_lock_target(target)):
            result = _gen_owned(prompt, target, retries, overwrite, progress)
    except (ImageOutputError, OutputLockError, OSError, TypeError, ValueError):
        return _result(GenerationStatus.LOCAL_FAILURE, "unable to prepare output target")
    if not isinstance(result, GenerationResult):
        return _result(GenerationStatus.LOCAL_FAILURE, "internal generator returned an invalid result")
    return result


def generate_result(prompt: str, out_path: str, retries: int = 2, overwrite: bool = False, progress: Optional[Callable[[str], None]] = None) -> GenerationResult:
    return _generate_result_with_lock(prompt, out_path, retries, overwrite, progress)


def gen(prompt: str, out_path: str, retries: int = 2, overwrite: bool = False) -> bool:
    result = generate_result(prompt, out_path, retries=retries, overwrite=overwrite, progress=print)
    if result.ok:
        print(f"  OK: {safe_message(result.output_path, _environment_redaction_secrets())} ({result.safe_message})")
    else:
        print(f"  ERR: {result.safe_message}")
    return result.ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", help="Output .jpg, .jpeg, .png, or .webp")
    parser.add_argument("prompt", help="Slide image prompt")
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return 0 if gen(args.prompt, args.output_path, overwrite=args.force) else 1


if __name__ == "__main__":
    raise SystemExit(main())
