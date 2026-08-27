#!/usr/bin/env python3
"""Safely import a host-generated slide image into one workspace."""

import argparse
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import image_output
from generation_result import GenerationResult, GenerationStatus
from image_output import (
    ImageOutputError,
    PreparedTarget,
    decode_base64,
    prepare_workspace_target,
    prepared_lock_target,
    preflight_output,
    publish_bytes,
    read_bounded_image_stream,
    resolve_input_path,
    validate_image_bytes,
)
from output_lock import OutputLockError, output_lock


class HostArtifactKind(str, Enum):
    LOCAL_PATH = "local_path"
    INLINE_BYTES = "inline_bytes"
    BASE64 = "base64"
    DATA_URL = "data_url"


@dataclass(frozen=True)
class HostArtifact:
    kind: HostArtifactKind
    value: Union[str, bytes]
    mime_type: Optional[str] = None

    @classmethod
    def local_path(cls, path: str) -> "HostArtifact":
        return cls(HostArtifactKind.LOCAL_PATH, path)

    @classmethod
    def inline_bytes(cls, data: bytes, mime_type: str) -> "HostArtifact":
        return cls(HostArtifactKind.INLINE_BYTES, data, mime_type)

    @classmethod
    def base64(cls, encoded: str, mime_type: str) -> "HostArtifact":
        return cls(HostArtifactKind.BASE64, encoded, mime_type)

    @classmethod
    def data_url(cls, value: str) -> "HostArtifact":
        return cls(HostArtifactKind.DATA_URL, value)


def _failure(
    status: GenerationStatus,
    provider: str,
    message: str,
) -> GenerationResult:
    return GenerationResult(status, provider, "host", safe_message=message)


def _validate_provider(provider: object) -> str:
    if provider not in {"openai", "gemini", "doubao"}:
        raise ValueError("provider must be openai, gemini, or doubao")
    return provider


def _read_local_path(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise OSError("host local path must be a non-empty string")
    try:
        source = resolve_input_path(value)
        before = source.stat()
        if not stat.S_ISREG(before.st_mode):
            raise OSError("host local path is not a regular file")
        if before.st_size <= 0:
            raise OSError("host local image is empty")
        if before.st_size > image_output.MAX_IMAGE_BYTES:
            raise OSError("host local image exceeds maximum size")
        descriptor = os.open(str(source), os.O_RDONLY)
    except ImageOutputError as error:
        raise OSError("could not resolve host local path") from error
    except (OSError, ValueError) as error:
        raise OSError("could not open host local image") from error

    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
            ):
                raise OSError("host local image identity changed before reading")
            data = read_bounded_image_stream(
                stream, max_bytes=image_output.MAX_IMAGE_BYTES
            )
            after = os.fstat(stream.fileno())
        current = source.stat()
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
        ):
            raise OSError("host local image identity changed during reading")
        return data
    except (ImageOutputError, OSError, ValueError) as error:
        raise OSError("could not read host local image") from error


def _decode_data_url(value: object) -> Tuple[bytes, str]:
    if not isinstance(value, str) or not value:
        raise ImageOutputError("image data URL must be a non-empty string")
    max_length = image_output.MAX_ENCODED_IMAGE_BYTES + 1024
    if len(value) > max_length:
        raise ImageOutputError("image data URL exceeds maximum encoded size")
    matched = re.fullmatch(r"data:([^;,]+);base64,([A-Za-z0-9+/=]+)", value)
    if matched is None:
        raise ImageOutputError("image data URL must contain Base64 image data")
    return decode_base64(matched.group(2)), matched.group(1)


def _artifact_bytes(artifact: HostArtifact) -> Tuple[bytes, Optional[str]]:
    if not isinstance(artifact, HostArtifact):
        raise ImageOutputError("host artifact is invalid")
    if artifact.kind == HostArtifactKind.LOCAL_PATH:
        return _read_local_path(artifact.value), None
    if artifact.kind == HostArtifactKind.INLINE_BYTES:
        if not isinstance(artifact.value, bytes):
            raise ImageOutputError("inline image data must be bytes")
        return artifact.value, artifact.mime_type
    if artifact.kind == HostArtifactKind.BASE64:
        return decode_base64(artifact.value), artifact.mime_type
    if artifact.kind == HostArtifactKind.DATA_URL:
        return _decode_data_url(artifact.value)
    raise ImageOutputError("host artifact kind is unsupported or ambiguous")


def _validate_mime(mime_type: Optional[str], target: PreparedTarget) -> None:
    if mime_type is None:
        return
    if not isinstance(mime_type, str) or not mime_type:
        raise ImageOutputError("host image MIME type is invalid")
    expected = image_output.expected_mime_type(str(target))
    if mime_type.lower() != expected:
        raise ImageOutputError(
            f"host image MIME type {mime_type!r} does not match {expected}"
        )


def import_host_artifact(
    artifact: HostArtifact,
    out_path: object,
    workspace_root: object,
    provider: str,
    overwrite: bool = False,
) -> GenerationResult:
    """Validate and atomically publish a host artifact inside a workspace."""
    provider = _validate_provider(provider)
    if not isinstance(overwrite, bool):
        return _failure(
            GenerationStatus.LOCAL_FAILURE,
            provider,
            "output overwrite flag is invalid",
        )
    try:
        target = prepare_workspace_target(out_path, workspace_root)
        image_output.output_format(str(target))
        with output_lock(prepared_lock_target(target)):
            target = preflight_output(target, overwrite=overwrite)
            try:
                data, mime_type = _artifact_bytes(artifact)
            except OSError:
                return _failure(
                    GenerationStatus.LOCAL_FAILURE,
                    provider,
                    "host local artifact could not be read",
                )
            except ImageOutputError:
                return _failure(
                    GenerationStatus.INVALID_OUTPUT,
                    provider,
                    "host artifact is not a valid image output",
                )

            try:
                validate_image_bytes(data, target)
                _validate_mime(mime_type, target)
            except ImageOutputError:
                return _failure(
                    GenerationStatus.INVALID_OUTPUT,
                    provider,
                    "host artifact is not a valid image output",
                )

            try:
                publish_bytes(data, target, overwrite=overwrite)
            except ImageOutputError:
                return _failure(
                    GenerationStatus.LOCAL_FAILURE,
                    provider,
                    "host artifact could not be published",
                )
    except (ImageOutputError, OutputLockError):
        return _failure(
            GenerationStatus.LOCAL_FAILURE,
            provider,
            "workspace output is unavailable",
        )

    return GenerationResult(
        GenerationStatus.SUCCESS,
        provider,
        "host",
        output_path=str(target.path),
        safe_message="host artifact imported",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Host-local source image path")
    parser.add_argument("output", help="Workspace-relative output image path")
    parser.add_argument("--workspace-root", required=True, help="Existing workspace root")
    parser.add_argument(
        "--provider", required=True, choices=("openai", "gemini", "doubao")
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing file")
    parser.add_argument("--json", action="store_true", help="Write one JSON result")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = import_host_artifact(
        HostArtifact.local_path(args.source),
        args.output,
        args.workspace_root,
        args.provider,
        overwrite=args.force,
    )
    if args.json:
        print(result.to_json())
    elif result.ok:
        print(f"OK: imported host image to {result.output_path}")
    else:
        print(f"ERR: {result.safe_message}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
