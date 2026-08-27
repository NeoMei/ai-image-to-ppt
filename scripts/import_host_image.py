#!/usr/bin/env python3
"""Safely import a host-generated slide image into one workspace."""

import argparse
import io
import os
import re
import stat
from contextlib import ExitStack
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
    read_bounded_image_stream,
    resolve_input_path,
    validate_image_bytes,
    verify_parent_identity,
)
from output_lock import OutputLockError, output_lock


class HostArtifactKind(str, Enum):
    LOCAL_PATH = "local_path"
    INLINE_BYTES = "inline_bytes"
    BASE64 = "base64"
    DATA_URL = "data_url"


HOST_ASPECT_ERROR_DENOMINATOR = 200


@dataclass
class _OutputSnapshot:
    target: PreparedTarget
    original_bytes: Optional[bytes]
    original_identity: Optional[Tuple[int, int]]
    published_identity: Optional[Tuple[int, int]] = None


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
    except ImageOutputError as error:
        raise OSError("could not resolve host local path") from error
    except (OSError, ValueError) as error:
        raise OSError("could not open host local image") from error
    if before.st_size <= 0:
        raise ImageOutputError("host local image is empty")
    if before.st_size > image_output.MAX_IMAGE_BYTES:
        raise ImageOutputError("host local image exceeds maximum size")
    try:
        descriptor = os.open(str(source), os.O_RDONLY)
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
    except ImageOutputError:
        raise
    except (OSError, ValueError) as error:
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


def _raw_workspace_target(
    target: PreparedTarget, workspace_root: object
) -> PreparedTarget:
    """Reserve the recoverable raw sibling before publishing a master."""
    return prepare_workspace_target(
        target.path.parent / "raw" / target.name,
        workspace_root,
    )


def _normalize_host_image(data: bytes, target: PreparedTarget) -> bytes:
    """Return strict host master bytes without relaxing API validation."""
    loaded = image_output.validate_decoded_image_bytes(data, target, copy_image=True)
    width, height = loaded.width, loaded.height
    k = min(width // 16, height // 9)
    if k < 1:
        raise ImageOutputError("host image is too small for a 16:9 crop")
    cross_product_error = abs(width * 9 - height * 16)
    if cross_product_error * HOST_ASPECT_ERROR_DENOMINATOR > height * 16:
        raise ImageOutputError("host image is outside the 0.5% 16:9 tolerance")

    target_size = (16 * k, 9 * k)
    if (width, height) == target_size:
        return data

    image = loaded.image
    if image is None:
        raise ImageOutputError("host image could not be decoded for normalization")
    left = (width - target_size[0]) // 2
    top = (height - target_size[1]) // 2
    cropped = image.crop((left, top, left + target_size[0], top + target_size[1]))
    try:
        buffer = io.BytesIO()
        cropped.save(
            buffer,
            format=image_output.PIL_FORMATS[image_output.output_format(str(target))],
        )
        return buffer.getvalue()
    except (OSError, ValueError) as error:
        raise ImageOutputError("host image could not be encoded after normalization") from error


def _identity(path: Path) -> Optional[Tuple[int, int]]:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ImageOutputError("could not inspect transaction output") from error
    if not stat.S_ISREG(current.st_mode):
        raise ImageOutputError("transaction output identity is unsafe")
    return current.st_dev, current.st_ino


def _snapshot_output(target: PreparedTarget) -> _OutputSnapshot:
    """Capture an existing bounded output before a two-file host publication."""
    verify_parent_identity(target.parent)
    identity = _identity(target.path)
    if identity is None:
        return _OutputSnapshot(target, None, None)
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(target.path), flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != identity
        ):
            raise ImageOutputError("transaction output identity changed before reading")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = read_bounded_image_stream(stream)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != identity:
            raise ImageOutputError("transaction output identity changed during reading")
    except (ImageOutputError, OSError, ValueError) as error:
        raise ImageOutputError("could not snapshot transaction output") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    verify_parent_identity(target.parent)
    if _identity(target.path) != identity:
        raise ImageOutputError("transaction output identity changed before publication")
    return _OutputSnapshot(target, data, identity)


def _remove_owned_output(target: PreparedTarget, expected: Tuple[int, int]) -> None:
    verify_parent_identity(target.parent)
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(target.parent.path), flags)
        parent = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or (parent.st_dev, parent.st_ino)
            != (target.parent.device, target.parent.inode)
        ):
            raise ImageOutputError("transaction parent identity changed during rollback")
        current = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != expected
        ):
            raise ImageOutputError("transaction output identity changed during rollback")
        os.unlink(target.name, dir_fd=descriptor)
    except OSError as error:
        raise ImageOutputError("could not remove transaction output") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    verify_parent_identity(target.parent)


def _restore_snapshot(
    snapshot: _OutputSnapshot,
) -> None:
    """Compensate a failed two-file publication while both target locks hold."""
    current_identity = _identity(snapshot.target.path)
    if current_identity == snapshot.original_identity:
        return
    if snapshot.published_identity is None:
        if current_identity is None:
            return
        raise ImageOutputError("transaction output ownership is unknown during rollback")
    expected_identity = snapshot.published_identity

    if snapshot.original_bytes is None:
        _remove_owned_output(snapshot.target, expected_identity)
        return

    if _identity(snapshot.target.path) != expected_identity:
        raise ImageOutputError("transaction output identity changed during rollback")
    image_output.publish_opaque_bytes_with_identity(
        snapshot.original_bytes,
        snapshot.target,
        expected_existing_identity=expected_identity,
    )


def _publish_transaction_member(
    data: bytes,
    target: PreparedTarget,
    overwrite: bool,
    strict: bool,
) -> Tuple[int, int]:
    """Publish and return the inode installed by this transaction member."""
    if strict:
        published = image_output.publish_bytes_with_identity(
            data, target, overwrite=overwrite
        )
    else:
        published = image_output.publish_decoded_image_bytes_with_identity(
            data, target, overwrite=overwrite
        )
    return published.device, published.inode


def _transaction_checkpoint(_phase: str, _snapshot: _OutputSnapshot) -> None:
    """A no-op boundary after publication ownership has been recorded."""


def _publish_host_pair(
    raw_data: bytes,
    normalized_data: bytes,
    raw_target: PreparedTarget,
    target: PreparedTarget,
    overwrite: bool,
) -> None:
    """Publish raw and master as one compensated transaction under both locks."""
    raw_snapshot = _snapshot_output(raw_target)
    master_snapshot = _snapshot_output(target)
    try:
        raw_snapshot.published_identity = _publish_transaction_member(
            raw_data, raw_target, overwrite, strict=False
        )
        _transaction_checkpoint("raw", raw_snapshot)
        master_snapshot.published_identity = _publish_transaction_member(
            normalized_data, target, overwrite, strict=True
        )
        _transaction_checkpoint("master", master_snapshot)
    except ImageOutputError as original_error:
        rollback_error = None
        for snapshot in (master_snapshot, raw_snapshot):
            try:
                _restore_snapshot(snapshot)
            except ImageOutputError as error:
                rollback_error = error
        if rollback_error is not None:
            raise ImageOutputError(
                "host artifact transaction failed and rollback was incomplete"
            ) from rollback_error
        raise ImageOutputError("host artifact transaction could not be published") from original_error


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
        raw_target = _raw_workspace_target(target, workspace_root)
        image_output.output_format(str(raw_target))
        lock_targets = sorted(
            (target, raw_target), key=lambda item: str(prepared_lock_target(item))
        )
        with ExitStack() as locks:
            for lock_target in lock_targets:
                locks.enter_context(output_lock(prepared_lock_target(lock_target)))
            target = preflight_output(target, overwrite=overwrite)
            raw_target = preflight_output(raw_target, overwrite=overwrite)
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
                _validate_mime(mime_type, target)
                normalized = _normalize_host_image(data, target)
                validate_image_bytes(normalized, target)
            except ImageOutputError:
                return _failure(
                    GenerationStatus.INVALID_OUTPUT,
                    provider,
                    "host artifact is not a valid image output",
                )

            try:
                _publish_host_pair(
                    data,
                    normalized,
                    raw_target,
                    target,
                    overwrite,
                )
            except ImageOutputError as error:
                message = "host artifact could not be published"
                if "rollback was incomplete" in str(error):
                    message = "host artifact transaction rollback was incomplete"
                return _failure(
                    GenerationStatus.LOCAL_FAILURE,
                    provider,
                    message,
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
