"""Load and strictly validate the published Skill capability manifest."""

import json
import stat
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple


MANIFEST_PATH = Path("references/capabilities.json")
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_CONTRACTS = {
    "generationResult": 1,
    "serialStickyRouterReport": 1,
    "hostImageImport": 1,
    "editableInput": 1,
}
EXPECTED_ROUTES = (
    ("openai", "host", "host-owned"),
    ("openai", "api", "gpt-image-2"),
    ("gemini", "host", "host-owned"),
    ("gemini", "api", "gemini-3.1-flash-image"),
    ("doubao", "host", "host-owned"),
    ("doubao", "api", "doubao-seedream-5-0-pro-260628"),
)
EXPECTED_OUTPUTS = {
    "normalizedSlide": {"format": "image", "width": 1920, "height": 1080},
    "editableInput": {"format": "png", "width": 1280, "height": 720},
}
EXPECTED_REFERENCES = {
    "schemaVersion": 1, "apiProviders": ["doubao"], "cliOption": "--reference-image",
    "input": "local-file", "maxImages": 10, "maxBytesPerImage": 10 * 1024 * 1024,
    "maxTotalBytes": 30 * 1024 * 1024, "formats": ["png", "jpeg", "webp"],
    "unsupported": "unavailable; never drop references",
}
EXPECTED_SCRIPTS = {
    "generationResult",
    "hostRoutingPolicy",
    "importHostImage",
    "prepareEditableInput",
    "apiGenerator",
    "normalizedExport",
}


class CapabilityManifestError(ValueError):
    """A controlled, user-safe manifest validation failure."""


def _reject_duplicate_members(pairs: Iterable[Tuple[str, object]]) -> Dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CapabilityManifestError(
                "capability manifest contains a duplicate JSON member"
            )
        result[key] = value
    return result


def load_capability_manifest(skill_root: Path) -> dict:
    """Read and decode the capability manifest below ``skill_root``."""

    manifest_path = Path(skill_root) / MANIFEST_PATH
    try:
        content = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CapabilityManifestError("capability manifest is missing or unreadable") from error
    try:
        manifest = json.loads(content, object_pairs_hook=_reject_duplicate_members)
    except CapabilityManifestError:
        raise
    except (ValueError, RecursionError) as error:
        raise CapabilityManifestError("capability manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise CapabilityManifestError("capability manifest has wrong type")
    return manifest


def _require_exact_keys(value: object, expected: Iterable[str], label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise CapabilityManifestError(f"{label} has wrong type")
    expected_keys = set(expected)
    if set(value) != expected_keys:
        raise CapabilityManifestError(f"{label} keys do not match the supported contract")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CapabilityManifestError(f"{label} has wrong type")
    return value


def _require_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise CapabilityManifestError(f"{label} has wrong type")
    return value


def _validate_contracts(value: object) -> None:
    contracts = _require_exact_keys(value, SUPPORTED_CONTRACTS, "contracts")
    for name, supported_version in SUPPORTED_CONTRACTS.items():
        version = _require_integer(contracts[name], f"contract {name}")
        if version != supported_version:
            raise CapabilityManifestError(f"contract {name} uses an unsupported version")


def _validate_routes(value: object) -> None:
    if not isinstance(value, list):
        raise CapabilityManifestError("routingOrder has wrong type")

    route_pairs = []
    for index, route_value in enumerate(value):
        if not isinstance(route_value, dict):
            raise CapabilityManifestError(f"routingOrder item {index} has wrong type")
        if "provider" not in route_value or "channel" not in route_value:
            raise CapabilityManifestError(f"routingOrder item {index} keys do not match the supported contract")
        provider = _require_string(route_value["provider"], f"routingOrder item {index} provider")
        channel = _require_string(route_value["channel"], f"routingOrder item {index} channel")
        route_pairs.append((provider, channel))

    expected_pairs = [(provider, channel) for provider, channel, _ in EXPECTED_ROUTES]
    if len(set(route_pairs)) != len(route_pairs):
        raise CapabilityManifestError("routing order must contain unique routes")
    if route_pairs != expected_pairs:
        raise CapabilityManifestError("routing order must match the host-first sticky contract")

    for index, (route_value, expected_route) in enumerate(zip(value, EXPECTED_ROUTES)):
        _provider, channel, expected_model = expected_route
        model_key = "modelSelection" if channel == "host" else "defaultModel"
        route = _require_exact_keys(
            route_value,
            {"provider", "channel", model_key},
            f"routingOrder item {index}",
        )
        model = _require_string(route[model_key], f"routingOrder item {index} model")
        if model != expected_model:
            raise CapabilityManifestError(
                f"routingOrder item {index} model does not match the supported contract"
            )


def _validate_outputs(value: object) -> None:
    outputs = _require_exact_keys(value, EXPECTED_OUTPUTS, "outputs")
    for output_name, expected_output in EXPECTED_OUTPUTS.items():
        output = _require_exact_keys(
            outputs[output_name], expected_output, f"output {output_name}"
        )
        output_format = _require_string(output["format"], f"output {output_name} format")
        width = _require_integer(output["width"], f"output {output_name} width")
        height = _require_integer(output["height"], f"output {output_name} height")
        if {
            "format": output_format,
            "width": width,
            "height": height,
        } != expected_output:
            raise CapabilityManifestError(
                f"output {output_name} does not match the required format and dimensions"
            )


def _validate_script_path(skill_root: Path, declared_path: object) -> None:
    relative_path = _require_string(declared_path, "script path")
    path_parts = relative_path.split("/")
    if (
        not relative_path
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in path_parts)
        or any(
            unicodedata.category(character) in {"Cc", "Cs"}
            for character in relative_path
        )
    ):
        raise CapabilityManifestError("script path must be a safe relative script path")

    try:
        resolved_root = skill_root.resolve(strict=True)
        declared = resolved_root.joinpath(*path_parts)
        resolved_script = declared.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CapabilityManifestError("declared script must resolve to a regular file") from error

    try:
        resolved_script.relative_to(resolved_root)
    except ValueError as error:
        raise CapabilityManifestError("declared script must remain under the resolved Skill root") from error

    current = resolved_root
    try:
        for part in path_parts:
            current = current / part
            if stat.S_ISLNK(current.lstat().st_mode):
                raise CapabilityManifestError("declared script path must not contain a symlink")
        if not stat.S_ISREG(declared.lstat().st_mode):
            raise CapabilityManifestError("declared script must resolve to a regular file")
    except CapabilityManifestError:
        raise
    except OSError as error:
        raise CapabilityManifestError("declared script must resolve to a regular file") from error


def _validate_scripts(skill_root: Path, value: object) -> None:
    scripts = _require_exact_keys(value, EXPECTED_SCRIPTS, "scripts")
    for declared_path in scripts.values():
        _validate_script_path(skill_root, declared_path)


def _validate_manifest(skill_root: Path, manifest: dict) -> None:
    top_level = _require_exact_keys(
        manifest,
        {"schemaVersion", "skill", "contracts", "routingOrder", "outputs", "scripts"} | ({"referenceImages"} if "referenceImages" in manifest else set()),
        "capability manifest",
    )
    schema_version = _require_integer(top_level["schemaVersion"], "schemaVersion")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise CapabilityManifestError("capability manifest uses an unsupported schema version")
    skill = _require_string(top_level["skill"], "skill")
    if skill != "ai-image-to-ppt":
        raise CapabilityManifestError("capability manifest identifies the wrong skill")
    if "referenceImages" in top_level:
        references = _require_exact_keys(top_level["referenceImages"], EXPECTED_REFERENCES, "referenceImages")
        if json.dumps(references, sort_keys=True) != json.dumps(EXPECTED_REFERENCES, sort_keys=True):
            raise CapabilityManifestError("referenceImages does not match the supported contract")
    _validate_contracts(top_level["contracts"])
    _validate_routes(top_level["routingOrder"])
    _validate_outputs(top_level["outputs"])
    _validate_scripts(skill_root, top_level["scripts"])
    if "referenceImages" in top_level:
        _validate_script_path(skill_root, "scripts/reference_images.py")


def validate_capability_manifest(skill_root: Path) -> Tuple[bool, str]:
    """Return a controlled validation result for the Skill capability manifest."""

    root = Path(skill_root)
    try:
        manifest = load_capability_manifest(root)
        _validate_manifest(root, manifest)
    except CapabilityManifestError as error:
        return False, str(error)
    return True, "Capability manifest is valid!"
