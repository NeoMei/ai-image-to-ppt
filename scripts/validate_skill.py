#!/usr/bin/env python3
"""Validate this repository's Agent Skill package without machine-specific paths."""

import argparse
import re
from pathlib import Path
from typing import Tuple

import yaml

from capability_manifest import validate_capability_manifest

MAX_SKILL_NAME_LENGTH = 64
ALLOWED_PROPERTIES = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "metadata",
}


def validate_skill(skill_path: str) -> Tuple[bool, str]:
    skill_file = Path(skill_path) / "SKILL.md"
    if not skill_file.is_file():
        return False, "SKILL.md not found"
    try:
        content = skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return False, f"Cannot read SKILL.md: {error}"
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if match is None:
        return False, "Invalid or missing YAML frontmatter"
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        return False, f"Invalid YAML in frontmatter: {error}"
    if not isinstance(frontmatter, dict):
        return False, "Frontmatter must be a YAML dictionary"
    unexpected = set(frontmatter) - ALLOWED_PROPERTIES
    if unexpected:
        return False, "Unexpected frontmatter keys: " + ", ".join(sorted(unexpected))

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip():
        return False, "Missing or invalid 'name' in frontmatter"
    name = name.strip()
    if (
        len(name) > MAX_SKILL_NAME_LENGTH
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
    ):
        return False, "Skill name must be hyphen-case and at most 64 characters"
    if not isinstance(description, str) or not description.strip():
        return False, "Missing or invalid 'description' in frontmatter"
    description = description.strip()
    if len(description) > 1024 or "<" in description or ">" in description:
        return False, "Description must be at most 1024 characters without angle brackets"
    if description.startswith("[TODO:"):
        return False, "Description contains an unfinished TODO placeholder"

    body = content[match.end() :]
    fence_marker = None
    fence_length = 0
    for line in body.splitlines():
        fence = re.match(
            r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?(`{3,}|~{3,})(.*)$",
            line,
        )
        if fence:
            marker = fence.group(1)
            if fence_marker is None:
                fence_marker, fence_length = marker[0], len(marker)
            elif (
                marker[0] == fence_marker
                and len(marker) >= fence_length
                and not fence.group(2).strip()
            ):
                fence_marker, fence_length = None, 0
            continue
        if fence_marker is None and re.fullmatch(
            r"[ ]{0,3}\[TODO:[^\n]*\][ \t]*", line
        ):
            return False, "Skill instructions contain an unfinished TODO placeholder"
    manifest_valid, manifest_message = validate_capability_manifest(Path(skill_path))
    if not manifest_valid:
        return False, f"Capability manifest invalid: {manifest_message}"
    return True, "Skill is valid!"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill_directory", nargs="?", default=".")
    args = parser.parse_args()
    valid, message = validate_skill(args.skill_directory)
    print(message)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
