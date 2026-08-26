"""Shared loading and validation for provider API keys."""

import os
from pathlib import Path


class APIKeyError(ValueError):
    """Raised when a configured provider key is unsafe for an HTTP header."""


def load_api_key(environment_name: str, secret_path: Path) -> str:
    """Load an environment key first, then a newline-terminated secret file."""
    environment_value = os.environ.get(environment_name)
    if environment_value is not None and environment_value != "":
        return environment_value
    try:
        return secret_path.read_text(encoding="utf-8").rstrip("\r\n")
    except (OSError, UnicodeError):
        return ""


def validate_api_key(value: object) -> str:
    """Return a header-safe key without ever including it in an error."""
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise APIKeyError(
            "configured API key is invalid; use non-empty printable ASCII "
            "without whitespace"
        )
    return value
