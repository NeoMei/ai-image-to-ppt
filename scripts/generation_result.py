from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
import re
from typing import AbstractSet, Iterable, Optional


class GenerationStatus(str, Enum):
    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    AUTH_UNAVAILABLE = "auth_unavailable"
    RETRYABLE_EXHAUSTED = "retryable_exhausted"
    POLICY_REFUSED = "policy_refused"
    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    LOCAL_FAILURE = "local_failure"


FALLBACK_STATUSES = frozenset({
    GenerationStatus.UNAVAILABLE,
    GenerationStatus.AUTH_UNAVAILABLE,
    GenerationStatus.RETRYABLE_EXHAUSTED,
})
FATAL_STATUSES = frozenset(set(GenerationStatus) - FALLBACK_STATUSES - {
    GenerationStatus.SUCCESS,
})


@dataclass(frozen=True)
class GenerationResult:
    status: GenerationStatus
    provider: str
    channel: str
    output_path: Optional[str] = None
    safe_message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, GenerationStatus):
            raise ValueError("status must be a GenerationStatus")
        if self.provider not in {"openai", "gemini", "doubao"}:
            raise ValueError("provider must be openai, gemini, or doubao")
        if self.channel not in {"host", "api"}:
            raise ValueError("channel must be host or api")
        if self.status is GenerationStatus.SUCCESS:
            if not isinstance(self.output_path, str) or not os.path.isabs(self.output_path):
                raise ValueError("success requires an absolute output_path")
        elif self.output_path is not None:
            raise ValueError("only success may carry output_path")

    @property
    def ok(self) -> bool:
        return self.status is GenerationStatus.SUCCESS

    @property
    def can_fallback(self) -> bool:
        return self.status in FALLBACK_STATUSES

    def to_json(self) -> str:
        payload = asdict(self)
        payload["status"] = self.status.value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def classify_http_failure(
    status_code: int,
    error_code: Optional[str],
    policy_codes: AbstractSet[str],
    retryable_codes: AbstractSet[str],
) -> GenerationStatus:
    normalized = error_code.strip().lower() if isinstance(error_code, str) else ""
    if normalized in policy_codes:
        return GenerationStatus.POLICY_REFUSED
    if normalized in retryable_codes:
        return GenerationStatus.RETRYABLE_EXHAUSTED
    if status_code in (401, 403):
        return GenerationStatus.AUTH_UNAVAILABLE
    if status_code == 429 or status_code >= 500:
        return GenerationStatus.RETRYABLE_EXHAUSTED
    return GenerationStatus.INVALID_INPUT


def safe_message(message: object, secrets: Iterable[str] = ()) -> str:
    text = str(message)
    for secret in sorted((secret for secret in secrets if secret), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer[ ]+[A-Za-z0-9._~+/-]+", "Bearer [REDACTED]", text)
    return text[:300]
