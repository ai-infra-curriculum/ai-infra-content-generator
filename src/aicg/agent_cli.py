"""Local subscription CLI execution helpers for Claude and Codex."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

LIMIT_PATTERNS = (
    "usage limit",
    "rate limit",
    "limit reached",
    "quota exceeded",
    "session limit",
    "weekly limit",
    "5-hour",
    "5 hour",
    "try again",
    "too many requests",
)


@dataclass(frozen=True)
class AgentCommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    limit_reached: bool
    retry_after: str | None
    limit_scope: str | None
    limit_pattern_unmatched: bool = False


class AgentLimitReached(RuntimeError):
    def __init__(self, result: AgentCommandResult):
        self.result = result
        super().__init__(
            f"Agent subscription limit reached; retry after {result.retry_after or 'later'}."
        )


def run_agent_command(command: str, cwd: Path) -> AgentCommandResult:
    completed = subprocess.run(
        shlex.split(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    limit_scope = classify_limit_scope(output)
    limit_reached = completed.returncode != 0 and limit_scope is not None
    # When the command failed but no known limit-pattern matched the
    # output, surface that as a separate signal so operators can extend
    # LIMIT_PATTERNS when vendors change their error wording.
    limit_pattern_unmatched = completed.returncode != 0 and limit_scope is None
    return AgentCommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout[-4000:],
        stderr=completed.stderr[-4000:],
        limit_reached=limit_reached,
        retry_after=retry_after_for_scope(limit_scope) if limit_scope else None,
        limit_scope=limit_scope,
        limit_pattern_unmatched=limit_pattern_unmatched,
    )


def classify_limit_scope(output: str) -> str | None:
    text = output.lower()
    if not any(pattern in text for pattern in LIMIT_PATTERNS):
        return None
    if "weekly" in text:
        return "weekly"
    if "5-hour" in text or "5 hour" in text or "five hour" in text:
        return "five_hour"
    return "unknown"


def retry_after_for_scope(scope: str) -> str:
    now = datetime.now(timezone.utc)
    if scope == "weekly":
        retry_at = now + timedelta(days=7)
    elif scope == "five_hour":
        retry_at = now + timedelta(hours=5)
    else:
        retry_at = now + timedelta(hours=6)
    return retry_at.isoformat(timespec="seconds").replace("+00:00", "Z")


def retry_after_has_passed(value: str | None) -> bool:
    if not value:
        return True
    try:
        retry_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(timezone.utc) >= retry_at
