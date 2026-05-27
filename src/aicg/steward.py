"""PR / issue / discussion stewardship for autonomous org operation.

The steward closes the audit -> plan -> generate -> verify -> PR loop by
performing the final merge step under explicit guardrails.

Lifecycle per PR:

    open
      |
      v
    ci_pending   (statusCheckRollup is PENDING / IN_PROGRESS)
      |
      v
    ci_passed    (rollup SUCCESS)            ci_failed (FAILURE)  -> end
      |
      v
    guardrails_ok                            guardrails_blocked   -> end
      |
      v
    merge_requested
      |
      v
    merged

Every transition is recorded in ``steward-report.json`` so the next
invocation can resume mid-loop. The merge call uses
``gh pr merge --auto --squash --delete-branch`` so GitHub does the
final flip once required status checks are green, which means the
runner does not have to hold the lock until merge actually completes.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .audit import PlaceholderCache, scan_placeholders
from .guardrails import (
    RESTRICTED_AUTO_MERGE_PATTERNS,
    GuardrailDecision,
    matches_any,
)
from .org_config import OrgManifest, state_dir_for_manifest
from .state import utc_now, write_json

STEWARD_REPORT = "steward-report.json"


class StewardError(RuntimeError):
    """Raised when the steward cannot proceed."""


def steward_run(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path | None = None,
    apply: bool = False,
    ci_timeout_seconds: int = 600,
    ci_poll_seconds: int = 30,
    merge_method: str = "squash",
) -> dict[str, Any]:
    """Steward every open PR in the manifest's repos.

    When ``apply`` is False the steward operates in dry-run mode: it
    reads PR + CI state and prints the decision it would have made, but
    does not call ``gh pr merge``.
    """
    state_dir = state_dir_for_manifest(manifest, state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    repo_reports: list[dict[str, Any]] = []
    for repo in manifest.repo_names:
        repo_path = workspace / repo
        repo_reports.append(
            steward_repo(
                repo=repo,
                repo_path=repo_path,
                apply=apply,
                ci_timeout_seconds=ci_timeout_seconds,
                ci_poll_seconds=ci_poll_seconds,
                merge_method=merge_method,
            )
        )

    report = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "operation": "steward",
        "status": "applied" if apply else "dry_run",
        "ci_timeout_seconds": ci_timeout_seconds,
        "ci_poll_seconds": ci_poll_seconds,
        "merge_method": merge_method,
        "repos": repo_reports,
    }
    write_json(state_dir / STEWARD_REPORT, report)
    return report


def steward_repo(
    repo: str,
    repo_path: Path,
    apply: bool,
    ci_timeout_seconds: int,
    ci_poll_seconds: int,
    merge_method: str,
) -> dict[str, Any]:
    if not repo_path.exists():
        return {
            "repo": repo,
            "path": str(repo_path),
            "present": False,
            "prs": [],
        }

    prs = list_open_prs(repo_path)
    pr_outcomes: list[dict[str, Any]] = []
    for pr in prs:
        pr_outcomes.append(
            steward_pr(
                repo=repo,
                repo_path=repo_path,
                pr=pr,
                apply=apply,
                ci_timeout_seconds=ci_timeout_seconds,
                ci_poll_seconds=ci_poll_seconds,
                merge_method=merge_method,
            )
        )

    return {
        "repo": repo,
        "path": str(repo_path),
        "present": True,
        "pr_count": len(pr_outcomes),
        "merged_count": sum(1 for item in pr_outcomes if item["state"] == "merged"),
        "blocked_count": sum(
            1 for item in pr_outcomes if item["state"] == "guardrails_blocked"
        ),
        "ci_failed_count": sum(
            1 for item in pr_outcomes if item["state"] == "ci_failed"
        ),
        "prs": pr_outcomes,
    }


def steward_pr(
    repo: str,
    repo_path: Path,
    pr: dict[str, Any],
    apply: bool,
    ci_timeout_seconds: int,
    ci_poll_seconds: int,
    merge_method: str,
) -> dict[str, Any]:
    number = pr.get("number")
    history: list[dict[str, Any]] = [
        {"timestamp": utc_now(), "state": "open"}
    ]

    ci_result = wait_for_ci(
        repo_path=repo_path,
        pr_number=number,
        timeout_seconds=ci_timeout_seconds,
        poll_seconds=ci_poll_seconds,
    )
    history.append(
        {
            "timestamp": utc_now(),
            "state": "ci_pending",
            "elapsed_seconds": ci_result["elapsed_seconds"],
        }
    )
    if ci_result["status"] != "success":
        terminal = "ci_failed" if ci_result["status"] == "failure" else "ci_timeout"
        history.append({"timestamp": utc_now(), "state": terminal})
        return {
            "repo": repo,
            "pr_number": number,
            "title": pr.get("title"),
            "state": terminal,
            "head_ref": pr.get("headRefName"),
            "ci": ci_result,
            "history": history,
        }
    history.append({"timestamp": utc_now(), "state": "ci_passed"})

    # Pull the full PR view (with file list + labels + mergeable state)
    # before evaluating guardrails. ``pr_view`` was already called by
    # ``wait_for_ci`` but its return is not propagated, so we ask
    # again. Cheap enough — this is the only place guardrails run.
    full_pr = pr_view(repo_path, number) or pr
    # Merge the list-summary fields into the full view so callers see
    # the union (list-only callers got ``title``, view-only callers
    # got ``files`` etc.).
    merged_pr = {**pr, **full_pr}
    decision = evaluate_pr_guardrails(repo_path=repo_path, pr=merged_pr)
    if not decision.allowed:
        history.append(
            {
                "timestamp": utc_now(),
                "state": "guardrails_blocked",
                "blockers": list(decision.blockers),
            }
        )
        return {
            "repo": repo,
            "pr_number": number,
            "title": pr.get("title"),
            "state": "guardrails_blocked",
            "head_ref": pr.get("headRefName"),
            "blockers": list(decision.blockers),
            "warnings": list(decision.warnings),
            "history": history,
        }
    history.append({"timestamp": utc_now(), "state": "guardrails_ok"})

    if not apply:
        history.append({"timestamp": utc_now(), "state": "dry_run_skipped_merge"})
        return {
            "repo": repo,
            "pr_number": number,
            "title": pr.get("title"),
            "state": "would_merge",
            "head_ref": pr.get("headRefName"),
            "history": history,
        }

    merge_result = enable_auto_merge(repo_path, number, method=merge_method)
    if merge_result["returncode"] != 0:
        history.append(
            {
                "timestamp": utc_now(),
                "state": "merge_failed",
                "stderr": merge_result["stderr"],
            }
        )
        return {
            "repo": repo,
            "pr_number": number,
            "title": pr.get("title"),
            "state": "merge_failed",
            "head_ref": pr.get("headRefName"),
            "merge_result": merge_result,
            "history": history,
        }
    history.append({"timestamp": utc_now(), "state": "merge_requested"})
    final_state = pr_state(repo_path, number)
    if final_state == "MERGED":
        history.append({"timestamp": utc_now(), "state": "merged"})
        return {
            "repo": repo,
            "pr_number": number,
            "title": pr.get("title"),
            "state": "merged",
            "head_ref": pr.get("headRefName"),
            "history": history,
        }
    return {
        "repo": repo,
        "pr_number": number,
        "title": pr.get("title"),
        "state": "merge_pending",
        "head_ref": pr.get("headRefName"),
        "history": history,
    }


# ---------------------------------------------------------------------------
# gh wrappers (one wrapper per command so they're easy to mock in tests)
# ---------------------------------------------------------------------------


def list_open_prs(repo_path: Path) -> list[dict[str, Any]]:
    """Return open PRs for ``repo_path`` via ``gh pr list``."""
    completed = _run_gh(
        repo_path,
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,headRefName,headRefOid,author,labels,isDraft",
            "--limit",
            "100",
        ],
    )
    if completed.returncode != 0:
        return []
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []


def pr_view(repo_path: Path, pr_number: int) -> dict[str, Any]:
    completed = _run_gh(
        repo_path,
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,headRefName,mergeable,mergeStateStatus,state,labels,statusCheckRollup,changedFiles,files",
        ],
    )
    if completed.returncode != 0:
        return {}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}


def pr_state(repo_path: Path, pr_number: int) -> str:
    """Return ``OPEN``, ``MERGED``, or ``CLOSED`` for a PR."""
    completed = _run_gh(
        repo_path,
        ["gh", "pr", "view", str(pr_number), "--json", "state"],
    )
    if completed.returncode != 0:
        return "UNKNOWN"
    try:
        return json.loads(completed.stdout).get("state", "UNKNOWN")
    except json.JSONDecodeError:
        return "UNKNOWN"


def enable_auto_merge(
    repo_path: Path, pr_number: int, method: str = "squash"
) -> dict[str, Any]:
    method = method.lower()
    flag = {"squash": "--squash", "merge": "--merge", "rebase": "--rebase"}.get(
        method, "--squash"
    )
    completed = _run_gh(
        repo_path,
        ["gh", "pr", "merge", str(pr_number), "--auto", flag, "--delete-branch"],
    )
    return {
        "command": " ".join(["gh", "pr", "merge", str(pr_number), "--auto", flag, "--delete-branch"]),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def _run_gh(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# CI rollup
# ---------------------------------------------------------------------------


_CI_TERMINAL_SUCCESS = {"SUCCESS"}
_CI_TERMINAL_FAILURE = {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}


def wait_for_ci(
    repo_path: Path,
    pr_number: int,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    """Poll ``gh pr view`` until the status-check rollup is terminal."""
    deadline = time.monotonic() + timeout_seconds
    last_rollup: list[dict[str, Any]] = []
    while True:
        view = pr_view(repo_path, pr_number)
        last_rollup = view.get("statusCheckRollup", []) or []
        status = classify_rollup(last_rollup)
        if status == "success":
            return {
                "status": "success",
                "rollup": last_rollup,
                "elapsed_seconds": int(timeout_seconds - max(deadline - time.monotonic(), 0)),
            }
        if status == "failure":
            return {
                "status": "failure",
                "rollup": last_rollup,
                "elapsed_seconds": int(timeout_seconds - max(deadline - time.monotonic(), 0)),
            }
        if time.monotonic() >= deadline:
            return {
                "status": "timeout",
                "rollup": last_rollup,
                "elapsed_seconds": timeout_seconds,
            }
        time.sleep(poll_seconds)


_PENDING_STATES = {"IN_PROGRESS", "QUEUED", "PENDING", "WAITING", "EXPECTED"}


def classify_rollup(rollup: list[dict[str, Any]]) -> str:
    """Reduce GitHub's per-check rollup to a single status.

    The runner treats ``SUCCESS`` and ``SKIPPED`` as passes. Any
    ``IN_PROGRESS`` / ``QUEUED`` / ``PENDING`` keeps the rollup
    pending. Any ``FAILURE``/``ERROR``/etc. classifies the whole rollup
    as failure (one bad apple).
    """
    if not rollup:
        return "pending"
    has_pending = False
    for entry in rollup:
        # GitHub mixes two shapes: check runs (conclusion + status) and
        # commit statuses (state).
        status = (entry.get("status") or "").upper()
        conclusion = (entry.get("conclusion") or "").upper()
        state = (entry.get("state") or "").upper()
        # Pending signals can show up on either field.
        if status in _PENDING_STATES or state in _PENDING_STATES:
            has_pending = True
            continue
        terminal = conclusion or state
        if not terminal:
            has_pending = True
            continue
        if terminal in _CI_TERMINAL_FAILURE:
            return "failure"
        if terminal in {"NEUTRAL", "STALE", "SKIPPED"}:
            continue
        if terminal not in _CI_TERMINAL_SUCCESS:
            # Unknown terminal — treat as failure rather than silently
            # passing.
            return "failure"
    return "pending" if has_pending else "success"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def evaluate_pr_guardrails(
    repo_path: Path,
    pr: dict[str, Any],
) -> GuardrailDecision:
    """Re-check guardrails against the merged state of a PR.

    This duplicates ``guardrails.evaluate_guardrails`` but operates on
    the *PR view* (changed file list comes from ``gh``) so the steward
    does not need a local checkout of the head ref.
    """
    blockers: list[str] = []
    warnings: list[str] = []

    if pr.get("isDraft"):
        blockers.append("PR is in draft state.")

    label_names = [
        (item.get("name") or "").lower()
        for item in (pr.get("labels") or [])
    ]
    if "do-not-merge" in label_names or "wip" in label_names:
        blockers.append("PR carries a do-not-merge / wip label.")

    head_ref = pr.get("headRefName") or ""
    if head_ref in {"main", "master"}:
        blockers.append("PR head ref is main/master.")

    changed = [
        item.get("path")
        for item in (pr.get("files") or [])
        if item.get("path")
    ]
    restricted = [
        path for path in changed if matches_any(path, RESTRICTED_AUTO_MERGE_PATTERNS)
    ]
    if restricted:
        blockers.append(
            "Restricted files in PR: " + ", ".join(sorted(restricted))
        )

    # Marker freedom: scan only the changed files in the local working
    # tree. ``gh pr checkout`` would be more accurate but for the
    # default aicg branch shape the head ref is local already.
    marker_findings = scan_placeholders(repo_path, cache=PlaceholderCache(repo_path))
    blocker_markers = [
        item
        for item in marker_findings
        if item.get("type") in {"needs_research", "manual_review"}
        and (not changed or item.get("path") in changed)
    ]
    if blocker_markers:
        sample = ", ".join(item.get("path", "?") for item in blocker_markers[:3])
        blockers.append(f"Marker remains in changed file(s): {sample}")

    mergeable = (pr.get("mergeable") or "").upper()
    if mergeable == "CONFLICTING":
        blockers.append("PR has merge conflicts.")
    if mergeable not in {"MERGEABLE", "UNKNOWN", "CONFLICTING", ""}:
        warnings.append(f"Unexpected mergeable state: {mergeable}")

    return GuardrailDecision(
        allowed=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
