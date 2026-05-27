"""Propagate shipped work into the target repo's changelog.

After a work item is generated + verified, the runner appends an entry
to ``VERSIONS.md`` so the changelog reflects what just shipped. This is
the safe, repo-agnostic half of "content-update propagation."
``CURRICULUM.md`` edits are deliberately *not* applied automatically —
their table shape varies per role tier and we don't want to silently
re-format authored docs. Instead the runner emits a suggested edit in
the report that an operator / agent can review.

The propagator is idempotent: if an entry for the same work id is
already present under the current month, the file is left alone.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import (
    ensure_state_dir,
    read_state,
    relative_path,
    state_path,
    utc_now,
    write_json,
)

PROPAGATE_REPORT = "propagate-report.json"
WORK_PLAN = "work-plan.json"
VERSIONS_FILE = "VERSIONS.md"


class PropagateError(RuntimeError):
    """Raised when propagation cannot run."""


@dataclass(frozen=True)
class WorkSummary:
    work_id: str
    module: str | None
    project: str | None
    work_type: str
    title: str


def propagate_repo(
    workspace: Path,
    repo_name: str,
    work_id: str | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Update ``VERSIONS.md`` for every verified work item in the plan."""
    from .inventory import WorkspaceInventory

    inventory = WorkspaceInventory(workspace)
    target = inventory.require(repo_name)
    repo_path = target.path

    try:
        plan = read_state(repo_path, WORK_PLAN)
    except FileNotFoundError as exc:
        raise PropagateError(
            f"No work plan at {state_path(repo_path, WORK_PLAN)}; "
            "run `aicg plan` first."
        ) from exc

    summaries: list[WorkSummary] = []
    for item in plan.get("work_items", []):
        if work_id is not None and item.get("id") != work_id:
            continue
        if item.get("status") not in {"verified", "generated"}:
            # Skip planned items that have not been generated yet.
            continue
        summaries.append(_summarise(item))

    if not summaries:
        return _emit_report(
            repo_path=repo_path,
            plan_repo=plan.get("repo", {}),
            updated=[],
            already_present=[],
            curriculum_suggestions=[],
            write_report=write_report,
            status="no_items",
        )

    versions_path = repo_path / VERSIONS_FILE
    existing = versions_path.read_text(encoding="utf-8") if versions_path.exists() else ""

    updated: list[dict[str, Any]] = []
    already_present: list[dict[str, Any]] = []
    new_content = existing
    today = _dt.date.today()

    for summary in summaries:
        if _entry_present(new_content, summary.work_id):
            already_present.append(_summary_dict(summary))
            continue
        new_content = _insert_changelog_entry(new_content, summary, today)
        updated.append(
            {
                **_summary_dict(summary),
                "date": today.isoformat(),
            }
        )

    curriculum_suggestions = [
        _curriculum_suggestion(summary) for summary in summaries
    ]

    if updated:
        if not new_content.endswith("\n"):
            new_content += "\n"
        versions_path.write_text(new_content, encoding="utf-8")

    return _emit_report(
        repo_path=repo_path,
        plan_repo=plan.get("repo", {}),
        updated=updated,
        already_present=already_present,
        curriculum_suggestions=curriculum_suggestions,
        write_report=write_report,
        status="updated" if updated else "noop",
        versions_path=versions_path,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _summarise(work_item: dict[str, Any]) -> WorkSummary:
    return WorkSummary(
        work_id=work_item.get("id", ""),
        module=work_item.get("module"),
        project=work_item.get("project"),
        work_type=work_item.get("type", ""),
        title=work_item.get("title", work_item.get("id", "")),
    )


def _summary_dict(summary: WorkSummary) -> dict[str, Any]:
    return {
        "work_id": summary.work_id,
        "module": summary.module,
        "project": summary.project,
        "type": summary.work_type,
        "title": summary.title,
    }


def _entry_present(content: str, work_id: str) -> bool:
    if not content or not work_id:
        return False
    return f"`{work_id}`" in content or work_id in content


def _insert_changelog_entry(
    content: str, summary: WorkSummary, today: _dt.date
) -> str:
    """Insert a row for ``summary`` under today's month heading."""
    month_label = today.strftime("%Y-%m")
    month_heading = f"## {month_label}"
    row = _format_row(summary, today)

    if not content:
        return _seed_versions_file(month_heading, row)

    if month_heading not in content:
        # Insert a fresh month block right after the title line.
        lines = content.splitlines()
        title_idx = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith("# ")
            ),
            None,
        )
        if title_idx is None:
            lines = [f"# Versions", ""] + lines
            title_idx = 0
        insertion = [
            "",
            month_heading,
            "",
            "| Date | Work ID | Scope | Title |",
            "|---|---|---|---|",
            row,
            "",
        ]
        lines[title_idx + 1 : title_idx + 1] = insertion
        return "\n".join(lines) + "\n"

    # Append the new row directly under the existing month heading's
    # table. We find the table by walking until we encounter a blank
    # line that follows table rows.
    lines = content.splitlines()
    out_lines: list[str] = []
    inserted = False
    i = 0
    while i < len(lines):
        out_lines.append(lines[i])
        if not inserted and lines[i].strip() == month_heading.strip():
            # Skip until we find the start of the table (header row).
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("|"):
                out_lines.append(lines[i])
                i += 1
            # Now we're at the header row. Append header + separator
            # rows, then any existing data rows, then our new row.
            if i < len(lines) and lines[i].lstrip().startswith("|"):
                out_lines.append(lines[i])  # header
                i += 1
            if i < len(lines) and lines[i].lstrip().startswith("|"):
                out_lines.append(lines[i])  # separator
                i += 1
            # Append existing data rows verbatim.
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                out_lines.append(lines[i])
                i += 1
            out_lines.append(row)
            inserted = True
            continue
        i += 1
    if not inserted:
        out_lines.append(row)
    return "\n".join(out_lines) + "\n"


def _seed_versions_file(month_heading: str, row: str) -> str:
    return (
        "# Versions\n\n"
        f"{month_heading}\n\n"
        "| Date | Work ID | Scope | Title |\n"
        "|---|---|---|---|\n"
        f"{row}\n"
    )


def _format_row(summary: WorkSummary, today: _dt.date) -> str:
    scope = summary.module or summary.project or summary.work_type or "—"
    title = summary.title.replace("|", "\\|")
    return f"| {today.isoformat()} | `{summary.work_id}` | `{scope}` | {title} |"


def _curriculum_suggestion(summary: WorkSummary) -> dict[str, Any]:
    scope = summary.module or summary.project
    if not scope:
        return {
            "work_id": summary.work_id,
            "suggestion": "Unknown scope; review manually.",
        }
    return {
        "work_id": summary.work_id,
        "scope": scope,
        "suggestion": (
            f"Update `CURRICULUM.md` so the row for `{scope}` reflects the "
            "newly-landed content. Schema varies per repo; this is a "
            "suggested edit, not an automated change."
        ),
    }


def _emit_report(
    repo_path: Path,
    plan_repo: dict[str, Any],
    updated: list[dict[str, Any]],
    already_present: list[dict[str, Any]],
    curriculum_suggestions: list[dict[str, Any]],
    write_report: bool,
    status: str,
    versions_path: Path | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "repo": plan_repo,
        "status": status,
        "versions_path": (
            relative_path(versions_path, repo_path)
            if versions_path is not None
            else VERSIONS_FILE
        ),
        "updated": updated,
        "already_present": already_present,
        "curriculum_suggestions": curriculum_suggestions,
    }
    if write_report:
        ensure_state_dir(repo_path)
        write_json(state_path(repo_path, PROPAGATE_REPORT), report)
    return report
