"""Manifest-driven org automation operations."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent_cli import AgentLimitReached, retry_after_has_passed
from .audit import audit_repo
from .generator import GenerationNotConfigured, generate_from_plan
from .org_config import OrgManifest, RoleConfig, state_dir_for_manifest
from .planner import plan_from_audit
from .state import read_state, utc_now, write_json

ORG_QUEUE = "work-queue.json"
ORG_RESEARCH_PLAN = "job-research-plan.json"
ORG_STEWARD_REPORT = "steward-report.json"


class OrgRunnerError(RuntimeError):
    """Raised when an org-level operation cannot proceed."""


def sync_repositories(
    manifest: OrgManifest,
    workspace: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    actions: list[dict[str, Any]] = []
    if not dry_run:
        workspace.mkdir(parents=True, exist_ok=True)

    for repo in manifest.repo_names:
        repo_path = workspace / repo
        remote = manifest.remote_for(repo)
        if repo_path.exists():
            dirty = is_git_dirty(repo_path)
            command = ["git", "-C", str(repo_path), "pull", "--ff-only"]
            status = "dirty_skip" if dirty else "planned" if dry_run else "pulled"
            action = repo_action(repo, repo_path, remote, command, status)
            if dirty:
                action["warning"] = "Repository has local changes; skipping pull."
            elif not dry_run:
                action.update(run_command(command))
        else:
            command = ["git", "clone", remote, str(repo_path)]
            action = repo_action(repo, repo_path, remote, command, "planned" if dry_run else "cloned")
            if not dry_run:
                action.update(run_command(command))
        actions.append(action)

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "sync",
        "dry_run": dry_run,
        "workspace": str(workspace),
        "actions": actions,
    }


def plan_monthly_release(
    manifest: OrgManifest,
    workspace: Path,
    today: date | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    tag = release_tag(manifest, today or date.today())
    actions: list[dict[str, Any]] = []
    for repo in manifest.release_repo_names:
        repo_path = workspace / repo
        if not repo_path.exists():
            actions.append(
                {
                    "repo": repo,
                    "path": str(repo_path),
                    "status": "missing",
                    "message": "Repository is not present in workspace.",
                }
            )
            continue
        if is_git_dirty(repo_path):
            actions.append(
                {
                    "repo": repo,
                    "path": str(repo_path),
                    "status": "blocked",
                    "message": "Repository has local changes.",
                }
            )
            continue
        if git_tag_exists(repo_path, tag):
            actions.append({"repo": repo, "path": str(repo_path), "status": "exists", "tag": tag})
            continue
        tag_command = ["git", "-C", str(repo_path), "tag", "-a", tag, "-m", f"{tag} curriculum release"]
        push_command = ["git", "-C", str(repo_path), "push", "origin", tag]
        action = {
            "repo": repo,
            "path": str(repo_path),
            "status": "planned" if not apply else "tagged",
            "tag": tag,
            "commands": [shell_join(tag_command), shell_join(push_command)],
        }
        if apply:
            tag_result = run_command(tag_command)
            push_result = run_command(push_command)
            action["results"] = [tag_result, push_result]
            if tag_result["returncode"] != 0 or push_result["returncode"] != 0:
                action["status"] = "failed"
        actions.append(action)

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "monthly_release",
        "apply": apply,
        "tag": tag,
        "actions": actions,
        "note": "Tag pushes trigger per-repo release packaging workflows when installed.",
    }


def release_tag(manifest: OrgManifest, today: date) -> str:
    return today.strftime(manifest.release.get("tag_format", "v%Y.%m"))


def generate_research_packets(
    manifest: OrgManifest,
    workspace: Path,
    month: str | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    month = month or date.today().strftime("%Y-%m")
    state_dir = state_dir_for_manifest(manifest, state_dir)
    prompt_dir = state_dir / "research" / month
    prompt_dir.mkdir(parents=True, exist_ok=True)

    packets: list[dict[str, Any]] = []
    for role in sorted(manifest.roles, key=lambda item: item.level):
        learning_path = workspace / role.learning_repo
        prompt_path = prompt_dir / f"{role.id}.md"
        prompt_path.write_text(build_research_prompt(manifest, role, month), encoding="utf-8")
        packets.append(
            {
                "role": role.id,
                "title": role.title,
                "learning_repo": role.learning_repo,
                "learning_path": str(learning_path),
                "prompt_path": str(prompt_path),
                "job_requirements_markdown": str(
                    learning_path
                    / manifest.job_requirements.get("markdown_file", "JOB_REQUIREMENTS.md")
                ),
                "job_requirements_json": str(
                    learning_path
                    / manifest.job_requirements.get("structured_file", ".aicg/job-requirements.json")
                ),
                "status": "prompt_ready",
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "monthly_research",
        "month": month,
        "minimum_postings_per_role": manifest.research.get("minimum_postings_per_role", 25),
        "source_window_days": manifest.research.get("source_window_days", 45),
        "ownership_strategy": manifest.job_requirements.get(
            "ownership_strategy", "lowest_level_role"
        ),
        "packets": packets,
    }
    write_json(state_dir / ORG_RESEARCH_PLAN, report)
    return report


def run_org_audit(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    state_dir = state_dir_for_manifest(manifest, state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    queue_items: list[dict[str, Any]] = []
    repo_reports: list[dict[str, Any]] = []

    for repo in manifest.solution_repo_names:
        repo_path = workspace / repo
        if not repo_path.exists():
            repo_reports.append({"repo": repo, "status": "missing", "path": str(repo_path)})
            continue
        audit = audit_repo(workspace, repo)
        plan = plan_from_audit(audit, repo_path=repo_path)
        repo_reports.append(
            {
                "repo": repo,
                "status": audit["summary"]["status"],
                "errors": audit["summary"]["error_count"],
                "warnings": audit["summary"]["warning_count"],
                "work_items": plan["work_item_count"],
            }
        )
        for item in plan.get("work_items", []):
            queue_items.append(
                {
                    "id": f"{repo}:{item['id']}",
                    "repo": repo,
                    "work_id": item["id"],
                    "module": item.get("module"),
                    "type": item["type"],
                    "title": item["title"],
                    "status": "ready",
                    "priority": queue_priority(manifest, repo, item),
                    "created_at": utc_now(),
                }
            )

        # Backlog items (exercise_depth_followup etc.) become medium-
        # severity work so they sit below structural gaps but get
        # worked eventually rather than living only in per-repo state.
        for item in plan.get("backlog_items", []):
            biased_item = {**item, "severity": "medium"}
            queue_items.append(
                {
                    "id": f"{repo}:{item['id']}",
                    "repo": repo,
                    "work_id": item["id"],
                    "module": item.get("module"),
                    "type": item["type"],
                    "severity": "medium",
                    "title": item["title"],
                    "status": "ready",
                    "priority": queue_priority(manifest, repo, biased_item),
                    "created_at": utc_now(),
                }
            )

        # Splice in freshness work items if the most recent audit-links /
        # audit-versions / review runs left reports behind. These are
        # additive — structural gaps remain in the queue too.
        for refresh_item in _collect_freshness_items(repo_path):
            queue_items.append(
                {
                    "id": f"{repo}:{refresh_item['id']}",
                    "repo": repo,
                    "work_id": refresh_item["id"],
                    "type": refresh_item["type"],
                    "severity": refresh_item.get("severity", "low"),
                    "title": refresh_item.get("title", refresh_item["id"]),
                    "path": refresh_item.get("path"),
                    "status": "ready",
                    "priority": queue_priority(manifest, repo, refresh_item),
                    "created_at": utc_now(),
                }
            )

    queue_items.sort(key=lambda item: (item["priority"], item["repo"], item["work_id"]))
    queue = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "weekly_audit",
        "workspace": str(workspace.resolve()),
        "work_item_count": len(queue_items),
        "repo_reports": repo_reports,
        "work_items": queue_items,
    }
    write_json(state_dir / ORG_QUEUE, queue)
    return queue


def run_daily_remediation(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    state_dir = state_dir_for_manifest(manifest, state_dir)
    queue_path = state_dir / ORG_QUEUE
    if not queue_path.exists():
        run_org_audit(manifest, workspace, state_dir=state_dir)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    refresh_deferred_items(queue)
    ready = [item for item in queue.get("work_items", []) if item.get("status") == "ready"]
    if not ready:
        return generate_supplemental_packet(manifest, workspace, state_dir)

    item = ready[0]
    repo_path = workspace / item["repo"]
    plan = read_state(repo_path, "work-plan.json")
    result = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "daily_remediate",
        "selected": item,
        "status": "started",
    }
    try:
        run_state = generate_from_plan(
            repo_path,
            plan,
            module=item.get("module"),
            work_id=item["work_id"],
            command_override=content_generation_command(manifest),
        )
        item["status"] = "generated"
        item["updated_at"] = utc_now()
        item["retry_count"] = 0
        result["status"] = "generated"
        result["run_state"] = run_state
        from .judge import JudgeConfig
        from .verify import verify_repo

        judge_config = JudgeConfig.from_manifest(manifest)
        verify_report = verify_repo(
            workspace,
            item["repo"],
            work_id=item["work_id"],
            judge_config=judge_config if judge_config.enabled else None,
        )
        result["verify"] = {
            "status": verify_report["status"],
            "work_item_count": verify_report["work_item_count"],
        }
        if verify_report["status"] == "verified":
            item["status"] = "verified"
            from .propagate import propagate_repo

            propagate_report = propagate_repo(
                workspace, item["repo"], work_id=item["work_id"]
            )
            result["propagate"] = {
                "status": propagate_report["status"],
                "updated_count": len(propagate_report.get("updated", [])),
            }
            # Close the loop: commit changes and open a PR so the
            # steward can pick it up. PR failures are recorded but do
            # not flip the item back to failed — the on-disk content
            # is good, just unmerged.
            pr_outcome = _open_work_item_pr(repo_path, plan, item)
            result["pr"] = pr_outcome
            if pr_outcome.get("status") == "opened":
                item["pr_url"] = pr_outcome.get("pr_url")
                item["pr_branch"] = pr_outcome.get("branch")
        else:
            item["status"] = "verification_failed"
        item["verified_at"] = utc_now()
    except GenerationNotConfigured as exc:
        item["status"] = "prompt_ready"
        item["updated_at"] = utc_now()
        result["status"] = "prompt_ready"
        result["prompt_path"] = str(exc.prompt_path)
        result["output_dir"] = str(exc.output_dir)
    except AgentLimitReached as exc:
        item["status"] = "deferred"
        item["updated_at"] = utc_now()
        item["defer_reason"] = "agent_subscription_limit"
        item["limit_scope"] = exc.result.limit_scope
        item["retry_after"] = exc.result.retry_after
        result["status"] = "deferred"
        result["defer_reason"] = "agent_subscription_limit"
        result["limit_scope"] = exc.result.limit_scope
        result["retry_after"] = exc.result.retry_after
    except RuntimeError as exc:
        # generate_from_plan raises RuntimeError when the agent
        # returned a non-zero exit code and no known limit pattern
        # matched. Treat that as a transient opaque failure and
        # schedule a retry; surface ``failed_permanently`` after
        # ``max_retries`` attempts.
        retry_config = _opaque_retry_config(manifest)
        retry_count = int(item.get("retry_count", 0)) + 1
        if retry_count >= retry_config["max_retries"]:
            item["status"] = "failed_permanently"
            item["updated_at"] = utc_now()
            item["retry_count"] = retry_count
            item["last_failure"] = str(exc)
            result["status"] = "failed_permanently"
            result["retry_count"] = retry_count
            result["error"] = str(exc)
        else:
            item["status"] = "deferred"
            item["updated_at"] = utc_now()
            item["retry_count"] = retry_count
            item["defer_reason"] = "opaque_generator_failure"
            item["retry_after"] = _retry_after_in_minutes(
                retry_config["retry_delay_minutes"]
            )
            item["last_failure"] = str(exc)
            result["status"] = "deferred"
            result["defer_reason"] = "opaque_generator_failure"
            result["retry_count"] = retry_count
            result["retry_after"] = item["retry_after"]
            result["error"] = str(exc)
    write_json(queue_path, queue)
    write_json(state_dir / "daily-run-state.json", result)
    return result


def _opaque_retry_config(manifest: OrgManifest) -> dict[str, Any]:
    automation = manifest.automation or {}
    retry = automation.get("opaque_retry", {}) if isinstance(automation, dict) else {}
    max_retries = int(retry.get("max_retries", 3))
    delay_minutes = int(retry.get("retry_delay_minutes", 15))
    return {"max_retries": max_retries, "retry_delay_minutes": delay_minutes}


def _retry_after_in_minutes(minutes: int) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def refresh_deferred_items(queue: dict[str, Any]) -> None:
    for item in queue.get("work_items", []):
        if item.get("status") != "deferred":
            continue
        if retry_after_has_passed(item.get("retry_after")):
            item["status"] = "ready"
            item["updated_at"] = utc_now()
            item.pop("defer_reason", None)


def steward_report(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path | None = None,
    apply: bool = False,
    ci_timeout_seconds: int = 600,
    ci_poll_seconds: int = 30,
    merge_method: str = "squash",
) -> dict[str, Any]:
    """Run the PR steward (auto-merger or dry-run)."""
    from .steward import steward_run

    return steward_run(
        manifest,
        workspace,
        state_dir=state_dir,
        apply=apply,
        ci_timeout_seconds=ci_timeout_seconds,
        ci_poll_seconds=ci_poll_seconds,
        merge_method=merge_method,
    )


def generate_supplemental_packet(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path,
) -> dict[str, Any]:
    month = date.today().strftime("%Y-%m")
    prompt_dir = state_dir / "supplemental" / month
    prompt_dir.mkdir(parents=True, exist_ok=True)
    packets = []
    for role in manifest.roles:
        prompt_path = prompt_dir / f"{role.id}.md"
        supplemental_dir = manifest.job_requirements.get("supplemental_dir", "supplemental")
        prompt_path.write_text(
            build_supplemental_prompt(manifest, role, month, supplemental_dir),
            encoding="utf-8",
        )
        packets.append(
            {
                "role": role.id,
                "learning_repo": role.learning_repo,
                "prompt_path": str(prompt_path),
                "target_dir": str(workspace / role.learning_repo / supplemental_dir),
            }
        )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "supplemental",
        "status": "prompt_ready",
        "packets": packets,
    }
    write_json(state_dir / "supplemental-plan.json", report)
    return report


def build_research_prompt(manifest: OrgManifest, role: RoleConfig, month: str) -> str:
    hierarchy = "\n".join(
        f"- level {item.level}: {item.title} (`{item.learning_repo}`)" for item in manifest.roles
    )
    return (
        f"# Job Requirements Research Packet - {role.title} - {month}\n\n"
        f"Preferred content agent: {content_generation_label(manifest)}.\n\n"
        "## Goal\n\n"
        f"Research current job postings for `{role.title}` and update "
        f"`{role.learning_repo}` requirements without duplicating lower-level coverage.\n\n"
        "## Research Requirements\n\n"
        f"- Analyze at least {manifest.research.get('minimum_postings_per_role', 25)} relevant postings.\n"
        f"- Prefer postings from the last {manifest.research.get('source_window_days', 45)} days.\n"
        "- Capture employer, title, URL, date observed, location, and requirements.\n"
        "- Store raw normalized findings in `.aicg/job-requirements.json`.\n"
        "- Update `JOB_REQUIREMENTS.md` with requirement coverage links.\n\n"
        "## Ownership Rule\n\n"
        "If a requirement belongs to multiple roles, assign the primary coverage to the "
        "lowest-level role where it is genuinely required. Higher-level roles should link "
        "to that owner unless they need different depth, architecture context, or leadership context.\n\n"
        "## Role Hierarchy\n\n"
        f"{hierarchy}\n\n"
        "## Output Contract\n\n"
        "Write exactly these files inside the learning repo (paths are relative to the repo root):\n\n"
        "1. `JOB_REQUIREMENTS.md` — grouped requirements, posting evidence, curriculum links, and external resources for out-of-scope items.\n"
        "2. `.aicg/job-requirements.json` — machine-readable requirements, evidence, owner role, coverage path, and status.\n"
        "3. `.aicg/curriculum-plan-delta.json` — *proposed* additions, strictly additive (never delete existing items). The runner will apply per-run caps and reject items lacking evidence; nothing is auto-merged. Use this schema:\n\n"
        "```json\n"
        "{\n"
        '  "schema_version": 1,\n'
        '  "role_id": "<role>",\n'
        '  "month": "<YYYY-MM>",\n'
        '  "rationale": "One paragraph: what specific change in the job market over the last 90 days drives every item below. Vague rationales get the proposal rejected by humans on review.",\n'
        '  "modules": [\n'
        '    {\n'
        '      "id": "mod-XXX-<slug>",\n'
        '      "title": "...",\n'
        '      "description": "...",\n'
        '      "exercises": [],\n'
        '      "rationale": "Why THIS module, why now, and which exact requirement(s) from JOB_REQUIREMENTS.md it covers.",\n'
        '      "evidence": [\n'
        '        {"employer": "...", "title": "...", "url": "...", "date_observed": "YYYY-MM-DD", "quote": "..."}\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "exercises": [\n'
        '    {\n'
        '      "module_id": "mod-XXX-<existing-module>",\n'
        '      "exercise": {"slug": "exercise-NN-<slug>", "title": "...", "summary": "..."},\n'
        '      "evidence": [{"employer": "...", "title": "...", "url": "...", "date_observed": "YYYY-MM-DD", "quote": "..."}]\n'
        '    }\n'
        '  ],\n'
        '  "projects": [\n'
        '    {"id": "project-NN-<slug>", "title": "...", "description": "...", "rationale": "...", "evidence": [...]}\n'
        '  ]\n'
        "}\n"
        "```\n\n"
        "Every item MUST carry an `evidence` array citing at least 3 distinct job postings from the last 90 days that demonstrate the gap. Items with fewer citations get rejected mechanically; do not pad with low-quality postings. PREFER ZERO ADDITIONS over weakly-justified ones — empty arrays are the expected output when nothing has materially shifted. If a requirement is already covered at a lower level, link to it in `JOB_REQUIREMENTS.md` and do NOT add it.\n\n"
        "Do NOT edit `CURRICULUM.md`, `CURRICULUM_INDEX.md`, `README.md`, or `VERSIONS.md` directly — the runner regenerates those from the merged plan. Mark unresolved claims with `<!-- needs-research: ... -->`.\n"
    )


def build_supplemental_prompt(
    manifest: OrgManifest,
    role: RoleConfig,
    month: str,
    supplemental_dir: str,
) -> str:
    return (
        f"# Supplemental Content Packet - {role.title} - {month}\n\n"
        f"Preferred content agent: {content_generation_label(manifest)}.\n\n"
        "No ready gap-remediation work items were available. Review the role's "
        "`JOB_REQUIREMENTS.md` and `.aicg/job-requirements.json`, then create brief "
        f"source-backed supplemental content under `{supplemental_dir}/` for requirements "
        "that are useful but do not belong in the main curriculum path.\n\n"
        "Constraints:\n\n"
        "- Keep each supplemental page short and focused.\n"
        "- Link to main curriculum coverage when it exists.\n"
        "- Prefer official sources and clearly label practitioner references.\n"
        "- Do not duplicate lower-level role content; link to it instead.\n"
    )


def repo_action(
    repo: str,
    repo_path: Path,
    remote: str,
    command: list[str],
    status: str,
) -> dict[str, Any]:
    return {
        "repo": repo,
        "path": str(repo_path),
        "remote": remote,
        "command": shell_join(command),
        "status": status,
    }


def queue_priority(manifest: OrgManifest, repo: str, item: dict[str, Any]) -> int:
    """Compute a deterministic priority for the work queue.

    Lower number = higher priority (sorted ascending). The score is:

        severity_bias + role_level × 1000 + item.priority

    where ``severity_bias`` lets high-severity refresh items jump ahead
    of structural gaps. The default work item gets bias = 0; a
    high-severity refresh (broken security guidance, EOL'd tool) gets
    a large negative bias.
    """
    role = next((role for role in manifest.roles if role.solution_repo == repo), None)
    base = role.level if role else 100
    severity_bias = _severity_bias(item)
    return severity_bias + base * 1000 + int(item.get("priority", 100))


def _open_work_item_pr(
    repo_path: Path, plan: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    """Commit and open a PR for the just-verified work item.

    Failures bubble up as ``status=error`` with a stderr tail — the
    runner does not rollback the on-disk content if PR creation
    fails. Operators can re-run ``aicg pr --repo X --work-id Y``.
    """
    from .gitops import GitOpsError, prepare_pr

    # Re-audit the target so the PR body reflects post-generation
    # state, not the weekly-audit snapshot. The result is written to
    # .aicg/audit-report.json which prepare_pr reads.
    try:
        audit_report = audit_repo(
            workspace=repo_path.parent,
            repo_name=repo_path.name,
        )
    except Exception as exc:  # noqa: BLE001
        # Fall back to whatever audit-report.json existed before; we
        # never block PR creation on a stale audit.
        try:
            audit_report = read_state(repo_path, "audit-report.json")
        except FileNotFoundError:
            return {
                "status": "skipped",
                "reason": f"audit failed and no prior audit-report.json: {exc}",
            }

    try:
        validation_report = read_state(repo_path, "validation-report.json")
    except FileNotFoundError:
        validation_report = {"status": "not_run", "checks": []}

    try:
        pr_result = prepare_pr(
            repo_path,
            work_plan=plan,
            audit_report=audit_report,
            validation_report=validation_report,
            auto_merge=False,
            work_id=item["work_id"],
        )
    except GitOpsError as exc:
        return {"status": "error", "reason": str(exc)}

    return {
        "status": "opened",
        "pr_url": pr_result.get("pr_url"),
        "branch": pr_result.get("branch"),
        "changed_files": pr_result.get("changed_files", [])[:20],
    }


def _collect_freshness_items(repo_path: Path) -> list[dict[str, Any]]:
    """Read freshness reports from the repo and return their work items.

    Tolerates missing files (each report is optional) and malformed
    JSON (logs nothing, skips silently — the audit run shouldn't blow
    up because a stale report can't be parsed).
    """
    items: list[dict[str, Any]] = []
    state_dir = repo_path / ".aicg"
    for filename in (
        "freshness-links-report.json",
        "freshness-versions-report.json",
        "freshness-review-report.json",
    ):
        path = state_dir / filename
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in report.get("work_items", []) or []:
            if isinstance(entry, dict) and entry.get("id"):
                items.append(entry)
    return items


_BACKLOG_TYPES = {"exercise_depth_followup"}


def _severity_bias(item: dict[str, Any]) -> int:
    """Convert work item severity + type into a priority bias.

    Lower number = higher priority (queue is sorted ascending).

    - High-severity refresh items: -100000 (jumps above every
      structural gap regardless of role level).
    - Structural gaps (module_solution_gap, project_solution_gap): 0.
    - Medium-severity refresh + backlog (exercise_depth_followup): +5000
      (just below structural within the same role).
    - Low-severity refresh items: +20000 (deepest backlog).
    """
    severity = str(item.get("severity", "")).lower()
    work_type = str(item.get("type", ""))
    is_refresh = work_type.startswith("refresh_")
    is_backlog = work_type in _BACKLOG_TYPES
    if not is_refresh and not is_backlog:
        return 0
    if severity == "high":
        return -100_000
    if severity == "medium":
        return 5_000
    return 20_000


def is_git_dirty(repo_path: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path == ".aicg" or path.startswith(".aicg/"):
            continue
        return True
    return False


def git_tag_exists(repo_path: Path, tag: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--verify", "--quiet", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def run_command(command: list[str]) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": shell_join(command),
        "started_at": started,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def content_generation_command(manifest: OrgManifest) -> str | None:
    agent = manifest.content_generation.get("agent", {})
    command = agent.get("agent_command") or manifest.research.get("agent_command")
    return command or None


def content_generation_label(manifest: OrgManifest) -> str:
    agent = manifest.content_generation.get("agent", {})
    provider = agent.get("provider", "configured content agent")
    model = agent.get("model", "configured model")
    return f"{provider} {model}"
