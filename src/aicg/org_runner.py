"""Manifest-driven org automation operations."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import date, datetime
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
        result["status"] = "generated"
        result["run_state"] = run_state
        from .verify import verify_repo

        verify_report = verify_repo(
            workspace, item["repo"], work_id=item["work_id"]
        )
        result["verify"] = {
            "status": verify_report["status"],
            "work_item_count": verify_report["work_item_count"],
        }
        if verify_report["status"] == "verified":
            item["status"] = "verified"
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
    write_json(queue_path, queue)
    write_json(state_dir / "daily-run-state.json", result)
    return result


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
) -> dict[str, Any]:
    state_dir = state_dir_for_manifest(manifest, state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    repos = []
    for repo in manifest.repo_names:
        repo_path = workspace / repo
        repos.append(
            {
                "repo": repo,
                "path": str(repo_path),
                "present": repo_path.exists(),
                "planned_checks": [
                    "inspect open PRs",
                    "merge PRs with green CI and passing AICG guardrails",
                    "update issues from audit/work queue state",
                    "summarize unresolved discussion items for review",
                ],
            }
        )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "steward",
        "status": "dry_run",
        "repos": repos,
    }
    write_json(state_dir / ORG_STEWARD_REPORT, report)
    return report


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
        "- `JOB_REQUIREMENTS.md`: grouped requirements, posting evidence, curriculum links, and external resources for out-of-scope items.\n"
        "- `.aicg/job-requirements.json`: machine-readable requirements, evidence, owner role, coverage path, and status.\n"
        "- Update `CURRICULUM.md`, `CURRICULUM_INDEX.md`, `README.md`, and `VERSIONS.md` when coverage changes.\n"
        "- Preserve the existing format of those documentation files: heading order, table shape, link style, and version-history conventions.\n"
        "- Update the org-level README in the `.github` repo when org-wide curriculum navigation changes.\n"
        "- Curriculum plan updates only when the requirement is relevant and not already covered at a lower level.\n"
        "- Mark unresolved claims with `<!-- needs-research: ... -->`.\n"
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
    role = next((role for role in manifest.roles if role.solution_repo == repo), None)
    base = role.level if role else 100
    return base * 1000 + int(item.get("priority", 100))


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
