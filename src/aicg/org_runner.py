"""Manifest-driven org automation operations."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
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

        # Splice in PR-response items emitted by the previous steward
        # run. These are high-severity so they jump ahead of structural
        # gaps — keeping an in-flight PR's loop closed beats opening
        # new ones.
        for refresh_item in _collect_pr_response_items(state_dir, repo):
            queue_items.append(
                {
                    "id": refresh_item["id"],
                    "repo": repo,
                    "work_id": refresh_item["id"].split(":", 1)[-1],
                    "type": refresh_item["type"],
                    "severity": refresh_item.get("severity", "high"),
                    "title": refresh_item.get(
                        "title",
                        f"Respond to PR #{refresh_item.get('pr_number')}",
                    ),
                    "pr_number": refresh_item.get("pr_number"),
                    "head_ref": refresh_item.get("head_ref"),
                    "blockers": refresh_item.get("blockers", []),
                    "response_count": refresh_item.get("response_count", 0),
                    "max_response_attempts": refresh_item.get(
                        "max_response_attempts", 3
                    ),
                    "status": refresh_item.get("status", "ready"),
                    "priority": queue_priority(
                        manifest,
                        repo,
                        {**refresh_item, "severity": "high"},
                    ),
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


DRAIN_WALL_CLOCK_SECONDS = 7200  # 2h cap per tick
VERIFY_SELF_HEAL_MAX_ATTEMPTS = 3
INLINE_MERGE_POLL_SECONDS = 15
INLINE_MERGE_TIMEOUT_SECONDS = 900  # 15 min per PR
CI_SELF_HEAL_MAX_ATTEMPTS = 3
CI_WAIT_PER_ATTEMPT_SECONDS = 180  # 3 min per CI cycle


def run_daily_remediation(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path | None = None,
    drain_until_empty: bool = False,
    wall_clock_cap_seconds: int = DRAIN_WALL_CLOCK_SECONDS,
) -> dict[str, Any]:
    """Drive ONE work item end-to-end from ready to merged.

    Default is single-item-per-tick: pick the highest-priority ready
    item, run the full pipeline (generate → verify-with-self-heal →
    propagate → commit/PR → inline-merge), exit. Pass
    ``drain_until_empty=True`` to keep processing items until the queue
    is empty, subscription limit hits, or the wall-clock cap fires.

    Single-item mode is the intended default: one item lands fully
    (committed, PR opened, merged on main) per tick. The next hourly
    tick picks up the next item. No partial state left between ticks.
    """
    state_dir = state_dir_for_manifest(manifest, state_dir)
    queue_path = state_dir / ORG_QUEUE
    if not queue_path.exists():
        run_org_audit(manifest, workspace, state_dir=state_dir)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    refresh_deferred_items(queue)

    deadline = time.monotonic() + max(60, wall_clock_cap_seconds)
    drained: list[dict[str, Any]] = []
    aggregate_status = "no_items"
    exit_reason = "queue_empty"

    while True:
        ready = [
            item for item in queue.get("work_items", []) if item.get("status") == "ready"
        ]
        if not ready:
            if not drained:
                # Preserve old behavior on the very first iteration.
                supp = generate_supplemental_packet(manifest, workspace, state_dir)
                return supp
            break
        if time.monotonic() >= deadline:
            exit_reason = "wall_clock_cap"
            break

        item = ready[0]
        item_result = _process_one_item(
            manifest=manifest,
            workspace=workspace,
            state_dir=state_dir,
            queue=queue,
            queue_path=queue_path,
            item=item,
        )
        drained.append(item_result)
        aggregate_status = item_result.get("status", aggregate_status)

        # Hard exits: subscription limit + escalation force us to stop
        # so other timers / a human can take a look.
        if item_result.get("status") == "deferred" and item_result.get(
            "defer_reason"
        ) == "agent_subscription_limit":
            exit_reason = "subscription_limit"
            break
        if not drain_until_empty:
            exit_reason = "single_item_mode"
            break

    summary = {
        "schema_version": 2,
        "generated_at": utc_now(),
        "operation": "daily_remediate",
        "status": aggregate_status,
        "exit_reason": exit_reason,
        "items_processed": len(drained),
        "items": drained,
    }
    write_json(state_dir / "daily-run-state.json", summary)
    return summary


def _process_one_item(
    manifest: OrgManifest,
    workspace: Path,
    state_dir: Path,
    queue: dict[str, Any],
    queue_path: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Drive a single work item from ready → merged (or to a failure)."""
    repo_path = workspace / item["repo"]
    result = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "process_work_item",
        "selected": item,
        "status": "started",
    }
    if item.get("type") == "respond_pr_review":
        return _handle_pr_response_item(
            manifest=manifest,
            workspace=workspace,
            repo_path=repo_path,
            item=item,
            state_dir=state_dir,
            result=result,
        )

    plan = read_state(repo_path, "work-plan.json")
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

        # Verify with self-heal: if the agent's output misses a required
        # heading, has a broken source citation, etc., feed the findings
        # back to the agent so it can fix them. Cap retries; escalate
        # past the cap.
        verify_outcome = _verify_with_self_heal(
            manifest=manifest,
            workspace=workspace,
            repo_path=repo_path,
            item=item,
            plan=plan,
        )
        result["verify"] = verify_outcome["verify"]
        result["self_heal_attempts"] = verify_outcome["attempts"]

        if verify_outcome["status"] == "verified":
            item["status"] = "verified"
            from .propagate import propagate_repo

            propagate_report = propagate_repo(
                workspace, item["repo"], work_id=item["work_id"]
            )
            result["propagate"] = {
                "status": propagate_report["status"],
                "updated_count": len(propagate_report.get("updated", [])),
            }
            pr_outcome = _open_work_item_pr(repo_path, plan, item)
            result["pr"] = pr_outcome
            if pr_outcome.get("status") == "opened":
                item["pr_url"] = pr_outcome.get("pr_url")
                item["pr_branch"] = pr_outcome.get("branch")
                # Inline drive PR to merge: poll CI + reviews, auto-
                # respond to anything blocking, merge when clean.
                merge_outcome = _drive_pr_to_merge(
                    manifest=manifest,
                    repo_path=repo_path,
                    pr_url=pr_outcome["pr_url"],
                    branch=pr_outcome["branch"],
                )
                result["inline_merge"] = merge_outcome
                if merge_outcome.get("status") == "merged":
                    item["status"] = "merged"
        else:
            item["status"] = "verification_failed"
            item["last_failure"] = verify_outcome.get("last_finding_summary", "")
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
    # Each item also persists its own per-item state file for debugging.
    write_json(state_dir / "daily-item-last.json", result)
    return result


def _verify_with_self_heal(
    manifest: OrgManifest,
    workspace: Path,
    repo_path: Path,
    item: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Verify the work item; on failure, re-invoke the agent with the
    findings as context and retry. Caps at VERIFY_SELF_HEAL_MAX_ATTEMPTS.
    """
    from .judge import JudgeConfig
    from .verify import verify_repo

    judge_config = JudgeConfig.from_manifest(manifest)
    attempts: list[dict[str, Any]] = []
    last_report: dict[str, Any] = {}
    last_findings: list[dict[str, Any]] = []

    for attempt_num in range(1, VERIFY_SELF_HEAL_MAX_ATTEMPTS + 1):
        last_report = verify_repo(
            workspace,
            item["repo"],
            work_id=item["work_id"],
            judge_config=judge_config if judge_config.enabled else None,
        )
        attempts.append(
            {
                "attempt": attempt_num,
                "status": last_report["status"],
                "work_item_count": last_report["work_item_count"],
            }
        )
        if last_report["status"] in {"verified", "no_items"}:
            return {
                "status": "verified" if last_report["status"] == "verified" else "no_items",
                "verify": {
                    "status": last_report["status"],
                    "work_item_count": last_report["work_item_count"],
                },
                "attempts": attempts,
            }
        # Collect findings from this attempt.
        last_findings = _collect_verify_findings(last_report)
        if attempt_num >= VERIFY_SELF_HEAL_MAX_ATTEMPTS:
            break
        # Ask the agent to address the findings, then re-verify.
        heal_outcome = _invoke_verify_self_heal_agent(
            manifest=manifest,
            repo_path=repo_path,
            item=item,
            plan=plan,
            findings=last_findings,
            attempt=attempt_num,
        )
        attempts[-1]["self_heal"] = heal_outcome
        if heal_outcome.get("status") not in {"ok", "no_changes"}:
            # Agent failed or hit a limit; stop trying.
            break

    summary = "; ".join(
        f"{f['type']}: {f['message'][:160]}" for f in last_findings[:3]
    ) or "verification_failed (no findings reported)"
    return {
        "status": "verification_failed",
        "verify": {
            "status": last_report.get("status", "verification_failed"),
            "work_item_count": last_report.get("work_item_count", 0),
        },
        "attempts": attempts,
        "last_finding_summary": summary,
    }


def _collect_verify_findings(verify_report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for wi in verify_report.get("work_items", []):
        for finding in wi.get("findings", []):
            findings.append(finding)
        for action in wi.get("actions", []):
            for finding in action.get("findings", []) or []:
                # Tag with target path for clarity.
                f = dict(finding)
                f.setdefault("target", action.get("path"))
                findings.append(f)
    return findings


def _invoke_verify_self_heal_agent(
    manifest: OrgManifest,
    repo_path: Path,
    item: dict[str, Any],
    plan: dict[str, Any],
    findings: list[dict[str, Any]],
    attempt: int,
) -> dict[str, Any]:
    """Build a self-heal prompt and re-invoke the content agent."""
    if not findings:
        return {"status": "no_changes", "reason": "no findings to address"}

    output_dir = (
        repo_path / ".aicg" / "self-heal" / item["work_id"] / f"attempt-{attempt}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(_build_self_heal_prompt(item, findings), encoding="utf-8")

    command = content_generation_command(manifest).format(
        prompt=str(prompt_path),
        output_dir=str(output_dir),
        repo=str(repo_path),
        work_id=f"self-heal:{item['work_id']}:{attempt}",
        runner=str(Path(__file__).resolve().parents[2]),
    )
    from .agent_cli import run_agent_command

    agent_result = run_agent_command(command, cwd=repo_path)
    if agent_result.limit_reached:
        return {
            "status": "subscription_limit",
            "retry_after": agent_result.retry_after,
        }
    if agent_result.returncode != 0:
        return {
            "status": "agent_failed",
            "returncode": agent_result.returncode,
            "stderr_tail": agent_result.stderr[-400:],
        }
    return {"status": "ok"}


def _build_self_heal_prompt(
    item: dict[str, Any], findings: list[dict[str, Any]]
) -> str:
    lines = [
        f"# Self-heal: address verify findings for {item['work_id']}",
        "",
        "## Goal",
        "",
        "The previous attempt at this work item produced content that",
        "failed contract verification. Fix the specific findings listed",
        "below by editing only the affected files. Do NOT regenerate",
        "from scratch and do NOT broaden the scope.",
        "",
        "## Findings",
        "",
    ]
    for i, f in enumerate(findings, 1):
        target = f.get("target") or "?"
        lines.append(f"### {i}. `{f.get('type')}` ({f.get('severity', 'error')})")
        lines.append("")
        lines.append(f"- Target: `{target}`")
        msg = (f.get("message") or "").strip()
        if msg:
            lines.append(f"- Message: {msg}")
        for key, value in f.items():
            if key in {"type", "severity", "message", "target"}:
                continue
            if value in (None, "", [], {}):
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "## Output contract",
            "",
            "- Edit ONLY the files listed in the findings.",
            "- Preserve the existing content; add or rename headings",
            "  rather than rewriting whole sections.",
            "- Do NOT touch CURRICULUM.md, VERSIONS.md, or anything",
            "  outside the affected files.",
        ]
    )
    return "\n".join(lines) + "\n"


def _drive_pr_to_merge(
    manifest: OrgManifest,
    repo_path: Path,
    pr_url: str,
    branch: str,
) -> dict[str, Any]:
    """Inline-poll a freshly-opened PR through to merge, with CI self-heal.

    Per CI cycle:
      1. wait_for_ci (CI_WAIT_PER_ATTEMPT_SECONDS budget per attempt)
      2. CI failure → fetch annotations → ask agent to address → push
         to the same branch → loop, up to CI_SELF_HEAL_MAX_ATTEMPTS
      3. CI passes (or no_ci_present) → guardrails → reviews → merge
    """
    from .steward import (
        classify_review_state,
        enable_auto_merge,
        evaluate_pr_guardrails,
        fetch_failed_checks,
        fetch_review_state,
        pr_view,
        wait_for_ci,
        _owner_repo_from_url,
    )

    pr_number = _pr_number_from_url(pr_url)
    if pr_number is None:
        return {"status": "no_pr_number", "pr_url": pr_url}

    owner, repo_name = _owner_repo_from_url(pr_url)
    history: list[dict[str, Any]] = []
    deadline = time.monotonic() + INLINE_MERGE_TIMEOUT_SECONDS

    ci_attempts = 0
    while time.monotonic() < deadline:
        ci_attempts += 1
        ci_result = wait_for_ci(
            repo_path=repo_path,
            pr_number=pr_number,
            timeout_seconds=min(
                CI_WAIT_PER_ATTEMPT_SECONDS, int(deadline - time.monotonic())
            ),
            poll_seconds=INLINE_MERGE_POLL_SECONDS,
        )
        history.append({"phase": "ci", "attempt": ci_attempts, "result": ci_result["status"]})

        if ci_result["status"] == "failure":
            if ci_attempts >= CI_SELF_HEAL_MAX_ATTEMPTS:
                return {
                    "status": "ci_failed_escalated",
                    "pr_url": pr_url,
                    "ci_attempts": ci_attempts,
                    "history": history,
                }
            if not owner or not repo_name:
                return {
                    "status": "ci_failed",
                    "pr_url": pr_url,
                    "reason": "could not parse owner/repo from PR URL",
                    "history": history,
                }
            failed_checks = fetch_failed_checks(
                repo_path, owner, repo_name, pr_number
            )
            heal_outcome = _invoke_ci_self_heal_agent(
                manifest=manifest,
                repo_path=repo_path,
                pr_number=pr_number,
                branch=branch,
                failed_checks=failed_checks,
                attempt=ci_attempts,
            )
            history.append(
                {
                    "phase": "ci_self_heal",
                    "attempt": ci_attempts,
                    "result": heal_outcome.get("status"),
                    "failed_checks": [c["name"] for c in failed_checks],
                }
            )
            if heal_outcome.get("status") not in {"pushed", "no_changes"}:
                return {
                    "status": "ci_failed",
                    "pr_url": pr_url,
                    "history": history,
                    "self_heal": heal_outcome,
                }
            # Pushed a fix — loop to re-wait for CI.
            continue
        if ci_result["status"] == "timeout":
            return {
                "status": "ci_timeout",
                "pr_url": pr_url,
                "history": history,
            }
        # success or no_ci_present → continue to guardrails / reviews
        pr = pr_view(repo_path, pr_number) or {}
        guardrails = evaluate_pr_guardrails(repo_path=repo_path, pr=pr)
        if not guardrails.allowed:
            history.append({"phase": "guardrails", "blockers": list(guardrails.blockers)})
            return {
                "status": "guardrails_blocked",
                "pr_url": pr_url,
                "blockers": list(guardrails.blockers),
                "history": history,
            }

        if owner and repo_name:
            review_state = fetch_review_state(repo_path, owner, repo_name, pr_number)
            if not review_state.get("error"):
                cls = classify_review_state(
                    review_state, pr_author=(pr.get("author") or {}).get("login")
                )
                if cls["verdict"] == "blocked":
                    history.append({"phase": "review", "verdict": "blocked"})
                    return {
                        "status": "review_blocked",
                        "pr_url": pr_url,
                        "blockers": cls["blockers"],
                        "history": history,
                    }

        merge_result = enable_auto_merge(repo_path, pr_number, method="squash")
        history.append({"phase": "merge", "result": merge_result.get("status")})
        if merge_result.get("status") == "merged":
            return {"status": "merged", "pr_url": pr_url, "history": history}
        if merge_result.get("status") == "auto_merge_enabled":
            return {
                "status": "auto_merge_enabled",
                "pr_url": pr_url,
                "history": history,
            }
        # Merge call returned something else — wait and retry.
        time.sleep(INLINE_MERGE_POLL_SECONDS)

    return {"status": "merge_timeout", "pr_url": pr_url, "history": history}


def _invoke_ci_self_heal_agent(
    manifest: OrgManifest,
    repo_path: Path,
    pr_number: int,
    branch: str,
    failed_checks: list[dict[str, Any]],
    attempt: int,
) -> dict[str, Any]:
    """Invoke the agent on the failed CI checks, then commit + push."""
    # Make sure the local working tree is on the PR branch — the runner
    # restored main after PR creation.
    subprocess.run(
        ["git", "-C", str(repo_path), "fetch", "origin", branch],
        capture_output=True, text=True, check=False,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "checkout", branch],
        capture_output=True, text=True, check=False,
    )

    output_dir = (
        repo_path / ".aicg" / "ci-self-heal" / f"pr-{pr_number}" / f"attempt-{attempt}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(
        _build_ci_self_heal_prompt(pr_number, failed_checks), encoding="utf-8"
    )

    command = content_generation_command(manifest).format(
        prompt=str(prompt_path),
        output_dir=str(output_dir),
        repo=str(repo_path),
        work_id=f"ci-heal:pr-{pr_number}:{attempt}",
        runner=str(Path(__file__).resolve().parents[2]),
    )
    from .agent_cli import run_agent_command

    agent_result = run_agent_command(command, cwd=repo_path)
    if agent_result.limit_reached:
        _checkout_main(repo_path)
        return {
            "status": "subscription_limit",
            "retry_after": agent_result.retry_after,
        }
    if agent_result.returncode != 0:
        _checkout_main(repo_path)
        return {
            "status": "agent_failed",
            "returncode": agent_result.returncode,
            "stderr_tail": agent_result.stderr[-400:],
        }

    # Stage + commit + push whatever the agent changed.
    status = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    )
    if not status.stdout.strip():
        _checkout_main(repo_path)
        return {"status": "no_changes"}

    for args in (
        ["git", "-C", str(repo_path), "add", "--all"],
        ["git", "-C", str(repo_path), "commit", "-m",
         f"aicg: address CI failures on PR #{pr_number} (attempt {attempt})"],
        ["git", "-C", str(repo_path), "push"],
    ):
        c = subprocess.run(args, capture_output=True, text=True, check=False)
        if c.returncode != 0:
            _checkout_main(repo_path)
            return {
                "status": "push_failed",
                "step": args[3],
                "stderr_tail": c.stderr[-400:],
            }
    _checkout_main(repo_path)
    return {"status": "pushed"}


def _build_ci_self_heal_prompt(
    pr_number: int, failed_checks: list[dict[str, Any]]
) -> str:
    lines = [
        f"# Address CI failures on PR #{pr_number}",
        "",
        "## Goal",
        "",
        "The PR you just opened failed CI. Fix the failures listed",
        "below by editing files on the current branch. Do NOT regenerate",
        "the content from scratch — make the minimal edit needed to",
        "satisfy each failing check.",
        "",
        "## Failed checks",
        "",
    ]
    for i, check in enumerate(failed_checks, 1):
        lines.append(f"### {i}. `{check.get('name')}` ({check.get('conclusion')})")
        lines.append("")
        if check.get("details_url"):
            lines.append(f"- Details: <{check['details_url']}>")
        annotations = check.get("annotations") or []
        if annotations:
            lines.append("- Annotations:")
            for ann in annotations[:20]:
                path = ann.get("path") or "?"
                line = ann.get("start_line") or "?"
                msg = (ann.get("message") or "").splitlines()[0][:300]
                level = ann.get("level") or "?"
                lines.append(f"  - `{path}:{line}` ({level}): {msg}")
        else:
            lines.append(
                "- No annotations exposed by this check. Infer the failure"
                " from the check name; if you cannot tell what's wrong from"
                " the name alone, leave a brief comment in the PR and exit"
                " without editing rather than guessing."
            )
        lines.append("")
    lines.extend(
        [
            "## Output contract",
            "",
            "- Edit ONLY files inside this repo on the current branch.",
            "- Preserve the existing structure; do not delete sections.",
            "- Do NOT touch CURRICULUM.md, README.md, or VERSIONS.md.",
            "- One atomic commit covering all fixes is fine.",
        ]
    )
    return "\n".join(lines) + "\n"


def _pr_number_from_url(url: str) -> int | None:
    if not url:
        return None
    parts = url.rstrip("/").split("/")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


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


def _handle_pr_response_item(
    manifest: OrgManifest,
    workspace: Path,
    repo_path: Path,
    item: dict[str, Any],
    state_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Process a respond_pr_review work item.

    Checks out the PR branch, builds a response prompt that lists each
    blocker, invokes the response agent, and pushes follow-up commits
    to the PR. The next steward pass re-checks reviews; if blockers
    clear, it merges. If the agent runs out of retries the item is
    flipped to escalated and daily-issues picks it up.
    """
    pr_number = item.get("pr_number")
    head_ref = item.get("head_ref")
    blockers = item.get("blockers", [])
    response_count = int(item.get("response_count", 0))
    max_attempts = int(item.get("max_response_attempts", 3))

    if response_count >= max_attempts:
        item["status"] = "escalated"
        result["status"] = "escalated"
        result["reason"] = (
            f"PR #{pr_number} hit max response attempts ({max_attempts}); "
            "daily-issues will surface to a human."
        )
        _persist_pr_response_state(state_dir, item)
        return result

    if not pr_number or not head_ref:
        item["status"] = "failed_permanently"
        result["status"] = "failed_permanently"
        result["reason"] = "missing pr_number or head_ref"
        _persist_pr_response_state(state_dir, item)
        return result

    # Check out the PR branch so the agent edits the right tree.
    co = subprocess.run(
        ["git", "-C", str(repo_path), "fetch", "origin", head_ref],
        capture_output=True, text=True, check=False,
    )
    if co.returncode != 0:
        result["status"] = "fetch_failed"
        result["stderr_tail"] = co.stderr[-400:]
        return result
    subprocess.run(
        ["git", "-C", str(repo_path), "checkout", head_ref],
        capture_output=True, text=True, check=False,
    )

    prompt = _build_pr_response_prompt(pr_number, blockers, item)
    prompt_dir = repo_path / ".aicg" / "pr-responses" / f"pr-{pr_number}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    output_dir = prompt_dir / f"attempt-{response_count + 1}"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = content_generation_command(manifest).format(
        prompt=str(prompt_path),
        output_dir=str(output_dir),
        repo=str(repo_path),
        work_id=item["id"],
        runner=str(Path(__file__).resolve().parents[2]),
    )
    from .agent_cli import run_agent_command

    agent_result = run_agent_command(command, cwd=repo_path)
    item["response_count"] = response_count + 1

    if agent_result.limit_reached:
        item["status"] = "deferred"
        item["retry_after"] = agent_result.retry_after
        result["status"] = "deferred"
        result["retry_after"] = agent_result.retry_after
        _checkout_main(repo_path)
        _persist_pr_response_state(state_dir, item)
        return result

    if agent_result.returncode != 0:
        result["status"] = "agent_failed"
        result["returncode"] = agent_result.returncode
        result["stderr_tail"] = agent_result.stderr[-400:]
        _checkout_main(repo_path)
        _persist_pr_response_state(state_dir, item)
        return result

    # Commit + push whatever the agent changed.
    push_outcome = _commit_and_push_pr_response(repo_path, pr_number, response_count + 1)
    result["status"] = push_outcome["status"]
    result["push"] = push_outcome
    item["status"] = (
        "ready"
        if push_outcome["status"] in {"pushed", "no_changes"}
        else "failed_permanently"
    )
    _checkout_main(repo_path)
    _persist_pr_response_state(state_dir, item)
    return result


def _build_pr_response_prompt(
    pr_number: int, blockers: list[dict[str, Any]], item: dict[str, Any]
) -> str:
    lines = [
        f"# Respond to PR #{pr_number} review comments",
        "",
        "## Goal",
        "",
        "Address every blocker listed below by editing the code on the",
        "current branch. You are NOT writing new curriculum content — you",
        "are responding to a reviewer (human or bot). Make the smallest",
        "change that resolves each blocker.",
        "",
        "## Source policy",
        "",
        "Same as content generation: official sources first, no invented",
        "facts, mark unresolved claims with `<!-- needs-research: ... -->`.",
        "",
        "## Blockers",
        "",
    ]
    for i, b in enumerate(blockers, 1):
        if b.get("kind") == "changes_requested":
            lines.append(f"### {i}. CHANGES_REQUESTED from @{b.get('author')}")
            lines.append("")
            lines.append(f"> {(b.get('body') or '').strip() or '(no body)'}")
            lines.append("")
        elif b.get("kind") == "unresolved_thread":
            actor = "bot" if b.get("is_bot") else "human"
            lines.append(
                f"### {i}. Unresolved review thread "
                f"({actor}: @{b.get('author')}) "
                f"in `{b.get('path')}:{b.get('line')}`"
            )
            lines.append("")
            lines.append(f"> {(b.get('body') or '').strip() or '(no body)'}")
            lines.append("")
    lines.extend(
        [
            "## Output contract",
            "",
            "- Edit only files in this repo on the current branch.",
            "- Make atomic commits per blocker where possible.",
            "- Do NOT mark review threads resolved — only the reviewer can do",
            "  that. Your job is to push commits that address the underlying",
            "  issue; bot threads will auto-resolve when their metric recovers,",
            "  human threads stay open until the human resolves them.",
            "- Do NOT touch CURRICULUM.md, README.md, VERSIONS.md, or anything",
            "  outside the scope of the reviewer's comment.",
        ]
    )
    return "\n".join(lines) + "\n"


def _commit_and_push_pr_response(
    repo_path: Path, pr_number: int, attempt: int
) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    )
    if not status.stdout.strip():
        return {"status": "no_changes"}
    msg = f"aicg: respond to PR #{pr_number} review (attempt {attempt})"
    for args in (
        ["git", "-C", str(repo_path), "add", "--all"],
        ["git", "-C", str(repo_path), "commit", "-m", msg],
        ["git", "-C", str(repo_path), "push"],
    ):
        c = subprocess.run(args, capture_output=True, text=True, check=False)
        if c.returncode != 0:
            return {
                "status": "push_failed",
                "step": args[3],
                "stderr_tail": c.stderr[-400:],
            }
    return {"status": "pushed", "message": msg}


def _checkout_main(repo_path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(repo_path), "checkout", "main"],
        capture_output=True, text=True, check=False,
    )


def _persist_pr_response_state(state_dir: Path, item: dict[str, Any]) -> None:
    """Write the updated item back into pr-response-queue.json."""
    from .steward import PR_RESPONSE_QUEUE

    path = state_dir / PR_RESPONSE_QUEUE
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    items = payload.get("items", [])
    for i, existing in enumerate(items):
        if existing.get("id") == item["id"]:
            items[i] = {**existing, **item, "updated_at": utc_now()}
            break
    payload["items"] = items
    write_json(path, payload)


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


def _collect_pr_response_items(state_dir: Path, repo: str) -> list[dict[str, Any]]:
    """Read pr-response-queue.json from the org state and filter to ``repo``.

    Steward populates this queue when it finds CHANGES_REQUESTED reviews
    or unresolved review threads. Items with status='escalated' are
    skipped here — daily-issues handles those.
    """
    path = state_dir / "pr-response-queue.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    # Both 'ready' (work the daily loop will pick up) and 'escalated'
    # (work that daily-issues should open a tracking issue for) belong
    # in the queue — daily-remediate skips escalated items, daily-issues
    # acts on them.
    return [
        item
        for item in (payload.get("items") or [])
        if item.get("repo") == repo and item.get("status") in {"ready", "escalated"}
    ]


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
_HIGH_PRIORITY_TYPES = {"respond_pr_review"}


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
    is_high_priority = work_type in _HIGH_PRIORITY_TYPES
    if is_high_priority:
        # PR responses always jump structural gaps — keeping an open PR
        # moving beats opening another one.
        return -200_000
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
