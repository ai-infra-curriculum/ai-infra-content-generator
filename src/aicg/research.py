"""Process monthly research packets into curriculum updates.

:func:`generate_research_packets` (in :mod:`aicg.org_runner`) writes a
prompt packet per role describing what to research. This module is the
matching consumer: it invokes the configured research agent on each
packet and expects three outputs per role:

1. ``JOB_REQUIREMENTS.md`` — human-readable requirements roll-up
2. ``.aicg/job-requirements.json`` — machine-readable normalized
   requirements with evidence, owner role, and coverage status
3. ``.aicg/curriculum-plan-delta.json`` — proposed additions to
   the role's ``curriculum-plan.json``: new modules, new exercises,
   new projects. Strictly additive; the runner refuses to remove
   existing items via this channel.

The delta is merged into the existing ``curriculum-plan.json`` via
:func:`merge_curriculum_plan_delta`, after which :mod:`aicg.bootstrap`'s
``execute_curriculum_plan`` can scaffold the new skeletons. The next
audit picks up the resulting gaps and the daily-remediate loop fills
them in.

The processor is subscription-limit-aware: when the agent hits a
five-hour or weekly cap it returns ``deferred`` for that role and the
next monthly-research run picks up where it left off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .agent_cli import AgentCommandResult, run_agent_command
from .org_config import OrgManifest, RoleConfig, state_dir_for_manifest
from .state import utc_now, write_json

RESEARCH_APPLY_REPORT = "research-apply-report.json"
CURRICULUM_PLAN_FILE = "curriculum-plan.json"
CURRICULUM_PLAN_DELTA_FILE = ".aicg/curriculum-plan-delta.json"
JOB_REQUIREMENTS_JSON = ".aicg/job-requirements.json"
JOB_REQUIREMENTS_MD = "JOB_REQUIREMENTS.md"


class ResearchError(RuntimeError):
    """Raised when research processing cannot proceed."""


@dataclass(frozen=True)
class ResearchAgentConfig:
    enabled: bool
    agent_command: str | None
    timeout_seconds: int | None

    @classmethod
    def from_manifest(cls, manifest: OrgManifest) -> "ResearchAgentConfig":
        """Pull the research agent config from manifest.research.agent.

        Falls back to ``content_generation.agent`` when there is no
        explicit research block, since the research wrapper is just a
        sibling of the content wrapper that also allows web tools.
        """
        cfg = (manifest.research or {}).get("agent")
        if not cfg:
            cfg = (manifest.content_generation or {}).get("agent") or {}
        agent_command = cfg.get("agent_command") if isinstance(cfg, dict) else None
        return cls(
            enabled=bool(agent_command),
            agent_command=agent_command,
            timeout_seconds=(cfg.get("timeout_seconds") if isinstance(cfg, dict) else None),
        )


def research_apply(
    manifest: OrgManifest,
    workspace: Path,
    month: str | None = None,
    state_dir: Path | None = None,
    runner_root: Path | None = None,
) -> dict[str, Any]:
    """Process every role's research packet via the configured agent."""
    state_root = state_dir_for_manifest(manifest, state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    month = month or date.today().strftime("%Y-%m")
    prompt_dir = state_root / "research" / month
    if not prompt_dir.exists():
        raise ResearchError(
            f"No research packets at {prompt_dir}. Run `aicg org research` first."
        )

    config = ResearchAgentConfig.from_manifest(manifest)
    runner_root = runner_root or Path(__file__).resolve().parents[2]

    role_reports: list[dict[str, Any]] = []
    for role in sorted(manifest.roles, key=lambda item: item.level):
        prompt_path = prompt_dir / f"{role.id}.md"
        if not prompt_path.exists():
            role_reports.append(_skip(role, "prompt_missing", str(prompt_path)))
            continue
        learning_path = workspace / role.learning_repo
        if not learning_path.exists():
            role_reports.append(_skip(role, "learning_repo_missing", str(learning_path)))
            continue
        if not config.enabled:
            role_reports.append(_skip(role, "agent_not_configured", None))
            continue

        result = _invoke_agent(
            config=config,
            prompt_path=prompt_path,
            learning_path=learning_path,
            role=role,
            runner_root=runner_root,
        )
        if result.limit_reached:
            role_reports.append(
                {
                    "role": role.id,
                    "status": "deferred",
                    "reason": (
                        f"Agent subscription limit ({result.limit_scope}); "
                        f"retry after {result.retry_after}."
                    ),
                    "retry_after": result.retry_after,
                }
            )
            continue
        if result.returncode != 0:
            role_reports.append(
                {
                    "role": role.id,
                    "status": "agent_failed",
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-800:],
                }
            )
            continue

        # Verify the agent produced the expected outputs.
        outputs = _detect_outputs(learning_path)
        merge_report: dict[str, Any] | None = None
        if outputs["delta_present"]:
            merge_report = merge_curriculum_plan_delta(learning_path)
        role_reports.append(
            {
                "role": role.id,
                "status": "applied",
                "outputs": outputs,
                "delta_merge": merge_report,
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "research_apply",
        "month": month,
        "roles": role_reports,
    }
    write_json(state_root / RESEARCH_APPLY_REPORT, report)
    return report


def merge_curriculum_plan_delta(learning_path: Path) -> dict[str, Any]:
    """Merge ``.aicg/curriculum-plan-delta.json`` into ``curriculum-plan.json``.

    Additive only — never removes existing modules, exercises, or
    projects. New items are deduplicated by stable id/slug. Returns a
    report describing what was added vs. skipped (already present).
    """
    delta_path = learning_path / CURRICULUM_PLAN_DELTA_FILE
    plan_path = learning_path / CURRICULUM_PLAN_FILE
    if not delta_path.exists():
        return {"present": False, "added": [], "skipped": []}

    try:
        delta = json.loads(delta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"present": True, "error": f"invalid JSON: {exc}"}

    plan: dict[str, Any]
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"present": True, "error": f"existing curriculum-plan.json invalid: {exc}"}
    else:
        plan = {"schema_version": 1, "modules": [], "projects": []}

    added: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    modules = plan.setdefault("modules", [])
    existing_module_ids = {str(m.get("id")) for m in modules if isinstance(m, dict)}
    for entry in delta.get("modules", []) or []:
        if not isinstance(entry, dict):
            continue
        mod_id = str(entry.get("id") or "")
        if not mod_id:
            continue
        if mod_id in existing_module_ids:
            skipped.append({"kind": "module", "id": mod_id})
            continue
        modules.append(entry)
        existing_module_ids.add(mod_id)
        added.append({"kind": "module", "id": mod_id})

    # Exercises live under their parent module.
    modules_by_id = {str(m.get("id")): m for m in modules if isinstance(m, dict)}
    for entry in delta.get("exercises", []) or []:
        if not isinstance(entry, dict):
            continue
        parent_id = str(entry.get("module_id") or entry.get("module") or "")
        exercise = entry.get("exercise") or entry
        exercise_slug = str(
            exercise.get("slug")
            or exercise.get("id")
            or ""
        )
        if not parent_id or not exercise_slug or parent_id not in modules_by_id:
            skipped.append({"kind": "exercise", "id": exercise_slug, "reason": "no_parent"})
            continue
        parent = modules_by_id[parent_id]
        parent_exercises = parent.setdefault("exercises", [])
        existing_slugs = {
            str((ex or {}).get("slug") or (ex or {}).get("id"))
            for ex in parent_exercises
            if isinstance(ex, dict)
        }
        if exercise_slug in existing_slugs:
            skipped.append({"kind": "exercise", "id": exercise_slug})
            continue
        parent_exercises.append(exercise)
        added.append({"kind": "exercise", "id": exercise_slug, "module_id": parent_id})

    projects = plan.setdefault("projects", [])
    existing_project_ids = {str(p.get("id")) for p in projects if isinstance(p, dict)}
    for entry in delta.get("projects", []) or []:
        if not isinstance(entry, dict):
            continue
        proj_id = str(entry.get("id") or "")
        if not proj_id:
            continue
        if proj_id in existing_project_ids:
            skipped.append({"kind": "project", "id": proj_id})
            continue
        projects.append(entry)
        existing_project_ids.add(proj_id)
        added.append({"kind": "project", "id": proj_id})

    if added:
        plan.setdefault("research_provenance", []).append(
            {
                "applied_at": utc_now(),
                "added": added,
                "rationale": delta.get("rationale", ""),
            }
        )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    return {
        "present": True,
        "plan_path": str(plan_path),
        "added": added,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _skip(role: RoleConfig, status: str, detail: str | None) -> dict[str, Any]:
    return {
        "role": role.id,
        "status": status,
        "detail": detail,
    }


def _detect_outputs(learning_path: Path) -> dict[str, Any]:
    return {
        "job_requirements_md": (learning_path / JOB_REQUIREMENTS_MD).exists(),
        "job_requirements_json": (learning_path / JOB_REQUIREMENTS_JSON).exists(),
        "delta_present": (learning_path / CURRICULUM_PLAN_DELTA_FILE).exists(),
    }


def _invoke_agent(
    config: ResearchAgentConfig,
    prompt_path: Path,
    learning_path: Path,
    role: RoleConfig,
    runner_root: Path,
) -> AgentCommandResult:
    output_dir = learning_path / ".aicg" / "research" / "runs" / role.id
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted = config.agent_command.format(  # type: ignore[union-attr]
        prompt=str(prompt_path),
        output_dir=str(output_dir),
        repo=str(learning_path),
        work_id=f"research:{role.id}",
        runner=str(runner_root),
    )
    return run_agent_command(formatted, cwd=learning_path)
