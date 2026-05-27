"""Process quarterly research packets into curriculum proposals.

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

The delta is NOT auto-merged. Instead :func:`validate_delta_against_caps`
applies per-run caps and evidence thresholds from the manifest, then
:func:`write_proposal` emits a human-readable ``RESEARCH_PROPOSAL_*.md``
plus the filtered delta into a branch and opens a PR for human review.
A separate command (:mod:`aicg.cli.cmd_org_promote_plan`) is what
actually merges an approved proposal into ``curriculum-plan.json``.

This split is intentional: research can propose, but the runner does
not add curriculum to the catalog autonomously. A human must approve
every new module / exercise / project.

The processor is subscription-limit-aware: when the agent hits a
five-hour or weekly cap it returns ``deferred`` for that role and the
next research run picks up where it left off.
"""

from __future__ import annotations

import json
import shlex
import subprocess
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
CURRICULUM_PLAN_DELTA_FILTERED = ".aicg/curriculum-plan-delta-filtered.json"
JOB_REQUIREMENTS_JSON = ".aicg/job-requirements.json"
JOB_REQUIREMENTS_MD = "JOB_REQUIREMENTS.md"

DEFAULT_CAPS = {
    "max_modules_per_run": 1,
    "max_exercises_per_run": 3,
    "max_projects_per_run": 0,
}
DEFAULT_MIN_EVIDENCE_COUNT = 3


@dataclass(frozen=True)
class ResearchCaps:
    max_modules: int
    max_exercises: int
    max_projects: int
    min_evidence_count: int

    @classmethod
    def from_manifest(cls, manifest: OrgManifest) -> "ResearchCaps":
        research = manifest.research or {}
        caps = research.get("caps") or {}
        return cls(
            max_modules=int(caps.get("max_modules_per_run", DEFAULT_CAPS["max_modules_per_run"])),
            max_exercises=int(caps.get("max_exercises_per_run", DEFAULT_CAPS["max_exercises_per_run"])),
            max_projects=int(caps.get("max_projects_per_run", DEFAULT_CAPS["max_projects_per_run"])),
            min_evidence_count=int(
                research.get("min_evidence_count", DEFAULT_MIN_EVIDENCE_COUNT)
            ),
        )


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
    open_pr: bool = True,
) -> dict[str, Any]:
    """Process every role's research packet via the configured agent.

    Per role, this:
    1. Invokes the configured research agent on the prompt packet.
    2. Validates the resulting curriculum-plan-delta against the
       manifest's caps and evidence threshold.
    3. Writes a human-readable RESEARCH_PROPOSAL_<month>.md plus the
       filtered delta into a branch in the learning repo.
    4. Opens a GitHub PR (``open_pr=True``, default) requesting human
       approval. The runner never merges curriculum-plan.json
       autonomously.

    Set ``open_pr=False`` for dry-run / tests / first inspection.
    """
    state_root = state_dir_for_manifest(manifest, state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    month = month or date.today().strftime("%Y-%m")
    prompt_dir = state_root / "research" / month
    if not prompt_dir.exists():
        raise ResearchError(
            f"No research packets at {prompt_dir}. Run `aicg org research` first."
        )

    config = ResearchAgentConfig.from_manifest(manifest)
    caps = ResearchCaps.from_manifest(manifest)
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

        outputs = _detect_outputs(learning_path)
        if not outputs["delta_present"]:
            role_reports.append(
                {
                    "role": role.id,
                    "status": "applied_no_delta",
                    "outputs": outputs,
                    "note": "Agent updated JOB_REQUIREMENTS only; no curriculum proposal.",
                }
            )
            continue

        delta_path = learning_path / CURRICULUM_PLAN_DELTA_FILE
        validation = validate_delta_against_caps(delta_path, caps)
        proposal = write_proposal(
            learning_path=learning_path,
            role=role,
            month=month,
            validation=validation,
            caps=caps,
        )
        pr_outcome: dict[str, Any] | None = None
        if open_pr and (validation["accepted"]["modules"]
                        or validation["accepted"]["exercises"]
                        or validation["accepted"]["projects"]
                        or validation["rejected"]):
            pr_outcome = _open_proposal_pr(
                learning_path=learning_path,
                role=role,
                month=month,
                proposal_paths=proposal["written"],
            )

        role_reports.append(
            {
                "role": role.id,
                "status": "proposal_ready",
                "outputs": outputs,
                "validation": _validation_summary(validation),
                "proposal_files": proposal["written"],
                "pr": pr_outcome,
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "research_apply",
        "month": month,
        "caps": {
            "max_modules_per_run": caps.max_modules,
            "max_exercises_per_run": caps.max_exercises,
            "max_projects_per_run": caps.max_projects,
            "min_evidence_count": caps.min_evidence_count,
        },
        "roles": role_reports,
    }
    write_json(state_root / RESEARCH_APPLY_REPORT, report)
    return report


def validate_delta_against_caps(
    delta_path: Path, caps: ResearchCaps
) -> dict[str, Any]:
    """Apply per-run caps and evidence threshold to a delta.

    Returns ``{"accepted": {...}, "rejected": [...], "rationale": str}``.
    Items beyond the cap, or below the evidence threshold, are moved to
    ``rejected`` with a reason. Order matters: the highest-quality items
    (by evidence count) are accepted first.
    """
    if not delta_path.exists():
        return _empty_validation("delta file not present")
    try:
        delta = json.loads(delta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _empty_validation(f"delta JSON invalid: {exc}")

    rationale = str(delta.get("rationale", "")).strip()
    raw_modules = delta.get("modules", []) or []
    raw_exercises = delta.get("exercises", []) or []
    raw_projects = delta.get("projects", []) or []

    accepted_modules, rejected = _filter_with_caps(
        raw_modules,
        cap=caps.max_modules,
        kind="module",
        min_evidence=caps.min_evidence_count,
    )
    accepted_exercises, rejected_ex = _filter_with_caps(
        raw_exercises,
        cap=caps.max_exercises,
        kind="exercise",
        min_evidence=caps.min_evidence_count,
    )
    accepted_projects, rejected_pj = _filter_with_caps(
        raw_projects,
        cap=caps.max_projects,
        kind="project",
        min_evidence=caps.min_evidence_count,
    )
    rejected.extend(rejected_ex)
    rejected.extend(rejected_pj)

    return {
        "rationale": rationale,
        "accepted": {
            "modules": accepted_modules,
            "exercises": accepted_exercises,
            "projects": accepted_projects,
        },
        "rejected": rejected,
        "input_counts": {
            "modules": len(raw_modules),
            "exercises": len(raw_exercises),
            "projects": len(raw_projects),
        },
    }


def _empty_validation(reason: str) -> dict[str, Any]:
    return {
        "rationale": "",
        "accepted": {"modules": [], "exercises": [], "projects": []},
        "rejected": [],
        "input_counts": {"modules": 0, "exercises": 0, "projects": 0},
        "note": reason,
    }


def _filter_with_caps(
    items: list[dict[str, Any]],
    cap: int,
    kind: str,
    min_evidence: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") or item.get("postings") or []
        if not isinstance(evidence, list):
            evidence = []
        evidence_count = len(evidence)
        if evidence_count < min_evidence:
            rejected.append(
                {
                    "kind": kind,
                    "id": str(item.get("id") or item.get("slug") or "?"),
                    "reason": (
                        f"evidence_below_threshold: {evidence_count} "
                        f"< required {min_evidence}"
                    ),
                    "item": item,
                }
            )
            continue
        qualified.append((evidence_count, item))

    # Highest evidence count first.
    qualified.sort(key=lambda pair: pair[0], reverse=True)
    accepted = [item for _, item in qualified[:cap]]
    for _, item in qualified[cap:]:
        rejected.append(
            {
                "kind": kind,
                "id": str(item.get("id") or item.get("slug") or "?"),
                "reason": f"cap_exceeded: cap={cap}",
                "item": item,
            }
        )
    return accepted, rejected


def write_proposal(
    learning_path: Path,
    role: RoleConfig,
    month: str,
    validation: dict[str, Any],
    caps: ResearchCaps,
) -> dict[str, Any]:
    """Materialize the proposal as files inside the learning repo.

    Returns ``{"written": [<paths>]}``. Idempotent: rerunning overwrites
    the same proposal artifacts so a re-opened PR shows the latest.
    """
    md_path = learning_path / f"RESEARCH_PROPOSAL_{month}.md"
    filtered_path = learning_path / CURRICULUM_PLAN_DELTA_FILTERED
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "role_id": role.id,
        "month": month,
        "rationale": validation["rationale"],
        "modules": validation["accepted"]["modules"],
        "exercises": validation["accepted"]["exercises"],
        "projects": validation["accepted"]["projects"],
    }
    filtered_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_path.write_text(_render_proposal_markdown(role, month, validation, caps), encoding="utf-8")
    return {"written": [str(md_path), str(filtered_path)]}


def _render_proposal_markdown(
    role: RoleConfig,
    month: str,
    validation: dict[str, Any],
    caps: ResearchCaps,
) -> str:
    accepted = validation["accepted"]
    rejected = validation["rejected"]
    lines: list[str] = [
        f"# Curriculum proposal: {role.title} — {month}",
        "",
        "Generated by `aicg org research --apply`. Caps:",
        "",
        f"- `max_modules_per_run`: {caps.max_modules}",
        f"- `max_exercises_per_run`: {caps.max_exercises}",
        f"- `max_projects_per_run`: {caps.max_projects}",
        f"- `min_evidence_count`: {caps.min_evidence_count} job postings per item",
        "",
        "## Rationale",
        "",
        validation["rationale"] or "_No rationale supplied by the research agent._",
        "",
        "## Accepted",
        "",
    ]
    for kind, items in (
        ("Modules", accepted["modules"]),
        ("Exercises", accepted["exercises"]),
        ("Projects", accepted["projects"]),
    ):
        lines.append(f"### {kind}")
        lines.append("")
        if not items:
            lines.append("_None._")
            lines.append("")
            continue
        for item in items:
            ident = item.get("id") or item.get("slug") or "?"
            title = item.get("title") or ""
            evidence = item.get("evidence") or item.get("postings") or []
            lines.append(f"- **`{ident}`** — {title}  ({len(evidence)} citations)")
            rationale = (item.get("rationale") or "").strip()
            if rationale:
                lines.append(f"  - rationale: {rationale}")
        lines.append("")

    lines.extend(["## Rejected", ""])
    if not rejected:
        lines.append("_None._")
    else:
        for entry in rejected:
            lines.append(
                f"- `{entry['kind']}` `{entry['id']}` — {entry['reason']}"
            )
    lines.append("")
    lines.extend(
        [
            "---",
            "",
            "## Approval",
            "",
            "**This proposal does NOT add anything to the curriculum automatically.**",
            "",
            "If you approve some/all of the accepted items, merge this PR. After merge,",
            "run `aicg org promote-plan --role <role>` (or wait for the next weekly",
            "audit) to apply the filtered delta to `curriculum-plan.json` and scaffold",
            "skeletons.",
            "",
            "If you reject everything, close the PR. No further action needed; the",
            "next research run will repropose anything still warranted by the job",
            "market.",
            "",
        ]
    )
    return "\n".join(lines)


def _validation_summary(validation: dict[str, Any]) -> dict[str, Any]:
    accepted = validation["accepted"]
    return {
        "rationale_present": bool(validation["rationale"]),
        "accepted_counts": {
            "modules": len(accepted["modules"]),
            "exercises": len(accepted["exercises"]),
            "projects": len(accepted["projects"]),
        },
        "rejected_count": len(validation["rejected"]),
        "input_counts": validation["input_counts"],
    }


def _open_proposal_pr(
    learning_path: Path,
    role: RoleConfig,
    month: str,
    proposal_paths: list[str],
) -> dict[str, Any]:
    """Push a branch with the proposal files and `gh pr create` it.

    Failures are reported but do not raise — the proposal files still
    exist locally so the human can open the PR manually.
    """
    branch = f"aicg/research/{month}/{role.id}"
    title = f"[aicg] curriculum proposal: {role.title} — {month}"
    body_path = learning_path / f"RESEARCH_PROPOSAL_{month}.md"

    commands: list[tuple[str, list[str]]] = [
        ("checkout", ["git", "-C", str(learning_path), "checkout", "-B", branch]),
        ("add", ["git", "-C", str(learning_path), "add", *proposal_paths]),
        (
            "commit",
            [
                "git",
                "-C",
                str(learning_path),
                "commit",
                "-m",
                f"aicg: curriculum proposal {month} ({role.id})",
            ],
        ),
        (
            "push",
            ["git", "-C", str(learning_path), "push", "-u", "origin", branch],
        ),
        (
            "pr_create",
            [
                "gh",
                "pr",
                "create",
                "--title",
                title,
                "--body-file",
                str(body_path),
                "--label",
                "aicg",
                "--label",
                "aicg:plan-proposal",
            ],
        ),
    ]
    outcomes: list[dict[str, Any]] = []
    for label, cmd in commands:
        completed = subprocess.run(
            cmd,
            cwd=learning_path,
            capture_output=True,
            text=True,
            check=False,
        )
        outcomes.append(
            {
                "step": label,
                "command": shlex.join(cmd),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-400:],
                "stderr_tail": completed.stderr[-400:],
            }
        )
        if completed.returncode != 0 and label != "commit":
            # `git commit` returning non-zero is fine if nothing changed.
            break
    # Restore main so subsequent agents start from a clean tree.
    subprocess.run(
        ["git", "-C", str(learning_path), "checkout", "main"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"branch": branch, "steps": outcomes}


def promote_plan(learning_path: Path) -> dict[str, Any]:
    """Apply a human-approved proposal to curriculum-plan.json.

    Reads ``.aicg/curriculum-plan-delta-filtered.json`` (written by
    :func:`write_proposal` and committed when a human merges the
    proposal PR) and merges it into ``curriculum-plan.json`` via
    :func:`merge_curriculum_plan_delta`. After a successful promotion
    the filtered file is renamed with a ``.promoted-<UTC>`` suffix so
    it doesn't get re-applied on the next pass.

    Returns a report describing what was promoted; raises
    ``ResearchError`` if no filtered proposal exists.
    """
    filtered_path = learning_path / CURRICULUM_PLAN_DELTA_FILTERED
    if not filtered_path.exists():
        raise ResearchError(
            f"No filtered proposal at {filtered_path}; run "
            "`aicg org research --apply` first and merge the PR."
        )

    # The filtered file has the same shape as a delta, so the existing
    # merger handles it. We temporarily write it as the delta input.
    plan_delta_path = learning_path / CURRICULUM_PLAN_DELTA_FILE
    plan_delta_path.parent.mkdir(parents=True, exist_ok=True)
    plan_delta_path.write_text(
        filtered_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    merge_report = merge_curriculum_plan_delta(learning_path)

    # Archive the consumed proposal so re-running promote is a no-op.
    stamp = utc_now().replace(":", "").replace("-", "")
    archived = filtered_path.with_suffix(f".promoted-{stamp}.json")
    filtered_path.rename(archived)
    # Also archive the now-merged delta file.
    if plan_delta_path.exists():
        plan_delta_path.rename(plan_delta_path.with_suffix(f".promoted-{stamp}.json"))

    return {
        "schema_version": 1,
        "operation": "promote_plan",
        "learning_path": str(learning_path),
        "archived_proposal": str(archived),
        "merge_report": merge_report,
    }


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
