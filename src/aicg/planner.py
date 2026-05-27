"""Deterministic work-plan generation from audit reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .state import utc_now, write_state

WORK_PLAN = "work-plan.json"


def plan_from_audit(audit_report: dict[str, Any], repo_path: Path | None = None) -> dict[str, Any]:
    work_items: list[dict[str, Any]] = []
    for module in audit_report.get("modules", []):
        missing = [
            exercise
            for exercise in module.get("exercises", [])
            if exercise.get("status") in {"missing_solution", "source_gap"}
        ]
        if not missing:
            continue

        module_id = module["module_id"]
        work_id = f"fill-{module_id}-solutions"
        actions = [
            {
                "type": "create_directory",
                "path": module.get("solution_path") or f"modules/{module_id}",
            }
        ]
        for exercise in missing:
            actions.append(
                {
                    "type": "write_solution",
                    "exercise_id": exercise["exercise_id"],
                    "source_exercise": exercise["learning_path"],
                    "path": f"modules/{module_id}/{exercise['exercise_id']}/SOLUTION.md",
                }
            )

        work_items.append(
            {
                "id": work_id,
                "type": "module_solution_gap",
                "repo": audit_report["repo"]["name"],
                "module": module_id,
                "title": f"Fill {module_id} module exercise solutions",
                "priority": len(work_items) + 1,
                "status": "planned",
                "why": (
                    "Learning exercises exist in the paired learning repo, but the solutions "
                    "repo is missing module-level reference solutions."
                ),
                "source_policy": {
                    "official_first": True,
                    "required_default_sources": recommended_source_ids(module_id),
                    "needs_research_blocks_merge": True,
                },
                "exercises": [
                    {
                        "exercise_id": exercise["exercise_id"],
                        "title": exercise.get("title", exercise["exercise_id"]),
                        "learning_path": exercise["learning_path"],
                        "expected_solution_dir": exercise["expected_solution_dir"],
                        "required_artifacts": exercise.get("required_artifacts", ["SOLUTION.md"]),
                    }
                    for exercise in missing
                ],
                "actions": actions,
            }
        )

    plan = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "repo": audit_report["repo"],
        "audit_summary": audit_report["summary"],
        "work_item_count": len(work_items),
        "work_items": work_items,
    }
    if repo_path is not None:
        write_state(repo_path, WORK_PLAN, plan)
    return plan


def recommended_source_ids(module_id: str) -> list[str]:
    sources = ["owasp-ml-top-10", "mitre-atlas", "nist-ai-rmf"]
    if "zero-trust" in module_id:
        sources.extend(["nist-sp-800-207", "kubernetes-docs"])
    if "supply-chain" in module_id:
        sources.extend(["slsa", "sigstore-docs", "openssf-scorecard"])
    if "compliance" in module_id or "governance" in module_id:
        sources.extend(["nist-ai-600-1", "veriswarm-trust-center"])
    return sources
