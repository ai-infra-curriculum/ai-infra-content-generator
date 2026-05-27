from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_security_workspace

from aicg.propagate import PropagateError, propagate_repo


def _plan_with_verified_item(solutions_path: Path) -> dict:
    return {
        "schema_version": 2,
        "repo": {"name": "ai-infra-security-solutions", "path": str(solutions_path)},
        "work_items": [
            {
                "id": "fill-mod-001-ml-security-foundations-solutions",
                "type": "module_solution_gap",
                "repo": "ai-infra-security-solutions",
                "module": "mod-001-ml-security-foundations",
                "title": "Fill mod-001 module exercise solutions",
                "status": "verified",
                "exercises": [],
                "actions": [],
            }
        ],
    }


def _write_plan(solutions: Path, plan: dict) -> None:
    state_dir = solutions / ".aicg"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "work-plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )


def test_propagate_seeds_versions_when_file_missing(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    _write_plan(solutions, _plan_with_verified_item(solutions))

    report = propagate_repo(workspace, "ai-infra-security-solutions")

    assert report["status"] == "updated"
    versions = (solutions / "VERSIONS.md").read_text(encoding="utf-8")
    assert "fill-mod-001-ml-security-foundations-solutions" in versions
    assert "mod-001-ml-security-foundations" in versions
    assert "| Date | Work ID | Scope | Title |" in versions


def test_propagate_idempotent_on_second_run(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    _write_plan(solutions, _plan_with_verified_item(solutions))

    first = propagate_repo(workspace, "ai-infra-security-solutions")
    second = propagate_repo(workspace, "ai-infra-security-solutions")

    assert first["status"] == "updated"
    assert second["status"] == "noop"
    assert second["updated"] == []
    assert len(second["already_present"]) == 1


def test_propagate_appends_to_existing_month_block(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    versions = solutions / "VERSIONS.md"
    versions.parent.mkdir(parents=True, exist_ok=True)

    from datetime import date

    month_label = date.today().strftime("%Y-%m")
    versions.write_text(
        "# Versions\n\n"
        f"## {month_label}\n\n"
        "| Date | Work ID | Scope | Title |\n"
        "|---|---|---|---|\n"
        "| 2026-05-26 | `seed-row` | `mod-000` | Pre-existing |\n",
        encoding="utf-8",
    )
    _write_plan(solutions, _plan_with_verified_item(solutions))

    propagate_repo(workspace, "ai-infra-security-solutions")

    contents = versions.read_text(encoding="utf-8")
    # Pre-existing row preserved.
    assert "seed-row" in contents
    # New row appended.
    assert "fill-mod-001-ml-security-foundations-solutions" in contents


def test_propagate_skips_unverified_items(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    plan = _plan_with_verified_item(solutions)
    plan["work_items"][0]["status"] = "planned"
    _write_plan(solutions, plan)

    report = propagate_repo(workspace, "ai-infra-security-solutions")

    assert report["status"] == "no_items"
    assert not (solutions / "VERSIONS.md").exists()


def test_propagate_filters_by_work_id(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    plan = _plan_with_verified_item(solutions)
    plan["work_items"].append(
        {
            "id": "fill-mod-002-solutions",
            "type": "module_solution_gap",
            "repo": "ai-infra-security-solutions",
            "module": "mod-002-foo",
            "title": "Second module",
            "status": "verified",
            "exercises": [],
            "actions": [],
        }
    )
    _write_plan(solutions, plan)

    report = propagate_repo(
        workspace,
        "ai-infra-security-solutions",
        work_id="fill-mod-002-solutions",
    )

    assert len(report["updated"]) == 1
    assert report["updated"][0]["work_id"] == "fill-mod-002-solutions"
    versions = (solutions / "VERSIONS.md").read_text(encoding="utf-8")
    assert "fill-mod-002-solutions" in versions
    assert "fill-mod-001" not in versions


def test_propagate_raises_when_plan_missing(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    with pytest.raises(PropagateError):
        propagate_repo(workspace, "ai-infra-security-solutions")


def test_propagate_emits_curriculum_suggestions(tmp_path: Path) -> None:
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    _write_plan(solutions, _plan_with_verified_item(solutions))

    report = propagate_repo(workspace, "ai-infra-security-solutions")

    assert report["curriculum_suggestions"]
    suggestion = report["curriculum_suggestions"][0]
    assert suggestion["scope"] == "mod-001-ml-security-foundations"
    assert "CURRICULUM.md" in suggestion["suggestion"]
