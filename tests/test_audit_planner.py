from __future__ import annotations

from conftest import make_security_workspace, write_file

from aicg.audit import audit_repo, scan_placeholders
from aicg.planner import plan_from_audit


def test_audit_detects_missing_module_level_solution_parity(tmp_path):
    workspace = make_security_workspace(tmp_path)

    report = audit_repo(workspace, "ai-infra-security-solutions", write_report=False)

    assert report["summary"]["status"] == "fail"
    assert report["modules"][0]["module_id"] == "mod-001-ml-security-foundations"
    assert report["modules"][0]["missing_solution_count"] == 1
    assert any(gap["type"] == "missing_solution_module" for gap in report["gaps"])
    assert any(gap["type"] == "missing_exercise_solution" for gap in report["gaps"])


def test_plan_groups_missing_exercise_solutions_by_module(tmp_path):
    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    report = audit_repo(workspace, "ai-infra-security-solutions", write_report=False)

    plan = plan_from_audit(report, repo_path=solutions)

    assert plan["work_item_count"] == 1
    item = plan["work_items"][0]
    assert item["id"] == "fill-mod-001-ml-security-foundations-solutions"
    assert item["module"] == "mod-001-ml-security-foundations"
    assert item["exercises"][0]["exercise_id"] == "exercise-01"
    assert item["actions"][-1]["path"].endswith("/exercise-01/SOLUTION.md")


def test_placeholder_scan_flags_manual_review_and_needs_research(tmp_path):
    repo = tmp_path / "repo"
    write_file(
        repo / "modules" / "mod-001" / "exercise-01" / "SOLUTION.md",
        "# Solution\n\n# manual-review\n\n<!-- needs-research: verify source -->\n",
    )

    findings = scan_placeholders(repo)

    assert {finding["type"] for finding in findings} >= {"manual_review", "needs_research"}
    assert all(finding["severity"] == "error" for finding in findings)
