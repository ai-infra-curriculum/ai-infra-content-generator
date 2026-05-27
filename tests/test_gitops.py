from __future__ import annotations

from aicg.gitops import is_aicg_state_path


def test_aicg_state_paths_are_not_pr_content():
    assert is_aicg_state_path(".aicg")
    assert is_aicg_state_path(".aicg/audit-report.json")
    assert not is_aicg_state_path("modules/mod-001/exercise-01/SOLUTION.md")
