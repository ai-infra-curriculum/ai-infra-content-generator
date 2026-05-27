from __future__ import annotations

from conftest import make_security_workspace

from aicg.inventory import WorkspaceInventory


def test_discovers_paired_learning_and_solutions_repos(tmp_path):
    workspace = make_security_workspace(tmp_path)

    inventory = WorkspaceInventory(workspace)
    repos = inventory.by_name()
    solutions = repos["ai-infra-security-solutions"]
    learning = repos["ai-infra-security-learning"]

    assert solutions.kind == "solutions"
    assert learning.kind == "learning"
    assert inventory.paired_repo(solutions) == learning
    assert inventory.paired_repo(learning) == solutions
