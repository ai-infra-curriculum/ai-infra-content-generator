from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_minimal_manifest

from aicg.bootstrap import BootstrapError, bootstrap_role
from aicg.org_config import load_manifest


def test_bootstrap_creates_paired_repos(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"

    report = bootstrap_role(
        manifest=manifest,
        workspace=workspace,
        role_id="data-engineer",
        title="Data Engineer",
        level=25,
        state_dir=state_dir,
    )

    learning = workspace / "ai-infra-data-engineer-learning"
    solutions = workspace / "ai-infra-data-engineer-solutions"

    assert learning.is_dir()
    assert solutions.is_dir()
    for required in (
        "README.md",
        "LICENSE",
        ".gitignore",
        "CURRICULUM.md",
        "PREREQUISITES.md",
        "VERSIONS.md",
        "lessons/README.md",
        "projects/README.md",
        ".github/workflows/ci.yml",
    ):
        assert (learning / required).exists(), f"missing {required} in learning"
    for required in (
        "README.md",
        "LICENSE",
        ".gitignore",
        "SOLUTIONS_INDEX.md",
        "modules/README.md",
        "projects/README.md",
        ".github/workflows/ci.yml",
    ):
        assert (solutions / required).exists(), f"missing {required} in solutions"

    prompt = report["plan"]["prompt_path"]
    text = Path(prompt).read_text(encoding="utf-8")
    assert "Phase 1 — Research" in text
    assert "data-engineer" in text
    assert "curriculum-plan.json" in text


def test_bootstrap_refuses_existing_directory(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    workspace = tmp_path / "workspace"
    (workspace / "ai-infra-data-engineer-learning").mkdir(parents=True)
    (workspace / "ai-infra-data-engineer-learning" / "existing.md").write_text(
        "do not clobber\n", encoding="utf-8"
    )

    with pytest.raises(BootstrapError) as excinfo:
        bootstrap_role(
            manifest=manifest,
            workspace=workspace,
            role_id="data-engineer",
            title="Data Engineer",
            level=25,
        )
    assert "already exists" in str(excinfo.value)


def test_bootstrap_overwrite_keeps_existing_unrelated_files(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    workspace = tmp_path / "workspace"
    learning = workspace / "ai-infra-data-engineer-learning"
    learning.mkdir(parents=True)
    legacy = learning / "legacy.md"
    legacy.write_text("legacy note\n", encoding="utf-8")

    bootstrap_role(
        manifest=manifest,
        workspace=workspace,
        role_id="data-engineer",
        title="Data Engineer",
        level=25,
        overwrite=True,
    )

    assert legacy.read_text(encoding="utf-8") == "legacy note\n"
    assert (learning / "README.md").exists()


def test_bootstrap_rejects_invalid_role_id(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    workspace = tmp_path / "workspace"
    with pytest.raises(BootstrapError):
        bootstrap_role(
            manifest=manifest,
            workspace=workspace,
            role_id="Bad ID!",
            title="Bad",
            level=10,
        )


def test_bootstrap_appends_role_to_json_manifest(tmp_path: Path) -> None:
    manifest_path = write_minimal_manifest(tmp_path / "aicg-org.yaml")
    manifest = load_manifest(manifest_path)
    workspace = tmp_path / "workspace"

    bootstrap_role(
        manifest=manifest,
        workspace=workspace,
        role_id="data-engineer",
        title="Data Engineer",
        level=25,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_ids = [role["id"] for role in payload["roles"]]
    assert "data-engineer" in role_ids
    new_role = next(role for role in payload["roles"] if role["id"] == "data-engineer")
    assert new_role["learning_repo"] == "ai-infra-data-engineer-learning"
    assert new_role["solution_repo"] == "ai-infra-data-engineer-solutions"
    assert new_role["level"] == 25


def test_bootstrap_no_update_manifest_leaves_file_unchanged(tmp_path: Path) -> None:
    manifest_path = write_minimal_manifest(tmp_path / "aicg-org.yaml")
    original = manifest_path.read_text(encoding="utf-8")
    manifest = load_manifest(manifest_path)
    workspace = tmp_path / "workspace"

    bootstrap_role(
        manifest=manifest,
        workspace=workspace,
        role_id="data-engineer",
        title="Data Engineer",
        level=25,
        write_manifest=False,
    )

    assert manifest_path.read_text(encoding="utf-8") == original


def test_bootstrap_report_written_to_state_dir(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"

    bootstrap_role(
        manifest=manifest,
        workspace=workspace,
        role_id="data-engineer",
        title="Data Engineer",
        level=25,
        state_dir=state_dir,
    )

    report = state_dir / "bootstrap-report.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["operation"] == "bootstrap_role"
    assert payload["plan"]["role_id"] == "data-engineer"
