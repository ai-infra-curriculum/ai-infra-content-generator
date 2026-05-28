from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_file, write_minimal_manifest

from aicg.learning_audit import LearningAuditError, audit_learning_repo
from aicg.nav_audit import (
    CurriculumNavError,
    audit_curriculum_nav,
    audit_org_profile,
)
from aicg.org_config import load_manifest
from aicg.pairing_audit import audit_pairing


# ---------------------------------------------------------------------------
# Learning audit
# ---------------------------------------------------------------------------


def test_learning_audit_flags_missing_module_container(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-learning"
    repo.mkdir(parents=True)
    report = audit_learning_repo(repo)
    types = [g["type"] for g in report["gaps"]]
    assert "missing_module_container" in types
    assert report["summary"]["error_count"] >= 1


def test_learning_audit_flags_missing_readme(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-learning"
    (repo / "modules" / "mod-001").mkdir(parents=True)
    # No README, no exercises
    report = audit_learning_repo(repo)
    types = [g["type"] for g in report["gaps"]]
    assert "missing_module_readme" in types
    assert "missing_exercises" in types


def test_learning_audit_clean_when_complete(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-learning"
    module = repo / "modules" / "mod-001"
    write_file(
        module / "README.md",
        "# Mod 001\n\n" + "Content " * 50 + "\n",
    )
    write_file(module / "exercises" / "exercise-01-foo.md", "# Ex 01\n\n" + "x" * 300)
    report = audit_learning_repo(repo)
    assert report["summary"]["error_count"] == 0


def test_learning_audit_detects_placeholder_module(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-learning"
    module = repo / "modules" / "mod-001"
    write_file(module / "README.md", "TODO\nTBD\n")
    write_file(module / "exercises" / "exercise-01.md", "# Ex 01\n" + "x" * 300)
    report = audit_learning_repo(repo)
    types = [g["type"] for g in report["gaps"]]
    assert "placeholder_module_readme" in types


# ---------------------------------------------------------------------------
# Pairing audit
# ---------------------------------------------------------------------------


def test_pairing_audit_flags_module_only_in_learning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "ai-infra-security-learning" / "modules" / "mod-001").mkdir(parents=True)
    (workspace / "ai-infra-security-solutions" / "modules").mkdir(parents=True)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = audit_pairing(manifest, workspace, state_dir=tmp_path / "state")

    types = {f["type"] for f in report["findings"]}
    assert "module_only_in_learning" in types


def test_pairing_audit_flags_exercise_slug_drift(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    learning = workspace / "ai-infra-security-learning" / "modules" / "mod-001"
    solutions = workspace / "ai-infra-security-solutions" / "modules" / "mod-001"
    (learning / "exercises").mkdir(parents=True)
    (solutions / "exercise-01-different-slug").mkdir(parents=True)
    write_file(learning / "exercises" / "exercise-01-threat-model.md", "")
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = audit_pairing(manifest, workspace, state_dir=tmp_path / "state")

    types = {f["type"] for f in report["findings"]}
    assert "exercise_slug_drift" in types


def test_pairing_audit_clean_when_perfect_pair(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    learning = workspace / "ai-infra-security-learning" / "modules" / "mod-001"
    solutions = workspace / "ai-infra-security-solutions" / "modules" / "mod-001"
    (learning / "exercises").mkdir(parents=True)
    write_file(learning / "exercises" / "exercise-01-threat-model.md", "")
    (solutions / "exercise-01-threat-model").mkdir(parents=True)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = audit_pairing(manifest, workspace, state_dir=tmp_path / "state")

    # No findings for that role
    role_report = next(r for r in report["roles"] if r["role"] == "security")
    assert role_report.get("finding_count", 0) == 0


def test_pairing_audit_flags_project_only_in_learning(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "ai-infra-security-learning" / "projects" / "project-01-zero-trust").mkdir(parents=True)
    (workspace / "ai-infra-security-solutions" / "projects").mkdir(parents=True)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = audit_pairing(manifest, workspace, state_dir=tmp_path / "state")

    types = {f["type"] for f in report["findings"]}
    assert "project_only_in_learning" in types


# ---------------------------------------------------------------------------
# Curriculum-nav audit
# ---------------------------------------------------------------------------


def test_curriculum_nav_flags_orphan_module(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-solutions"
    (repo / "modules" / "mod-001").mkdir(parents=True)
    (repo / "modules" / "mod-002").mkdir(parents=True)
    write_file(repo / "CURRICULUM.md", "# Curriculum\n\n- `mod-001`\n")

    report = audit_curriculum_nav(repo)

    types = [g["type"] for g in report["findings"]]
    assert "nav_missing_reference" in types
    # mod-002 should be the orphan
    msgs = " ".join(g["message"] for g in report["findings"])
    assert "mod-002" in msgs


def test_curriculum_nav_flags_broken_reference(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-solutions"
    (repo / "modules" / "mod-001").mkdir(parents=True)
    write_file(
        repo / "CURRICULUM.md",
        "# Curriculum\n\n- `mod-001`\n- `mod-999`\n",
    )

    report = audit_curriculum_nav(repo)

    types = [g["type"] for g in report["findings"]]
    assert "nav_broken_reference" in types


def test_curriculum_nav_flags_missing_doc_when_modules_exist(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-solutions"
    (repo / "modules" / "mod-001").mkdir(parents=True)
    report = audit_curriculum_nav(repo)
    types = [g["type"] for g in report["findings"]]
    assert "missing_curriculum_md" in types


def test_curriculum_nav_clean_when_aligned(tmp_path: Path) -> None:
    repo = tmp_path / "ai-infra-engineer-solutions"
    (repo / "modules" / "mod-001").mkdir(parents=True)
    write_file(
        repo / "CURRICULUM.md",
        "# Curriculum\n\nSee `mod-001` in modules/mod-001.\n",
    )
    report = audit_curriculum_nav(repo)
    assert report["gap_count"] == 0


# ---------------------------------------------------------------------------
# Org-profile audit
# ---------------------------------------------------------------------------


def test_org_profile_flags_missing_repos(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".github" / "profile").mkdir(parents=True)
    # Empty profile README
    write_file(
        workspace / ".github" / "profile" / "README.md",
        "# AI Infra Curriculum\n\n" + "Welcome." * 30,
    )
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = audit_org_profile(manifest, workspace, state_dir=tmp_path / "state")

    types = [f["type"] for f in report["findings"]]
    assert "profile_missing_repos" in types


def test_org_profile_flags_thin_doc(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".github" / "profile").mkdir(parents=True)
    write_file(workspace / ".github" / "profile" / "README.md", "# Hi")
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = audit_org_profile(manifest, workspace, state_dir=tmp_path / "state")

    types = [f["type"] for f in report["findings"]]
    assert "profile_too_thin" in types


def test_org_profile_flags_missing_repo_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    report = audit_org_profile(manifest, workspace, state_dir=tmp_path / "state")
    types = [f["type"] for f in report["findings"]]
    assert "missing_profile_repo" in types


# ---------------------------------------------------------------------------
# Integration: run_org_audit splices the new types into the queue
# ---------------------------------------------------------------------------


def test_run_org_audit_includes_pairing_findings(tmp_path: Path) -> None:
    from aicg.org_runner import run_org_audit

    workspace = tmp_path / "workspace"
    # Solution + learning side both present but learning has an extra module
    (workspace / "ai-infra-security-learning" / "modules" / "mod-001").mkdir(parents=True)
    write_file(
        workspace / "ai-infra-security-learning" / "modules" / "mod-001" / "README.md",
        "# Mod 001\n\n" + "Content " * 50,
    )
    write_file(
        workspace
        / "ai-infra-security-learning"
        / "modules"
        / "mod-001"
        / "exercises"
        / "exercise-01-threat-model.md",
        "# Ex\n" + "x" * 400,
    )
    (workspace / "ai-infra-security-learning" / "modules" / "mod-002").mkdir(parents=True)
    write_file(
        workspace / "ai-infra-security-learning" / "modules" / "mod-002" / "README.md",
        "# Mod 002\n\n" + "Content " * 50,
    )
    write_file(
        workspace
        / "ai-infra-security-learning"
        / "modules"
        / "mod-002"
        / "exercises"
        / "exercise-01-foo.md",
        "# Ex\n" + "x" * 400,
    )
    (workspace / "ai-infra-security-solutions" / "modules" / "mod-001").mkdir(parents=True)
    write_file(workspace / "ai-infra-security-solutions" / "modules" / "README.md", "")
    write_file(
        workspace / "ai-infra-security-solutions" / ".github" / "workflows" / "ci.yml",
        "name: CI\n",
    )

    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    queue = run_org_audit(manifest, workspace, state_dir=tmp_path / "state")

    types = {item.get("type") for item in queue["work_items"]}
    # Pairing + learning + nav all run and at least pairing mismatch surfaces
    assert "pairing_mismatch" in types
