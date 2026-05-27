from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import write_minimal_manifest

from aicg.agent_cli import AgentCommandResult
from aicg.org_config import load_manifest
from aicg.research import (
    ResearchAgentConfig,
    ResearchError,
    merge_curriculum_plan_delta,
    research_apply,
)


def _setup(tmp_path: Path) -> tuple[Path, Path, "OrgManifest"]:
    workspace = tmp_path / "workspace"
    (workspace / "ai-infra-security-learning").mkdir(parents=True)
    (workspace / "ai-infra-security-solutions").mkdir(parents=True)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    return workspace, tmp_path / "state", manifest


def _write_prompt(state_dir: Path, role_id: str, month: str = "2026-05") -> Path:
    packet_dir = state_dir / "research" / month
    packet_dir.mkdir(parents=True, exist_ok=True)
    path = packet_dir / f"{role_id}.md"
    path.write_text("# packet\n", encoding="utf-8")
    return path


def _enable_research_agent(manifest_path: Path, command: str) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["research"]["agent"] = {
        "provider": "anthropic",
        "model": "claude-opus-4.7",
        "interface": "local_cli_subscription",
        "agent_command": command,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_research_agent_config_falls_back_to_content_generation(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    config = ResearchAgentConfig.from_manifest(manifest)
    # The minimal manifest has no research.agent block; it should fall
    # through to content_generation.agent's agent_command.
    assert config.enabled is True
    assert "run-claude-content.sh" in (config.agent_command or "")


def test_research_apply_raises_when_no_packets(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup(tmp_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ResearchError):
        research_apply(manifest, workspace, month="2026-05", state_dir=state_dir)


def test_research_apply_skips_role_when_prompt_missing(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup(tmp_path)
    # Create the month dir but no role prompt.
    (state_dir / "research" / "2026-05").mkdir(parents=True)
    report = research_apply(manifest, workspace, month="2026-05", state_dir=state_dir)
    assert report["roles"][0]["status"] == "prompt_missing"


def test_research_apply_records_applied_when_agent_succeeds(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup(tmp_path)
    _write_prompt(state_dir, "security")
    learning = workspace / "ai-infra-security-learning"
    fake_result = AgentCommandResult(
        command="fake",
        returncode=0,
        stdout="",
        stderr="",
        limit_reached=False,
        retry_after=None,
        limit_scope=None,
    )

    def _fake_agent(command: str, cwd: Path) -> AgentCommandResult:
        # Simulate what the agent would do: write the three contract files.
        (learning / "JOB_REQUIREMENTS.md").write_text("# req\n", encoding="utf-8")
        (learning / ".aicg").mkdir(parents=True, exist_ok=True)
        (learning / ".aicg/job-requirements.json").write_text(
            json.dumps({"requirements": []}), encoding="utf-8"
        )
        (learning / ".aicg/curriculum-plan-delta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "role_id": "security",
                    "month": "2026-05",
                    "rationale": "Adding ML supply-chain content.",
                    "modules": [
                        {"id": "mod-099-supply-chain", "title": "Supply Chain Security"}
                    ],
                    "exercises": [],
                    "projects": [
                        {"id": "project-99-attestation", "title": "Build SLSA-3 pipeline"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        return fake_result

    with patch("aicg.research.run_agent_command", side_effect=_fake_agent):
        report = research_apply(
            manifest, workspace, month="2026-05", state_dir=state_dir
        )

    role_report = report["roles"][0]
    assert role_report["status"] == "applied"
    assert role_report["outputs"]["job_requirements_md"] is True
    assert role_report["outputs"]["delta_present"] is True
    # Plan file should have been created with the new module + project.
    plan = json.loads((learning / "curriculum-plan.json").read_text())
    assert any(m.get("id") == "mod-099-supply-chain" for m in plan["modules"])
    assert any(p.get("id") == "project-99-attestation" for p in plan["projects"])


def test_research_apply_records_deferred_on_subscription_limit(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup(tmp_path)
    _write_prompt(state_dir, "security")
    limit_result = AgentCommandResult(
        command="fake",
        returncode=1,
        stdout="usage limit",
        stderr="",
        limit_reached=True,
        retry_after="2026-05-27T20:00:00Z",
        limit_scope="five_hour",
    )
    with patch("aicg.research.run_agent_command", return_value=limit_result):
        report = research_apply(
            manifest, workspace, month="2026-05", state_dir=state_dir
        )
    assert report["roles"][0]["status"] == "deferred"
    assert report["roles"][0]["retry_after"] == "2026-05-27T20:00:00Z"


def test_research_apply_records_failure_on_nonzero_exit(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup(tmp_path)
    _write_prompt(state_dir, "security")
    fail_result = AgentCommandResult(
        command="fake",
        returncode=2,
        stdout="",
        stderr="agent crashed",
        limit_reached=False,
        retry_after=None,
        limit_scope=None,
    )
    with patch("aicg.research.run_agent_command", return_value=fail_result):
        report = research_apply(
            manifest, workspace, month="2026-05", state_dir=state_dir
        )
    assert report["roles"][0]["status"] == "agent_failed"
    assert report["roles"][0]["returncode"] == 2


def test_merge_delta_creates_plan_when_missing(tmp_path: Path) -> None:
    learning = tmp_path / "learning"
    (learning / ".aicg").mkdir(parents=True)
    (learning / ".aicg/curriculum-plan-delta.json").write_text(
        json.dumps(
            {
                "modules": [{"id": "mod-001", "title": "Foundations"}],
                "projects": [{"id": "project-1", "title": "Capstone"}],
            }
        ),
        encoding="utf-8",
    )
    report = merge_curriculum_plan_delta(learning)
    assert report["present"] is True
    assert {"kind": "module", "id": "mod-001"} in report["added"]
    plan = json.loads((learning / "curriculum-plan.json").read_text())
    assert plan["modules"][0]["id"] == "mod-001"
    assert plan["projects"][0]["id"] == "project-1"


def test_merge_delta_is_idempotent(tmp_path: Path) -> None:
    learning = tmp_path / "learning"
    (learning / ".aicg").mkdir(parents=True)
    (learning / "curriculum-plan.json").write_text(
        json.dumps({"modules": [{"id": "mod-001"}], "projects": []}),
        encoding="utf-8",
    )
    (learning / ".aicg/curriculum-plan-delta.json").write_text(
        json.dumps({"modules": [{"id": "mod-001"}, {"id": "mod-002"}]}),
        encoding="utf-8",
    )

    first = merge_curriculum_plan_delta(learning)
    second = merge_curriculum_plan_delta(learning)

    assert any(a["id"] == "mod-002" for a in first["added"])
    assert second["added"] == []
    assert {"kind": "module", "id": "mod-001"} in second["skipped"]
    assert {"kind": "module", "id": "mod-002"} in second["skipped"]


def test_merge_delta_adds_exercise_under_existing_module(tmp_path: Path) -> None:
    learning = tmp_path / "learning"
    (learning / ".aicg").mkdir(parents=True)
    (learning / "curriculum-plan.json").write_text(
        json.dumps(
            {"modules": [{"id": "mod-001", "exercises": []}], "projects": []}
        ),
        encoding="utf-8",
    )
    (learning / ".aicg/curriculum-plan-delta.json").write_text(
        json.dumps(
            {
                "exercises": [
                    {
                        "module_id": "mod-001",
                        "exercise": {"slug": "exercise-99-adversarial", "title": "Adversarial"},
                    },
                    # Orphan exercise — parent does not exist; should be skipped.
                    {
                        "module_id": "mod-999",
                        "exercise": {"slug": "exercise-x", "title": "Orphan"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = merge_curriculum_plan_delta(learning)
    plan = json.loads((learning / "curriculum-plan.json").read_text())
    mod_001 = next(m for m in plan["modules"] if m["id"] == "mod-001")
    assert any(ex["slug"] == "exercise-99-adversarial" for ex in mod_001["exercises"])
    assert any(s["id"] == "exercise-x" for s in report["skipped"])


def test_merge_delta_returns_present_false_when_no_delta(tmp_path: Path) -> None:
    report = merge_curriculum_plan_delta(tmp_path / "learning")
    assert report["present"] is False
    assert report["added"] == []
