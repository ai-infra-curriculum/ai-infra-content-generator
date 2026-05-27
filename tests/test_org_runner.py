from __future__ import annotations

import subprocess
from datetime import date

from conftest import make_security_workspace, write_minimal_manifest

from aicg.cli import main
from aicg.org_config import load_manifest
from aicg.org_runner import (
    generate_research_packets,
    is_git_dirty,
    plan_monthly_release,
    release_tag,
    run_org_audit,
    sync_repositories,
)


def test_manifest_loads_role_and_agent_policy(tmp_path):
    manifest_path = write_minimal_manifest(tmp_path / "aicg-org.yaml")

    manifest = load_manifest(manifest_path)

    assert manifest.repo_names == [
        "ai-infra-security-learning",
        "ai-infra-security-solutions",
        ".github",
    ]
    assert manifest.release_repo_names == [
        "ai-infra-security-learning",
        "ai-infra-security-solutions",
    ]
    assert manifest.automation["agent"]["model"] == "codex-gpt-5.5"
    assert manifest.automation["agent"]["interface"] == "local_cli_subscription"
    assert manifest.content_generation["agent"]["model"] == "claude-opus-4.7"
    assert manifest.content_generation["agent"]["interface"] == "local_cli_subscription"


def test_release_tag_uses_date_based_versioning(tmp_path):
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    assert release_tag(manifest, date(2026, 5, 27)) == "v2026.05"


def test_sync_repositories_dry_run_plans_clones(tmp_path):
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    workspace = tmp_path / "workspace"

    report = sync_repositories(manifest, workspace, dry_run=True)

    assert len(report["actions"]) == 3
    assert all(action["status"] == "planned" for action in report["actions"])
    assert "git clone" in report["actions"][0]["command"]


def test_monthly_release_dry_run_reports_missing_repos(tmp_path):
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = plan_monthly_release(manifest, tmp_path / "workspace", today=date(2026, 5, 27))

    assert report["tag"] == "v2026.05"
    assert len(report["actions"]) == 2
    assert {action["status"] for action in report["actions"]} == {"missing"}


def test_research_packets_include_claude_policy(tmp_path):
    workspace = make_security_workspace(tmp_path)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    report = generate_research_packets(
        manifest,
        workspace,
        month="2026-05",
        state_dir=tmp_path / "state",
    )

    prompt = report["packets"][0]["prompt_path"]
    assert report["packets"][0]["role"] == "security"
    with open(prompt, encoding="utf-8") as prompt_file:
        assert "claude-opus-4.7" in prompt_file.read()


def test_org_audit_writes_queue_from_solution_gaps(tmp_path):
    workspace = make_security_workspace(tmp_path)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))

    queue = run_org_audit(manifest, workspace, state_dir=tmp_path / "state")

    assert queue["work_item_count"] == 1
    assert queue["work_items"][0]["repo"] == "ai-infra-security-solutions"
    assert queue["work_items"][0]["status"] == "ready"


def test_cli_org_research(tmp_path):
    workspace = make_security_workspace(tmp_path)
    manifest_path = write_minimal_manifest(tmp_path / "aicg-org.yaml")
    state_dir = tmp_path / "state"

    rc = main(
        [
            "org",
            "research",
            "--workspace",
            str(workspace),
            "--manifest",
            str(manifest_path),
            "--state-dir",
            str(state_dir),
            "--month",
            "2026-05",
        ]
    )

    assert rc == 0
    assert (state_dir / "job-research-plan.json").exists()


def test_aicg_state_does_not_make_repo_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / ".aicg").mkdir()
    (repo / ".aicg" / "audit-report.json").write_text("{}\n", encoding="utf-8")

    assert not is_git_dirty(repo)


def test_daily_remediation_verifies_after_generate(tmp_path):
    import textwrap

    from aicg.org_runner import run_daily_remediation

    workspace = make_security_workspace(tmp_path)
    solutions = workspace / "ai-infra-security-solutions"
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    state_dir = tmp_path / "state"

    # Configure a generator script that produces a complete SOLUTION.md.
    generator = tmp_path / "fake-claude.sh"
    target_file = (
        solutions
        / "modules"
        / "mod-001-ml-security-foundations"
        / "exercise-01-threat-model-a-small-ml-system"
        / "SOLUTION.md"
    )
    body = textwrap.dedent(
        """\
        # Threat Model Solution

        ## Overview
        Brief walkthrough.

        ## Implementation
        Build it.

        ## Validation
        Run tests.

        ## Rubric
        Graded.

        ## Common mistakes
        Listed.

        ## References
        - https://www.nist.gov/itl/ai-risk-management-framework
        """
    )
    target_file.parent.mkdir(parents=True, exist_ok=True)
    generator.write_text(
        "#!/usr/bin/env bash\n"
        f'cat > "{target_file}" <<EOF\n{body}EOF\n',
        encoding="utf-8",
    )
    generator.chmod(0o755)

    # Override the manifest's configured agent command for this run.
    import aicg.org_runner as org_runner

    original = org_runner.content_generation_command
    org_runner.content_generation_command = lambda _m: str(generator)
    try:
        # First run org audit to populate the work queue.
        org_runner.run_org_audit(manifest, workspace, state_dir=state_dir)
        result = run_daily_remediation(manifest, workspace, state_dir=state_dir)
    finally:
        org_runner.content_generation_command = original

    assert result["status"] == "generated"
    assert result.get("verify", {}).get("status") == "verified"
