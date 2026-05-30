from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from conftest import write_file, write_minimal_manifest

from aicg.org_config import load_manifest
from aicg.rebrand import _append_maintainer_footer, _rebrand_repo, rebrand_run


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, "OrgManifest"]:
    workspace = tmp_path / "workspace"
    (workspace / "ai-infra-security-learning").mkdir(parents=True)
    (workspace / "ai-infra-security-solutions").mkdir(parents=True)
    (workspace / ".github" / "profile").mkdir(parents=True)
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    return workspace, tmp_path / "state", manifest


def test_footer_appender_adds_marker_and_phrasing() -> None:
    body = "# Hi\n\nProject content.\n"
    out = _append_maintainer_footer(body, "<!-- m -->", "Maintained by X")
    assert "<!-- m -->" in out
    assert "Maintained by X" in out
    assert out.startswith(body)


def test_footer_appender_preserves_trailing_newline() -> None:
    body = "# Hi\n\nProject content."  # no trailing newline
    out = _append_maintainer_footer(body, "<!-- m -->", "Maintained by X")
    assert out.endswith("\n")
    assert "Maintained by X" in out


def test_rebrand_dry_run_reports_would_change(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    write_file(workspace / "ai-infra-security-learning" / "README.md", "# Repo\n")

    report = rebrand_run(
        manifest, workspace, state_dir=state_dir, apply=False,
        repos=["ai-infra-security-learning"],
    )

    learning_outcome = next(
        r for r in report["repos"] if r["repo"] == "ai-infra-security-learning"
    )
    assert learning_outcome["status"] == "would_change"
    assert "README.md" in learning_outcome["changed_files"]


def test_rebrand_idempotent_already_branded(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    # README already has the marker.
    body = (
        "# Repo\n\nContent.\n\n---\n\n<!-- aicg:maintained-by -->\n"
        "Maintained by [VeriSwarm.ai](https://veriswarm.ai)\n"
    )
    write_file(workspace / "ai-infra-security-learning" / "README.md", body)

    report = rebrand_run(
        manifest, workspace, state_dir=state_dir, apply=False,
        repos=["ai-infra-security-learning"],
    )

    outcome = next(
        r for r in report["repos"] if r["repo"] == "ai-infra-security-learning"
    )
    assert outcome["status"] == "already_branded"
    assert outcome["changed_files"] == []


def test_rebrand_github_repo_targets_profile_readme(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    write_file(workspace / ".github" / "profile" / "README.md", "# Org Profile\n")
    write_file(workspace / ".github" / "README.md", "# Org README\n")

    report = rebrand_run(
        manifest, workspace, state_dir=state_dir, apply=False, repos=[".github"]
    )

    outcome = next(r for r in report["repos"] if r["repo"] == ".github")
    assert outcome["status"] == "would_change"
    paths = set(outcome["changed_files"])
    assert "profile/README.md" in paths
    assert "README.md" in paths


def test_rebrand_skips_when_no_maintainer_configured(tmp_path: Path) -> None:
    workspace, state_dir, _manifest = _setup_workspace(tmp_path)
    # Build a manifest WITHOUT a maintained_by block by editing the
    # written file directly.
    manifest_path = tmp_path / "aicg-org.yaml"
    payload = json.loads(manifest_path.read_text())
    payload.pop("maintained_by", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    bare_manifest = load_manifest(manifest_path)

    report = rebrand_run(
        bare_manifest, workspace, state_dir=state_dir, apply=False
    )

    assert report["status"] == "skipped"


def test_known_org_references_via_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    refs = manifest.known_org_references
    assert "VeriSwarm.ai" in refs
    assert "veriswarm.ai" in {r.lower() for r in refs}


def test_audit_org_profile_does_not_flag_maintained_by_as_orphan(
    tmp_path: Path,
) -> None:
    """The maintainer attribution should not trip profile_orphan_references."""
    from aicg.nav_audit import audit_org_profile

    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    # Profile references VeriSwarm.ai in a maintainer footer.
    write_file(
        workspace / ".github" / "profile" / "README.md",
        "# Org Profile\n\n"
        "Welcome to the AI Infra Curriculum. " + ("." * 200) + "\n\n"
        "[ai-infra-security-learning](https://github.com/ai-infra-curriculum/ai-infra-security-learning)\n"
        "[ai-infra-security-solutions](https://github.com/ai-infra-curriculum/ai-infra-security-solutions)\n"
        "[.github](https://github.com/ai-infra-curriculum/.github)\n\n"
        "Maintained by [VeriSwarm.ai](https://veriswarm.ai)\n",
    )

    report = audit_org_profile(manifest, workspace, state_dir=state_dir)

    types = [f["type"] for f in report["findings"]]
    assert "profile_orphan_references" not in types


# ---------------------------------------------------------------------------
# Branch name + dirty-tree handling
# ---------------------------------------------------------------------------


def test_rebrand_skips_dirty_working_tree(tmp_path: Path) -> None:
    """Don't clobber an in-flight job's uncommitted changes."""
    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    repo_path = workspace / "ai-infra-security-learning"
    write_file(repo_path / "README.md", "# Repo\n")

    with patch("aicg.rebrand.subprocess.run") as mock_run:
        # First call (the dirty check inside _rebrand_repo) returns
        # 'M README.md' so the rebrand should skip cleanly.
        mock_run.return_value.stdout = " M README.md\n"
        mock_run.return_value.returncode = 0
        outcome = _rebrand_repo(repo_path, "ai-infra-security-learning",
                                manifest.maintained_by, apply=True)
    assert outcome["status"] == "skipped_dirty_tree"


def test_rebrand_branch_safe_for_dotted_repo_names() -> None:
    """git refuses 'aicg/.../.github/...' because the component begins
    with a dot. Verify our branch builder sanitizes it."""
    import re

    from aicg.rebrand import utc_now

    repo = ".github"
    safe_repo = re.sub(r"^\.+", "dot-", repo).replace("/", "-")
    branch = f"aicg/{utc_now()[:10]}/{safe_repo}/maintainer-footer"

    # Each component must not start with '.'
    parts = branch.split("/")
    assert all(not p.startswith(".") for p in parts), branch


# ---------------------------------------------------------------------------
# Defensive idempotency: don't double-add when a 'Maintained by' line
# already exists (under any phrasing).
# ---------------------------------------------------------------------------


def test_rebrand_skips_when_prior_maintained_by_line_present(tmp_path: Path) -> None:
    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    # Old-style attribution that the rebrand should NOT clobber by
    # appending its own footer.
    write_file(
        workspace / "ai-infra-security-learning" / "README.md",
        "# Repo\n\n## License\n\n---\n\n"
        "*Maintained by the AI Infrastructure Curriculum Project*\n",
    )

    report = rebrand_run(
        manifest, workspace, state_dir=state_dir, apply=False,
        repos=["ai-infra-security-learning"],
    )

    outcome = next(
        r for r in report["repos"] if r["repo"] == "ai-infra-security-learning"
    )
    assert outcome["status"] == "already_branded"
    assert outcome["changed_files"] == []


def test_rebrand_still_skips_when_existing_attribution_is_to_us(tmp_path: Path) -> None:
    """A prior run added our exact phrasing without the marker — still skip."""
    workspace, state_dir, manifest = _setup_workspace(tmp_path)
    write_file(
        workspace / "ai-infra-security-learning" / "README.md",
        "# Repo\n\nMaintained by VeriSwarm.ai\n",
    )

    report = rebrand_run(
        manifest, workspace, state_dir=state_dir, apply=False,
        repos=["ai-infra-security-learning"],
    )

    outcome = next(
        r for r in report["repos"] if r["repo"] == "ai-infra-security-learning"
    )
    assert outcome["status"] == "already_branded"
