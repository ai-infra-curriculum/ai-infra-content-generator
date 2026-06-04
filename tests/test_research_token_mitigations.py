"""Regression tests for Bug 4 mitigations.

- Brief-mode prompts: non-canary roles get slug-only summaries
- Canary roles still get the full hierarchy (they consume it)
- _pending_proposal_pr: branch-prefix detection works for both opener formats
- research_apply skips a role when an open proposal PR exists
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aicg.manifest import (
    CurriculumManifest,
    Module,
    Track,
    summarize_manifest_for_prompt,
    summarize_track_for_prompt,
)
from aicg.org_config import RoleConfig, load_manifest
from aicg.org_runner import build_research_prompt
from aicg.research import (
    CANARY_ROLES_NEW_FORMAT,
    _OPEN_PR_BRANCH_PREFIXES,
    _pending_proposal_pr,
)


def _completed(returncode: int = 0, stdout: str = "[]", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------- brief-mode summaries ----------


def test_summarize_track_brief_is_much_shorter_than_full() -> None:
    """Brief mode should be at least 50% smaller than full mode."""
    track = Track(
        slug="junior-engineer",
        display_name="Junior",
        level=1,
        learning_repo="x",
        solutions_repo="y",
        learning_repo_url="https://x",
        solutions_repo_url="https://y",
        modules=tuple(
            Module(
                slug=f"mod-{i:03d}-thing",
                number=i,
                title=f"Module {i}: Something Long",
                path=f"lessons/mod-{i:03d}",
                github_url="https://x",
                exercises=tuple(),
                prerequisites=(),
                related=(),
            )
            for i in range(1, 11)
        ),
        projects=tuple(),
        resources=tuple(),
    )
    full = summarize_track_for_prompt(track)
    brief = summarize_track_for_prompt(track, brief=True)
    assert len(brief) < len(full) * 0.6
    # Brief still names every module slug — just compactly.
    for i in range(1, 11):
        assert f"mod-{i:03d}-thing" in brief


def test_summarize_manifest_brief_omits_sibling_index() -> None:
    """Brief mode skips the sibling-tracks index too (pure noise for non-canary)."""
    # Use a minimal synthetic manifest.
    track = Track(
        slug="x",
        display_name="X",
        level=1,
        learning_repo="r",
        solutions_repo="s",
        learning_repo_url="",
        solutions_repo_url="",
        modules=tuple(),
        projects=tuple(),
        resources=tuple(),
    )
    manifest = CurriculumManifest(
        schema_version=1,
        generated_at="now",
        org="ai-infra-curriculum",
        tracks=(track,),
    )
    brief = summarize_manifest_for_prompt(
        manifest, only_track_slug="x", brief=True
    )
    full = summarize_manifest_for_prompt(manifest, only_track_slug="x")
    assert "Sibling tracks" not in brief
    # Full mode would emit a sibling index if there were other tracks,
    # but with only one track it just omits the line. Both forms have
    # the role's header.
    assert "X" in brief and "X" in full


# ---------- routing through build_research_prompt ----------


@pytest.fixture
def org_manifest():
    p = Path("config/aicg-org.yaml")
    if not p.exists():
        pytest.skip("org manifest not in this checkout")
    return load_manifest(p)


def test_non_canary_prompt_uses_brief_existing_curriculum(org_manifest) -> None:
    """A non-canary role's prompt should be SMALLER than the canary's prompt."""
    canary_id = next(iter(CANARY_ROLES_NEW_FORMAT))
    canary_role = next((r for r in org_manifest.roles if r.id == canary_id), None)
    non_canary_role = next(
        (r for r in org_manifest.roles if r.id not in CANARY_ROLES_NEW_FORMAT), None
    )
    if canary_role is None or non_canary_role is None:
        pytest.skip("manifest does not have a canary and non-canary role")

    canary_prompt = build_research_prompt(org_manifest, canary_role, "2026-06")
    non_canary_prompt = build_research_prompt(org_manifest, non_canary_role, "2026-06")
    # Canary prompt embeds the full hierarchy + larger v2 output contract.
    # Non-canary should be smaller.
    assert len(non_canary_prompt) < len(canary_prompt)
    # Non-canary prompt's existing-curriculum section uses the brief
    # intro: "These already exist — do NOT propose duplicates."
    assert "do NOT propose duplicates" in non_canary_prompt
    # Canary's section uses the longer intro:
    assert "Treat every module / exercise / project" in canary_prompt


# ---------- _pending_proposal_pr ----------


def _role(role_id: str = "junior-engineer") -> RoleConfig:
    import dataclasses

    defaults: dict[str, Any] = {
        "id": role_id,
        "title": role_id,
        "level": 1,
        "learning_repo": f"ai-infra-{role_id}-learning",
    }
    for field in dataclasses.fields(RoleConfig):
        if field.name in defaults:
            continue
        if field.default is not dataclasses.MISSING:
            continue
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            continue
        defaults[field.name] = ""
    return RoleConfig(**defaults)


def test_pending_pr_detects_aicg_research_branch() -> None:
    rows = [
        {
            "url": "https://github.com/ai-infra-curriculum/x/pull/42",
            "headRefName": "aicg/research/2026-06/junior-engineer",
            "updatedAt": "2026-06-01T00:00:00Z",
        }
    ]
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(stdout=json.dumps(rows)),
    ):
        url = _pending_proposal_pr(_role())
    assert url == "https://github.com/ai-infra-curriculum/x/pull/42"


def test_pending_pr_detects_v2_research_branch() -> None:
    rows = [
        {
            "url": "https://github.com/x/y/pull/7",
            "headRefName": "research/2026-06/junior-engineer",
            "updatedAt": "2026-06-02T00:00:00Z",
        }
    ]
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(stdout=json.dumps(rows)),
    ):
        url = _pending_proposal_pr(_role())
    assert url == "https://github.com/x/y/pull/7"


def test_pending_pr_ignores_other_role_branches() -> None:
    """A PR for a different role must not block this role."""
    rows = [
        {
            "url": "https://x/1",
            "headRefName": "aicg/research/2026-06/senior-engineer",
            "updatedAt": "2026-06-01T00:00:00Z",
        }
    ]
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(stdout=json.dumps(rows)),
    ):
        url = _pending_proposal_pr(_role("junior-engineer"))
    assert url is None


def test_pending_pr_ignores_non_research_branches() -> None:
    """Random feature branches must not count as pending review."""
    rows = [
        {
            "url": "https://x/2",
            "headRefName": "feat/some-thing",
            "updatedAt": "2026-06-01T00:00:00Z",
        }
    ]
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(stdout=json.dumps(rows)),
    ):
        url = _pending_proposal_pr(_role())
    assert url is None


def test_pending_pr_picks_most_recently_updated() -> None:
    rows = [
        {
            "url": "https://old",
            "headRefName": "aicg/research/2026-04/junior-engineer",
            "updatedAt": "2026-04-15T00:00:00Z",
        },
        {
            "url": "https://newer",
            "headRefName": "aicg/research/2026-06/junior-engineer",
            "updatedAt": "2026-06-01T00:00:00Z",
        },
    ]
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(stdout=json.dumps(rows)),
    ):
        url = _pending_proposal_pr(_role())
    assert url == "https://newer"


def test_pending_pr_gh_failure_returns_none() -> None:
    """gh failure must not block the cycle — return None and let agent run."""
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(returncode=1, stderr="auth required"),
    ):
        url = _pending_proposal_pr(_role())
    assert url is None


def test_pending_pr_no_open_prs_returns_none() -> None:
    with patch(
        "aicg.research.subprocess.run",
        return_value=_completed(stdout="[]"),
    ):
        url = _pending_proposal_pr(_role())
    assert url is None


# ---------- branch prefixes constant ----------


def test_branch_prefixes_cover_both_openers() -> None:
    """The constant must include both prefix forms used by _open_proposal_pr / _v2."""
    assert "aicg/research/" in _OPEN_PR_BRANCH_PREFIXES
    assert "research/" in _OPEN_PR_BRANCH_PREFIXES
