from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import write_file

from aicg.freshness import (
    FreshnessError,
    VersionTarget,
    audit_links,
    audit_versions,
    review_existing_artifacts,
)


# ---------------------------------------------------------------------------
# Link checker
# ---------------------------------------------------------------------------


def _seed_repo_with_links(tmp_path: Path, urls_by_file: dict[str, list[str]]) -> Path:
    repo = tmp_path / "repo"
    for rel, urls in urls_by_file.items():
        body = "\n".join(f"See [link]({url}) for more." for url in urls) + "\n"
        write_file(repo / rel, body)
    return repo


def test_audit_links_flags_404(tmp_path: Path) -> None:
    repo = _seed_repo_with_links(
        tmp_path,
        {"modules/mod-001/README.md": ["https://example.com/broken"]},
    )

    def fake_fetch(url: str) -> tuple[int, str]:
        return 404, "Not Found"

    report = audit_links(repo, url_fetcher=fake_fetch)
    assert report["broken_count"] == 1
    assert report["work_items"][0]["type"] == "refresh_links"
    assert report["work_items"][0]["severity"] == "low"


def test_audit_links_severity_escalates_with_count(tmp_path: Path) -> None:
    repo = _seed_repo_with_links(
        tmp_path,
        {
            "modules/mod-001/README.md": [
                f"https://example.com/broken/{i}" for i in range(6)
            ]
        },
    )

    report = audit_links(repo, url_fetcher=lambda url: (404, "Not Found"))
    item = report["work_items"][0]
    assert item["severity"] == "high"
    assert item["broken_count"] == 6


def test_audit_links_ignores_200_ok(tmp_path: Path) -> None:
    repo = _seed_repo_with_links(
        tmp_path,
        {"modules/mod-001/README.md": ["https://example.com/ok"]},
    )
    report = audit_links(repo, url_fetcher=lambda url: (200, "OK"))
    assert report["broken_count"] == 0
    assert report["work_items"] == []


def test_audit_links_raises_when_repo_missing(tmp_path: Path) -> None:
    with pytest.raises(FreshnessError):
        audit_links(tmp_path / "missing")


def test_audit_links_skips_archive_and_aicg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_file(
        repo / "modules/mod-001/README.md", "[ok](https://example.com/in-repo)\n"
    )
    write_file(
        repo / "_archive/old.md", "[old](https://example.com/in-archive)\n"
    )
    write_file(
        repo / ".aicg/notes.md", "[ai](https://example.com/in-aicg)\n"
    )

    seen: list[str] = []
    def fake_fetch(url: str) -> tuple[int, str]:
        seen.append(url)
        return 200, "OK"

    audit_links(repo, url_fetcher=fake_fetch)
    assert "https://example.com/in-repo" in seen
    assert "https://example.com/in-archive" not in seen
    assert "https://example.com/in-aicg" not in seen


# ---------------------------------------------------------------------------
# Version-pin scanner
# ---------------------------------------------------------------------------


def _registry_with(*targets) -> list[VersionTarget]:
    return list(targets)


def _make_target(**kwargs) -> VersionTarget:
    raw = {
        "id": kwargs.get("id", "pytorch"),
        "name": kwargs.get("name", "PyTorch"),
        "current": kwargs.get("current", "2.5"),
        "deprecated": kwargs.get("deprecated", ["2.0", "2.1"]),
        "pattern": kwargs.get(
            "pattern", r"(?i)pytorch[\s=:@v]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)"
        ),
        "eol": kwargs.get("eol", False),
    }
    return VersionTarget.from_dict(raw)


def test_audit_versions_flags_deprecated(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_file(
        repo / "modules/mod-001/README.md",
        "Set up PyTorch 2.0 with CUDA 11.8.\n",
    )
    report = audit_versions(repo, targets=_registry_with(_make_target()))
    assert report["finding_count"] == 1
    finding = report["findings"][0]
    assert finding["matched_version"] == "2.0"
    assert finding["severity"] == "high"
    assert report["work_items"][0]["severity"] == "high"


def test_audit_versions_flags_minor_older_as_medium(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_file(
        repo / "modules/mod-001/README.md",
        "We use PyTorch 2.3 for training.\n",
    )
    report = audit_versions(
        repo,
        targets=_registry_with(
            _make_target(deprecated=["1.10", "1.11", "2.0", "2.1"])
        ),
    )
    finding = report["findings"][0]
    assert finding["severity"] == "medium"


def test_audit_versions_no_flags_when_current(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_file(repo / "modules/mod-001/README.md", "PyTorch 2.5 is recommended.\n")
    report = audit_versions(
        repo, targets=_registry_with(_make_target(current="2.5"))
    )
    assert report["finding_count"] == 0
    assert report["work_items"] == []


def test_audit_versions_eol_always_high(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_file(repo / "modules/mod-001/README.md", "Use Theano 1.0 for ...\n")
    target = _make_target(
        id="theano",
        name="Theano",
        current="(none)",
        deprecated=[],
        pattern=r"(?i)theano[\s=:v]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
        eol=True,
    )
    report = audit_versions(repo, targets=[target])
    assert report["findings"][0]["severity"] == "high"


def test_audit_versions_from_registry_file(tmp_path: Path) -> None:
    registry = tmp_path / "version-targets.yaml"
    registry.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "id": "pytorch",
                        "name": "PyTorch",
                        "current": "2.5",
                        "deprecated": ["2.0"],
                        "pattern": r"(?i)pytorch[\s=:@v]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    write_file(repo / "README.md", "PyTorch 2.0 here.\n")
    report = audit_versions(repo, registry_path=registry)
    assert report["finding_count"] == 1


# ---------------------------------------------------------------------------
# Freshness review
# ---------------------------------------------------------------------------


def test_review_skips_when_judge_disabled(tmp_path: Path) -> None:
    from aicg.judge import JudgeConfig

    repo = tmp_path / "repo"
    write_file(repo / "modules/mod-001/SOLUTION.md", "# stale\n")
    judge_config = JudgeConfig(
        enabled=False,
        agent_command=None,
        dimensions=(),
        thresholds={"default": 75},
        timeout_seconds=None,
    )

    def fake_judge(**kwargs):
        return None  # judge disabled → skipped

    report = review_existing_artifacts(
        repo, judge_config=judge_config, artifact_judge=fake_judge
    )
    assert report["stale_count"] == 0
    assert report["work_items"] == []
    assert report["findings"][0]["status"] == "skipped"


def test_review_emits_work_item_when_stale(tmp_path: Path) -> None:
    from aicg.judge import JudgeConfig, JudgeVerdict

    repo = tmp_path / "repo"
    write_file(repo / "modules/mod-001/SOLUTION.md", "Use PyTorch 1.10\n")
    judge_config = JudgeConfig(
        enabled=True,
        agent_command="fake",
        dimensions=(),
        thresholds={"default": 75, "freshness": 75},
        timeout_seconds=None,
    )

    def fake_judge(**kwargs):
        return JudgeVerdict(
            score=40,
            dimensions={"api_currency": 10, "version_currency": 5},
            blockers=["References PyTorch 1.10"],
            summary="Stale",
            passed=False,
            threshold=75,
            raw="",
        )

    report = review_existing_artifacts(
        repo, judge_config=judge_config, artifact_judge=fake_judge
    )
    assert report["stale_count"] == 1
    item = report["work_items"][0]
    assert item["type"] == "refresh_stale"
    assert item["severity"] == "high"  # blockers escalate to high


def test_review_defers_on_subscription_limit(tmp_path: Path) -> None:
    from aicg.judge import JudgeConfig, JudgeVerdict

    repo = tmp_path / "repo"
    # Two artifacts: the first triggers the defer, the second should be skipped.
    write_file(repo / "modules/mod-001/SOLUTION.md", "")
    write_file(repo / "modules/mod-002/SOLUTION.md", "")
    judge_config = JudgeConfig(
        enabled=True,
        agent_command="fake",
        dimensions=(),
        thresholds={"default": 75, "freshness": 75},
        timeout_seconds=None,
    )

    def fake_judge(**kwargs):
        return JudgeVerdict(
            score=0,
            dimensions={},
            blockers=["Judge subscription limit reached (five_hour); retry after 2026-05-27T20:00:00Z."],
            summary="",
            passed=False,
            threshold=75,
            raw="",
        )

    report = review_existing_artifacts(
        repo, judge_config=judge_config, artifact_judge=fake_judge
    )
    # Loop breaks on defer; only one artifact recorded.
    assert report["deferred"] is not None
    assert len(report["findings"]) == 1


# ---------------------------------------------------------------------------
# queue_priority severity bias
# ---------------------------------------------------------------------------


def test_severity_bias_promotes_high_refresh_above_structural(tmp_path: Path) -> None:
    from conftest import write_minimal_manifest
    from aicg.org_config import load_manifest
    from aicg.org_runner import queue_priority

    manifest = load_manifest(write_minimal_manifest(tmp_path / "aicg-org.yaml"))
    repo = "ai-infra-security-solutions"

    structural = {"id": "mod-001", "type": "module_solution_gap", "priority": 100}
    refresh_high = {
        "id": "refresh-stale-mod-001",
        "type": "refresh_stale",
        "severity": "high",
        "priority": 100,
    }
    refresh_low = {
        "id": "refresh-links-mod-001",
        "type": "refresh_links",
        "severity": "low",
        "priority": 100,
    }

    p_structural = queue_priority(manifest, repo, structural)
    p_high = queue_priority(manifest, repo, refresh_high)
    p_low = queue_priority(manifest, repo, refresh_low)

    # Lower priority number = higher priority. High-severity refresh
    # MUST jump structural gaps; low-severity refresh MUST sit after.
    assert p_high < p_structural
    assert p_structural < p_low
