"""Manifest model for AI Infrastructure Curriculum org automation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_loader import ConfigError, load_config


class ManifestError(RuntimeError):
    """Raised when the org manifest is missing or malformed."""


@dataclass(frozen=True)
class RoleConfig:
    id: str
    title: str
    level: int
    learning_repo: str
    solution_repo: str


@dataclass(frozen=True)
class ExtraRepoConfig:
    name: str
    kind: str
    release: bool = False


@dataclass(frozen=True)
class OrgManifest:
    org: str
    default_remote: str
    roles: tuple[RoleConfig, ...]
    extra_repos: tuple[ExtraRepoConfig, ...]
    release: dict[str, Any]
    documentation: dict[str, Any]
    schedules: dict[str, str]
    automation: dict[str, Any]
    content_generation: dict[str, Any]
    job_requirements: dict[str, Any]
    research: dict[str, Any]
    path: Path

    @property
    def repo_names(self) -> list[str]:
        names: list[str] = []
        for role in self.roles:
            names.extend([role.learning_repo, role.solution_repo])
        names.extend(repo.name for repo in self.extra_repos)
        return list(dict.fromkeys(names))

    @property
    def learning_repo_names(self) -> list[str]:
        return [role.learning_repo for role in self.roles]

    @property
    def solution_repo_names(self) -> list[str]:
        return [role.solution_repo for role in self.roles]

    @property
    def release_repo_names(self) -> list[str]:
        names = self.learning_repo_names + self.solution_repo_names
        names.extend(repo.name for repo in self.extra_repos if repo.release)
        return list(dict.fromkeys(names))

    def remote_for(self, repo: str) -> str:
        return self.default_remote.format(org=self.org, repo=repo)

    def role_for_learning_repo(self, repo: str) -> RoleConfig | None:
        return next((role for role in self.roles if role.learning_repo == repo), None)


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "aicg-org.yaml"


def load_manifest(path: Path | None = None) -> OrgManifest:
    manifest_path = (path or default_manifest_path()).resolve()
    if not manifest_path.exists():
        raise ManifestError(f"Org manifest not found: {manifest_path}")
    try:
        raw = load_config(manifest_path)
    except ConfigError as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise ManifestError(
            f"Org manifest at {manifest_path} must be a mapping; got {type(raw).__name__}."
        )
    roles = tuple(
        RoleConfig(
            id=item["id"],
            title=item["title"],
            level=int(item["level"]),
            learning_repo=item["learning_repo"],
            solution_repo=item["solution_repo"],
        )
        for item in raw.get("roles", [])
    )
    if not roles:
        raise ManifestError("Org manifest must define at least one role.")
    extra_repos = tuple(
        ExtraRepoConfig(
            name=item["name"],
            kind=item.get("kind", "extra"),
            release=bool(item.get("release", False)),
        )
        for item in raw.get("extra_repos", [])
    )
    return OrgManifest(
        org=raw["org"],
        default_remote=raw["default_remote"],
        roles=roles,
        extra_repos=extra_repos,
        release=raw.get("release", {}),
        documentation=raw.get("documentation", {}),
        schedules=raw.get("schedules", {}),
        automation=raw.get("automation", {}),
        content_generation=raw.get("content_generation", {}),
        job_requirements=raw.get("job_requirements", {}),
        research=raw.get("research", {}),
        path=manifest_path,
    )


def state_dir_for_manifest(manifest: OrgManifest, override: Path | None = None) -> Path:
    if override is not None:
        return override.resolve()
    configured = Path(manifest.automation.get("state_dir", ".aicg/org"))
    if configured.is_absolute():
        return configured
    return (manifest.path.parent.parent / configured).resolve()
