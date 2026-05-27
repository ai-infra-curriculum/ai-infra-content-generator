"""Objective-based curriculum repository audit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .inventory import RepositoryInfo, WorkspaceInventory
from .source_registry import SourceRegistry, extract_urls
from .state import relative_path, utc_now, write_state

AUDIT_REPORT = "audit-report.json"
TEXT_EXTENSIONS = {".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh", ".hcl"}
SKIP_DIRS = {".git", ".aicg", ".venv", "node_modules", "__pycache__"}
PLACEHOLDER_PATTERNS = {
    "needs_research": re.compile(r"needs-research", re.IGNORECASE),
    "manual_review": re.compile(r"#\s*manual-review|manual-review", re.IGNORECASE),
    "todo": re.compile(r"\bTODO\b|FIXME|\bTBD\b", re.IGNORECASE),
    "placeholder": re.compile(r"\bplaceholder\b|coming soon|lorem ipsum", re.IGNORECASE),
}
EXERCISE_ID_PATTERN = re.compile(r"^(exercise-\d+)")


class AuditError(RuntimeError):
    """Raised when an audit cannot be completed."""


def audit_repo(
    workspace: Path,
    repo_name: str,
    module: str | None = None,
    write_report: bool = True,
    source_registry: SourceRegistry | None = None,
) -> dict[str, Any]:
    inventory = WorkspaceInventory(workspace)
    target = inventory.require(repo_name)
    paired = inventory.paired_repo(target)
    registry = source_registry or SourceRegistry.load()

    if target.kind == "solutions" and paired is None:
        raise AuditError(f"Solutions repo '{target.name}' has no paired learning repo in {workspace}.")

    learning_repo = paired if target.kind == "solutions" else target
    solution_repo = target if target.kind == "solutions" else paired

    if learning_repo is None:
        raise AuditError(f"Learning repo for '{target.name}' could not be resolved.")

    modules = audit_learning_solution_parity(
        learning_repo=learning_repo,
        solution_repo=solution_repo,
        module_filter=module,
        registry=registry,
    )
    placeholder_findings = scan_placeholders(target.path)
    repo_checks = audit_repo_checks(target)

    gaps: list[dict[str, Any]] = []
    for item in modules:
        gaps.extend(item["gaps"])
    gaps.extend(placeholder_findings)
    gaps.extend(repo_checks["gaps"])

    error_count = sum(1 for gap in gaps if gap["severity"] == "error")
    warning_count = sum(1 for gap in gaps if gap["severity"] == "warning")
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "workspace": str(Path(workspace).resolve()),
        "repo": repo_dict(target),
        "paired_repo": repo_dict(paired) if paired else None,
        "source_policy": {
            "registry_last_verified": registry.last_verified,
            "official_first": True,
            "practitioner_reference_note": (
                "VeriSwarm sources may be used only as practitioner implementation references, "
                "not standards authorities."
            ),
        },
        "summary": {
            "status": "fail" if error_count else "pass",
            "error_count": error_count,
            "warning_count": warning_count,
            "module_count": len(modules),
            "gap_count": len(gaps),
        },
        "checks": {
            "learning_solution_parity": {
                "status": "fail" if any(module_item["status"] != "ok" for module_item in modules) else "pass"
            },
            "placeholder_scan": {
                "status": "fail"
                if any(gap["severity"] == "error" for gap in placeholder_findings)
                else "pass",
                "finding_count": len(placeholder_findings),
            },
            "repo_structure": repo_checks["summary"],
        },
        "modules": modules,
        "gaps": gaps,
    }

    if write_report:
        write_state(target.path, AUDIT_REPORT, report)
    return report


def audit_learning_solution_parity(
    learning_repo: RepositoryInfo,
    solution_repo: RepositoryInfo | None,
    module_filter: str | None,
    registry: SourceRegistry,
) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for learning_module in discover_learning_modules(learning_repo):
        module_id = learning_module.name
        if module_filter and module_filter != module_id:
            continue

        solution_module = solution_repo.path / "modules" / module_id if solution_repo else None
        exercises = []
        module_gaps: list[dict[str, Any]] = []

        if solution_repo is None:
            module_gaps.append(
                gap(
                    "missing_paired_solutions_repo",
                    "error",
                    f"Learning module {module_id} has no paired solutions repo.",
                    module_id=module_id,
                    path=relative_path(learning_module, learning_repo.path),
                )
            )
        elif solution_module is None or not solution_module.exists():
            module_gaps.append(
                gap(
                    "missing_solution_module",
                    "error",
                    f"Missing solution module directory for {module_id}.",
                    module_id=module_id,
                    expected_path=relative_path(solution_module, solution_repo.path),
                )
            )

        for exercise_path in discover_exercise_files(learning_module):
            exercise_id = normalize_exercise_id(exercise_path)
            expected_dir = solution_module / exercise_id if solution_module else None
            expected_solution = expected_dir / "SOLUTION.md" if expected_dir else None
            exercise_status = "ok"
            exercise_gaps: list[dict[str, Any]] = []

            if expected_solution is None or not expected_solution.exists():
                exercise_status = "missing_solution"
                exercise_gaps.append(
                    gap(
                        "missing_exercise_solution",
                        "error",
                        f"Missing SOLUTION.md for {module_id}/{exercise_id}.",
                        module_id=module_id,
                        exercise_id=exercise_id,
                        learning_path=relative_path(exercise_path, learning_repo.path),
                        expected_path=relative_path(expected_solution, solution_repo.path)
                        if expected_solution and solution_repo
                        else None,
                    )
                )
            else:
                source_gaps = audit_solution_sources(
                    expected_solution,
                    solution_repo.path,
                    module_id,
                    exercise_id,
                    registry,
                )
                if any(item["severity"] == "error" for item in source_gaps):
                    exercise_status = "source_gap"
                exercise_gaps.extend(source_gaps)

            exercise_title = extract_title(exercise_path)
            exercises.append(
                {
                    "exercise_id": exercise_id,
                    "slug": exercise_path.stem,
                    "title": exercise_title,
                    "learning_path": relative_path(exercise_path, learning_repo.path),
                    "expected_solution_dir": relative_path(expected_dir, solution_repo.path)
                    if expected_dir and solution_repo
                    else None,
                    "required_artifacts": ["SOLUTION.md"],
                    "status": exercise_status,
                    "gaps": exercise_gaps,
                }
            )
            module_gaps.extend(exercise_gaps)

        status = "ok" if not any(item["severity"] == "error" for item in module_gaps) else "gap"
        modules.append(
            {
                "module_id": module_id,
                "learning_path": relative_path(learning_module, learning_repo.path),
                "solution_path": relative_path(solution_module, solution_repo.path)
                if solution_module and solution_repo
                else None,
                "status": status,
                "exercise_count": len(exercises),
                "missing_solution_count": sum(
                    1 for exercise in exercises if exercise["status"] == "missing_solution"
                ),
                "exercises": exercises,
                "gaps": module_gaps,
            }
        )
    return modules


def discover_learning_modules(repo: RepositoryInfo) -> list[Path]:
    root = repo.module_root
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("mod-"))


def discover_exercise_files(module_path: Path) -> list[Path]:
    exercises_dir = module_path / "exercises"
    if not exercises_dir.exists():
        return []
    return sorted(
        path
        for path in exercises_dir.iterdir()
        if path.is_file() and path.suffix == ".md" and path.name.lower() != "readme.md"
    )


def normalize_exercise_id(path: Path) -> str:
    match = EXERCISE_ID_PATTERN.match(path.stem)
    return match.group(1) if match else path.stem


def extract_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
    except UnicodeDecodeError:
        return path.stem
    return path.stem.replace("-", " ").title()


def audit_solution_sources(
    solution_file: Path,
    repo_path: Path,
    module_id: str,
    exercise_id: str,
    registry: SourceRegistry,
) -> list[dict[str, Any]]:
    content = solution_file.read_text(encoding="utf-8")
    urls = extract_urls(content)
    if not urls and len(content.split()) >= 250:
        return [
            gap(
                "missing_source_references",
                "warning",
                f"{module_id}/{exercise_id} has substantial explanation but no source URLs.",
                module_id=module_id,
                exercise_id=exercise_id,
                path=relative_path(solution_file, repo_path),
            )
        ]

    classified = [registry.classify_url(url) for url in urls]
    has_official = any(source and source.is_official for source in classified)
    has_practitioner = any(source and source.is_practitioner_reference for source in classified)
    findings: list[dict[str, Any]] = []
    if has_practitioner and not has_official:
        findings.append(
            gap(
                "practitioner_reference_without_official_source",
                "warning",
                f"{module_id}/{exercise_id} uses practitioner references without an official source.",
                module_id=module_id,
                exercise_id=exercise_id,
                path=relative_path(solution_file, repo_path),
            )
        )
    return findings


def scan_placeholders(repo_path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in iter_text_files(repo_path):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, 1):
            for marker, pattern in PLACEHOLDER_PATTERNS.items():
                if pattern.search(line):
                    severity = "error" if marker in {"needs_research", "manual_review"} else "warning"
                    findings.append(
                        gap(
                            marker,
                            severity,
                            f"Found {marker.replace('_', '-')} marker.",
                            path=relative_path(path, repo_path),
                            line=line_number,
                            excerpt=line.strip()[:160],
                        )
                    )
    return findings


def iter_text_files(repo_path: Path) -> list[Path]:
    paths: list[Path] = []
    for path in repo_path.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            paths.append(path)
    return paths


def audit_repo_checks(repo: RepositoryInfo) -> dict[str, Any]:
    gaps: list[dict[str, Any]] = []
    if not (repo.path / ".github" / "workflows").exists():
        gaps.append(
            gap(
                "missing_ci_workflow",
                "warning",
                "Repository has no .github/workflows directory for validation.",
            )
        )
    if repo.kind == "solutions" and not (repo.path / "modules").exists():
        gaps.append(
            gap(
                "missing_modules_dir",
                "error",
                "Solutions repository has no modules directory.",
                expected_path="modules",
            )
        )
    return {
        "summary": {
            "status": "fail" if any(item["severity"] == "error" for item in gaps) else "pass",
            "finding_count": len(gaps),
        },
        "gaps": gaps,
    }


def gap(kind: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    data = {"type": kind, "severity": severity, "message": message}
    data.update({key: value for key, value in extra.items() if value is not None})
    return data


def repo_dict(repo: RepositoryInfo | None) -> dict[str, Any] | None:
    if repo is None:
        return None
    return {
        "name": repo.name,
        "path": str(repo.path),
        "kind": repo.kind,
        "track": repo.track,
        "has_git": repo.has_git,
    }
