"""Scaffold a new role's learning + solutions repos.

A new role is two paired repos plus a manifest entry. This module
generates the skeleton on disk so subsequent ``aicg`` cycles
(research -> plan -> generate -> verify -> PR -> steward) can fill
the curriculum in without any further human bootstrapping.

Phase A (this module) writes:

- ``ai-infra-<id>-learning/``: README.md, LICENSE, .gitignore, CI
  workflow, CURRICULUM.md placeholder, empty ``lessons/`` and
  ``projects/`` directories.
- ``ai-infra-<id>-solutions/``: README.md, LICENSE, .gitignore, CI
  workflow, ``modules/`` and ``projects/`` directories, plus
  ``SOLUTIONS_INDEX.md``.
- A role entry appended to the org manifest (if the manifest is JSON
  the file is rewritten; YAML is left for the user to update
  manually with the printed snippet).
- A prompt packet at ``<state-dir>/bootstrap/<id>.md`` that asks the
  content agent to (a) research the role's job requirements and (b)
  draft the initial curriculum plan as
  ``ai-infra-<id>-learning/.aicg/curriculum-plan.json``.

Phase B (out of scope for now): execute the curriculum plan, create
per-module skeletons, mark them as ready in the org queue.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .org_config import OrgManifest, state_dir_for_manifest
from .state import utc_now, write_json

BOOTSTRAP_REPORT = "bootstrap-report.json"

_ROLE_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")


class BootstrapError(RuntimeError):
    """Raised when a role cannot be scaffolded."""


@dataclass(frozen=True)
class BootstrapPlan:
    role_id: str
    title: str
    level: int
    learning_repo: str
    solution_repo: str
    learning_path: Path
    solution_path: Path
    prompt_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "title": self.title,
            "level": self.level,
            "learning_repo": self.learning_repo,
            "solution_repo": self.solution_repo,
            "learning_path": str(self.learning_path),
            "solution_path": str(self.solution_path),
            "prompt_path": str(self.prompt_path),
        }


def bootstrap_role(
    manifest: OrgManifest,
    workspace: Path,
    role_id: str,
    title: str,
    level: int,
    description: str | None = None,
    overwrite: bool = False,
    write_manifest: bool = True,
    create_remotes: bool = False,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Scaffold a paired learning + solutions repo set for ``role_id``."""
    if not _ROLE_ID_RE.match(role_id):
        raise BootstrapError(
            f"Invalid role id {role_id!r}; expected lowercase alphanumeric "
            "with hyphens (e.g. 'data-engineer')."
        )

    learning_repo = f"ai-infra-{role_id}-learning"
    solution_repo = f"ai-infra-{role_id}-solutions"
    workspace = workspace.resolve()
    learning_path = workspace / learning_repo
    solution_path = workspace / solution_repo

    if not overwrite:
        for path in (learning_path, solution_path):
            if path.exists() and any(path.iterdir()):
                raise BootstrapError(
                    f"Target path already exists with content: {path}. "
                    "Pass overwrite=True to scaffold on top."
                )

    learning_path.mkdir(parents=True, exist_ok=True)
    solution_path.mkdir(parents=True, exist_ok=True)

    description = description or f"Curriculum for the {title} role."
    files_written: list[str] = []

    files_written.extend(
        _write_learning_skeleton(learning_path, role_id, title, description)
    )
    files_written.extend(
        _write_solutions_skeleton(solution_path, role_id, title, description)
    )

    state_root = state_dir_for_manifest(manifest, state_dir)
    bootstrap_dir = state_root / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = bootstrap_dir / f"{role_id}.md"
    prompt_path.write_text(
        _build_bootstrap_prompt(role_id, title, level, description), encoding="utf-8"
    )

    plan = BootstrapPlan(
        role_id=role_id,
        title=title,
        level=level,
        learning_repo=learning_repo,
        solution_repo=solution_repo,
        learning_path=learning_path,
        solution_path=solution_path,
        prompt_path=prompt_path,
    )

    manifest_update = None
    if write_manifest:
        manifest_update = _append_role_to_manifest(manifest, plan)

    remote_result: dict[str, Any] | None = None
    if create_remotes:
        remote_result = _create_github_repos(manifest, plan)

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "bootstrap_role",
        "plan": plan.to_dict(),
        "files_written": files_written,
        "manifest_update": manifest_update,
        "remotes": remote_result,
    }
    write_json(state_root / BOOTSTRAP_REPORT, report)
    return report


# ---------------------------------------------------------------------------
# Skeleton writers
# ---------------------------------------------------------------------------


def _write_learning_skeleton(
    repo_path: Path, role_id: str, title: str, description: str
) -> list[str]:
    written: list[str] = []
    written.append(
        _write(
            repo_path / "README.md",
            _learning_readme(role_id, title, description),
        )
    )
    written.append(_write(repo_path / "LICENSE", _mit_license()))
    written.append(_write(repo_path / ".gitignore", _gitignore()))
    written.append(
        _write(
            repo_path / "CURRICULUM.md",
            _learning_curriculum_placeholder(role_id, title),
        )
    )
    written.append(
        _write(
            repo_path / "PREREQUISITES.md",
            _learning_prerequisites_placeholder(title),
        )
    )
    written.append(
        _write(
            repo_path / "VERSIONS.md",
            _versions_placeholder(role_id),
        )
    )
    written.append(
        _write(
            repo_path / ".github" / "workflows" / "ci.yml",
            _learning_ci_workflow(),
        )
    )
    # Empty content directories with placeholder READMEs so they show
    # up under git.
    written.append(
        _write(
            repo_path / "lessons" / "README.md",
            "# Lessons\n\nModule directories live here as `lessons/mod-XXX-name/`.\n",
        )
    )
    written.append(
        _write(
            repo_path / "projects" / "README.md",
            "# Projects\n\nCapstones live here as `projects/project-XXX-name/`.\n",
        )
    )
    return written


def _write_solutions_skeleton(
    repo_path: Path, role_id: str, title: str, description: str
) -> list[str]:
    written: list[str] = []
    written.append(
        _write(
            repo_path / "README.md",
            _solutions_readme(role_id, title, description),
        )
    )
    written.append(_write(repo_path / "LICENSE", _mit_license()))
    written.append(_write(repo_path / ".gitignore", _gitignore_solutions()))
    written.append(
        _write(
            repo_path / "SOLUTIONS_INDEX.md",
            _solutions_index_placeholder(role_id, title),
        )
    )
    written.append(
        _write(
            repo_path / ".github" / "workflows" / "ci.yml",
            _solutions_ci_workflow(),
        )
    )
    written.append(
        _write(
            repo_path / "modules" / "README.md",
            "# Modules\n\nReference solutions per module live here.\n",
        )
    )
    written.append(
        _write(
            repo_path / "projects" / "README.md",
            "# Projects\n\nReference solutions per project live here.\n",
        )
    )
    return written


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Skeleton content
# ---------------------------------------------------------------------------


def _learning_readme(role_id: str, title: str, description: str) -> str:
    return (
        f"# AI Infrastructure {title} — Learning Repository\n\n"
        f"{description}\n\n"
        "> **Status**: scaffolded by `aicg org bootstrap-role`. The curriculum is "
        "not authored yet. Run `aicg org research` and `aicg org daily` to drive "
        "the autonomous fill-in loop.\n\n"
        "## Layout\n\n"
        "```\n"
        f"ai-infra-{role_id}-learning/\n"
        "├── lessons/mod-XXX-*/        modules with lectures, exercises, labs, quizzes\n"
        "├── projects/project-XXX-*/   multi-module capstones\n"
        "├── CURRICULUM.md             role-level coverage map\n"
        "├── PREREQUISITES.md          assumed entry skills\n"
        "├── VERSIONS.md               release history\n"
        "└── README.md                 this file\n"
        "```\n\n"
        "## Paired Solutions Repo\n\n"
        f"[`ai-infra-{role_id}-solutions`](https://github.com/ai-infra-curriculum/ai-infra-{role_id}-solutions) "
        "carries the reference implementations.\n"
    )


def _solutions_readme(role_id: str, title: str, description: str) -> str:
    return (
        f"# AI Infrastructure {title} — Solutions Repository\n\n"
        "Reference implementations for the paired "
        f"[`ai-infra-{role_id}-learning`](https://github.com/ai-infra-curriculum/ai-infra-{role_id}-learning) "
        "track.\n\n"
        "> **Status**: scaffolded by `aicg org bootstrap-role`. Module and "
        "project solutions arrive over subsequent autonomous cycles.\n\n"
        "## Layout\n\n"
        "```\n"
        f"ai-infra-{role_id}-solutions/\n"
        "├── modules/mod-XXX-*/                 module-level rationale + per-exercise solutions\n"
        "├── projects/project-XXX-*/            capstone walkthroughs\n"
        "├── SOLUTIONS_INDEX.md                 inventory + completion map\n"
        "└── README.md                          this file\n"
        "```\n"
    )


def _learning_curriculum_placeholder(role_id: str, title: str) -> str:
    return (
        f"# {title} Curriculum\n\n"
        "> Scaffolded placeholder. The autonomous loop will populate this file "
        "as modules are designed and shipped.\n\n"
        "## Overview\n\n"
        "TBD.\n\n"
        "## Module Plan\n\n"
        "| Module | Status |\n"
        "|---|---|\n"
        "| mod-XXX | planned |\n"
    )


def _learning_prerequisites_placeholder(title: str) -> str:
    return (
        f"# Prerequisites for the {title} track\n\n"
        "> Scaffolded placeholder. Update during the curriculum-planning step.\n"
    )


def _versions_placeholder(role_id: str) -> str:
    return (
        f"# Versions — ai-infra-{role_id}-learning\n\n"
        "| Tag | Date | Highlights |\n"
        "|---|---|---|\n"
        "| (unreleased) | TBD | initial scaffold |\n"
    )


def _solutions_index_placeholder(role_id: str, title: str) -> str:
    return (
        f"# Solutions Index — {title}\n\n"
        f"Reference implementations for `ai-infra-{role_id}-learning`.\n\n"
        "## Coverage\n\n"
        "| Module | Solution Status |\n"
        "|---|---|\n"
        "| mod-XXX | planned |\n"
    )


def _mit_license() -> str:
    return (
        "MIT License\n\n"
        "Copyright (c) 2026 AI Infrastructure Curriculum\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy "
        'of this software and associated documentation files (the "Software"), to deal '
        "in the Software without restriction, including without limitation the rights "
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell "
        "copies of the Software, and to permit persons to whom the Software is "
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in "
        "all copies or substantial portions of the Software.\n\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR '
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE "
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER "
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING "
        "FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS "
        "IN THE SOFTWARE.\n"
    )


def _gitignore() -> str:
    return (
        "__pycache__/\n"
        "*.py[cod]\n"
        ".venv/\n"
        "venv/\n"
        ".DS_Store\n"
        "# AICG runner state — never commit.\n"
        ".aicg/\n"
    )


def _gitignore_solutions() -> str:
    # Same content for the solutions side but explicit so future
    # additions (e.g. wheel artefacts) can diverge cleanly.
    return _gitignore()


def _learning_ci_workflow() -> str:
    return (
        "name: CI\n\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [main]\n"
        "  push:\n"
        "    branches: [main]\n\n"
        "permissions:\n"
        "  contents: read\n\n"
        "jobs:\n"
        "  markdown-lint:\n"
        "    name: Markdown lint\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: DavidAnson/markdownlint-cli2-action@v16\n"
        "        with:\n"
        "          globs: |\n"
        "            lessons/**/*.md\n"
        "            projects/**/*.md\n"
        "            !**/node_modules/**\n"
    )


def _solutions_ci_workflow() -> str:
    return (
        "name: CI\n\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [main]\n"
        "  push:\n"
        "    branches: [main]\n\n"
        "permissions:\n"
        "  contents: read\n\n"
        "jobs:\n"
        "  markdown-lint:\n"
        "    name: Markdown lint\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: DavidAnson/markdownlint-cli2-action@v16\n"
        "        with:\n"
        "          globs: |\n"
        "            modules/**/*.md\n"
        "            projects/**/*.md\n"
        "            !**/node_modules/**\n\n"
        "  python-syntax:\n"
        "    name: Python syntax (modules/)\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: \"3.10\"\n"
        "      - name: Compile every .py under modules/\n"
        "        shell: bash\n"
        "        run: |\n"
        "          set -euo pipefail\n"
        "          shopt -s globstar nullglob\n"
        "          files=( modules/**/*.py )\n"
        "          if [ ${#files[@]} -eq 0 ]; then\n"
        "            echo \"No modules/**/*.py — nothing to check.\"\n"
        "            exit 0\n"
        "          fi\n"
        "          fails=0\n"
        "          for f in \"${files[@]}\"; do\n"
        "            if ! python -m py_compile \"$f\"; then\n"
        "              echo \"::error file=$f::SyntaxError\"\n"
        "              fails=$((fails+1))\n"
        "            fi\n"
        "          done\n"
        "          echo \"Compiled ${#files[@]} file(s); ${fails} failure(s).\"\n"
        "          exit \"$fails\"\n"
    )


# ---------------------------------------------------------------------------
# Manifest update + remote creation
# ---------------------------------------------------------------------------


def _append_role_to_manifest(
    manifest: OrgManifest, plan: BootstrapPlan
) -> dict[str, Any]:
    if plan.role_id in {role.id for role in manifest.roles}:
        return {
            "status": "already_present",
            "manifest_path": str(manifest.path),
        }

    text = manifest.path.read_text(encoding="utf-8")
    new_role = {
        "id": plan.role_id,
        "title": plan.title,
        "level": plan.level,
        "learning_repo": plan.learning_repo,
        "solution_repo": plan.solution_repo,
    }

    stripped = text.lstrip()
    if stripped.startswith("{"):
        # JSON manifest — rewrite in place.
        payload = json.loads(text)
        payload.setdefault("roles", []).append(new_role)
        manifest.path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "appended_json",
            "manifest_path": str(manifest.path),
        }

    # YAML manifest — the runner does not commit to rewriting arbitrary
    # YAML in-place. Print the snippet for the operator to paste.
    return {
        "status": "yaml_manifest_manual_update_required",
        "manifest_path": str(manifest.path),
        "snippet": (
            "  - id: " + plan.role_id + "\n"
            "    title: " + json.dumps(plan.title) + "\n"
            "    level: " + str(plan.level) + "\n"
            "    learning_repo: " + plan.learning_repo + "\n"
            "    solution_repo: " + plan.solution_repo + "\n"
        ),
    }


def _create_github_repos(
    manifest: OrgManifest, plan: BootstrapPlan
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for repo_name in (plan.learning_repo, plan.solution_repo):
        completed = subprocess.run(
            [
                "gh",
                "repo",
                "create",
                f"{manifest.org}/{repo_name}",
                "--public",
                "--source",
                str(plan.learning_path if repo_name.endswith("-learning") else plan.solution_path),
                "--remote",
                "origin",
                "--push",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(
            {
                "repo": repo_name,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
        )
    return {"actions": results}


# ---------------------------------------------------------------------------
# Bootstrap prompt
# ---------------------------------------------------------------------------


def _build_bootstrap_prompt(
    role_id: str, title: str, level: int, description: str
) -> str:
    return (
        f"# Bootstrap Curriculum Packet — {title} (level {level})\n\n"
        f"`ai-infra-{role_id}-learning` and `ai-infra-{role_id}-solutions` were "
        "scaffolded with empty curriculum directories. Your job is to research "
        "the role and produce the initial curriculum plan.\n\n"
        "## Phase 1 — Research\n\n"
        "- Analyse at least 25 current job postings for this role.\n"
        f"- Description seed: {description}\n"
        "- Capture employer, posting URL, date, location, required skills, "
        "preferred skills, salary range.\n"
        "- Normalise findings into "
        f"`ai-infra-{role_id}-learning/.aicg/job-requirements.json`.\n"
        "- Write a readable summary to "
        f"`ai-infra-{role_id}-learning/JOB_REQUIREMENTS.md`.\n"
        "- Cite official sources where claims need backing.\n\n"
        "## Phase 2 — Curriculum Plan\n\n"
        "Author "
        f"`ai-infra-{role_id}-learning/.aicg/curriculum-plan.json` with this shape:\n\n"
        "```json\n"
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"role_id\": \"" + role_id + "\",\n"
        "  \"title\": \"" + title + "\",\n"
        "  \"level\": " + str(level) + ",\n"
        "  \"modules\": [\n"
        "    {\n"
        "      \"id\": \"mod-101-foundations\",\n"
        "      \"title\": \"...\",\n"
        "      \"hours\": 12,\n"
        "      \"objectives\": [\"...\"],\n"
        "      \"exercises\": [\n"
        "        {\"id\": \"exercise-01\", \"slug\": \"...\", \"hours\": 2}\n"
        "      ],\n"
        "      \"labs\": [...],\n"
        "      \"quizzes\": 1\n"
        "    }\n"
        "  ],\n"
        "  \"projects\": [\n"
        "    {\"id\": \"project-101-...\", \"title\": \"...\", \"hours\": 20}\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "## Ownership Rule\n\n"
        "If a requirement is already covered by a lower-level role, link to "
        "that owner instead of duplicating. Higher-level coverage should add "
        "depth, architectural context, or leadership framing — not repeat the "
        "fundamentals.\n\n"
        "## Constraints\n\n"
        "- Do not invent facts, incidents, or salary figures. Cite sources.\n"
        "- Unverified claims must use `<!-- needs-research: ... -->`.\n"
        "- Preserve the standard layout (`lessons/mod-XXX-*/` for learning, "
        "`modules/mod-XXX-*/` for solutions).\n"
        "- Update `CURRICULUM.md`, `PREREQUISITES.md`, `VERSIONS.md`, "
        "`README.md` to reflect the planned curriculum once `curriculum-plan.json` "
        "lands.\n"
    )
