"""File-based generation adapter for local Claude/Codex commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_cli import AgentLimitReached, run_agent_command
from .source_registry import SourceRegistry
from .state import ensure_state_dir, state_path, utc_now, write_json, write_state

RUN_STATE = "run-state.json"


class GenerationNotConfigured(RuntimeError):
    def __init__(self, prompt_path: Path, output_dir: Path):
        self.prompt_path = prompt_path
        self.output_dir = output_dir
        super().__init__(
            "No generator command configured. Prompt packet was written for manual execution."
        )


def generate_from_plan(
    repo_path: Path,
    work_plan: dict[str, Any],
    module: str | None = None,
    work_id: str | None = None,
    config_paths: list[Path] | None = None,
    registry: SourceRegistry | None = None,
    command_override: str | None = None,
) -> dict[str, Any]:
    item = select_work_item(work_plan, module=module, work_id=work_id)
    if item is None:
        raise ValueError("No matching work item was found.")

    registry = registry or SourceRegistry.load()
    state_dir = ensure_state_dir(repo_path)
    prompt_dir = state_dir / "prompts"
    output_dir = state_dir / "generated" / item["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{item['id']}.md"
    prompt_path.write_text(build_prompt_packet(item, registry), encoding="utf-8")

    command = command_override or load_generator_command(repo_path, config_paths or [])
    run_state = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "repo": work_plan["repo"]["name"],
        "work_id": item["id"],
        "prompt_path": str(prompt_path),
        "output_dir": str(output_dir),
        "status": "prompt_ready",
    }
    write_state(repo_path, RUN_STATE, run_state)

    if not command:
        raise GenerationNotConfigured(prompt_path=prompt_path, output_dir=output_dir)

    formatted = command.format(
        prompt=str(prompt_path),
        output_dir=str(output_dir),
        repo=str(repo_path),
        work_id=item["id"],
        runner=str(Path(__file__).resolve().parents[2]),
    )
    result = run_agent_command(formatted, cwd=repo_path)
    if result.limit_reached:
        run_state.update(
            {
                "status": "agent_limit_reached",
                "deferred": True,
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "limit_scope": result.limit_scope,
                "retry_after": result.retry_after,
                "updated_at": utc_now(),
            }
        )
        write_json(state_path(repo_path, RUN_STATE), run_state)
        raise AgentLimitReached(result)

    run_state.update(
        {
            "status": "generated" if result.returncode == 0 else "generator_failed",
            "command": result.command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "updated_at": utc_now(),
        }
    )
    write_json(state_path(repo_path, RUN_STATE), run_state)
    if result.returncode != 0:
        raise RuntimeError(f"Generator command failed with exit code {result.returncode}.")
    return run_state


def select_work_item(
    work_plan: dict[str, Any],
    module: str | None = None,
    work_id: str | None = None,
) -> dict[str, Any] | None:
    for item in work_plan.get("work_items", []):
        if work_id and item["id"] != work_id:
            continue
        if module and item.get("module") != module:
            continue
        return item
    return None


def build_prompt_packet(work_item: dict[str, Any], registry: SourceRegistry) -> str:
    source_ids = work_item.get("source_policy", {}).get("required_default_sources", [])
    source_lines = []
    for source_id in source_ids:
        source = registry.get(source_id)
        if source is None:
            continue
        source_lines.append(f"- `{source.id}` ({source.authority}): {source.name} - {source.url}")

    exercise_lines = []
    for exercise in work_item.get("exercises", []):
        exercise_lines.append(
            "\n".join(
                [
                    f"### {exercise['exercise_id']} - {exercise.get('title', exercise['exercise_id'])}",
                    f"- Learning file: `{exercise['learning_path']}`",
                    f"- Output directory: `{exercise['expected_solution_dir']}`",
                    "- Required artifact: `SOLUTION.md`",
                ]
            )
        )

    return (
        f"# AICG Work Packet: {work_item['id']}\n\n"
        f"## Goal\n\n"
        f"Create module-level reference solutions for `{work_item['repo']}` module "
        f"`{work_item['module']}`.\n\n"
        "## Scope\n\n"
        "- Modify only the target solutions repository.\n"
        "- Create one directory per learning exercise under the module solution directory.\n"
        "- Each exercise directory must contain `SOLUTION.md`.\n"
        "- Design exercises need a worked example, decision rationale, and grading rubric.\n"
        "- Implementation exercises need runnable or statically valid artifacts where feasible.\n\n"
        "## Source Policy\n\n"
        "- Use official standards and project documentation first.\n"
        "- VeriSwarm references may only be used as practitioner implementation examples.\n"
        "- Do not invent facts, metrics, incidents, or case studies.\n"
        "- If a factual claim cannot be verified, write `<!-- needs-research: ... -->`; this blocks auto-merge.\n\n"
        + ("\n".join(source_lines) if source_lines else "- No default sources matched this work item.")
        + "\n\n"
        "## Exercises\n\n"
        + "\n\n".join(exercise_lines)
        + "\n\n"
        "## Output Contract\n\n"
        "For every exercise, write a `SOLUTION.md` with these sections:\n\n"
        "1. Solution overview\n"
        "2. Worked answer or implementation\n"
        "3. Validation steps\n"
        "4. Rubric or review checklist\n"
        "5. Common mistakes\n"
        "6. References\n\n"
        "Keep claims tied to the listed official sources or to clearly labeled local exercise context.\n"
    )


def load_generator_command(repo_path: Path, config_paths: list[Path]) -> str | None:
    candidate_paths = [
        repo_path / "aicg.yaml",
        repo_path / "aicg.yml",
        repo_path / "aicg.json",
        *config_paths,
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        if path.suffix == ".json":
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw.get("generator_command") or raw.get("agent_command")
        parsed = parse_simple_yaml(path.read_text(encoding="utf-8"))
        command = parsed.get("generator_command") or parsed.get("agent_command")
        if command:
            return command
    return None


def parse_simple_yaml(content: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        data[key.strip()] = value
    return data
