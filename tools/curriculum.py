#!/usr/bin/env python3
"""
Curriculum CLI

Provides lightweight helpers for interacting with pipeline manifests, metadata,
and validation profiles. This is a foundation for future automation or agent integrations.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_list_pipelines(_: argparse.Namespace) -> None:
    pipelines_dir = ROOT / "pipelines"
    for manifest in sorted(pipelines_dir.glob("*.yaml")):
        data = load_yaml(manifest)
        name = data.get("metadata", {}).get("name", manifest.stem)
        desc = data.get("metadata", {}).get("description", "")
        print(f"{name}: {desc}")


def cmd_show_pipeline(args: argparse.Namespace) -> None:
    manifest_path = ROOT / "pipelines" / f"{args.name}.yaml"
    if not manifest_path.exists():
        raise SystemExit(f"Pipeline {args.name} not found at {manifest_path}")

    data = load_yaml(manifest_path)
    metadata = data.get("metadata", {})
    phases = data.get("phases", [])

    print(f"# {metadata.get('title', metadata.get('name', args.name))}")
    if metadata.get("description"):
        print(metadata["description"])
    print("\nPhases:")
    for phase in phases:
        print(f"- [{phase.get('id')}] {phase.get('title')}")
        for key in ("workflow", "prompts", "templates", "commands"):
            if phase.get(key):
                print(f"    {key}: {phase[key]}")


def cmd_validate_metadata(args: argparse.Namespace) -> None:
    import jsonschema

    schema = json.loads((ROOT / "schemas" / "asset-metadata.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    metadata_path = Path(args.path)
    with metadata_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        for error in errors:
            loc = ".".join(str(elem) for elem in error.path)
            print(f"[ERROR] {loc}: {error.message}")
        raise SystemExit(1)
    print("Metadata is valid ✅")


def cmd_list_validation_profiles(_: argparse.Namespace) -> None:
    profiles = load_validation_profiles()
    for name, cfg in profiles.items():
        print(f"{name}: {cfg.get('description', '')}")
        if cfg.get("commands"):
            for command in cfg["commands"]:
                print(f"  - {command}")


def load_validation_profiles() -> Dict[str, Any]:
    return load_yaml(ROOT / "configs" / "validation-profiles.yaml")["profiles"]


def cmd_run_validation(args: argparse.Namespace) -> None:
    profiles = load_validation_profiles()
    if args.profile not in profiles:
        raise SystemExit(f"Validation profile '{args.profile}' not found.")

    profile = profiles[args.profile]
    commands: List[str] = profile.get("commands", [])

    if not commands:
        print("No commands defined for this profile.")
        return

    target = args.target
    for command in commands:
        formatted = format_command(command, target)
        print(f"→ Running: {formatted}")
        subprocess.run(shlex.split(formatted), check=True)


def format_command(command: str, target: Optional[str]) -> str:
    if target:
        if "{target}" in command:
            return command.format(target=target)
        # Append target if command appears to expect a path parameter
        return f"{command} {target}"
    return command


def cmd_scaffold_metadata(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    roles = [role.strip() for role in args.roles.split(",")] if args.roles else []

    data = {
        "id": args.id,
        "type": args.type,
        "title": args.title,
        "roles": roles or ["<role-slug>"],
        "stage": args.stage,
        "proficiency_target": args.proficiency,
        "competencies": [
            {
                "id": "<competency-id>",
                "level": args.proficiency,
            }
        ],
        "dependencies": [],
        "validation_profile": args.validation_profile,
        "status": "draft",
        "owners": [
            {
                "name": args.owner,
                "github": args.owner_github,
                "role": "author",
            }
        ],
        "metadata_version": "1.0.0",
    }

    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Metadata scaffold created at {output}")


def cmd_generate_mkdocs_nav(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path {root} does not exist")

    entries: List[str] = []
    for metadata_path in sorted(root.rglob("metadata.yaml")):
        data = load_yaml(metadata_path)
        if data.get("type") != "module":
            continue
        title = data.get("title", metadata_path.parent.name)
        lesson_path = (metadata_path.parent / "lesson.md").resolve()
        if not lesson_path.exists():
            for candidate in ("README.md", "lecture-notes.md"):
                candidate_path = (metadata_path.parent / candidate).resolve()
                if candidate_path.exists():
                    lesson_path = candidate_path
                    break
        try:
            rel_path = lesson_path.relative_to(ROOT)
        except ValueError:
            rel_path = lesson_path
        entries.append(f'    - "{title}": {rel_path.as_posix()}')

    if not entries:
        print("# No module metadata files found.")
        return

    print("nav:")
    print("  - Modules:")
    for line in entries:
        print(line)


def cmd_export_graph(args: argparse.Namespace) -> None:
    roots = [Path(path).resolve() for path in args.paths]
    metadata_items: List[Dict[str, Any]] = []

    for root in roots:
        if not root.exists():
            print(f"# Skipping missing path: {root}")
            continue
        for metadata_path in root.rglob("metadata.yaml"):
            data = load_yaml(metadata_path)
            data["_path"] = metadata_path
            metadata_items.append(data)

    roles: Dict[str, Dict[str, Any]] = {}
    modules: List[Dict[str, Any]] = []
    projects: List[Dict[str, Any]] = []

    for item in metadata_items:
        item_type = item.get("type")
        role_slugs = item.get("roles", [])

        for role in role_slugs:
            entry = roles.setdefault(
                role,
                {"slug": role, "name": role.replace("-", " ").title(), "modules": [], "projects": []},
            )
            if item_type == "module":
                entry["modules"].append(item.get("id"))
            elif item_type == "project":
                entry["projects"].append(item.get("id"))

        if item_type == "module":
            modules.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "roleSlugs": role_slugs,
                    "stage": item.get("stage"),
                    "dependencies": item.get("dependencies", []),
                    "validationProfile": item.get("validation_profile"),
                    "path": str(item.get("_path").parent.relative_to(ROOT)),
                }
            )
        elif item_type == "project":
            projects.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "roleSlugs": role_slugs,
                    "difficulty": item.get("stage"),
                    "dependencies": item.get("dependencies", []),
                    "sharedComponents": item.get("shared_components", []),
                    "path": str(item.get("_path").parent.relative_to(ROOT)),
                }
            )

    graph = {
        "roles": list(roles.values()),
        "modules": modules,
        "projects": projects,
    }

    output = args.output
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        print(f"Graph exported to {out_path}")
    else:
        print(json.dumps(graph, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum automation helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pipelines", help="List available pipeline manifests").set_defaults(
        func=cmd_list_pipelines
    )

    show_pipeline = sub.add_parser("pipeline", help="Show details for a pipeline manifest")
    show_pipeline.add_argument("name")
    show_pipeline.set_defaults(func=cmd_show_pipeline)

    validate_metadata = sub.add_parser("validate-metadata", help="Validate asset metadata YAML against schema")
    validate_metadata.add_argument("path")
    validate_metadata.set_defaults(func=cmd_validate_metadata)

    sub.add_parser("validation-profiles", help="List validation profile commands").set_defaults(
        func=cmd_list_validation_profiles
    )

    run_validation = sub.add_parser("run-validation", help="Execute a validation profile against a target path")
    run_validation.add_argument("profile", help="Validation profile name")
    run_validation.add_argument("target", help="Target file or directory")
    run_validation.set_defaults(func=cmd_run_validation)

    scaffold_metadata = sub.add_parser("scaffold-metadata", help="Create a metadata YAML stub")
    scaffold_metadata.add_argument("output", help="Output path for metadata YAML")
    scaffold_metadata.add_argument("--id", required=True, help="Asset identifier (e.g., MOD-01)")
    scaffold_metadata.add_argument("--type", required=True, choices=["module", "project", "exercise", "assessment", "solution"])
    scaffold_metadata.add_argument("--title", required=True, help="Human-friendly title")
    scaffold_metadata.add_argument("--roles", default="", help="Comma-separated list of role slugs")
    scaffold_metadata.add_argument("--stage", default="core", choices=["foundational", "core", "advanced", "capstone"])
    scaffold_metadata.add_argument("--proficiency", default="working", choices=["awareness", "working", "proficient", "expert"])
    scaffold_metadata.add_argument("--validation-profile", default="python-light", help="Validation profile to apply")
    scaffold_metadata.add_argument("--owner", default="Curriculum Team", help="Primary owner name")
    scaffold_metadata.add_argument("--owner-github", default="@curriculum-team", help="Primary owner GitHub handle")
    scaffold_metadata.set_defaults(func=cmd_scaffold_metadata)

    mkdocs_nav = sub.add_parser("generate-mkdocs-nav", help="Generate MkDocs nav from module metadata")
    mkdocs_nav.add_argument("root", help="Root directory to scan (e.g., modules/)")
    mkdocs_nav.set_defaults(func=cmd_generate_mkdocs_nav)

    export_graph = sub.add_parser("export-graph", help="Export curriculum graph JSON from metadata")
    export_graph.add_argument("paths", nargs="+", help="Directories to scan for metadata")
    export_graph.add_argument("--output", help="Optional output file (defaults to stdout)")
    export_graph.set_defaults(func=cmd_export_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
