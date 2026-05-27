"""Command-line interface for AICG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent_cli import AgentLimitReached
from .audit import AuditError, audit_repo
from .bootstrap import (
    BootstrapError,
    CurriculumPlanError,
    bootstrap_role,
    execute_curriculum_plan,
)
from .diff import diff_repo
from .generator import GenerationNotConfigured, generate_all_pending, generate_from_plan
from .gitops import GitOpsError, prepare_pr
from .propagate import PropagateError, propagate_repo
from .inventory import InventoryError, WorkspaceInventory, default_workspace
from .org_config import ManifestError, load_manifest, state_dir_for_manifest
from .org_runner import (
    OrgRunnerError,
    generate_research_packets,
    plan_monthly_release,
    run_daily_remediation,
    run_org_audit,
    steward_report,
    sync_repositories,
)
from .planner import plan_from_audit
from .state import read_state
from .validator import validate_repo
from .verify import VerifyError, verify_repo


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        AuditError,
        BootstrapError,
        CurriculumPlanError,
        GitOpsError,
        InventoryError,
        ManifestError,
        OrgRunnerError,
        PropagateError,
        VerifyError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aicg", description="AI curriculum runner")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace containing paired curriculum repos. Defaults to the parent of this repo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit curriculum repo structure and guardrails")
    add_repo_args(audit_parser)
    audit_parser.set_defaults(func=cmd_audit)

    plan_parser = subparsers.add_parser("plan", help="Turn audit gaps into deterministic work items")
    add_repo_args(plan_parser)
    plan_parser.set_defaults(func=cmd_plan)

    generate_parser = subparsers.add_parser("generate", help="Write prompt packet and invoke local agent")
    add_repo_args(generate_parser)
    generate_parser.add_argument("--work-id", default=None, help="Specific work item to generate")
    generate_parser.add_argument(
        "--all",
        action="store_true",
        help="Drain every pending work item. Stops on subscription-limit or missing-config.",
    )
    generate_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Additional aicg.yaml/json config path containing generator_command.",
    )
    generate_parser.set_defaults(func=cmd_generate)

    validate_parser = subparsers.add_parser("validate", help="Run objective validation checks")
    add_repo_args(validate_parser)
    validate_parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit 0 after writing the validation report.",
    )
    validate_parser.set_defaults(func=cmd_validate)

    propagate_parser = subparsers.add_parser(
        "propagate",
        help="Append shipped work items to VERSIONS.md (and suggest CURRICULUM.md edits)",
    )
    add_repo_args(propagate_parser)
    propagate_parser.add_argument(
        "--work-id",
        default=None,
        help="Limit propagation to a single work item.",
    )
    propagate_parser.set_defaults(func=cmd_propagate)

    diff_parser = subparsers.add_parser(
        "diff",
        help="Show what the agent changed for a work item",
    )
    add_repo_args(diff_parser)
    diff_parser.add_argument(
        "--work-id",
        default=None,
        help="Limit expected-path matching to one work item.",
    )
    diff_parser.add_argument(
        "--full",
        action="store_true",
        help="Include the unified `git diff` output in the report.",
    )
    diff_parser.set_defaults(func=cmd_diff)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Confirm generated artifacts match the plan's output contract",
    )
    add_repo_args(verify_parser)
    verify_parser.add_argument(
        "--work-id",
        default=None,
        help="Verify a specific work item id; defaults to every item in the plan.",
    )
    verify_parser.set_defaults(func=cmd_verify)

    pr_parser = subparsers.add_parser("pr", help="Create guarded branch, commit, and GitHub PR")
    add_repo_args(pr_parser)
    pr_parser.add_argument(
        "--work-id",
        default=None,
        help="Specific work item id to PR. Defaults to the highest-priority item.",
    )
    pr_parser.add_argument("--auto-merge", action="store_true", help="Require auto-merge guardrails")
    pr_parser.set_defaults(func=cmd_pr)

    run_parser = subparsers.add_parser("run", help="Run a predefined orchestration loop")
    add_repo_args(run_parser)
    run_parser.add_argument("--mode", choices=["pilot"], default="pilot")
    run_parser.set_defaults(func=cmd_run)

    org_parser = subparsers.add_parser("org", help="Run org-level automation operations")
    org_subparsers = org_parser.add_subparsers(dest="org_command", required=True)

    org_sync = org_subparsers.add_parser("sync", help="Clone or fast-forward all manifest repos")
    add_org_args(org_sync)
    org_sync.add_argument("--dry-run", action="store_true", help="Print actions without running git.")
    org_sync.set_defaults(func=cmd_org_sync)

    org_release = org_subparsers.add_parser("release", help="Create monthly release tags")
    add_org_args(org_release)
    org_release.add_argument(
        "--apply",
        action="store_true",
        help="Actually create and push tags. Omit for dry-run planning.",
    )
    org_release.set_defaults(func=cmd_org_release)

    org_research = org_subparsers.add_parser(
        "research",
        help="Create monthly job-requirements research packets",
    )
    add_org_args(org_research)
    org_research.add_argument("--month", default=None, help="Month key such as 2026-05.")
    org_research.set_defaults(func=cmd_org_research)

    org_audit = org_subparsers.add_parser("audit", help="Audit all solution repos and write queue")
    add_org_args(org_audit)
    org_audit.set_defaults(func=cmd_org_audit)

    org_daily = org_subparsers.add_parser("daily", help="Consume the next ready work item")
    add_org_args(org_daily)
    org_daily.set_defaults(func=cmd_org_daily)

    org_execute_plan = org_subparsers.add_parser(
        "execute-plan",
        help="Scaffold module + project skeletons from a curriculum-plan.json",
    )
    add_org_args(org_execute_plan)
    org_execute_plan.add_argument(
        "--role", required=True, help="Role id whose plan should be executed."
    )
    org_execute_plan.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Optional explicit path to curriculum-plan.json (defaults to .aicg/ in the learning repo).",
    )
    org_execute_plan.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-scaffold modules/projects that already have on-disk content.",
    )
    org_execute_plan.set_defaults(func=cmd_org_execute_plan)

    org_bootstrap = org_subparsers.add_parser(
        "bootstrap-role",
        help="Scaffold a new role's learning + solutions repos and research prompt",
    )
    add_org_args(org_bootstrap)
    org_bootstrap.add_argument(
        "--role",
        required=True,
        help="Role id (lowercase, hyphenated, e.g. 'data-engineer').",
    )
    org_bootstrap.add_argument("--title", required=True, help="Human-readable role title.")
    org_bootstrap.add_argument(
        "--level", type=int, required=True, help="Numeric seniority level (e.g. 25)."
    )
    org_bootstrap.add_argument(
        "--description",
        default=None,
        help="One-line description used inside the bootstrap prompt and READMEs.",
    )
    org_bootstrap.add_argument(
        "--overwrite",
        action="store_true",
        help="Scaffold even when the target directories already exist with content.",
    )
    org_bootstrap.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Skip appending the role to the org manifest (print a YAML snippet instead).",
    )
    org_bootstrap.add_argument(
        "--create-remotes",
        action="store_true",
        help="Create the matching GitHub repos via `gh repo create --push`.",
    )
    org_bootstrap.set_defaults(func=cmd_org_bootstrap_role)

    org_steward = org_subparsers.add_parser(
        "steward",
        help="Inspect open PRs and merge ones that clear CI + guardrails",
    )
    add_org_args(org_steward)
    org_steward.add_argument(
        "--apply",
        action="store_true",
        help="Actually merge eligible PRs. Omit for dry-run / planning.",
    )
    org_steward.add_argument(
        "--ci-timeout",
        type=int,
        default=600,
        help="Per-PR CI rollup timeout in seconds (default 600).",
    )
    org_steward.add_argument(
        "--ci-poll",
        type=int,
        default=30,
        help="Per-PR CI rollup poll interval in seconds (default 30).",
    )
    org_steward.add_argument(
        "--merge-method",
        choices=["squash", "merge", "rebase"],
        default="squash",
        help="Merge strategy for `gh pr merge --auto` (default squash).",
    )
    org_steward.set_defaults(func=cmd_org_steward)

    return parser


def add_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=argparse.SUPPRESS,
        help="Workspace containing paired curriculum repos.",
    )
    parser.add_argument("--repo", required=True, help="Target curriculum repository name")
    parser.add_argument("--module", default=None, help="Optional module id such as mod-001-...")


def add_org_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=argparse.SUPPRESS,
        help="Workspace containing curriculum repos.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Org manifest path. Defaults to config/aicg-org.yaml.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Override org state directory.",
    )


def resolve_workspace(args: argparse.Namespace) -> Path:
    return (getattr(args, "workspace", None) or default_workspace()).resolve()


def target_repo_path(workspace: Path, repo_name: str) -> Path:
    return WorkspaceInventory(workspace).require(repo_name).path


def resolve_manifest(args: argparse.Namespace):
    return load_manifest(args.manifest)


def resolve_org_state_dir(args: argparse.Namespace, manifest):
    return state_dir_for_manifest(manifest, args.state_dir)


def cmd_audit(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    report = audit_repo(workspace, args.repo, module=args.module)
    print_audit_summary(report)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    audit = audit_repo(workspace, args.repo, module=args.module)
    repo_path = target_repo_path(workspace, args.repo)
    plan = plan_from_audit(audit, repo_path=repo_path)
    print(f"Wrote work plan for {args.repo}: {plan['work_item_count']} item(s)")
    for item in plan["work_items"]:
        print(f"- {item['id']} ({len(item['exercises'])} exercise solution(s))")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    repo_path = target_repo_path(workspace, args.repo)
    try:
        plan = read_state(repo_path, "work-plan.json")
    except FileNotFoundError:
        audit = audit_repo(workspace, args.repo, module=args.module)
        plan = plan_from_audit(audit, repo_path=repo_path)

    if args.all and args.work_id:
        print("error: --all and --work-id are mutually exclusive", file=sys.stderr)
        return 2

    if args.all:
        try:
            batch = generate_all_pending(
                repo_path,
                plan,
                module=args.module,
                config_paths=args.config,
            )
        except GenerationNotConfigured as exc:
            print("Prompt packet ready; no generator command configured.")
            print(f"Prompt: {exc.prompt_path}")
            print(f"Expected output directory: {exc.output_dir}")
            return 2
        print(
            f"Generated {batch['completed_count']}/{batch['pending_count']} pending work item(s)."
        )
        if batch.get("deferred"):
            print(
                f"Stopped at {batch['deferred']['work_id']}: subscription "
                f"limit ({batch['deferred']['limit_scope']}); retry after "
                f"{batch['deferred']['retry_after']}."
            )
            return 75
        return 0

    try:
        state = generate_from_plan(
            repo_path,
            plan,
            module=args.module,
            work_id=args.work_id,
            config_paths=args.config,
        )
    except GenerationNotConfigured as exc:
        print("Prompt packet ready; no generator command configured.")
        print(f"Prompt: {exc.prompt_path}")
        print(f"Expected output directory: {exc.output_dir}")
        return 2
    except AgentLimitReached as exc:
        print("Agent subscription limit reached; work deferred.")
        print(f"Retry after: {exc.result.retry_after}")
        return 75
    print(f"Generated work item {state['work_id']} with status {state['status']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    report = validate_repo(workspace, args.repo, module=args.module)
    print(f"Validation {report['status']} for {args.repo}")
    for check in report["checks"]:
        print(f"- {check['name']}: {check['status']} ({check['finding_count']} finding(s))")
    if args.report_only:
        return 0
    return 0 if report["status"] == "passed" else 1


def cmd_propagate(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    report = propagate_repo(workspace, args.repo, work_id=args.work_id)
    if report["status"] == "no_items":
        print(f"Propagate noop: no verified work items in plan for {args.repo}.")
        return 0
    print(
        f"Propagated {len(report['updated'])} work item(s) to "
        f"{report['versions_path']}; {len(report['already_present'])} already present."
    )
    for entry in report["updated"][:8]:
        print(f"  + {entry['date']} {entry['work_id']} ({entry.get('module') or entry.get('project') or entry.get('type')})")
    for suggestion in report["curriculum_suggestions"][:4]:
        scope = suggestion.get("scope")
        if scope:
            print(f"  ~ CURRICULUM.md: review the row for `{scope}`.")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    repo_path = target_repo_path(workspace, args.repo)
    report = diff_repo(repo_path, work_id=args.work_id, show_full=args.full)
    summary = report["summary"]
    print(
        f"Diff for {args.repo}: {summary['added']} added, "
        f"{summary['modified']} modified, {summary['deleted']} deleted, "
        f"{summary['untracked']} untracked "
        f"({summary['unexpected']} unexpected)"
    )
    for entry in report["entries"][:25]:
        tag = "  " if entry["expected"] else " !"
        print(
            f"{tag} {entry['status']:>9}  {entry['path']}"
            + (f"  ({entry['line_count']} lines)" if entry['line_count'] else "")
        )
        for line in entry["preview_head"][:6]:
            print(f"    | {line}")
        if entry["preview_tail"]:
            print("    | ...")
            for line in entry["preview_tail"][:4]:
                print(f"    | {line}")
    if args.full and report.get("full_diff"):
        print("\n--- full diff ---")
        print(report["full_diff"])
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    report = verify_repo(workspace, args.repo, work_id=args.work_id)
    print(f"Verify {report['status']} for {args.repo} ({report['work_item_count']} item(s))")
    for item in report["work_items"]:
        print(f"- {item['work_id']}: {item['status']} ({item['finding_count']} finding(s))")
        for action in item["actions"][:4]:
            print(f"    {action['status']:>10}  {action['type']:<22}  {action['path']}")
        if len(item["actions"]) > 4:
            print(f"    ... {len(item['actions']) - 4} more action(s) in .aicg/verify-report.json")
    return 0 if report["status"] in {"verified", "no_items"} else 1


def cmd_pr(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    repo_path = target_repo_path(workspace, args.repo)
    work_plan = read_state(repo_path, "work-plan.json")
    audit = read_state(repo_path, "audit-report.json")
    validation = read_state(repo_path, "validation-report.json")
    result = prepare_pr(
        repo_path,
        work_plan=work_plan,
        audit_report=audit,
        validation_report=validation,
        auto_merge=args.auto_merge,
        work_id=args.work_id,
    )
    print(f"Created PR: {result['pr_url']}")
    print(f"Branch: {result['branch']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    workspace = resolve_workspace(args)
    audit = audit_repo(workspace, args.repo, module=args.module)
    repo_path = target_repo_path(workspace, args.repo)
    plan = plan_from_audit(audit, repo_path=repo_path)
    print_audit_summary(audit)
    print(f"Planned {plan['work_item_count']} work item(s).")
    if not plan["work_items"]:
        validate_report = validate_repo(workspace, args.repo, module=args.module)
        print(f"Validation {validate_report['status']}.")
        return 0 if validate_report["status"] == "passed" else 1
    try:
        run_state = generate_from_plan(repo_path, plan, module=args.module)
    except AgentLimitReached as exc:
        print("Pilot stopped because the content agent hit a subscription limit.")
        print(f"Retry after: {exc.result.retry_after}")
        return 75
    except GenerationNotConfigured as exc:
        print("Pilot stopped before content mutation because no generator command is configured.")
        print(f"Prompt: {exc.prompt_path}")
        return 2
    verify_report = verify_repo(
        workspace, args.repo, work_id=run_state.get("work_id")
    )
    print(f"Verify {verify_report['status']} ({verify_report['work_item_count']} item(s)).")
    if verify_report["status"] == "verified":
        propagate_report = propagate_repo(
            workspace, args.repo, work_id=run_state.get("work_id")
        )
        print(
            f"Propagate {propagate_report['status']}: "
            f"{len(propagate_report['updated'])} VERSIONS.md row(s) added."
        )
    return 0 if verify_report["status"] in {"verified", "no_items"} else 1


def cmd_org_sync(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = sync_repositories(manifest, resolve_workspace(args), dry_run=args.dry_run)
    print(f"Org sync {'dry-run' if args.dry_run else 'completed'}: {len(report['actions'])} repo(s)")
    for action in report["actions"]:
        print(f"- {action['repo']}: {action['status']} ({action['command']})")
    return 0


def cmd_org_release(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = plan_monthly_release(manifest, resolve_workspace(args), apply=args.apply)
    state_dir = resolve_org_state_dir(args, manifest)
    state_dir.mkdir(parents=True, exist_ok=True)
    from .state import write_json

    write_json(state_dir / "monthly-release-plan.json", report)
    print(
        f"Monthly release {'applied' if args.apply else 'planned'} for tag {report['tag']}: "
        f"{len(report['actions'])} repo(s)"
    )
    for action in report["actions"]:
        print(f"- {action['repo']}: {action['status']}")
    return 0 if all(action["status"] != "failed" for action in report["actions"]) else 1


def cmd_org_research(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = generate_research_packets(
        manifest,
        resolve_workspace(args),
        month=args.month,
        state_dir=resolve_org_state_dir(args, manifest),
    )
    print(f"Research packets ready for {report['month']}: {len(report['packets'])} role(s)")
    for packet in report["packets"]:
        print(f"- {packet['role']}: {packet['prompt_path']}")
    return 0


def cmd_org_audit(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    queue = run_org_audit(
        manifest,
        resolve_workspace(args),
        state_dir=resolve_org_state_dir(args, manifest),
    )
    print(f"Org audit queued {queue['work_item_count']} work item(s)")
    for repo in queue["repo_reports"]:
        print(f"- {repo['repo']}: {repo['status']}")
    return 0


def cmd_org_daily(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = run_daily_remediation(
        manifest,
        resolve_workspace(args),
        state_dir=resolve_org_state_dir(args, manifest),
    )
    print(f"Daily remediation status: {report['status']}")
    if report.get("selected"):
        selected = report["selected"]
        print(f"- selected: {selected['repo']} {selected['work_id']}")
    if report.get("prompt_path"):
        print(f"- prompt: {report['prompt_path']}")
    return 0


def cmd_org_execute_plan(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = execute_curriculum_plan(
        manifest=manifest,
        workspace=resolve_workspace(args),
        role_id=args.role,
        plan_path=args.plan,
        overwrite=args.overwrite,
        state_dir=resolve_org_state_dir(args, manifest),
    )
    print(
        f"Executed curriculum plan for {report['role_id']}: "
        f"{len(report['modules_created'])} module(s) scaffolded, "
        f"{len(report['modules_skipped'])} skipped; "
        f"{len(report['projects_created'])} project(s) scaffolded, "
        f"{len(report['projects_skipped'])} skipped; "
        f"{report['files_written_count']} file(s) written."
    )
    if report["modules_created"]:
        print("Modules scaffolded:")
        for mod in report["modules_created"][:12]:
            print(f"  - {mod}")
        if len(report["modules_created"]) > 12:
            print(f"  ... {len(report['modules_created']) - 12} more")
    return 0


def cmd_org_bootstrap_role(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = bootstrap_role(
        manifest=manifest,
        workspace=resolve_workspace(args),
        role_id=args.role,
        title=args.title,
        level=args.level,
        description=args.description,
        overwrite=args.overwrite,
        write_manifest=not args.no_update_manifest,
        create_remotes=args.create_remotes,
        state_dir=resolve_org_state_dir(args, manifest),
    )
    plan = report["plan"]
    print(
        f"Bootstrapped role '{plan['role_id']}' ({plan['title']}, level "
        f"{plan['level']})"
    )
    print(f"- Learning repo: {plan['learning_path']}")
    print(f"- Solutions repo: {plan['solution_path']}")
    print(f"- Prompt packet: {plan['prompt_path']}")
    manifest_update = report.get("manifest_update")
    if manifest_update and manifest_update.get("status") == "yaml_manifest_manual_update_required":
        print("Manifest is YAML; append this snippet under `roles:`:")
        print(manifest_update["snippet"])
    elif manifest_update and manifest_update.get("status") == "appended_json":
        print(f"- Manifest updated: {manifest_update['manifest_path']}")
    remotes = report.get("remotes")
    if remotes:
        for action in remotes.get("actions", []):
            outcome = "ok" if action["returncode"] == 0 else "failed"
            print(f"- gh repo create {action['repo']}: {outcome}")
    return 0


def cmd_org_steward(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = steward_report(
        manifest,
        resolve_workspace(args),
        state_dir=resolve_org_state_dir(args, manifest),
        apply=args.apply,
        ci_timeout_seconds=args.ci_timeout,
        ci_poll_seconds=args.ci_poll,
        merge_method=args.merge_method,
    )
    merged = sum(repo.get("merged_count", 0) for repo in report.get("repos", []))
    blocked = sum(repo.get("blocked_count", 0) for repo in report.get("repos", []))
    failed = sum(repo.get("ci_failed_count", 0) for repo in report.get("repos", []))
    pr_count = sum(repo.get("pr_count", 0) for repo in report.get("repos", []))
    mode = "apply" if args.apply else "dry-run"
    print(
        f"Steward {mode}: {pr_count} open PR(s); merged={merged} "
        f"blocked={blocked} ci_failed={failed}"
    )
    for repo in report["repos"]:
        if not repo.get("pr_count"):
            continue
        for pr in repo.get("prs", []):
            print(f"- {repo['repo']}#{pr['pr_number']}: {pr['state']} — {pr.get('title', '')}")
    return 0


def print_audit_summary(report: dict) -> None:
    summary = report["summary"]
    print(
        f"Audit {summary['status']} for {report['repo']['name']}: "
        f"{summary['error_count']} error(s), {summary['warning_count']} warning(s), "
        f"{summary['module_count']} module(s)"
    )
    for gap in report.get("gaps", [])[:12]:
        location = gap.get("expected_path") or gap.get("path") or gap.get("learning_path") or ""
        suffix = f" [{location}]" if location else ""
        print(f"- {gap['severity']}: {gap['type']}: {gap['message']}{suffix}")
    if len(report.get("gaps", [])) > 12:
        print(f"- ... {len(report['gaps']) - 12} more gap(s) in .aicg/audit-report.json")
