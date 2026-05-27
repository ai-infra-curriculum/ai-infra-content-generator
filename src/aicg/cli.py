"""Command-line interface for AICG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent_cli import AgentLimitReached
from .audit import AuditError, audit_repo
from .generator import GenerationNotConfigured, generate_all_pending, generate_from_plan
from .gitops import GitOpsError, prepare_pr
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
        GitOpsError,
        InventoryError,
        ManifestError,
        OrgRunnerError,
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

    org_steward = org_subparsers.add_parser(
        "steward",
        help="Write PR/issue/discussion stewardship report",
    )
    add_org_args(org_steward)
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


def cmd_org_steward(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    report = steward_report(
        manifest,
        resolve_workspace(args),
        state_dir=resolve_org_state_dir(args, manifest),
    )
    print(f"Steward report written for {len(report['repos'])} repo(s); status={report['status']}")
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
