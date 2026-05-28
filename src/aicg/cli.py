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

    audit_links_parser = subparsers.add_parser(
        "audit-links",
        help="HEAD-ping every external URL in committed markdown; emit refresh_links work items",
    )
    add_repo_args(audit_links_parser)
    audit_links_parser.add_argument(
        "--timeout", type=float, default=8.0, help="Per-URL HEAD timeout seconds (default 8)."
    )
    audit_links_parser.add_argument(
        "--workers", type=int, default=16, help="Concurrent HEAD requests (default 16)."
    )
    audit_links_parser.set_defaults(func=cmd_audit_links)

    audit_versions_parser = subparsers.add_parser(
        "audit-versions",
        help="Scan committed markdown for stale version pins from a curated registry",
    )
    add_repo_args(audit_versions_parser)
    audit_versions_parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to version-targets.yaml. Defaults to config/version-targets.yaml.",
    )
    audit_versions_parser.set_defaults(func=cmd_audit_versions)

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
    verify_parser.add_argument(
        "--with-quality-grade",
        action="store_true",
        help="Invoke the configured judge to grade artifact quality.",
    )
    verify_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Org manifest path (only needed when --with-quality-grade is set).",
    )
    verify_parser.set_defaults(func=cmd_verify)

    pr_drive_parser = subparsers.add_parser(
        "pr-drive",
        help=(
            "Run the inline-merge loop against an existing PR: CI wait → "
            "self-heal failed checks → guardrails → reviews → merge."
        ),
    )
    add_repo_args(pr_drive_parser)
    pr_drive_parser.add_argument(
        "--pr",
        type=int,
        required=True,
        help="PR number to drive to merge.",
    )
    pr_drive_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Org manifest path. Defaults to config/aicg-org.yaml.",
    )
    pr_drive_parser.set_defaults(func=cmd_pr_drive)

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
    org_research.add_argument(
        "--apply",
        action="store_true",
        help="After writing prompts, invoke the configured agent on each role's packet so it updates JOB_REQUIREMENTS + curriculum-plan-delta in the learning repo.",
    )
    org_research.add_argument(
        "--no-pr",
        action="store_true",
        help="Skip opening proposal PRs. Useful for first-cycle inspection and tests; the proposal files still get written.",
    )
    org_research.set_defaults(func=cmd_org_research)

    org_promote = org_subparsers.add_parser(
        "promote-plan",
        help="Apply a human-approved research proposal to curriculum-plan.json (run AFTER merging the proposal PR)",
    )
    add_org_args(org_promote)
    org_promote.add_argument(
        "--role",
        default=None,
        help="Role id to promote. Omit to promote every role with a pending proposal.",
    )
    org_promote.set_defaults(func=cmd_org_promote_plan)

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

    org_issues = org_subparsers.add_parser(
        "issues",
        help="Reconcile GitHub issues with the work-queue state (open / comment / close)",
    )
    add_org_args(org_issues)
    org_issues.add_argument(
        "--apply",
        action="store_true",
        help="Actually open / comment / close issues. Omit for dry-run.",
    )
    org_issues.add_argument(
        "--stuck-after",
        type=float,
        default=None,
        help="Hours an item must be deferred before it gets an issue (default 24).",
    )
    org_issues.set_defaults(func=cmd_org_issues)

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

    org_discussions = org_subparsers.add_parser(
        "discussions",
        help="Summarize open GitHub Discussions across the org (dry-run only)",
    )
    add_org_args(org_discussions)
    org_discussions.set_defaults(func=cmd_org_discussions)

    org_list_roles = org_subparsers.add_parser(
        "list-roles",
        help="Print role ids from the manifest, one per line (for shell loops).",
    )
    add_org_args(org_list_roles)
    org_list_roles.set_defaults(func=cmd_org_list_roles)

    org_audit_links = org_subparsers.add_parser(
        "audit-links",
        help="Run the link checker against every solution repo in the manifest",
    )
    add_org_args(org_audit_links)
    org_audit_links.add_argument("--timeout", type=float, default=8.0)
    org_audit_links.add_argument("--workers", type=int, default=16)
    org_audit_links.set_defaults(func=cmd_org_audit_links)

    org_audit_versions = org_subparsers.add_parser(
        "audit-versions",
        help="Run the version-pin scanner against every solution repo in the manifest",
    )
    add_org_args(org_audit_versions)
    org_audit_versions.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="version-targets.yaml. Defaults to config/version-targets.yaml.",
    )
    org_audit_versions.set_defaults(func=cmd_org_audit_versions)

    org_review = org_subparsers.add_parser(
        "review",
        help="LLM freshness review of committed artifacts across all solution repos",
    )
    add_org_args(org_review)
    org_review.add_argument(
        "--max-artifacts",
        type=int,
        default=None,
        help="Cap per-repo artifacts reviewed in this run (default: unlimited).",
    )
    org_review.set_defaults(func=cmd_org_review)

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


def cmd_audit_links(args: argparse.Namespace) -> int:
    from .freshness import audit_links

    workspace = resolve_workspace(args)
    repo_path = target_repo_path(workspace, args.repo)
    report = audit_links(
        repo_path,
        timeout=args.timeout,
        max_workers=args.workers,
    )
    print(
        f"Link audit for {args.repo}: {report['broken_count']} broken / "
        f"{report['urls_unique']} unique URLs across {report['files_scanned']} file(s)"
    )
    for item in report["work_items"][:8]:
        print(f"  ! {item['severity']:>6}  {item['broken_count']:>2}  {item['path']}")
    return 0 if report["broken_count"] == 0 else 1


def cmd_audit_versions(args: argparse.Namespace) -> int:
    from .freshness import audit_versions

    workspace = resolve_workspace(args)
    repo_path = target_repo_path(workspace, args.repo)
    registry = args.registry or _default_version_registry()
    report = audit_versions(repo_path, registry_path=registry)
    print(
        f"Version audit for {args.repo}: {report['finding_count']} stale ref(s) "
        f"across {report['target_count']} target(s)"
    )
    for item in report["work_items"][:8]:
        print(f"  ! {item['severity']:>6}  {item['target']:>14}  {item['path']}")
    return 0 if report["finding_count"] == 0 else 1


def _default_version_registry() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "version-targets.yaml"


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
    judge_config = None
    if getattr(args, "with_quality_grade", False):
        from .judge import JudgeConfig

        manifest = load_manifest(args.manifest)
        judge_config = JudgeConfig.from_manifest(manifest)
        if not judge_config.enabled:
            print(
                "warning: --with-quality-grade requested but manifest's "
                "quality_judge.enabled is false; skipping judge invocation.",
                file=sys.stderr,
            )
    report = verify_repo(
        workspace,
        args.repo,
        work_id=args.work_id,
        judge_config=judge_config,
    )
    print(f"Verify {report['status']} for {args.repo} ({report['work_item_count']} item(s))")
    for item in report["work_items"]:
        print(f"- {item['work_id']}: {item['status']} ({item['finding_count']} finding(s))")
        for action in item["actions"][:4]:
            quality = action.get("quality") or {}
            quality_tag = (
                f"  q={quality['score']}/100"
                if isinstance(quality.get("score"), int)
                else ""
            )
            print(
                f"    {action['status']:>10}  {action['type']:<22}  "
                f"{action['path']}{quality_tag}"
            )
        if len(item["actions"]) > 4:
            print(f"    ... {len(item['actions']) - 4} more action(s) in .aicg/verify-report.json")
    return 0 if report["status"] in {"verified", "no_items"} else 1


def cmd_pr_drive(args: argparse.Namespace) -> int:
    """Run the inline-merge loop (with CI self-heal) on an existing PR."""
    import json as _json
    import subprocess as _sp

    from .org_runner import _drive_pr_to_merge

    workspace = resolve_workspace(args)
    repo_path = target_repo_path(workspace, args.repo)
    manifest = load_manifest(args.manifest)

    # Look up the PR's url + head branch via gh.
    completed = _sp.run(
        [
            "gh",
            "pr",
            "view",
            str(args.pr),
            "--json",
            "url,headRefName,state",
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print(f"error: gh pr view {args.pr} failed: {completed.stderr.strip()}", file=sys.stderr)
        return 1
    try:
        pr = _json.loads(completed.stdout)
    except _json.JSONDecodeError as exc:
        print(f"error: could not parse gh pr view output: {exc}", file=sys.stderr)
        return 1

    if pr.get("state") != "OPEN":
        print(f"PR #{args.pr} is not OPEN (state={pr.get('state')}); nothing to do.")
        return 0

    print(
        f"Driving PR #{args.pr} ({pr['url']}) — branch `{pr['headRefName']}`. "
        "This may invoke the agent and push commits."
    )
    outcome = _drive_pr_to_merge(
        manifest=manifest,
        repo_path=repo_path,
        pr_url=pr["url"],
        branch=pr["headRefName"],
    )
    print(f"Outcome: {outcome.get('status')}")
    for entry in outcome.get("history", [])[-10:]:
        print(f"  - {entry}")
    if outcome.get("status") in {"merged", "auto_merge_enabled"}:
        return 0
    if outcome.get("status") in {"review_blocked", "guardrails_blocked"}:
        return 0  # not a failure — surfaces to the operator's attention
    return 1


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
    from .research import ResearchError, research_apply

    manifest = resolve_manifest(args)
    state_dir = resolve_org_state_dir(args, manifest)
    workspace = resolve_workspace(args)
    report = generate_research_packets(
        manifest,
        workspace,
        month=args.month,
        state_dir=state_dir,
    )
    print(f"Research packets ready for {report['month']}: {len(report['packets'])} role(s)")
    for packet in report["packets"]:
        print(f"- {packet['role']}: {packet['prompt_path']}")

    if not getattr(args, "apply", False):
        return 0

    try:
        apply_report = research_apply(
            manifest,
            workspace,
            month=args.month,
            state_dir=state_dir,
            open_pr=not getattr(args, "no_pr", False),
        )
    except ResearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    proposal_ready = sum(1 for r in apply_report["roles"] if r["status"] == "proposal_ready")
    no_delta = sum(1 for r in apply_report["roles"] if r["status"] == "applied_no_delta")
    deferred = sum(1 for r in apply_report["roles"] if r["status"] == "deferred")
    failed = sum(1 for r in apply_report["roles"] if r["status"] == "agent_failed")
    skipped = sum(
        1
        for r in apply_report["roles"]
        if r["status"] not in {"proposal_ready", "applied_no_delta", "deferred", "agent_failed"}
    )
    print(
        f"Research apply: {proposal_ready} proposal(s), {no_delta} no-delta, "
        f"{deferred} deferred, {failed} failed, {skipped} skipped."
    )
    for role in apply_report["roles"]:
        if role["status"] == "proposal_ready":
            summary = role.get("validation", {})
            counts = summary.get("accepted_counts", {})
            rejected = summary.get("rejected_count", 0)
            print(
                f"  + {role['role']}: proposal "
                f"(mods={counts.get('modules', 0)} "
                f"ex={counts.get('exercises', 0)} "
                f"proj={counts.get('projects', 0)} "
                f"rejected={rejected})"
            )
        elif role["status"] == "applied_no_delta":
            print(f"  = {role['role']}: requirements updated, no curriculum additions")
        elif role["status"] == "deferred":
            print(f"  ~ {role['role']}: deferred ({role.get('reason', '')})")
        elif role["status"] == "agent_failed":
            print(f"  ! {role['role']}: agent_failed (rc={role.get('returncode')})")
        else:
            print(f"  - {role['role']}: {role['status']}")
    return 0 if failed == 0 else 1


def cmd_org_promote_plan(args: argparse.Namespace) -> int:
    from .research import ResearchError, promote_plan

    manifest = resolve_manifest(args)
    workspace = resolve_workspace(args)
    roles_to_promote = (
        [next((r for r in manifest.roles if r.id == args.role), None)]
        if args.role
        else list(manifest.roles)
    )
    if args.role and roles_to_promote[0] is None:
        print(f"error: role not in manifest: {args.role}", file=sys.stderr)
        return 1

    promoted_count = 0
    skipped_count = 0
    for role in roles_to_promote:
        if role is None:
            continue
        learning_path = workspace / role.learning_repo
        try:
            report = promote_plan(learning_path)
        except ResearchError:
            skipped_count += 1
            continue
        added = report.get("merge_report", {}).get("added", [])
        print(
            f"+ {role.id}: promoted ({len(added)} item(s) added to curriculum-plan.json)"
        )
        promoted_count += 1
    if skipped_count:
        print(f"({skipped_count} role(s) had no pending proposal — skipped)")
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


def cmd_org_issues(args: argparse.Namespace) -> int:
    from .issues import IssuesError, issues_run

    manifest = resolve_manifest(args)
    try:
        report = issues_run(
            manifest=manifest,
            workspace=resolve_workspace(args),
            state_dir=resolve_org_state_dir(args, manifest),
            apply=args.apply,
            stuck_after_hours=args.stuck_after,
        )
    except IssuesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    mode = "apply" if args.apply else "dry-run"
    opened = sum(repo.get("opened", 0) for repo in report["repos"])
    commented = sum(repo.get("commented", 0) for repo in report["repos"])
    closed = sum(repo.get("closed", 0) for repo in report["repos"])
    print(
        f"Issues {mode}: {opened} opened, {commented} commented, {closed} closed."
    )
    for repo in report["repos"]:
        for decision in repo.get("decisions", [])[:6]:
            print(
                f"- {repo['repo']}/{decision['work_id']}: "
                f"{decision['action']} — {decision['reason']}"
            )
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


def cmd_org_audit_links(args: argparse.Namespace) -> int:
    from .freshness import audit_links
    from .state import write_json

    manifest = resolve_manifest(args)
    workspace = resolve_workspace(args)
    state_dir = resolve_org_state_dir(args, manifest)
    state_dir.mkdir(parents=True, exist_ok=True)

    all_work_items: list = []
    repo_summaries: list = []
    for repo in manifest.repo_names:
        repo_path = workspace / repo
        if not repo_path.exists():
            continue
        report = audit_links(
            repo_path, timeout=args.timeout, max_workers=args.workers
        )
        all_work_items.extend(report["work_items"])
        repo_summaries.append(
            {
                "repo": repo,
                "files_scanned": report["files_scanned"],
                "urls_unique": report["urls_unique"],
                "broken_count": report["broken_count"],
                "work_items": len(report["work_items"]),
            }
        )

    org_report = {
        "schema_version": 1,
        "operation": "org_audit_links",
        "repo_count": len(repo_summaries),
        "broken_total": sum(r["broken_count"] for r in repo_summaries),
        "work_item_total": len(all_work_items),
        "repos": repo_summaries,
        "work_items": all_work_items,
    }
    write_json(state_dir / "freshness-links-report.json", org_report)
    print(
        f"Link audit: {org_report['broken_total']} broken across "
        f"{org_report['repo_count']} repo(s); {org_report['work_item_total']} work item(s)"
    )
    for repo in repo_summaries:
        if repo["broken_count"]:
            print(f"  ! {repo['repo']}: {repo['broken_count']} broken")
    return 0


def cmd_org_audit_versions(args: argparse.Namespace) -> int:
    from .freshness import audit_versions
    from .state import write_json

    manifest = resolve_manifest(args)
    workspace = resolve_workspace(args)
    state_dir = resolve_org_state_dir(args, manifest)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry = args.registry or _default_version_registry()

    all_work_items: list = []
    repo_summaries: list = []
    for repo in manifest.repo_names:
        repo_path = workspace / repo
        if not repo_path.exists():
            continue
        report = audit_versions(repo_path, registry_path=registry)
        all_work_items.extend(report["work_items"])
        repo_summaries.append(
            {
                "repo": repo,
                "finding_count": report["finding_count"],
                "work_items": len(report["work_items"]),
            }
        )

    org_report = {
        "schema_version": 1,
        "operation": "org_audit_versions",
        "repo_count": len(repo_summaries),
        "finding_total": sum(r["finding_count"] for r in repo_summaries),
        "work_item_total": len(all_work_items),
        "repos": repo_summaries,
        "work_items": all_work_items,
    }
    write_json(state_dir / "freshness-versions-report.json", org_report)
    print(
        f"Version audit: {org_report['finding_total']} stale ref(s) across "
        f"{org_report['repo_count']} repo(s); {org_report['work_item_total']} work item(s)"
    )
    for repo in repo_summaries:
        if repo["finding_count"]:
            print(f"  ! {repo['repo']}: {repo['finding_count']} ref(s)")
    return 0


def cmd_org_review(args: argparse.Namespace) -> int:
    from .freshness import review_existing_artifacts
    from .judge import JudgeConfig
    from .state import write_json

    manifest = resolve_manifest(args)
    judge_config = JudgeConfig.from_manifest(manifest)
    if not judge_config.enabled:
        print(
            "warning: quality_judge.enabled is false; review will mark every "
            "artifact 'skipped'. Enable judge in the manifest to actually scan.",
            file=sys.stderr,
        )

    workspace = resolve_workspace(args)
    state_dir = resolve_org_state_dir(args, manifest)
    state_dir.mkdir(parents=True, exist_ok=True)

    all_work_items: list = []
    repo_summaries: list = []
    for repo in manifest.solution_repo_names:
        repo_path = workspace / repo
        if not repo_path.exists():
            continue
        report = review_existing_artifacts(
            repo_path,
            judge_config=judge_config,
            max_artifacts=args.max_artifacts,
        )
        all_work_items.extend(report["work_items"])
        repo_summaries.append(
            {
                "repo": repo,
                "artifacts_reviewed": report["artifacts_reviewed"],
                "stale_count": report["stale_count"],
                "deferred": bool(report.get("deferred")),
                "work_items": len(report["work_items"]),
            }
        )

    org_report = {
        "schema_version": 1,
        "operation": "org_review",
        "repo_count": len(repo_summaries),
        "stale_total": sum(r["stale_count"] for r in repo_summaries),
        "work_item_total": len(all_work_items),
        "repos": repo_summaries,
        "work_items": all_work_items,
    }
    write_json(state_dir / "freshness-review-report.json", org_report)
    print(
        f"Freshness review: {org_report['stale_total']} stale artifact(s) across "
        f"{org_report['repo_count']} repo(s); {org_report['work_item_total']} work item(s)"
    )
    for repo in repo_summaries:
        if repo["stale_count"]:
            print(
                f"  ! {repo['repo']}: {repo['stale_count']} stale of "
                f"{repo['artifacts_reviewed']} reviewed"
            )
    return 0


def cmd_org_list_roles(args: argparse.Namespace) -> int:
    manifest = resolve_manifest(args)
    for role in sorted(manifest.roles, key=lambda item: item.level):
        print(role.id)
    return 0


def cmd_org_discussions(args: argparse.Namespace) -> int:
    from .discussions import DiscussionsError, discussions_run

    manifest = resolve_manifest(args)
    try:
        report = discussions_run(
            manifest=manifest,
            workspace=resolve_workspace(args),
            state_dir=resolve_org_state_dir(args, manifest),
        )
    except DiscussionsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    totals = report["totals"]
    print(
        f"Discussions: {totals['discussion_count']} open across "
        f"{totals['repos_with_content']} repo(s); "
        f"{totals['needs_attention_count']} needing attention"
        + (
            f"; {totals['repos_with_errors']} fetch error(s)"
            if totals["repos_with_errors"]
            else ""
        )
    )
    for repo in report["repos"]:
        if not repo.get("needs_attention"):
            continue
        print(f"- {repo['repo']}: {repo['needs_attention_count']} flagged")
        for item in repo["needs_attention"][:5]:
            reasons = "; ".join(item.get("reasons", []))[:160]
            print(f"    #{item['number']} {item['title']} — {reasons}")
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
