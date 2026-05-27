# Autonomous Org Automation

This is the target operating model for headless management of the
AI-Infra-Curriculum GitHub org.

## Agent Policy

- **Claude Opus 4.7**: content generation only. This includes curriculum,
  solution, job-requirement, and supplemental content.
- **Codex GPT-5.5**: everything else. This includes repository orchestration,
  audits, validation, work planning, PR/issue/discussion stewardship, release
  tagging, and scheduler control.

The policy is encoded in `config/aicg-org.yaml`. Command strings are intentionally
configurable because headless hosts may use different local wrappers.

The default command path is local CLI only:

- Claude content: `scripts/run-claude-content.sh`
- Codex control: `scripts/run-codex-control.sh`

These wrappers call the `claude` and `codex` commands installed on the headless
host. They do not use Anthropic or OpenAI API tokens.

## Subscription Limits

Claude and Codex subscription sessions can hit 5-hour or weekly limits. AICG
treats those failures as temporary:

- command output is scanned for limit messages
- `.aicg/run-state.json` records `agent_limit_reached`
- org queue items move to `deferred`
- `retry_after` is set to roughly 5 hours, 7 days, or 6 hours for unknown limits
- future daily runs re-enable deferred items after `retry_after`

This keeps cron/systemd jobs healthy while preserving the exact work item that
needs to resume.

## Schedule

Default timers:

- Monthly release: day 1, 02:00.
- Monthly job research: day 1, 05:30.
- Weekly audit: Sunday, 03:00.
- Daily remediation: every day, 04:00.
- Daily stewardship report: every day, 04:30.

Install on a host with:

```bash
scripts/bootstrap-headless-host.sh --workspace /srv/ai-infra-curriculum --scheduler systemd
```

For cron instead:

```bash
scripts/bootstrap-headless-host.sh --workspace /srv/ai-infra-curriculum --scheduler cron
```

Use `--dry-run` first to inspect the generated entries.

## Monthly Release

`aicg org release` plans date-based tags for all learning and solution repos.
The configured format is `v%Y.%m`, producing tags such as `v2026.05`.

The org `.github` repository is included in sync and stewardship operations, but
is not part of the curriculum/solution release tag set by default.

`aicg org release --apply` creates and pushes tags. Per-repo GitHub Actions
should use `templates/github/release-package.yml` to publish `.tar.gz` and `.zip`
assets for each tag.

## Monthly Job Research

`aicg org research` writes one research packet per role under
`.aicg/org/research/<YYYY-MM>/`.

Each packet requires:

- Current job postings and source URLs.
- Normalized requirements in `.aicg/job-requirements.json`.
- Human-readable coverage in `JOB_REQUIREMENTS.md`.
- Links to curriculum paths for covered requirements.
- External learning resources for useful but out-of-scope requirements.

Shared requirements use the lowest-level ownership rule: the lowest-level role
that genuinely needs the skill owns the primary curriculum coverage. Higher
roles link to that owner unless they need different depth or leadership context.

When research changes coverage, generated updates must also keep these files in
sync while preserving their existing format:

- `CURRICULUM.md`
- `CURRICULUM_INDEX.md`
- `README.md`
- `VERSIONS.md`
- org-level README files in the `.github` repo, including `README.md` and
  `profile/README.md` when present

## Weekly Audit

`aicg org audit` runs repo-level audit and planning across all solution repos and
writes `.aicg/org/work-queue.json`.

The work queue is deterministic and sorted by role level, repo, and work item
priority. Daily remediation consumes this queue instead of freely choosing work.

Validation includes format checks for the protected curriculum docs above. The
check is intentionally structural: it catches empty protected files, broken
tables, and suspicious heading hierarchy changes before PRs are considered safe.

## Daily Remediation

`aicg org daily` selects the next ready work item and prepares a content packet
for the configured Claude content agent. If no ready gaps exist, it creates
supplemental content packets for brief, source-backed material under each
learning repo's `supplemental/` directory.

## PR, Issue, And Discussion Stewardship

`aicg org steward` currently writes a dry-run stewardship report. The intended
mutation path is:

- inspect open PRs and required checks
- auto-merge only when AICG guardrails pass
- update issues from audit and work-queue state
- summarize discussions that require human judgment

This path should remain dry-run until the pilot PR flow proves green CI and
guardrail behavior.
