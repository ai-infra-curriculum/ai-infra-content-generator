#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKSPACE="${AICG_WORKSPACE:-$(cd "$RUNNER_DIR/.." && pwd)}"
MANIFEST="${AICG_MANIFEST:-$RUNNER_DIR/config/aicg-org.yaml}"
STATE_DIR="${AICG_STATE_DIR:-$RUNNER_DIR/.aicg/org}"
LOG_DIR="${AICG_LOG_DIR:-$RUNNER_DIR/.aicg/logs}"
AICG_BIN="${AICG_BIN:-$RUNNER_DIR/.venv/bin/aicg}"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME JOB [OPTIONS]

Jobs:
  sync
  monthly-release
  monthly-research      (quarterly: writes proposals, opens PRs, NO auto-merge)
  monthly-review        (quarterly: LLM freshness review of existing content)
  promote-plan          (on-demand: applies approved research proposals after PR merge)
  research-role ROLE    (nightly per-role research; replaces monthly-research)
  weekly-audit
  daily-remediate
  daily-issues
  daily-steward
  daily-discussions

Options:
  --workspace PATH   Curriculum workspace. Default: parent of runner repo.
  --manifest PATH    Org manifest. Default: config/aicg-org.yaml.
  --state-dir PATH   Org state directory. Default: .aicg/org.
  --aicg-bin PATH    aicg executable. Default: .venv/bin/aicg.
  -h, --help         Show this help.
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

parse_args() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 2
  fi

  JOB="$1"
  shift

  # research-role takes a positional ROLE argument. Capture it before
  # the option loop so the rest of parsing stays uniform.
  ROLE=""
  if [[ "$JOB" == "research-role" ]]; then
    if [[ $# -eq 0 || "$1" == --* ]]; then
      die "research-role requires a ROLE argument (e.g. junior-engineer)"
    fi
    ROLE="$1"
    shift
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workspace)
        WORKSPACE="${2:-}"
        shift 2
        ;;
      --manifest)
        MANIFEST="${2:-}"
        shift 2
        ;;
      --state-dir)
        STATE_DIR="${2:-}"
        shift 2
        ;;
      --aicg-bin)
        AICG_BIN="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

run_aicg_org() {
  "$AICG_BIN" org "$@" --workspace "$WORKSPACE" --manifest "$MANIFEST" --state-dir "$STATE_DIR"
}

main() {
  mkdir -p "$STATE_DIR" "$LOG_DIR"
  local log_slug="$JOB"
  if [[ -n "$ROLE" ]]; then
    log_slug="${JOB}-${ROLE}"
  fi
  local log_file="$LOG_DIR/${log_slug}-$(date '+%Y%m%d').log"
  exec >>"$log_file" 2>&1

  [[ -x "$AICG_BIN" ]] || die "aicg executable not found or not executable: $AICG_BIN"
  [[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"

  # Lock variable is intentionally global so the EXIT trap, which fires
  # after main() returns, can still see it under `set -u`.
  LOCK_DIR="$STATE_DIR/aicg-org.lock"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "Another AICG org job is running; exiting."
    exit 0
  fi
  trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

  log "Starting job: $JOB"
  case "$JOB" in
    sync)
      run_aicg_org sync
      ;;
    monthly-release)
      run_aicg_org sync
      run_aicg_org release --apply
      ;;
    monthly-research)
      # Quarterly research. The runner does NOT scaffold new modules
      # autonomously — `research --apply` opens proposal PRs for human
      # review. After a human merges a proposal PR, run
      # `aicg org promote-plan --role <role>` (or the next
      # weekly-audit) to apply the approved delta and scaffold
      # skeletons.
      run_aicg_org research --apply
      ;;
    research-role)
      # Per-role nightly research. One role per night at midnight, ordered
      # lowest-level-first across the month, opens a proposal PR if the
      # delta passes caps. Replaces the org-wide monthly-research job —
      # spreads token usage across 13 nights and bounds blast-radius
      # of any single failure to one role.
      [[ -n "$ROLE" ]] || die "research-role requires ROLE"
      run_aicg_org research --apply --role "$ROLE"
      ;;
    promote-plan)
      run_aicg_org promote-plan
      local roles
      roles=$("$AICG_BIN" --workspace "$WORKSPACE" org list-roles \
        --manifest "$MANIFEST" --state-dir "$STATE_DIR" 2>/dev/null || true)
      for role in $roles; do
        log "execute-plan for role=$role"
        run_aicg_org execute-plan --role "$role" 2>&1 | tail -3 || true
      done
      run_aicg_org audit
      ;;
    weekly-audit)
      run_aicg_org sync
      # Freshness checks first so their work items are picked up by audit.
      run_aicg_org audit-links || true
      run_aicg_org audit-versions || true
      # Broader structural audits also feed into the work-queue via
      # run_org_audit, but a standalone run here writes their reports
      # and surfaces summary lines in the log.
      run_aicg_org audit-learning || true
      run_aicg_org audit-pairing || true
      run_aicg_org audit-curriculum || true
      run_aicg_org audit-profile || true
      run_aicg_org audit
      ;;
    monthly-review)
      run_aicg_org sync
      run_aicg_org review
      run_aicg_org audit
      ;;
    daily-remediate)
      run_aicg_org daily
      ;;
    daily-issues)
      run_aicg_org issues --apply
      ;;
    daily-steward)
      run_aicg_org steward --apply
      # Also sweep Dependabot PRs each morning: queue auto-merge,
      # post @dependabot rebase on stale ones, escalate after 3 rebase
      # loops without progress.
      run_aicg_org dependabot --apply || true
      # Re-attempt the maintainer rebrand for any repos that were
      # skipped (dirty tree etc.) on a previous run. Idempotent —
      # already-branded repos are no-ops.
      run_aicg_org rebrand --apply || true
      ;;
    daily-discussions)
      run_aicg_org discussions
      ;;
    *)
      die "Unknown job: $JOB"
      ;;
  esac
  log "Completed job: $JOB"
}

parse_args "$@"
main
