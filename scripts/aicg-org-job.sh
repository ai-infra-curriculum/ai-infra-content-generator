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

# ntfy observability (optional). Topic comes from an env file so it stays out
# of git; if unset, every notify is a no-op.
[[ -f "$HOME/.config/aicg/ntfy.env" ]] && . "$HOME/.config/aicg/ntfy.env"
NTFY_TOPIC="${AICG_NTFY_TOPIC:-}"
NTFY_BASE="${AICG_NTFY_URL:-https://ntfy.sh}"

# notify_ntfy <title> <tags> <priority> <message>  — best-effort; never fails the job.
notify_ntfy() {
  [[ -n "$NTFY_TOPIC" ]] || return 0
  curl -fsS -m 10 \
    -H "Title: $1" -H "Tags: $2" -H "Priority: $3" \
    -d "$4" "$NTFY_BASE/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

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
  review-role ROLE      (nightly per-role freshness review; replaces monthly-review)
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

  # research-role / review-role take a positional ROLE argument. Capture
  # it before the option loop so the rest of parsing stays uniform.
  ROLE=""
  case "$JOB" in
    research-role|review-role|generate-role|seed-role)
      if [[ $# -eq 0 || "$1" == --* ]]; then
        die "$JOB requires a ROLE argument (e.g. junior-engineer)"
      fi
      ROLE="$1"
      shift
      ;;
  esac

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
  local lock_ttl="${AICG_LOCK_TTL:-10800}"  # 3h grace before a held lock is suspect
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    # Stale-lock recovery (C-M2): break the lock only if its owner PID is dead
    # AND it has been held past the TTL. A slow-but-live holder is never broken.
    local broke=0
    if [[ -f "$LOCK_DIR/owner" ]]; then
      local lpid lts age
      lpid="$(sed -n '1p' "$LOCK_DIR/owner" 2>/dev/null || true)"
      lts="$(sed -n '2p' "$LOCK_DIR/owner" 2>/dev/null || true)"
      if [[ "$lpid" =~ ^[0-9]+$ && "$lts" =~ ^[0-9]+$ ]]; then
        age=$(( $(date +%s) - lts ))
        if (( age > lock_ttl )) && ! kill -0 "$lpid" 2>/dev/null; then
          log "Breaking stale lock: pid $lpid dead, held ${age}s > ${lock_ttl}s."
          rm -rf "$LOCK_DIR" 2>/dev/null || true
          mkdir "$LOCK_DIR" 2>/dev/null && broke=1
        fi
      fi
    fi
    if (( broke == 0 )); then
      log "Another AICG org job is running; exiting."
      exit 0
    fi
  fi
  printf '%s\n%s\n' "$$" "$(date +%s)" > "$LOCK_DIR/owner" 2>/dev/null || true
  # On exit: drop the lock, and (on a hard failure) push an ntfy alert so a
  # silent crash is visible. A clean exit or the "already running" path (rc 0)
  # stays quiet.
  _on_exit() {
    local rc=$?
    rm -rf "$LOCK_DIR" 2>/dev/null || true
    if [[ "$rc" -ne 0 ]]; then
      notify_ntfy "🔴 AICG job failed: $JOB${ROLE:+ ($ROLE)}" "rotating_light" "high" \
        "domain=$(basename "$MANIFEST" .yaml) job=$JOB exit=$rc host=$(hostname 2>/dev/null)"
    fi
    return "$rc"
  }
  trap _on_exit EXIT

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
      # Pull every repo to current main first — research reads the role's
      # curriculum-plan.json and branches from local main, so a stale clone
      # would analyze old content and base the proposal PR on an old commit.
      run_aicg_org sync
      run_aicg_org research --apply --role "$ROLE"
      ;;
    seed-role)
      # One-time plan SEEDING for a fresh role. Curriculum-plan changes are
      # human-gated by design (research opens proposal PRs; the runner never
      # auto-merges curriculum-plan.json), so a brand-new repo has no plan for
      # generate-role to author from. This bypasses that gate for the INITIAL
      # seed only: (re)write the role's bootstrap prompt, run the content agent
      # to research + author curriculum-plan.json + JOB_REQUIREMENTS.md into the
      # learning repo, then commit + push just those plan files. Idempotent:
      # skips if a non-empty plan already exists.
      [[ -n "$ROLE" ]] || die "seed-role requires ROLE"
      run_aicg_org sync
      LREPO=$(PYTHONPATH="$RUNNER_DIR/src" python3 -c "
import json,sys
d=json.load(open('$MANIFEST'))
print(next((r['learning_repo'] for r in d['roles'] if r['id']=='$ROLE'),''))" 2>/dev/null || true)
      [[ -n "$LREPO" ]] || die "no learning repo for role $ROLE in $MANIFEST"
      LPATH="$WORKSPACE/$LREPO"
      [[ -d "$LPATH" ]] || die "learning repo not cloned: $LPATH (run sync)"
      if [[ -s "$LPATH/.aicg/curriculum-plan.json" ]]; then
        log "plan already exists for $ROLE; skipping seed"
      else
        PROMPT=$(run_aicg_org bootstrap-prompt --role "$ROLE" | tail -1)
        case "$PROMPT" in /*) ;; *) PROMPT="$RUNNER_DIR/$PROMPT" ;; esac
        [[ -f "$PROMPT" ]] || die "bootstrap prompt not written: $PROMPT"
        # Absolute output dir: run-claude-content.sh cd's into --repo before
        # writing response.md, so a relative path would resolve under the repo.
        SEED_OUT="$STATE_DIR/seed/$ROLE"
        case "$SEED_OUT" in /*) ;; *) SEED_OUT="$RUNNER_DIR/$SEED_OUT" ;; esac
        mkdir -p "$SEED_OUT"
        log "seeding plan for $ROLE via content agent (prompt=$PROMPT)"
        "$RUNNER_DIR/scripts/run-claude-content.sh" \
          --prompt "$PROMPT" --repo "$WORKSPACE" --output-dir "$SEED_OUT" \
          --work-id "seed-$ROLE" || log "content agent returned non-zero for $ROLE"
        if [[ -s "$LPATH/.aicg/curriculum-plan.json" ]]; then
          # The plan lives at .aicg/ which is gitignored by design (per-repo
          # runner state, never pushed — generate-role reads it locally). So
          # success = the local plan exists. Only the PUBLIC artifacts the
          # agent may also write (JOB_REQUIREMENTS.md, supplemental/) get
          # committed; git add'ing the ignored plan would fail the whole add.
          log "seeded plan present for $ROLE ($(wc -c <"$LPATH/.aicg/curriculum-plan.json") bytes, local runner state)"
          ( cd "$LPATH"
            git add JOB_REQUIREMENTS.md supplemental 2>/dev/null || true
            if ! git diff --cached --quiet 2>/dev/null; then
              git -c user.email="aicg@veriswarm.ai" -c user.name="AICG Runner" \
                  commit -m "seed: job requirements from initial research" \
              && git push origin HEAD && echo "  pushed public seed artifacts"
            fi ) || log "seed: public-artifact commit skipped for $ROLE"
        else
          log "seed produced NO plan for $ROLE (see $SEED_OUT/response.md)"
        fi
      fi
      ;;
    generate-role)
      # Per-role nightly CONTENT AUTHORING — the autonomous fill step that
      # research/daily-remediate alone don't cover for a fresh repo. Reads the
      # role's seeded curriculum-plan.json, scaffolds any missing module/project
      # skeletons, then authors the learning content for those modules directly
      # (learning_content.generate_role_learning_content bypasses the
      # evidence-gated research-PR flow — it authors from the plan). Solutions
      # are filled by daily-remediate from audit-queued solution gaps afterward.
      # No-op-safe: execute-plan + generate-learning are idempotent per module.
      [[ -n "$ROLE" ]] || die "generate-role requires ROLE"
      run_aicg_org sync
      # Self-seed: generate-learning authors from curriculum-plan.json, which a
      # fresh role lacks (plan changes are human-gated by design). Seed it first
      # if missing — seed-role no-ops when a plan already exists, so this is the
      # autonomous full chain: seed (once) -> author. Heavy LLM steps; the
      # shared-subscription session cap naturally spreads them across the
      # staggered per-role schedule.
      GR_LREPO=$(PYTHONPATH="$RUNNER_DIR/src" python3 -c "
import json
d=json.load(open('$MANIFEST'))
print(next((r['learning_repo'] for r in d['roles'] if r['id']=='$ROLE'),''))" 2>/dev/null || true)
      if [[ -n "$GR_LREPO" && ! -s "$WORKSPACE/$GR_LREPO/.aicg/curriculum-plan.json" ]]; then
        log "no plan for $ROLE; seeding before generate"
        bash "$0" seed-role "$ROLE" --workspace "$WORKSPACE" \
          --manifest "$MANIFEST" --state-dir "$STATE_DIR" || log "seed step failed/limited for $ROLE"
      fi
      run_aicg_org execute-plan --role "$ROLE" 2>&1 | tail -3 || true
      run_aicg_org generate-learning --role "$ROLE" || true
      # Observability: report the outcome by observable state (modules present
      # or not) — no agent-output parsing. One push per role-fill, so the
      # daily fill timer yields ~1 notify/domain/day, not spam.
      GR_DOM=$(basename "$MANIFEST" .yaml)
      GR_MODS=0
      [[ -n "$GR_LREPO" ]] && GR_MODS=$(ls -d "$WORKSPACE/$GR_LREPO"/lessons/mod-* 2>/dev/null | wc -l | tr -d ' ')
      if [[ "${GR_MODS:-0}" -gt 0 ]]; then
        notify_ntfy "✅ $GR_DOM · $ROLE authored" "white_check_mark,books" "default" \
          "$GR_MODS module(s) now in $GR_LREPO"
      else
        notify_ntfy "⏸️ $GR_DOM · $ROLE not filled" "hourglass" "default" \
          "generate-role produced no modules (likely the Claude usage cap; retries next cycle)"
      fi
      ;;
    review-role)
      # Per-role nightly freshness review. One role per night, days
      # 14-26, lowest-level-first. Reviews BOTH the learning and
      # solution repos for that role with a 50-artifact cap PER REPO.
      # Replaces the org-wide monthly-review job — same total review
      # cost (~1200 calls / month with the judge enabled) but spread
      # across 13 separate Claude-budget windows instead of one shot.
      # No-op when quality_judge.enabled is false (every artifact
      # gets marked 'skipped').
      [[ -n "$ROLE" ]] || die "review-role requires ROLE"
      # Pull every repo to current main first so freshness review judges
      # the latest shipped artifacts, not a stale local clone.
      run_aicg_org sync
      run_aicg_org review --role "$ROLE" --max-artifacts 50
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
    fill-next)
      # Continuous initial-fill: pick the FIRST role in this domain whose
      # learning repo has no authored modules yet and run the self-seeding
      # generate-role for just that one. Meant to run DAILY (not day-of-month),
      # so a fresh fleet fills one role per domain per day — within the shared
      # session cap — instead of stalling between the day-1-8 per-role timers.
      # Self-terminating: once every role has modules, this is a no-op.
      run_aicg_org sync
      FILL_PICK=$(PYTHONPATH="$RUNNER_DIR/src" python3 -c "
import json, os, glob, sys
d=json.load(open('$MANIFEST'))
ws='$WORKSPACE'
for r in sorted(d['roles'], key=lambda x: x.get('level',0)):
    lr=os.path.join(ws, r['learning_repo'])
    if not glob.glob(os.path.join(lr,'lessons','mod-*')) and not glob.glob(os.path.join(lr,'modules','mod-*')):
        print(r['id']); break" 2>/dev/null || true)
      if [[ -z "$FILL_PICK" ]]; then
        log "fill-next: every role has modules; nothing to fill"
      else
        log "fill-next: filling next unfilled role -> $FILL_PICK"
        bash "$0" generate-role "$FILL_PICK" --workspace "$WORKSPACE" \
          --manifest "$MANIFEST" --state-dir "$STATE_DIR"
      fi
      ;;
    daily-pipeline-tick)
      # Observe-mode tick: run every P2-P5 phase's decision on real data and
      # report what it WOULD do — writes nothing (the design's staged first
      # form). Promote a phase to act-mode by flipping its pipeline.phases flag.
      run_aicg_org sync
      run_aicg_org pipeline-tick
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
    fleet-digest)
      # Project-wide daily observability: one ntfy push summarizing fill
      # progress across all domains (N/total roles authored, mode, BAR).
      # Reads all domain manifests + the workspace; ignores --manifest.
      DIGEST=$("$AICG_BIN" --workspace "$WORKSPACE" fleet digest \
        --date "$(date +%Y-%m-%d)" 2>/dev/null || echo "fleet digest failed")
      log "$DIGEST"
      notify_ntfy "📊 AICG fleet digest" "bar_chart" "default" "$DIGEST"
      ;;
    *)
      die "Unknown job: $JOB"
      ;;
  esac
  log "Completed job: $JOB"

  # Out-of-band heartbeat (C-B2): ping an external dead-man's-switch on success
  # so a monitor independent of this host + token alerts if the pipeline goes
  # silent — the failure mode that bit us before (token expiry, silent stall).
  if [[ -n "${AICG_HEARTBEAT_URL:-}" ]]; then
    curl -fsS -m 10 "$AICG_HEARTBEAT_URL" >/dev/null 2>&1 || log "heartbeat ping failed"
  fi
}

parse_args "$@"
main
