#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKSPACE="$(cd "$RUNNER_DIR/.." && pwd)"
MANIFEST="$RUNNER_DIR/config/aicg-org.yaml"
STATE_DIR="$RUNNER_DIR/.aicg/org"
SCHEDULER="systemd"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS]

Options:
  --scheduler MODE   systemd or cron. Default: systemd.
  --workspace PATH   Curriculum workspace. Default: parent of runner repo.
  --manifest PATH    Org manifest. Default: config/aicg-org.yaml.
  --state-dir PATH   Org state directory. Default: .aicg/org.
  --dry-run          Print files/entries without installing.
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
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scheduler)
        SCHEDULER="${2:-}"
        shift 2
        ;;
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
      --dry-run)
        DRY_RUN=1
        shift
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

job_command() {
  local job="$1"
  printf '%q %q --workspace %q --manifest %q --state-dir %q' \
    "$RUNNER_DIR/scripts/aicg-org-job.sh" "$job" "$WORKSPACE" "$MANIFEST" "$STATE_DIR"
}

# research-role takes a positional ROLE that must follow the job name,
# before option flags. Build the command in that order.
job_command_research_role() {
  local role="$1"
  printf '%q %q %q --workspace %q --manifest %q --state-dir %q' \
    "$RUNNER_DIR/scripts/aicg-org-job.sh" "research-role" "$role" \
    "$WORKSPACE" "$MANIFEST" "$STATE_DIR"
}

install_cron() {
  local marker_begin="# BEGIN AICG ORG JOBS"
  local marker_end="# END AICG ORG JOBS"
  # Per-role nightly research at 00:00 on days 1-13. Matches the systemd
  # ordering — lowest-level role on day 1, chief-ai-officer on day 13.
  local per_role_lines=""
  local -a role_slots=(
    "1:junior-engineer"
    "2:engineer"
    "3:ml-platform"
    "4:mlops"
    "5:senior-engineer"
    "6:performance"
    "7:security"
    "8:team-lead"
    "9:architect"
    "10:principal-engineer"
    "11:senior-architect"
    "12:principal-architect"
    "13:chief-ai-officer"
  )
  local entry day role
  for entry in "${role_slots[@]}"; do
    day="${entry%%:*}"
    role="${entry##*:}"
    per_role_lines+=$'\n'"0 0 $day * * $(job_command_research_role "$role")"
  done

  local block
  block="$marker_begin
0 2 1 * * $(job_command monthly-release)
0 3 * * 0 $(job_command weekly-audit)
0 * * * * $(job_command daily-remediate)
20 4 * * * $(job_command daily-issues)
40 4 * * * $(job_command daily-steward)
5 5 * * * $(job_command daily-discussions)
0 6 1 3,6,9,12 * $(job_command monthly-review)${per_role_lines}
$marker_end"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n' "$block"
    return
  fi

  local tmp
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | sed "/$marker_begin/,/$marker_end/d" >"$tmp" || true
  printf '\n%s\n' "$block" >>"$tmp"
  crontab "$tmp"
  rm -f "$tmp"
  log "Installed cron jobs."
}

write_systemd_pair() {
  local unit_dir="$1"
  local name="$2"
  local job="$3"
  local calendar="$4"
  local service="$unit_dir/aicg-$name.service"
  local timer="$unit_dir/aicg-$name.timer"

  # Drop-in marker file lets operators keep a local override in
  # aicg-$name.timer.d/override.conf — we never touch *.d/ subdirs.
  # Anything written here gets the install-script default; existing
  # drop-ins take effect after `daemon-reload`.
  :

  local log_dir="$HOME/.cache/aicg"
  local service_content="[Unit]
Description=AICG org job: $job

[Service]
Type=oneshot
Environment=PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=$HOME
ExecStartPre=/bin/mkdir -p $log_dir
ExecStart=$(job_command "$job")
StandardOutput=append:$log_dir/$job.log
StandardError=append:$log_dir/$job.log
"
  local timer_content="[Unit]
Description=AICG org timer: $job

[Timer]
OnCalendar=$calendar
Persistent=true

[Install]
WantedBy=timers.target
"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n%s\n' "----- $service -----" "$service_content"
    printf '%s\n%s\n' "----- $timer -----" "$timer_content"
    return
  fi

  printf '%s' "$service_content" >"$service"
  printf '%s' "$timer_content" >"$timer"
}

install_per_role_research() {
  local unit_dir="$1"
  local log_dir="$HOME/.cache/aicg"
  local service_path="$unit_dir/aicg-research-role@.service"
  local template_path="$RUNNER_DIR/scripts/cron/aicg-research-role@.service.template"

  [[ -f "$template_path" ]] || die "missing template: $template_path"

  # Render the templated service file. systemd substitutes %i at
  # invocation time; we substitute the install-time variables now.
  local service_content
  service_content="$(
    sed \
      -e "s#{PATH}#$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin#g" \
      -e "s#{HOME}#$HOME#g" \
      -e "s#{LOG_DIR}#$log_dir#g" \
      -e "s#{RUNNER_DIR}#$RUNNER_DIR#g" \
      -e "s#{WORKSPACE}#$WORKSPACE#g" \
      -e "s#{MANIFEST}#$MANIFEST#g" \
      -e "s#{STATE_DIR}#$STATE_DIR#g" \
      "$template_path"
  )"

  # Roles + their day-of-month slot. Lowest-level-first per the curriculum
  # config so foundational tracks land their proposals before dependent
  # tracks could conflict with them. Chief-AI-officer (level 70) anchors
  # day 13. Update this list when the org manifest gains/loses a role.
  local -a role_slots=(
    "01:junior-engineer"
    "02:engineer"
    "03:ml-platform"
    "04:mlops"
    "05:senior-engineer"
    "06:performance"
    "07:security"
    "08:team-lead"
    "09:architect"
    "10:principal-engineer"
    "11:senior-architect"
    "12:principal-architect"
    "13:chief-ai-officer"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n%s\n' "----- $service_path -----" "$service_content"
  else
    printf '%s' "$service_content" >"$service_path"
  fi

  local entry day role timer_path timer_content
  for entry in "${role_slots[@]}"; do
    day="${entry%%:*}"
    role="${entry##*:}"
    timer_path="$unit_dir/aicg-research-role@${role}.timer"
    timer_content="[Unit]
Description=AICG per-role research timer: ${role}

[Timer]
OnCalendar=*-*-${day} 00:00:00
Persistent=true
Unit=aicg-research-role@${role}.service

[Install]
WantedBy=timers.target
"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '%s\n%s\n' "----- $timer_path -----" "$timer_content"
    else
      printf '%s' "$timer_content" >"$timer_path"
    fi
  done

  # Return the role list so the caller can enable each timer.
  PER_ROLE_SLUGS=()
  for entry in "${role_slots[@]}"; do
    PER_ROLE_SLUGS+=("${entry##*:}")
  done
}

install_systemd() {
  local unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$unit_dir"
  fi

  write_systemd_pair "$unit_dir" "monthly-release" "monthly-release" "*-*-01 02:00:00"
  # Per-role nightly research replaces the org-wide monthly-research job.
  # The legacy aicg-monthly-research.{service,timer} files are still
  # written for compatibility (so existing operators see the rename in
  # the dry-run), but install_per_role_research is the new path and the
  # enable list below disables monthly-research.timer in favor of the
  # per-role instances.
  write_systemd_pair "$unit_dir" "monthly-research" "monthly-research" "*-03,06,09,12-01 05:30:00"
  # Quarterly LLM freshness review (1st of Mar/Jun/Sep/Dec at 06:00).
  write_systemd_pair "$unit_dir" "monthly-review" "monthly-review" "*-03,06,09,12-01 06:00:00"
  write_systemd_pair "$unit_dir" "weekly-audit" "weekly-audit" "Sun 03:00:00"
  write_systemd_pair "$unit_dir" "daily-remediate" "daily-remediate" "*-*-* *:00:00"
  write_systemd_pair "$unit_dir" "daily-issues" "daily-issues" "*-*-* 04:20:00"
  write_systemd_pair "$unit_dir" "daily-steward" "daily-steward" "*-*-* 04:40:00"
  # Discussions at 05:05, not 05:00 — the hourly remediate timer also
  # fires at 05:00 and they raced for the org-job lock. Remediate won
  # consistently and discussions bailed with "Another AICG org job is
  # running" every day. 5 min is comfortably past remediate's typical
  # runtime (~1-100s).
  write_systemd_pair "$unit_dir" "daily-discussions" "daily-discussions" "*-*-* 05:05:00"

  install_per_role_research "$unit_dir"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    systemctl --user daemon-reload
    # Disable the legacy org-wide research timer in favor of per-role
    # instances. Tolerate failure (it may already be inactive).
    systemctl --user disable --now aicg-monthly-research.timer 2>/dev/null || true
    systemctl --user enable --now \
      aicg-monthly-release.timer \
      aicg-monthly-review.timer \
      aicg-weekly-audit.timer \
      aicg-daily-remediate.timer \
      aicg-daily-issues.timer \
      aicg-daily-steward.timer \
      aicg-daily-discussions.timer
    local slug
    for slug in "${PER_ROLE_SLUGS[@]}"; do
      systemctl --user enable --now "aicg-research-role@${slug}.timer"
    done
    log "Installed systemd user timers (including ${#PER_ROLE_SLUGS[@]} per-role research timers)."
  fi
}

main() {
  [[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"
  [[ -x "$RUNNER_DIR/scripts/aicg-org-job.sh" ]] || die "job wrapper is not executable"

  case "$SCHEDULER" in
    cron)
      install_cron
      ;;
    systemd)
      install_systemd
      ;;
    *)
      die "Unsupported scheduler: $SCHEDULER"
      ;;
  esac
}

parse_args "$@"
main
