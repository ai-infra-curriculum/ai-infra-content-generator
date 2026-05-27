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

install_cron() {
  local marker_begin="# BEGIN AICG ORG JOBS"
  local marker_end="# END AICG ORG JOBS"
  local block
  block="$marker_begin
0 2 1 * * $(job_command monthly-release)
30 5 1 * * $(job_command monthly-research)
0 3 * * 0 $(job_command weekly-audit)
0 4 * * * $(job_command daily-remediate)
30 4 * * * $(job_command daily-steward)
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

  local service_content="[Unit]
Description=AICG org job: $job

[Service]
Type=oneshot
ExecStart=$(job_command "$job")
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

install_systemd() {
  local unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$unit_dir"
  fi

  write_systemd_pair "$unit_dir" "monthly-release" "monthly-release" "*-*-01 02:00:00"
  write_systemd_pair "$unit_dir" "monthly-research" "monthly-research" "*-*-01 05:30:00"
  write_systemd_pair "$unit_dir" "weekly-audit" "weekly-audit" "Sun 03:00:00"
  write_systemd_pair "$unit_dir" "daily-remediate" "daily-remediate" "*-*-* 04:00:00"
  write_systemd_pair "$unit_dir" "daily-steward" "daily-steward" "*-*-* 04:30:00"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    systemctl --user daemon-reload
    systemctl --user enable --now \
      aicg-monthly-release.timer \
      aicg-monthly-research.timer \
      aicg-weekly-audit.timer \
      aicg-daily-remediate.timer \
      aicg-daily-steward.timer
    log "Installed systemd user timers."
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
