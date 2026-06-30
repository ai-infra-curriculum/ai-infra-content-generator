#!/usr/bin/env bash
#
# Install a namespaced systemd --user timer set for ONE sibling domain, so the
# autonomous pipeline researches + authors that domain's roles without
# colliding with ai-infra's timers (install-schedules.sh) or the other
# domains.
#
# Each domain gets:
#   - aicg-<domain>-research-role@<role>.timer  (seed curriculum plan; day=role index)
#   - aicg-<domain>-review-role@<role>.timer    (freshness review; day=14+index)
#   - aicg-<domain>-generate-role@<role>.timer  (author content from the plan; day=index)
#   - aicg-<domain>-daily.timer                 (daily authoring tick from the queue)
#   - aicg-<domain>-fill.timer                  (daily: fill the next unfilled role)
# all pointing at the domain's manifest + a per-domain state dir, and offset by
# --hour-offset so 4 domains don't hit the shared Claude subscription at once.
#
# Roles come from the manifest (via aicg org list-roles), so the set stays in
# lockstep with config. Stale per-role timers (role left the manifest) are
# removed. Run per domain:
#   install-domain-timers.sh --domain ml-engineering --hour-offset 2
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$RUNNER_DIR/.." && pwd)"
AICG_BIN="${AICG_BIN:-$RUNNER_DIR/.venv/bin/aicg}"
DOMAIN=""
HOUR_OFFSET=0
DRY_RUN=0

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="${2:-}"; shift 2;;
    --hour-offset) HOUR_OFFSET="${2:-0}"; shift 2;;
    --workspace) WORKSPACE="${2:-}"; shift 2;;
    --aicg-bin) AICG_BIN="${2:-}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) die "unknown arg: $1";;
  esac
done

[[ -n "$DOMAIN" ]] || die "--domain is required"
MANIFEST="$RUNNER_DIR/config/domains/$DOMAIN.yaml"
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"
STATE_DIR="$RUNNER_DIR/.aicg/$DOMAIN"
LOG_DIR="$HOME/.cache/aicg"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

# Roles from the manifest (authoritative ordering), with a parse fallback.
manifest_roles() {
  local out=""
  if [[ -x "$AICG_BIN" ]]; then
    out="$("$AICG_BIN" org list-roles --manifest "$MANIFEST" --state-dir "$STATE_DIR" 2>/dev/null || true)"
  fi
  if [[ -n "$out" ]]; then printf '%s\n' "$out"; return; fi
  python3 - "$MANIFEST" <<'PY'
import sys, json
d=json.load(open(sys.argv[1]))
for r in d.get("roles", []): print(r["id"])
PY
}

ROLES=()
while IFS= read -r _role; do
  [[ -n "$_role" ]] && ROLES+=("$_role")
done < <(manifest_roles)
[[ "${#ROLES[@]}" -gt 0 ]] || die "no roles in manifest"
log "Domain '$DOMAIN': ${#ROLES[@]} roles, hour-offset=$HOUR_OFFSET, state=$STATE_DIR"

# minute 15 keeps clear of any top-of-hour jobs; offset shifts the whole domain.
RHOUR=$(printf '%02d' $(( (0 + HOUR_OFFSET) % 24 )))   # research base 00:15 + offset
VHOUR=$(printf '%02d' $(( (12 + HOUR_OFFSET) % 24 )))  # review base 12:30 + offset
DHOUR=$(printf '%02d' $(( (3 + HOUR_OFFSET) % 24 )))   # daily author tick 03:40 + offset

job_cmd() {  # job role
  printf '%q %q %q --workspace %q --manifest %q --state-dir %q' \
    "$RUNNER_DIR/scripts/aicg-org-job.sh" "$1" "$2" "$WORKSPACE" "$MANIFEST" "$STATE_DIR"
}
job_cmd_noarg() {  # job
  printf '%q %q --workspace %q --manifest %q --state-dir %q' \
    "$RUNNER_DIR/scripts/aicg-org-job.sh" "$1" "$WORKSPACE" "$MANIFEST" "$STATE_DIR"
}

write_unit() {  # name exec-cmd calendar
  local name="$1" cmd="$2" cal="$3"
  local svc="$UNIT_DIR/aicg-$name.service" tmr="$UNIT_DIR/aicg-$name.timer"
  local svc_body="[Unit]
Description=AICG $DOMAIN: $name
[Service]
Type=oneshot
Environment=PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HOME=$HOME
ExecStartPre=/bin/mkdir -p $LOG_DIR
ExecStart=$cmd
StandardOutput=append:$LOG_DIR/$name.log
StandardError=append:$LOG_DIR/$name.log
"
  local tmr_body="[Unit]
Description=AICG $DOMAIN timer: $name
[Timer]
OnCalendar=$cal
Persistent=true
[Install]
WantedBy=timers.target
"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf -- '----- %s -----\n%s\n----- %s -----\n%s\n' "$svc" "$svc_body" "$tmr" "$tmr_body"
    return
  fi
  printf '%s' "$svc_body" >"$svc"
  printf '%s' "$tmr_body" >"$tmr"
  systemctl --user enable --now "aicg-$name.timer" >/dev/null 2>&1 || true
}

[[ "$DRY_RUN" -eq 1 ]] || mkdir -p "$UNIT_DIR" "$STATE_DIR"

# Per-role research -> generate -> review, day = role index (research+generate)
# / 14+index (review). Generate runs 3h after research the same day so the plan
# it authors from is already seeded.
GHOUR=$(printf '%02d' $(( (3 + HOUR_OFFSET) % 24 )))  # generate, after research
i=0
for role in "${ROLES[@]}"; do
  i=$((i + 1))
  rday=$(printf '%02d' "$i")
  vday=$(printf '%02d' "$((i + 14))")
  write_unit "$DOMAIN-research-role@$role" "$(job_cmd research-role "$role")" "*-*-$rday $RHOUR:15:00"
  write_unit "$DOMAIN-generate-role@$role" "$(job_cmd generate-role "$role")" "*-*-$rday $GHOUR:15:00"
  write_unit "$DOMAIN-review-role@$role"   "$(job_cmd review-role "$role")"   "*-*-$vday $VHOUR:30:00"
done

# One daily authoring tick for the domain (drains the work queue within budget).
write_unit "$DOMAIN-daily" "$(job_cmd_noarg daily-remediate)" "*-*-* $DHOUR:40:00"

# Daily continuous-fill tick: author the next unfilled role (self-seeding), so a
# fresh fleet fills one role/day instead of stalling between the day-1-8 per-role
# timers. Self-terminating once every role has modules. Offset 1h from the daily
# tick so they don't overlap on the session cap.
FHOUR=$(printf '%02d' $(( (4 + HOUR_OFFSET) % 24 )))
write_unit "$DOMAIN-fill" "$(job_cmd_noarg fill-next)" "*-*-* $FHOUR:40:00"

if [[ "$DRY_RUN" -eq 0 ]]; then
  systemctl --user daemon-reload
  # Tear down stale per-role timers for roles no longer in the manifest.
  current=" ${ROLES[*]} "
  for t in "$UNIT_DIR"/aicg-"$DOMAIN"-research-role@*.timer \
           "$UNIT_DIR"/aicg-"$DOMAIN"-generate-role@*.timer \
           "$UNIT_DIR"/aicg-"$DOMAIN"-review-role@*.timer; do
    [[ -e "$t" ]] || continue
    base="$(basename "$t")"; role="${base##*@}"; role="${role%.timer}"
    [[ "$current" == *" $role "* ]] && continue
    log "removing stale timer: $base"
    systemctl --user disable --now "$base" 2>/dev/null || true
    rm -f "$t"
  done
  log "Installed $DOMAIN timers (${#ROLES[@]} research + ${#ROLES[@]} generate + ${#ROLES[@]} review + 1 daily + 1 fill)."
fi
