#!/usr/bin/env bash
set -euo pipefail

# Local Claude judge wrapper. Mirrors run-claude-content.sh but
# scopes the agent to grading: it reads the judge prompt packet (which
# includes the absolute artifact path and the rubric) and is expected
# to write `response.json` into the output directory matching the
# contract documented in src/aicg/judge.py.

MODEL="claude-opus-4.7"
PROMPT=""
OUTPUT_DIR=""
REPO=""
WORK_ID=""
ARTIFACT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --prompt PATH --output-dir DIR [OPTIONS]

Options:
  --model NAME       Claude model. Default: claude-opus-4.7.
  --repo PATH        Target repo path, passed through for logging.
  --work-id ID       AICG work item id, passed through for logging.
  --artifact PATH    Absolute path to the artifact being graded.
  -h, --help         Show this help.

The wrapper invokes the local claude CLI; no Anthropic API token is
used. The judge must write response.json to the output directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="${2:-}";        shift 2 ;;
    --prompt)       PROMPT="${2:-}";       shift 2 ;;
    --output-dir)   OUTPUT_DIR="${2:-}";   shift 2 ;;
    --repo)         REPO="${2:-}";         shift 2 ;;
    --work-id)      WORK_ID="${2:-}";      shift 2 ;;
    --artifact)     ARTIFACT="${2:-}";     shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *)              echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$PROMPT" ]] || { echo "Prompt not found: $PROMPT" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required" >&2; exit 2; }
command -v claude >/dev/null || { echo "claude CLI not found in PATH" >&2; exit 127; }

mkdir -p "$OUTPUT_DIR"
{
  echo "model=$MODEL"
  echo "repo=$REPO"
  echo "work_id=$WORK_ID"
  echo "artifact=$ARTIFACT"
  echo "prompt=$PROMPT"
  date '+started_at=%Y-%m-%dT%H:%M:%S%z'
} >"$OUTPUT_DIR/judge-run.env"

# The judge is expected to use the prompt verbatim. We append a
# system-style suffix telling the agent that the output contract is
# response.json under OUTPUT_DIR.
PROMPT_BODY="$(cat "$PROMPT")
---
Write your verdict as JSON to: $OUTPUT_DIR/response.json
Follow the schema described in the prompt above."

claude --model "$MODEL" -p "$PROMPT_BODY" >"$OUTPUT_DIR/raw-output.md"

# If the agent wrote raw JSON to stdout instead of response.json,
# attempt a permissive extraction.
if [[ ! -f "$OUTPUT_DIR/response.json" ]]; then
  if grep -q '^{' "$OUTPUT_DIR/raw-output.md" 2>/dev/null; then
    awk '/^{/,/^}/' "$OUTPUT_DIR/raw-output.md" >"$OUTPUT_DIR/response.json"
  fi
fi
