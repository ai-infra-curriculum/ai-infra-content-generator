"""Fleet status — a read-only roll-up across every curriculum domain.

Roadmap §4 (control plane) starts here: now that the harness runs four
domains (ai-infra + ml-engineering + ai-engineering + ai-governance), an
operator needs one pane that answers "how live is each domain?" without
opening four manifests. This module is the read model; it makes no writes
and no network calls. The pure functions below take already-loaded
manifests so they're trivially testable; the CLI layer does the IO.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .judge import JudgeConfig
from .pipeline_config import PipelineConfig


@dataclass(frozen=True)
class DomainStatus:
    domain: str
    org: str
    role_count: int
    enabled_phases: tuple[str, ...]
    daily_budget: int
    judge_enabled: bool
    judge_flag_only: bool
    freshness_bar: int | None
    queue_depth: int | None  # None = no work-queue state file found

    @property
    def mode(self) -> str:
        """At-a-glance liveness: ACT (writing), OBSERVE (judging, flag-only),
        or INERT (judge off, no write-phases)."""
        if self.enabled_phases:
            return "ACT"
        if self.judge_enabled:
            return "OBSERVE"
        return "INERT"


def build_domain_status(
    domain: str, manifest: Any, queue_depth: int | None = None
) -> DomainStatus:
    """Derive a DomainStatus from a loaded manifest. Pure — no IO."""
    pc = PipelineConfig.from_manifest(manifest)
    jc = JudgeConfig.from_manifest(manifest)
    bar = None
    if jc.thresholds:
        bar = jc.thresholds.get("freshness", jc.thresholds.get("default"))
    return DomainStatus(
        domain=domain,
        org=manifest.org,
        role_count=len(manifest.roles),
        enabled_phases=tuple(pc.enabled_phases()),
        daily_budget=pc.daily_budget,
        judge_enabled=jc.enabled,
        judge_flag_only=jc.flag_only,
        freshness_bar=bar,
        queue_depth=queue_depth,
    )


def read_queue_depth(state_dir: Path) -> int | None:
    """Best-effort: count open items in ``<state_dir>/work-queue.json``.

    Returns None when the file is absent (a domain that hasn't run yet) or
    unreadable, so the fleet view degrades gracefully rather than erroring.
    """
    qpath = state_dir / "work-queue.json"
    if not qpath.is_file():
        return None
    try:
        data = json.loads(qpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None
    open_states = {"queued", "pending", "open", "in_progress", "ready"}
    counted = [
        it for it in items
        if not isinstance(it, dict) or it.get("status", "queued") in open_states
    ]
    return len(counted)


def render_fleet_table(statuses: list[DomainStatus]) -> str:
    """Render a fixed-width table of the fleet. Pure formatting."""
    header = (
        f"{'DOMAIN':<16} {'ORG':<26} {'ROLES':>5} {'MODE':<8} "
        f"{'PHASES':<22} {'JUDGE':<14} {'BAR':>4} {'BUDGET':>6} {'QUEUE':>6}"
    )
    lines = ["Fleet status — AI Career Curriculum ecosystem", "", header, "-" * len(header)]
    for s in sorted(statuses, key=lambda d: d.domain):
        phases = ",".join(s.enabled_phases) if s.enabled_phases else "—"
        judge = (
            f"{'on' if s.judge_enabled else 'off'}"
            + (" (flag)" if s.judge_enabled and s.judge_flag_only else "")
        )
        bar = str(s.freshness_bar) if s.freshness_bar is not None else "—"
        queue = "—" if s.queue_depth is None else str(s.queue_depth)
        lines.append(
            f"{s.domain:<16} {s.org:<26} {s.role_count:>5} {s.mode:<8} "
            f"{phases:<22} {judge:<14} {bar:>4} {s.daily_budget:>6} {queue:>6}"
        )
    total_roles = sum(s.role_count for s in statuses)
    act = sum(1 for s in statuses if s.mode == "ACT")
    observe = sum(1 for s in statuses if s.mode == "OBSERVE")
    inert = sum(1 for s in statuses if s.mode == "INERT")
    lines.append("")
    lines.append(
        f"{len(statuses)} domains · {total_roles} roles · "
        f"{act} ACT / {observe} OBSERVE / {inert} INERT"
    )
    return "\n".join(lines)
