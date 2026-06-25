# AICG Roadmap

The autonomous curriculum pipeline (design: `autonomous-curriculum-pipeline.md`)
is built and live for the AI-infra curriculum. This roadmap tracks where it goes
next: finishing the staged rollout, generalizing it into a domain-neutral engine,
and putting a control plane in front of it.

> Status legend: ✅ done · 🔄 in progress · ⏳ planned

---

## 0. Current state (✅)

All P0–P6 phases are implemented, tested (19 pipeline modules, 463 tests), and
live on the runner:

- **P0** eval-gate — flag-only judge, calibrated BAR=76 (ACT)
- **P1** author loop — autonomous authoring quality-gated before merge (ACT)
- **P2–P5** re-audit / research-add / retire / package — running in OBSERVE
  (dry-run) mode via `aicg org pipeline-tick`, deciding on real data, writing
  nothing
- **P6** fan-out — applies across all 17 roles / 34 repos

---

## 1. Finish the staged rollout — promote write-phases observe → act (⏳)

Each write-phase is live in observe mode; promotion is a watched flag-flip plus a
little act-mode wiring. Do these one at a time, watching each for a cycle.

- **P2 re-audit auto-fix** — add `refresh_stale` to `HANDLED_WORK_TYPES` + a
  regenerate-by-path handler; flip `pipeline.phases.reaudit_autofix` /
  `quality_judge.flag_only=false`.
- **P3 research auto-promote** — wire `research_cycle` promotion into the research
  job (auto-apply qualifying deltas instead of proposing); flip
  `research_autopromote`.
- **P4 retire** — build the retire executor (git rm + tombstone + reconciliation
  + dangling-ref scan), validate as a real dry-run to a branch, then flip
  `retire`. Consumes the evidence research already gathers.
- **P5 package** — wire `plan_monthly_package` into `monthly-release` (tag changed
  repos + build Release tarballs + `ARCHIVE_INDEX.md`); flip `package`.

---

## 2. Domain-neutral engine — generalize beyond AI (⏳)

The pipeline core is already domain-agnostic. Four changes turn "AI curriculum
tool" into a curriculum engine that works for any profession. Ordered by leverage.

1. **Domain-configurable freshness rubric** ✅ *(highest leverage — the seam; done 2026-06-25)*
   `quality_judge.freshness_dimensions` is now a list of `{name, description}`
   pairs (defaults to the AI set, so the AI curriculum is unchanged). A non-tech
   domain supplies its own — nursing grades `guideline_currency` /
   `regulation_currency`, law `statute_currency` / `caselaw_currency` — and the
   same pipeline grades it. The engine is now provably domain-neutral. (Also
   fixed a latent bug: the freshness prompt was iterating the *quality* dims.)

2. **Multi-tenant config + git/gh plumbing**
   `config/aicg-org.yaml` hardcodes one org, remote, and role list. Make the
   harness run N domain configs, each pointing at its own GitHub org + repos +
   role ladder. "Domain" becomes the tenancy boundary above "role."

3. **Per-domain calibration corpus**
   Each domain needs its own good/bad exemplars to pick its own BAR. Structure
   `calibration/<domain>/corpus/`; `calibrate-judge` runs per domain.

4. **Per-domain research source + role titles**
   The evidence gate / aliases / frequency are already generic — just point them
   at the domain's titles + posting source. (Postings exist for every
   profession; the easy one.)

---

## 3. Multi-tenancy & provisioning — `bootstrap-domain` (🟡 partial)

A provisioning capability above the existing per-role `bootstrap-role`:

- Create the GitHub org (or target an existing one). — **manual** (needs
  `admin:org`/web UI; org creation isn't in the runner's token scope).
- Seed the domain's role ladder (entry → senior → architect/lead → exec, or the
  domain's equivalent). — ✅ via `config/domains/<name>.yaml`.
- Scaffold the paired `-learning` / `-solutions` repos per role. — ✅
  **`bootstrap-role` is now domain-aware** (`--domain` + `--create-remotes`);
  repo names/branding resolve from the manifest, not a hardcoded `ai-infra-`
  prefix. Proven end-to-end standing up **ml-engineering-curriculum**
  (2026-06-25): 8 role repos + `.github` profile created & pushed.
- Drop in the domain rubric (§2.1) + a starter calibration corpus (§2.3). — ⏳
- Wire the per-role timers + register the domain config. — register ✅; timers
  are manifest-driven (`install-schedules.sh`) but ai-infra-only so far
  (sibling orgs are observe-only, phases off).
- Hand off to the same P0–P6 pipeline. — ⏳ (gated on enabling phases per domain).

Remaining for a true one-command `bootstrap-domain`: wrap org creation +
`.github` profile generation + 4-way cross-link wiring (all done by hand for the
first three sibling orgs) around the now-domain-aware `bootstrap-role`.

Outcome: standing up a new domain is a configuration + provisioning exercise, not
a rewrite.

---

## 4. Web UI / control plane (⏳)

Today's surfaces are CLI (`pipeline-status`, `pipeline-tick`, `calibrate-judge`)
and GitHub-native (status issues, PRs, the heartbeat). A control plane puts a
single pane over the multi-domain fleet.

**Observability**

- Fleet dashboard: per-domain / per-role phase state (observe vs act), queue
  depth, daily budget usage, quarantine flags, calibration status (BAR per
  domain), heartbeat health.
- Activity feed: recent merges, retirements, promotions, monthly changelogs.

**Control**

- Promote a phase observe → act per domain (the staged flag-flip, with a
  confirmation + the last observe output shown).
- Trigger a calibration run; view the good/bad separation + suggested BAR; accept.
- Review the quarantine queue: inspect failing artifacts + rubric scores;
  re-queue, override, or leave flagged.
- Pause / resume a domain or the whole fleet (kill switch).

**Provisioning**

- Launch `bootstrap-domain` from the UI: name the domain, pick the role ladder,
  point at a GitHub org, upload/seed the rubric + corpus.

**Architecture notes**

- Read model from the existing state files (`work-queue.json`,
  `daily-run-state.json`, the per-role manifests, calibration reports) + GitHub
  API; the runner stays the source of truth, the UI is a view + a thin control
  layer that writes config flags / enqueues commands.
- Keep the runner authoritative and the UI optional — the pipeline must keep
  running headless if the UI is down (consistent with the no-single-point-of-
  failure stance).

---

## Sequencing

```text
0 (done) ──> 1 promote P2..P5 (near-term)
         └─> 2.1 rubric-config ──> 2.2 multi-org ──> 3 bootstrap-domain ──> 2.3/2.4 per-domain corpus+source
                                                                        └─> 4 control plane (parallel once multi-domain exists)
```

§2.1 (rubric-config) is the cheapest highest-signal step and gates the rest of
the generalization. The control plane (§4) becomes worthwhile once there's more
than one domain to manage.
