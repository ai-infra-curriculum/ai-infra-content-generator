# Validation Status & Blockers

> Updated 2025-10-20 — tracking outstanding work required before running the end-to-end validation suite.

## Environment Gaps

- `pip`/Python tooling not installed in the current workspace, blocking:
  - `./tools/curriculum.py validate-metadata`
  - `./tools/curriculum.py generate-mkdocs-nav`
  - `./tools/curriculum.py export-graph`
  - Project/module validation runners referenced in roadmap checklists
- GPU-enabled runners unavailable for automated execution of senior distributed training and inference benchmarks (manual validation required or add mocked harness).
- Architect-level simulations (FinOps cost models, DR/chaos drills) require manual review or dedicated tooling outside current sandbox.

## Pending Validation Tasks

1. Install Python dependencies (`python -m venv .venv && source .venv/bin/activate && pip install -r requirements-tooling.txt`).
2. Execute `./tools/curriculum.py validate-metadata` across all roles (junior through team lead and principal tracks).
3. Run exporter samples:
   - `./tools/curriculum.py generate-mkdocs-nav --output exporters/generated/mkdocs-nav.yml`
   - `./tools/curriculum.py export-graph --output graphs/ai-infra.graph.json`
4. Trigger project-specific validation scripts documented in each solutions folder (e.g., `make validate`, `.github/workflows/*` dry-runs).
5. Capture benchmark outputs for senior projects (PROJ-301, PROJ-302) and attach to project directories; document leadership, platform, optimization, and security outcomes for advanced projects (PROJ-501…506, PROJ-601…604, PROJ-701…703, PROJ-801…803, PROJ-901…905).
6. Review architect FinOps models, DR simulations, and governance artifacts with finance/compliance stakeholders and record approvals; extend review to principal architect, principal engineer, team lead, security engineer, and ML platform engineer portfolios.

## Next Actions

- [ ] Create or reuse CI job that provisions Python environment and runs metadata/exporter validation.
- [ ] Define GPU validation strategy (cloud credits vs. simulation) and document fallback instructions in senior project READMEs.
- [ ] Update module/project roadmaps to check off validation items once runs succeed, including team lead, security engineer, and principal-level roadmaps/project plans.
