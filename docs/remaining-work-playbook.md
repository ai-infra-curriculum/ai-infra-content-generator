# Remaining Work Playbook

This reference tracks every outstanding item needed to finish the AI Infrastructure Curriculum migration and provides actionable steps for completing each task. Use it as the canonical punch list during the hardening phase.

---

## 1. Solution Content Backfill

**Goal:** Replace placeholder solution stubs with production-ready content across advanced roles.

### Roles & Assets Still Missing Full Solutions
- Principal Engineer & Team Lead (leadership scenarios)
- AI Infrastructure Security Engineer (threat simulations, compliance packs)
- AI Infrastructure Performance Engineer (benchmark artefacts, profiler traces)
- Cross-role shared labs referenced in `curriculum/roles/multi-role-alignment.md`

### Execution Steps
1. Inventory each role directory in `solutions/<role>/` and open the `HOUSEKEEPING_REPORT.md`.
2. For every module or project flagged as `TODO`:
   - Pull legacy material from `/home/claude/ai-infrastructure-project/` using the mapping in `curriculum/repository-strategy.yaml`.
   - Normalize filenames to the new ID structure (e.g., `PROJ-521`).
   - Update `solutions/<role>/<module-or-project>/metadata.yaml` with:
     - `solution_status: complete`
     - `artifacts:` list referencing the new files.
3. Document validation evidence in the same directory (`VALIDATION_NOTES.md` or existing README).
4. Mark the task as `DONE` in the housekeeping report and commit the new artefacts.

### Acceptance Criteria
- No modules/projects retain `placeholder` or `pending` status in solution metadata.
- README files clearly describe how to run or verify each solution.
- Validation evidence exists (screenshots, logs, benchmark tables where applicable).

---

## 2. Validation Pipeline

**Goal:** Ensure all metadata, exporters, and dependency graphs compile cleanly after content backfill.

### Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt  # if dependencies change
```

### Commands to Run
```bash
./tools/curriculum.py validate-metadata
./tools/curriculum.py generate-mkdocs-nav
./tools/curriculum.py export-graph
./tools/curriculum.py run-validation --profile performance
./tools/curriculum.py run-validation --profile leadership
```

> Adjust profiles as new validation bundles are added.

### Documentation Updates
- Append results and timestamps to `validation/README.md`.
- Log any remediation work in `docs/migration-plan.md` (Section 7: Verification & Sign-Off).
- If exporters produce new files, ensure `.gitignore` covers transient output.

---

## 3. Documentation Refresh

**Goal:** Align reference docs with the final structure and decision paths.

### Required Edits
- `docs/migration-plan.md`: tick completion boxes as tasks finish; add validation dates.
- `docs/ai-infrastructure-curriculum-guide.md`: link to new solution repositories or shared assets.
- `curriculum/roles/multi-role-alignment.md`: update matrices once cross-role content convergence is verified.
- `CHANGELOG.md`: summarize major milestones before release tagging.

### Suggested Workflow
1. After completing solution backfill for a role, update the relevant sections immediately.
2. Run `git diff` to review doc changes for accuracy and tone.
3. Solicit stakeholder review (Engineering leadership, Curriculum PM) before publishing.

---

## 4. Governance & Release Prep

**Goal:** Produce artifacts needed for an official curriculum release.

### Checklist
- [ ] Sign-off matrix populated in `governance/release-approvals.yaml`.
- [ ] `RELEASE_NOTES.md` drafted (use template in `templates/governance/release-notes-template.md`).
- [ ] Version bump performed in `pyproject.toml` (if tooling release accompanies curriculum push).
- [ ] Tag creation strategy documented in `docs/best-practices.md`.

---

## 5. Post-Migration Enhancements (Optional)

These items are not blockers for release but increase the framework’s long-term flexibility.

- Automate solution validation (CI workflow enhancements).
- Add content linting scripts (e.g., check metadata keys, verify IDs).
- Build exporter adapters for LMS targets beyond the sample JSON/YAML.
- Expand guidance for multi-role sequencing in `curriculum/roles/multi-role-alignment.md`.

---

## Execution Guidance

- Work in small batches: finish one role or project set before switching contexts.
- Use dedicated branches per task group (e.g., `feature/performance-solutions`).
- Document discoveries or deviations in `docs/migration-plan.md` to keep the plan audit-ready.
- Coordinate with content owners using the issue templates in `.github/ISSUE_TEMPLATE/`.

Once all mandatory sections are marked complete and validations pass, proceed with final QA and release packaging.
