# Curriculum Migration Plan

> For a consolidated execution checklist, see `docs/remaining-work-playbook.md`.

1. **Inventory & Snapshot** — _Completed 2025-10-15_
   - Archive current state of role repositories, research assets, and planning docs.
   - Ensure checkpoints/backups exist for rollback.

2. **Research Migration** — _Status: In progress_
- ✅ Junior AI Infrastructure Engineer (`research/junior-ai-infrastructure-engineer/*`)
- ✅ AI Infrastructure Engineer (`research/ai-infrastructure-engineer/*`)
- ✅ Senior AI Infrastructure Engineer (`research/senior-ai-infrastructure-engineer/*`)
- ✅ AI Infrastructure Architect (`research/ai-infrastructure-architect/*`)
- ✅ Principal AI Infrastructure Architect (`research/ai-infrastructure-principal-architect/*`)
- ✅ Principal AI Infrastructure Engineer (`research/ai-infrastructure-principal-engineer/*`)
- ✅ AI Infrastructure Team Lead (`research/ai-infrastructure-team-lead/*`)
- ✅ AI Infrastructure Security Engineer (`research/ai-infrastructure-security-engineer/*`)
- ✅ AI Infrastructure ML Platform Engineer (`research/ai-infrastructure-ml-platform-engineer/*`)
- ✅ AI Infrastructure MLOps Engineer (`research/ai-infrastructure-mlops-engineer/*`)
- ✅ AI Infrastructure Performance Engineer (`research/ai-infrastructure-performance-engineer/*`)
   - ☐ Remaining leadership roles (Principal Engineer, Team Lead) for solution backfill & validation
   - ☐ Validate migrated metadata via `./tools/curriculum.py validate-metadata` (blocked: Python tooling not yet installed)

3. **Curriculum Plans** — _Status: In progress_
- ✅ Junior master plan, module roadmaps, project plans
- ✅ AI Infrastructure Engineer master plan, roadmaps, projects
- ✅ Senior AI Infrastructure Engineer master plan, roadmaps, projects
- ✅ AI Infrastructure Architect master plan, roadmaps, projects
- ✅ Principal AI Infrastructure Architect master plan, roadmaps, projects
- ✅ Principal AI Infrastructure Engineer master plan, roadmaps, projects
- ✅ AI Infrastructure Team Lead master plan, roadmaps, projects
- ✅ AI Infrastructure Security Engineer master plan, roadmaps, projects
- ✅ AI Infrastructure ML Platform Engineer master plan, roadmaps, projects
- ✅ AI Infrastructure MLOps Engineer master plan, roadmaps, projects
- ✅ AI Infrastructure Performance Engineer master plan, roadmaps, projects
   - ☐ Update remaining role plan templates once prioritised

4. **Module & Lesson Content** — _Status: In progress_
- ✅ Junior and mid-level modules migrated with metadata
- ✅ Senior modules (MOD-201…210) migrated with metadata and placeholder solution notes
- ✅ Architect modules (MOD-301…310) migrated with metadata and solution stubs
- ✅ Principal modules (MOD-601…610) migrated with metadata and solution placeholders
- ✅ Principal engineer modules (MOD-701…706) migrated with metadata and solution placeholders
- ✅ Team lead modules (MOD-801…810) migrated with metadata and leadership solution stubs
- ✅ Security engineer modules (MOD-901…912) migrated with metadata and security solution placeholders
- ✅ ML platform modules (MOD-501…510) migrated with metadata and solution placeholders
- ✅ MLOps modules (MOD-551…560) migrated with metadata and solution placeholders
- ✅ Performance engineer modules (MOD-521…528) migrated with metadata and solution placeholders
   - ☐ Run validation profiles (`./tools/curriculum.py run-validation …`) after environment setup (pip missing)

5. **Projects & Solutions** — _Status: In progress_
- ✅ Junior projects + solutions metadata
- ✅ AI Infrastructure Engineer projects + solutions metadata
- ✅ Senior projects (PROJ-301…304) with legacy solutions copied and metadata linked
- ✅ Architect projects (PROJ-401…405) with legacy solutions copied and metadata linked
- ✅ Principal projects (PROJ-601…604) with metadata and solution stubs connected to legacy repos
- ✅ Principal engineer projects (PROJ-701…703) with metadata and solution placeholders linked to legacy repos
- ✅ Team lead projects (PROJ-801…803) with metadata and leadership solution placeholders synced to legacy repos
- ✅ Security engineer projects (PROJ-901…905) with metadata and solution placeholders tied to legacy security repositories
- ✅ ML platform projects (PROJ-501…506) with metadata and solution placeholders mapped to legacy platform assets
- ✅ MLOps projects (PROJ-551…555) with metadata and solution placeholders mapped to legacy assets
- ✅ Performance engineer projects (PROJ-521…524) with metadata and solution placeholders mapped to legacy assets
   - ☐ Curate senior module solution READMEs with direct references to legacy guides

6. **Exporters & Automation** — _Status: In progress_
   - ✅ Update LMS sample (`exporters/samples/lms-export.json`) with junior through principal coverage
   - ✅ Update MkDocs sample (`exporters/samples/mkdocs.yml`) navigation across roles
  - ☐ Execute automation commands (`generate-mkdocs-nav`, `export-graph`) once tooling is available

7. **Verification & Sign-Off** — _Status: Pending_
   - ☐ Install Python dependencies and run validation pipelines (metadata, exporters, graph checks)
   - ☐ Update `CHANGELOG.md` and governance docs after validation passes
   - ☐ Route senior curriculum for stakeholder review (Platform Architect, Reliability director)

## Next Steps & Blockers

- Provision Python tooling (pip/virtualenv) to run `./tools/curriculum.py validate-metadata`, exporter generation, and graph checks.
- Backfill senior module solution documentation with curated links to legacy guides and CI instructions.
- Align remaining advanced roles (Principal Engineer, Team Lead, specialized tracks) once priorities are confirmed.
