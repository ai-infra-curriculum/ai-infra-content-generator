# Renaming Workflow Guide

Use this guide when renaming any curriculum artifact (roles, modules, lessons, projects, or assessments). It lists every file that must be updated so references, exports, and automation continue to work.

---

## General Principles

- **IDs are canonical.** Every module (`MOD-XXX`), project (`PROJ-XXX`), assessment (`ASSMT-XXX`), and lesson (`LESSON-XXX`) uses an ID that ties metadata together. If the name changes but the scope remains, keep the ID. Only change IDs when there is a fundamental re-scope.
- **Directory names mirror IDs.** Folder names use the pattern `<id>-<slug>`. When title or ID changes, update the folder to match.
- **Search globally before committing.** Run `rg "OLD_NAME"` and `rg "OLD_ID"` to catch lingering references.
- **Update metadata timestamps.** Set `date_updated` or `last_modified` fields in YAML/Markdown files to reflect the change.

---

## 1. Renaming a Module

Assume we are renaming `modules/ai-infrastructure-engineer/module-104-kubernetes-fundamentals` to “Kubernetes Platform Foundations” while keeping ID `MOD-104`.

### Steps
1. **Directory & File Names**
   - Rename the directory: `modules/<role>/module-104-kubernetes-fundamentals` → `module-104-kubernetes-platform-foundations`.
   - Update lesson subdirectories under `lessons/` if their slugs include the old name.
   - Rename any associated solution directory: `solutions/<role>/module-104-kubernetes-fundamentals`.
2. **Module Metadata**
   - Edit `modules/<role>/module-104-*/metadata.yaml`:
     - Update `title`, `slug`, `summary`, and `last_updated`.
     - Adjust `tags` or `learning_objectives` if the scope shifts.
   - Update `modules/<role>/module-104-*/solutions/metadata.yaml` to reflect new module title.
3. **Curriculum References**
   - Update the title in `curriculum/<role>/master-plan.yaml` under the `program_structure.modules` list.
   - Update any module roadmaps (`curriculum/<role>/modules/module-104-*/roadmap.md`).
   - Adjust prerequisite or dependency references in other modules’ metadata (search for `MOD-104`).
4. **Exporters & Samples**
   - Edit `exporters/samples/mkdocs.yml` and `exporters/samples/lms-export.json` if the module is listed.
   - Update any CSV or JSON exports maintained under `exporters/`.
5. **Repository Strategy**
   - If repositories or solution paths changed, update `curriculum/repository-strategy.yaml` and the relevant file under `configs/repositories/`.
6. **Validation**
   - Run `./tools/curriculum.py validate-metadata` to confirm schema compliance.
   - Regenerate MkDocs navigation and LMS samples if names surfaced there.

### Files to Touch (Typical)
- `modules/<role>/module-XXX-*/metadata.yaml`
- `modules/<role>/module-XXX-*/solutions/metadata.yaml`
- `modules/<role>/module-XXX-*/lessons/*/metadata.yaml`
- `solutions/<role>/module-XXX-*/README.md` (and associated assets)
- `curriculum/<role>/master-plan.yaml`
- `curriculum/<role>/modules/module-XXX-*/roadmap.md`
- `exporters/samples/*`
- `curriculum/repository-strategy.yaml`
- `configs/repositories/<role>.yaml`

---

## 2. Renaming a Project

Example: Renaming `projects/ai-infrastructure-engineer/project-203-kubernetes-observability`.

### Steps
1. Rename the directory and update the slug inside `projects/<role>/project-203-*/metadata.yaml`.
2. Update linked module references:
   - Search for `PROJ-203` across `curriculum/<role>/master-plan.yaml` and module `linked_projects` lists.
   - Adjust `solutions/<role>/project-203-*/metadata.yaml`, README, and artefact paths.
3. Update exporter listings (`exporters/samples/*.json|yml`) and repository configs if the repo path changes.
4. Regenerate validation assets and update documentation (`docs/remaining-work-playbook.md`, if relevant).

### Additional Notes
- Projects often include `project-plan.md`, `rubric.md`, and `checklist.md`; ensure titles and internal references reflect the new name.
- If the project lives in a dedicated repository, rename the repo in `configs/repositories/<role>.yaml` and any Git submodule references.

---

## 3. Renaming a Lesson or Assessment

### Steps
1. Update the lesson directory under `modules/<role>/module-XXX-*/lessons/`.
2. Edit `metadata.yaml` within the lesson to change `title`, `slug`, and `last_updated`.
3. Update references in the module roadmap (lesson sequencing tables).
4. If the lesson has exercises or assessments with solution files, update those directories and metadata as well (`exercises/`, `assessments/` subfolders).
5. Rerun validation to ensure lesson IDs still match the module metadata.

---

## 4. Renaming an Entire Role

### Steps
1. Rename top-level directories:
   - `curriculum/<old-role>/` → `curriculum/<new-role>/`
   - `modules/<old-role>/`, `projects/<old-role>/`, `solutions/<old-role>/`, `research/<old-role>/`
   - `configs/repositories/<old-role>.yaml`
2. Update role metadata files:
   - `curriculum/<new-role>/master-plan.yaml` (`role`, `program_name`, `repository_config`).
   - `curriculum/roles/multi-role-alignment.md` (tables, narratives, matrices).
3. Adjust global references:
   - `docs/ai-infrastructure-curriculum-guide.md`
   - `docs/migration-plan.md`
   - `curriculum/repository-strategy.yaml`
   - `exporters/samples/*`
   - `validation/README.md` (role status tables)
4. Update prompts or templates if they are role-specific (`prompts/roles/`, `templates/research/`).
5. Run `rg "<old-role>"` and `rg "<Old Role Title>"` to ensure no lingering references remain.

---

## 5. Post-Rename Checklist

1. Run full validation suite (metadata, exporters, graph).
2. Update `CHANGELOG.md` with a summary of the rename and rationale.
3. If external repositories or stakeholders are affected, submit a communication via the governance process (`templates/governance/change-brief-template.md`).
4. Create or update issue templates or automation triggers if the rename affects them (`.github/ISSUE_TEMPLATE/`).

Following this workflow ensures that renames propagate consistently across the curriculum, automation tooling, and downstream repositories.
