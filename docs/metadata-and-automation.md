# Metadata & Automation Guide

This guide explains how to use the new automation components that make the framework flexible and program-aware.

## 1. Asset Metadata Schema

- Schema: `schemas/asset-metadata.schema.json`
- Purpose: Standardize metadata for modules, projects, exercises, assessments, and solutions.
- Recommended storage: YAML front matter or sidecar files (`metadata.yaml`).
- Validation: `./tools/curriculum.py validate-metadata path/to/metadata.yaml`

### Minimal Metadata Example

```yaml
id: MOD-03
type: module
title: Observability in Production ML
roles: [platform-engineer, mlops-engineer]
stage: core
proficiency_target: proficient
competencies:
  - id: obs.metrics
    level: working
  - id: obs.alerting
    level: proficient
dependencies: [MOD-01]
validation_profile: python-strict
status: in-review
owners:
  - name: Taylor Nguyen
    github: "@tay-dev"
    role: author
metadata_version: "1.0.0"
```

## 2. Pipeline Manifests

- Location: `pipelines/`
- Purpose: Declarative definitions of end-to-end workflows.
- Reference: `pipelines/README.md`
- Usage:
  ```bash
  ./tools/curriculum.py pipelines
  ./tools/curriculum.py pipeline ai-infra-program
  ```
- Combine with automation or notebook agents to drive execution.

## 3. Validation Profiles

- Located at `configs/validation-profiles.yaml`.
- Reference profile names in metadata or manifests (`validation_profile: python-strict`).
- Each profile lists required commands/tools; integrate with CI or local scripts.

## 4. Cross-Role Content Graph

- Schema: `graphs/ai-infra.graphql`
- Purpose: Model relationships across roles, modules, projects, and solutions.
- Tooling ideas:
  - Generate graph JSON for dashboards.
  - Build queries to visualize prerequisite chains.
  - Integrate with LMS/API exports.

## 5. CLI Helper

- Script: `tools/curriculum.py`
- Capabilities:
  - List/show pipelines.
  - Validate metadata against JSON schema.
  - List validation profiles.
  - Run validation profiles: `./tools/curriculum.py run-validation python-strict modules/module-01`
  - Scaffold metadata stubs: `./tools/curriculum.py scaffold-metadata modules/module-01/metadata.yaml --id MOD-01 --type module --title "Foundations"`
- Dependencies (install once):
  ```bash
  pip install -r requirements-tooling.txt
  ```

## 6. Prompt Versioning

- Documentation: `prompts/README.md`
- Version map: `prompts/version-map.yaml`
- Best practice: record prompt versions in asset metadata to ensure reproducibility.

## 7. Template Partials

- Directory: `templates/partials/`
- Purpose: Reuse content blocks across modules/projects (e.g., production best practices).
- Include via your static-site generator or manual copy.

---

### Suggested Workflow Integration

1. Create metadata for each asset (`metadata.yaml` adhering to schema). Use `templates/curriculum/module-metadata-template.yaml` or `templates/curriculum/project-metadata-template.yaml` as starting points.
2. Select validation profile(s) and embed in metadata.
3. Choose a pipeline manifest (`pipelines/*.yaml`) and follow the phases.
4. Use CLI helper for quick introspection and metadata validation.
5. Update cross-role graph as new modules/projects are added.
6. Record prompt versions in metadata and changelog for transparency.
7. Configure exporters (see `exporters/README.md`) to publish content downstream. Use `./tools/curriculum.py generate-mkdocs-nav modules/` to produce navigation snippets.
8. Export graph data for dashboards: `./tools/curriculum.py export-graph modules/ projects/ lessons/ --output graphs/generated.json`
9. Automate validation via CI (sample workflow: `.github/workflows/validation.yml`).

These components make the framework configurable, automation-ready, and adaptable to new program types without rewriting templates or workflows.
