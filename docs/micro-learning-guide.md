# Micro-Learning Production Guide

This guide describes how to create a focused micro-learning experience (2–4 hours seat time) using the framework’s micro-learning pipeline manifest.

---

## When to Use

- Rapid enablement on a single competency or tool.
- Supplemental lessons between full modules.
- Pre-work or follow-up reinforcement for workshops.

## Prerequisites

- Role research and competency mapping (`research/<role>/`).
- A micro-learning idea scoped to a single learning objective.
- Repository strategy (inline vs separate solutions) defined in `curriculum/repository-strategy.yaml`.

## Pipeline Overview

Manifest: `pipelines/micro-learning.yaml`

| Phase | Goal | Key Outputs |
|-------|------|-------------|
| Scope | Define objective & prerequisites | `lessons/<slug>/metadata.yaml` |
| Lesson Draft | Produce 2–3k word lesson | `lesson.md` |
| Exercise | Hands-on activity | `exercise.md` |
| Validation | Run docs-quality checks | Validation logs |
| Solution (optional) | Provide answer key | `solutions/exercise/solution.md` |
| Publish | Update changelog & notify team | Release announcement |

## Step-by-Step

1. **Scaffold directory**
   ```bash
   mkdir -p lessons/<slug> solutions/exercises/<slug>
   ./tools/curriculum.py scaffold-metadata lessons/<slug>/metadata.yaml \
     --id ML-<NN> --type module --title "Lesson Title" --validation-profile docs-quality
   ```
   _Reference implementation_: `lessons/sample-micro/`

2. **Draft lesson content**
   - Use relevant sections of `prompts/lecture-generation/comprehensive-module-prompt.md` (trim to essential sections).
   - Target 2,000–3,000 words focused on the single objective.

3. **Create exercise**
   - Copy `templates/exercises/exercise-template.md`.
   - Emphasize immediate application of the lesson content.

4. **Optional solution**
   - Copy `templates/solutions/exercise-solution-template.md`.
   - Provide concise steps and validation commands.

5. **Run validation profile**
   ```bash
   ./tools/curriculum.py run-validation docs-quality lessons/<slug>
   ```

6. **Publish or stage**
   - Update metadata status to `released`.
   - Add entry to `CHANGELOG.md`.
   - If exporting to docs/LMS, update `exporters/samples/*` as needed.
    - Use GitHub Actions (see `.github/workflows/validation.yml`) or another automation method to run validation before publishing.

## Best Practices

- Keep scope tight: one core objective, one primary exercise.
- Reference the parent module or competency in metadata (`dependencies` field).
- Reuse template partials (`templates/partials/`) for production best practices.
- Log prompt versions in metadata (`prompts_used`) for reproducibility.

For deeper automation and exporter configuration, see `docs/metadata-and-automation.md` and `exporters/README.md`.
