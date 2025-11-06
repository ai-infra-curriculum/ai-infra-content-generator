# Pipeline Manifests

Pipeline manifests describe reusable sequences of phases, templates, prompts, and validation steps. They make it easy to automate or orchestrate curriculum production.

## Available Pipelines

| Manifest | Description |
|----------|-------------|
| `single-module.yaml` | End-to-end workflow for generating one module (research context, drafting, validation, solutions). |
| `micro-learning.yaml` | Lightweight sprint for micro-lessons (2-4 hours seat time) with focused validation. |
| `ai-infra-program.yaml` | Complete multi-role AI Infrastructure Curriculum pipeline. |

## Inspecting Pipelines

Use the CLI helper:

```bash
./tools/curriculum.py pipelines
./tools/curriculum.py pipeline ai-infra-program
```

## Executing Pipelines

Pipelines are declarative. You can:
- Feed them into an automation agent or orchestration script.
- Translate phases into GitHub Actions, notebooks, or task trackers.
- Customize manifests or create new ones for different program types (micro-learning, certification sprints, etc.).

## Creating New Pipelines

1. Copy an existing manifest.
2. Update `metadata` (name, description, roles).
3. List phases with relevant instructions:
   - `workflow`: reference Markdown workflow guides.
   - `prompts` / `templates`: direct contributors to assets.
   - `commands`: automated scripts to run.
4. Document the manifest in this README.
