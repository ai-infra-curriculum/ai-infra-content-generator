# Migration Notes – AI Infrastructure MLOps Engineer

## Legacy Source Repositories
- Learning: `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-mlops-learning`
- Solutions: `/home/claude/ai-infrastructure-project/repositories/solutions/ai-infra-mlops-solutions`

## Legacy Module Inventory (Lessons)
| Legacy Module | Proposed New ID | Notes |
| --- | --- | --- |
| 01-mlops-foundations | MOD-551 | Establishes lifecycle terminology; references junior MOD-009 monitoring basics. |
| 02-cicd-for-ml | MOD-552 | Builds on MOD-109 IaC and senior MOD-206 automation to introduce GitOps/CI-CD for ML. |
| 03-model-monitoring | MOD-553 | Extends junior monitoring (MOD-009) and senior observability (MOD-207) with drift detection. |
| 04-data-quality | MOD-554 | Reuses data validation artifacts from MOD-105 pipelines; introduces Great Expectations checks. |
| 05-experimentation | MOD-555 | Builds on senior experimentation governance (MOD-206) and ML platform MOD-504 feature store outputs. |
| 06-automation | MOD-556 | Advanced orchestration combining Airflow/Kubeflow; references ML platform MOD-505 workflows. |
| 07-governance | MOD-557 | Shares governance templates with security MOD-907 and architect MOD-603. |
| 08-production-ops | MOD-558 | Extends senior SRE practices (MOD-207) for ML runbooks; ties into platform observability MOD-508. |
| 09-security | MOD-559 | Pulls from security modules (MOD-901–910) adapted to MLOps context. |
| 10-advanced-topics | MOD-560 | Capstone synthesizing LLMOps, multi-cloud rollouts, and responsible AI (architect MOD-605). |

## Legacy Project Inventory
| Legacy Project | Proposed New ID | Notes |
| --- | --- | --- |
| project-1-ml-pipeline | PROJ-551 | Production-ready pipeline; preserve full solution artefacts. |
| project-2-model-serving | PROJ-552 | Needs solution metadata; reuses platform serving templates. |
| project-3-experimentation | PROJ-553 | Align with experimentation module (MOD-555). |
| project-4-governance | PROJ-554 | Connect to governance module (MOD-557) and security guidance. |
| project-5-llmops | PROJ-555 | Bridge to architect LLM program (PROJ-405) to prevent duplication. |

Older directories `project-01-cicd-pipeline` … `project-05-governance` contain pre-migration drafts; archive references but prioritize the newer `project-1` … `project-5` artifacts completed in 2025-11.

## Solution Assets
- High-quality implementation for `project-1-ml-pipeline` (~9,500 LOC) must be linked and documented.
- Placeholder or draft specs exist for projects 2–5; capture status/TODOs in solution metadata and READMEs.
- Solutions remain in separate repos per org policy; ensure repository strategy reflects per-role ownership while supporting consolidated option per latest requirements.

## Cross-Role Alignment Goals
- Reference junior and senior engineer modules for baseline knowledge to avoid re-teaching fundamentals.
- Leverage ML Platform Engineer assets (feature store, workflow orchestration, observability) for modules 554–558.
- Reuse security governance controls from AI Infrastructure Security Engineer for modules 557–559.
- Tie advanced topics (MOD-560) to architect/principal tracks for portfolio-level continuity and reduced duplication.

## Outstanding Questions
1. Confirm numbering scheme (MOD-551…560 / PROJ-551…555) works with exporters/mapping scripts.
2. Determine validation profiles for code-heavy modules (likely `python-strict` for automation-focused lessons).
3. Decide whether experimentation automation scripts from `project-03-validation` drafts should be archived or integrated as supplemental resources.

Convert these notes into full research briefs, master plans, module/project metadata, and solution documentation during migration.
