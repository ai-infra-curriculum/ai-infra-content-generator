# Module Roadmap

> AI Infrastructure MLOps Engineer | Module 556 modernizes `lessons/06-automation` with shared orchestration assets.

## Module Overview

- **Module ID**: MOD-556
- **Module Title**: Automation & Orchestration at Scale
- **Target Role(s)**: AI Infrastructure MLOps Engineer
- **Duration**: 40 hours (12 lecture, 18 lab, 6 portfolio, 4 assessment)
- **Prerequisites**: MOD-552, MOD-555
- **Next Module(s)**: MOD-557, MOD-558, PROJ-552, PROJ-554

## Cross-Role Progression

- Extends senior orchestration (MOD-206) while reusing ML Platform workflow components (MOD-505).
- Coordinates with security pipelines to embed policy gates without duplicating logic.
- Feeds automation assets into architect enterprise programs (PROJ-401) for consistency.

## Learning Objectives

- Build resilient orchestration pipelines using Airflow, Kubeflow, Prefect, and Argo Workflows.
- Implement reusable components, dependency management, and failure handling patterns.
- Integrate compliance checks, approvals, and rollout strategies into automation.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| workflow-orchestration | Expert | DAG/flow implementation review | AI Infrastructure MLOps Engineer |
| automation | Expert | Reliability & policy gate demo | AI Infrastructure MLOps Engineer |

## Content Outline

1. **Orchestration Landscape** – tools, control/data plane separation, scaling considerations.
2. **Reusable Components** – templated tasks, data dependencies, artifact passing.
3. **Reliability Enhancements** – retries, compensation logic, circuit breakers.
4. **Policy Gates** – integration with governance/security modules.
5. **Deployment Strategies** – multi-environment rollout, blue/green, canary.

## Hands-On Activities

- Implement shared DAG templates referencing ML Platform assets to avoid duplication.
- Add policy-as-code checks and fallback logic to orchestration flows.
- Run chaos experiments on pipelines and document mitigation strategies.

## Assessments & Evidence

- Peer-reviewed orchestration design walkthrough.
- Automation resilience report including chaos test results and governance integration.

## Shared Assets & Legacy Mapping

- Aligns with ML Platform workflow modernization to keep automation primitives unified.
- Produces components consumed by compliance pipeline project (PROJ-554).
