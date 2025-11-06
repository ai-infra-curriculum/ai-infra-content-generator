# Lecture 01 · Orchestration Patterns

## Objectives
- Evaluate orchestration frameworks and choose components that maximize reuse across teams.
- Design control and data plane separation for reliability and scale.
- Standardize artifact passing, dependency management, and multi-environment deployment.

## Key Topics
1. **Framework Comparison** — Airflow vs Kubeflow vs Prefect vs Argo; when to unify.
2. **Component Design** — task templates, sensors, event-driven triggers, reusable libraries.
3. **Dependency Strategies** — fan-in/out, conditional flows, parallelization, retries.
4. **Environment Promotion** — staging, canary, blue/green flows leveraging GitOps.
5. **Observability Hooks** — logging, tracing, metrics, correlation with ML monitoring.

## Activities
- Map legacy orchestration scripts to reusable components.
- Implement a sample DAG referencing shared ML Platform assets.
- Document deployment strategy feeding into production ops module MOD-558.
