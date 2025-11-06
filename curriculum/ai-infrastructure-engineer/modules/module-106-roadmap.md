# Module Roadmap

> Captures the legacy MLOps module with updated objectives, artifacts, and validation requirements.

## Module Overview

- **Module ID**: MOD-106
- **Module Title**: MLOps & Experiment Tracking
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 40 hours (14 lecture, 18 lab, 6 project, 2 assessment)
- **Prerequisites**: MOD-101, MOD-105
- **Next Module(s)**: MOD-110, PROJ-202

## Cross-Role Progression

- **Builds On** (modules/roles): Data pipeline engineering (MOD-105) and junior automation foundations
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): MLflow quick-start from junior capstone reused for baseline workshop
- **Differentiators** (role-specific emphasis): Enterprise model governance, automated deployment, experiment lifecycle

## Learning Objectives

- Stand up MLflow (or equivalent) for experiment tracking, model registry, and artifact management.
- Design CI/CD pipelines for ML model deployment with testing, approvals, and rollbacks.
- Implement automated model validation, including data drift and performance checks.
- Integrate feature stores or metadata catalogs to support reproducible ML workflows.
- Produce governance artifacts (runbooks, compliance reports) for production ML systems.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| MLOps lifecycle | Proficient | End-to-end pipeline demonstration | AI Infrastructure Engineer |
| Automation & tooling | Proficient | CI/CD workflow with automated testing | AI Infrastructure Engineer |
| Governance & compliance | Working | Model approval documentation | AI Infrastructure Engineer |

## Content Outline

1. **MLOps Principles** – maturity models, roles, and production success metrics.
2. **Experiment Tracking** – MLflow setup, tracking APIs, artifact storage integration.
3. **Model Registry & Promotion** – lifecycle states, approval workflows, rollback strategies.
4. **CI/CD for ML** – GitHub Actions pipeline, automated testing, canary deployment triggers.
5. **Validation & Monitoring** – automated tests, data quality, drift detection, alerts.
6. **Feature & Metadata Stores** – design considerations, integration with pipelines.
7. **Governance & Documentation** – compliance artifacts, audit trail, communication templates.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Deploy MLflow tracking & registry | Authenticated access configured; S3/GCS backend attached | Infrastructure validation script |
| Lab 2 | CI/CD pipeline for model promotion | Pipeline executes tests, approvals, rollout/rollback | GitHub Actions dry run |
| Lab 3 | Automated validation suite | Drift detection implemented; alerts routed | Validation test suite |
| Assessment | MLOps architecture quiz | ≥80% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: MLflow deployment scripts, CI/CD workflows, validation notebooks, quiz key
- **Repository Strategy**: `per_role` with separate solutions repo for sensitive configs
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-106-mlops/solutions`
- **Validation Status**: Pending reconfiguration of secrets for CI and validation harness run

## Resource Plan

- **Primary References**:
  - Module README in `modules/ai-infrastructure-engineer/module-106-mlops`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-106-mlops`
- **Supplemental Resources**:
  - MLflow documentation, Kubeflow Pipelines references
  - MLOps maturity model whitepapers
- **Tooling Requirements**:
  - MLflow 2.x, GitHub Actions or equivalent CI, cloud storage bucket, feature store (Feast or similar)

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Airflow pipelines (MOD-105), foundational deployment skills (MOD-101)
- **Downstream Outputs**: Essential for Projects PROJ-202 and PROJ-203, plus MOD-110
- **Risks / Mitigations**: Secret management complexity—provide templated `.env` and secret rotation guidance.
