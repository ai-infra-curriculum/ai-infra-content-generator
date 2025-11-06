# Project Plan

> Migrated from legacy AI Infrastructure Engineer Project 2 to solidify automated ML lifecycle operations.

## Project Overview

- **Project ID**: PROJ-202
- **Project Title**: End-to-End MLOps Pipeline
- **Target Role(s)**: AI Infrastructure Engineer
- **Placement in Curriculum**: Culmination of Modules MOD-105 (Data Pipelines) and MOD-106 (MLOps)
- **Estimated Duration**: 40 hours
- **Prerequisite Modules / Skills**: MOD-105, MOD-106, MOD-102 (cloud provisioning)
- **Related Assessments**: Pipeline design review, governance checklist, automated test suite report

## Learning Objectives

- Design and implement an automated ML pipeline covering ingestion, training, validation, and deployment.
- Operate experiment tracking, model registry, and feature store integrations with governance artifacts.
- Automate CI/CD workflows that gate deployments on quality signals and compliance checks.
- Extend observability to cover pipeline health, data quality, and model performance post-deployment.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| MLOps automation | Proficient | Pipeline run logs, CI/CD workflow execution | AI Infrastructure Engineer |
| Data pipeline engineering | Proficient | Airflow DAG + monitoring report | AI Infrastructure Engineer |
| Governance & compliance | Working | Approval workflow, audit-ready documentation | AI Infrastructure Engineer |

## Project Narrative

The data science team needs a repeatable way to train and deploy models for a churn prediction product. As the infrastructure engineer you will:

1. Build an Airflow DAG orchestrating data ingestion, feature engineering, training, evaluation, and deployment.
2. Integrate MLflow (or equivalent) for experiment tracking, model registry, and promotion approvals.
3. Wire CI/CD workflows to validate code, enforce policy-as-code, and trigger deployments to staging/production.
4. Instrument data validation (Great Expectations) and model drift monitoring with automated notifications.
5. Produce governance artifacts including approval checklists, rollback procedures, and compliance evidence.

## Deliverables

- Airflow DAGs, configuration, and supporting scripts checked into source control.
- MLflow/registry configuration, promotion workflow documentation, and environment configuration files.
- CI/CD pipeline definition (GitHub Actions or similar) covering tests, policy checks, deployment automation.
- Monitoring dashboards for pipeline runtime, data quality metrics, and model performance.
- Governance packet: approval workflow, incident response plan, compliance checklist, final retrospective.

## Constraints & Assumptions

- Pipeline must run on managed cloud services or containerized Airflow; all infrastructure defined as code.
- Secrets handled via secret manager integration (Vault, AWS Secrets Manager, etc.).
- Promotion requires automated quality gates (tests, metrics) plus manual approval for production.
- Observability must integrate with program-wide monitoring stack (Prometheus/Grafana).

## Learner Experience

| Phase | Description | Estimated Time | Instructor Touchpoints |
|-------|-------------|----------------|------------------------|
| Design | Pipeline architecture, tool selection, backlog refinement | 6h | Design review workshop |
| Implement | Build orchestration, automation, observability, governance assets | 28h | Weekly stand-up + async reviews |
| Validate | Run dry-run deployment, finalize documentation, handoff | 6h | Final audit walkthrough |

## Solutions & Validation

- **Solutions Path**: `projects/ai-infrastructure-engineer/project-102-mlops-pipeline/solutions`
- **Validation Profiles**: `python-strict`, Airflow DAG unit tests, MLflow integration checks
- **Automation Hooks**: See `.github/workflows/pipeline-ci.yml` and `Makefile` in the solutions repository for reproducible validation

## Risks & Mitigations

- **Pipeline brittleness**: Require chaos experiment (task failure injection) before final approval.
- **Governance gaps**: Provide compliance checklist; instructor validates artifacts prior to sign-off.
- **Tooling fatigue**: Offer reference architecture and pre-approved stack to reduce decision overhead.

## References & Inspiration

- Legacy project docs at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/projects/project-102-mlops-pipeline`
- MLOps best practices from Google/Azure architecture guides
- Great Expectations and MLflow operator documentation

## Sign-off

- **Author**: Curriculum Migration Team
- **Reviewer(s)**: MLOps Lead (pending)
- **Date Approved**: Pending validation run
