# Project Plan

> Migrated from the legacy AI Infrastructure Engineer Project 1 to capture scope, sequencing, and validation in the new framework.

## Project Overview

- **Project ID**: PROJ-201
- **Project Title**: Basic Model Serving System
- **Target Role(s)**: AI Infrastructure Engineer
- **Placement in Curriculum**: Capstone for Modules MOD-101, MOD-103, MOD-104
- **Estimated Duration**: 30 hours
- **Prerequisite Modules / Skills**: MOD-101 Foundations, MOD-103 Containerization, MOD-104 Kubernetes
- **Related Assessments**: Module quizzes (MOD-101/103/104) and observability spot-check

## Learning Objectives

- Deploy a production-ready ML inference service behind a FastAPI/TorchServe endpoint.
- Containerize and publish the service using best-practice Docker and Helm workflows.
- Operate the service on Kubernetes with autoscaling, health checks, and rolling upgrades.
- Establish CI/CD pipelines, monitoring dashboards, and on-call runbooks for the service.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Infrastructure automation | Proficient | Deployment automation review, CI/CD pipeline execution | AI Infrastructure Engineer |
| Observability engineering | Working | Prometheus/Grafana dashboards, alert routing | AI Infrastructure Engineer |
| Reliability engineering | Working | Runbooks, incident rehearsal summary | AI Infrastructure Engineer |

## Project Narrative

You inherit a prototype image-classification model built by the data science team. It runs locally and lacks production readiness. Your task is to deliver a fully operational service:

1. Package the inference logic using FastAPI and TorchServe (or equivalent).
2. Containerize the service with a multi-stage Dockerfile that keeps runtime images lean.
3. Deploy to Kubernetes with Helm, enabling autoscaling and zero-downtime rollouts.
4. Implement monitoring (Prometheus, Grafana) and alerting tied to service SLIs.
5. Ship CI/CD workflows that validate builds, run tests, and promote to staging/production environments.

Stakeholders expect weekly demos plus a final deployment review showcasing resilience and observability.

## Deliverables

- Source code with modular service implementation and tests.
- Dockerfile, Helm chart, and Kubernetes manifests promoting dev → staging → prod.
- GitHub Actions (or equivalent) pipeline covering build, test, security scan, and deployment.
- Prometheus alert rules, Grafana dashboards, and operational runbooks.
- Final deployment report including architecture diagram, cost estimate, and post-deployment checklist.

## Constraints & Assumptions

- Use FastAPI + TorchServe (or similar) as baseline stack; substitutions documented and approved.
- Target deployment on managed Kubernetes (EKS/GKE/AKS) or local kind cluster with documented trade-offs.
- Monitoring stack must expose latency, error rate, throughput, and GPU utilization (if applicable).
- Secrets handled via Kubernetes Secrets or vault integration—no plaintext credentials in repo.

## Learner Experience

| Phase | Description | Estimated Time | Instructor Touchpoints |
|-------|-------------|----------------|------------------------|
| Discover | Requirements review, architecture design, backlog creation | 4h | Architecture working session |
| Build | Implement service, containerization, deployment, and monitoring | 22h | Weekly async feedback + office hours |
| Validate | Run CI/CD, observability drills, finalize documentation | 4h | Final demo & ops review |

## Solutions & Validation

- **Solutions Path**: `projects/ai-infrastructure-engineer/project-101-basic-model-serving/solutions`
- **Validation Profiles**: `python-strict`, container security scan, deployment smoke tests
- **Automation Hooks**: See `.github/workflows/deploy.yml` and `Makefile` in the solutions path for runnable checks

## Risks & Mitigations

- **Unclear service boundaries**: Provide architecture canvas template and require sign-off before build sprint begins.
- **Operational blind spots**: Enforce observability acceptance criteria (dashboards + alerts) prior to final sign-off.
- **Cost overrun**: Include cost estimation worksheet; add guardrails via resource limits and autoscaling policies.

## References & Inspiration

- Legacy project docs at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/projects/project-101-basic-model-serving`
- CNCF production readiness checklists
- Google SRE workbook (service level indicators and objectives)

## Sign-off

- **Author**: Curriculum Migration Team
- **Reviewer(s)**: Platform Lead (pending)
- **Date Approved**: Pending validation run
