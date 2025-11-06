# Module Roadmap

> AI Infrastructure MLOps Engineer | Module 553 modernizes `lessons/03-model-monitoring` to align with shared observability tooling.

## Module Overview

- **Module ID**: MOD-553
- **Module Title**: Model Monitoring & Drift Response
- **Target Role(s)**: AI Infrastructure MLOps Engineer
- **Duration**: 38 hours (12 lecture, 16 lab, 6 portfolio, 4 assessment)
- **Prerequisites**: MOD-551, MOD-207
- **Next Module(s)**: MOD-558, PROJ-552

## Cross-Role Progression

- Extends senior engineer observability (MOD-207) with model-specific metrics and drift pipelines.
- Shares dashboards and cost analytics with ML Platform (MOD-508) to minimize duplication.
- Coordinates with security (MOD-909) for anomaly detection and incident escalation.

## Learning Objectives

- Instrument inference and training pipelines with model health metrics, business KPIs, and data drift signals.
- Automate alerting, incident response, and retraining triggers linked to GitOps workflows.
- Integrate FinOps and cost tracking to control LLM/compute spend.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| model-observability | Proficient | Monitoring dashboard & alert playbook | AI Infrastructure MLOps Engineer |
| service-reliability | Proficient | Incident response tabletop | AI Infrastructure MLOps Engineer |

## Content Outline

1. **Observability Foundations** – SLIs/SLOs, telemetry, and model KPIs.
2. **Drift Detection Patterns** – statistical tests, supervised/unsupervised signals.
3. **Incident Response** – on-call rotations, runbooks, escalation paths.
4. **Retraining Automation** – pipeline triggers, approvals, rollback.
5. **FinOps Tie-In** – cost dashboards, capacity planning, budgeting.

## Hands-On Activities

- Configure Prometheus/Grafana dashboards using shared templates.
- Implement Evidently/Arize drift monitors with automated actions.
- Conduct an incident tabletop using shared runbooks from senior engineer track.

## Assessments & Evidence

- Monitoring implementation review with SRE stakeholders.
- Incident postmortem and remediation plan aligned with security requirements.

## Shared Assets & Legacy Mapping

- Reuses dashboards from ML Platform and security roles, customizing metrics for MLOps.
- Provides signals consumed by governance module (MOD-557) for compliance evidence.
