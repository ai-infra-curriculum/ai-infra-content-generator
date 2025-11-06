# Module Roadmap

> Updated from the legacy monitoring track to emphasize ML observability and cross-role reuse.

## Module Overview

- **Module ID**: MOD-108
- **Module Title**: Monitoring & Observability for ML Systems
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 40 hours (14 lecture, 18 lab, 6 project, 2 assessment)
- **Prerequisites**: MOD-103, MOD-104, MOD-009
- **Next Module(s)**: MOD-109, MOD-110, PROJ-203

## Cross-Role Progression

- **Builds On** (modules/roles): Junior monitoring basics (MOD-009) and Kubernetes operations (MOD-104)
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Baseline dashboard templates reused for junior role retrospectives
- **Differentiators** (role-specific emphasis): ML-specific metrics, alert automation, incident response leadership

## Learning Objectives

- Design and deploy observability stacks (Prometheus, Grafana, Alertmanager) for ML infrastructure.
- Instrument ML services with custom metrics, traces, and structured logs.
- Implement alerting strategies that account for model and infra health indicators.
- Build runbooks and SLO dashboards to support on-call rotations.
- Integrate logging and tracing pipelines (ELK/EFK, OpenTelemetry) for ML workloads.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Observability engineering | Proficient | Dashboard + alert review | AI Infrastructure Engineer |
| Incident response | Working | Runbook walkthrough and tabletop | AI Infrastructure Engineer |
| Automation & tooling | Working | Alert automation scripts | AI Infrastructure Engineer |

## Content Outline

1. **Observability Principles** – metrics, logs, traces, ML-specific KPIs.
2. **Metrics Pipeline** – Prometheus architecture, exporters, recording rules.
3. **Visualization & Dashboards** – Grafana best practices, SLO/SLA visualization.
4. **Alerting & Incident Flow** – Alertmanager routing, noise reduction, escalation policies.
5. **Logging & Tracing** – EFK stack, OpenTelemetry, correlation between requests and model predictions.
6. **ML Monitoring** – data drift, model performance metrics, business KPI instrumentation.
7. **Reliability Operations** – incident runbooks, postmortems, error budgets, on-call readiness.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Deploy Prometheus + Grafana stack | Dashboards published; recording rules committed | Observability validation script |
| Lab 2 | Instrument ML service | Custom metrics (latency, accuracy, drift) exposed | Metrics unit tests |
| Lab 3 | Alert & incident simulation | Pager workflow tested; postmortem drafted | Tabletop exercise rubric |
| Assessment | Observability quiz | ≥80% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Helm charts/manifests, dashboard exports, alert routing config, runbook templates, quiz key
- **Repository Strategy**: `per_role` with separate solutions repo due to sensitive configs
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-108-monitoring-observability/solutions`
- **Validation Status**: Pending integration test running kube-prometheus-stack in CI

## Resource Plan

- **Primary References**:
  - Module README in `modules/ai-infrastructure-engineer/module-108-monitoring-observability`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-108-monitoring-observability`
- **Supplemental Resources**:
  - Prometheus & Grafana documentation, Google SRE workbook
  - OpenTelemetry collector guides
- **Tooling Requirements**:
  - Kubernetes cluster, Helm, Prometheus stack, Loki/ELK (optional), incident management tool

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Junior observability basics (MOD-009), Kubernetes module (MOD-104)
- **Downstream Outputs**: Supports PROJ-201/202/203 reliability requirements
- **Risks / Mitigations**: Tooling sprawl—standardize on Prometheus stack for exercises; provide optional OpenTelemetry extension.
