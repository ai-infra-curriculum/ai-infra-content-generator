# Module Roadmap

> AI Infrastructure Architect | Module 305 advances high-availability and disaster recovery design for enterprise AI workloads.

## Module Overview

- **Module ID**: MOD-305
- **Module Title**: High Availability & Disaster Recovery
- **Target Role(s)**: AI Infrastructure Architect
- **Duration**: 50 hours (16 lecture, 20 lab, 10 project, 4 assessment)
- **Prerequisites**: MOD-205, MOD-302, MOD-304
- **Next Module(s)**: PROJ-402, PROJ-403

## Cross-Role Progression

- **Builds On**: Senior resiliency (MOD-205) and architect multi-cloud strategy (MOD-302) plus FinOps constraints (MOD-304).
- **Adds New Depth**: Active-active designs, chaos engineering programs, sustainability-aware resilience.
- **Shared Assets**: Shares DR automation scripts with PROJ-402, uses cost models from MOD-304.
- **Differentiators**: Emphasizes measurable reliability targets, cross-region DR, and executive readiness.

## Learning Objectives

- Architect ML/LLM platforms achieving ≥99.95% availability with automated failover.
- Implement DR plans, testing cadence, and chaos experiments across multi-cloud environments.
- Produce executive reliability dashboards and action plans balancing cost and resilience.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| resilience-architecture | Expert | Reliability architecture + test evidence | AI Infrastructure Architect |
| chaos-engineering | Proficient | Chaos program charter & drill report | AI Infrastructure Architect |

## Content Outline

1. **Reliability Goals** – SLOs, error budgets, business continuity requirements.
2. **Architectural Patterns** – active-active/active-passive, cell-based, regional isolation.
3. **Automation** – IaC-driven DR, GitOps promotion, automated failover workflows.
4. **Chaos & Testing** – chaos experiments, tabletop exercises, DR rehearsals, observability integration.
5. **Reporting & Governance** – reliability scorecards, KPIs, stakeholder communication.
6. **Cost & Sustainability** – balance redundancy with budget/sustainability targets.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Reliability architecture pack | Includes architecture diagrams, SLOs, risk register | Architecture review |
| Lab 2 | Chaos experiment suite | Chaos test executed with documented learnings | Chaos report |
| Lab 3 | DR automation playbook | Automated failover demo with <60s RTO | DR drill validation |
| Assessment | Executive briefing | ≥80% on reliability briefing rubric | Exec-style panel |

## Solutions Plan

- **Coverage**: Reliability templates, chaos scripts, DR automation runbooks, KPI dashboards.
- **Repository Strategy**: Resources located under `solutions/ai-infrastructure-architect/projects/project-402-multicloud-infrastructure`.
- **Solution Path**: `modules/ai-infrastructure-architect/module-305-ha-dr/solutions`.
- **Validation Status**: Requires DR/chaos simulation; integrate with validation backlog.

## Resource Plan

- **Primary References**: Module README, legacy HA/DR documentation.
- **Supplemental Resources**: Google SRE workbook, AWS/GCP/Azure DR guides, Gremlin chaos engineering playbooks.
- **Tooling Requirements**: IaC tooling, chaos platform (Chaos Mesh/Litmus/Gremlin), observability stack, communication tools.

## Quality Checklist

- [ ] Architecture includes RTO/RPO, dependency mapping, and cost analysis.
- [ ] Chaos program defines scope, schedule, and improvement backlog.
- [ ] DR drill evidence stored in validation repository.
- [ ] Deliverables inform PROJ-402 readiness review.

## Dependencies & Notes

- Align with security/compliance (MOD-303) to ensure DR plans satisfy regulatory requirements.
