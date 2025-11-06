# Module Roadmap

> AI Infrastructure Architect | Module 302 modernizes multi-cloud/hybrid architecture practices for AI platforms.

## Module Overview

- **Module ID**: MOD-302
- **Module Title**: Multi-Cloud & Hybrid Architecture
- **Target Role(s)**: AI Infrastructure Architect
- **Duration**: 60 hours (18 lecture, 26 lab, 12 project, 4 assessment)
- **Prerequisites**: MOD-205, MOD-208, MOD-301
- **Next Module(s)**: MOD-303, MOD-304, MOD-305

## Cross-Role Progression

- **Builds On**: Senior multi-cloud resiliency (MOD-205) and IaC/GitOps capabilities (MOD-208).
- **Adds New Depth**: Vendor-neutral design, sovereignty controls, active-active patterns, migration strategy.
- **Shared Assets**: Reuses GitOps and policy stacks from senior role; expands FinOps dashboards.
- **Differentiators**: Introduces executive-level cost modeling, regulatory constraints, and hybrid connectivity patterns.

## Learning Objectives

- Architect multi-cloud/hybrid AI platforms with consistent identity, networking, and data pipelines.
- Design active-active and DR topologies with measurable RTO/RPO, automated failover, and FinOps guardrails.
- Produce migration plans, vendor evaluations, and governance artifacts supporting enterprise rollout.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| multi-cloud-strategy | Expert | Reference architecture + migration plan | AI Infrastructure Architect |
| resilience-architecture | Proficient | DR simulation report + metrics | AI Infrastructure Architect |

## Content Outline

1. **Strategy & Assessment** – vendor comparison, cloud adoption frameworks, capability gaps.
2. **Identity & Networking** – multi-cloud IAM, private connectivity, service mesh federation.
3. **Data & Sovereignty** – residency, replication, encryption, compliance boundaries.
4. **Deployment Patterns** – active-active, blue/green DR, hub-and-spoke, cloud bursting.
5. **Automation & Governance** – IaC module design, GitOps for environments, policy-as-code.
6. **FinOps & Performance** – placement strategies, cost modeling, PMO integration.
7. **Migration & Change Management** – phased rollout, communication plans, risk mitigation.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Multi-cloud landing zone | Connectivity + IAM config validated across providers | Security review |
| Lab 2 | Active-active reference architecture | Architecture pack with RTO/RPO, FinOps metrics | Architecture review |
| Lab 3 | Migration & change plan | Stakeholder-approved roadmap and risk register | Executive simulation |
| Assessment | Scenario defense | Panel evaluates architecture trade-offs ≥80% | Panel review |

## Solutions Plan

- **Coverage**: Landing zone templates, network architecture diagrams, migration playbooks, financial models.
- **Repository Strategy**: Stored under `frameworks/` and `projects/` in `solutions/ai-infrastructure-architect`.
- **Solution Path**: `modules/ai-infrastructure-architect/module-302-multicloud-hybrid/solutions`.
- **Validation Status**: Requires tabletop simulation rather than automated checks.

## Resource Plan

- **Primary References**: Module README and legacy lesson artifacts.
- **Supplemental Resources**: AWS/GCP/Azure multi-cloud guides, Anthos/Azure Arc references, CNCF multi-cluster patterns.
- **Tooling Requirements**: Terraform/Pulumi, Crossplane, ArgoCD/Flux, networking design tools, cost modeling spreadsheets.

## Quality Checklist

- [ ] Architecture addresses sovereignty, identity, network, data, and FinOps dimensions.
- [ ] Migration plan includes communications, milestones, and risk mitigation.
- [ ] DR/HA documentation references chaos/DR drills planned in MOD-305 and PROJ-402.
- [ ] Executive summary aligns with business objectives and ROI targets.

## Dependencies & Notes

- Coordinate with MOD-304 FinOps labs to reuse dashboards and chargeback models for cost tracking.
