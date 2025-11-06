# Module Roadmap

> Updated with concrete objectives and activities sourced from the legacy AI Infrastructure Engineer curriculum.

## Module Overview

- **Module ID**: MOD-102
- **Module Title**: Cloud Computing for ML
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 50 hours (18 lecture, 22 lab, 8 project, 2 assessment)
- **Prerequisites**: MOD-101, MOD-010
- **Next Module(s)**: MOD-103, MOD-105

## Cross-Role Progression

- **Builds On** (modules/roles): Junior MOD-010 Cloud Platforms and MOD-001 Python Fundamentals
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Budget and access policy templates seeded from junior curriculum
- **Differentiators** (role-specific emphasis): Multi-cloud decision frameworks, IAM governance, cost optimization simulations

## Learning Objectives

- Provision GPU-enabled compute, storage, and networking workloads for ML on major cloud providers.
- Configure IAM policies, service principals, and secrets storage that comply with least-privilege principles.
- Design resilient VPC/network topologies that support Kubernetes clusters and data pipelines.
- Implement cost monitoring, tagging, and alerting systems tailored for ML infrastructure.
- Deploy a secure end-to-end ML workload encompassing ingestion, training, and serving.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Cloud architecture | Proficient | Terraform/IaC deployment review | AI Infrastructure Engineer |
| Security & compliance | Proficient | IAM and secrets management workshop | AI Infrastructure Engineer |
| Cost management | Working | Cloud spend dashboard and mitigation plan | AI Infrastructure Engineer |

## Content Outline

1. **Cloud Fundamentals Refresh** – service models, regional design, AI workload patterns.
2. **IAM & Secrets Management** – roles, policies, federated identities, vault integrations.
3. **Core Services for ML** – compute SKUs, storage tiers, managed ML services, GPU provisioning.
4. **Networking & Security** – VPC design, ingress/egress controls, service mesh integration.
5. **Automation & Provisioning** – Terraform modules, CLI/SDK workflows, tagging strategies.
6. **Cost Governance** – alerting, anomaly detection, FinOps playbooks for ML workloads.
7. **Operational Readiness** – logging pipelines, incident response hooks, backup/DR considerations.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Provision GPU-ready cloud environment with IaC | Terraform plan/apply output stored; teardown verified | Terraform fmt/validate |
| Lab 2 | Implement IAM and secrets workflow | Policies peer-reviewed; rotation procedure documented | Security checklist review |
| Lab 3 | Build cost monitoring dashboard | Budget alerts active; monthly spend report generated | Cost report peer review |
| Assessment | Cloud architecture quiz | ≥75% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Terraform module set, IAM policy samples, budget calculator notebook, quiz key
- **Repository Strategy**: `per_role` (see `curriculum/repository-strategy.yaml`)
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-102-cloud-computing/solutions`
- **Validation Status**: Pending execution of provisioning tests inside updated CI harness

## Resource Plan

- **Primary References**:
  - `modules/ai-infrastructure-engineer/module-102-cloud-computing/README.md`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-102-cloud-computing`
- **Supplemental Resources**:
  - AWS/GCP/Azure architecture blueprints for ML infrastructure
  - CNCF FinOps and cost management whitepapers
- **Tooling Requirements**:
  - Terraform ≥1.6, cloud CLIs/SDKs, sandbox cloud accounts with GPU quota

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Junior cloud (MOD-010) and environment readiness from MOD-101
- **Downstream Outputs**: Feed infrastructure for Projects PROJ-201 (serving) and PROJ-202 (pipeline)
- **Risks / Mitigations**: Cloud cost overruns—require spend simulation deliverable before project approval.
