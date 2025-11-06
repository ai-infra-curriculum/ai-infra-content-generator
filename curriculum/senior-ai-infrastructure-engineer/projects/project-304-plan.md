# Project Plan

> Senior AI Infrastructure Engineer | Migrated from legacy Project 4 (Custom Kubernetes Operator) with modern GitOps integration.

## Project Overview

- **Project ID**: PROJ-304
- **Project Title**: Custom Kubernetes Operator Suite
- **Target Role(s)**: Senior AI Infrastructure Engineer
- **Placement in Curriculum**: Capstone integrating MOD-201, MOD-206, MOD-208, MOD-210.
- **Estimated Duration**: 65 hours
- **Prerequisite Modules / Skills**: MOD-201 Advanced Kubernetes, MOD-206 Advanced MLOps, MOD-208 IaC & GitOps, MOD-210 Leadership
- **Related Assessments**: Operator design review, policy compliance audit, enablement workshop

## Learning Objectives

- Build production-ready Kubernetes operators/controllers to manage ML workloads end-to-end.
- Integrate operators with GitOps, policy-as-code, and platform APIs to deliver self-service capabilities.
- Enable downstream teams through documentation, demos, and adoption metrics.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| kubernetes-platform-engineering | Expert | Operator code review & functional demo | Senior AI Infrastructure Engineer |
| platform-engineering | Expert | Platform API + GitOps integration | Senior AI Infrastructure Engineer |
| mentorship | Expert | Enablement workshop & feedback | Senior AI Infrastructure Engineer |

## Project Narrative

The organization needs a standardized way to provision ML training, serving, and monitoring workloads via GitOps. Engineers spend weeks configuring each project manually. You will:

1. Define operator scope (CRDs, reconciliation logic) to manage lifecycle of ML workloads (training jobs, inference services, observability stacks).
2. Implement operators/controllers in Go/Python with comprehensive testing, metrics, and alerting.
3. Integrate operators with GitOps workflows, policy enforcement, and secrets automation.
4. Produce platform documentation, SDK/CLI integrations, and run enablement sessions for downstream teams.
5. Capture adoption metrics and feedback to drive continuous improvement roadmap.

## Deliverables

- Operator source code, CRDs, manifests, and automated tests.
- GitOps integration (ArgoCD/Flux configurations), policy-as-code suites, secret management pipelines.
- Observability dashboards and alerting for operator health.
- Enablement assets: documentation portal, tutorial recordings, Q&A logs.
- Adoption report: usage metrics, feedback survey, prioritized improvement backlog.

## Constraints & Assumptions

- Operators must support multi-tenant isolation, approval workflows, and rollback.
- GitOps pipeline must include signed releases, policy checks, and multi-environment promotion.
- Documentation must align with organizational templates and include onboarding journey.
- Success measured by pilot adoption and feedback from target teams.

## Learner Experience

| Phase | Description | Estimated Time | Instructor Touchpoints |
|-------|-------------|----------------|------------------------|
| Define | Scope CRDs, roadmap, success metrics, design review | 10h | Architecture council |
| Build | Implement operators, automation, documentation | 45h | Weekly engineering review |
| Enable | Run pilot rollout, collect feedback, finalize adoption report | 10h | Enablement workshop |

## Solutions & Validation

- **Solutions Path**: `projects/senior-ai-infrastructure-engineer/project-304-k8s-operator/solutions`
- **Validation Profiles**: `python-strict`, operator unit/integration tests, policy compliance checks
- **Automation Hooks**: `.github/workflows/operator-ci.yml`, operator Makefile targets

## Risks & Mitigations

- **Operator complexity**: Limit MVP scope, iterate with stakeholders, implement feature flags.
- **Adoption resistance**: Run enablement workshops, gather feedback, integrate requested features quickly.
- **Policy violations**: Embed policy-as-code from MOD-208/209; include compliance officer in review.

## References & Inspiration

- Legacy assets: `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-senior-engineer-learning/projects/project-204-k8s-operator`
- CNCF operator pattern references, Kubebuilder docs, Openshift/Weaveworks GitOps operator case studies

## Sign-off

- **Author**: Curriculum Migration Team
- **Reviewer(s)**: Staff Platform Engineer (pending)
- **Date Approved**: Pending validation run
