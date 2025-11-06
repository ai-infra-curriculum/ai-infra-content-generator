# Module Roadmap

> Senior AI Infrastructure Engineer | Module 208 advances IaC and GitOps governance practices.

## Module Overview

- **Module ID**: MOD-208
- **Module Title**: Advanced Infrastructure as Code & GitOps
- **Target Role(s)**: Senior AI Infrastructure Engineer
- **Duration**: 40 hours (14 lecture, 18 lab, 6 project, 2 assessment)
- **Prerequisites**: MOD-109, MOD-206
- **Next Module(s)**: MOD-209, PROJ-304

## Cross-Role Progression

- **Builds On**: Mid-level IaC fundamentals and advanced MLOps platform automation.
- **Adds New Depth**: Modular IaC frameworks, policy-as-code, secure GitOps, testing automation.
- **Shared Assets**: Integrates with security checks in MOD-209 and operator deliverables in PROJ-304.
- **Differentiators**: Enterprise governance and automation of multi-environment promotion.

## Learning Objectives

- Design reusable Terraform/Pulumi modules with automated testing, versioning, and documentation standards.
- Implement GitOps workflows (ArgoCD/Flux) with policy enforcement (OPA/Kyverno, Conftest) and secrets automation.
- Establish compliance pipelines covering drift detection, change audits, and code signing.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| iac-enterprise | Expert | Module registry + test harness | Senior AI Infrastructure Engineer |
| gitops-automation | Expert | GitOps pipeline demo with policy gating | Senior AI Infrastructure Engineer |

## Content Outline

1. IaC architecture patterns (module registries, platform APIs, multi-account strategy).
2. Testing automation (Terratest, InSpec, policy unit tests, drift pipelines).
3. GitOps at scale (ArgoCD App-of-Apps, multi-tenant management, change approval).
4. Security/compliance (image/signature verification, secret management, audit logs).
5. Documentation & onboarding (module READMEs, changelog automation, support models).

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Terraform/Pulumi module | Published to registry with tests & docs | Test harness + doc review |
| Lab 2 | GitOps pipeline | Promotion flow with policy gating and drift detection | CI validation |
| Lab 3 | Compliance automation | Change audit trail + SBOM/signing integrated | Compliance review |
| Assessment | Governance presentation | Approved by platform steering committee | Presentation rubric |

## Solutions Plan

- **Coverage**: Sample modules, test harnesses, GitOps pipelines, policy bundles, presentation deck.
- **Repository Strategy**: Solutions tracked in separate repo; metadata stored locally.
- **Solution Path**: `modules/senior-ai-infrastructure-engineer/module-208-iac-gitops/solutions`.
- **Validation Status**: Pending integration of Terratest suite in CI and policy checks in automation workflows.

## Resource Plan

- **Primary References**: Module README, templates, and scripts in module directory.
- **Supplemental Resources**: ArgoCD/Flux operator docs, Terraform module design guides, OPA/Kyverno best practices.
- **Tooling Requirements**: Terraform/Pulumi, Terratest (Go), Conftest/OPA, ArgoCD/Flux, Vault/Secrets Manager, Cosign/Sigstore.

## Quality Checklist

- [ ] Module registry entry includes API docs, versioning strategy, and support contact.
- [ ] GitOps pipelines enforce signed commits/artifacts and produce audit logs.
- [ ] Policy tests integrated into CI gating; failure scenarios documented.
- [ ] Deliverables align with PROJ-304 operator automation plans.

## Dependencies & Notes

- Coordinate with MOD-209 security requirements to include compliance evidence within IaC pipeline outputs.
