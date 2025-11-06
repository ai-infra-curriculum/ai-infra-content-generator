# Module Roadmap

> Reconstructed from the legacy infrastructure-as-code module with emphasis on ML platform automation.

## Module Overview

- **Module ID**: MOD-109
- **Module Title**: Infrastructure as Code for ML Platforms
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 40 hours (14 lecture, 18 lab, 6 project, 2 assessment)
- **Prerequisites**: MOD-102, MOD-103
- **Next Module(s)**: MOD-110, PROJ-203

## Cross-Role Progression

- **Builds On** (modules/roles): Junior automation patterns (MOD-001) and cloud setup (MOD-102)
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Terraform starter modules reused for junior refresher workshops
- **Differentiators** (role-specific emphasis): Multi-environment pipelines, testing frameworks, policy enforcement

## Learning Objectives

- Author reusable Terraform/Pulumi modules that provision ML infrastructure components.
- Implement environment promotion workflows with GitOps, CI/CD, and policy checks.
- Integrate secrets management, observability, and security baselines into IaC modules.
- Test infrastructure code using automated unit/integration testing and policy-as-code (OPA) rules.
- Document and govern IaC modules to support cross-team adoption and compliance.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Infrastructure automation | Proficient | Module code review + deployment demo | AI Infrastructure Engineer |
| Security & compliance | Working | Policy-as-code and drift reports | AI Infrastructure Engineer |
| Collaboration & governance | Working | Module documentation and onboarding guide | AI Infrastructure Engineer |

## Content Outline

1. **IaC Architecture Patterns** – module structure, state management, environment strategy.
2. **Terraform/Pulumi Deep Dive** – reusable components, versioning, registries.
3. **Pipelines & GitOps** – CI/CD workflows, environment promotion, drift detection.
4. **Policy & Security** – OPA/Conftest, Sentinel, baseline tagging, compliance gates.
5. **Testing Infrastructure** – unit tests (Terratest), integration tests, sandbox environments.
6. **Secrets & Observability** – secret backends, monitoring hooks, logging integration.
7. **Documentation & Governance** – module READMEs, changelog, support model.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Build reusable Terraform module | Module published with versioning and tests | `terraform validate` + Terratest |
| Lab 2 | Implement GitOps pipeline | Promotion workflow demo; drift detected & reconciled | CI pipeline run |
| Lab 3 | Apply policy-as-code | Failing policy blocks deployment; report generated | OPA policy check |
| Assessment | IaC quiz | ≥80% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Terraform modules, Terratest suites, OPA policies, pipeline configs, quiz key
- **Repository Strategy**: `per_role` with separate solutions repo for IaC blueprints
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-109-infrastructure-as-code/solutions`
- **Validation Status**: Pending execution of Terratest suite and policy checks in CI

## Resource Plan

- **Primary References**:
  - Module README in `modules/ai-infrastructure-engineer/module-109-infrastructure-as-code`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-109-infrastructure-as-code`
- **Supplemental Resources**:
  - Terraform Best Practices guides, Open Policy Agent docs
  - GitOps playbooks (Flux/ArgoCD)
- **Tooling Requirements**:
  - Terraform ≥1.6 or Pulumi, Terratest (Go), Conftest/OPA, GitHub Actions or equivalent CI, secret backend (Vault/Secrets Manager)

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Cloud provisioning (MOD-102), containerization (MOD-103), monitoring (MOD-108 for integration)
- **Downstream Outputs**: Enables LLM infrastructure module (MOD-110) and Project PROJ-203
- **Risks / Mitigations**: State management mishandling—enforce remote state and locking in labs.
