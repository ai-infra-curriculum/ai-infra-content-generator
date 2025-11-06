# Module Roadmap

> AI Infrastructure MLOps Engineer | Module 552 repackages `lessons/02-cicd-for-ml` with GitOps integration and cross-role reuse.

## Module Overview

- **Module ID**: MOD-552
- **Module Title**: CI/CD for Machine Learning
- **Target Role(s)**: AI Infrastructure MLOps Engineer
- **Duration**: 38 hours (12 lecture, 18 lab, 4 portfolio, 4 assessment)
- **Prerequisites**: MOD-551, MOD-109
- **Next Module(s)**: MOD-556, PROJ-551, PROJ-552

## Cross-Role Progression

- Builds directly on engineer IaC (MOD-109) and senior automation (MOD-206) to avoid reiterating fundamentals.
- Shares pipeline templates with ML Platform (MOD-505) so orchestration patterns stay consistent.
- Introduces policy hooks that governance module (MOD-557) will extend.

## Learning Objectives

- Design ML-aware CI/CD pipelines covering data prep, training, validation, deployment, and rollback.
- Apply GitOps patterns with ArgoCD/Flux that mesh with platform automation.
- Embed automated testing, security scans, and approvals without slowing experimentation.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| pipeline-automation | Proficient | Pipeline architecture review | AI Infrastructure MLOps Engineer |
| model-lifecycle | Proficient | Deployment playbooks | AI Infrastructure MLOps Engineer |

## Content Outline

1. **Pipeline Design Principles** – ML vs. traditional software differences.
2. **Template Implementation** – reusable GitHub Actions / GitLab CI specs.
3. **GitOps Integration** – ArgoCD/Flux workflows, environment promotion.
4. **Testing & Validation** – unit, integration, data, and drift checks in CI.
5. **Security & Compliance** – secrets, vulnerability scanning, approvals.

## Hands-On Activities

- Build reusable CI template referencing senior engineer pipeline assets.
- Configure GitOps deployment flow with staged promotion and rollback.
- Add automated validation gates referencing data quality and security modules.

## Assessments & Evidence

- Peer-reviewed pipeline design document.
- Demo of automated promotion with rollback triggered via GitOps.

## Shared Assets & Legacy Mapping

- Reuses runner configuration from senior engineer repository to avoid duplication.
- Hooks into ML Platform developer experience module for portal integration.
