# Module Roadmap

> AI Infrastructure MLOps Engineer | Module 559 reuses `lessons/09-security` while embedding responsible AI expectations.

## Module Overview

- **Module ID**: MOD-559
- **Module Title**: Secure & Responsible MLOps
- **Target Role(s)**: AI Infrastructure MLOps Engineer
- **Duration**: 34 hours (10 lecture, 14 lab, 6 portfolio, 4 assessment)
- **Prerequisites**: MOD-557, MOD-558
- **Next Module(s)**: MOD-560, PROJ-554, PROJ-555

## Cross-Role Progression

- Shares security guardrails with AI Infrastructure Security Engineer (MOD-901–910) to avoid duplicating controls.
- Aligns with architect responsible AI program (MOD-605) for strategic governance.
- Provides risk mitigation deliverables referenced by principal engineer and team lead tracks.

## Learning Objectives

- Conduct threat modelling, vulnerability analysis, and defense-in-depth for ML pipelines/services.
- Implement responsible AI guardrails (bias checks, explainability, human-in-the-loop) within automation.
- Coordinate security/compliance sign-off processes with minimal manual overhead.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| platform-security | Expert | Threat model & mitigation plan | AI Infrastructure MLOps Engineer |
| responsible-ai | Expert | Bias & explainability guardrail demo | AI Infrastructure MLOps Engineer |

## Content Outline

1. **Threat Landscape** – adversarial ML, supply-chain attacks, data poisoning.
2. **Security Controls** – secrets, RBAC, network policies, scanning.
3. **Responsible AI Checks** – bias detection, explainability, human review cadence.
4. **Governance Integration** – approvals, exception handling, documentation.
5. **Incident Collaboration** – security + SRE + product alignment.

## Hands-On Activities

- Create threat model referencing security role templates and update mitigation backlog.
- Add responsible AI checks to CI/CD + orchestration pipelines.
- Run joint tabletop with security and compliance teams focusing on ML-specific incidents.

## Assessments & Evidence

- Security review including automated evidence capture.
- Responsible AI compliance packet reused by architect and leadership programs.

## Shared Assets & Legacy Mapping

- Inherits policy libraries from security engineers to prevent divergence.
- Provides guardrails consumed in LLMOps capstone (PROJ-555).
