# Multi-Role Alignment Dashboard

> Tracks how the AI Infrastructure Curriculum scaffolds learners from junior to senior responsibilities while minimizing duplication.

## Program Context

- **Program Name**: AI Infrastructure Curriculum  
- **Roles Covered**: Junior AI Infrastructure Engineer, AI Infrastructure Engineer, Senior AI Infrastructure Engineer, AI Infrastructure Security Engineer, AI Infrastructure MLOps Engineer, AI Infrastructure Performance Engineer, AI Infrastructure ML Platform Engineer, AI Infrastructure Team Lead, Principal AI Infrastructure Engineer, AI Infrastructure Architect, Senior AI Infrastructure Architect, Principal AI Infrastructure Architect  
- **Version**: 1.5.0  
- **Last Updated**: 2025-10-20  
- **Repository Strategy**: per_role (see `curriculum/repository-strategy.yaml`)  
- **Solutions Placement**: Separate repositories (`ai-infra-<role>-solutions`)

### Role Highlights
- **AI Infrastructure Performance Engineer** bridges senior engineering foundations with specialized optimization (kernel, compression, distributed inference) and shares benchmarking/FinOps assets with the MLOps and ML Platform tracks to minimize duplication.

## Progression Ladder

| Sequence | Role | Builds On | New Capabilities Introduced | Shared Assets Reused |
|----------|------|-----------|-----------------------------|----------------------|
| 1 | Junior AI Infrastructure Engineer | Entry-level fundamentals | Python automation, Linux ops, Docker/K8s awareness | Foundational modules reused by higher roles |
| 2 | AI Infrastructure Engineer | Junior curriculum | Production Kubernetes, IaC, observability, MLOps automation | Modules 101-110 extend junior projects with production focus |
| 3 | Senior AI Infrastructure Engineer | AI Infrastructure Engineer specialization | Multi-region architecture, distributed training, governance, leadership | Senior modules reference mid-level artifacts for advanced extensions |
| 4 | AI Infrastructure Security Engineer | Senior AI Infrastructure Engineer | Zero-trust architecture, policy-as-code, adversarial defense, compliance automation | Reuses senior/platform assets and extends them with security guardrails |
| 5 | AI Infrastructure MLOps Engineer | Senior AI Infrastructure Engineer, Security Engineer | Lifecycle automation, monitoring, governance, and LLMOps readiness | Shares CI/CD templates, validation suites, and governance assets with security and platform tracks |
| 6 | AI Infrastructure Performance Engineer | Senior AI Infrastructure Engineer, MLOps Engineer | Kernel optimization, model compression, distributed inference, FinOps analytics | Reuses senior/MLOps benchmarking assets and feeds platform & architect initiatives |
| 7 | AI Infrastructure ML Platform Engineer | Senior AI Infrastructure Engineer, MLOps Engineer, Performance Engineer | Feature store engineering, workflow orchestration, self-service tooling, observability/FinOps | Shares automation templates, feature assets, and developer tooling with security, team lead, and principal tracks |
| 8 | AI Infrastructure Team Lead | Senior AI Infrastructure Engineer | People leadership, team operations, cross-functional delivery, performance coaching | Reuses senior technical assets, incident playbooks, mentorship frameworks |
| 9 | Principal AI Infrastructure Engineer | Senior AI Infrastructure Engineer, Team Lead partnership | Complex systems design, performance optimization, cross-org execution without authority | Shares benchmarking harnesses, excellence playbooks, and program tooling with architect tracks |
| 10 | AI Infrastructure Architect | Senior/Principal Engineer collaboration | Enterprise architecture, multi-cloud strategy, compliance, innovation leadership | Architect modules build on principal engineer technical outputs and add governance/business focus |
| 11 | Senior AI Infrastructure Architect | AI Infrastructure Architect | Enterprise AI strategy, executive advisory, global governance, innovation portfolio | Senior architect modules reuse architect, principal engineer, security, and team lead deliverables |
| 12 | Principal AI Infrastructure Architect | Senior AI Infrastructure Architect | Enterprise portfolio governance, board advisory, innovation ecosystems, multi-cloud modernization | Reuses senior/principal engineer artifacts as executive inputs, extends with principal-level playbooks |

## Competency Alignment By Domain

### Foundational Engineering & Mindset
- **Junior AI Infrastructure Engineer** – Establishes Python, Linux, Git, networking, and container fundamentals (MOD-001…006).  
- **AI Infrastructure Engineer** – Extends those foundations into production Kubernetes, IaC, observability, and CI/CD (MOD-101…110).  
- **Senior AI Infrastructure Engineer** – Adds multi-region architecture, distributed training, governance, and technical leadership (MOD-201…210).  
- **AI Infrastructure Security Engineer** – Layers security-by-design, threat modeling, and compliance automation (MOD-901…908).  
- **AI Infrastructure MLOps Engineer** – Reframes foundations around ML lifecycle governance and validation (MOD-551).  
- **AI Infrastructure Performance Engineer** – Builds GPU architecture, profiling, and optimization baselines (MOD-521).  
- **AI Infrastructure ML Platform Engineer** – Synthesizes engineer/MLOps/performance outputs into platform vision and FinOps framing (MOD-501).  
- **Leadership & Architect Tracks** – Translate these bases into coaching (Team Lead, Principal Engineer) and enterprise strategy (Architect, Senior Architect, Principal Architect).

### Automation, Pipelines & Governance
- **Engineer Tracks** – Provide scripting, IaC, and automation governance groundwork (MOD-109, MOD-206).  
- **Security Engineer** – Wraps policy-as-code enforcement and audit evidence around automation.  
- **MLOps Engineer** – Specializes ML-aware CI/CD, GitOps, and governance automation (MOD-552, MOD-556, MOD-557).  
- **Performance Engineer** – Automates benchmarking, profiling regressions, and validation harnesses (MOD-523).  
- **ML Platform Engineer** – Delivers orchestration, feature pipelines, and developer tooling to scale platform adoption (MOD-505).  
- **Leadership & Architect Tracks** – Govern rollout strategies, review boards, and enterprise pipeline policies.

### Observability, Reliability & FinOps
- **Senior Engineer** – Introduces SRE patterns, error budgets, and incident management (MOD-207).  
- **Security Engineer** – Extends telemetry with threat detection and compliance dashboards.  
- **MLOps Engineer** – Implements model monitoring, drift detection, and cost analytics (MOD-553, MOD-558).  
- **Performance Engineer** – Supplies GPU/DCGM dashboards, benchmarking scorecards, and FinOps reporting (MOD-527).  
- **ML Platform Engineer** – Adds platform service observability and developer analytics (MOD-508).  
- **Leadership & Architect Tracks** – Consume consolidated dashboards for executive, innovation, and board-level reporting.

### Security, Responsible AI & Compliance
- **Security Engineer** – Leads zero-trust architecture, adversarial defense, and compliance automation.  
- **MLOps Engineer** – Embeds responsible AI guardrails and audit evidence (MOD-557, MOD-559).  
- **Performance Engineer** – Validates optimization efforts against accuracy/safety thresholds (MOD-525, MOD-527).  
- **ML Platform Engineer** – Integrates guardrails, RBAC, and audit workflows across shared services (MOD-509, MOD-510).  
- **Architect & Leadership Tracks** – Scale governance frameworks to enterprise contexts, handling policy exceptions and executive accountability.

### Optimization & Scaling
- **Senior Engineer** – Provides distributed training and reliability baselines.  
- **Performance Engineer** – Owns kernel optimization (MOD-522), benchmarking automation (MOD-523), LLM efficiency (MOD-524), distributed inference (MOD-526), and hardware acceleration pilots (MOD-528).  
- **MLOps Engineer** – Operationalizes optimizations via automated pipelines and LLMOps practices (MOD-556, MOD-560).  
- **ML Platform Engineer** – Builds platform primitives and developer experience assets that consume performance artifacts.  
- **Principal Engineer & Architect Tracks** – Convert optimization outputs into org-wide excellence programs, modernization roadmaps, and investment strategies.

### Leadership & Enablement
- **Senior Engineer** – Seeds mentorship and technical leadership skills relied upon by higher-level roles.  
- **Security, MLOps, Performance, Platform Engineers** – Produce domain-specific playbooks and enablement sessions.  
- **Team Lead & Principal Engineer** – Scale coaching, culture, and cross-functional alignment using leadership modules (MOD-801) and principal projects (PROJ-702).  
- **Architect, Senior Architect, Principal Architect** – Craft executive narratives, governance forums, and board-level storytelling rooted in shared technical deliverables.

### Shared Assets & Reuse Highlights
- **Foundations** – MOD-001/002/003, MOD-101, and MOD-201 appear as prerequisites across every track.  
- **Automation** – MOD-109, MOD-206, and the MLOps automation suite (MOD-552, MOD-556, MOD-557) underpin security, performance, and platform pipelines.  
- **Observability** – Dashboards from MOD-207 (senior), MOD-553 (MLOps), and MOD-527 (performance) feed FinOps reporting for leadership and architects.  
- **Optimization** – Performance modules (MOD-522…528) and projects (PROJ-521…524) provide reusable assets for MLOps, ML Platform, principal engineer, and architect initiatives.  
- **Governance** – Security modules and MLOps responsible AI tooling supply shared policy libraries consumed by performance, platform, and architect tracks.  
- **Enablement** – Leadership modules (MOD-210, MOD-801) and cross-role projects (PROJ-803, PROJ-702) keep mentorship and executive communication consistent.

## Role-Specific Differentiators

- **Junior** – Guided labs, documentation focus, supervised incident response.  
- **AI Engineer** – Leads production operations, automation, observability, and project delivery.  
- **Senior** – Owns architecture, governance, performance optimization, and people leadership.  
- **Security Engineer** – Embeds zero-trust, adversarial defense, compliance automation, and SOC integration across AI pipelines and platforms.  
- **MLOps Engineer** – Operationalizes the ML lifecycle with automated validation, governance, and LLMOps orchestration.  
- **Performance Engineer** – Drives CUDA/LLM optimization, distributed inference, and benchmarking governance.  
- **ML Platform Engineer** – Builds self-service ML platforms and developer experience tooling to accelerate delivery.  
- **Team Lead** – Converts technical strategy into squad execution and culture.  
- **Principal Engineer** – Leads complex systems design, cross-org execution, mentoring, and excellence programs.  
- **Architect / Senior Architect / Principal Architect** – Define enterprise strategy, communicate with executives and boards, and steward AI portfolio governance.

## Dependencies & Timeline

| Milestone | Roles Impacted | Owner | Due Date | Status |
|-----------|----------------|-------|----------|--------|
| Research migration | Junior, AI, Senior, Architect | Curriculum Team | 2025-11-15 | In progress |
| Curriculum draft | Junior, AI, Senior, Architect | Curriculum Team | 2025-11-30 | In progress |
| Principal architect curriculum launch | Principal | Program Lead | 2025-12-10 | In progress |
| Principal engineer curriculum launch | Principal IC | Program Lead | 2025-12-05 | In progress |
| Team lead curriculum launch | Team Lead | Program Lead | 2025-11-28 | In progress |
| Security engineer curriculum launch | Security | Program Lead | 2025-11-25 | In progress |
| ML platform engineer curriculum launch | ML Platform | Program Lead | 2025-11-22 | In progress |
| Performance engineer curriculum launch | Performance | Curriculum Team | 2025-12-02 | In progress |
| Validation pipeline updates | AI, Senior, Architect | QA Lead | 2025-12-15 | Pending |
| Leadership enablement rollout | Senior, Architect | Program Lead | 2026-01-15 | Planned |

## Change Log

| Date | Change | Roles | Author |
|------|--------|-------|--------|
| 2025-10-17 | Initial migration for junior role | Junior | Migration script |
| 2025-10-18 | Added AI Infrastructure Engineer role alignment | Junior, AI | Migration script |
| 2025-10-18 | Integrated Senior AI Infrastructure Engineer role alignment | Junior, AI, Senior | Curriculum Team |
| 2025-10-19 | Added AI Infrastructure Architect role alignment | Junior, AI, Senior, Architect | Curriculum Team |
| 2025-10-20 | Added Principal AI Infrastructure Architect role alignment | Principal Architect | Curriculum Team |
| 2025-10-20 | Added Principal AI Infrastructure Engineer role alignment | Principal Engineer | Curriculum Team |
| 2025-10-20 | Added AI Infrastructure Team Lead role alignment | Team Lead | Curriculum Team |
| 2025-10-20 | Added AI Infrastructure Security Engineer role alignment | Security | Curriculum Team |
| 2025-10-20 | Added AI Infrastructure ML Platform Engineer role alignment | ML Platform | Curriculum Team |
| 2025-11-05 | Added AI Infrastructure MLOps & Performance Engineer alignment refresh | MLOps, Performance | Curriculum Team |
