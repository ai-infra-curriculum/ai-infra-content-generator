# Multi-Role Alignment Dashboard

> Tracks how the AI Infrastructure Curriculum scaffolds learners from junior to senior responsibilities while minimizing duplication.

## Program Context

- **Program Name**: AI Infrastructure Curriculum
- **Roles Covered**: Junior AI Infrastructure Engineer, AI Infrastructure Engineer, Senior AI Infrastructure Engineer, AI Infrastructure Team Lead, Principal AI Infrastructure Engineer, AI Infrastructure Architect, Senior AI Infrastructure Architect, Principal AI Infrastructure Architect
- **Version**: 1.5.0
- **Last Updated**: 2025-10-20
- **Repository Strategy**: per_role (see `curriculum/repository-strategy.yaml`)
- **Solutions Placement**: Separate repositories (`ai-infra-<role>-solutions`)

## Progression Ladder

| Sequence | Role | Builds On | New Capabilities Introduced | Shared Assets Reused |
|----------|------|-----------|-----------------------------|----------------------|
| 1 | Junior AI Infrastructure Engineer | Entry-level fundamentals | Python automation, Linux ops, Docker/K8s awareness | Foundational modules reused by higher roles |
| 2 | AI Infrastructure Engineer | Junior curriculum | Production Kubernetes, IaC, observability, MLOps automation | Modules 101-110 extend junior projects with production focus |
| 3 | Senior AI Infrastructure Engineer | AI Infrastructure Engineer specialization | Multi-region architecture, distributed training, governance, leadership | Senior modules reference mid-level artifacts for advanced extensions |
| 4 | AI Infrastructure Team Lead | Senior AI Infrastructure Engineer | People leadership, team operations, cross-functional delivery, performance coaching | Reuses senior technical assets, incident playbooks, and mentorship frameworks to create leadership guides |
| 5 | Principal AI Infrastructure Engineer | Senior AI Infrastructure Engineer, Team Lead partnership | Complex systems design, performance optimization, cross-org execution without authority | Shares benchmarking harnesses, excellence playbooks, and program tooling with architect tracks |
| 6 | AI Infrastructure Architect | Senior/Principal Engineer collaboration | Enterprise architecture, multi-cloud strategy, compliance, innovation leadership | Architect modules build on principal engineer technical outputs and add governance/business focus |
| 7 | Senior AI Infrastructure Architect | AI Infrastructure Architect | Enterprise AI strategy, executive advisory, global governance, innovation portfolio | Senior architect modules leverage architect, principal engineer, and team lead deliverables for executive engagements |
| 8 | Principal AI Infrastructure Architect | Senior AI Infrastructure Architect | Enterprise portfolio governance, board advisory, innovation ecosystems, multi-cloud modernization | Reuses senior/principal engineer artifacts as executive inputs, extends with principal-level playbooks |

## Role Comparison Matrix

| Competency Domain | Junior AI Infrastructure Engineer | AI Infrastructure Engineer | Senior AI Infrastructure Engineer | AI Infrastructure Team Lead | Principal AI Infrastructure Engineer | AI Infrastructure Architect | Senior AI Infrastructure Architect | Principal AI Infrastructure Architect | Shared Modules | Divergent Needs |
|--------------------|-----------------------------------|---------------------------|----------------------------------|---------------------------------------------|----------------------------------------------|---------------------------|-------------------------------|-----------------------------------------------|----------------|-----------------|
| Foundational Engineering | Python scripting, Linux CLI, Git basics | Own automation tooling and CI workflows | Drive platform-wide standards, author ADRs, mentor teams | Champion engineering excellence, mentor squads, translate technical direction into team execution | Set org-wide engineering bar, mentor staff/principal talent, defend technical direction | Define enterprise strategy, governance, and executive communication | Advise board/investors, lead transformation portfolios, mentor architects | Set enterprise-wide engineering vision, steward principal talent pipeline, align architecture practice with corporate strategy | MOD-001/002/003, MOD-301, MOD-401 | Team lead converts individual excellence into team-wide practices; principal architect ensures global consistency while empowering specialization tracks |
| Infrastructure Foundations | Docker/K8s awareness, cloud basics | Production Kubernetes, IaC, multi-cloud | Multi-region, operators, GitOps governance | Operate squads’ platform stacks, coordinate with platform teams on upgrades and cost | Own complex systems design, hardware strategy, capacity & FinOps execution | Multi-cloud strategy, vendor management, FinOps | Global platform strategy, sovereignty, sustainability, geopolitical risk | Owns global infrastructure investments, custom hardware strategy, and sovereign compliance across regions | MOD-005/006, MOD-103/104, MOD-201, MOD-302, MOD-406 | Team lead balances platform constraints with squad velocity; principal architect balances performance, cost, ESG, and geopolitical risk |
| Operations & Monitoring | Follow runbooks, respond to guided incidents | Design observability platforms, lead on-call | Establish SLO programs, chaos drills, executive reporting | Lead on-call rotations, postmortems, and operational rituals for squads | Run incident command, automate SLO enforcement, institutionalize postmortems | Set organization-wide reliability standards and reporting | Align reliability with board risk appetite, sustainability, and crisis response | Define enterprise resilience KPIs, board-level incident governance, and sustainability accountability | MOD-009, MOD-108, MOD-207, MOD-305, MOD-406 | Team lead converts ops strategy into daily practice and wellbeing; principal architect reconciles resilience trade-offs with investment pressures |
| MLOps & Pipelines | Assist with basic pipelines | Build automated pipelines, MLflow, feature stores | Deliver self-service platforms, compliance workflows, analytics | Coordinate backlog with platform/product, ensure adoption of platform capabilities | Extend platform capabilities, ensure performance/compliance, partner with research | Govern enterprise MLOps product strategy, adoption, and ROI | Drive enterprise AI transformation and portfolio measurement | Govern AI investment portfolio, value realization, and cross-business capability scaling | MOD-004/007, MOD-105/106, MOD-206, MOD-306, MOD-401 | Team lead prevents duplicate tooling and enforces adoption at squad level; principal architect sets funding guardrails |
| Data & LLM Platforms | Intro data tooling | Operate data pipelines, basic LLM serving | GPU optimization, LLM deployment | Align squad work with data/LLM platform roadmaps, manage readiness and rollout | Optimize training & inference pipelines, orchestrate data performance at scale | Architect enterprise LLM/RAG and data platforms with governance | Govern global data/LLM strategy, responsible AI, and regulatory engagement | Shape enterprise data/LLM strategy with regulators, partners, and industry coalitions | MOD-107, MOD-203/204, MOD-307, MOD-308, MOD-405 | Team lead ensures successful adoption and feedback loops; principal architect orchestrates external influence and policy alignment |
| Security & Compliance | Apply baseline security guidance | Implement IAM, secrets, observability guardrails | Lead zero-trust strategy, audit automation, threat modeling | Embed security-by-design within squad workflows, coordinate audits and incident comms | Embed security-by-design, partner with compliance on critical launches | Govern enterprise compliance, responsible AI, and executive risk reporting | Advise board/regulators, integrate RAI, sustainability, and legal demands | Chair global governance, define risk appetite, and engage regulators/boards on AI compliance | MOD-208/209, MOD-303, MOD-405, MOD-503 | Team lead keeps compliance actionable and humane; principal architect ensures unified governance across jurisdictions |

## Module Assignment by Role

| Module ID | Module Title | Junior Status | AI Engineer Status | Senior Status | Team Lead Status | Principal Engineer Status | AI Infrastructure Architect Status | Senior AI Infrastructure Architect Status | Principal AI Infrastructure Architect Status | Notes |
|-----------|--------------|---------------|--------------------|---------------|-------------------|---------------------------|-------------------|-------------------------|----------------------------|-------|
| MOD-101 – MOD-110 | AI Engineer specialization | N/A | Core | Prerequisite | Reference | Reference | Reference | Reference | Reference | Prepares learners for advanced depth |
| MOD-201 | Advanced Kubernetes & Cloud-Native Architecture | N/A | N/A | Core | Reference | Core | Prerequisite | Reference | Reference | Operator, GitOps, multi-cluster focus |
| MOD-202 | Distributed Training at Scale | N/A | N/A | Core | Reference | Core | Prerequisite | Reference | Reference | Builds on MOD-107 + MOD-201 |
| MOD-203 | Advanced GPU Computing & Optimization | N/A | N/A | Core | Reference | Core | Reference | Reference | Reference | CUDA, TensorRT, fleet management |
| MOD-204 | Advanced Model Optimization & Inference | N/A | N/A | Core | Reference | Core | Prerequisite | Reference | Reference | LLM performance optimization |
| MOD-205 | Multi-Cloud Architecture & Resilience | N/A | N/A | Advanced | Reference | Core | Core | Core | Prerequisite | Active-active, DR, FinOps |
| MOD-206 | Advanced MLOps & Platform Engineering | N/A | N/A | Advanced | Reference | Core | Core | Core | Prerequisite | Platform product mindset |
| MOD-207 | Advanced Observability & SRE Practices | N/A | N/A | Advanced | Reference | Advanced | Reference | Reference | Reference | SLO programs, chaos engineering |
| MOD-208 | Advanced IaC & GitOps | N/A | N/A | Advanced | Reference | Core | Core | Core | Prerequisite | Policy-as-code, compliance automation |
| MOD-209 | Security & Compliance for ML Systems | N/A | N/A | Advanced | Reference | Core | Core | Core | Prerequisite | Zero-trust, audit automation |
| MOD-210 | Technical Leadership & Mentorship | N/A | N/A | Advanced | Reference | Core | Reference | Reference | Reference | Strategy, communication, mentorship |
| MOD-301 | Enterprise Architecture Fundamentals | N/A | N/A | Reference | Reference | Reference | Core | Prerequisite | Prerequisite | Enterprise governance, ADRs, TOGAF |
| MOD-302 | Multi-Cloud & Hybrid Architecture | N/A | N/A | Reference | Reference | Reference | Core | Prerequisite | Prerequisite | Multi-cloud strategy, migration, sovereignty |
| MOD-303 | Enterprise Security & Compliance Architecture | N/A | N/A | Reference | Reference | Reference | Core | Core | Prerequisite | Zero-trust, compliance automation, RAI |
| MOD-304 | Cost Optimization & FinOps | N/A | N/A | Reference | Reference | Reference | Core | Core | Prerequisite | FinOps operating model, ROI cases |
| MOD-305 | High Availability & Disaster Recovery | N/A | N/A | Reference | Reference | Reference | Core | Core | Prerequisite | Active-active, chaos programs |
| MOD-306 | Enterprise MLOps Platform Architecture | N/A | N/A | Reference | Reference | Reference | Advanced | Prerequisite | Prerequisite | Platform product strategy |
| MOD-307 | Data Architecture for AI | N/A | N/A | Reference | Reference | Reference | Advanced | Core | Prerequisite | Lakehouse, governance, lineage |
| MOD-308 | LLM Platform & RAG Architecture | N/A | N/A | Reference | Reference | Reference | Advanced | Core | Prerequisite | Enterprise LLM/RAG, safety |
| MOD-309 | Architecture Communication & Leadership | N/A | N/A | Reference | Reference | Reference | Advanced | Core | Prerequisite | Executive storytelling, governance boards |
| MOD-310 | Emerging Technologies & Innovation | N/A | N/A | Reference | Reference | Reference | Advanced | Prerequisite | Reference | Technology radar, innovation strategy |
| MOD-401 | Enterprise AI Strategy & Vision | N/A | N/A | Reference | Reference | Reference | Reference | Core | Prerequisite | Enterprise AI transformation strategy |
| MOD-402 | Executive Leadership & Communication | N/A | N/A | Reference | Reference | Reference | Reference | Core | Prerequisite | Executive presence, crisis leadership |
| MOD-403 | Enterprise Architecture Governance & Standards | N/A | N/A | Reference | Reference | Reference | Reference | Core | Prerequisite | Governance operating model |
| MOD-404 | Innovation & R&D Leadership | N/A | N/A | Reference | Reference | Reference | Reference | Core | Prerequisite | Innovation portfolio leadership |
| MOD-405 | Responsible AI & Ethics Leadership | N/A | N/A | Reference | Reference | Reference | Reference | Core | Prerequisite | Responsible AI governance |
| MOD-406 | Global Infrastructure & Sustainability | N/A | N/A | Reference | Reference | Reference | Reference | Core | Prerequisite | Global topology, sustainability |
| MOD-407 | Strategic Partnerships & Vendor Ecosystems | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Prerequisite | Partnership ecosystem leadership |
| MOD-408 | M&A Technical Due Diligence | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Prerequisite | Acquisition assessment & integration |
| MOD-409 | Thought Leadership & Industry Influence | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Reference | Publications, standards influence |
| MOD-410 | Future of AI Infrastructure | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Reference | Scenario planning, emerging tech strategy |
| MOD-601 | Enterprise Architecture Leadership | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Core | Principal-level operating model playbook |
| MOD-602 | Strategic Alignment & Value Mapping | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Core | Portfolio economics, funding guardrails |
| MOD-603 | Global Governance & Risk Orchestration | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Core | Global governance automation & reporting |
| MOD-604 | Innovation Ecosystems & Thought Leadership | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Ecosystem orchestration, industry influence |
| MOD-605 | Responsible AI Programs at Enterprise Scale | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Core | Enterprise responsible AI governance |
| MOD-606 | Multi-Cloud Transformation Leadership | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Core | Sovereign-aware modernization & FinOps |
| MOD-607 | Strategic Partnership & Ecosystem Strategy | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Alliance governance, commercial strategy |
| MOD-608 | M&A Technical Due Diligence for AI | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Advanced | Enterprise M&A integration leadership |
| MOD-609 | Executive Communication & Influence at Board Level | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Core | Board communication, crisis response |
| MOD-610 | Future-Proofing & Horizon Scanning | N/A | N/A | N/A | Reference | Reference | Reference | Reference | Capstone | Scenario planning, foresight integration |
| MOD-701 | Technical Excellence at Principal Scale | N/A | N/A | Reference | Reference | Core | Reference | Reference | Reference | Principal IC engineering excellence playbook |
| MOD-702 | Complex Systems Design Leadership | N/A | N/A | Reference | Reference | Core | Reference | Reference | Reference | Large-scale systems design & governance |
| MOD-703 | Performance Engineering at Extreme Scale | N/A | N/A | Reference | Reference | Core | Reference | Reference | Reference | Benchmarking, optimization, capacity planning |
| MOD-704 | Innovation & Applied Research Leadership | N/A | N/A | Reference | Reference | Advanced | Reference | Reference | Reference | Applied research incubation for IC track |
| MOD-705 | Mentorship & Engineering Leadership | N/A | N/A | Reference | Reference | Core | Reference | Reference | Reference | Mentorship and talent acceleration |
| MOD-706 | Cross-Org Initiatives & Execution | N/A | N/A | Reference | Reference | Capstone | Reference | Reference | Reference | Program leadership without direct authority |

## Shared Assets

- **Projects**:
  - Junior: `projects/junior-ai-infrastructure-engineer/project-01-simple-model-api` … `project-05-production-ml-capstone`
  - AI Engineer: `projects/ai-infrastructure-engineer/project-101-basic-model-serving` … `project-103-llm-deployment`
  - Senior: `projects/senior-ai-infrastructure-engineer/project-301-distributed-training` … `project-304-k8s-operator`
  - Principal Engineer: `projects/ai-infrastructure-principal-engineer/project-701-technical-excellence` … `project-703-cross-org-initiative`
  - Architect: `projects/ai-infrastructure-architect/project-401-enterprise-mlops-platform` … `project-405-security-framework`
  - Principal Architect: `projects/ai-infrastructure-principal-architect/project-601-enterprise-architecture-playbook` … `project-604-ecosystem-leadership-initiative`
- **Solutions**: Separate repos with consistent metadata (`solutions/ai-infra-<role>-solutions`)
- **Case Studies & Resources**: Module resource folders reused with advanced annotations
- **Assessments**: Quizzes, architecture reviews, and leadership evaluations aligned across levels

## Reuse Strategy & Anti-Duplication

- Junior assets scaffold core skills; AI Engineer reuses labs as refresh modules or optional warm-ups.
- Senior modules reference mid-level codebases to extend functionality (e.g., upgrade GitOps pipelines, add policy layers) rather than duplicating artifacts.
- Team lead modules convert senior/principal technical standards into squad-level playbooks, incident guides, and people-development frameworks.
- Architect modules reuse senior and principal deliverables (operators, DR automation, FinOps dashboards) as inputs to enterprise-level governance, reducing duplication.
- Principal engineer modules consume senior artifacts (platform codebases, reliability frameworks) and extend them with performance harnesses, excellence playbooks, and cross-org tooling that architects reuse.
- Principal architect modules consume senior and principal engineer artifacts as executive inputs (transformation strategies, governance playbooks) and extend them with board-ready portfolios instead of recreating work.
- Repository strategy ensures cross-role synchronization through mapping files in `configs/repositories/`.
- Solutions metadata links back to legacy repos, preventing redundant storage while migration completes.

## Role-Specific Differentiators

- **Junior**: Guided labs, documentation focus, supervised incident response.
- **AI Engineer**: Leads production operations, automation, observability, and project delivery.
- **Senior**: Owns architecture, governance, performance optimizations, and people leadership.
- **Team Lead**: Converts technical strategy into squad execution, manages operations, culture, and cross-functional delivery.
- **Principal Engineer**: Leads complex systems design, optimization programs, and cross-org execution while mentoring staff/principal talent.
- **Architect**: Defines enterprise strategy, governs standards, optimizes cost/resilience, and communicates with executives.
- **Principal Architect**: Governs enterprise AI portfolio, advises boards/investors, orchestrates innovation ecosystems, and directs multi-cloud modernization strategy.

## Dependencies & Timeline

| Milestone | Roles Impacted | Owner | Due Date | Status |
|-----------|----------------|-------|----------|--------|
| Research migration | Junior, AI, Senior, Architect | Curriculum Team | 2025-11-15 | In progress |
| Curriculum draft | Junior, AI, Senior, Architect | Curriculum Team | 2025-11-30 | In progress |
| Principal architect curriculum launch | Principal | Program Lead | 2025-12-10 | In progress |
| Principal engineer curriculum launch | Principal IC | Program Lead | 2025-12-05 | In progress |
| Team lead curriculum launch | Team Lead | Program Lead | 2025-11-28 | In progress |
| Validation pipeline updates | AI, Senior, Architect | QA Lead | 2025-12-15 | Pending |
| Leadership enablement rollout | Senior, Architect | Program Lead | 2026-01-15 | Planned |

## Change Log

| Date | Change | Roles | Author |
|------|--------|-------|--------|
| 2025-10-17 | Initial migration for junior role | Junior | Migration script |
| 2025-10-18 | Added AI Infrastructure Engineer role alignment | Junior, AI | Migration script |
| 2025-10-18 | Integrated Senior AI Infrastructure Engineer role alignment | Junior, AI, Senior | Curriculum Team |
| 2025-10-19 | Added AI Infrastructure Architect role alignment | Junior, AI, Senior, Architect | Curriculum Team |
| 2025-10-20 | Added Principal AI Infrastructure Architect role alignment | Principal | Curriculum Team |
| 2025-10-20 | Added Principal AI Infrastructure Engineer role alignment | Principal IC | Curriculum Team |
| 2025-10-20 | Added AI Infrastructure Team Lead role alignment | Team Lead | Curriculum Team |
