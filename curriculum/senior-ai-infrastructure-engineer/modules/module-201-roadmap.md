# Module Roadmap

> Senior AI Infrastructure Engineer | Derived from legacy Module 201 assets and refreshed with current platform practices.

## Module Overview

- **Module ID**: MOD-201
- **Module Title**: Advanced Kubernetes & Cloud-Native Architecture
- **Target Role(s)**: Senior AI Infrastructure Engineer
- **Duration**: 60 hours (20 lecture, 24 lab, 12 project, 4 assessment)
- **Prerequisites**: MOD-104, MOD-109
- **Next Module(s)**: MOD-202, MOD-205, MOD-208

## Cross-Role Progression

- **Builds On**: AI Infrastructure Engineer modules covering production Kubernetes and IaC.
- **Adds New Depth**: Operator development, multi-cluster strategy, service mesh, disaster recovery.
- **Shared Assets**: Reuses mid-level Helm charts as baselines; expands GitOps pipelines created in MOD-109.
- **Differentiators**: Focus on platform-level governance, chaos engineering, and fleet-wide lifecycle management.

## Learning Objectives

- Design custom Kubernetes operators/CRDs to automate ML workload orchestration.
- Architect multi-cluster, multi-region environments with service mesh, advanced networking, and GitOps promotion flows.
- Implement hardened security (OPA/Kyverno, network policies) and chaos-driven resiliency practices.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| kubernetes-platform-engineering | Expert | Capstone operator demo, GitOps workflow review | Senior AI Infrastructure Engineer |
| platform-automation | Proficient | Helm/ArgoCD automation, policy enforcement artifacts | Senior AI Infrastructure Engineer |

## Content Outline

1. Operator pattern deep dive (controller-runtime, reconciliation loops, custom metrics).
2. Multi-cluster topologies (hub-and-spoke, fleet API, service mesh federation).
3. Advanced scheduling (topology spread, device plugins, workload classes for ML).
4. Security hardening (zero-trust networking, pod security standards, secret automation).
5. Resiliency engineering (chaos experiments, disaster recovery drills, kube-bench/kube-hunter audits).
6. GitOps + policy stacks (ArgoCD/Flux, OPA/Kyverno, pipeline gating, drift detection).

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Build & deploy custom operator for ML training jobs | Operator reconciles job life-cycle, exposes metrics | `python-strict` |
| Lab 2 | Configure multi-cluster GitOps promotion flow | Fleet promotion executes with policy checks | Policy-as-code review |
| Lab 3 | Run chaos & DR scenario | Documented recovery within target RTO | Chaos experiment report |
| Assessment | Architecture review & oral defense | Score ≥ 80% on design rubric | Panel-driven review |

## Solutions Plan

- **Coverage**: Operator reference implementation, GitOps manifests, chaos experiment automation, review rubric.
- **Repository Strategy**: Solutions reside in dedicated repo (`ai-infra-senior-engineer-solutions`) per role.
- **Solution Path**: `modules/senior-ai-infrastructure-engineer/module-201-advanced-kubernetes/solutions` (metadata + legacy references).
- **Validation Status**: Requires execution of GitOps pipeline and chaos scenario under new CI harness.

## Resource Plan

- **Primary References**:
  - Module README in `modules/senior-ai-infrastructure-engineer/module-201-advanced-kubernetes/README.md`
  - Legacy labs within same directory
- **Supplemental Resources**:
  - CNCF operator pattern guides, ArgoCD/Flux documentation
  - Service mesh references (Istio, Linkerd)
- **Tooling Requirements**:
  - kind/AKS/EKS clusters, controller-runtime SDK, ArgoCD/Flux, OPA/Kyverno, Chaos Mesh or Litmus

## Quality Checklist

- [ ] Operator labs validated against policy-as-code gates
- [ ] Chaos experiment outcomes documented with mitigation backlog
- [ ] GitOps pipelines include automated drift detection
- [ ] Security posture reviewed (network policies, secrets, image signing)
- [ ] Module deliverables linked to PROJ-304 planning

## Dependencies & Notes

- Coordinates with MOD-205 (multi-cloud) for network patterns and MOD-208 for reusable IaC modules.
- Encourage learners to schedule hardware simulations early to avoid cluster quota issues.
