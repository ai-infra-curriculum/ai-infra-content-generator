# Module Roadmap

> Derived from the legacy Kubernetes fundamentals track to capture the production ML emphasis.

## Module Overview

- **Module ID**: MOD-104
- **Module Title**: Kubernetes Fundamentals for ML
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 60 hours (20 lecture, 24 lab, 12 project, 4 assessment)
- **Prerequisites**: MOD-103, MOD-006
- **Next Module(s)**: MOD-105, MOD-107, PROJ-201

## Cross-Role Progression

- **Builds On** (modules/roles): Junior MOD-006 Kubernetes Introduction labs
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Base manifests adopted from junior labs but extended with production features
- **Differentiators** (role-specific emphasis): GPU scheduling, Helm packaging, observability, incident playbooks

## Learning Objectives

- Deploy, scale, and upgrade ML services on Kubernetes clusters.
- Configure ingress, storage, and secrets management for production workloads.
- Implement Helm-based packaging and release management for ML services.
- Operate GPU workloads and optimize resource allocation in mixed clusters.
- Troubleshoot and remediate common Kubernetes failures for ML deployments.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Kubernetes operations | Proficient | Cluster deployment + rolling upgrade demonstration | AI Infrastructure Engineer |
| Observability & reliability | Proficient | Kubernetes monitoring dashboards | AI Infrastructure Engineer |
| Automation & tooling | Working | Helm chart packaging workflow | AI Infrastructure Engineer |

## Content Outline

1. **Kubernetes Architecture Review** – control plane, nodes, workloads, lifecycle.
2. **Core Resources** – Deployments, Services, ConfigMaps, Secrets, StatefulSets for ML.
3. **Networking & Ingress** – service discovery, ingress controllers, service mesh options.
4. **Storage & Data** – PersistentVolumeClaims, CSI drivers, artifact storage patterns.
5. **Helm & Package Management** – chart structure, values, promotion strategies.
6. **GPU Workloads** – node pools, device plugins, scheduling constraints, quotas.
7. **Monitoring & Troubleshooting** – logs, events, metrics, debugging pods, chaos drills.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Create production-grade cluster manifests | Namespace isolation, RBAC policies implemented | Policy lint (`kube-score`) |
| Lab 2 | Deploy ML serving stack with Helm | Chart published; canary release executed | Helm chart test |
| Lab 3 | GPU scheduling exercise | GPU workload runs; utilization report generated | GPU profiler script |
| Assessment | Kubernetes troubleshooting exam | ≥75% scenario accuracy | Practical evaluation rubric |

## Solutions Plan

- **Solution Coverage Required**: Helm chart, cluster configs, GPU scheduling walkthrough, troubleshooting answers
- **Repository Strategy**: `per_role` with separate solutions repository
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-104-kubernetes/solutions`
- **Validation Status**: Requires executing kubeconform/policy tests within CI pipeline

## Resource Plan

- **Primary References**:
  - `modules/ai-infrastructure-engineer/module-104-kubernetes/README.md`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-104-kubernetes`
- **Supplemental Resources**:
  - CNCF Kubernetes whitepapers, CKAD exam prep
  - NVIDIA GPU Operator documentation
- **Tooling Requirements**:
  - kind/minikube, kubectl, Helm, kube-score, NVIDIA device plugin (for GPU labs)

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: MOD-103 containerization and junior Kubernetes intro
- **Downstream Outputs**: Critical for Projects PROJ-201 and PROJ-203, plus modules MOD-108/109
- **Risks / Mitigations**: Cluster access variance—provide local kind cluster fallback instructions.
