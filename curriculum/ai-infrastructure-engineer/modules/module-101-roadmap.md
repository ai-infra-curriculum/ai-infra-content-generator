# Module Roadmap

> Updated from the legacy AI Infrastructure Engineer curriculum to provide concrete objectives, activities, and validation hooks.

## Module Overview

- **Module ID**: MOD-101
- **Module Title**: Foundations of ML Infrastructure
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 40 hours (12 lecture, 20 lab, 6 project, 2 assessment)
- **Prerequisites**: MOD-005, MOD-006, MOD-010
- **Next Module(s)**: MOD-102, MOD-103

## Cross-Role Progression

- **Builds On** (modules/roles): Junior AI Infrastructure Engineer Modules 005–010 (containers, Kubernetes, cloud basics)
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Exercises from `modules/junior-ai-infrastructure-engineer/module-005-docker-containers` reused as warmups
- **Differentiators** (role-specific emphasis): Production-ready environment setup, ML lifecycle ownership, and deployment independence

## Learning Objectives

- Explain the ML lifecycle and articulate where infrastructure enables each stage.
- Configure a professional ML infrastructure workstation (Python 3.11+, Docker, GPU drivers, cloud CLIs).
- Deploy a pre-trained model behind a FastAPI/TorchServe endpoint with automated tests.
- Containerize the service with optimized Dockerfiles and publish to a registry.
- Stand up a monitored single-node deployment on a cloud VM with health checks and logging.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Infrastructure automation | Proficient | Containerized service, deployment scripts | AI Infrastructure Engineer |
| ML operations foundations | Proficient | Deployment guide, monitoring checklist | AI Infrastructure Engineer |
| Cloud platform literacy | Proficient | Cloud VM deployment with observability | AI Infrastructure Engineer |

## Content Outline

1. **Role of ML Infrastructure** – lifecycle review, stakeholder interactions, success metrics.
2. **Environment Build Out** – Python environment, Git workflow, Docker baseline, GPU tooling.
3. **Model Serving Fundamentals** – TorchServe/TensorFlow Serving, FastAPI interface, testing.
4. **Containerization Pipeline** – Dockerfile patterns, registry publishing, image optimization.
5. **First Deployment** – Provision cloud VM, secure endpoints, implement logging/metrics.
6. **Operational Readiness** – Health checks, runbooks, incident triage basics, cost snapshot.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Environment bootstrap automation | Reproducible setup script committed | `python-strict` profile |
| Lab 2 | Containerized inference service | Docker image <2GB, tests pass | `./tools/curriculum.py run-validation python-strict` |
| Lab 3 | Cloud deployment with monitoring | Endpoint live, Prometheus metrics exposed | Infrastructure checklist review |
| Assessment | ML infrastructure concepts quiz | ≥70% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Environment automation script, containerized service, deployment manifests, quiz answer key
- **Repository Strategy**: See `curriculum/repository-strategy.yaml` (`per_role` with separate solutions repo)
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-101-foundations/solutions`
- **Validation Status**: Migrated from legacy repo; needs `python-strict` pipeline run under new harness

## Resource Plan

- **Primary References**:
  - `modules/ai-infrastructure-engineer/module-101-foundations/README.md`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-101-foundations`
- **Supplemental Resources**:
  - TorchServe and FastAPI official docs
  - CNCF landscape overview for ML infrastructure
- **Tooling Requirements**:
  - Python 3.11+, Docker 24+, NVIDIA Container Toolkit (optional), AWS/GCP CLI

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Junior modules MOD-005 (Docker), MOD-006 (Kubernetes Intro), MOD-010 (Cloud Platforms)
- **Downstream Outputs**: Enables Project PROJ-201, unlocks deeper automation in MOD-103/104
- **Risks / Mitigations**: Learners may skip environment hardening—add readiness checklist before moving forward.
