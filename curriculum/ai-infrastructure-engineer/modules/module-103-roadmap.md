# Module Roadmap

> Refined from the legacy AI Infrastructure Engineer containerization curriculum.

## Module Overview

- **Module ID**: MOD-103
- **Module Title**: Containerization & Docker for ML
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 35 hours (12 lecture, 16 lab, 5 project, 2 assessment)
- **Prerequisites**: MOD-101, MOD-005
- **Next Module(s)**: MOD-104, MOD-107

## Cross-Role Progression

- **Builds On** (modules/roles): Junior MOD-005 Docker & Containerization labs
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): Base Dockerfiles and Compose templates from junior curriculum reused as starting points
- **Differentiators** (role-specific emphasis): Production optimization, GPU integration, registry automation

## Learning Objectives

- Build production-grade Docker images for ML applications with multi-stage techniques.
- Optimize image size and runtime performance, including GPU support.
- Manage networking, volumes, and secrets for multi-service ML stacks.
- Use Docker Compose to orchestrate inference, data, and monitoring services.
- Publish images to registries with automated scanning and promotion workflows.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| Container engineering | Proficient | Optimized image and Compose stack review | AI Infrastructure Engineer |
| Automation & tooling | Working | Registry automation scripts | AI Infrastructure Engineer |
| GPU operations | Working | GPU-enabled container lab | AI Infrastructure Engineer |

## Content Outline

1. **Containers in ML Infrastructure** – benefits, trade-offs, security considerations.
2. **Docker Architecture Deep Dive** – engine, storage drivers, networking modes.
3. **Writing Efficient Dockerfiles** – multi-stage builds, caching, build args, secrets.
4. **Managing Services with Compose** – multiple containers, health checks, overrides.
5. **Image Distribution** – registries (Docker Hub/ECR/GCR), tagging strategies, scanning.
6. **GPU Support** – NVIDIA Container Toolkit, runtime configuration, performance tuning.
7. **Testing & Troubleshooting** – CI integration, linting, debugging containerized ML services.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Multi-stage Dockerfile optimization | Image size reduced ≥30%; automated tests pass | `python-strict` linting |
| Lab 2 | Docker Compose ML stack | Services healthy; logs aggregated; README updated | Compose integration test |
| Lab 3 | GPU-enabled container deployment | GPU device exposed; benchmark target met | GPU validation script |
| Assessment | Containerization quiz | ≥80% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Optimized Dockerfiles, Compose manifests, GPU deployment guide, quiz key
- **Repository Strategy**: `per_role` with dedicated solutions repo
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-103-containerization/solutions`
- **Validation Status**: Needs container security scan and GPU benchmarking in CI

## Resource Plan

- **Primary References**:
  - Module README materials in `modules/ai-infrastructure-engineer/module-103-containerization`
  - Legacy content at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-103-containerization`
- **Supplemental Resources**:
  - Docker official production guidelines
  - NVIDIA Container Toolkit documentation
- **Tooling Requirements**:
  - Docker 24+, Docker Compose v2, NVIDIA Container Toolkit (for GPU labs), Trivy/Grype scanners

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: Junior container fundamentals (MOD-005) and environment from MOD-101
- **Downstream Outputs**: Required for Kubernetes deployment (MOD-104) and GPU operations (MOD-107)
- **Risks / Mitigations**: Learners may skip security scans—make Trivy scan a gated requirement.
