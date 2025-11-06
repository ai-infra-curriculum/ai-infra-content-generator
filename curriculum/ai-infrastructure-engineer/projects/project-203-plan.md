# Project Plan

> Migrated from legacy AI Infrastructure Engineer Project 3 with updated sequencing, validation, and cross-role hooks.

## Project Overview

- **Project ID**: PROJ-203
- **Project Title**: Production LLM Deployment
- **Target Role(s)**: AI Infrastructure Engineer
- **Placement in Curriculum**: Final capstone after MOD-104, MOD-107, MOD-109, MOD-110
- **Estimated Duration**: 50 hours
- **Prerequisite Modules / Skills**: MOD-104 Kubernetes, MOD-107 GPU Ops, MOD-109 IaC, MOD-110 LLM Infrastructure
- **Related Assessments**: LLM deployment readiness review, cost optimization audit, runbook tabletop

## Learning Objectives

- Deploy an enterprise-ready LLM/RAG platform using infrastructure-as-code and modern serving frameworks.
- Optimize inference latency, GPU utilization, and cost through quantization, batching, and autoscaling.
- Implement safety guardrails, observability, and incident response processes specific to LLM workloads.
- Produce business-ready documentation and runbooks enabling handoff to platform/ops teams.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| LLM infrastructure | Proficient | Deployment demo, optimization benchmarks | AI Infrastructure Engineer |
| Performance optimization | Proficient | Latency/cost report, load-test results | AI Infrastructure Engineer |
| Reliability & safety | Working | Guardrail configuration, incident drills | AI Infrastructure Engineer |

## Project Narrative

Your company is launching an internal assistant powered by open-source LLMs. As the infrastructure owner, you must:

1. Provision the serving environment (GPU nodes, networking, storage) via Terraform/Helm.
2. Deploy a 7B open-source model using vLLM or text-generation-inference with autoscaling enabled.
3. Integrate a RAG pipeline (vector database, embeddings service, orchestration layer) with latency targets <500ms P95.
4. Implement safety guardrails (content moderation, rate limiting) and observability across tokens, requests, and costs.
5. Deliver runbooks, escalation procedures, and cost-management scripts tailored to the platform.

## Deliverables

- IaC bundles (Terraform + Helm charts) for serving, vector database, and observability stacks.
- RAG service implementation (API gateway, retrieval layer, prompt orchestration) with automated tests.
- Optimization toolkit (quantization configs, batching strategy, load testing results).
- Monitoring/alerting dashboards (Prometheus/Grafana, tracing), safety guardrail policies, compliance documentation.
- Cost analysis, deployment report, and post-launch operations plan.

## Constraints & Assumptions

- Model must run on GPUs with quantization strategy documented; fallback CPU path optional but documented.
- Use vector database approved in repository strategy (Weaviate/PGVector) to encourage reuse across roles.
- Deployment must support blue/green or canary releases with rollback instructions.
- Observability must capture token usage, latency, error rate, and safety events.

## Learner Experience

| Phase | Description | Estimated Time | Instructor Touchpoints |
|-------|-------------|----------------|------------------------|
| Architect | Design target architecture, select serving stack, plan optimization backlog | 6h | Architecture review +
security sign-off |
| Implement | Build infrastructure, deploy model + RAG pipeline, integrate monitoring/safety | 34h | Weekly demo + async check-ins |
| Optimize & Launch | Tune performance, run load tests, finalize cost + runbooks, launch review | 10h | Launch readiness review |

## Solutions & Validation

- **Solutions Path**: `projects/ai-infrastructure-engineer/project-103-llm-deployment/solutions`
- **Validation Profiles**: `python-strict`, load-testing harness, IaC policy checks
- **Automation Hooks**: See `.github/workflows/llm-deploy.yml` and `Makefile` in the solutions repo for reproducible validation

## Risks & Mitigations

- **GPU resource contention**: Include scheduling strategies and fallback plans; verify quotas early.
- **Latency regressions**: Load-test required before sign-off; enforce performance budget in CI.
- **Safety gaps**: Provide red-team checklist and require guardrail tests ahead of launch.

## References & Inspiration

- Legacy project docs at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/projects/project-103-llm-deployment`
- vLLM, text-generation-inference, and LangChain deployment guides
- FinOps frameworks for AI workloads

## Sign-off

- **Author**: Curriculum Migration Team
- **Reviewer(s)**: Principal AI Infrastructure Engineer (pending)
- **Date Approved**: Pending validation run
