# Module Roadmap

> Final module synthesized from the legacy LLM infrastructure curriculum with focus on production deployment.

## Module Overview

- **Module ID**: MOD-110
- **Module Title**: LLM Infrastructure & Optimization
- **Target Role(s)**: AI Infrastructure Engineer
- **Duration**: 40 hours (14 lecture, 18 lab, 6 project, 2 assessment)
- **Prerequisites**: MOD-106, MOD-107, MOD-109
- **Next Module(s)**: PROJ-203 (Production LLM Deployment)

## Cross-Role Progression

- **Builds On** (modules/roles): MLOps pipelines (MOD-106), GPU operations (MOD-107), IaC automation (MOD-109)
- **Adds New Depth For** (roles): AI Infrastructure Engineer
- **Shared Assets** (reuse / adapt): RAG starter kit referenced by senior/principal curricula
- **Differentiators** (role-specific emphasis): Model serving frameworks, cost/performance balancing, production RAG integration

## Learning Objectives

- Deploy open-source LLMs (Llama, Mistral) using modern inference frameworks (vLLM, TGI).
- Implement retrieval-augmented generation pipelines with vector databases and orchestration frameworks.
- Optimize inference latency and throughput via quantization, batching, and GPU scheduling.
- Instrument and monitor LLM workloads for reliability, safety, and cost controls.
- Package LLM deployment with IaC and CI/CD, including rollback and blue/green strategies.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| LLM infrastructure | Proficient | End-to-end deployment demo | AI Infrastructure Engineer |
| Performance optimization | Proficient | Latency/cost optimization report | AI Infrastructure Engineer |
| Reliability & safety | Working | Monitoring + guardrail checklist | AI Infrastructure Engineer |

## Content Outline

1. **LLM Workload Landscape** – model families, serving challenges, sizing considerations.
2. **Serving Frameworks** – vLLM, text-generation-inference, DeepSpeed Inference, Triton.
3. **RAG Pipelines** – vector store selection, embedding services, orchestration (LangChain/LlamaIndex).
4. **Optimization Techniques** – quantization (GGML, GPTQ), batching, caching, speculative decoding.
5. **Reliability & Observability** – tracing tokens, safety filters, user feedback loops.
6. **Cost Management** – GPU sizing, autoscaling, serverless/offloading strategies.
7. **Packaging & Deployment** – IaC modules, CI/CD pipelines, rollout strategies, rollback readiness.

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Deploy open-source LLM with vLLM | Endpoint live with auth; baseline throughput recorded | Load testing script |
| Lab 2 | Build RAG pipeline | Vector DB integrated; evaluation metrics captured | RAG integration tests |
| Lab 3 | Optimize latency & cost | Achieve target latency (<500ms) and cost report | Optimization summary review |
| Assessment | LLM infrastructure quiz | ≥80% score | Quiz auto-grader |

## Solutions Plan

- **Solution Coverage Required**: Terraform + Helm deployments, RAG reference implementation, optimization notebooks, quiz key
- **Repository Strategy**: `per_role`; solutions stored separately due to proprietary configs
- **Solution Path / Repo**: `modules/ai-infrastructure-engineer/module-110-llm-infrastructure/solutions`
- **Validation Status**: Requires GPU-capable test run plus load/latency validation in CI or manual review

## Resource Plan

- **Primary References**:
  - Module README in `modules/ai-infrastructure-engineer/module-110-llm-infrastructure`
  - Legacy lessons at `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-engineer-learning/lessons/mod-110-llm-infrastructure`
- **Supplemental Resources**:
  - vLLM, TGI, and LangChain/LlamaIndex documentation
  - Industry cost optimization case studies for LLM deployment
- **Tooling Requirements**:
  - GPU instances, vLLM/TGI Docker images, vector database (Weaviate/Pinecone/PGVector), load testing tool (Locust/K6)

## Quality Checklist

- [x] Module objectives map to competency framework
- [x] Hands-on assets cover key skills
- [ ] Observability/security considerations included
- [ ] Automation examples validated by scripts
- [ ] Assessment rubric aligned with learning objectives
- [ ] Solutions validated per repository strategy
- [ ] Module advances prior-role content without duplicating material

## Dependencies & Notes

- **Upstream Inputs**: MLOps (MOD-106), GPU operations (MOD-107), IaC (MOD-109)
- **Downstream Outputs**: Capstone Project PROJ-203 and cross-role specialized tracks
- **Risks / Mitigations**: Hardware expense—provide local quantized model option and recorded demo for learners without GPU access.
