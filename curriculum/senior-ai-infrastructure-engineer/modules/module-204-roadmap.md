# Module Roadmap

> Senior AI Infrastructure Engineer | Module 204 aligns legacy inference optimization content with modern LLM serving.

## Module Overview

- **Module ID**: MOD-204
- **Module Title**: Advanced Model Optimization & Inference
- **Target Role(s)**: Senior AI Infrastructure Engineer
- **Duration**: 55 hours (16 lecture, 24 lab, 10 project, 5 assessment)
- **Prerequisites**: MOD-203, MOD-106
- **Next Module(s)**: MOD-205, MOD-302

## Cross-Role Progression

- **Builds On**: Mid-level LLM infrastructure (MOD-110) and senior GPU optimization (MOD-203).
- **Adds New Depth**: TensorRT-LLM, quantization/pruning pipelines, inference cost governance.
- **Shared Assets**: Uses datasets/models from PROJ-201/302; extends automation from MOD-206.
- **Differentiators**: Emphasizes measured throughput/latency improvements and production deployment strategies.

## Learning Objectives

- Optimize LLM/deep learning inference through TensorRT-LLM, ONNX, quantization, and pruning workflows.
- Design high-throughput serving pipelines (vLLM, Triton, TGI) with autoscaling and canary rollouts.
- Implement monitoring, guardrails, and cost telemetry specific to inference workloads.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| inference-optimization | Expert | Latency/cost reduction case study | Senior AI Infrastructure Engineer |
| llm-infrastructure | Proficient | Serving pipeline runbook & automation | Senior AI Infrastructure Engineer |

## Content Outline

1. Optimization frameworks (TensorRT-LLM, ONNX Runtime, DeepSpeed inference).
2. Quantization & pruning strategies (GPTQ, AWQ, sparsity techniques).
3. Serving architectures (vLLM, TGI, Triton) with GPU-aware autoscaling.
4. Observability & guardrails (token-level metrics, safety filters, fallback paths).
5. Cost management (spot/preemptible, rightsizing, batch scheduling).
6. Deployment automation (blue/green, canary, rollback pipelines integrating policy checks).

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Quantization + benchmarking | Document ≥40% cost reduction or 2x throughput | Benchmark validation |
| Lab 2 | Multi-model serving pipeline | Deployed with autoscaling + guardrails | Load & safety testing |
| Lab 3 | Rollout automation | Blue/green pipeline with automated rollback triggers | CI/CD validation |
| Assessment | Optimization playbook | Approved by peer review panel | Architecture panel |

## Solutions Plan

- **Coverage**: Optimization notebooks, deployment manifests, cost calculator, assessment rubric.
- **Repository Strategy**: Solutions stored separately (`ai-infra-senior-engineer-solutions`) with metadata linkbacks.
- **Solution Path**: `modules/senior-ai-infrastructure-engineer/module-204-model-optimization/solutions`.
- **Validation Status**: Requires inference benchmarking pipeline; integrate with load testing harness.

## Resource Plan

- **Primary References**: Module README and lab resources in module directory.
- **Supplemental Resources**: TensorRT-LLM docs, vLLM performance guides, cost optimization playbooks.
- **Tooling Requirements**: GPU cluster, TensorRT/Triton, vLLM/TGI, load testing tools (Locust/K6).

## Quality Checklist

- [ ] Optimization outcomes documented with reproducible scripts and metrics.
- [ ] Guardrail testing includes adversarial prompts and failure injection.
- [ ] Rollout automation integrates compliance checks from MOD-208/209.
- [ ] Deliverables connect to PROJ-302 milestone checkpoints.

## Dependencies & Notes

- Encourage learners to align optimization targets with business SLAs defined for PROJ-302.
