# Module Roadmap

> AI Infrastructure Performance Engineer | Module 526 focuses on distributed inference and scaling strategies.

## Module Overview

- **Module ID**: MOD-526
- **Module Title**: Distributed Inference & Scaling
- **Target Role(s)**: AI Infrastructure Performance Engineer
- **Duration**: 40 hours (12 lecture, 18 lab, 6 portfolio, 4 assessment)
- **Prerequisites**: MOD-524, MOD-525
- **Next Module(s)**: MOD-527, MOD-528, PROJ-523

## Cross-Role Progression

- Builds on senior engineer distributed systems knowledge and MLOps automation pipelines.
- Shares autoscaling and batching assets with ML Platform and MLOps roles to ensure consistent operations.
- Provides data for architect FinOps and reliability programs.

## Learning Objectives

- Architect multi-GPU and multi-node inference deployments using tensor/sequence/pipeline parallelism.
- Implement continuous batching, dynamic routing, and autoscaling strategies tied to performance KPIs.
- Instrument distributed inference for observability and incident response coordination.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| workflow-orchestration | Proficient | Distributed inference pipeline | AI Infrastructure Performance Engineer |
| service-reliability | Proficient | Autoscaling and incident playbook | AI Infrastructure Performance Engineer |

## Content Outline

1. **Parallelism Strategies** – tensor, pipeline, expert, and sequence parallelism trade-offs.
2. **Serving Frameworks** – DeepSpeed-Inference, TensorRT-LLM, vLLM, Ray Serve, Triton.
3. **Batching & Routing** – continuous batching, dynamic scheduling, traffic shaping.
4. **Autoscaling & Capacity Planning** – GPU scheduling, spot/on-demand mix, FinOps considerations.
5. **Observability & Incident Response** – metrics, tracing, chaos experiments.

## Hands-On Activities

- Deploy a distributed inference stack with selected parallelism mode and benchmark improvements.
- Implement continuous batching and autoscaling policies with regression tests.
- Document incident response plan aligned with production operations module obligations.

## Assessments & Evidence

- Distributed inference pipeline demonstration and metrics review.
- Incident response tabletop results and remediation backlog.

## Shared Assets & Legacy Mapping

- Legacy source: `lessons/mod-006-distributed-inference`
- Outputs support PROJ-523 distributed inference platform optimization and inform ML Platform/MLOps deployments.
