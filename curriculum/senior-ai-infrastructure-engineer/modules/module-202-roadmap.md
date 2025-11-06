# Module Roadmap

> Senior AI Infrastructure Engineer | Legacy Module 202 uplifted with 2025 distributed training practices.

## Module Overview

- **Module ID**: MOD-202
- **Module Title**: Distributed Training at Scale
- **Target Role(s)**: Senior AI Infrastructure Engineer
- **Duration**: 55 hours (18 lecture, 24 lab, 10 project, 3 assessment)
- **Prerequisites**: MOD-201, MOD-107
- **Next Module(s)**: MOD-203, MOD-204

## Cross-Role Progression

- **Builds On**: Mid-level distributed training exposure (Ray/Horovod basics) and GPU operations from MOD-107.
- **Adds New Depth**: Elastic scaling, advanced scheduling, performance diagnostics, heterogenous hardware support.
- **Shared Assets**: Reuses GPU benchmark harness from MOD-107; extends data pipeline assets from MOD-105.
- **Differentiators**: Introduces high-performance networking (RDMA/InfiniBand), checkpoint orchestration, and cost telemetry.

## Learning Objectives

- Configure Ray, Horovod, and PyTorch DDP/FSDP for multi-node, multi-GPU training with elastic scaling.
- Optimize end-to-end throughput using performance profiling, sharding strategies, and smart batching/checkpointing.
- Implement resiliency patterns (fault injection, restart orchestration, cost-aware scheduling) for production training jobs.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| distributed-training-optimization | Expert | Benchmark delta ≥ 2x over baseline | Senior AI Infrastructure Engineer |
| performance-engineering | Proficient | Profiling report + remediation plan | Senior AI Infrastructure Engineer |

## Content Outline

1. Distributed paradigms (data/model/pipeline parallelism) and workload characterization.
2. Ray architecture (autoscaler, datasets, Train/Tune integration) with GPU orchestration.
3. Horovod/FSDP best practices, gradient compression, activation checkpointing.
4. High-performance networking (InfiniBand, NCCL tuning, RDMA), storage considerations.
5. Fault tolerance (elastic training, checkpoint orchestration, job recovery automation).
6. Observability for training systems (DCGM, TensorBoard, cost telemetry dashboards).

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | Deploy elastic Ray cluster with autoscaler | Cluster adapts to load and recovers from node loss | Benchmark validation |
| Lab 2 | Horovod vs FSDP performance shootout | Submit comparative report with metrics + recommendations | Performance review |
| Lab 3 | Failure injection tabletop | Demonstrate checkpoint restore + SLA impact analysis | Incident simulation |
| Assessment | Design doc & oral defense | Pass rate ≥ 80% on evaluator rubric | Architecture review panel |

## Solutions Plan

- **Coverage**: Ray/Horovod cluster templates, benchmarking scripts, failure-injection automation, assessment rubric.
- **Repository Strategy**: Separate solutions repository per role, mapped via `curriculum/repository-strategy.yaml`.
- **Solution Path**: `modules/senior-ai-infrastructure-engineer/module-202-distributed-training/solutions` (metadata + legacy pointers).
- **Validation Status**: Requires GPU-backed CI/CD execution or documented manual validation workflow.

## Resource Plan

- **Primary References**:
  - Module README `modules/senior-ai-infrastructure-engineer/module-202-distributed-training/README.md`
  - Legacy lab notebooks and scripts in same directory
- **Supplemental Resources**:
  - Ray Train documentation, Horovod best practices, NCCL tuning guides
  - Papers on large-scale distributed training (Megatron-LM, DeepSpeed)
- **Tooling Requirements**:
  - GPU cluster (on-prem or cloud), Ray/Horovod environments, profiling tools (Nsight, PyTorch profiler)

## Quality Checklist

- [ ] Benchmark scripts reproduce ≥2x improvement narratives
- [ ] Failure injection runbooks include rollback and communication plan
- [ ] Cost telemetry integrated into dashboards with actionable thresholds
- [ ] Labs reused by PROJ-301 plan without duplication

## Dependencies & Notes

- Coordinate scheduling of GPU resources with MOD-203 to avoid contention.
- Encourage learners to log benchmark metadata to support PROJ-301 deliverables.
