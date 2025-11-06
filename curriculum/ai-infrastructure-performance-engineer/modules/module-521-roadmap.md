# Module Roadmap

> AI Infrastructure Performance Engineer | Module 521 introduces GPU architecture fundamentals and baseline profiling practices.

## Module Overview

- **Module ID**: MOD-521
- **Module Title**: GPU & Performance Foundations
- **Target Role(s)**: AI Infrastructure Performance Engineer
- **Duration**: 32 hours (10 lecture, 16 lab, 4 portfolio, 2 assessment)
- **Prerequisites**: MOD-201, MOD-202
- **Next Module(s)**: MOD-522, MOD-523, PROJ-521

## Cross-Role Progression

- Builds on senior engineer distributed training modules to avoid reintroducing core GPU concepts.
- Shares telemetry instrumentation patterns with MLOps observability modules to ensure consistency.
- Provides baseline metrics consumed by ML Platform FinOps dashboards for cost tracking.

## Learning Objectives

- Explain GPU architecture, memory hierarchy, and occupancy considerations for ML workloads.
- Configure profiling toolchains (Nsight Systems/Compute, PyTorch Profiler) and interpret key metrics.
- Establish baseline benchmarking scripts and data-driven performance goals for downstream modules.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| kernel-engineering | Working | Profiling lab report | AI Infrastructure Performance Engineer |
| pipeline-automation | Working | Baseline benchmarking script | AI Infrastructure Performance Engineer |

## Content Outline

1. **GPU Architecture Primer** – SMs, warp scheduling, memory hierarchy.
2. **Profiling Tooling** – Nsight overview, PyTorch Profiler, DCGM metrics.
3. **Benchmark Design** – workload selection, reproducibility, regression tracking.
4. **Performance Metrics** – latency, throughput, utilization, cost per inference.
5. **Goal Setting** – linking benchmarks to business KPIs and FinOps reporting.

## Hands-On Activities

- Install and configure profiling toolchain; gather baseline traces for reference models.
- Build reproducible benchmarking scripts with logging and reporting.
- Translate profiling results into optimization backlog items for PROJ-521.

## Assessments & Evidence

- Profiling lab submission with annotated Nsight screenshots and improvement backlog.
- Benchmark baseline report reviewed with MLOps stakeholders.

## Shared Assets & Legacy Mapping

- Legacy source: `lessons/mod-001-performance-fundamentals`
- Outputs feed into MOD-522 kernel optimization labs and PROJ-521 performance pipeline work.
