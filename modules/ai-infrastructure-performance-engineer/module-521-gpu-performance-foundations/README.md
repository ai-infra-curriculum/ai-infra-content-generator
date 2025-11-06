# MOD-521 · GPU & Performance Foundations

Introduces GPU architecture, memory hierarchy, and baseline profiling practices necessary for advanced optimization work.

## Learning Goals
- Understand GPU execution model (SMs, warps, occupancy) and implications for ML workloads.
- Configure profiling toolchains (Nsight Systems/Compute, PyTorch Profiler) to capture actionable metrics.
- Establish reproducible benchmarking scripts that feed performance backlog prioritization.

## Legacy Source
- `learning/ai-infra-performance-learning/lessons/mod-001-performance-fundamentals`

## Cross-Role Integration
- Shares telemetry instrumentation patterns with MLOps observability modules to avoid duplication.
- Provides baseline metrics referenced by ML Platform FinOps dashboards and architect cost reports.
- Sets expectations for accuracy validation reused when compression or kernel changes are introduced later.
