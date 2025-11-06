# Role Research – AI Infrastructure Performance Engineer

## Mission Statement
Deliver cost-efficient, high-performance ML systems by optimizing model architectures, kernels, and deployment pipelines across the entire lifecycle.

## Key Responsibilities
- Profile training/inference workloads, identify bottlenecks, and implement targeted optimizations.
- Apply model compression (quantization, pruning, distillation) while maintaining accuracy guardrails.
- Develop and tune custom CUDA/Triton kernels, fused operators, and memory-efficient attention mechanisms.
- Design benchmarking suites, regression tests, and performance dashboards for continuous monitoring.
- Collaborate with platform/MLOps teams to productionize optimizations at scale, balancing throughput, latency, and cost.

## Success Indicators
- Latency/throughput improvements vs. baseline (e.g., 2x speedup, 40% cost reduction).
- GPU utilization increases and reduced idle time in inference clusters.
- Accuracy deltas post-optimization kept within defined tolerances.
- Repeatable benchmarking harness adopted across product teams.
- Documented best practices reused by platform, security, and leadership tracks.

## Role Evolution & Dependencies
- Builds on Senior AI Infrastructure Engineer (distributed systems) and MLOps (observability/governance) capabilities.
- Shares optimized artifacts with ML Platform Engineer (feature services, developer experience) and Architect tracks (FinOps narratives).
- Provides inputs to Security Engineer (guardrails for kernel-level changes) and Leadership roles (performance communication).

## Risks & Constraints
- Optimizations may degrade accuracy or reliability if validation is insufficient.
- Hardware diversity (A100, H100, L4, Inferentia) requires adaptable tooling.
- Benchmark drift due to dataset/model updates; need tight integration with MLOps observability.

## Research Inputs
- Legacy curriculum docs (`CURRICULUM.md`, `ROLE_REQUIREMENTS.md`, project READMEs`).
- Job postings from NVIDIA, OpenAI, Meta, AWS (October 2025) highlighting transformer and inference efficiency.
- Internal reports on inference FinOps challenges captured in prior security/platform migrations.
