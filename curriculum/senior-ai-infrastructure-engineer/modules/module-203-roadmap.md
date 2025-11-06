# Module Roadmap

> Senior AI Infrastructure Engineer | Module 203 (legacy GPU optimization) refreshed for 2025 GPU platforms.

## Module Overview

- **Module ID**: MOD-203
- **Module Title**: Advanced GPU Computing & Optimization
- **Target Role(s)**: Senior AI Infrastructure Engineer
- **Duration**: 60 hours (18 lecture, 26 lab, 12 project, 4 assessment)
- **Prerequisites**: MOD-202
- **Next Module(s)**: MOD-204, MOD-302 (project)

## Cross-Role Progression

- **Builds On**: MOD-107 GPU fundamentals and MOD-202 distributed training.
- **Adds New Depth**: CUDA kernel authoring, MIG/vGPU strategies, Nsight-driven performance engineering.
- **Shared Assets**: Leverages benchmarks from MOD-202; informs optimization work in PROJ-301 and PROJ-302.
- **Differentiators**: Heavy focus on measurable performance gains and hardware-aware automation.

## Learning Objectives

- Author and optimize CUDA kernels, integrating them into ML workloads.
- Use Nsight Systems/Compute, DCGM, and custom telemetry to diagnose GPU bottlenecks.
- Design GPU fleet strategy (MIG/vGPU, scheduling policies) that balances performance and cost.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| gpu-infrastructure-advanced | Expert | CUDA optimization lab (≥2x improvement) | Senior AI Infrastructure Engineer |
| performance-engineering | Expert | Profiling dossier + remediation plan | Senior AI Infrastructure Engineer |

## Content Outline

1. GPU architecture deep dive (SM layout, memory hierarchy, interconnects).
2. CUDA programming practices (kernels, streams, shared memory, warp optimization).
3. TensorRT/TensorRT-LLM integration, Triton inference server tuning.
4. GPU virtualization (MIG, vGPU), scheduling policies, capacity planning.
5. Profiling workflow (Nsight Systems/Compute, DCGM exporters, flamegraphs).
6. Automation for GPU maintenance (driver rollout, firmware management, health remediation).

## Practical Components

| Asset | Description | Completion Criteria | Linked Validation |
|-------|-------------|---------------------|-------------------|
| Lab 1 | CUDA kernel optimization | Benchmark shows ≥2x speedup vs baseline | Performance regression tests |
| Lab 2 | GPU fleet scheduler | MIG/vGPU policy codified, monitored via DCGM | Automation validation |
| Lab 3 | Inference tuning | TensorRT pipeline meets latency SLA | Load testing harness |
| Assessment | Performance review board | Score ≥ 80% on peer panel rubric | Panel evaluation |

## Solutions Plan

- **Coverage**: CUDA sample solutions, scheduling automation scripts, DCGM dashboards, assessment rubrics.
- **Repository Strategy**: Solutions tracked in `ai-infra-senior-engineer-solutions` with metadata linking.
- **Solution Path**: `modules/senior-ai-infrastructure-engineer/module-203-gpu-computing/solutions`.
- **Validation Status**: Requires GPU access; provide simulation fallback guidance for CI.

## Resource Plan

- **Primary References**: Module README and lab assets in module directory.
- **Supplemental Resources**: NVIDIA DLI courses, Nsight documentation, TensorRT-LLM guides.
- **Tooling Requirements**: CUDA toolkit, Nsight suite, TensorRT/Triton, GPU cluster w/ MIG support.

## Quality Checklist

- [ ] Benchmark improvements documented with reproducible scripts.
- [ ] GPU scheduling policies tested against resource contention scenarios.
- [ ] Observability includes GPU health, errors, and utilization SLOs.
- [ ] Outputs feed into PROJ-301 and PROJ-302 planning docs.

## Dependencies & Notes

- Coordinate hardware allocation with program operations; record fallback instructions for learners without direct GPU access.
