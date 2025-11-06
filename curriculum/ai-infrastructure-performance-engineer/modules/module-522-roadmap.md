# Module Roadmap

> AI Infrastructure Performance Engineer | Module 522 advances from foundational profiling into CUDA/Triton kernel optimization.

## Module Overview

- **Module ID**: MOD-522
- **Module Title**: CUDA & Kernel Optimization
- **Target Role(s)**: AI Infrastructure Performance Engineer
- **Duration**: 38 hours (12 lecture, 18 lab, 4 portfolio, 4 assessment)
- **Prerequisites**: MOD-521
- **Next Module(s)**: MOD-523, MOD-524, PROJ-522

## Cross-Role Progression

- Leverages senior engineer GPU modules to avoid duplicating introductory content.
- Coordinates with ML Platform kernel libraries for shared fused operators.
- Provides optimized kernels reused by architect/performance projects evaluating hardware migrations.

## Learning Objectives

- Design, implement, and profile custom CUDA/Triton kernels for common ML operations.
- Apply optimization techniques (memory coalescing, occupancy tuning, loop unrolling) to achieve measurable speedups.
- Integrate kernels into PyTorch/TensorFlow workflows with automated testing and fallbacks.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| kernel-engineering | Proficient | Kernel optimization challenge | AI Infrastructure Performance Engineer |
| automation | Working | Integration tests & CI setup | AI Infrastructure Performance Engineer |

## Content Outline

1. **Kernel Anatomy** – thread/block configuration, shared memory usage, warp divergence.
2. **Optimization Techniques** – tiling, loop fusion, mixed precision, occupancy analysis.
3. **Tooling** – Nsight Compute deep dive, SASS inspection, Triton language patterns.
4. **Integration Strategies** – binding kernels to PyTorch/TensorFlow/Triton runtime.
5. **Testing & Validation** – numerical correctness, regression tests, fallback paths.

## Hands-On Activities

- Optimize baseline kernel using Nsight metrics to achieve defined speedup targets.
- Implement fused operator (e.g., attention, GEMM) with benchmarking harness.
- Document optimization decisions and integrate into PROJ-522 custom kernel project.

## Assessments & Evidence

- Kernel optimization challenge evaluated via benchmark improvements and code review.
- Integration pipeline demonstration showing automated tests and regression tracking.

## Shared Assets & Legacy Mapping

- Legacy source: `lessons/mod-002-cuda-programming`
- Outputs flow into MOD-524 transformer optimization and PR0J-522 custom kernel acceleration initiative.
