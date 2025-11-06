# Project Plan

> AI Infrastructure Performance Engineer | Built on legacy `projects/project-02-custom-cuda-kernels` artifacts.

## Project Overview

- **Project ID**: PROJ-522
- **Project Title**: Custom Kernel Acceleration
- **Target Role(s)**: AI Infrastructure Performance Engineer
- **Placement in Curriculum**: Follows MOD-522 and MOD-523
- **Estimated Duration**: 120 hours
- **Prerequisite Modules / Skills**: MOD-522, MOD-523
- **Related Assessments**: Kernel optimization code review, benchmarking presentation

## Learning Objectives

- Design custom CUDA/Triton kernels for performance-critical operations and integrate them into ML frameworks.
- Establish benchmarking harnesses and regression tests to validate performance gains and correctness.
- Produce documentation and enablement materials for platform teams adopting optimized kernels.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| kernel-engineering | Expert | Custom kernel implementation & benchmark | AI Infrastructure Performance Engineer |
| automation | Proficient | CI pipeline for kernel tests | AI Infrastructure Performance Engineer |
| innovation | Working | Enablement playbook | AI Infrastructure Performance Engineer |

## Key Deliverables

- Custom fused kernels with benchmarking results demonstrating target improvements.
- Automated test suite (numerical correctness, regression detection) integrated with CI/CD.
- Enablement guide for platform engineers detailing usage patterns, guardrails, and troubleshooting.

## Learning Activities & Milestones

### Guided Activities

- Profiling workload to identify kernel-level bottlenecks.
- Implementing fused kernel versions and measuring gains against baseline operations.
- Documenting integration steps, fallbacks, and best practices for adoption.

### Milestone Schedule

- Weeks 1-2: Workload analysis, kernel design, scaffolding.
- Weeks 3-4: Implementation, profiling, and iterative optimization.
- Weeks 5-6: Testing automation, documentation, and stakeholder review.

## Assessment & Validation

- Kernel code review focusing on performance, readability, and reliability.
- Benchmark comparison and regression test results validated by GPU platform stakeholders.
- Enablement session or documentation handoff to ML Platform teams.

## Legacy Alignment & Next Steps

- Legacy source: `projects/project-02-custom-cuda-kernels`
- Solutions repo: `solutions/ai-infra-performance-solutions/project-02-custom-cuda-kernels`
- Outputs feed PROJ-523 distributed inference optimization and inform architect hardware evaluations.
