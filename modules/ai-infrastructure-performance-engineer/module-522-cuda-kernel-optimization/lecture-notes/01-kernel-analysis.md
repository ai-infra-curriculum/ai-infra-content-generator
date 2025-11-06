# Lecture 01 · Kernel Analysis Techniques

## Objectives
- Interpret profiling outputs at instruction and memory hierarchy levels.
- Identify occupancy, memory throughput, and divergence issues in custom kernels.
- Prioritize optimization opportunities based on data-driven analysis.

## Key Topics
1. **Profiler Deep Dive** – Nsight Compute metrics (achieved occupancy, memory throughput, warp stall reasons).
2. **Bottleneck Classification** – Memory-bound vs compute-bound, latency vs throughput limitations.
3. **Workload Characterization** – Input sizes, tensor layouts, batch dynamics affecting performance.
4. **Diagnostic Methodology** – Hypothesis-driven profiling, experiment planning, regression tracking.

## Activities
- Review sample profiling sessions for GEMM and attention kernels, identify top stall reasons.
- Create a decision matrix mapping profiling metrics to optimization actions.
- Plan experiments to validate hypotheses, documenting expected improvements.
