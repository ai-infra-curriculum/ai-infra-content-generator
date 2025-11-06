# Lecture 01 · GPU Architecture Overview

## Objectives
- Review GPU components (SMs, warps, schedulers, memory hierarchy) and their impact on ML workloads.
- Identify common bottlenecks caused by memory access patterns, occupancy limits, and warp divergence.
- Connect architectural concepts to profiling metrics learners will encounter in subsequent labs.

## Key Topics
1. **Compute Units** – Streaming multiprocessors, warp scheduling, latency hiding strategies.
2. **Memory Hierarchy** – Registers, shared memory, L2 cache, global memory, unified memory.
3. **Occupancy & Utilization** – Theoretical occupancy, active warps, latency hiding heuristics.
4. **Common Bottlenecks** – Memory-bound kernels, launch configuration issues, host-device transfers.
5. **Profiling Signals** – Mapping Nsight Systems/Compute metrics back to architecture behaviors.

## Activities
- Examine architecture diagrams for current GPU generations (A100, H100, L4). Annotate differences relevant to ML workloads.
- Analyze sample profiling traces to locate architectural bottlenecks.
- Draft hypotheses on how kernel or workload changes might address observed constraints.
