# Migration Notes – AI Infrastructure Performance Engineer

## Legacy Source Repositories
- Learning: `/home/claude/ai-infrastructure-project/repositories/learning/ai-infra-performance-learning`
- Solutions: `/home/claude/ai-infrastructure-project/repositories/solutions/ai-infra-performance-solutions`

## Legacy Module Inventory (Lessons)
| Legacy Module | Proposed New ID | Notes |
| --- | --- | --- |
| mod-001-gpu-fundamentals | MOD-521 | Establishes GPU architecture, CUDA basics, and profiling primers. |
| mod-002-cuda-programming | MOD-522 | Deep dive on CUDA kernel development and optimization. |
| mod-003-performance-profiling / mod-003-profiling-optimization | MOD-523 | Consolidate profiling content (Nsight, PyTorch Profiler). |
| mod-004-transformer-optimization | MOD-524 | Focus on attention optimizations, FlashAttention, KV cache tuning. |
| mod-005-model-compression | MOD-525 | Quantization, pruning, distillation patterns with accuracy guardrails. |
| mod-006-distributed-inference | MOD-526 | Multi-GPU, tensor/sequence parallelism, continuous batching. |
| mod-007-production-deployment | MOD-527 | Production monitoring, autoscaling, cost-performance trade-offs. |
| mod-008-advanced-topics / mod-008-hardware-acceleration | MOD-528 | Specialized hardware (TPUs, Inferentia, Trainium), compiler stacks (TVM, XLA). |

Duplicates such as `01-gpu-fundamentals` etc. are older exports; prefer `mod-00x` naming for consistent mapping.

## Legacy Project Inventory
| Legacy Project | Proposed New ID | Notes |
| --- | --- | --- |
| project-01-model-optimization | PROJ-521 | Baseline optimization pipeline; combines profiling + quantization. |
| project-02-custom-cuda-kernels | PROJ-522 | Custom kernel development and benchmarking harness. |
| project-03-distributed-inference | PROJ-523 | Multi-GPU / multi-node inference scaling project. |
| project-03-llm-inference | PROJ-524 | LLM serving optimization (KV cache, continuous batching); leverage latest content. |

Archive older directories (`project-01-model-compression`, `project-02-gpu-optimization`) as historical references unless additional scope is needed.

## Solution Assets
- Performance benchmarking suites and scripts in solutions repo; need integration with new validation profiles.
- CUDA kernel samples, Nsight reports, and cost benchmarking dashboards to copy into module/project solution folders.
- Housekeeping reports highlight missing documentation for production deployment scenario.

## Cross-Role Alignment Goals
- Build on Senior AI Infrastructure Engineer MOD-202/MOD-203 for distributed training foundations.
- Share observability dashboards and governance guardrails with MLOps (MOD-553, MOD-557) and ML Platform roles to avoid duplication.
- Provide optimized artifacts consumed by Architect/Principal initiatives (enterprise performance/FinOps roadmaps).

## Outstanding Questions
1. Determine exact number of modules (8 vs 10). Current mapping assumes 8; confirm if additional FinOps/performance governance modules required.
2. Select validation profile mix (likely `python-strict` for CUDA-heavy modules, plus custom performance checks).
3. Decide on automation for benchmarking (CI pipelines vs manual instructions) before finalizing solution documentation.
