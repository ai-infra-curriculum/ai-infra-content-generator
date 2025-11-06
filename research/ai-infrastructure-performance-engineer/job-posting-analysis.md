# Job Posting Analysis — AI Infrastructure Performance Engineer

## Sample Employers & Focus Areas
- **Hyperscalers / Cloud Providers** (NVIDIA, AWS, Google) emphasize GPU kernel optimization, TensorRT/vLLM expertise, and large-scale inference efficiency.
- **AI Product Companies** (OpenAI, Anthropic, Stability AI) require transformer optimization, KV-cache management, and cost/performance tuning for LLM deployment.
- **Enterprise ML Teams** (Meta, Salesforce, ServiceNow) focus on quantization, pruning, and observability to reduce inference spend across business units.

## Common Requirements
- Profiling skillset across training and inference (Nsight, PyTorch Profiler, TensorBoard).
- Model compression experience (INT8/FP8 quantization, AWQ/GPTQ, distillation, structured sparsity).
- CUDA kernel authoring/tuning, familiarity with CUTLASS and custom fused operators.
- Mastery of transformer performance tricks (FlashAttention, continuous batching, paged KV cache).
- Ability to benchmark multi-GPU/multi-node inference frameworks (TensorRT-LLM, DeepSpeed-Inference, vLLM, SGLang).
- Cost analysis + FinOps mindset to quantify ROI of optimization work.

## Tooling & Infrastructure Expectations
- Hardware: NVIDIA A100/H100/L4, emerging inference accelerators (AWS Inferentia, Trainium, Groq).
- Serving stacks: Triton Inference Server, TensorRT-LLM, vLLM, TGI, Ray Serve.
- Compiler ecosystems: TVM, XLA, PyTorch Inductor, OpenAI Triton.
- Observability: NVIDIA DCGM, Prometheus/Grafana dashboards for GPU metrics, custom perf regression suites.

## Noted Skill Gaps
- Engineers with both low-level CUDA expertise and high-level ML pipeline knowledge are rare.
- Many teams lack repeatable benchmarking harnesses to track regressions across hardware SKUs.
- KV cache and continuous batching optimizations for LLMs still under-documented; high demand for practical experience.
- Responsible AI/performance coupling (ensuring optimizations do not compromise model fairness) emerging requirement.

## Curriculum Implications
- Emphasize reproducible benchmarking workflows and data-driven decision making.
- Provide cross-role integration with MLOps/Platform for shared observability, governance, and deployment practices.
- Include advanced case studies on LLM inference and specialized hardware adoption.
- Highlight cost-performance trade-off analysis and communication skills for leadership stakeholders.
