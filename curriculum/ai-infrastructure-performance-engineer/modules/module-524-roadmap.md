# Module Roadmap

> AI Infrastructure Performance Engineer | Module 524 targets transformer and LLM optimization techniques.

## Module Overview

- **Module ID**: MOD-524
- **Module Title**: Transformer & LLM Optimization
- **Target Role(s)**: AI Infrastructure Performance Engineer
- **Duration**: 40 hours (12 lecture, 18 lab, 6 portfolio, 4 assessment)
- **Prerequisites**: MOD-522, MOD-523
- **Next Module(s)**: MOD-526, MOD-560 (LLMOps), PROJ-524

## Cross-Role Progression

- Shares LLM optimization frameworks with MLOps MOD-560 to maintain unified guardrails.
- Provides optimized components reused by ML Platform developer experience and architect responsible AI programs.
- Aligns with security track for validation of fused kernels and cache modifications.

## Learning Objectives

- Implement attention optimizations (FlashAttention, xFormers, custom kernels) and quantify gains.
- Optimize KV cache management, continuous batching, and streaming inference strategies for LLMs.
- Maintain accuracy and responsible AI commitments when applying aggressive optimizations.

## Competency Alignment

| Competency | Proficiency Target | Evidence / Assessment | Role Alignment |
|------------|--------------------|-----------------------|----------------|
| llmops | Proficient | LLM optimization lab | AI Infrastructure Performance Engineer |
| responsible-ai | Working | Accuracy & safety validation report | AI Infrastructure Performance Engineer |

## Content Outline

1. **Transformer Performance Anatomy** – attention bottlenecks, memory footprints, batching dynamics.
2. **Optimized Kernels & Libraries** – FlashAttention, DeepSpeed-Inference, TensorRT-LLM, vLLM, SGLang.
3. **Cache & Streaming Techniques** – paged attention, KV cache partitioning, continuous batching.
4. **Accuracy & Safety** – evaluation harnesses, guardrails, responsible AI validation.
5. **Case Studies** – real-world LLM optimization projects and ROI analysis.

## Hands-On Activities

- Implement attention optimization using multiple libraries and compare metrics.
- Tune continuous batching configuration for throughput/latency targets on sample workloads.
- Run accuracy and safety validation to confirm acceptable degradation.

## Assessments & Evidence

- Optimization report quantifying improvements and guardrail compliance.
- Technical presentation prepared for platform/leadership stakeholders.

## Shared Assets & Legacy Mapping

- Legacy source: `lessons/mod-004-transformer-optimization`
- Outputs feed PROJ-524 LLM inference efficiency program and inform architect LLM strategy documents.
