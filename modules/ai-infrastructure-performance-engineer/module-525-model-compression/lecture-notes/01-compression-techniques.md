# Lecture 01 · Compression Techniques Overview

## Objectives
- Survey compression strategies (quantization, pruning, distillation, low-rank adaptation) and when to apply them.
- Understand tooling (TensorRT, ONNX Runtime, bitsandbytes, AWQ, GPTQ) for different workloads.
- Plan experiments to balance performance gains with acceptable accuracy degradation.

## Key Topics
1. **Quantization Variants** – post-training vs QAT, FP8/INT8, AWQ, GPTQ, AutoAWQ.
2. **Pruning Strategies** – structured/unstructured pruning, sparsity-aware kernels, LoRA/QLoRA adaptations.
3. **Distillation** – student-teacher frameworks, loss functions, evaluation metrics.
4. **Low-Rank Adaptations** – LoRA, adapters, and synergy with quantization.
5. **Experiment Design** – baselines, metrics, tolerance thresholds, risk assessment.

## Activities
- Map compression strategy to target workload characteristics and business goals.
- Review sample experiment logs, identify successful vs unsuccessful attempts.
- Draft plan for integrating compression into optimization pipeline with validation checkpoints.
