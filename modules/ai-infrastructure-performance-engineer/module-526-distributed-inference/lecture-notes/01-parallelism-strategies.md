# Lecture 01 · Parallelism Strategies

## Objectives
- Compare tensor, pipeline, sequence, and expert parallelism approaches for inference workloads.
- Select appropriate strategies based on latency, throughput, memory, and hardware constraints.
- Plan migration paths from single-node deployments to distributed architectures.

## Key Topics
1. **Parallelism Overview** – data/model/pipeline parallelism for inference vs training.
2. **Framework Support** – TensorRT-LLM, DeepSpeed-Inference, vLLM, Ray Serve.
3. **Trade-offs** – latency vs throughput, cross-node communication, memory footprint.
4. **Hybrid Strategies** – combining techniques (tensor + pipeline, mixture-of-experts) for workload-specific goals.

## Activities
- Analyze workload characteristics and map to recommended parallelism strategies.
- Evaluate communication overhead/benefits using profiling tools.
- Draft migration checklist for transitioning from single-node to distributed inference.
