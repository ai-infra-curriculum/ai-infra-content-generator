# Module 07: GPU Computing for ML - Quiz

**Time Limit:** 30 minutes
**Passing Score:** 80% (20/25 questions)
**Coverage:** GPU architecture, CUDA, optimization

---

## Section 1: GPU Fundamentals (5 questions)

### Q1. Why are GPUs faster than CPUs for ML training?

a) Higher clock speed
b) Massive parallelism with thousands of cores
c) More cache memory
d) Better cooling

**Answer:** B

---

### Q2. What does CUDA stand for?

a) Computer Unified Device Architecture
b) Compute Unified Device Architecture
c) Central Unit Data Access
d) Core Unified Development API

**Answer:** B

---

### Q3. What is the difference between a GPU and a TPU?

a) No difference
b) TPUs are specialized for tensor operations, GPUs general-purpose
c) TPUs are slower
d) GPUs don't work for ML

**Answer:** B

---

### Q4. What is VRAM?

a) Virtual RAM
b) Video RAM / GPU memory
c) Very fast RAM
d) Variable RAM

**Answer:** B

---

### Q5. What happens when model doesn't fit in GPU memory?

a) Automatic optimization
b) Out of memory error or need to use CPU
c) GPU gets more memory
d) Nothing

**Answer:** B

---

## Section 2: CUDA and cuDNN (5 questions)

### Q6. What is cuDNN?

a) CUDA Network
b) CUDA Deep Neural Network library
c) CPU library
d) Database

**Answer:** B

---

### Q7. Why must CUDA versions match between driver and toolkit?

a) No need to match
b) Compatibility and functionality requirements
c) Just a suggestion
d) Only for performance

**Answer:** B

---

### Q8. What is the NVIDIA Container Toolkit used for?

a) General containers
b) Enable GPU access in Docker containers
c) CPU optimization
d) Network configuration

**Answer:** B

---

### Q9. How do you check CUDA version on a system?

a) `cuda --version`
b) `nvidia-smi` or `nvcc --version`
c) `gpu --version`
d) `python --cuda`

**Answer:** B

---

### Q10. What is the purpose of CUDA streams?

a) Video streaming
b) Concurrent execution of operations
c) Data streaming
d) Network streams

**Answer:** B

---

## Section 3: Multi-GPU Training (5 questions)

### Q11. What is data parallelism?

a) Parallel databases
b) Split batch across GPUs, same model on each
c) Split model across GPUs
d) No parallelism

**Answer:** B

---

### Q12. What is model parallelism?

a) Training multiple models
b) Split model across GPUs
c) Same as data parallelism
d) Not possible

**Answer:** B

---

### Q13. When should you use model parallelism?

a) Always
b) When model too large for single GPU memory
c) Never
d) Only for inference

**Answer:** B

---

### Q14. What is NCCL?

a) Network protocol
b) NVIDIA Collective Communications Library for multi-GPU
c) Neural network
d) Cloud service

**Answer:** B

---

### Q15. What is gradient accumulation?

a) Storing all gradients
b) Accumulate gradients over multiple batches before update
c) Deleting gradients
d) Gradient backup

**Answer:** B

---

## Section 4: GPU Optimization (5 questions)

### Q16. What is mixed precision training?

a) Random precision
b) Using FP16 and FP32 together to speed up training
c) Low quality training
d) Multiple models

**Answer:** B

---

### Q17. What is the benefit of using FP16 (half precision)?

a) Better accuracy
b) Faster computation and lower memory usage
c) Easier debugging
d) No benefits

**Answer:** B

---

### Q18. What is tensor cores on NVIDIA GPUs?

a) CPU cores
b) Specialized hardware for matrix operations
c) Storage cores
d) Network cores

**Answer:** B

---

### Q19. How can you optimize GPU utilization?

a) Use smaller batches
b) Maximize batch size, use mixed precision, pipeline data
c) Use CPU instead
d) Add more RAM

**Answer:** B

---

### Q20. What is the purpose of GPU profiling?

a) User profiling
b) Identify performance bottlenecks
c) Security auditing
d) Cost tracking

**Answer:** B

---

## Section 5: Cloud GPUs and Best Practices (5 questions)

### Q21. Which AWS instance type provides GPU access?

a) t3.large
b) p3, p4, g4 instances
c) m5.xlarge
d) c5.4xlarge

**Answer:** B

---

### Q22. What is a GPU spot instance used for?

a) Critical production workloads
b) Cost-effective training with interruption tolerance
c) Real-time inference
d) Databases

**Answer:** B

---

### Q23. What is the typical cost difference between GPU and CPU instances?

a) Same cost
b) GPUs 3-10x more expensive per hour
c) CPUs more expensive
d) GPUs are free

**Answer:** B

---

### Q24. Best practice for GPU resource allocation in Kubernetes?

a) Oversubscribe GPUs
b) Request exact GPU count needed, use limits
c) Never use GPUs in Kubernetes
d) Share GPUs randomly

**Answer:** B

---

### Q25. What is Multi-Instance GPU (MIG)?

a) Multiple GPU purchases
b) Partition single GPU into multiple instances
c) Running multiple models
d) Cloud service

**Answer:** B

---

## Answer Key

1. B   2. B   3. B   4. B   5. B
6. B   7. B   8. B   9. B   10. B
11. B  12. B  13. B  14. B  15. B
16. B  17. B  18. B  19. B  20. B
21. B  22. B  23. B  24. B  25. B

---

## Scoring

- **23-25 correct (92-100%)**: Excellent! GPU expert
- **20-22 correct (80-88%)**: Good! Ready for production
- **18-19 correct (72-76%)**: Fair. Review concepts
- **Below 18 (< 72%)**: Review module materials

---

**Next Module:** Module 08 - Monitoring and Observability
