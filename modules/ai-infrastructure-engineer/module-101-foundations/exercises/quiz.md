# Module 01: Foundations of ML Infrastructure - Quiz

**Duration:** 60 minutes
**Total Questions:** 30
**Passing Score:** 70% (21 correct answers)

## Instructions

- Read each question carefully
- Select the best answer for multiple choice questions
- Some questions may have multiple correct answers (marked with [Multiple Choice])
- Short answer questions should be 2-3 sentences
- Answers and explanations are provided at the end

---

## Section 1: ML Infrastructure Fundamentals (Questions 1-5)

### Question 1
What is the primary role of an ML Infrastructure Engineer?

A) Training machine learning models to achieve highest accuracy
B) Building and maintaining systems that enable ML model deployment and operation
C) Conducting research on new ML algorithms
D) Analyzing data to extract business insights

---

### Question 2
Which of the following is NOT a stage in the ML lifecycle?

A) Data Collection
B) Model Training
C) Customer Acquisition
D) Model Monitoring

---

### Question 3
What is the main difference between MLOps and traditional DevOps?

A) MLOps uses different programming languages
B) MLOps includes data and model versioning in addition to code versioning
C) MLOps doesn't require CI/CD pipelines
D) MLOps only works in cloud environments

---

### Question 4
**[Short Answer]**
Explain why model retraining is necessary in production ML systems.

_Your answer:_

---

### Question 5
Which salary range is typical for an entry-level AI Infrastructure Engineer in the US?

A) $50k - $80k
B) $107k - $230k
C) $300k - $500k
D) $40k - $60k

---

## Section 2: ML Frameworks (Questions 6-10)

### Question 6
What is the main difference between PyTorch and TensorFlow computation graphs?

A) PyTorch uses static graphs, TensorFlow uses dynamic graphs
B) PyTorch uses dynamic graphs, TensorFlow traditionally uses static graphs
C) Both use the same type of computation graphs
D) Neither framework uses computation graphs

---

### Question 7
Which method is recommended for saving PyTorch models in production?

A) `torch.save(model, 'model.pth')` - save entire model
B) `torch.save(model.state_dict(), 'model.pth')` - save state dictionary
C) Copy the model file manually
D) Use pickle to serialize the model

---

### Question 8
Why is it critical to call `model.eval()` before inference in PyTorch?

A) It makes the model run faster
B) It changes behavior of layers like dropout and batch normalization
C) It's not actually necessary, just a convention
D) It loads the model weights

---

### Question 9
**[Multiple Choice]**
Which of the following techniques can optimize model inference? (Select all that apply)

A) Model quantization (FP32 → INT8)
B) TorchScript compilation
C) Batch inference
D) Running inference in training mode
E) Model warm-up

---

### Question 10
What is ONNX and why is it useful?

A) A Python framework for training models
B) A universal model format for framework-agnostic deployment
C) A cloud service for hosting models
D) A type of neural network architecture

---

## Section 3: Cloud Computing (Questions 11-15)

### Question 11
Which cloud service model gives you the most control over the infrastructure?

A) SaaS (Software as a Service)
B) PaaS (Platform as a Service)
C) IaaS (Infrastructure as a Service)
D) FaaS (Function as a Service)

---

### Question 12
What is a spot/preemptible instance and when should you use it?

A) A premium instance type with guaranteed availability
B) A discounted instance that can be terminated with short notice, suitable for non-critical workloads
C) An instance that only runs during business hours
D) A small instance type for testing

---

### Question 13
**[Multiple Choice]**
Which of the following are valid strategies for cloud cost optimization? (Select all that apply)

A) Right-sizing instances based on actual usage
B) Using spot/preemptible instances for batch workloads
C) Running all resources 24/7 for consistency
D) Using reserved instances for steady workloads
E) Auto-shutting down development environments after hours

---

### Question 14
What is the typical cost structure for cloud GPU instances?

A) Free for ML workloads
B) ~$1-5 per hour depending on GPU type
C) ~$100-200 per month fixed rate
D) Pay only for successful inferences

---

### Question 15
Which cloud provider is generally considered best for ML/AI workloads?

A) AWS (most mature, widest service range)
B) GCP (ML-focused, TPU access, Kubernetes)
C) Azure (enterprise integration)
D) All three are equally suitable depending on requirements

---

## Section 4: Model Serving (Questions 16-20)

### Question 16
What is the main difference between training and serving workloads?

A) Training is faster than serving
B) Training optimizes for accuracy, serving optimizes for latency and throughput
C) Training uses GPUs, serving always uses CPUs
D) There is no difference

---

### Question 17
What is batch inference and what are its benefits?

A) Training multiple models at once; saves time
B) Processing multiple inference requests together; improves GPU utilization and throughput
C) Serving multiple model versions; enables A/B testing
D) Storing predictions in batches; reduces database load

---

### Question 18
In a blue-green deployment strategy, what happens during the switch?

A) Traffic is gradually shifted from old to new version
B) Traffic is instantly switched from old version to new version
C) Both versions serve traffic permanently
D) Old version is deleted before new version starts

---

### Question 19
**[Short Answer]**
Explain why model warm-up is important before serving production traffic.

_Your answer:_

---

### Question 20
Which deployment strategy allows testing a new model with real traffic without impacting users?

A) Blue-green deployment
B) Canary deployment
C) Shadow deployment
D) Rolling update

---

## Section 5: Docker (Questions 21-25)

### Question 21
What is the primary benefit of multi-stage Docker builds?

A) Faster build times
B) Smaller final image size by excluding build dependencies
C) Better security by using multiple base images
D) Support for multiple programming languages

---

### Question 22
Why should you run containers as non-root users?

A) Non-root users have better performance
B) It's a security best practice to minimize potential damage from container escape
C) Root users cannot access network ports
D) Docker requires non-root users

---

### Question 23
**[Multiple Choice]**
Which Dockerfile instructions affect layer caching? (Select all that apply)

A) FROM
B) RUN
C) COPY
D) CMD
E) ENV

---

### Question 24
What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?

A) CMD is required, ENTRYPOINT is optional
B) CMD provides default arguments that can be overridden; ENTRYPOINT sets the main command
C) They are exactly the same
D) CMD runs at build time, ENTRYPOINT runs at runtime

---

### Question 25
How do you enable GPU access in Docker containers?

A) GPUs work automatically in all containers
B) Install NVIDIA Container Toolkit and use --gpus flag
C) Use a special GPU-enabled base image
D) Mount /dev/nvidia into the container

---

## Section 6: API Development (Questions 26-30)

### Question 26
Why is FastAPI particularly well-suited for ML model serving?

A) It's the only framework that supports ML models
B) It's fast (async), has auto-documentation, and built-in validation
C) It requires less code than other frameworks
D) It's free and open source

---

### Question 27
What does Pydantic provide in FastAPI applications?

A) Database connectivity
B) Automatic request/response validation based on Python types
C) Machine learning model hosting
D) Frontend UI components

---

### Question 28
**[Short Answer]**
What is the difference between a 400 and 500 HTTP status code, and when would you use each?

_Your answer:_

---

### Question 29
What is the purpose of Prometheus metrics in an ML serving API?

A) To train models faster
B) To monitor system performance, request rates, latencies, and errors
C) To automatically scale the application
D) To generate API documentation

---

### Question 30
Which of the following is a best practice for production API design?

A) Return detailed error messages including stack traces to users
B) Use generic error messages and log detailed errors server-side
C) Never use HTTP status codes
D) Disable all logging in production for performance

---

---

# Answers and Explanations

## Section 1: ML Infrastructure Fundamentals

**Question 1: B**
ML Infrastructure Engineers build and maintain systems that enable ML model deployment and operation. They focus on infrastructure, not model training or research.

**Question 2: C**
Customer Acquisition is a business function, not part of the technical ML lifecycle. The ML lifecycle includes: Data Collection → Training → Evaluation → Deployment → Monitoring → Retraining.

**Question 3: B**
MLOps extends DevOps practices by adding data versioning, model versioning, and model-specific monitoring. Code versioning is common to both, but MLOps handles the additional complexity of data and model artifacts.

**Question 4: Sample Answer**
Model retraining is necessary because model performance degrades over time due to data drift (input data distribution changes) and concept drift (relationship between inputs and outputs changes). Regular retraining with fresh data keeps models accurate and relevant to current conditions.

**Question 5: B**
Entry-level AI Infrastructure Engineers in the US typically earn $107k-$230k, reflecting the high demand and specialized skills required for the role.

---

## Section 2: ML Frameworks

**Question 6: B**
PyTorch uses dynamic computation graphs (define-by-run), allowing flexibility and easier debugging. TensorFlow traditionally used static graphs (TF 1.x), though TF 2.x added eager execution for dynamic behavior.

**Question 7: B**
Saving the state dictionary (`model.state_dict()`) is recommended for production because it's more portable, smaller (~50% reduction), and separates model architecture from weights. You need the architecture code to load it back.

**Question 8: B**
`model.eval()` changes the behavior of layers like Dropout (disables during inference) and BatchNormalization (uses running statistics instead of batch statistics). Forgetting this leads to incorrect predictions.

**Question 9: A, B, C, E**
Correct optimization techniques:
- A: Quantization reduces model size and increases speed (4x smaller, 2-4x faster)
- B: TorchScript compiles models for 10-30% faster inference
- C: Batch inference is 3-10x faster than single inference
- E: Warm-up eliminates slow first inference (JIT compilation, CUDA initialization)
- D is WRONG: Training mode should never be used for inference

**Question 10: B**
ONNX (Open Neural Network Exchange) is a universal model format that allows converting models between frameworks (PyTorch ↔ TensorFlow) and often provides faster inference with ONNX Runtime.

---

## Section 3: Cloud Computing

**Question 11: C**
IaaS (Infrastructure as a Service) provides the most control—you manage the OS, runtime, and applications. PaaS manages more for you, and SaaS manages everything.

**Question 12: B**
Spot/preemptible instances are discounted (up to 90% off) but can be terminated with 30 seconds notice. Suitable for batch processing, training with checkpoints, and non-critical workloads.

**Question 13: A, B, D, E**
Valid cost optimization strategies:
- A: Right-sizing prevents over-provisioning
- B: Spot instances save up to 90% on batch workloads
- D: Reserved instances offer 30-75% discount for committed usage
- E: Auto-shutdown saves costs during non-business hours
- C is WRONG: Running 24/7 wastes money; use auto-scaling instead

**Question 14: B**
GPU instances typically cost $1-5 per hour depending on GPU type:
- Entry (T4, K80): ~$1-2/hour
- Mid (V100, A10): ~$2-4/hour
- High-end (A100): ~$4-5/hour

**Question 15: D**
All three cloud providers are suitable depending on requirements:
- AWS: Most mature, widest services
- GCP: Best for ML/AI, TPUs, Kubernetes
- Azure: Best for enterprise, Microsoft integration
Choice depends on your specific needs, existing infrastructure, and team expertise.

---

## Section 4: Model Serving

**Question 16: B**
Training optimizes for accuracy over hours/days with large datasets. Serving optimizes for low latency (milliseconds) and high throughput with single samples or small batches.

**Question 17: B**
Batch inference processes multiple requests together, which dramatically improves GPU utilization (GPUs are optimized for parallel processing) and increases throughput 3-10x. Tradeoff: adds latency waiting for batch to fill.

**Question 18: B**
In blue-green deployment, traffic is switched instantly from the old version (blue) to the new version (green). Both versions exist simultaneously, allowing instant rollback if needed. Gradual shifting is canary deployment.

**Question 19: Sample Answer**
Model warm-up is important because the first inference is 10-100x slower than subsequent inferences due to JIT compilation, CUDA kernel initialization, and memory allocation. Running dummy inferences during startup ensures production requests get consistent, fast response times.

**Question 20: C**
Shadow deployment copies production traffic to the new model but doesn't return its predictions to users. This allows testing with real data without any risk to users. The new model's predictions are logged for comparison.

---

## Section 5: Docker

**Question 21: B**
Multi-stage builds create smaller final images by using a builder stage (with all build tools) and a runtime stage (with only what's needed to run). Build dependencies are excluded from the final image, reducing size by 50-80%.

**Question 22: B**
Running as non-root is a security best practice. If an attacker exploits a container vulnerability to escape, they'll have limited privileges on the host system. Root users in containers can potentially gain root access to the host.

**Question 23: B, C, E**
RUN, COPY, and ENV create new layers and affect caching:
- Changing files copied by COPY invalidates cache
- Modifying commands in RUN invalidates cache
- Changing ENV values invalidates cache
FROM and CMD don't create cached layers in the same way.

**Question 24: B**
CMD provides default arguments that can be overridden when running the container. ENTRYPOINT sets the main executable that always runs. They're often used together: ENTRYPOINT for the command, CMD for default arguments.

Example:
```dockerfile
ENTRYPOINT ["python"]
CMD ["app.py"]
```
Run with: `docker run image script.py` (overrides CMD)

**Question 25: B**
GPU access requires:
1. NVIDIA Container Toolkit installed on host
2. Using `--gpus` flag when running: `docker run --gpus all ...`
3. NVIDIA drivers on host (not in container)

---

## Section 6: API Development

**Question 26: B**
FastAPI is ideal for ML serving because:
- Fast: One of the fastest Python frameworks (async/await)
- Auto-documentation: Swagger UI generated automatically
- Validation: Pydantic validates requests/responses automatically
- Type hints: Type safety catches errors early

**Question 27: B**
Pydantic provides automatic request/response validation based on Python type hints. It validates data types, required fields, value ranges, and more—returning clear error messages when validation fails.

**Question 28: Sample Answer**
400 (Bad Request) indicates a client error—the request was malformed or contained invalid data. Use for validation errors, missing required fields, or invalid input format. 500 (Internal Server Error) indicates a server error—something went wrong processing a valid request. Use for unexpected exceptions, database errors, or service failures.

**Question 29: B**
Prometheus metrics provide observability into your API's performance:
- Request rates (requests/sec)
- Latencies (p50, p95, p99)
- Error rates
- Model-specific metrics (confidence scores, prediction distribution)
This enables monitoring, alerting, and capacity planning.

**Question 30: B**
In production, use generic error messages for users (avoid exposing internal details) but log detailed information server-side for debugging. Never expose stack traces, database details, or system information to users—this helps attackers.

---

## Scoring Guide

- **27-30 correct (90-100%)**: Excellent! You have a strong grasp of ML infrastructure fundamentals.
- **24-26 correct (80-89%)**: Very Good! Minor review needed on a few topics.
- **21-23 correct (70-79%)**: Good! Passing score, but review weaker areas.
- **18-20 correct (60-69%)**: Fair. Significant review needed before moving to Module 02.
- **Below 18 (< 60%)**: Please review the lessons and retake the quiz.

---

## Review Topics by Section

If you scored poorly in a section, review these lessons:

- **Section 1**: Lesson 01 (Introduction) and Lesson 03 (ML Infrastructure Basics)
- **Section 2**: Lesson 04 (ML Frameworks Fundamentals)
- **Section 3**: Lesson 05 (Cloud Computing Intro)
- **Section 4**: Lesson 06 (Model Serving Basics)
- **Section 5**: Lesson 07 (Docker Basics)
- **Section 6**: Lesson 08 (API Development)

---

## Next Steps

1. **Score < 70%**: Review the lessons for topics you struggled with, then retake the quiz
2. **Score 70-89%**: Review weak areas, complete additional exercises
3. **Score ≥ 90%**: Proceed to Module 02 or work on Project 01

**Additional Resources:**
- Complete the exercises in `exercises/` directory
- Work through Project 01 with the provided code stubs
- Review the resources.md file for each lesson

Good luck! 🚀
