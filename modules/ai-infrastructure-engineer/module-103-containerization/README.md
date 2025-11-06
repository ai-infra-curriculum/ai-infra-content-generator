# Module 03: Containerization with Docker

**Duration:** 35 hours
**Difficulty:** Intermediate
**Prerequisites:** Module 01 and 02 complete

## Module Overview

This module dives deep into Docker containerization specifically for machine learning applications. You'll learn to build optimized Docker images, manage ML-specific dependencies, handle GPU access in containers, and orchestrate multi-container ML systems with Docker Compose.

By the end of this module, you'll be proficient in containerizing any ML application, from simple inference services to complex multi-component ML pipelines.

## Learning Outcomes

By the end of this module, you will be able to:

1. **Understand Docker Architecture** - Explain how Docker works and its benefits for ML workloads
2. **Write Production Dockerfiles** - Create optimized Dockerfiles for ML applications
3. **Optimize Image Size** - Use multi-stage builds and layer caching to reduce image sizes by 50-80%
4. **Manage ML Dependencies** - Handle PyTorch, TensorFlow, CUDA, and system dependencies correctly
5. **Enable GPU Access** - Configure Docker containers to use NVIDIA GPUs
6. **Use Docker Compose** - Orchestrate multi-container ML systems (model + API + database + cache)
7. **Work with Container Registries** - Push and pull images from Docker Hub, ECR, GCR, ACR
8. **Debug Container Issues** - Troubleshoot common Docker problems in ML contexts

## Why Containerization Matters for ML

### The Problem Without Containers

**"It works on my machine!"** - The classic developer problem, amplified for ML:

- ML models depend on specific library versions (PyTorch 2.0 vs 2.1)
- CUDA versions must match between driver, toolkit, and libraries
- System libraries (libcudnn, libnccl) have complex dependencies
- Python environments are fragile and version-sensitive
- Deployment environments differ from development environments

### The Solution: Docker Containers

Docker solves these problems by:

1. **Packaging Everything** - Code, dependencies, system libraries, all in one image
2. **Reproducibility** - Same container runs identically everywhere
3. **Isolation** - No conflicts with other applications
4. **Portability** - Run on laptop, cloud VMs, Kubernetes, anywhere
5. **Version Control** - Tag and version container images like code

### Real-World Impact

**Example: LLM Deployment**
- Without Docker: 2-3 days setting up CUDA, PyTorch, dependencies on new machines
- With Docker: 5 minutes to pull image and start serving

**Example: Multi-Model Serving**
- Without Docker: Different models require different Python versions → separate VMs
- With Docker: Run 10 models with different dependencies on the same machine

## Module Structure

### Lesson 01: Docker Introduction (4 hours)
**File:** `01-docker-introduction.md`

- Container vs VM comparison
- Docker architecture (daemon, client, images, containers)
- Installing Docker and NVIDIA Container Toolkit
- Basic Docker commands
- First container: Hello World to ML model
- **Hands-on:** Run pre-built ML container

### Lesson 02: Dockerfiles for ML Applications (5 hours)
**File:** `02-dockerfile-ml-apps.md`

- Dockerfile syntax and instructions
- Base images for ML (python:3.11, nvidia/cuda, pytorch/pytorch)
- Installing ML frameworks (PyTorch, TensorFlow)
- Copying code and models into images
- Setting up working directories and entry points
- **Hands-on:** Write Dockerfile for a PyTorch model

### Lesson 03: Image Optimization (5 hours)
**File:** `03-image-optimization.md`

- Multi-stage builds (build vs runtime separation)
- Layer caching strategies
- Minimizing image size (500MB → 150MB techniques)
- .dockerignore for faster builds
- Choosing slim base images
- Security scanning and vulnerabilities
- **Hands-on:** Optimize image size by 70%

### Lesson 04: Docker Networking and Volumes (5 hours)
**File:** `04-docker-networking-volumes.md`

- Docker networking modes (bridge, host, none)
- Port mapping for inference services
- Container-to-container communication
- Docker volumes for persistent storage
- Bind mounts for development
- Managing datasets with volumes
- **Hands-on:** Multi-container application with networking

### Lesson 05: Docker Compose (5 hours)
**File:** `05-docker-compose.md`

- Docker Compose overview
- Writing docker-compose.yml files
- Multi-service ML applications (API + model + Redis + PostgreSQL)
- Environment variables and secrets
- Health checks and dependencies
- Scaling services with compose
- **Hands-on:** Build complete ML stack with Compose

### Lesson 06: Container Registries (4 hours)
**File:** `06-container-registries.md`

- Docker Hub: public and private registries
- AWS ECR (Elastic Container Registry)
- GCP GCR/Artifact Registry
- Azure ACR (Azure Container Registry)
- Image tagging strategies (latest, semantic versioning)
- CI/CD integration with registries
- **Hands-on:** Push images to multiple registries

### Lesson 07: GPU Support in Docker (5 hours)
**File:** `07-gpu-docker.md`

- NVIDIA Container Toolkit installation
- Running GPU-accelerated containers
- CUDA base images (nvidia/cuda)
- Mapping specific GPUs to containers
- GPU memory management
- Multi-GPU containers
- Monitoring GPU usage in containers
- **Hands-on:** Deploy GPU-accelerated inference service

## Hands-On Activities

### Activity 1: Containerize Image Classification Service
**Duration:** 3 hours
**Deliverable:** Docker image < 500MB serving ResNet model via FastAPI

### Activity 2: Multi-Stage Build Optimization
**Duration:** 2 hours
**Deliverable:** Reduce image from 2GB to < 500MB using multi-stage builds

### Activity 3: Multi-Container ML Application
**Duration:** 4 hours
**Deliverable:** Docker Compose stack with model server, API gateway, cache, and database

### Activity 4: GPU Container Deployment
**Duration:** 3 hours
**Deliverable:** GPU-accelerated container serving Stable Diffusion model

### Activity 5: CI/CD with Container Registry
**Duration:** 2 hours
**Deliverable:** GitHub Actions workflow building and pushing images

## Assessments

### Quiz: Docker Concepts and Best Practices
**Location:** `quiz.md`
**Questions:** 20 multiple choice and short answer
**Passing Score:** 70% (14/20 correct)
**Topics:** Docker architecture, Dockerfile syntax, optimization, GPU containers

### Practical: Build Production-Ready ML Container
**Objective:** Create optimized, secure container for ML model deployment

**Requirements:**
1. Multi-stage Dockerfile optimizing for size
2. Image < 500MB for CPU model or < 2GB for GPU model
3. Non-root user for security
4. Health check endpoint
5. Proper signal handling for graceful shutdown
6. Environment variable configuration
7. Documentation in README

**Submission:** GitHub repository with Dockerfile, code, and README

## Prerequisites Check

Before starting Module 03, ensure you have:

- [ ] Completed Modules 01 and 02
- [ ] Docker installed (Docker Desktop or Docker Engine)
- [ ] Basic command-line familiarity
- [ ] Understanding of REST APIs (from Module 01)
- [ ] (Optional) NVIDIA GPU with drivers for GPU lessons
- [ ] (Optional) NVIDIA Container Toolkit for GPU Docker support

### Installation Verification

```bash
# Verify Docker installation
docker --version
docker run hello-world

# Verify Docker Compose
docker compose version

# (Optional) Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Key Concepts Covered

### Docker Fundamentals
- Images vs Containers
- Dockerfile instructions (FROM, RUN, COPY, CMD, ENTRYPOINT)
- Layers and caching
- Docker daemon and client architecture

### ML-Specific Concerns
- Large model files (>1GB) in images
- CUDA and GPU dependencies
- Python package management in containers
- Data volume mounting strategies

### Production Best Practices
- Security (non-root users, scanning)
- Optimization (multi-stage builds, layer caching)
- Logging and monitoring
- Health checks and graceful shutdown

### DevOps Integration
- Container registries and versioning
- CI/CD integration
- Environment management
- Secrets handling

## Tools and Technologies

### Required
- **Docker Engine** (v20.10+) or **Docker Desktop**
- **Docker Compose** (v2.0+)
- **Python** (3.9+) for ML application code

### Optional but Recommended
- **NVIDIA Docker** (for GPU support)
- **Visual Studio Code** with Docker extension
- **Docker Registry Account** (Docker Hub, free tier)
- **GitHub Account** (for CI/CD exercises)

### Cloud Platform Integration
- **AWS ECR** - Elastic Container Registry
- **GCP GCR** - Google Container Registry / Artifact Registry
- **Azure ACR** - Azure Container Registry

## Common Pitfalls and Solutions

### Pitfall 1: Huge Image Sizes
**Problem:** ML images often exceed 5GB
**Solution:** Multi-stage builds, slim base images, layer optimization
**Target:** < 500MB for CPU, < 2GB for GPU images

### Pitfall 2: Build Cache Invalidation
**Problem:** Small code changes rebuild all layers
**Solution:** Proper layer ordering, separate dependency installation
**Impact:** 20 minute builds → 30 second builds

### Pitfall 3: GPU Not Accessible
**Problem:** CUDA errors inside containers
**Solution:** NVIDIA Container Toolkit, correct base images, GPU flags
**Verification:** `nvidia-smi` works inside container

### Pitfall 4: Model Files Too Large for Image
**Problem:** Cannot fit 10GB LLM model in image
**Solution:** Volume mounts, model registries, download on startup
**Alternative:** Use model serving platforms with external storage

### Pitfall 5: Container as Black Box
**Problem:** Cannot debug issues inside containers
**Solution:** `docker exec`, `docker logs`, attach debugger
**Practice:** Learn Docker debugging techniques

## Real-World Applications

### Use Case 1: Multi-Model Serving Platform
**Challenge:** Serve 50 different models with different dependencies
**Solution:** Each model in isolated container, orchestrated with Docker Compose
**Benefit:** No version conflicts, independent scaling

### Use Case 2: Reproducible ML Research
**Challenge:** Paper results not reproducible due to environment differences
**Solution:** Publish Docker image with exact dependencies
**Benefit:** Anyone can reproduce results identically

### Use Case 3: Edge AI Deployment
**Challenge:** Deploy models to edge devices with limited resources
**Solution:** Highly optimized Docker images (< 100MB)
**Benefit:** Fast deployment, minimal resource usage

### Use Case 4: CI/CD for ML Models
**Challenge:** Automate model testing and deployment
**Solution:** Docker containers in CI/CD pipeline
**Benefit:** Consistent testing and deployment environments

## Success Criteria

You've mastered Module 03 when you can:

- [ ] Write Dockerfiles from scratch for ML applications
- [ ] Build images < 500MB for CPU workloads
- [ ] Use multi-stage builds effectively
- [ ] Configure GPU access in containers
- [ ] Use Docker Compose for multi-container applications
- [ ] Push and pull images from container registries
- [ ] Debug container issues efficiently
- [ ] Integrate Docker into CI/CD workflows
- [ ] Apply security best practices (non-root users, scanning)
- [ ] Optimize build times with layer caching

## Time Estimates

| Component | Estimated Hours |
|-----------|----------------|
| Lessons (7) | 35 hours |
| Exercises (5) | 12-15 hours |
| Quiz | 1-2 hours |
| Practical Assessment | 4-6 hours |
| **Total** | **52-58 hours** |

**Recommended Pace:**
- Part-time (10h/week): 5-6 weeks
- Full-time (40h/week): 1.5 weeks

## Resources

**Official Documentation:**
- [Docker Documentation](https://docs.docker.com/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

**Additional Resources:**
See `resources.md` for comprehensive list of:
- Books and tutorials
- Video courses
- Blog posts and articles
- GitHub repositories
- Community forums

## What's Next?

After completing Module 03:

1. **Complete the quiz** with ≥70% score
2. **Finish the practical assessment** - Build production-ready container
3. **Choose your path:**
   - **Module 04: Kubernetes Fundamentals** - Container orchestration at scale
   - **Module 05: Data Pipelines** - Building ML data workflows
   - **Deep Dive: Advanced Docker** - Security, networking, optimization

## Tips for Success

1. **Practice building images iteratively** - Don't aim for perfection on first try
2. **Study layer caching carefully** - This is the #1 optimization technique
3. **Use .dockerignore religiously** - Exclude unnecessary files
4. **Test locally before pushing** - Verify images work before pushing to registry
5. **Learn to read build output** - Understand what each layer does
6. **Keep a cheat sheet** - Common Dockerfile patterns for ML
7. **Experiment with GPU containers** - Even without GPU, understand the concepts

---

**Ready to containerize everything?** Start with [Lesson 01: Docker Introduction](./01-docker-introduction.md)

**Need Docker installed?** Check [Docker Installation Guide](https://docs.docker.com/get-docker/)

**Have questions?** Open an issue or discussion in the GitHub repository!
