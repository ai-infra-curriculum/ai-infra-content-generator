# Module 01: Foundations of ML Infrastructure

**Duration:** 40 hours
**Difficulty:** Beginner
**Prerequisites:** Basic Python programming, command-line familiarity

## Module Overview

Welcome to the first module of your AI Infrastructure Engineering journey! This module establishes the foundational concepts, tools, and practices you'll need throughout your career in ML infrastructure.

You'll learn what ML infrastructure is, why it matters, and how to set up a professional development environment. By the end of this module, you'll deploy your first ML model using a basic serving framework.

## Learning Outcomes

By the end of this module, you will be able to:

1. **Understand the ML Lifecycle** - Explain the stages of ML development from training to production
2. **Set Up Development Environment** - Configure Python, Docker, Git, and cloud tools for ML infrastructure work
3. **Deploy First ML Model** - Use TorchServe or TensorFlow Serving to deploy a pre-trained model
4. **Understand the Role of Infrastructure** - Articulate how infrastructure enables ML systems
5. **Navigate ML Frameworks** - Work with PyTorch and TensorFlow at a basic level
6. **Apply MLOps Fundamentals** - Understand key MLOps concepts and their importance

## Topics Covered

### 1. Introduction to ML Infrastructure and MLOps
- What is ML infrastructure?
- The difference between ML engineering and ML infrastructure engineering
- MLOps: Definition, importance, and maturity levels
- Industry trends and career opportunities

### 2. The ML Lifecycle
- Training phase: data, model development, experimentation
- Serving phase: deployment, inference, monitoring
- Feedback loop: performance tracking, retraining
- The role of infrastructure at each stage

### 3. ML Frameworks Overview
- PyTorch: dynamic computation graphs, research-friendly
- TensorFlow: static graphs, production-optimized
- Hugging Face Transformers: pre-trained models and ecosystems
- When to use which framework

### 4. Development Environment Setup
- Python 3.9+ with virtual environments
- Git for version control
- Docker basics and installation
- Cloud platform setup (AWS/GCP/Azure)
- VS Code or PyCharm configuration for ML work

### 5. Cloud Platforms Introduction
- Infrastructure as a Service (IaaS) vs Platform as a Service (PaaS)
- Major cloud providers: AWS, GCP, Azure
- Core services for ML: compute, storage, networking
- Free tiers and cost management basics

### 6. Model Serving Fundamentals
- What is model serving?
- Serving frameworks: TorchServe, TensorFlow Serving, FastAPI
- REST APIs for model inference
- Synchronous vs asynchronous serving

## Lesson Plan

### Lesson 01: Introduction to ML Infrastructure (3 hours)
Introduction to the field, career paths, and key concepts

**Reading:** `lecture-notes/01-introduction.md`
**Exercise:** None (orientation lesson)

### Lesson 02: Environment Setup (5 hours)
Hands-on setup of all development tools and verification

**Reading:** `lecture-notes/02-environment-setup.md`
**Exercise:** `exercises/exercise-01-environment.md` (2-3 hours)

### Lesson 03: ML Infrastructure Basics (4 hours)
Deep dive into ML lifecycle and infrastructure requirements

**Reading:** `lecture-notes/03-ml-infrastructure-basics.md`
**Exercise:** Concepts reinforced in later exercises

### Lesson 04: ML Frameworks Fundamentals (6 hours)
Working with PyTorch and TensorFlow for infrastructure engineers

**Reading:** `lecture-notes/04-ml-frameworks.md`
**Exercise:** Concepts applied in Exercise 05 (ML Framework Benchmarking)

### Lesson 05: Cloud Computing Intro (5 hours)
Cloud platforms, services, and first deployments

**Reading:** `lecture-notes/05-cloud-intro.md`
**Exercise:** Concepts applied in Docker and Kubernetes exercises

### Lesson 06: Model Serving Basics (8 hours)
Deploy first ML model with TorchServe or TensorFlow Serving

**Reading:** `lecture-notes/06-model-serving-basics.md`
**Exercise:** Concepts applied in Exercise 07 (API Development)

### Lesson 07: Containerization Intro (6 hours)
Docker basics and containerizing ML applications

**Reading:** `lecture-notes/07-docker-basics.md`
**Exercise:** `exercises/exercise-02-docker.md` (2-3 hours)

### Lesson 08: API Development for ML (5 hours)
Building REST APIs for model inference with FastAPI

**Reading:** `lecture-notes/08-api-development.md`
**Exercises:**
- `exercises/exercise-03-kubernetes.md` (3-4 hours) - Deploy ML apps to Kubernetes
- `exercises/exercise-07-api.md` (4-5 hours) - Build FastAPI ML service

## Advanced Projects

After completing the foundational exercises, challenge yourself with these capstone-level projects:

### Exercise 04: Python Environment Manager (30-38 hours)
Build a comprehensive Python environment management tool similar to pyenv/poetry.

**Location:** `exercises/exercise-04-python-env-manager/`
**Prerequisites:** Completed Lessons 01-08
**Skills:** Advanced Python, CLI development, dependency management, subprocess handling

### Exercise 05: ML Framework Benchmarking Tool (31-40 hours)
Create an automated benchmarking system comparing PyTorch, TensorFlow, and JAX performance.

**Location:** `exercises/exercise-05-ml-framework-benchmark/`
**Prerequisites:** Completed Lessons 01-08, GPU access recommended
**Skills:** Performance profiling, framework APIs, benchmarking methodologies, visualization

### Exercise 06: FastAPI ML Service Template Generator (27-34 hours)
Build a code generator that creates production-ready FastAPI services for ML model serving.

**Location:** `exercises/exercise-06-fastapi-ml-template-generator/`
**Prerequisites:** Completed Lessons 01-08
**Skills:** Code generation, Jinja2 templates, FastAPI patterns, Docker, CI/CD

## Hands-On Activities

### Activity 1: Set Up Complete Development Environment
Install and configure all required tools: Python, Git, Docker, cloud CLI, VS Code

**Time:** 3-4 hours
**Deliverable:** Screenshot of working environment with all tools

### Activity 2: Deploy Pre-Trained Image Classification Model
Deploy a ResNet model using TorchServe or TensorFlow Serving

**Time:** 4-5 hours
**Deliverable:** Working model service accepting image uploads

### Activity 3: Create Simple REST API for Model Inference
Build FastAPI application that wraps model inference

**Time:** 3-4 hours
**Deliverable:** API with `/predict` endpoint and documentation

### Activity 4: Containerize ML Service with Docker
Package API and model into Docker container

**Time:** 2-3 hours
**Deliverable:** Docker image running model serving

## Assessments

### Quiz: ML Infrastructure Concepts and Terminology
**Location:** `quizzes/module-01-quiz.md`
**Questions:** 20 multiple choice and short answer
**Passing Score:** 70% (14/20 correct)
**Topics:** MLOps, ML lifecycle, cloud basics, serving concepts

### Practical: Deploy and Test Simple ML Model
**Objective:** Deploy a pre-trained model and demonstrate inference

**Requirements:**
1. Model deployed and accessible via HTTP
2. Successfully processes image input
3. Returns predictions with confidence scores
4. Response time < 500ms for small images
5. Includes basic error handling

**Submission:** GitHub repository with code, README, and demo video/screenshots

## Additional Resources

### Recommended Reading
- **"Machine Learning Engineering" by Andriy Burkov** - Chapters 1-2 (ML in production)
- **Google's "Rules of Machine Learning"** - [Best practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- **MLOps Maturity Model** - Microsoft's framework

### Video Tutorials
- **"Full Stack Deep Learning"** - Course introduction and ML infrastructure overview
- **Andrew Ng's MLOps Specialization** - First course (intro to MLOps)

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [TorchServe Documentation](https://pytorch.org/serve/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

### Interactive Tutorials
- [Katacoda Docker Scenarios](https://www.katacoda.com/courses/docker)
- [Google Cloud Skills Boost](https://www.cloudskillsboost.google/paths)

See `resources.md` in this folder for a comprehensive resource list.

## Common Questions

**Q: Do I need a GPU for this module?**
A: No, a CPU is sufficient for this module. We'll work with small models and pre-trained weights. GPU will be important in later modules.

**Q: Which cloud platform should I choose?**
A: Any of AWS, GCP, or Azure will work. If unsure, start with GCP (generous free tier) or AWS (most widely used in industry). You can try multiple platforms.

**Q: How long should each lesson take?**
A: The time estimates are guidelines. Expect 3-8 hours per lesson depending on your background. Take your time and understand concepts deeply.

**Q: What if I get stuck on an exercise?**
A: Check the solutions folder, review the lesson materials, search official documentation, and post questions in GitHub Discussions. Learning to debug and find solutions is part of the skill development.

**Q: Can I skip the environment setup if I already have Python/Docker installed?**
A: Review the setup lesson anyway to ensure your environment matches the curriculum requirements. Missing dependencies will cause issues in later modules.

## Tips for Success

1. **Hands-on practice is essential** - Don't just read, type the code and run examples
2. **Document your learning** - Keep notes on commands, errors, and solutions
3. **Build your portfolio** - Save all exercise code in a personal GitHub repository
4. **Join the community** - Connect with other learners for support and knowledge sharing
5. **Take breaks** - ML infrastructure concepts can be dense. Step away and return with fresh eyes.
6. **Ask questions** - No question is too basic. Understanding fundamentals is crucial.

## Module Prerequisites Check

Before starting, ensure you have:

- [ ] Computer with 8GB+ RAM and 20GB+ free disk space
- [ ] Linux, macOS, or Windows with WSL2
- [ ] Admin/sudo privileges to install software
- [ ] Internet connection for downloads and cloud services
- [ ] GitHub account for code repositories
- [ ] Text editor (VS Code recommended)
- [ ] Basic Python knowledge (functions, classes, imports)
- [ ] Command-line comfort (cd, ls, mkdir, cat)

If any prerequisites are missing, address them before starting Lesson 01.

## Next Steps

Once you complete this module:

1. **Take the module quiz** - Validate your understanding
2. **Complete the practical assessment** - Deploy a working model
3. **Review your work** - Ensure code quality and documentation
4. **Move to Module 02** - Cloud Computing for ML (deeper cloud platform skills)

---

**Ready to begin?** Start with `01-introduction.md` and begin your ML infrastructure journey!

**Questions or feedback?** Open an issue in the GitHub repository.
