# Lesson 07: Docker Basics for ML Infrastructure

**Duration:** 6 hours

## Learning Objectives

By the end of this lecture, you will be able to:
- Explain containerization and how Docker solves environment consistency problems
- Build Docker images for ML applications with appropriate base images
- Write optimized Dockerfiles using multi-stage builds and layer caching
- Manage Docker containers (start, stop, logs, exec)
- Use Docker volumes for persistent data and model storage
- Configure Docker networking for multi-container applications
- Push and pull images from Docker registries (Docker Hub, ECR, GCR)
- Apply Docker best practices for security and image size optimization

## Introduction

Docker is a platform for developing, shipping, and running applications in containers. For ML infrastructure engineers, Docker is essential for:

- Creating consistent environments across development, testing, and production
- Packaging models with their dependencies
- Enabling easy deployment and scaling
- Isolating applications for security
- Simplifying dependency management

This lesson covers Docker fundamentals specifically for ML infrastructure use cases.

## What is Docker?

### The Problem Docker Solves

**Before Docker:**
```
Developer's Machine:
✅ Python 3.11, PyTorch 2.1, CUDA 11.8
✅ Model works perfectly

Production Server:
❌ Python 3.9, PyTorch 1.12, CUDA 11.2
❌ "But it works on my machine!"
❌ Hours spent debugging environment issues
```

**With Docker:**
```
Developer's Machine:
✅ Build Docker image with exact dependencies
✅ Test in container

Production Server:
✅ Run same Docker image
✅ Guaranteed identical environment
✅ No surprises!
```

### Containers vs Virtual Machines

```
┌─────────────────────────────────────────────────────────┐
│              Virtual Machines (VMs)                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  App A   │  │  App B   │  │  App C   │             │
│  ├──────────┤  ├──────────┤  ├──────────┤             │
│  │  Bins    │  │  Bins    │  │  Bins    │             │
│  │  Libs    │  │  Libs    │  │  Libs    │             │
│  ├──────────┤  ├──────────┤  ├──────────┤             │
│  │ Guest OS │  │ Guest OS │  │ Guest OS │  (GBs each) │
│  └──────────┘  └──────────┘  └──────────┘             │
│              Hypervisor                                  │
│              Host OS                                     │
│              Hardware                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Docker Containers                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  App A   │  │  App B   │  │  App C   │             │
│  ├──────────┤  ├──────────┤  ├──────────┤             │
│  │  Bins    │  │  Bins    │  │  Bins    │             │
│  │  Libs    │  │  Libs    │  │  Libs    │             │
│  └──────────┘  └──────────┘  └──────────┘  (MBs each)  │
│              Docker Engine                               │
│              Host OS (shared kernel)                     │
│              Hardware                                    │
└─────────────────────────────────────────────────────────┘

Containers:                      VMs:
- Lightweight (MBs)             - Heavy (GBs)
- Fast startup (seconds)        - Slow startup (minutes)
- Share host OS kernel          - Each has own OS
- Less isolation                - Strong isolation
- More efficient                - More resource overhead
```

### Key Docker Concepts

1. **Image**: Read-only template with application and dependencies
2. **Container**: Running instance of an image
3. **Dockerfile**: Instructions to build an image
4. **Registry**: Repository for storing images (Docker Hub, ECR, GCR)
5. **Volume**: Persistent data storage
6. **Network**: Communication between containers

## Installing Docker

### Linux (Ubuntu/Debian)
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
sudo docker run hello-world

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### macOS
```bash
# Download Docker Desktop from: https://www.docker.com/products/docker-desktop
# Or use Homebrew
brew install --cask docker

# Verify
docker --version
docker run hello-world
```

### Windows
```bash
# Download Docker Desktop from: https://www.docker.com/products/docker-desktop
# Requires WSL2

# Verify in PowerShell or WSL2
docker --version
docker run hello-world
```

## Basic Docker Commands

### Working with Images

```bash
# Pull image from registry
docker pull python:3.11-slim

# List local images
docker images

# Remove image
docker rmi python:3.11-slim

# Search for images
docker search pytorch

# Build image from Dockerfile
docker build -t my-image:v1.0 .

# Tag image
docker tag my-image:v1.0 username/my-image:v1.0

# Push image to registry
docker push username/my-image:v1.0
```

### Working with Containers

```bash
# Run container
docker run python:3.11-slim

# Run container with name
docker run --name my-container python:3.11-slim

# Run container in background (detached)
docker run -d python:3.11-slim

# Run container with port mapping
docker run -p 8000:8000 my-api:v1.0

# Run container interactively
docker run -it python:3.11-slim /bin/bash

# Run container with environment variables
docker run -e MODEL_NAME=resnet18 my-model:v1.0

# Run container with volume
docker run -v /host/path:/container/path my-app:v1.0

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop my-container

# Start stopped container
docker start my-container

# Remove container
docker rm my-container

# Remove all stopped containers
docker container prune

# View container logs
docker logs my-container

# Follow logs (live)
docker logs -f my-container

# Execute command in running container
docker exec -it my-container /bin/bash

# Inspect container
docker inspect my-container

# View container resource usage
docker stats my-container
```

## Writing Dockerfiles for ML

### Basic Dockerfile Structure

```dockerfile
# 1. Base Image
FROM python:3.11-slim

# 2. Metadata
LABEL maintainer="you@example.com"
LABEL version="1.0"

# 3. Environment Variables
ENV PYTHONUNBUFFERED=1

# 4. Working Directory
WORKDIR /app

# 5. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Expose Port
EXPOSE 8000

# 8. Run Command
CMD ["python", "app.py"]
```

### ML-Specific Dockerfile Example

```dockerfile
# ML Model Serving Dockerfile

# Base image with Python
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME=resnet18 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Run application
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Builds for Optimization

**Problem**: Docker images can become very large

**Solution**: Multi-stage builds

```dockerfile
# ❌ Single-stage (large image: ~2GB)
FROM python:3.11
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]

# ✅ Multi-stage (smaller image: ~500MB)
# Stage 1: Build
FROM python:3.11 as builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

### Dockerfile Best Practices

```dockerfile
# ✅ GOOD: Layer caching optimization
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .           # Copy requirements first
RUN pip install -r requirements.txt  # Install dependencies (cached)
COPY . .                          # Copy code last (changes frequently)

# ❌ BAD: No caching
FROM python:3.11-slim
WORKDIR /app
COPY . .                          # Copy everything (invalidates cache)
RUN pip install -r requirements.txt  # Reinstall every time

# ✅ GOOD: Combine commands to reduce layers
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# ❌ BAD: Multiple layers
RUN apt-get update
RUN apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*

# ✅ GOOD: Use specific versions
FROM python:3.11-slim

# ❌ BAD: Use latest (unpredictable)
FROM python:latest

# ✅ GOOD: Non-root user
RUN useradd -r appuser
USER appuser

# ❌ BAD: Run as root (security risk)
# (no USER specified, defaults to root)

# ✅ GOOD: Clean up in same layer
RUN apt-get update && \
    apt-get install -y package && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ❌ BAD: Clean up in different layer (doesn't reduce size)
RUN apt-get update
RUN apt-get install -y package
RUN rm -rf /var/lib/apt/lists/*
```

## Building and Running ML Containers

### Building an Image

```bash
# Basic build
docker build -t ml-model:v1.0 .

# Build with build arguments
docker build --build-arg MODEL_NAME=resnet50 -t ml-model:v1.0 .

# Build with different Dockerfile
docker build -f Dockerfile.gpu -t ml-model-gpu:v1.0 .

# Build without cache
docker build --no-cache -t ml-model:v1.0 .

# View build history
docker history ml-model:v1.0
```

### Running ML Container

```bash
# Basic run
docker run ml-model:v1.0

# Run with port mapping
docker run -p 8000:8000 ml-model:v1.0

# Run in background
docker run -d -p 8000:8000 --name ml-api ml-model:v1.0

# Run with GPU support (NVIDIA)
docker run --gpus all -p 8000:8000 ml-model-gpu:v1.0

# Run with specific GPU
docker run --gpus '"device=0"' -p 8000:8000 ml-model-gpu:v1.0

# Run with resource limits
docker run \
    --memory=4g \
    --cpus=2 \
    -p 8000:8000 \
    ml-model:v1.0

# Run with environment variables
docker run \
    -e MODEL_NAME=resnet50 \
    -e BATCH_SIZE=32 \
    -p 8000:8000 \
    ml-model:v1.0

# Run with volume for models
docker run \
    -v /host/models:/app/models \
    -p 8000:8000 \
    ml-model:v1.0

# Run with restart policy
docker run \
    --restart=always \
    -d \
    -p 8000:8000 \
    ml-model:v1.0
```

## Docker Compose for Multi-Container Apps

### What is Docker Compose?

Tool for defining and running multi-container applications using YAML.

### Docker Compose File Example

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ML API Service
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: ml-model:v1.0
    container_name: ml-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=resnet18
      - DEVICE=cpu
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - ml-network

  # Prometheus (Monitoring)
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    networks:
      - ml-network

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - ml-network
    depends_on:
      - prometheus

networks:
  ml-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs

# Follow logs for specific service
docker-compose logs -f ml-api

# Scale service
docker-compose up -d --scale ml-api=3

# Rebuild images
docker-compose build

# Rebuild and start
docker-compose up -d --build

# View running services
docker-compose ps

# Execute command in service
docker-compose exec ml-api /bin/bash

# View resource usage
docker-compose stats
```

## Docker Networking

### Network Types

```bash
# Bridge (default): Containers on same host
docker network create ml-bridge

# Host: Container uses host networking (no isolation)
docker run --network host ml-model:v1.0

# None: No networking
docker run --network none ml-model:v1.0

# List networks
docker network ls

# Inspect network
docker network inspect ml-bridge

# Connect container to network
docker network connect ml-bridge ml-api

# Disconnect
docker network disconnect ml-bridge ml-api
```

### Container Communication

```bash
# Create network
docker network create ml-network

# Run containers on same network
docker run -d --name ml-api --network ml-network ml-model:v1.0
docker run -d --name redis --network ml-network redis:latest

# Containers can communicate using service names
# ml-api can reach redis at: redis:6379
```

## Docker Volumes (Persistent Storage)

### Volume Types

```bash
# Named volume (managed by Docker)
docker volume create model-storage
docker run -v model-storage:/app/models ml-model:v1.0

# Bind mount (specific host path)
docker run -v /host/models:/app/models ml-model:v1.0

# tmpfs (in-memory, temporary)
docker run --tmpfs /tmp ml-model:v1.0

# List volumes
docker volume ls

# Inspect volume
docker volume inspect model-storage

# Remove volume
docker volume rm model-storage

# Remove unused volumes
docker volume prune
```

## Docker for GPU Workloads

### NVIDIA Container Toolkit Setup

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### GPU Dockerfile

```dockerfile
# Use CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENV DEVICE=cuda

CMD ["python3", "serve.py"]
```

## Docker Registry (Image Storage)

### Docker Hub (Public Registry)

```bash
# Login
docker login

# Tag image
docker tag ml-model:v1.0 username/ml-model:v1.0

# Push to Docker Hub
docker push username/ml-model:v1.0

# Pull from Docker Hub
docker pull username/ml-model:v1.0
```

### Private Registry (AWS ECR)

```bash
# Install AWS CLI
aws configure

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name ml-model

# Tag image
docker tag ml-model:v1.0 \
    123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0

# Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0
```

### Private Registry (GCP GCR)

```bash
# Configure gcloud
gcloud auth configure-docker

# Tag image
docker tag ml-model:v1.0 gcr.io/project-id/ml-model:v1.0

# Push to GCR
docker push gcr.io/project-id/ml-model:v1.0
```

## Troubleshooting Docker

### Common Issues

```bash
# Issue: Permission denied
# Solution: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Issue: Container exits immediately
# Check logs
docker logs container-name

# Issue: Port already in use
# Find process using port
sudo lsof -i :8000
# Kill process or use different port

# Issue: Out of disk space
# Clean up unused resources
docker system prune -a

# Issue: Container can't connect to network
# Check network configuration
docker network inspect network-name

# Issue: Image too large
# Use multi-stage builds
# Use smaller base images (alpine, slim)

# Issue: Slow build times
# Optimize layer caching
# Use .dockerignore

# Debug container
docker run -it ml-model:v1.0 /bin/bash
docker exec -it container-name /bin/bash

# View container processes
docker top container-name

# Monitor resource usage
docker stats

# Inspect everything
docker inspect container-name
```

## Practical Exercise

**Build and run a containerized ML API:**

```bash
# 1. Create project structure
mkdir ml-docker-demo && cd ml-docker-demo

# 2. Create simple FastAPI app
cat > app.py << 'EOF'
from fastapi import FastAPI
import torch

app = FastAPI()

@app.get("/")
def root():
    return {"message": "ML API Running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "torch_version": torch.__version__}

@app.get("/gpu")
def gpu_check():
    return {"cuda_available": torch.cuda.is_available()}
EOF

# 3. Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
EOF

# 4. Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# 5. Build image
docker build -t ml-api:v1.0 .

# 6. Run container
docker run -d -p 8000:8000 --name ml-api ml-api:v1.0

# 7. Test
curl http://localhost:8000/health

# 8. View logs
docker logs ml-api

# 9. Cleanup
docker stop ml-api
docker rm ml-api
```

## Key Takeaways

1. **Docker** creates consistent, portable environments using containers
2. **Containers** are lightweight, fast, and isolated
3. **Dockerfile** defines how to build an image
4. **Multi-stage builds** reduce image size
5. **Docker Compose** manages multi-container applications
6. **Volumes** provide persistent storage
7. **Networks** enable container communication
8. **GPU support** requires NVIDIA Container Toolkit
9. **Best practices**: Use specific versions, non-root users, layer caching

## Self-Check Questions

1. What's the difference between an image and a container?
2. What is the purpose of multi-stage builds?
3. How do you expose a port from a Docker container?
4. What's the difference between COPY and ADD in Dockerfile?
5. How do you persist data in Docker?
6. How do you enable GPU access in Docker containers?
7. What is Docker Compose used for?

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker for ML Guide](https://docs.docker.com/get-started/)

---

**Next Lesson:** [08-api-development.md](./08-api-development.md) - Building production-ready APIs
