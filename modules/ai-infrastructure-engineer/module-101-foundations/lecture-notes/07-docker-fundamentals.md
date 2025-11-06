# Lesson 07: Docker Fundamentals for ML

## Learning Objectives
- Understand containerization concepts
- Write Dockerfiles for ML applications
- Build and optimize Docker images
- Use Docker Compose for multi-container apps

## Duration: 3-4 hours

---

## 1. Why Docker for ML?

**Benefits:**
- **Reproducibility**: Same environment everywhere
- **Isolation**: Dependencies don't conflict
- **Portability**: Run anywhere (local, cloud, k8s)
- **Consistency**: Dev = Staging = Production

---

## 2. Docker Basics

### Key Concepts
- **Image**: Template (read-only)
- **Container**: Running instance of image
- **Dockerfile**: Instructions to build image
- **Registry**: Store for images (Docker Hub, ECR, GCR)

### Essential Commands
```bash
# Build image
docker build -t my-ml-app:v1 .

# Run container
docker run -p 8000:8000 my-ml-app:v1

# List containers
docker ps

# Stop container
docker stop <container-id>

# Remove container
docker rm <container-id>

# List images
docker images

# Remove image
docker rmi my-ml-app:v1
```

---

## 3. Writing Dockerfiles for ML

### 3.1 Basic Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t ml-api:v1 .
docker run -p 8000:8000 ml-api:v1
```

### 3.2 Optimized Multi-stage Dockerfile

```dockerfile
# Stage 1: Build environment
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Benefits:**
- Smaller image size (no build tools in final image)
- Security (non-root user)
- Health checks

### 3.3 GPU-enabled Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "train.py"]
```

---

## 4. Docker Compose for Multi-container Apps

### Example: API + Database + Monitoring

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mldb
    depends_on:
      - db
    volumes:
      - ./models:/app/models

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mldb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  postgres_data:
```

**Run:**
```bash
docker-compose up -d
docker-compose logs -f api
docker-compose down
```

---

## 5. Best Practices for ML Docker Images

### 5.1 Layer Caching

```dockerfile
# Good: Install dependencies first (changes less frequently)
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Bad: Copy everything first (cache invalidated on every code change)
COPY . .
RUN pip install -r requirements.txt
```

### 5.2 Minimize Image Size

```dockerfile
# Use slim base images
FROM python:3.11-slim  # 50MB vs python:3.11 (350MB)

# Clean up in same layer
RUN apt-get update && apt-get install -y package \
    && rm -rf /var/lib/apt/lists/*

# Use .dockerignore
# .dockerignore file:
__pycache__
*.pyc
.git
.env
tests/
```

### 5.3 Security

```dockerfile
# Don't run as root
RUN useradd -m appuser
USER appuser

# Don't include secrets
# Use build args or environment variables
ARG API_KEY
ENV API_KEY=${API_KEY}
```

---

## 6. Hands-On Exercise

### TODO: Dockerize the FastAPI ML app from Lesson 06

**Requirements:**
1. Create optimized multi-stage Dockerfile
2. Image size < 500MB
3. Run as non-root user
4. Include health check
5. Create docker-compose.yml with API + Prometheus

**Test:**
```bash
docker build -t ml-api:v1 .
docker images ml-api:v1  # Check size
docker run -p 8000:8000 ml-api:v1
curl http://localhost:8000/health
```

---

## 7. Common Docker Issues for ML

### Issue 1: Large Image Sizes

**Problem**: Images >5GB
**Solution**: Use multi-stage builds, slim base images, .dockerignore

### Issue 2: Slow Builds

**Problem**: Builds take 10+ minutes
**Solution**: Order layers by change frequency, cache dependencies

### Issue 3: GPU Not Detected

**Problem**: `torch.cuda.is_available()` returns False
**Solution**: Use nvidia/cuda base image, install NVIDIA Container Toolkit

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU
docker run --gpus all my-ml-app:v1
```

---

## 8. Key Takeaways

- ✅ Docker ensures reproducible ML environments
- ✅ Multi-stage builds reduce image size
- ✅ Use slim base images and layer caching
- ✅ Docker Compose simplifies multi-container apps
- ✅ Always run containers as non-root user

---

## Next Steps

✅ Complete Dockerization exercise
✅ Push image to Docker Hub or container registry
✅ Proceed to [Lesson 08: API Development & Testing](./08-api-development-testing.md)
