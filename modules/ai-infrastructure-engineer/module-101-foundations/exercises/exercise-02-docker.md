# Exercise 02: Docker Fundamentals for ML

**Duration:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 07 (Docker Basics)

## Learning Objectives

By completing this exercise, you will:
- Build Docker images for ML applications
- Optimize Dockerfile for size and build speed
- Use multi-stage builds effectively
- Manage Docker volumes and networks
- Understand layer caching
- Deploy a simple ML model in a container

---

## Part 1: Build Your First ML Container (30 minutes)

### Task 1.1: Create a Simple ML Application

Create a directory structure:
```bash
mkdir ml-docker-exercise && cd ml-docker-exercise
mkdir src
```

Create `src/app.py`:
```python
# Simple ML inference app
from fastapi import FastAPI
import torch
import torchvision.models as models

app = FastAPI()

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = models.resnet18(pretrained=True)
    model.eval()
    print("Model loaded successfully!")

@app.get("/")
def root():
    return {"message": "ML Model API", "model": "ResNet-18"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "torch_version": torch.__version__
    }

@app.get("/info")
def info():
    if model is None:
        return {"error": "Model not loaded"}

    # Count parameters
    params = sum(p.numel() for p in model.parameters())

    return {
        "model": "ResNet-18",
        "parameters": params,
        "device": "cpu"
    }
```

Create `requirements.txt`:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
torchvision==0.16.0
```

**✅ Checkpoint:** Test locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --reload
# Visit http://localhost:8000/docs
```

---

### Task 1.2: Create a Basic Dockerfile

Create `Dockerfile`:
```dockerfile
# TODO: Complete this Dockerfile
FROM python:3.11-slim

WORKDIR /app

# TODO: Copy requirements and install dependencies

# TODO: Copy application code

# TODO: Expose port 8000

# TODO: Set command to run the application
# Use: uvicorn src.app:app --host 0.0.0.0 --port 8000
```

**Your Task:**
1. Complete the Dockerfile
2. Build the image: `docker build -t ml-app:v1 .`
3. Run the container: `docker run -p 8000:8000 ml-app:v1`
4. Test: `curl http://localhost:8000/health`

**Expected Results:**
- Image builds successfully
- Container runs without errors
- Health endpoint returns 200

**Questions to Answer:**
1. What is the size of your Docker image? (`docker images ml-app:v1`)
2. How long did the build take?
3. What happens if you change `app.py` and rebuild?

---

## Part 2: Optimize with Multi-Stage Builds (45 minutes)

### Task 2.1: Implement Multi-Stage Build

The basic Dockerfile creates a large image (~2GB). Let's optimize it!

Create `Dockerfile.optimized`:
```dockerfile
# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Your Task:**
1. Build the optimized image: `docker build -f Dockerfile.optimized -t ml-app:v2 .`
2. Compare image sizes:
```bash
docker images | grep ml-app
```
3. Run and test: `docker run -p 8000:8000 ml-app:v2`

**Expected Results:**
- v2 image is 30-50% smaller than v1
- Both versions work identically
- Build time might be slightly longer but rebuilds are faster

**Questions:**
1. Why is the multi-stage image smaller?
2. What does `--no-cache-dir` do?
3. Why copy Python packages instead of reinstalling in stage 2?

---

### Task 2.2: Optimize Layer Caching

Modify your Dockerfile to optimize layer caching:

```dockerfile
# ❌ Bad: Copies everything first
COPY . .
RUN pip install -r requirements.txt

# ✅ Good: Copies requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

**Your Task:**
1. Modify `src/app.py` (change a message)
2. Rebuild the image
3. Note which steps use cache (look for "CACHED")
4. Now modify `requirements.txt` (add a comment)
5. Rebuild and observe the difference

**Expected Behavior:**
- Changing app code only invalidates later layers
- Changing requirements invalidates dependency install and everything after

---

## Part 3: Docker Compose for Multi-Container Setup (45 minutes)

### Task 3.1: Add Monitoring with Prometheus

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  # ML API
  ml-api:
    build: .
    image: ml-app:v2
    container_name: ml-api
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - ml-network
    restart: unless-stopped

  # Prometheus monitoring
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
    restart: unless-stopped

  # Grafana visualization
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
    restart: unless-stopped

networks:
  ml-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:8000']
```

**Your Task:**
1. Start all services: `docker-compose up -d`
2. Check logs: `docker-compose logs -f ml-api`
3. Visit services:
   - ML API: http://localhost:8000/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)
4. Stop services: `docker-compose down`

**Expected Results:**
- All three containers start successfully
- ML API is accessible and healthy
- Prometheus shows ml-api as a target
- Grafana can connect to Prometheus

---

### Task 3.2: Add Prometheus Metrics to ML API

Modify `src/app.py` to add metrics:

```python
from prometheus_client import Counter, Histogram, make_asgi_app
import time

# Add metrics
request_count = Counter(
    'ml_api_requests_total',
    'Total API requests',
    ['endpoint', 'status']
)

request_duration = Histogram(
    'ml_api_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Update endpoints to record metrics
@app.get("/info")
def info():
    start_time = time.time()

    try:
        # ... existing code ...

        request_count.labels(endpoint='info', status='success').inc()
        return result
    except Exception as e:
        request_count.labels(endpoint='info', status='error').inc()
        raise
    finally:
        duration = time.time() - start_time
        request_duration.labels(endpoint='info').observe(duration)
```

**Your Task:**
1. Add `prometheus-client` to requirements.txt
2. Rebuild: `docker-compose build ml-api`
3. Restart: `docker-compose up -d`
4. Make requests to `/info` endpoint
5. Check metrics: http://localhost:8000/metrics
6. View in Prometheus: http://localhost:9090/graph
   - Query: `ml_api_requests_total`
   - Query: `rate(ml_api_requests_total[1m])`

---

## Part 4: Docker Volumes and Persistence (30 minutes)

### Task 4.1: Mount Model Directory

Models can be large. Instead of including in image, mount from host.

Modify `docker-compose.yml`:
```yaml
services:
  ml-api:
    # ... existing config ...
    volumes:
      - ./models:/app/models:ro  # Read-only mount
    environment:
      - MODEL_PATH=/app/models/resnet18.pth
```

Modify `src/app.py`:
```python
import os

@app.on_event("startup")
async def load_model():
    global model

    model_path = os.getenv("MODEL_PATH")

    if model_path and os.path.exists(model_path):
        # Load from file
        model = torch.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        # Load pretrained
        model = models.resnet18(pretrained=True)
        print("Model loaded (pretrained)")

    model.eval()
```

**Your Task:**
1. Create `models/` directory
2. Save a model:
```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
torch.save(model, 'models/resnet18.pth')
```
3. Restart services and verify model loads from file

---

### Task 4.2: Persist Logs

Create a volume for logs:

```yaml
services:
  ml-api:
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs  # Log directory
```

Add logging to `src/app.py`:
```python
import logging
from datetime import datetime

# Setup file logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.get("/info")
def info():
    logger.info("Info endpoint called")
    # ... rest of code ...
```

**Your Task:**
1. Restart services
2. Make requests to `/info`
3. Check `logs/app.log` on host
4. Verify logs persist after container restart

---

## Part 5: Docker Networks and Communication (30 minutes)

### Task 5.1: Multi-Service Communication

Add a Redis cache service:

```yaml
services:
  # ... existing services ...

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    networks:
      - ml-network
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

volumes:
  # ... existing volumes ...
  redis-data:
```

Modify `src/app.py` to use Redis:
```python
import redis
import json

# Connect to Redis
redis_client = redis.Redis(
    host='redis',  # Service name from docker-compose
    port=6379,
    decode_responses=True
)

@app.get("/info")
def info():
    # Check cache
    cache_key = "model_info"
    cached = redis_client.get(cache_key)

    if cached:
        logger.info("Returning cached response")
        return json.loads(cached)

    # Generate response
    if model is None:
        return {"error": "Model not loaded"}

    params = sum(p.numel() for p in model.parameters())

    response = {
        "model": "ResNet-18",
        "parameters": params,
        "device": "cpu",
        "cached": False
    }

    # Cache for 60 seconds
    redis_client.setex(cache_key, 60, json.dumps(response))

    logger.info("Generated and cached response")
    return response
```

**Your Task:**
1. Add `redis` to requirements.txt
2. Rebuild and restart: `docker-compose up -d --build`
3. Test caching:
```bash
# First request (slow)
time curl http://localhost:8000/info

# Second request (fast, from cache)
time curl http://localhost:8000/info
```
4. Verify network connectivity:
```bash
docker-compose exec ml-api ping redis
```

---

## Part 6: Challenge Tasks (Optional, 1-2 hours)

### Challenge 1: GPU Support

Modify Dockerfile to support both CPU and GPU:

```dockerfile
# Use a build arg to choose base image
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# For GPU: docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 -t ml-app:gpu .
```

### Challenge 2: Model Hot-Reload

Implement file watching to reload model when file changes:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ModelReloader(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.pth'):
            logger.info(f"Model file changed: {event.src_path}")
            load_model()
```

### Challenge 3: Health Check with Dependencies

Improve health check to verify all dependencies:

```python
@app.get("/health")
def health():
    checks = {
        "model": model is not None,
        "redis": False,
        "disk_space": False
    }

    # Check Redis
    try:
        redis_client.ping()
        checks["redis"] = True
    except:
        pass

    # Check disk space
    import shutil
    disk = shutil.disk_usage("/")
    checks["disk_space"] = (disk.free / disk.total) > 0.1  # >10% free

    healthy = all(checks.values())

    return {
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks
    }
```

---

## Verification Checklist

After completing this exercise, you should be able to:

- [ ] Build a Docker image for an ML application
- [ ] Use multi-stage builds to reduce image size
- [ ] Optimize Dockerfile for layer caching
- [ ] Create a docker-compose.yml for multi-container apps
- [ ] Use Docker volumes for persistent data
- [ ] Configure Docker networks for service communication
- [ ] Add health checks to containers
- [ ] Integrate Prometheus metrics
- [ ] Run non-root containers for security
- [ ] Debug container issues using logs

---

## Submission (If part of course)

Create a ZIP file with:
1. All Dockerfiles (basic and optimized)
2. docker-compose.yml
3. Source code (src/app.py)
4. Requirements.txt
5. Screenshot of running services
6. Screenshot of Prometheus metrics
7. Answers to questions throughout exercise

---

## Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Build Guide](https://docs.docker.com/build/building/multi-stage/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

---

**Next Exercise:** [Exercise 03: Kubernetes Deployment](./exercise-03-kubernetes.md)
