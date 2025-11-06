# Lesson 06: Model Serving Basics

**Duration:** 8 hours

## Learning Objectives

By the end of this lecture, you will be able to:
- Explain the difference between model training and model serving
- Compare batch inference vs. real-time inference serving patterns
- Implement basic model serving using Flask and FastAPI
- Deploy models using TorchServe, TensorFlow Serving, or ONNX Runtime
- Understand model versioning and A/B testing strategies
- Implement health checks, logging, and basic monitoring for model endpoints
- Optimize model serving for latency and throughput
- Handle common serving challenges (batching, caching, error handling)

## Introduction

Model serving is the process of making ML models available for inference in production environments. As an ML infrastructure engineer, this is one of your primary responsibilities.

**Why Model Serving is Critical:**
- Models have no value unless they're deployed
- Serving requirements differ dramatically from training
- Performance, latency, and reliability are critical
- Must handle real-world traffic patterns
- Need monitoring, versioning, and rollback capabilities

This lesson covers the fundamentals of model serving, from simple approaches to production-grade solutions.

## What is Model Serving?

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Model Lifecycle                        │
└─────────────────────────────────────────────────────────────┘

Training Phase              Serving Phase
┌─────────────┐            ┌──────────────────────────────┐
│             │            │                              │
│  Training   │            │   Input Data                 │
│  Data       │            │       ↓                      │
│     ↓       │            │   Preprocessing              │
│  Training   │            │       ↓                      │
│  Pipeline   │            │   Model Inference            │
│     ↓       │   ──────►  │       ↓                      │
│  Model      │            │   Post-processing            │
│  (weights)  │            │       ↓                      │
│             │            │   Prediction/Response        │
│             │            │                              │
└─────────────┘            └──────────────────────────────┘

Training: Hours/Days        Serving: Milliseconds
Batch Processing           Real-time Processing
Optimize for Accuracy      Optimize for Latency
Can Retry                  Must Succeed First Try
```

### Key Differences: Training vs Serving

| Aspect | Training | Serving |
|--------|----------|---------|
| **Time Constraint** | Hours/Days | Milliseconds |
| **Data Volume** | Millions of samples | 1 sample at a time |
| **Compute** | Parallel/Distributed | Single inference |
| **Optimization Goal** | Accuracy | Latency + Throughput |
| **Fault Tolerance** | Can retry | Must be reliable |
| **Scaling** | Horizontal (more GPUs) | Horizontal (more replicas) |
| **Cost Model** | Temporary | Continuous |

## Model Serving Requirements

### 1. Performance Requirements

**Latency**: Time from request to response
```
Latency Budget Examples:

Real-time Applications:
- Search ranking: < 50ms
- Ad serving: < 20ms
- Fraud detection: < 100ms
- Chatbots: < 500ms

Batch Processing:
- Email spam filtering: < 5 seconds
- Content recommendation: < 1 second
- Image processing: < 2 seconds
```

**Throughput**: Requests handled per second
```
Throughput Requirements:

Small Application:     10 req/sec
Medium Application:    100 req/sec
Large Application:     1,000 req/sec
Massive Scale:         10,000+ req/sec
```

**Availability**: Uptime percentage
```
Availability Levels:

99% (Two 9s):     ~3.65 days downtime/year
99.9% (Three 9s): ~8.76 hours downtime/year
99.99% (Four 9s): ~52 minutes downtime/year
99.999% (Five 9s): ~5 minutes downtime/year
```

### 2. Functional Requirements

- **Versioning**: Support multiple model versions
- **A/B Testing**: Route traffic to different models
- **Monitoring**: Track predictions and performance
- **Logging**: Audit trail of predictions
- **Batching**: Combine multiple requests for efficiency
- **Caching**: Store frequent predictions
- **Fallback**: Handle model failures gracefully

### 3. Operational Requirements

- **Easy Deployment**: Simple to update models
- **Rollback**: Quick recovery from bad deployments
- **Scalability**: Handle traffic spikes
- **Resource Efficiency**: Optimize cost
- **Security**: Protect model and data
- **Observability**: Understand system behavior

## Model Serving Approaches

### 1. Embedded Model (Simplest)

**Description**: Model runs within application code

```python
# Example: Flask app with embedded model
from flask import Flask, request, jsonify
import torch
import torchvision.models as models

app = Flask(__name__)

# Load model at startup
model = models.resnet18(pretrained=True)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    image = process_image(request.files['image'])

    # Run inference
    with torch.no_grad():
        prediction = model(image)

    return jsonify({'class': get_class_name(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Pros:**
- ✅ Simple to implement
- ✅ No additional infrastructure
- ✅ Low latency (no network calls)

**Cons:**
- ❌ Model updates require app restart
- ❌ Hard to scale independently
- ❌ No model versioning
- ❌ Language-specific (Python only)

**Use When**: Prototypes, simple apps, low traffic

### 2. Model as a Service (Microservice)

**Description**: Model runs in separate service

```
┌──────────────┐       HTTP/gRPC      ┌──────────────┐
│              │  ────────────────►   │              │
│  Application │                      │ Model Service│
│              │  ◄────────────────   │              │
└──────────────┘       Response       └──────────────┘
```

**Example Architecture:**
```python
# Model Service (FastAPI)
from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()

# Load model
model = load_model("resnet18")

@app.post("/predict")
async def predict(image: UploadFile):
    processed = preprocess(await image.read())
    prediction = model(processed)
    return {"prediction": prediction.tolist()}

# Application calls this service
import httpx

async def get_prediction(image_data):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://model-service:8000/predict",
            files={"image": image_data}
        )
    return response.json()
```

**Pros:**
- ✅ Independent scaling
- ✅ Model updates without app changes
- ✅ Can serve multiple applications
- ✅ Can use different tech stacks

**Cons:**
- ❌ Network latency overhead
- ❌ More infrastructure complexity
- ❌ Need load balancing

**Use When**: Multiple apps use same model, need independent scaling

### 3. Model Serving Frameworks

**Description**: Use specialized frameworks for production serving

**Popular Frameworks:**

#### TorchServe (PyTorch)

```bash
# Install
pip install torchserve torch-model-archiver

# Create model archive
torch-model-archiver \
  --model-name resnet18 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet18.pth \
  --handler image_classifier

# Start server
torchserve \
  --start \
  --model-store model_store \
  --models resnet18=resnet18.mar

# Make prediction
curl http://localhost:8080/predictions/resnet18 -T image.jpg
```

**Features:**
- Multi-model serving
- Model versioning
- Batch inference
- Metrics (Prometheus)
- Model management API
- GPU support

#### TensorFlow Serving

```bash
# Install
apt-get install tensorflow-model-server

# Serve model
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=resnet \
  --model_base_path=/models/resnet

# Make prediction
curl -X POST http://localhost:8501/v1/models/resnet:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"input": [...]}]}'
```

**Features:**
- High-performance serving
- gRPC and REST APIs
- Model versioning
- Batching
- GPU support
- Production-tested at Google scale

#### ONNX Runtime

```python
# Convert model to ONNX
import torch.onnx

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=12
)

# Serve with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_data})
```

**Features:**
- Framework-agnostic
- Often faster than native frameworks
- Cross-platform
- Hardware acceleration

**Pros:**
- ✅ Production-ready
- ✅ Built-in best practices
- ✅ Optimized performance
- ✅ Monitoring and metrics
- ✅ Model management

**Cons:**
- ❌ Learning curve
- ❌ Less flexible than custom
- ❌ Framework-specific

**Use When**: Production deployments, need reliability and performance

### 4. Cloud ML Platforms

**Description**: Fully managed serving on cloud platforms

#### AWS SageMaker

```python
import boto3

# Deploy model
sagemaker = boto3.client('sagemaker')

sagemaker.create_model(
    ModelName='my-model',
    PrimaryContainer={
        'Image': 'pytorch-inference:1.12',
        'ModelDataUrl': 's3://bucket/model.tar.gz'
    },
    ExecutionRoleArn='arn:aws:iam::role'
)

sagemaker.create_endpoint(
    EndpointName='my-endpoint',
    EndpointConfigName='my-config'
)

# Make prediction
runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    Body=json.dumps(data)
)
```

#### GCP Vertex AI

```python
from google.cloud import aiplatform

# Deploy model
model = aiplatform.Model.upload(
    display_name="my-model",
    artifact_uri="gs://bucket/model",
    serving_container_image_uri="pytorch-prediction-latest"
)

endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=10
)

# Make prediction
prediction = endpoint.predict(instances=[data])
```

#### Azure ML

```python
from azure.ai.ml import MLClient

# Deploy model
ml_client = MLClient.from_config()

deployment = ml_client.online_deployments.begin_create_or_update(
    deployment_name="my-deployment",
    model="my-model:1",
    instance_type="Standard_DS3_v2",
    instance_count=2
)

# Make prediction
response = ml_client.online_endpoints.invoke(
    endpoint_name="my-endpoint",
    request_file="request.json"
)
```

**Pros:**
- ✅ Fully managed
- ✅ Auto-scaling
- ✅ No infrastructure management
- ✅ Integrated monitoring
- ✅ A/B testing support

**Cons:**
- ❌ Vendor lock-in
- ❌ Expensive
- ❌ Less control
- ❌ Framework limitations

**Use When**: Want managed solution, have cloud budget, need quick deployment

## Inference Optimization Techniques

### 1. Batch Inference

**Problem**: Processing one request at a time is inefficient

**Solution**: Combine multiple requests into a batch

```python
# Without batching (slow)
for request in requests:
    result = model(request)
    send_response(result)

# With batching (fast)
batch = []
for request in requests:
    batch.append(request)
    if len(batch) >= BATCH_SIZE:
        results = model(batch)
        for result in results:
            send_response(result)
        batch = []
```

**Performance Impact:**
- 3-10x throughput improvement
- Tradeoff: Adds latency (wait for batch to fill)

**Configuration:**
```python
BATCH_SIZE = 32  # How many requests to batch
MAX_WAIT_TIME = 50  # Max milliseconds to wait for batch

# Send batch when either:
# 1. Batch is full (batch_size reached)
# 2. Max wait time exceeded
```

### 2. Model Quantization

**Problem**: Models are large and slow

**Solution**: Reduce precision (FP32 → INT8)

```python
# PyTorch quantization
import torch

model = load_model()

# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Benefits:
# - 4x smaller model
# - 2-4x faster inference
# - Minimal accuracy loss (<1%)
```

### 3. Model Caching

**Problem**: Same inputs requested repeatedly

**Solution**: Cache predictions

```python
from functools import lru_cache
import hashlib

class CachedModel:
    def __init__(self, model, cache_size=1000):
        self.model = model
        self.cache = {}
        self.cache_size = cache_size

    def predict(self, input_data):
        # Create cache key
        cache_key = hashlib.md5(input_data).hexdigest()

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Run inference
        result = self.model(input_data)

        # Store in cache
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = result

        return result
```

**When to Use:**
- Expensive models
- Limited input space
- Read-heavy workload

### 4. Model Warm-Up

**Problem**: First inference is very slow

**Solution**: Pre-warm model during startup

```python
def warm_up_model(model, num_warmup=10):
    """
    Run dummy inferences to warm up model
    """
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Warming up model...")
    for i in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    print("Model warmed up!")

# In serving code
model = load_model()
warm_up_model(model)
# Now ready for real requests
```

### 5. Asynchronous Processing

**Problem**: Blocking I/O slows down serving

**Solution**: Use async/await patterns

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile):
    # Read file asynchronously
    image_data = await image.read()

    # Preprocess asynchronously
    processed = await preprocess_async(image_data)

    # Run inference (CPU-bound, use thread pool)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict, processed)

    return {"prediction": result}
```

## Deployment Strategies

### 1. Blue-Green Deployment

```
Initial State (Blue):
Users → Load Balancer → Blue Deployment (v1.0)

Deploy Green:
Users → Load Balancer → Blue Deployment (v1.0)
                      → Green Deployment (v2.0) [testing]

Switch Traffic:
Users → Load Balancer → Green Deployment (v2.0)
        Old Blue (v1.0) [kept for rollback]

Rollback if needed:
Users → Load Balancer → Blue Deployment (v1.0)
```

**Pros:**
- Zero downtime
- Instant rollback
- Full testing before switch

**Cons:**
- 2x resources during deployment
- All-or-nothing switch

### 2. Canary Deployment

```
Initial State:
Users → Load Balancer → v1.0 (100%)

Canary (5%):
Users → Load Balancer → v1.0 (95%)
                      → v2.0 (5%)

Increase (50%):
Users → Load Balancer → v1.0 (50%)
                      → v2.0 (50%)

Full Rollout:
Users → Load Balancer → v2.0 (100%)
```

**Pros:**
- Gradual rollout
- Early error detection
- Minimal impact if issues

**Cons:**
- Slower rollout
- Need traffic routing logic
- Monitoring complexity

### 3. Shadow Deployment

```
Production:
Users → Load Balancer → v1.0 (serves users)
                      ↓
                    Copy requests
                      ↓
                    v2.0 (shadow, no response to users)
```

**Pros:**
- Zero risk to users
- Test with real traffic
- Compare model performance

**Cons:**
- 2x compute cost
- Doesn't catch all issues (no user impact)

## Monitoring Model Serving

### Key Metrics

**Infrastructure Metrics:**
```python
# Latency
p50_latency: 45ms   # 50% of requests < 45ms
p95_latency: 120ms  # 95% of requests < 120ms
p99_latency: 300ms  # 99% of requests < 300ms

# Throughput
requests_per_second: 150

# Resource Usage
cpu_usage: 65%
memory_usage: 3.2 GB
gpu_utilization: 80%

# Availability
uptime: 99.95%
error_rate: 0.05%
```

**Model Metrics:**
```python
# Prediction distribution
class_0: 35%
class_1: 45%
class_2: 20%

# Confidence scores
avg_confidence: 0.87
low_confidence_count: 45 (predictions < 0.5)

# Data drift
input_mean: 0.52 (expected: 0.5)
input_std: 0.31 (expected: 0.3)
```

### Implementing Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
prediction_count = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds'
)

model_confidence = Histogram(
    'model_prediction_confidence',
    'Prediction confidence scores'
)

# Use in serving code
@app.post("/predict")
async def predict(image: UploadFile):
    start_time = time.time()

    try:
        # Run inference
        result = await model.predict(image)

        # Record metrics
        prediction_count.labels(
            model_version='v1.0',
            status='success'
        ).inc()

        model_confidence.observe(result['confidence'])

        return result

    except Exception as e:
        prediction_count.labels(
            model_version='v1.0',
            status='error'
        ).inc()
        raise

    finally:
        latency = time.time() - start_time
        prediction_latency.observe(latency)
```

## Practical Example: Complete Serving Solution

```python
# production_serving.py
from fastapi import FastAPI, UploadFile, HTTPException
from prometheus_client import make_asgi_app, Counter, Histogram
import torch
import logging
from typing import Dict
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
predictions = Counter('predictions_total', 'Total predictions', ['status'])
latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# Create app
app = FastAPI(title="ML Model Serving")

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Global model (loaded once)
MODEL = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global MODEL
    logger.info("Loading model...")
    MODEL = load_and_warmup_model()
    logger.info("Model loaded and ready!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

@app.post("/predict")
async def predict(image: UploadFile) -> Dict:
    """Prediction endpoint"""
    start_time = time.time()

    try:
        # Validate input
        if not image.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid image type")

        # Process image
        image_data = await image.read()
        processed = preprocess(image_data)

        # Run inference
        with torch.no_grad():
            prediction = MODEL(processed)

        # Post-process
        result = {
            "class": get_class_name(prediction),
            "confidence": float(prediction.max()),
            "latency_ms": (time.time() - start_time) * 1000
        }

        # Record success
        predictions.labels(status='success').inc()

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        predictions.labels(status='error').inc()
        raise HTTPException(500, str(e))

    finally:
        latency.observe(time.time() - start_time)

def load_and_warmup_model():
    """Load model and warm up"""
    import torchvision.models as models

    # Load model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Warm up
    dummy = torch.randn(1, 3, 224, 224)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    return model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Key Takeaways

1. **Model serving** is making models available for production inference
2. **Key requirements**: Low latency, high throughput, high availability
3. **Approaches**: Embedded, microservice, serving frameworks, cloud platforms
4. **Optimization**: Batching, quantization, caching, warm-up, async processing
5. **Deployment**: Blue-green, canary, shadow strategies
6. **Monitoring**: Track latency, throughput, errors, model metrics
7. **Production frameworks**: TorchServe, TensorFlow Serving, ONNX Runtime

## Self-Check Questions

1. What's the difference between training and serving workloads?
2. What are the pros and cons of embedded models vs microservices?
3. How does batch inference improve throughput?
4. What is quantization and what are its benefits?
5. What's the difference between blue-green and canary deployments?
6. What metrics should you monitor for model serving?
7. When would you use TorchServe vs a simple FastAPI service?

## Additional Resources

- [TorchServe Documentation](https://pytorch.org/serve/)
- [TensorFlow Serving Guide](https://www.tensorflow.org/tfx/guide/serving)
- [ONNX Runtime](https://onnxruntime.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Model Serving Best Practices](https://ml-ops.org/content/model-serving)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

---

**Next Lesson:** [07-docker-basics.md](./07-docker-basics.md) - Containerization fundamentals
