# Lesson 08: API Development for ML Services

**Duration:** 5 hours

## Learning Objectives

By the end of this lecture, you will be able to:
- Build RESTful APIs using FastAPI for ML model serving
- Implement request validation using Pydantic models
- Handle asynchronous requests for improved throughput
- Add authentication and authorization to ML APIs
- Implement rate limiting and request throttling
- Generate automatic API documentation using OpenAPI/Swagger
- Add comprehensive error handling and logging
- Test APIs using pytest and integration testing strategies
- Deploy FastAPI applications with Uvicorn and Gunicorn

## Introduction

APIs (Application Programming Interfaces) are how users interact with your ML models. As an ML infrastructure engineer, you'll build APIs that:

- Accept user input (images, text, data)
- Run model inference
- Return predictions
- Handle errors gracefully
- Scale to handle many requests
- Provide monitoring and logging

This lesson covers building production-ready ML APIs using modern Python frameworks, focusing on FastAPI.

## Why FastAPI for ML?

**FastAPI** is a modern, fast web framework for building APIs with Python 3.7+.

### FastAPI vs Alternatives

| Framework | Speed | Async | Docs | Type Safety | ML Use |
|-----------|-------|-------|------|-------------|---------|
| **FastAPI** | ⚡⚡⚡ | ✅ | Auto | ✅ | ⭐⭐⭐ |
| Flask | ⚡ | ❌ | Manual | ❌ | ⭐⭐ |
| Django | ⚡ | ❌ | Good | ❌ | ⭐ |
| Tornado | ⚡⚡ | ✅ | Manual | ❌ | ⭐⭐ |

**Why FastAPI wins for ML:**
1. **Speed**: One of the fastest Python frameworks (on par with NodeJS, Go)
2. **Async**: Handle multiple requests efficiently
3. **Auto Documentation**: Interactive API docs (Swagger UI)
4. **Type Safety**: Pydantic validation catches errors early
5. **Modern**: Built on latest Python features

## FastAPI Basics

### Installation

```bash
pip install fastapi uvicorn[standard]
```

### Hello World API

```python
# app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**Run the server:**
```bash
uvicorn app:app --reload
```

**Access:**
- API: http://localhost:8000/
- Auto Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Path Parameters

```python
@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    return {"model": model_name, "status": "loaded"}

# Access: /models/resnet18
# Returns: {"model": "resnet18", "status": "loaded"}
```

### Query Parameters

```python
@app.get("/predict")
def predict(
    model: str = "resnet18",
    top_k: int = 5,
    threshold: float = 0.5
):
    return {
        "model": model,
        "top_k": top_k,
        "threshold": threshold
    }

# Access: /predict?model=resnet50&top_k=10&threshold=0.7
```

### Request Body (POST)

```python
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image_url: str
    model: str = "resnet18"
    top_k: int = 5

@app.post("/predict")
def predict(request: PredictionRequest):
    return {
        "image_url": request.image_url,
        "model": request.model,
        "predictions": []
    }

# POST /predict
# Body: {"image_url": "http://example.com/image.jpg", "top_k": 3}
```

### File Upload

```python
from fastapi import File, UploadFile

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Read file content
    contents = await file.read()

    # Process image
    # ... (your ML inference code)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "prediction": "cat"
    }

# Test with curl:
# curl -X POST http://localhost:8000/predict/image \
#   -F "file=@image.jpg"
```

## Building an ML Prediction API

### Complete Example

```python
# ml_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="ML Model Serving API",
    description="API for image classification using ResNet",
    version="1.0.0"
)

# Global model (loaded once on startup)
MODEL = None
DEVICE = None
TRANSFORM = None
CLASSES = None

# Response models
class PredictionResult(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    model: str
    inference_time_ms: float

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global MODEL, DEVICE, TRANSFORM, CLASSES

    logger.info("Loading model...")

    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # Load model
    MODEL = models.resnet18(pretrained=True)
    MODEL.to(DEVICE)
    MODEL.eval()

    # Define transforms
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load ImageNet class labels
    CLASSES = load_imagenet_classes()

    logger.info("Model loaded successfully!")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "ML Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image file)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Predict image class

    Args:
        file: Image file (JPEG, PNG)
        top_k: Number of top predictions to return

    Returns:
        Predictions with class names and confidence scores
    """
    import time
    start_time = time.time()

    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. "
                       f"Supported types: image/jpeg, image/png"
            )

        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        predictions = [
            PredictionResult(
                class_name=CLASSES[idx.item()],
                confidence=float(prob.item())
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            predictions=predictions,
            model="resnet18",
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def load_imagenet_classes():
    """Load ImageNet class labels"""
    # Simplified - in production, load from file
    return ["class_" + str(i) for i in range(1000)]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Testing the API

```bash
# Test health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict?top_k=3 \
  -F "file=@cat.jpg"

# Response:
# {
#   "predictions": [
#     {"class_name": "tabby_cat", "confidence": 0.87},
#     {"class_name": "egyptian_cat", "confidence": 0.09},
#     {"class_name": "tiger_cat", "confidence": 0.03}
#   ],
#   "model": "resnet18",
#   "inference_time_ms": 45.23
# }
```

## Request Validation with Pydantic

### Input Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class PredictionRequest(BaseModel):
    image_url: str = Field(..., description="URL of image to classify")
    model_name: str = Field("resnet18", description="Model to use")
    top_k: int = Field(5, ge=1, le=10, description="Number of top predictions")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Confidence threshold")

    @validator('image_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Invalid URL format')
        return v

    @validator('model_name')
    def validate_model(cls, v):
        allowed_models = ['resnet18', 'resnet50', 'mobilenet_v2']
        if v not in allowed_models:
            raise ValueError(f'Model must be one of: {allowed_models}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://example.com/cat.jpg",
                "model_name": "resnet18",
                "top_k": 5,
                "threshold": 0.5
            }
        }

@app.post("/predict/url")
async def predict_from_url(request: PredictionRequest):
    # Input is automatically validated
    # If invalid, FastAPI returns 422 error with details
    return {"message": "Processing image", "request": request}
```

## Error Handling

### HTTP Exception Handling

```python
from fastapi import HTTPException, status

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Input validation error
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type"
        )

    try:
        # Process image
        result = process_image(file)
        return result

    except ValueError as e:
        # Client error (bad input)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        # Server error
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

### Custom Exception Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class ModelNotLoadedError(Exception):
    pass

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Model not loaded",
            "detail": "Service is starting up, please try again"
        }
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise ModelNotLoadedError()

    # ... rest of prediction logic
```

## Async Processing

### Why Async?

```python
# ❌ Synchronous (blocking)
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # This blocks the entire server while reading file
    contents = file.file.read()
    # This blocks while running inference
    result = model.predict(contents)
    return result

# ✅ Asynchronous (non-blocking)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # This doesn't block - other requests can be handled
    contents = await file.read()

    # For CPU-bound operations, use thread pool
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict, contents)

    return result
```

### Background Tasks

```python
from fastapi import BackgroundTasks

def log_prediction(image_id: str, prediction: str):
    """Background task to log prediction"""
    # This runs after response is sent
    logger.info(f"Image {image_id}: predicted {prediction}")
    # Could also save to database, send metrics, etc.

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks
):
    # Run inference
    result = await run_inference(file)

    # Add background task
    background_tasks.add_task(
        log_prediction,
        image_id=file.filename,
        prediction=result['class']
    )

    # Response sent immediately, background task runs after
    return result
```

## Monitoring and Metrics

### Adding Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, make_asgi_app
import time

# Define metrics
prediction_count = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model']
)

model_confidence = Histogram(
    'ml_prediction_confidence',
    'Prediction confidence scores',
    ['model']
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    model_name = "resnet18"

    try:
        # Run inference
        result = await run_inference(file)

        # Record metrics
        prediction_count.labels(model=model_name, status='success').inc()
        model_confidence.labels(model=model_name).observe(result['confidence'])

        return result

    except Exception as e:
        prediction_count.labels(model=model_name, status='error').inc()
        raise

    finally:
        latency = time.time() - start_time
        prediction_latency.labels(model=model_name).observe(latency)
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        return json.dumps(log_data)

# Configure logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Use in endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info("Prediction started", extra={'filename': file.filename})
    try:
        result = await run_inference(file)
        logger.info("Prediction successful", extra={'result': result})
        return result
    except Exception as e:
        logger.error("Prediction failed", extra={'error': str(e)})
        raise
```

## API Versioning

### Path-Based Versioning

```python
from fastapi import APIRouter

# Version 1 API
v1_router = APIRouter(prefix="/v1")

@v1_router.post("/predict")
async def predict_v1(file: UploadFile = File(...)):
    # Old prediction logic
    return {"version": "v1", "result": "..."}

# Version 2 API (with enhanced features)
v2_router = APIRouter(prefix="/v2")

@v2_router.post("/predict")
async def predict_v2(
    file: UploadFile = File(...),
    model: str = "resnet50",  # New parameter
    return_embeddings: bool = False  # New feature
):
    # New prediction logic
    return {"version": "v2", "result": "..."}

# Include routers
app.include_router(v1_router)
app.include_router(v2_router)

# Access:
# POST /v1/predict
# POST /v2/predict
```

## Rate Limiting

### Basic Rate Limiting

```python
from fastapi import Request, HTTPException
from datetime import datetime, timedelta
from collections import defaultdict

# Simple in-memory rate limiter (production: use Redis)
request_counts = defaultdict(list)
RATE_LIMIT = 100  # requests
TIME_WINDOW = 60  # seconds

async def rate_limit(request: Request):
    client_ip = request.client.host
    now = datetime.now()

    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if now - req_time < timedelta(seconds=TIME_WINDOW)
    ]

    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT} requests per {TIME_WINDOW}s"
        )

    # Add current request
    request_counts[client_ip].append(now)

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    await rate_limit(request)
    # ... rest of prediction logic
```

## API Documentation

### Customizing Auto-Generated Docs

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ML Model Serving API",
        version="1.0.0",
        description="""
        ## Image Classification API

        This API provides image classification using pre-trained models.

        ### Features:
        * Multiple model support (ResNet, MobileNet)
        * Batch prediction
        * Confidence thresholds
        * Real-time inference

        ### Rate Limits:
        * 100 requests per minute per IP

        ### Contact:
        * Email: ml-team@example.com
        * Docs: https://docs.example.com
        """,
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

## Production Deployment

### Running with Uvicorn

```bash
# Development (auto-reload)
uvicorn app:app --reload

# Production (single worker)
uvicorn app:app --host 0.0.0.0 --port 8000

# Production (multiple workers)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# With custom settings
uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --limit-concurrency 1000 \
  --timeout-keep-alive 5
```

### Running with Gunicorn + Uvicorn Workers

```bash
# Install
pip install gunicorn

# Run with Uvicorn workers
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

## Key Takeaways

1. **FastAPI** is ideal for ML APIs: fast, async, auto-docs, type-safe
2. **Pydantic** provides automatic request validation
3. **Async/await** enables efficient handling of I/O-bound operations
4. **Error handling** is critical for production reliability
5. **Monitoring** with Prometheus provides observability
6. **Rate limiting** protects against abuse
7. **Versioning** enables safe API evolution
8. **Background tasks** for post-processing without blocking responses

## Self-Check Questions

1. What are the advantages of FastAPI over Flask for ML serving?
2. How does Pydantic help with input validation?
3. What's the difference between sync and async endpoints?
4. How do you handle file uploads in FastAPI?
5. What metrics should you track for an ML API?
6. How do you implement API versioning?
7. What's the difference between Uvicorn and Gunicorn?

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [API Design Best Practices](https://restfulapi.net/)
- [HTTP Status Codes](https://httpstatuses.com/)

---

**Next:** Complete [Project 01](../../projects/project-101-basic-model-serving/) to apply these concepts!
